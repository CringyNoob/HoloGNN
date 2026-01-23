import os
# --- CRITICAL WINDOWS FIX ---
# Prevents "OMP: Error #15" crash on Anaconda/Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import time

# Custom Modules
from src.dataset import MegaScaleDataset
from src.full_model import HoloGNN

# --- CONFIGURATION FOR GTX 1050 Ti ---
BATCH_SIZE = 4           # Optimized for 4GB VRAM
LEARNING_RATE = 1e-4     
EPOCHS = 3               # 3 Epochs gives a strong research result
DATA_PATH = "data/mega_scale_cdna/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"

# --- DATASET SIZE CONTROL ---
# Set to 200000 for Publication-Grade results (approx 4-5 hours)
# If you want a quick test, change this to 5000
MAX_SAMPLES = 200000 

def train():
    print("--- 1. INITIALIZING HOLO-GNN ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    model = HoloGNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- 2. LOADING DATASET ---")
    full_dataset = MegaScaleDataset(DATA_PATH)
    total_available = len(full_dataset)
    
    # Smart Subsetting
    if MAX_SAMPLES and MAX_SAMPLES < total_available:
        print(f"!!! OPTIMIZATION: Using {MAX_SAMPLES} samples (out of {total_available}) for efficient training.")
        subset_indices = range(MAX_SAMPLES)
        active_dataset = Subset(full_dataset, subset_indices)
    else:
        print(f"!!! FULL RUN: Using all {total_available} samples. This may take days.")
        active_dataset = full_dataset

    # Split 90/10 for maximum training signal
    train_size = int(0.9 * len(active_dataset))
    val_size = len(active_dataset) - train_size
    train_dataset, val_dataset = random_split(active_dataset, [train_size, val_size])
    
    # num_workers=0 is mandatory for Windows to avoid crashes
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    print(f"Training on: {len(train_dataset)} sequences")
    print(f"Validating on: {len(val_dataset)} sequences")

    print("--- 3. STARTING PRODUCTION TRAINING ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            # Move to GPU
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Data Object
            class DataBatch: pass
            data = DataBatch()
            data.input_ids = input_ids
            data.mask = mask
            # CRITICAL: Passing None forces the Backbone to build the Graph from Attention
            data.edge_index = None 
            
            # Forward & Backward
            optimizer.zero_grad()
            predictions = model(data, task="idr") # Predicting Stability
            loss = criterion(predictions.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
        
        # Validation Loop
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint with a consistent name
        save_name = f"holognn_stability_final.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Checkpoint saved: {save_name}")

    total_time = (time.time() - start_time) / 3600
    print(f"--- TRAINING COMPLETE in {total_time:.2f} hours ---")

if __name__ == "__main__":
    train()