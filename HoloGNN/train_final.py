import os
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

# --- PRO SETTINGS FOR GTX 1050 Ti ---
BATCH_SIZE = 4            # Physical limit of your GPU
ACCUMULATION_STEPS = 8    # 4 * 8 = Effective Batch Size of 32 (Stable!)
LEARNING_RATE = 1e-4      # Standard scientific rate
EPOCHS = 5                # Increased to ensure convergence
DATA_PATH = "data/mega_scale_cdna/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"

# Use 100,000 samples for the perfect balance of Speed vs Accuracy
MAX_SAMPLES = 100000 

def train():
    print("--- 1. INITIALIZING PRO TRAINING ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Strategy: Gradient Accumulation (Effective Batch Size = {BATCH_SIZE * ACCUMULATION_STEPS})")
    
    model = HoloGNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- 2. PREPARING DATA ---")
    full_dataset = MegaScaleDataset(DATA_PATH)
    
    # Select the optimal subset
    subset_indices = range(MAX_SAMPLES)
    active_dataset = Subset(full_dataset, subset_indices)
    
    # 90/10 Split
    train_size = int(0.9 * len(active_dataset))
    val_size = len(active_dataset) - train_size
    train_dataset, val_dataset = random_split(active_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0) # Validation doesn't need accumulation

    print(f"Training on: {len(train_dataset)} sequences")
    
    print("--- 3. STARTING TRAINING LOOP ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad() # Initialize gradients
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, batch in enumerate(progress_bar):
            # Move data to GPU
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Setup Data
            class DataBatch: pass
            data = DataBatch()
            data.input_ids = input_ids
            data.mask = mask
            # Mechanistic features (B, L, 3) are REQUIRED by the backbone.
            data.mechanistic_features = batch['mechanistic_features'].to(device)
            data.edge_index = None # Trigger Graph Builder

            # Forward Pass — single-sequence absolute ΔG regression (MegaScale).
            predictions = model(data, task="stability")
            loss = criterion(predictions.squeeze(-1), labels)
            
            # --- THE TRICK: NORMALIZE LOSS ---
            # We divide loss by steps so the gradients don't get too big
            loss = loss / ACCUMULATION_STEPS 
            loss.backward()
            
            # Only update weights every 'ACCUMULATION_STEPS'
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Add back the full loss for reporting
            running_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})
        
        # Save "Golden" Checkpoint (This name matches predict.py)
        torch.save(model.state_dict(), "holognn_stability_final.pth")
        print(f"Epoch {epoch+1} Saved.")

    print(f"--- COMPLETE. Time: {(time.time() - start_time)/3600:.2f} hours ---")

if __name__ == "__main__":
    train()