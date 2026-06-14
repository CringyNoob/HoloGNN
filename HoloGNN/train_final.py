import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time

# Custom Modules
from src.dataset import MegaScaleDataset
from src.full_model import HoloGNN
from src.device import describe_device
from src.metrics import regression_metrics, format_report
from src.splits import split_indices, DEFAULT_SEED
from src.checkpoint import save_checkpoint

# --- PRO SETTINGS FOR GTX 1050 Ti ---
BATCH_SIZE = 4            # Physical limit of your GPU
ACCUMULATION_STEPS = 8    # 4 * 8 = Effective Batch Size of 32 (Stable!)
LEARNING_RATE = 1e-4      # Standard scientific rate
EPOCHS = 5                # Increased to ensure convergence
DATA_PATH = "CLEANED_DATA/mega_scale_clean.parquet"

# Use 100,000 samples for the perfect balance of Speed vs Accuracy
MAX_SAMPLES = 100000

# Shared with evaluate.py so the held-out test partition is disjoint by construction.
SPLIT_SEED = DEFAULT_SEED
# Distinct name so it does not clobber the Siamese checkpoint the app relies on.
SAVE_PATH  = "holognn_stability_regression.pth"

def train():
    print("--- 1. INITIALIZING PRO TRAINING ---")
    device = describe_device()
    print(f"Strategy: Gradient Accumulation (Effective Batch Size = {BATCH_SIZE * ACCUMULATION_STEPS})")
    
    model = HoloGNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- 2. PREPARING DATA ---")
    full_dataset = MegaScaleDataset(DATA_PATH)
    
    # Deterministic, leakage-free split shared with evaluate.py (src/splits.py).
    total_available = len(full_dataset)
    splits = split_indices(total_available, seed=SPLIT_SEED)
    train_idx, val_idx = splits["train"], splits["val"]
    if MAX_SAMPLES and MAX_SAMPLES < len(train_idx):
        train_idx = train_idx[:MAX_SAMPLES]
    train_dataset = Subset(full_dataset, train_idx.tolist())
    val_dataset   = Subset(full_dataset, val_idx.tolist())
    
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
        
        # --- Validation: held-out regression metrics ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                class DataBatch: pass
                data = DataBatch()
                data.input_ids = batch['input_ids'].to(device)
                data.mask = batch['attention_mask'].to(device)
                data.mechanistic_features = batch['mechanistic_features'].to(device)
                data.edge_index = None
                preds = model(data, task="stability").squeeze(-1)
                val_preds.extend(preds.detach().cpu().tolist())
                val_labels.extend(batch['label'].tolist())
        print(format_report(regression_metrics(val_labels, val_preds),
                            f"Epoch {epoch+1} validation (stability dG)"))

        # Save "Golden" Checkpoint with provenance metadata (name matches predict.py)
        save_checkpoint(SAVE_PATH, model, trained_task="stability",
                        trained_heads=["stability_head"], split_seed=SPLIT_SEED,
                        dataset_n=total_available)
        print(f"Epoch {epoch+1} Saved.")

    print(f"--- COMPLETE. Time: {(time.time() - start_time)/3600:.2f} hours ---")

if __name__ == "__main__":
    train()