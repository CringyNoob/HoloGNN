import os
# --- CRITICAL WINDOWS FIX ---
# Prevents "OMP: Error #15" crash on Anaconda/Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time

# Custom Modules
from src.dataset import MegaScaleDataset
from src.full_model import HoloGNN
from src.device import describe_device
from src.metrics import regression_metrics, format_report
from src.splits import split_indices, DEFAULT_SEED
from src.checkpoint import save_checkpoint

# --- CONFIGURATION FOR RTX 5070 Ti (16GB) ---
BATCH_SIZE = 32          # 5070 Ti easily fits this for ESM2-8M (was 4 for a 1050 Ti)
NUM_WORKERS = 0          # raise to 4 on WSL/Linux for faster data loading
LOG_EVERY = 25           # print a plain progress line every N steps
LEARNING_RATE = 1e-4
EPOCHS = 3               # 3 Epochs gives a strong research result
DATA_PATH = "CLEANED_DATA/mega_scale_clean.parquet"

# --- DATASET SIZE CONTROL ---
# Set to 200000 for Publication-Grade results (approx 4-5 hours)
# If you want a quick test, change this to 5000
MAX_SAMPLES = 200000

# Shared with evaluate.py so the held-out test partition is disjoint by construction.
SPLIT_SEED = DEFAULT_SEED
# Distinct name so the single-seq stability head does NOT clobber the Siamese
# checkpoint (holognn_stability_final.pth) that the app/predict.py rely on.
SAVE_PATH  = "holognn_stability_regression.pth"

def train():
    print("--- 1. INITIALIZING HOLO-GNN ---")
    device = describe_device()

    model = HoloGNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- 2. LOADING DATASET ---")
    full_dataset = MegaScaleDataset(DATA_PATH)
    total_available = len(full_dataset)
    
    # Deterministic, leakage-free split shared with evaluate.py (src/splits.py):
    # we train on the 'train' partition; evaluate.py scores the disjoint 'test' one.
    splits = split_indices(total_available, seed=SPLIT_SEED)
    train_idx, val_idx = splits["train"], splits["val"]
    if MAX_SAMPLES and MAX_SAMPLES < len(train_idx):
        print(f"!!! OPTIMIZATION: Using {MAX_SAMPLES} of {len(train_idx)} train samples "
              f"(of {total_available} total) for efficient training.")
        train_idx = train_idx[:MAX_SAMPLES]
    else:
        print(f"!!! FULL RUN: Using all {len(train_idx)} train samples (of {total_available}).")
    train_dataset = Subset(full_dataset, train_idx.tolist())
    val_dataset   = Subset(full_dataset, val_idx.tolist())
    
    # num_workers=0 is safest on native Windows; raise NUM_WORKERS on WSL/Linux.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Training on: {len(train_dataset)} sequences")
    print(f"Validating on: {len(val_dataset)} sequences")

    print("--- 3. STARTING PRODUCTION TRAINING ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        n_steps = len(train_loader)
        t_ep = time.time()
        for step, batch in enumerate(train_loader, 1):
            # Move to GPU
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Data Object
            class DataBatch: pass
            data = DataBatch()
            data.input_ids = input_ids
            data.mask = mask
            # Mechanistic features (B, L, 3) are REQUIRED by the backbone.
            data.mechanistic_features = batch['mechanistic_features'].to(device)
            # CRITICAL: Passing None forces the Backbone to build the Graph from Attention
            data.edge_index = None

            # Forward & Backward
            # MegaScale yields one sequence + one absolute ΔG label, so we use the
            # single-sequence 'stability' regression task (not the Siamese 'idr').
            optimizer.zero_grad()
            predictions = model(data, task="stability")
            loss = criterion(predictions.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if step % LOG_EVERY == 0 or step == n_steps:
                rate = step / max(time.time() - t_ep, 1e-9)
                print(f"  Epoch {epoch+1}/{EPOCHS} | step {step}/{n_steps} | "
                      f"loss={running_loss/step:.4f} | {rate:.1f} it/s", flush=True)
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")

        # --- Validation: held-out regression metrics (Pearson/Spearman/RMSE/MAE/R2) ---
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

        # Save with provenance metadata (trained_task/heads + split seed) so
        # evaluate.py can verify the right head was trained and reproduce the split.
        save_checkpoint(SAVE_PATH, model, trained_task="stability",
                        trained_heads=["stability_head"], split_seed=SPLIT_SEED,
                        dataset_n=total_available)
        print(f"Checkpoint saved: {SAVE_PATH}")

    total_time = (time.time() - start_time) / 3600
    print(f"--- TRAINING COMPLETE in {total_time:.2f} hours ---")

if __name__ == "__main__":
    train()