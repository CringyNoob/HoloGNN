"""
train.py
========
Lightweight single-sequence absolute ΔG trainer for Holo-GNN on MegaScale data.
Similar to train_final.py but without gradient accumulation (simpler loop).

Run:
    python train.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time

from src.dataset import MegaScaleDataset
from src.full_model import HoloGNN
from src.device import describe_device
from src.metrics import regression_metrics, format_report

# --- CONFIGURATION ---
BATCH_SIZE    = 4
LEARNING_RATE = 1e-4
EPOCHS        = 3
GRAD_CLIP     = 1.0
DATA_PATH     = "CLEANED_DATA/mega_scale_clean.parquet"
MAX_SAMPLES   = 200000
SAVE_DIR      = "checkpoints"


def train():
    print("--- 1. INITIALIZING HOLO-GNN ---")
    device = describe_device()
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    model = HoloGNN(
        mech_feature_dim=6,
        freeze_esm_layers=4,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- 2. LOADING DATASET ---")
    full_dataset = MegaScaleDataset(DATA_PATH, expanded_mech=True)
    total_available = len(full_dataset)

    if MAX_SAMPLES and MAX_SAMPLES < total_available:
        print(f"Using {MAX_SAMPLES} samples (out of {total_available}) for efficient training.")
        active_dataset = Subset(full_dataset, range(MAX_SAMPLES))
    else:
        print(f"Using all {total_available} samples.")
        active_dataset = full_dataset

    train_size = int(0.9 * len(active_dataset))
    val_size   = len(active_dataset) - train_size
    train_dataset, val_dataset = random_split(active_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, num_workers=0)
    print(f"Training on: {len(train_dataset)} | Validating on: {len(val_dataset)}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    print("--- 3. STARTING TRAINING ---")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)
            labels    = batch['label'].to(device)

            class DataBatch: pass
            data = DataBatch()
            data.input_ids            = input_ids
            data.mask                 = mask
            data.mechanistic_features = batch['mechanistic_features'].to(device)
            data.edge_index           = None

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                predictions = model(data, task="stability")
                loss = criterion(predictions.squeeze(-1), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

        # --- Validation ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                class DataBatch: pass
                data = DataBatch()
                data.input_ids            = batch['input_ids'].to(device)
                data.mask                 = batch['attention_mask'].to(device)
                data.mechanistic_features = batch['mechanistic_features'].to(device)
                data.edge_index           = None
                preds = model(data, task="stability").squeeze(-1)
                val_preds.extend(preds.detach().cpu().tolist())
                val_labels.extend(batch['label'].tolist())
        print(format_report(regression_metrics(val_labels, val_preds),
                            f"Epoch {epoch+1} validation (stability ΔG)"))

        ckpt_path = os.path.join(SAVE_DIR, f"holognn_stability_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    total_time = (time.time() - start_time) / 3600
    print(f"--- TRAINING COMPLETE in {total_time:.2f} hours ---")

if __name__ == "__main__":
    train()