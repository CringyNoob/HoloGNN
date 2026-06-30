"""
train_final.py
==============
Single-sequence absolute ΔG trainer for Holo-GNN on MegaScale data.

Regresses the absolute free energy of folding (ΔG) for each cDNA variant.
Uses gradient accumulation for an effective batch size of 32.

Run:
    python train_final.py
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
BATCH_SIZE         = 4
ACCUMULATION_STEPS = 8        # effective batch = 4 * 8 = 32
LEARNING_RATE      = 1e-4
EPOCHS             = 5
GRAD_CLIP          = 1.0
DATA_PATH          = "CLEANED_DATA/mega_scale_clean.parquet"
MAX_SAMPLES        = 100000
SAVE_DIR           = "checkpoints"


def train():
    print("--- 1. INITIALIZING TRAINING ---")
    device = describe_device()
    use_amp = device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)
    print(f"Strategy: Gradient Accumulation (Effective Batch Size = {BATCH_SIZE * ACCUMULATION_STEPS})")

    model = HoloGNN(
        mech_feature_dim=6,
        freeze_esm_layers=4,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- 2. PREPARING DATA ---")
    full_dataset = MegaScaleDataset(DATA_PATH, expanded_mech=True)

    subset_indices = range(min(MAX_SAMPLES, len(full_dataset)))
    active_dataset = Subset(full_dataset, subset_indices)

    train_size = int(0.9 * len(active_dataset))
    val_size   = len(active_dataset) - train_size
    train_dataset, val_dataset = random_split(active_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, num_workers=0)
    print(f"Training on: {len(train_dataset)} sequences")

    os.makedirs(SAVE_DIR, exist_ok=True)
    print("--- 3. STARTING TRAINING LOOP ---")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)
            labels    = batch['label'].to(device)

            class DataBatch: pass
            data = DataBatch()
            data.input_ids            = input_ids
            data.mask                 = mask
            data.mechanistic_features = batch['mechanistic_features'].to(device)
            data.edge_index           = None

            with autocast(device_type=device.type, enabled=use_amp):
                predictions = model(data, task="stability")
                loss = criterion(predictions.squeeze(-1), labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})

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
        print(f"Epoch {epoch+1} saved → {ckpt_path}")

    print(f"--- COMPLETE. Time: {(time.time() - start_time)/3600:.2f} hours ---")

if __name__ == "__main__":
    train()