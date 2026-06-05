import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- V6 Architecture Imports ---
from src.device import describe_device
from src.full_model import HoloGNN
from src.dataset import MegaScaleDataset, _make_batch  # Adjust import based on your dataset.py
from src.metrics import regression_metrics, format_report

# --- PRODUCTION CONFIGURATION ---
DATA_PATH     = "CLEANED_DATA/mega_scale_clean.parquet"
BATCH_SIZE    = 32     # VRAM Constraint (16GB RTX 5070 Ti)
NUM_WORKERS   = 4      # RAM Advantage (32GB System RAM)
EPOCHS        = 5
LEARNING_RATE = 1e-4
SAVE_PATH     = "holognn_stability_final.pth"

def train():
    print("--- MEGA-SCALE STABILITY TRAINING (V6 FULL PIPELINE) ---")
    
    # 1. Hardware Verification (Blackwell sm_120 Check)
    device = describe_device()

    # 2. V6 Model Instantiation
    # We omit antisym_head=True because this is single-sequence absolute ΔG, not Siamese paired.
    model = HoloGNN(
        mech_feature_dim=6,      # V6: Expanded protein-only descriptors (Hydropathy, Volume, Helix)
        freeze_esm_layers=4,     # V6: Lock bottom 4 layers to save VRAM and prevent overfitting
    ).to(device)

    # Standard MSE for regression stability
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Hardware Optimization: Mixed Precision for Tensor Cores
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # 3. Data Pipeline
    print(f"Loading MegaScale dataset from {DATA_PATH}...")
    dataset = MegaScaleDataset(DATA_PATH)
    n_val   = max(1, int(0.05 * len(dataset))) # 5% of millions of rows is plenty for validation
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    # Leveraging the 32GB RAM: 4 workers with prefetching
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    print(f"Train: {n_train:,} samples | Val: {n_val:,} samples")

    # 4. Training Loop
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in bar:
            # Unpack batch (Ensure _make_batch matches your dataset.py structure for MegaScale)
            data   = _make_batch(batch, device) 
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            
            # Forward pass with Automatic Mixed Precision (AMP)
            with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                # Explicitly route to the stability regression head
                preds = model(data, task="stability") 
                loss  = criterion(preds.squeeze(-1), labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            bar.set_postfix({"loss": run_loss / (bar.n + 1)})

        # 5. Validation & V6 Metrics
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        print("\nRunning Validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                data   = _make_batch(batch, device)
                labels = batch["label"].to(device)
                
                with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    preds = model(data, task="stability")
                    l = criterion(preds.squeeze(-1), labels)
                
                val_loss += l.item()
                val_preds.extend(preds.squeeze(-1).detach().cpu().tolist())
                val_labels.extend(labels.detach().cpu().tolist())
                
        # V6 Shared Metrics Reporting
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Val Loss (MSE): {val_loss/max(1, len(val_loader)):.4f}")
        print(format_report(regression_metrics(val_labels, val_preds), "MegaScale Held-Out Split"))

        # Save Checkpoint
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Checkpoint saved → {SAVE_PATH}\n")

    print(f"--- TRAINING COMPLETE in {(time.time()-start)/3600:.2f} h ---")

if __name__ == "__main__":
    train()