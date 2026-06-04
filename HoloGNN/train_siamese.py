"""
train_siamese.py
================
Flagship Siamese ΔΔG trainer (the V4/V5 production objective).

Unlike train.py / train_final.py — which train the single-sequence absolute-ΔG
'stability' task on MegaScale — this script trains the **paired Siamese 'idr'
task** with the physics-constrained AntisymmetricLoss:

    L = (dG_wt_to_mt + dG_mt_to_wt)²  +  (dG_pred − dG_exp)²
        └ antisymmetry constraint ┘     └ regression fidelity ┘

It consumes FireProtDB (WT/MT sequence pairs + ΔΔG labels) produced by
master_etl_pipeline.py → fireprotdb_clean.parquet.

Run:
    python train_siamese.py
(Adjust DATA_PATH / BATCH_SIZE / EPOCHS below for your hardware.)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import FireProtDataset
from src.full_model import HoloGNN
from src.loss import AntisymmetricLoss

# --- CONFIGURATION ---
DATA_PATH     = "data/fireprotdb/fireprotdb_clean.parquet"
BATCH_SIZE    = 8
LEARNING_RATE = 1e-4
EPOCHS        = 5
ALPHA         = 1.0          # antisymmetry-term weight
SAVE_PATH     = "holognn_stability_final.pth"


def _make_batch(batch, suffix, device):
    """Build a DataBatch from the '<field>_wt' / '<field>_mt' dataset fields."""
    class DataBatch:
        pass
    d = DataBatch()
    d.input_ids            = batch[f"input_ids_{suffix}"].to(device)
    d.mask                 = batch[f"attention_mask_{suffix}"].to(device)
    d.mechanistic_features = batch[f"mechanistic_features_{suffix}"].to(device)
    d.edge_index           = None     # build the graph dynamically from attention
    return d


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SIAMESE ΔΔG TRAINING on {device} ---")

    model     = HoloGNN().to(device)
    criterion = AntisymmetricLoss(alpha=ALPHA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = FireProtDataset(DATA_PATH)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train: {n_train:,} pairs | Val: {n_val:,} pairs")

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        run_loss = run_anti = run_fid = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in bar:
            data_wt = _make_batch(batch, "wt", device)
            data_mt = _make_batch(batch, "mt", device)
            labels  = batch["label"].to(device)

            optimizer.zero_grad()
            dG_fwd, dG_rev = model((data_wt, data_mt), task="idr")
            loss, comp = criterion(dG_fwd, dG_rev, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item(); run_anti += comp["antisymmetry"]; run_fid += comp["fidelity"]
            n = bar.n + 1
            bar.set_postfix({"loss": run_loss/n, "anti": run_anti/n, "fid": run_fid/n})

        # --- validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data_wt = _make_batch(batch, "wt", device)
                data_mt = _make_batch(batch, "mt", device)
                labels  = batch["label"].to(device)
                dG_fwd, dG_rev = model((data_wt, data_mt), task="idr")
                l, _ = criterion(dG_fwd, dG_rev, labels)
                val_loss += l.item()
        print(f"Epoch {epoch+1}: val_loss = {val_loss/max(1,len(val_loader)):.4f}")

        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  checkpoint saved → {SAVE_PATH}")

    print(f"--- COMPLETE in {(time.time()-start)/3600:.2f} h ---")


if __name__ == "__main__":
    train()
