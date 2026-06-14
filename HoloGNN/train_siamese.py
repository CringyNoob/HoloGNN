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
from torch.utils.data import DataLoader, Subset

from src.dataset import FireProtDataset
from src.full_model import HoloGNN
from src.loss import AntisymmetricLoss
from src.device import describe_device
from src.metrics import regression_metrics, format_report
from src.splits import split_indices, DEFAULT_SEED
from src.checkpoint import save_checkpoint

# --- CONFIGURATION ---
# NOTE: FireProtDB clean has NO sequences (SOURCE/TARGET all empty) → train on the
# MegaScale-derived WT/MT pairs built by build_megascale_pairs.py instead.
DATA_PATH     = "data/megascale_siamese/megascale_ddg_pairs.parquet"
BATCH_SIZE    = 32           # RTX 5070 Ti (16GB) easily fits this for ESM2-8M
NUM_WORKERS   = 0            # raise to 4 on WSL/Linux for a big data-loading speedup
LOG_EVERY     = 25           # print a plain progress line every N steps
LEARNING_RATE = 1e-4
EPOCHS        = 5
ALPHA         = 1.0          # antisymmetry-term weight
SPLIT_SEED    = DEFAULT_SEED  # shared with evaluate.py for a disjoint test split
SAVE_PATH     = "holognn_stability_final.pth"   # the file the app / predict.py load


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
    print("--- SIAMESE ΔΔG TRAINING ---")
    device = describe_device()

    model     = HoloGNN().to(device)
    criterion = AntisymmetricLoss(alpha=ALPHA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = FireProtDataset(DATA_PATH)
    # Deterministic, leakage-free split shared with evaluate.py (src/splits.py):
    # evaluate.py's disjoint 'test' partition removes the train/eval overlap the
    # old unseeded random_split caused for the ddg/pathogenicity tasks.
    n_total = len(dataset)
    splits  = split_indices(n_total, seed=SPLIT_SEED)
    train_ds = Subset(dataset, splits["train"].tolist())
    val_ds   = Subset(dataset, splits["val"].tolist())
    n_train, n_val = len(train_ds), len(val_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {n_train:,} pairs | Val: {n_val:,} pairs")

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        run_loss = run_anti = run_fid = 0.0
        n_steps = len(train_loader)
        t_ep = time.time()
        for step, batch in enumerate(train_loader, 1):
            data_wt = _make_batch(batch, "wt", device)
            data_mt = _make_batch(batch, "mt", device)
            labels  = batch["label"].to(device)

            optimizer.zero_grad()
            dG_fwd, dG_rev = model((data_wt, data_mt), task="idr")
            loss, comp = criterion(dG_fwd, dG_rev, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item(); run_anti += comp["antisymmetry"]; run_fid += comp["fidelity"]
            if step % LOG_EVERY == 0 or step == n_steps:
                rate = step / max(time.time() - t_ep, 1e-9)
                print(f"  Epoch {epoch+1}/{EPOCHS} | step {step}/{n_steps} | "
                      f"loss={run_loss/step:.4f} anti={run_anti/step:.4f} fid={run_fid/step:.4f} | "
                      f"{rate:.1f} it/s", flush=True)

        # --- validation: loss + held-out ΔΔG regression metrics ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                data_wt = _make_batch(batch, "wt", device)
                data_mt = _make_batch(batch, "mt", device)
                labels  = batch["label"].to(device)
                dG_fwd, dG_rev = model((data_wt, data_mt), task="idr")
                l, _ = criterion(dG_fwd, dG_rev, labels)
                val_loss += l.item()
                val_preds.extend(dG_fwd.squeeze(-1).detach().cpu().tolist())
                val_labels.extend(labels.detach().cpu().tolist())
        print(f"Epoch {epoch+1}: val_loss = {val_loss/max(1,len(val_loader)):.4f}")
        print(format_report(regression_metrics(val_labels, val_preds),
                            f"Epoch {epoch+1} validation (ddG)"))

        save_checkpoint(SAVE_PATH, model, trained_task="idr",
                        trained_heads=["siamese_head"], split_seed=SPLIT_SEED,
                        dataset_n=n_total)
        print(f"  checkpoint saved -> {SAVE_PATH}")

    print(f"--- COMPLETE in {(time.time()-start)/3600:.2f} h ---")


if __name__ == "__main__":
    train()
