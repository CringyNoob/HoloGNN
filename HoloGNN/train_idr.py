"""
train_idr.py
============
Train a **dedicated ClinVar pathogenicity classifier** on the Holo-GNN
``idr_head`` (``EnsembleIDRHead``), so evaluation no longer has to abuse the
ΔΔG stability head as a ``sigmoid(-ΔΔG)`` proxy.

The model encodes the REF/ALT allele pair through the shared backbone and runs
the ``pathogenicity`` task route (``src/full_model.py``) → a single logit trained
with ``BCEWithLogitsLoss`` against the ClinVar benign(0)/pathogenic(1) label.

Design
------
* Warm-starts the backbone from an existing checkpoint (``--init``, default the
  stability/siamese weights) so it inherits learned protein representations.
* Freezes the backbone by default (fast, avoids forgetting) and trains only the
  small ``idr_head``; ``--unfreeze`` fine-tunes the whole stack.
* Uses the **same deterministic split** as evaluate.py (``src/splits.py``) — it
  trains on the ``train`` partition; evaluate.py scores the disjoint ``test`` one.
* fp16 AMP for speed on the RTX 5070 Ti; atomic, metadata-tagged checkpoints.

Run:
    python train_idr.py --data CLEANED_DATA/clinvar_clean.parquet
    python evaluate.py  --task pathogenicity --data CLEANED_DATA/clinvar_clean.parquet \
                        --weights holognn_pathogenicity.pth
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.dataset import ClinVarDataset
from src.full_model import HoloGNN
from src.device import describe_device
from src.metrics import classification_metrics, format_report, best_threshold
from src.splits import split_indices, DEFAULT_SEED
from src.checkpoint import save_checkpoint, load_checkpoint

LOG_EVERY = 25   # print a plain progress line every N steps


def _make_batch(batch, suffix, device):
    class DataBatch:
        pass
    d = DataBatch()
    d.input_ids            = batch[f"input_ids_{suffix}"].to(device, non_blocking=True)
    d.mask                 = batch[f"attention_mask_{suffix}"].to(device, non_blocking=True)
    d.mechanistic_features = batch[f"mechanistic_features_{suffix}"].to(device, non_blocking=True)
    d.edge_index           = None
    return d


def train(args):
    print("--- ClinVar PATHOGENICITY HEAD TRAINING ---")
    device = describe_device()
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HoloGNN().to(device)
    if args.init and os.path.exists(args.init):
        _, meta = load_checkpoint(args.init, model, device, strict=False)
        print(f"[ok] warm-started backbone from {args.init} "
              f"(source task: {meta.get('trained_task')}).")
    else:
        print(f"[warn] no init weights at {args.init!r}; training from scratch.")

    # Freeze backbone unless explicitly fine-tuning the whole stack.
    if not args.unfreeze:
        for p in model.backbone.parameters():
            p.requires_grad = False
        trainable = list(model.idr_head.parameters())
        print("Backbone frozen; training idr_head only.")
    else:
        trainable = list(model.parameters())
        print("Fine-tuning the whole model.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    dataset = ClinVarDataset(args.data, max_length=args.max_length)
    n_total = len(dataset)
    splits = split_indices(n_total, seed=args.seed)
    train_loader = DataLoader(Subset(dataset, splits["train"].tolist()),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=cuda)
    val_loader = DataLoader(Subset(dataset, splits["val"].tolist()),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=cuda)
    print(f"Train: {len(splits['train']):,} | Val: {len(splits['val']):,} variants")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        n_steps = len(train_loader)
        t_ep = time.time()
        for i, batch in enumerate(train_loader, 1):
            data_wt = _make_batch(batch, "wt", device)
            data_mt = _make_batch(batch, "mt", device)
            labels  = batch["label"].float().to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if cuda else "cpu",
                                    dtype=torch.float16, enabled=cuda):
                logits = model((data_wt, data_mt), task="pathogenicity")
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
            if i % LOG_EVERY == 0 or i == n_steps:
                rate = i / max(time.time() - t_ep, 1e-9)
                print(f"  Epoch {epoch}/{args.epochs} | step {i}/{n_steps} | "
                      f"loss={run_loss/i:.4f} | {rate:.1f} it/s", flush=True)

        # --- validation ---
        model.eval()
        v_true, v_prob = [], []
        with torch.inference_mode():
            for batch in val_loader:
                data_wt = _make_batch(batch, "wt", device)
                data_mt = _make_batch(batch, "mt", device)
                with torch.amp.autocast(device_type="cuda" if cuda else "cpu",
                                        dtype=torch.float16, enabled=cuda):
                    logits = model((data_wt, data_mt), task="pathogenicity")
                v_prob.extend(torch.sigmoid(logits).float().cpu().tolist())
                v_true.extend(batch["label"].tolist())
        thr = best_threshold(v_true, v_prob, objective="mcc").get("threshold", 0.5)
        print(format_report(classification_metrics(v_true, v_prob, threshold=thr),
                            f"Epoch {epoch} validation (pathogenicity @ thr={thr:.3f})"))

        save_checkpoint(args.output, model, trained_task="pathogenicity",
                        trained_heads=["idr_head"], split_seed=args.seed,
                        dataset_n=n_total)
        print(f"  checkpoint saved -> {args.output}")

    print(f"--- COMPLETE in {(time.time()-start)/3600:.2f} h ---")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the ClinVar pathogenicity head.")
    ap.add_argument("--data", default="CLEANED_DATA/clinvar_clean.parquet")
    ap.add_argument("--init", default="holognn_stability_final.pth",
                    help="Checkpoint to warm-start the backbone from.")
    ap.add_argument("--output", default="holognn_pathogenicity.pth")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--unfreeze", action="store_true",
                    help="Fine-tune the whole model (default: train idr_head only).")
    args = ap.parse_args()
    train(args)
