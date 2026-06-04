"""
evaluate.py
===========
Held-out **evaluation** for a trained Holo-GNN checkpoint — the "is the model
actually good on this dataset?" tool.  It loads a ``.pth``, builds the matching
dataset, runs inference over a held-out tail split, and prints + saves the right
metrics for the task:

    regression  (stability / ddg / proteomics) → Pearson, Spearman, RMSE, MAE, R²
    classification (pathogenicity / ClinVar)    → AUROC, AUPRC, F1, precision,
                                                  recall, accuracy, MCC

Tasks → datasets
----------------
    stability       MegaScaleDataset       (K50 / Processed_K50_dG; single-seq ΔG)
    ddg             FireProtDataset        (FireProtDB; Siamese ΔΔG pairs)
    pathogenicity   ClinVarDataset         (clinvar_clean.parquet; benign/pathogenic)
    proteomics      MassIVEKBDataset       (ProteoSAFe .sptxt library; RT regression)

For **pathogenicity** the model has no dedicated ClinVar-trained head, so we use
its Siamese **ΔΔG as a destabilisation score** (pathogenic ≈ destabilising):
``prob_pathogenic = sigmoid(-ΔΔG)``.  AUROC/AUPRC are threshold-free (any
monotonic score works); the 0.5 threshold ⇔ "destabilising → pathogenic".

Examples
--------
    python evaluate.py --task stability     --weights holognn_stability_final.pth \
                       --data data/.../mega_scale_clean.parquet
    python evaluate.py --task ddg           --data data/fireprotdb/fireprotdb_clean.parquet
    python evaluate.py --task pathogenicity --data CLEANED_DATA/clinvar_clean.parquet
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from src.device import describe_device
from src.full_model import HoloGNN
from src.metrics import regression_metrics, classification_metrics, format_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _DataBatch:
    """Lightweight attribute bag matching what the backbone expects."""
    pass


def _single_batch(batch, device):
    d = _DataBatch()
    d.input_ids            = batch["input_ids"].to(device)
    d.mask                 = batch["attention_mask"].to(device)
    d.mechanistic_features = batch["mechanistic_features"].to(device)
    d.edge_index           = None
    return d


def _pair_batch(batch, suffix, device):
    d = _DataBatch()
    d.input_ids            = batch[f"input_ids_{suffix}"].to(device)
    d.mask                 = batch[f"attention_mask_{suffix}"].to(device)
    d.mechanistic_features = batch[f"mechanistic_features_{suffix}"].to(device)
    d.edge_index           = None
    return d


def _tail_subset(dataset, max_samples):
    """Deterministic held-out split: the *tail* of the dataset.

    train.py / train_final.py train on the *head* (``range(MAX_SAMPLES)``), so a
    tail slice is disjoint from what those scripts see.
    """
    n = len(dataset)
    if max_samples and max_samples < n:
        return Subset(dataset, range(n - max_samples, n))
    return dataset


def _build_dataset(task, data_path, max_length):
    if task == "stability":
        from src.dataset import MegaScaleDataset
        return MegaScaleDataset(data_path, max_length=max_length or 100)
    if task == "ddg":
        from src.dataset import FireProtDataset
        return FireProtDataset(data_path, max_length=max_length or 100)
    if task == "pathogenicity":
        from src.dataset import ClinVarDataset
        return ClinVarDataset(data_path, max_length=max_length or 64)
    if task == "proteomics":
        from src.dataset import MassIVEKBDataset
        return MassIVEKBDataset(data_path, max_length=max_length or 100)
    raise ValueError(f"Unknown task '{task}'")


# ---------------------------------------------------------------------------
# Inference loops
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run(task, model, loader, device):
    """Return (y_true, y_pred_or_prob) lists for the given task."""
    y_true, y_out = [], []
    for batch in loader:
        if task == "stability":
            preds = model(_single_batch(batch, device), task="stability").squeeze(-1)
            y_out.extend(preds.cpu().tolist())
            y_true.extend(batch["label"].tolist())

        elif task == "proteomics":
            preds = model(_single_batch(batch, device), task="proteomics").squeeze(-1)
            y_out.extend(preds.cpu().tolist())
            y_true.extend(batch["label"].tolist())

        elif task == "ddg":
            data_wt = _pair_batch(batch, "wt", device)
            data_mt = _pair_batch(batch, "mt", device)
            ddg_fwd, _ = model((data_wt, data_mt), task="idr")
            y_out.extend(ddg_fwd.squeeze(-1).cpu().tolist())
            y_true.extend(batch["label"].tolist())

        elif task == "pathogenicity":
            data_wt = _pair_batch(batch, "wt", device)
            data_mt = _pair_batch(batch, "mt", device)
            ddg_fwd, _ = model((data_wt, data_mt), task="idr")
            # destabilisation score → pathogenicity probability
            prob = torch.sigmoid(-ddg_fwd.squeeze(-1))
            y_out.extend(prob.cpu().tolist())
            y_true.extend(batch["label"].tolist())
    return y_true, y_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate a Holo-GNN checkpoint on a dataset.")
    ap.add_argument("--task", required=True,
                    choices=["stability", "ddg", "pathogenicity", "proteomics"])
    ap.add_argument("--data", required=True, help="Dataset path (parquet/csv) or .sptxt dir.")
    ap.add_argument("--weights", default=os.environ.get("HOLOGNN_WEIGHTS",
                                                         "holognn_stability_final.pth"))
    ap.add_argument("--max-samples", type=int, default=5000,
                    help="Evaluate on at most this many held-out (tail) samples.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=0, help="0 = task default.")
    ap.add_argument("--output", default="", help="metrics JSON path (default metrics_<task>.json).")
    args = ap.parse_args()

    is_classification = args.task == "pathogenicity"

    print(f"--- HOLO-GNN EVALUATION · task={args.task} ---")
    device = describe_device()

    # --- model ---
    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            "Train first (train.py / train_siamese.py) or pass --weights.")
    model = HoloGNN().to(device)
    # strict=False so a checkpoint trained for a sibling task (e.g. the disease
    # classifier) still loads its shared backbone for evaluation.
    missing = model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    print(f"[eval] loaded {args.weights} (missing={len(missing.missing_keys)}, "
          f"unexpected={len(missing.unexpected_keys)})")
    model.eval()

    # --- data ---
    dataset = _build_dataset(args.task, args.data, args.max_length)
    test_set = _tail_subset(dataset, args.max_samples)
    loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)
    print(f"[eval] evaluating on {len(test_set):,} held-out samples "
          f"(of {len(dataset):,}) …")

    # --- run + score ---
    y_true, y_out = _run(args.task, model, loader, device)
    if is_classification:
        metrics = classification_metrics(y_true, y_out)
        title = f"{args.task} (classification)"
    else:
        metrics = regression_metrics(y_true, y_out)
        title = f"{args.task} (regression)"

    print(format_report(metrics, title))

    out_path = args.output or f"metrics_{args.task}.json"
    payload = {"task": args.task, "weights": args.weights, "data": args.data,
               "n_eval": len(test_set), "metrics": metrics}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] metrics written → {out_path}")


if __name__ == "__main__":
    main()
