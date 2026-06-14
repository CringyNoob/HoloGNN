"""
evaluate.py
===========
Held-out **evaluation** for a trained Holo-GNN checkpoint — the "is the model
actually good on this dataset?" tool.  It loads a ``.pth``, builds the matching
dataset, runs inference over a **reproducible, leakage-free held-out split**, and
prints + saves the right metrics for the task:

    regression  (stability / ddg / proteomics) → Pearson, Spearman, RMSE, MAE, R²
    classification (pathogenicity / ClinVar)    → AUROC, AUPRC, F1, precision,
                                                  recall, accuracy, MCC

What changed vs. the previous version
-------------------------------------
* **Head-trust guard.**  Because :class:`HoloGNN` instantiates *all* heads, a
  bare ``state_dict`` load silently brought in *untrained* (random) head weights
  for whichever task wasn't trained, yielding believable-looking noise.  We now
  read the checkpoint's provenance metadata (see ``src/checkpoint.py``) and
  **refuse** to score a head that was never trained (override with
  ``--allow-untrained-head``).  Legacy checkpoints with no metadata get a loud
  warning.
* **Reproducible split.**  Uses ``src/splits.py`` to take the deterministic
  ``test`` partition — disjoint from what training consumed (same seed/fractions)
  — instead of the old biased "tail" slice.  ``--split tail`` restores the old
  behaviour for legacy checkpoints.
* **Speed.**  fp16 AMP autocast + ``inference_mode`` + TF32, larger default batch,
  and parallel pinned data loading (RAM-safe under the 16 GB WSL cap).
* **Trustworthy pathogenicity.**  Prefers a dedicated ClinVar-trained
  ``idr_head`` (task ``pathogenicity``); falls back to the ``sigmoid(-ΔΔG)``
  proxy only when no classifier head was trained.
* **Richer reporting.**  Bootstrap confidence intervals per metric, tuned
  classification threshold (chosen on the val split), and a ``--task all`` runner.

Examples
--------
    python evaluate.py --task stability     --data CLEANED_DATA/mega_scale_clean.parquet
    python evaluate.py --task ddg           --data CLEANED_DATA/fireprotdb_clean.parquet
    python evaluate.py --task pathogenicity --data CLEANED_DATA/clinvar_clean.parquet
    python evaluate.py --task all           --data-dir CLEANED_DATA
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.device import describe_device
from src.full_model import HoloGNN
from src.checkpoint import load_checkpoint, TASK_TO_HEADS
from src.metrics import (
    regression_metrics, classification_metrics, format_report,
    bootstrap_ci, best_threshold,
)
from src.splits import indices_for, DEFAULT_SEED

# UTF-8 safe stdout (✓ / … crash on cp1252 consoles).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

REGRESSION_TASKS = ("stability", "ddg", "proteomics")
CLASSIFICATION_TASKS = ("pathogenicity",)
ALL_TASKS = ("stability", "ddg", "pathogenicity", "proteomics")

# Default filenames probed by `--task all` inside `--data-dir`.
ALL_TASK_DATA = {
    "stability":     "mega_scale_clean.parquet",
    "ddg":           "fireprotdb_clean.parquet",
    "pathogenicity": "clinvar_clean.parquet",
    # proteomics needs an .sptxt directory; skipped in all-mode unless present.
}


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
class _DataBatch:
    """Lightweight attribute bag matching what the backbone expects."""
    pass


def _single_batch(batch, device):
    d = _DataBatch()
    d.input_ids            = batch["input_ids"].to(device, non_blocking=True)
    d.mask                 = batch["attention_mask"].to(device, non_blocking=True)
    d.mechanistic_features = batch["mechanistic_features"].to(device, non_blocking=True)
    d.edge_index           = None
    return d


def _pair_batch(batch, suffix, device):
    d = _DataBatch()
    d.input_ids            = batch[f"input_ids_{suffix}"].to(device, non_blocking=True)
    d.mask                 = batch[f"attention_mask_{suffix}"].to(device, non_blocking=True)
    d.mechanistic_features = batch[f"mechanistic_features_{suffix}"].to(device, non_blocking=True)
    d.edge_index           = None
    return d


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
# Head-trust resolution
# ---------------------------------------------------------------------------
def _resolve_pathogenicity_mode(meta, patho_mode):
    """Return ('head'|'proxy', required_head_name) for the pathogenicity task."""
    trained = meta.get("trained_heads")
    if patho_mode == "head":
        return "head", "idr_head"
    if patho_mode == "proxy":
        return "proxy", "siamese_head"
    # auto: prefer a trained classifier head, else the ΔΔG proxy.
    if trained and "idr_head" in trained:
        return "head", "idr_head"
    return "proxy", "siamese_head"


def _required_head(task, meta, patho_mode):
    if task == "pathogenicity":
        _, head = _resolve_pathogenicity_mode(meta, patho_mode)
        return head
    # one head per regression task
    return TASK_TO_HEADS[task][0]


def _head_trust(required_head, meta):
    """'ok' | 'legacy' | 'untrained'."""
    trained = meta.get("trained_heads")
    if trained is None:
        return "legacy"
    return "ok" if required_head in trained else "untrained"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _run(task, model, loader, device, *, use_amp, patho_head):
    """Return (y_true, y_pred_or_prob) lists for the given task."""
    cuda = device.type == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if (use_amp and cuda) else nullcontext())
    y_true, y_out = [], []
    with torch.inference_mode():
        for batch in loader:
            with amp_ctx:
                if task in ("stability", "proteomics"):
                    preds = model(_single_batch(batch, device), task=task).squeeze(-1)
                    out = preds
                elif task == "ddg":
                    data_wt = _pair_batch(batch, "wt", device)
                    data_mt = _pair_batch(batch, "mt", device)
                    res = model((data_wt, data_mt), task="idr")
                    out = res[0].squeeze(-1)                    # dG_wt_to_mt
                elif task == "pathogenicity":
                    data_wt = _pair_batch(batch, "wt", device)
                    data_mt = _pair_batch(batch, "mt", device)
                    if patho_head == "head":
                        logit = model((data_wt, data_mt), task="pathogenicity")
                        out = torch.sigmoid(logit.squeeze(-1))
                    else:  # proxy: destabilisation score → pathogenicity prob
                        res = model((data_wt, data_mt), task="idr")
                        out = torch.sigmoid(-res[0].squeeze(-1))
                else:
                    raise ValueError(f"Unknown task '{task}'")
            y_out.extend(out.float().cpu().tolist())
            y_true.extend(batch["label"].tolist())
    return y_true, y_out


def _make_loader(subset, batch_size, num_workers, cuda):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


# ---------------------------------------------------------------------------
# Single-task evaluation
# ---------------------------------------------------------------------------
def evaluate_one(task, data_path, model, meta, device, args, *, all_mode=False):
    """Evaluate one task; return a payload dict or None if skipped."""
    is_cls = task in CLASSIFICATION_TASKS
    cuda = device.type == "cuda"

    # --- head-trust guard -------------------------------------------------
    patho_head = None
    if task == "pathogenicity":
        patho_head, required = _resolve_pathogenicity_mode(meta, args.patho_mode)
    else:
        required = _required_head(task, meta, args.patho_mode)
    trust = _head_trust(required, meta)

    if trust == "untrained" and not args.allow_untrained_head:
        msg = (f"[eval] checkpoint was trained for {meta.get('trained_task')!r} "
               f"(heads={meta.get('trained_heads')}); task '{task}' needs an UNTRAINED "
               f"head '{required}'. Its random weights would score pure noise.")
        if all_mode:
            print(msg + "  → skipping.")
            return None
        raise SystemExit(msg + "\n        Pass --allow-untrained-head to force, or "
                               "evaluate a checkpoint trained for this task.")
    if trust == "legacy":
        warnings.warn(
            f"[eval] '{Path(args.weights).name}' has no provenance metadata "
            f"(legacy checkpoint). Cannot verify head '{required}' was trained for "
            f"'{task}'; results may be meaningless if it was not.", RuntimeWarning)
    if task == "pathogenicity":
        print(f"[eval] pathogenicity scoring mode: {patho_head} "
              f"({'trained classifier' if patho_head == 'head' else 'sigmoid(-ΔΔG) proxy'})")

    # --- split seed reconciliation ---------------------------------------
    split_seed = args.split_seed
    if split_seed is None:
        split_seed = meta.get("split_seed") if meta.get("split_seed") is not None else DEFAULT_SEED
    if meta.get("split_seed") is not None and split_seed != meta["split_seed"]:
        warnings.warn(f"[eval] --split-seed {split_seed} differs from the checkpoint's "
                      f"training seed {meta['split_seed']}; the test slice may overlap "
                      f"training data.", RuntimeWarning)

    # --- data + held-out test partition ----------------------------------
    dataset = _build_dataset(task, data_path, args.max_length)
    n = len(dataset)
    if meta.get("dataset_n") is not None and meta["dataset_n"] != n:
        warnings.warn(f"[eval] dataset has {n} rows but the checkpoint was split on "
                      f"{meta['dataset_n']} rows; index-based split may not be disjoint. "
                      f"Re-run on the same data file.", RuntimeWarning)

    if args.split == "tail":
        test_idx = list(range(max(0, n - args.max_samples), n))
    else:
        test_idx = indices_for(n, args.split, seed=split_seed,
                               max_samples=args.max_samples).tolist()
    test_set = Subset(dataset, test_idx)
    loader = _make_loader(test_set, args.batch_size, args.num_workers, cuda)
    print(f"[eval] {task}: {len(test_set):,} held-out '{args.split}' samples "
          f"(of {n:,}, seed={split_seed}) …")

    y_true, y_out = _run(task, model, loader, device,
                         use_amp=not args.no_amp, patho_head=patho_head)

    # --- score ------------------------------------------------------------
    if is_cls:
        threshold = 0.5
        if args.tune_threshold:
            val_idx = indices_for(n, "val", seed=split_seed,
                                  max_samples=args.max_samples).tolist()
            v_true, v_out = _run(task, model,
                                 _make_loader(Subset(dataset, val_idx),
                                              args.batch_size, args.num_workers, cuda),
                                 device, use_amp=not args.no_amp, patho_head=patho_head)
            tuned = best_threshold(v_true, v_out, objective="mcc")
            if tuned["threshold"] == tuned["threshold"]:   # not NaN
                threshold = tuned["threshold"]
                print(f"[eval] tuned threshold (max MCC on val): {threshold:.4f}")
        metrics = classification_metrics(y_true, y_out, threshold=threshold)
        metrics["threshold"] = threshold
        ci_metrics = ("auroc", "auprc", "mcc")
        ci = (bootstrap_ci(y_true, y_out, kind="classification", metrics=ci_metrics,
                           n_boot=args.bootstrap, threshold=threshold)
              if args.bootstrap else {})
        title = f"{task} (classification · {patho_head})"
    else:
        metrics = regression_metrics(y_true, y_out)
        ci_metrics = ("pearson", "spearman", "rmse")
        ci = (bootstrap_ci(y_true, y_out, kind="regression", metrics=ci_metrics,
                           n_boot=args.bootstrap)
              if args.bootstrap else {})
        title = f"{task} (regression)"

    print(format_report(metrics, title, ci))
    return {
        "task": task, "weights": args.weights, "data": data_path,
        "trained_task": meta.get("trained_task"), "split": args.split,
        "split_seed": split_seed, "n_eval": len(test_set),
        "metrics": metrics, "ci95": ci,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate a Holo-GNN checkpoint on a dataset.")
    ap.add_argument("--task", required=True, choices=list(ALL_TASKS) + ["all"])
    ap.add_argument("--data", default="", help="Dataset path (parquet/csv) or .sptxt dir.")
    ap.add_argument("--data-dir", default="CLEANED_DATA",
                    help="Directory probed for per-task files when --task all.")
    ap.add_argument("--weights", default=os.environ.get("HOLOGNN_WEIGHTS",
                                                         "holognn_stability_final.pth"))
    ap.add_argument("--max-samples", type=int, default=5000,
                    help="Evaluate on at most this many held-out samples per task.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=0, help="0 = task default.")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "tail"],
                    help="Held-out partition (default 'test'). 'tail' = legacy slice.")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="Override split seed (default: checkpoint's, else 42).")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="Bootstrap resamples for CIs (0 disables).")
    ap.add_argument("--no-tune-threshold", dest="tune_threshold", action="store_false",
                    help="Disable val-tuned classification threshold (use 0.5).")
    ap.add_argument("--patho-mode", default="auto", choices=["auto", "head", "proxy"],
                    help="pathogenicity scoring: trained head, ΔΔG proxy, or auto.")
    ap.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast.")
    ap.add_argument("--allow-untrained-head", action="store_true",
                    help="Score even if the task's head was not trained (unsafe).")
    ap.add_argument("--output", default="", help="metrics JSON path (default metrics_<task>.json).")
    args = ap.parse_args()

    print(f"--- HOLO-GNN EVALUATION · task={args.task} ---")
    device = describe_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # --- model (built once, reused across tasks) -------------------------
    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Weights not found: {args.weights}\n"
            "Train first (train.py / train_siamese.py) or pass --weights.")
    _peek = torch.load(args.weights, map_location="cpu")
    cfg = (_peek.get("config") if isinstance(_peek, dict) else None) or {}
    import inspect
    valid = {k: v for k, v in cfg.items()
             if k in inspect.signature(HoloGNN.__init__).parameters}
    del _peek
    model = HoloGNN(**valid).to(device)
    load_result, meta = load_checkpoint(args.weights, model, device, strict=False)
    print(f"[eval] loaded {args.weights} "
          f"(trained_task={meta.get('trained_task')}, "
          f"missing={len(load_result.missing_keys)}, "
          f"unexpected={len(load_result.unexpected_keys)})")
    model.eval()

    # --- resolve task list ------------------------------------------------
    if args.task == "all":
        jobs = []
        for t, fname in ALL_TASK_DATA.items():
            p = Path(args.data_dir) / fname
            if p.exists():
                jobs.append((t, str(p)))
            else:
                print(f"[eval] {t}: data not found at {p} → skipping.")
        if not jobs:
            raise SystemExit(f"No task data found under {args.data_dir}.")
    else:
        if not args.data:
            raise SystemExit("--data is required (or use --task all with --data-dir).")
        jobs = [(args.task, args.data)]

    # --- run --------------------------------------------------------------
    results = {}
    for task, data_path in jobs:
        payload = evaluate_one(task, data_path, model, meta, device, args,
                               all_mode=(args.task == "all"))
        if payload is not None:
            results[task] = payload
            if args.task != "all":
                out_path = args.output or f"metrics_{task}.json"
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"[eval] metrics written → {out_path}")

    if args.task == "all":
        out_path = args.output or "metrics_all.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] combined metrics written → {out_path}")
        # compact summary table
        print("\n=== SUMMARY ===")
        for task, p in results.items():
            m = p["metrics"]
            head = (f"AUROC={m.get('auroc'):.4f}" if task in CLASSIFICATION_TASKS
                    else f"Spearman={m.get('spearman'):.4f}")
            print(f"  {task:<14} n={p['n_eval']:<6} {head}")


if __name__ == "__main__":
    main()
