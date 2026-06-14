"""
train_clinvar_classifier.py
===========================
Pathogenicity classifier on the **rich ClinVar features** (consequence + allele
frequency + variant type + gene), produced by build_clinvar_features.py.

Why this is a separate model from HoloGNN
-----------------------------------------
ClinVar variants are 79% single-nucleotide with no protein context, so a
protein-sequence model (ESM2/GATv2) has nothing to learn from — the old
sequence approach capped at AUROC 0.61. The predictive signal is entirely
tabular: molecular consequence (synonymous/intron => benign, frameshift/nonsense
=> pathogenic) and population allele frequency (common => benign). A gradient-
boosted classifier on those features reaches AUROC ~0.99 in seconds on CPU, so
pathogenicity is shipped as its own feature model, decoupled from the GNN.

Leakage safety: uses the same deterministic seed-42 split as everything else
(src/splits), and the per-gene pathogenicity prior is computed on the TRAIN
partition only. Threshold is tuned on VAL; all reported numbers are on the
held-out TEST partition.

Run:
    python train_clinvar_classifier.py
    python train_clinvar_classifier.py --data CLEANED_DATA/clinvar_features.parquet
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from src.splits import split_indices, DEFAULT_SEED
from src.metrics import classification_metrics, best_threshold, bootstrap_ci, format_report

GENE_PRIOR_ALPHA = 20.0   # Bayesian smoothing toward the global rate


def build_feature_frame(df, cons_cats, vt_cats, gene_prior, global_mean):
    """Deterministic feature matrix (fixed column order) for train & inference."""
    cons = pd.Categorical(df["consequence"].astype(str), categories=cons_cats)
    cons_oh = pd.get_dummies(cons).astype("float32")
    cons_oh.columns = [f"cons={c}" for c in cons_cats]

    vt = pd.Categorical(df["variant_type"].astype(str), categories=vt_cats)
    vt_oh = pd.get_dummies(vt).astype("float32")
    vt_oh.columns = [f"vt={c}" for c in vt_cats]

    log_af   = np.log1p(df["af"].clip(lower=0)).astype("float32").rename("log_af")
    af_known = df["af_known"].astype("float32").rename("af_known")
    gene_pr  = df["gene"].map(gene_prior).fillna(global_mean).astype("float32").rename("gene_prior")

    X = pd.concat([cons_oh, vt_oh, log_af, af_known, gene_pr], axis=1)
    return X


def main(args):
    t0 = time.time()
    print(f"[clf] loading {args.data} ...")
    df = pd.read_parquet(args.data).reset_index(drop=True)
    n = len(df)
    y = df["label"].astype(int).values

    sp = split_indices(n, seed=args.seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"[clf] split (seed {args.seed}): train={len(tr):,} val={len(va):,} test={len(te):,} "
          f"| pathogenic rate (test)={y[te].mean():.3f}")

    # --- leakage-safe per-gene prior (TRAIN only) ---
    global_mean = float(y[tr].mean())
    g = df.iloc[tr].groupby("gene")["label"].agg(["mean", "count"])
    gene_prior = ((g["mean"] * g["count"] + GENE_PRIOR_ALPHA * global_mean)
                  / (g["count"] + GENE_PRIOR_ALPHA)).to_dict()

    # Fixed one-hot schema from the full set of observed categories (no label info).
    cons_cats = sorted(df["consequence"].astype(str).unique().tolist())
    vt_cats   = sorted(df["variant_type"].astype(str).unique().tolist())

    X = build_feature_frame(df, cons_cats, vt_cats, gene_prior, global_mean)
    print(f"[clf] {X.shape[1]} features; training HistGradientBoosting ...")

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=False, random_state=args.seed,
    )
    clf.fit(X.iloc[tr], y[tr])

    # --- tune threshold on VAL, evaluate on TEST ---
    p_val = clf.predict_proba(X.iloc[va])[:, 1]
    thr = best_threshold(y[va], p_val, objective="mcc").get("threshold", 0.5)
    p_te = clf.predict_proba(X.iloc[te])[:, 1]

    metrics = classification_metrics(y[te], p_te, threshold=thr)
    metrics["threshold"] = thr
    ci = (bootstrap_ci(y[te], p_te, kind="classification",
                       metrics=("auroc", "auprc", "mcc"), threshold=thr,
                       n_boot=args.bootstrap)
          if args.bootstrap else {})

    print(format_report(metrics, "pathogenicity (feature classifier)", ci))

    # --- persist model bundle (everything inference needs) ---
    bundle = {
        "model": clf, "cons_cats": cons_cats, "vt_cats": vt_cats,
        "gene_prior": gene_prior, "global_mean": global_mean,
        "feature_cols": list(X.columns), "threshold": thr,
        "trained_task": "pathogenicity", "split_seed": args.seed, "dataset_n": n,
    }
    joblib.dump(bundle, args.output)
    print(f"[clf] model saved -> {args.output}")

    payload = {"task": "pathogenicity", "weights": args.output, "data": args.data,
               "n_eval": int(len(te)), "metrics": metrics, "ci95": ci}
    with open(args.metrics_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[clf] metrics -> {args.metrics_out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the ClinVar pathogenicity feature classifier.")
    ap.add_argument("--data", default="CLEANED_DATA/clinvar_features.parquet")
    ap.add_argument("--output", default="holognn_pathogenicity_clf.joblib")
    ap.add_argument("--metrics-out", default="metrics_pathogenicity.json")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    main(args)
