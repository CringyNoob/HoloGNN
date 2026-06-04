"""
src/metrics.py
==============
Shared evaluation metrics so every training / eval script reports performance the
same way.  Two task families:

* **Regression** (ΔΔG / absolute ΔG / expression) → Pearson r, Spearman ρ, RMSE,
  MAE, R².
* **Binary classification** (ClinVar pathogenic-vs-benign, disease classifier)
  → AUROC, AUPRC (average precision), F1, precision, recall, accuracy, MCC.

All functions accept plain lists / numpy arrays / detached tensors and return a
``dict[str, float]`` (NaN for any metric that is undefined for the given batch —
e.g. AUROC when only one class is present — rather than raising).
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Sequence

import numpy as np


def _as_1d(x) -> np.ndarray:
    """Coerce list / numpy / torch tensor to a 1-D float numpy array."""
    if hasattr(x, "detach"):          # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Pearson r, Spearman ρ, RMSE, MAE, R² for a regression task."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]

    out: Dict[str, float] = {
        "n": float(n),
        "pearson": float("nan"),
        "spearman": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "r2": float("nan"),
    }
    if n == 0:
        return out

    err = yp - yt
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))

    # Correlations need variance in both vectors.
    if n >= 2 and np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
        try:
            from scipy.stats import pearsonr, spearmanr
            out["pearson"] = float(pearsonr(yt, yp)[0])
            out["spearman"] = float(spearmanr(yt, yp)[0])
        except Exception:  # pragma: no cover - scipy missing / degenerate
            out["pearson"] = float(np.corrcoef(yt, yp)[0, 1])
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """AUROC, AUPRC, F1, precision, recall, accuracy, MCC for a binary task.

    ``y_prob`` is the positive-class probability (or score).  Threshold-based
    metrics use ``threshold``.  Ranking metrics (AUROC / AUPRC) are NaN when only
    one class is present in ``y_true`` (they are undefined there).
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_prob)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]

    out: Dict[str, float] = {
        "n": float(n),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "accuracy": float("nan"),
        "mcc": float("nan"),
        "pos_rate": float("nan"),
    }
    if n == 0:
        return out

    yt_bin = (yt >= 0.5).astype(int)            # tolerate float labels
    out["pos_rate"] = float(yt_bin.mean())
    both_classes = 0 < yt_bin.sum() < n

    try:
        from sklearn.metrics import (
            average_precision_score,
            accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        y_hat = (yp >= threshold).astype(int)
        out["accuracy"] = float(accuracy_score(yt_bin, y_hat))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["f1"] = float(f1_score(yt_bin, y_hat, zero_division=0))
            out["precision"] = float(precision_score(yt_bin, y_hat, zero_division=0))
            out["recall"] = float(recall_score(yt_bin, y_hat, zero_division=0))
            try:
                out["mcc"] = float(matthews_corrcoef(yt_bin, y_hat))
            except Exception:
                out["mcc"] = float("nan")
        if both_classes:
            out["auroc"] = float(roc_auc_score(yt_bin, yp))
            out["auprc"] = float(average_precision_score(yt_bin, yp))
        else:
            warnings.warn(
                "Only one class present in y_true; AUROC/AUPRC are undefined (NaN).",
                RuntimeWarning,
            )
    except ImportError:  # pragma: no cover - sklearn missing
        # Minimal numpy fallback for the threshold metrics.
        y_hat = (yp >= threshold).astype(int)
        tp = int(((y_hat == 1) & (yt_bin == 1)).sum())
        fp = int(((y_hat == 1) & (yt_bin == 0)).sum())
        fn = int(((y_hat == 0) & (yt_bin == 1)).sum())
        tn = int(((y_hat == 0) & (yt_bin == 0)).sum())
        out["accuracy"] = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out["precision"], out["recall"] = prec, rec
        out["f1"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
_LABELS = {
    "pearson": "Pearson r", "spearman": "Spearman rho", "rmse": "RMSE", "mae": "MAE",
    "r2": "R^2", "auroc": "AUROC", "auprc": "AUPRC", "f1": "F1",
    "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy",
    "mcc": "MCC", "pos_rate": "Positive rate", "n": "N",
}


def format_report(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Render a metrics dict as an aligned multi-line block for the console."""
    lines = [f"=== {title} ==="]
    for key, val in metrics.items():
        label = _LABELS.get(key, key)
        if key == "n":
            lines.append(f"  {label:<14}: {int(val)}")
        elif isinstance(val, float) and math.isnan(val):
            lines.append(f"  {label:<14}: n/a")
        else:
            lines.append(f"  {label:<14}: {val:.4f}")
    return "\n".join(lines)


__all__ = ["regression_metrics", "classification_metrics", "format_report"]
