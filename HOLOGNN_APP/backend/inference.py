"""
HOLOGNN_APP/backend/inference.py
================================
Inference engine that wraps the sibling Holo-GNN model package for the web UI.

Design goals
------------
1. **Single shared code path.**  Mechanistic features and the biophysical demo
   heuristic are imported from the model package (``src.dataset`` /
   ``src.heuristics``) so the app and ``predict.py`` never diverge.
2. **Always usable offline (demo mode).**  The public repository ships no trained
   weights and the heavy stack (torch / transformers / torch_geometric) may be
   absent.  When either is missing the engine transparently falls back to the
   deterministic biophysical heuristic and reports ``demo_mode = True`` so the UI
   can show a banner.  Every endpoint returns sensible, reproducible numbers.
3. **Lazy & cached.**  ESM-2 / weights are loaded once on first use.

Sign convention (ΔG_mut − ΔG_wt):  ΔΔG > 0 → stabilising;  ΔΔG < 0 → destabilising.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Locate the sibling model package (…/HoloGNN) and import the shared helpers.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "HoloGNN"
MODEL_DIR = Path(os.environ.get("HOLOGNN_MODEL_DIR", _DEFAULT_MODEL_DIR)).resolve()
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# The heuristic module is dependency-light (only stdlib) — always importable.
from src.heuristics import (  # noqa: E402
    AA_ORDER,
    apply_mutation,
    heuristic_confidence,
    heuristic_ddg,
    heuristic_radius_of_gyration,
)

WEIGHTS_PATH = Path(
    os.environ.get("HOLOGNN_WEIGHTS", MODEL_DIR / "holognn_stability_final.pth")
)
ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
MAX_LENGTH = int(os.environ.get("HOLOGNN_MAX_LENGTH", "512"))
VERSION = "5.0"

# Hard caps to keep a local laptop responsive.
MAX_SCAN_POSITIONS = 400
MAX_SEQ_LENGTH = 2000


def _clean_sequence(seq: str) -> str:
    """Uppercase, strip whitespace/FASTA headers, validate amino-acid alphabet."""
    if not seq:
        raise ValueError("Empty sequence.")
    lines = [ln for ln in seq.splitlines() if not ln.startswith(">")]
    seq = "".join(lines) if lines else seq
    seq = "".join(seq.split()).upper()
    if not seq:
        raise ValueError("Sequence contains no residues.")
    bad = set(seq) - set(AA_ORDER)
    if bad:
        raise ValueError(f"Invalid amino-acid symbol(s): {''.join(sorted(bad))}")
    if len(seq) > MAX_SEQ_LENGTH:
        raise ValueError(f"Sequence too long ({len(seq)} > {MAX_SEQ_LENGTH}).")
    return seq


class HoloGNNEngine:
    """Lazy, cached wrapper around the Holo-GNN model with a heuristic fallback."""

    def __init__(self) -> None:
        self._loaded = False        # True once a real model is in memory
        self._tried = False         # True once a load attempt has been made
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ state
    def _ensure_loaded(self) -> None:
        """Attempt to load the real model exactly once; stay in demo mode on failure."""
        if self._tried:
            return
        self._tried = True

        if not WEIGHTS_PATH.exists():
            self._load_error = f"weights not found at {WEIGHTS_PATH}"
            return
        try:
            import torch
            from transformers import EsmTokenizer
            from src.full_model import HoloGNN

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HoloGNN()
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            model.to(device).eval()

            self._torch = torch
            self._model = model
            self._tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL)
            self._device = device
            self._loaded = True
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{exc.__class__.__name__}: {exc}"
            self._loaded = False

    @property
    def demo_mode(self) -> bool:
        self._ensure_loaded()
        return not self._loaded

    def health(self) -> Dict:
        self._ensure_loaded()
        return {
            "version": VERSION,
            "model_loaded": self._loaded,
            "demo_mode": not self._loaded,
            "weights_path": str(WEIGHTS_PATH),
            "load_note": None if self._loaded else self._load_error,
        }

    # --------------------------------------------------------------- full mode
    def _make_batch(self, seq: str):
        from src.dataset import mechanistic_features_for_protein

        torch = self._torch
        enc = self._tokenizer(
            seq, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        )
        input_ids = enc["input_ids"].to(self._device)
        mask = enc["attention_mask"].to(self._device)
        L = input_ids.size(1)
        mech = mechanistic_features_for_protein(seq, L).unsqueeze(0).to(self._device)

        class _DataBatch:
            pass

        data = _DataBatch()
        data.input_ids = input_ids
        data.mask = mask
        data.mechanistic_features = mech
        data.edge_index = None
        return data

    def _ddg_full(self, wt_seq: str, mt_seq: str) -> float:
        torch = self._torch
        data_wt = self._make_batch(wt_seq)
        data_mt = self._make_batch(mt_seq)
        with torch.no_grad():
            dG_wt_to_mt, _ = self._model((data_wt, data_mt), task="idr")
        return float(dG_wt_to_mt.item())

    # ----------------------------------------------------------- public methods
    def predict_ddg(self, wt_sequence: str, mutation: str) -> Dict:
        """ΔΔG for a single ``<WT><pos><MUT>`` mutation, with a confidence band."""
        wt = _clean_sequence(wt_sequence)
        mt, idx, wt_aa, mut_aa = apply_mutation(wt, mutation)
        self._ensure_loaded()

        if self._loaded:
            ddg = self._ddg_full(wt, mt)
        else:
            ddg = heuristic_ddg(wt_aa, mut_aa)
        lo, hi = heuristic_confidence(ddg)

        return {
            "mutation": f"{wt_aa}{idx + 1}{mut_aa}",
            "position": idx + 1,
            "wt_residue": wt_aa,
            "mut_residue": mut_aa,
            "ddg": round(ddg, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "stabilizing": ddg > 0,
            "verdict": "stabilizing" if ddg > 0 else "destabilizing",
            "demo_mode": not self._loaded,
        }

    def mutation_scan(
        self, sequence: str, start: int = 1, end: Optional[int] = None
    ) -> Dict:
        """
        Full deep-mutational-scan: ΔΔG of every substitution over a window.

        Returns a 20×N matrix (rows = ``AA_ORDER``, cols = positions).  The
        wild-type cell at each position is reported as 0.0.
        """
        seq = _clean_sequence(sequence)
        start = max(1, int(start))
        # Note: an explicit end=0 is treated as "to the end" (0 is not a valid
        # 1-based position); any other value is honoured.
        end = int(end) if (end is not None and int(end) != 0) else len(seq)
        end = min(end, len(seq))
        if end < start:
            raise ValueError("end must be ≥ start.")
        if end - start + 1 > MAX_SCAN_POSITIONS:
            raise ValueError(
                f"Scan window too wide ({end - start + 1} > {MAX_SCAN_POSITIONS} positions)."
            )

        self._ensure_loaded()
        positions = list(range(start, end + 1))
        wt_residues = [seq[p - 1] for p in positions]

        matrix: List[List[float]] = []
        for aa in AA_ORDER:
            row: List[float] = []
            for p in positions:
                wt_aa = seq[p - 1]
                if aa == wt_aa:
                    row.append(0.0)
                elif self._loaded:
                    mt = seq[: p - 1] + aa + seq[p:]
                    row.append(round(self._ddg_full(seq, mt), 4))
                else:
                    row.append(round(heuristic_ddg(wt_aa, aa), 4))
            matrix.append(row)

        return {
            "positions": positions,
            "wt_residues": wt_residues,
            "aa_order": AA_ORDER,
            "matrix": matrix,
            "demo_mode": not self._loaded,
        }

    def idr_ensemble(self, sequence: str) -> Dict:
        """
        Radius-of-gyration ensemble (μ, σ) plus a per-residue compaction score.

        Full mode uses ``EnsembleIDRHead``; demo mode uses the Flory-scaling
        heuristic.  ``per_residue`` ∈ [0, 1]: higher = more expanded/disordered.
        """
        seq = _clean_sequence(sequence)
        self._ensure_loaded()

        if self._loaded:
            torch = self._torch
            data = self._make_batch(seq)
            with torch.no_grad():
                z = self._model._encode(data)            # (1, 320)
                mu_t, sigma_t = self._model.idr_head(z)  # (1, 1), (1, 1)
            mu, sigma = float(mu_t.item()), float(sigma_t.item())
        else:
            mu, sigma = heuristic_radius_of_gyration(seq)

        # Per-residue disorder propensity (deterministic, charge/flexibility based).
        disorder_promoting = set("PEKSQGRDHNT")
        window = 7
        half = window // 2
        per_residue: List[float] = []
        for i in range(len(seq)):
            lo = max(0, i - half)
            hi = min(len(seq), i + half + 1)
            frac = sum(1 for a in seq[lo:hi] if a in disorder_promoting) / (hi - lo)
            per_residue.append(round(frac, 4))

        return {
            "length": len(seq),
            "mu": round(mu, 4),
            # Clamp σ away from 0 so the frontend Gaussian (1/σ…) never divides by zero.
            "sigma": round(max(sigma, 1e-3), 4),
            "per_residue": per_residue,
            "demo_mode": not self._loaded,
        }


@lru_cache(maxsize=1)
def get_engine() -> HoloGNNEngine:
    """Process-wide singleton engine."""
    return HoloGNNEngine()


if __name__ == "__main__":  # tiny smoke test
    eng = get_engine()
    print("health:", eng.health())
    print("ddg:", eng.predict_ddg(
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "L8P",
    ))
    scan = eng.mutation_scan("MQIFVKTLTG", 1, 5)
    print("scan cols:", scan["positions"], "rows:", len(scan["matrix"]))
    print("idr:", eng.idr_ensemble("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ"))
