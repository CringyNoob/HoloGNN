"""
src/heuristics.py
=================
Deterministic biophysical fallback for ΔΔG estimation.

Holo-GNN ships **no trained weights** in the public repository (``*.pth`` is
``.gitignore``d), and the heavy inference stack (torch / transformers /
torch_geometric) may not be installed on a reviewer's laptop.  To keep both
``predict.py`` and the HOLOGNN_APP backend fully usable offline, this module
provides a small, **deterministic** (no randomness) biophysical estimator of the
single-point-mutation stability change ΔΔG.

It is a physically-motivated heuristic — *not* the trained model — and is only
used when ``demo_mode`` is active.  It captures the dominant first-order effects
that govern point-mutation stability:

    • Hydropathy change   (Kyte–Doolittle)         — burying/exposing hydrophobics
    • Side-chain volume change                     — steric packing strain
    • Net charge change   (Henderson–Hasselbalch)  — buried-charge penalty
    • Helix-breaker proline / flexible glycine      — backbone conformational cost

Sign convention (MegaScale / ΔG_mut − ΔG_wt):
    ΔΔG  >  0  →  stabilising
    ΔΔG  <  0  →  destabilising
This matches the original predict.py phrasing ("diff < 0 → DESTABILIZING").
"""

from __future__ import annotations

import math
from typing import Tuple

# The 20 canonical amino acids, in a fixed order used everywhere (scan rows etc.)
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
_AA_SET = set(AA_ORDER)

# --- Kyte–Doolittle hydropathy index (positive = hydrophobic) ---------------
KD_HYDROPATHY = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# --- Side-chain van der Waals volume (Å³) -----------------------------------
RESIDUE_VOLUME = {
    "A":  88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G":  60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S":  89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# --- Approximate residue net charge at pH 7.4 -------------------------------
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}


def _charge(aa: str) -> float:
    return _CHARGE.get(aa.upper(), 0.0)


# ---------------------------------------------------------------------------
# Mutation parsing
# ---------------------------------------------------------------------------
def apply_mutation(wt_seq: str, mutation: str) -> Tuple[str, int, str, str]:
    """
    Apply a point mutation written as ``<WT><1-based-pos><MUT>`` (e.g. ``"M1A"``).

    Returns ``(mutant_sequence, zero_based_index, wt_aa, mut_aa)``.
    Raises ``ValueError`` on malformed input or a WT-residue mismatch.
    """
    mutation = mutation.strip().upper()
    if len(mutation) < 3:
        raise ValueError(f"Malformed mutation code: {mutation!r}")

    wt_aa, mut_aa = mutation[0], mutation[-1]
    pos_str = mutation[1:-1]
    if not pos_str.isdigit():
        raise ValueError(f"Could not parse position from {mutation!r}")
    pos = int(pos_str)  # 1-based

    if wt_aa not in _AA_SET or mut_aa not in _AA_SET:
        raise ValueError(f"Non-standard amino acid in {mutation!r}")
    if pos < 1 or pos > len(wt_seq):
        raise ValueError(f"Position {pos} out of range for length {len(wt_seq)}")

    idx = pos - 1
    if wt_seq[idx].upper() != wt_aa:
        raise ValueError(
            f"WT residue mismatch: sequence has {wt_seq[idx]!r} at position {pos}, "
            f"mutation says {wt_aa!r}"
        )

    mut_seq = wt_seq[:idx] + mut_aa + wt_seq[idx + 1:]
    return mut_seq, idx, wt_aa, mut_aa


# ---------------------------------------------------------------------------
# Core ΔΔG heuristic
# ---------------------------------------------------------------------------
def heuristic_ddg(wt_aa: str, mut_aa: str, *, buried: float = 0.6) -> float:
    """
    Deterministic ΔΔG estimate (kcal/mol) for a single substitution.

    ``buried`` ∈ [0, 1] is an assumed burial fraction (0.6 ≈ average residue);
    hydrophobic/charge penalties scale with it.  Returns 0.0 for a no-op
    substitution (wt == mut).
    """
    wt_aa, mut_aa = wt_aa.upper(), mut_aa.upper()
    if wt_aa == mut_aa or wt_aa not in _AA_SET or mut_aa not in _AA_SET:
        return 0.0

    # 1) Hydropathy: losing hydrophobicity in the core is destabilising.
    d_hydro = KD_HYDROPATHY[mut_aa] - KD_HYDROPATHY[wt_aa]
    ddg_h = 0.18 * d_hydro * buried

    # 2) Volume: any large packing change strains the core (always penalised).
    d_vol = abs(RESIDUE_VOLUME[mut_aa] - RESIDUE_VOLUME[wt_aa])
    ddg_v = -0.012 * d_vol * buried

    # 3) Charge: introducing/!flipping a buried charge is destabilising.
    d_charge = abs(_charge(mut_aa) - _charge(wt_aa))
    ddg_c = -0.6 * d_charge * buried

    # 4) Backbone conformation: proline = helix breaker; glycine = flexible hinge.
    ddg_bb = 0.0
    if mut_aa == "P" and wt_aa != "P":
        ddg_bb -= 2.3
    elif wt_aa == "P" and mut_aa != "P":
        ddg_bb += 0.6
    if mut_aa == "G" and wt_aa != "G":
        ddg_bb -= 1.0
    elif wt_aa == "G" and mut_aa != "G":
        ddg_bb += 0.4

    ddg = ddg_h + ddg_v + ddg_c + ddg_bb
    # Clamp to a physically sensible band.
    return max(-7.0, min(4.0, ddg))


def heuristic_confidence(ddg: float) -> Tuple[float, float]:
    """A simple symmetric uncertainty band that widens with |ΔΔG|."""
    half = 0.4 + 0.12 * abs(ddg)
    return ddg - half, ddg + half


# ---------------------------------------------------------------------------
# IDR / radius-of-gyration heuristic  (demo for the EnsembleIDRHead)
# ---------------------------------------------------------------------------
def heuristic_radius_of_gyration(seq: str) -> Tuple[float, float]:
    """
    Deterministic (μ, σ) for the radius of gyration (Å) of a chain.

    Uses Flory scaling Rg ≈ R0 · N^ν, with the apparent exponent ν interpolated
    between a compact globule (0.33) and an expanded coil (0.60) by the chain's
    mean disorder propensity (fraction of polar/charged/Gly/Pro residues).  σ
    grows with disorder (expanded ensembles are more heterogeneous).
    """
    n = max(len(seq), 1)
    disorder_promoting = set("PEKSQGRDHNT")
    frac = sum(1 for a in seq.upper() if a in disorder_promoting) / n
    nu = 0.33 + 0.27 * frac                 # 0.33 (globule) … 0.60 (coil)
    r0 = 2.0
    mu = r0 * (n ** nu)
    sigma = mu * (0.06 + 0.18 * frac)
    return mu, sigma
