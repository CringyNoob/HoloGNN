"""
src/splits.py
=============
Deterministic, reproducible, leakage-free train / val / test splits shared by
**both** the training scripts and ``evaluate.py``.

The problem this fixes
----------------------
``train_siamese.py`` used ``random_split(full_dataset, 90/10)`` with no fixed
seed, and ``evaluate.py`` scored the *tail* of the same full dataset — so the
"held-out" rows had almost certainly been trained on (inflated metrics).  And
the MegaScale head/tail convention produced a biased, non-random test slice.

The fix: one deterministic split keyed only on ``(n, seed, fractions)`` (or on
stable per-row keys).  Training takes the ``train`` partition, evaluation takes
the ``test`` partition — disjoint *by construction* and reproducible across runs
and machines.

Two strategies
--------------
* :func:`split_indices` — seeded permutation of ``range(n)``.  Simple; requires
  the same ``n`` on both sides (guaranteed when both build the dataset from the
  same parquet, since the dataset ``dropna`` is deterministic).
* :func:`assign_by_key` — hashes a stable per-row key (e.g. the sequence) into
  ``[0, 1)`` and buckets it.  Independent of row count / ordering, so it stays
  correct even if a few rows are added or filtered later.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Sequence

import numpy as np

DEFAULT_SEED = 42
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)   # train / val / test
_SPLIT_NAMES = ("train", "val", "test")


def _check_fractions(fractions: Sequence[float]) -> None:
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test).")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions} (sum={sum(fractions)}).")


def split_indices(
    n: int,
    seed: int = DEFAULT_SEED,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> Dict[str, np.ndarray]:
    """Return disjoint ``{"train","val","test"}`` index arrays for ``range(n)``.

    A fixed-seed permutation, so two callers with the same ``(n, seed,
    fractions)`` get identical, non-overlapping partitions.
    """
    _check_fractions(fractions)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return {
        "train": perm[:n_train],
        "val":   perm[n_train:n_train + n_val],
        "test":  perm[n_train + n_val:],
    }


def assign_by_key(
    keys: Sequence,
    seed: int = DEFAULT_SEED,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> np.ndarray:
    """Assign each row to ``"train"/"val"/"test"`` by hashing its stable key.

    Independent of ``n`` and ordering: the same key always lands in the same
    partition for a given ``seed``.
    """
    _check_fractions(fractions)
    cum = np.cumsum(fractions)
    out = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        digest = hashlib.sha1(f"{seed}:{k}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)   # → [0, 1)
        if u < cum[0]:
            out[i] = "train"
        elif u < cum[1]:
            out[i] = "val"
        else:
            out[i] = "test"
    return out


def indices_for(
    n: int,
    which: str,
    seed: int = DEFAULT_SEED,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    max_samples: int | None = None,
) -> np.ndarray:
    """Convenience: the index array for one partition, optionally capped.

    ``which`` ∈ {"train","val","test"}.  ``max_samples`` truncates *after*
    splitting (deterministic, since the partition order is seed-fixed).
    """
    if which not in _SPLIT_NAMES:
        raise ValueError(f"which must be one of {_SPLIT_NAMES}, got {which!r}.")
    idx = split_indices(n, seed=seed, fractions=fractions)[which]
    if max_samples is not None and max_samples < len(idx):
        idx = idx[:max_samples]
    return idx


def subset(dataset, which: str, *, seed: int = DEFAULT_SEED,
           fractions: Sequence[float] = DEFAULT_FRACTIONS,
           max_samples: int | None = None):
    """Return a ``torch.utils.data.Subset`` for one partition of ``dataset``."""
    from torch.utils.data import Subset
    idx = indices_for(len(dataset), which, seed=seed, fractions=fractions,
                      max_samples=max_samples)
    return Subset(dataset, idx.tolist())


__all__ = [
    "DEFAULT_SEED", "DEFAULT_FRACTIONS",
    "split_indices", "assign_by_key", "indices_for", "subset",
]
