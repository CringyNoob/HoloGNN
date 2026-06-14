"""
src/checkpoint.py
=================
Unified, power-cut-safe checkpoint I/O with provenance metadata.

Why this module exists
----------------------
Every HoloGNN trainer used to do a bare ``torch.save(model.state_dict())``.
Because :class:`~src.full_model.HoloGNN` always instantiates *all* task heads,
that single file contains every head's weights — including heads that were never
trained for the task at hand.  ``evaluate.py`` then loaded with ``strict=False``,
so an untrained head's **random** weights loaded silently (zero missing keys)
and produced noise-level metrics that looked legitimate.

``save_checkpoint`` records WHICH task / heads were actually trained (plus the
split seed used, so evaluation can reproduce the exact held-out set).
``load_checkpoint`` returns that metadata so callers can refuse to score an
untrained head.  Both remain backward compatible with the legacy bare
``state_dict`` format (older ``.pth`` files still load, flagged ``legacy=True``).

Saves are atomic (write ``*.tmp`` → ``os.replace``) so a power cut mid-write
can never corrupt the checkpoint — the same pattern used by ``pretrain_uniref.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

CHECKPOINT_FORMAT = 2   # bump when the payload schema changes

# Canonical task → required-head mapping.  A checkpoint is only trustworthy for a
# task if the corresponding head was actually trained.  ``pathogenicity`` lists
# both the dedicated classifier head (preferred) and the Siamese head (the
# legacy ``sigmoid(-ΔΔG)`` proxy) — evaluate.py decides which to require.
TASK_TO_HEADS: dict[str, list[str]] = {
    "stability":     ["stability_head"],
    "ddg":           ["siamese_head"],
    "idr":           ["siamese_head"],          # train_siamese's task name
    "pathogenicity": ["idr_head", "siamese_head"],
    "proteomics":    ["proteomics_head"],
}


def _atomic_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    try:  # best-effort durability before the atomic rename
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    except Exception:
        pass
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    trained_task: str,
    trained_heads: Sequence[str],
    config: Optional[dict] = None,
    split_seed: Optional[int] = None,
    dataset_n: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    """Atomically save a HoloGNN checkpoint with provenance metadata.

    Args:
        trained_task  : the task this checkpoint was trained for (see TASK_TO_HEADS).
        trained_heads : the head attribute names that received gradient updates.
        config        : model construction kwargs (so eval can rebuild faithfully).
        split_seed    : seed passed to ``src.splits`` so eval can reproduce the
                        exact held-out test partition.
        dataset_n     : len(full_dataset) at train time (eval warns on mismatch).
    """
    payload: dict[str, Any] = {
        "format":           CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "trained_task":     trained_task,
        "trained_heads":    list(trained_heads),
        "config":           config or {},
        "split_seed":       split_seed,
        "dataset_n":        dataset_n,
    }
    if extra:
        payload.update(extra)
    _atomic_save(payload, Path(path))


def load_checkpoint(path: str | Path, model: nn.Module, device, strict: bool = False):
    """Load weights into ``model``; return ``(load_result, meta)``.

    Handles both the metadata-wrapped format and a legacy bare ``state_dict``.
    ``meta`` always has ``trained_task`` / ``trained_heads`` keys (``None`` for
    legacy files) plus ``legacy: bool``.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        meta.setdefault("trained_task", None)
        meta.setdefault("trained_heads", None)
        meta["legacy"] = False
    else:
        state_dict = ckpt
        meta = {"legacy": True, "trained_task": None, "trained_heads": None,
                "split_seed": None, "dataset_n": None, "config": {}}
    load_result = model.load_state_dict(state_dict, strict=strict)
    return load_result, meta


__all__ = ["save_checkpoint", "load_checkpoint", "TASK_TO_HEADS", "CHECKPOINT_FORMAT"]
