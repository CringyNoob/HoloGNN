"""
src/device.py
=============
Single source of truth for device selection + a runtime "is this GPU usable?"
guard.  Every training / evaluation / inference script imports ``get_device`` and
``describe_device`` from here instead of repeating
``torch.device("cuda" if torch.cuda.is_available() else "cpu")``.

Why the guard matters
---------------------
The NVIDIA RTX 50-series (Blackwell, compute capability **sm_120**) is only
supported by **torch >= 2.7 built against CUDA 12.8 (cu128)**.  A plain
``pip install torch`` can pull a wheel compiled only up to sm_90, in which case
the 5070 Ti either raises a "no kernel image is available for execution" error
or silently runs on the CPU.  ``describe_device`` detects exactly this situation
and prints the one-line fix instead of failing deep inside a training run.
"""

from __future__ import annotations

import torch

# The pip command that installs a Blackwell-capable wheel.
CU128_INSTALL = "pip install torch --index-url https://download.pytorch.org/whl/cu128"


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device (CUDA if present, else CPU).

    A single helper so the whole codebase agrees on the device.  ``prefer_cuda``
    lets a caller force CPU (e.g. for a deterministic unit test).
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _arch_supports(capability: tuple[int, int]) -> bool:
    """True if the installed torch wheel was built for this compute capability.

    ``torch.cuda.get_arch_list()`` looks like ``['sm_70', 'sm_80', 'sm_90',
    'sm_120']``.  ``compute_120`` (PTX) also counts because it JITs forward.
    """
    major, minor = capability
    tag = f"sm_{major}{minor}"
    ptx = f"compute_{major}{minor}"
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:  # pragma: no cover - very old torch
        return True  # can't tell; assume fine
    if not arch_list:
        return True
    return tag in arch_list or ptx in arch_list


def describe_device(device: torch.device | None = None, *, verbose: bool = True) -> torch.device:
    """Print a human-readable summary of the active device and return it.

    On CUDA it reports the GPU name, total VRAM and compute capability, and warns
    loudly (with the cu128 fix) if the installed torch cannot target the card —
    the typical RTX 5070 Ti / Blackwell footgun.  On CPU it notes that training
    will be slow and points at the GPU install command.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        if verbose:
            print(f"[device] CUDA: {name}  (sm_{cap[0]}{cap[1]}, {total_gb:.1f} GB, "
                  f"torch {torch.__version__})")
        if not _arch_supports(cap):
            print("[device] " + "!" * 60)
            print(f"[device] WARNING: this torch build ({torch.__version__}) was NOT "
                  f"compiled for sm_{cap[0]}{cap[1]} ({name}).")
            print(f"[device]          arch list = {torch.cuda.get_arch_list()}")
            print("[device]          Kernels may fail or fall back to slow paths.")
            print(f"[device]          Fix (Blackwell / RTX 50-series): {CU128_INSTALL}")
            print("[device] " + "!" * 60)
    else:
        if verbose:
            cuda_built = torch.version.cuda is not None
            print(f"[device] CPU only (torch {torch.__version__}). Training will be slow.")
            if not torch.cuda.is_available() and cuda_built:
                print("[device] A CUDA build is installed but no GPU was detected "
                      "(driver / visibility issue?).")
            elif not cuda_built:
                print(f"[device] This is a CPU-only torch build. For an NVIDIA GPU: {CU128_INSTALL}")
    return device


__all__ = ["get_device", "describe_device", "CU128_INSTALL"]
