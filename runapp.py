#!/usr/bin/env python3
"""
runapp.py — single cross-platform launcher for the Holo-GNN web app
===================================================================
One command to run the whole app (FastAPI backend + built React frontend) on
Windows / macOS / Linux:

    python runapp.py                 # demo mode (no heavy ML deps); opens browser
    python runapp.py --full          # also install the model stack (torch, …) for
                                     #   real inference (needs trained weights)
    python runapp.py --weights path/to/holognn_stability_final.pth
    python runapp.py --port 8080 --no-browser
    python runapp.py --rebuild-frontend
    python runapp.py --dev           # backend + Vite dev server (hot reload)

What it does
------------
1. Creates an isolated virtualenv in ``HOLOGNN_APP/backend/.venv`` (PEP-668 safe)
   and installs the lightweight backend requirements into it.
2. Builds the frontend (``npm install`` + ``npm run build``) if ``dist/`` is
   missing — skipped gracefully if Node/npm is not installed (the backend then
   serves a "build the frontend" message).
3. Launches uvicorn via the venv's Python, waits for ``/api/health``, and opens
   the browser.

GPU note (RTX 50-series / Blackwell): ``--full`` installs ``torch>=2.7`` from the
default index, which may NOT include sm_120 kernels.  For an NVIDIA 50-series GPU,
install the CUDA 12.8 wheel FIRST, then run with ``--full``:
    pip install torch --index-url https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "HOLOGNN_APP"
BACKEND = APP_DIR / "backend"
FRONTEND = APP_DIR / "frontend"
MODEL_DIR = ROOT / "HoloGNN"
VENV_DIR = BACKEND / ".venv"

IS_WINDOWS = os.name == "nt"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def _run(cmd, cwd=None, env=None, check=True):
    print(f"[runapp] $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def _ensure_venv() -> Path:
    py = _venv_python()
    if not py.exists():
        print(f"[runapp] creating virtualenv → {VENV_DIR}")
        _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    return py


def _pip_install(py: Path, *args: str):
    _run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=False)
    _run([str(py), "-m", "pip", "install", "--quiet", *args])


def _npm() -> str | None:
    """Return the npm executable path (npm.cmd on Windows) or None if absent."""
    return shutil.which("npm")


def _build_frontend(force: bool) -> None:
    dist = FRONTEND / "dist"
    if dist.exists() and not force:
        print(f"[runapp] frontend already built ({dist}); use --rebuild-frontend to rebuild.")
        return
    npm = _npm()
    if not npm:
        print("[runapp] WARNING: npm not found — skipping frontend build. The backend "
              "will serve a placeholder page. Install Node.js to build the UI, or run "
              "with --dev once npm is available.")
        return
    if not (FRONTEND / "node_modules").exists():
        _run([npm, "install"], cwd=FRONTEND)
    _run([npm, "run", "build"], cwd=FRONTEND)


def _wait_for_health(port: int, timeout: float = 90.0) -> bool:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Launch the Holo-GNN web app.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    ap.add_argument("--rebuild-frontend", action="store_true", help="Force npm run build.")
    ap.add_argument("--full", action="store_true",
                    help="Also install the model stack (torch, transformers, …) for real inference.")
    ap.add_argument("--weights", default="", help="Path to trained weights (sets HOLOGNN_WEIGHTS).")
    ap.add_argument("--dev", action="store_true",
                    help="Run the Vite dev server (hot reload) alongside the backend.")
    args = ap.parse_args()

    if not BACKEND.exists():
        print(f"[runapp] ERROR: backend not found at {BACKEND}", file=sys.stderr)
        return 1

    # 1. backend venv + deps
    py = _ensure_venv()
    print("[runapp] installing backend dependencies …")
    _pip_install(py, "-r", str(BACKEND / "requirements.txt"))
    if args.full:
        print("[runapp] --full: installing the model stack (this can take a while) …")
        print("[runapp] NOTE: for an NVIDIA RTX 50-series GPU, install the cu128 torch wheel "
              "first:\n          pip install torch --index-url https://download.pytorch.org/whl/cu128")
        _pip_install(py, "-r", str(MODEL_DIR / "requirements.txt"))

    # 2. frontend build (skipped in --dev; the dev server serves it live)
    if not args.dev:
        _build_frontend(args.rebuild_frontend)

    # 3. environment for the backend process
    env = os.environ.copy()
    env["PORT"] = str(args.port)
    if args.weights:
        env["HOLOGNN_WEIGHTS"] = str(Path(args.weights).resolve())
        print(f"[runapp] HOLOGNN_WEIGHTS = {env['HOLOGNN_WEIGHTS']}")

    # 4. launch uvicorn (via the venv python) + optional Vite dev server
    uvicorn_cmd = [str(py), "-m", "uvicorn", "app:app",
                   "--host", "127.0.0.1", "--port", str(args.port)]
    print(f"[runapp] starting backend on http://127.0.0.1:{args.port} …")
    backend_proc = subprocess.Popen(uvicorn_cmd, cwd=str(BACKEND), env=env)

    dev_proc = None
    open_url = f"http://127.0.0.1:{args.port}"
    if args.dev:
        npm = _npm()
        if npm:
            if not (FRONTEND / "node_modules").exists():
                _run([npm, "install"], cwd=FRONTEND)
            print("[runapp] starting Vite dev server on http://localhost:5173 …")
            dev_proc = subprocess.Popen([npm, "run", "dev"], cwd=str(FRONTEND))
            open_url = "http://localhost:5173"
        else:
            print("[runapp] WARNING: --dev requested but npm not found; serving the built UI instead.")

    try:
        if _wait_for_health(args.port):
            print(f"[runapp] backend healthy ✓  →  {open_url}")
            if not args.no_browser:
                webbrowser.open(open_url)
        else:
            print("[runapp] WARNING: backend did not report healthy in time; check the logs above.")
        # Block on the backend process until Ctrl-C / exit.
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\n[runapp] shutting down …")
    finally:
        for proc in (dev_proc, backend_proc):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
