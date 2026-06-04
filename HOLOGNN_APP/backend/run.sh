#!/usr/bin/env bash
# run.sh — Start the Holo-GNN backend server (macOS / Linux)
#
# Creates an isolated virtual environment on first run (so it works on modern,
# PEP-668 "externally managed" Python installs), installs dependencies, then
# launches the server at http://127.0.0.1:8000.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[run.sh] Creating virtual environment in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[run.sh] Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

PORT="${PORT:-8000}"
echo "[run.sh] Starting Holo-GNN backend on http://127.0.0.1:${PORT}"
python -m uvicorn app:app --host 127.0.0.1 --port "${PORT}"
