@echo off
REM run.bat - Start the Holo-GNN backend server (Windows)
REM
REM Creates an isolated virtual environment on first run, installs dependencies,
REM then launches the server at http://127.0.0.1:8000.

cd /d "%~dp0"

if not exist ".venv" (
    echo [run.bat] Creating virtual environment in .venv ...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo [run.bat] Installing dependencies ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if "%PORT%"=="" set PORT=8000
echo [run.bat] Starting Holo-GNN backend on http://127.0.0.1:%PORT%
python -m uvicorn app:app --host 127.0.0.1 --port %PORT%
