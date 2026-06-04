# Holo-GNN App

A local, browser-based dashboard for **Holo-GNN** — predict protein stability changes (ΔΔG),
explore mutational stability landscapes, visualise IDR ensembles, and compare against AlphaFold2,
all without writing code. This is the reference implementation of the paper's **Future Work §8.1**
interactive UI.

> **Run it on your own PC.** Download the repository, start the backend, and open the page in your
> browser. Nothing is sent to a server you don't control. (Public deployment is planned as future
> work — for now it's a download-and-run-locally app.)

<p align="center"><em>React + Vite frontend · FastAPI backend · works fully offline in demo mode</em></p>

---

## Features

| Tab | What it does | Maps to paper §8.1 bullet |
|-----|--------------|---------------------------|
| **ΔΔG Predictor** | Enter a wild-type sequence + a point mutation (e.g. `L8P`) → predicted ΔΔG with a **confidence interval** | "ΔΔG predictions with confidence intervals in real time" |
| **Stability Landscape** | Interactive **heatmap** of every substitution × position; colour = predicted thermodynamic effect | "visualise predicted stability landscapes as interactive heatmaps" |
| **IDR Ensemble** | Radius-of-gyration distribution (μ/σ) + per-residue compaction/expansion profile | "explore IDR ensemble distributions … compaction/expansion annotations" |
| **AlphaFold Compare** | Fetches an AlphaFold2 model by UniProt ID and shows **pLDDT-coloured 3-D structure + PAE** beside Holo-GNN's prediction | "compare Holo-GNN predictions against AlphaFold2 pLDDT and PAE side-by-side" |
| **Export** | Download results as **CSV / JSON / PDB** (B-factor-annotated); re-hydrates from history on load so results survive a page refresh | "export results in standardised formats" |
| **History** | Browse, view, re-export, and delete every past prediction stored in the local SQLite database | persistent local run log |

---

## Demo mode vs. full mode

The public repository **ships no trained weights** (`*.pth` is git-ignored). So out of the box the
app runs in **demo mode**: every prediction uses a deterministic biophysical heuristic
(hydropathy + side-chain volume + charge + helix-breaker terms — see `../HoloGNN/src/heuristics.py`).
This makes the whole UI explorable with zero setup, and an amber banner makes the mode obvious.

To enable **full Holo-GNN inference**:

1. Place the trained weights at `../HoloGNN/holognn_stability_final.pth`
   (or set the `HOLOGNN_WEIGHTS` environment variable to their path).
2. Install the heavy stack in the backend venv: `pip install -r ../HoloGNN/requirements.txt`
   (torch, transformers, torch-geometric).
3. Restart the backend. `/api/health` will report `model_loaded: true` and the banner disappears.

The app locates the model package as a **sibling folder** (`../HoloGNN`). Override with
`HOLOGNN_MODEL_DIR=/path/to/HoloGNN` if your layout differs.

---

## Prerequisites

Check you have these before starting:

| Tool | Version | Check |
|------|---------|-------|
| **Python** | 3.9+ (3.10–3.12 recommended) | `python3 --version` |
| **Node.js + npm** | Node 18+ | `node --version` && `npm --version` |
| **Git** | any | `git --version` |

Internet is needed only for: the **AlphaFold Compare** tab (queries the EBI AlphaFold DB), and — in
full mode — the one-time ESM-2 weights download. Everything else works offline.

---

## One-command launcher (recommended)

From the **repository root** a single script handles everything — venv creation, dep install, frontend build, server start, and browser open:

```bash
python runapp.py
```

The app will be available at **http://127.0.0.1:8000**.

What `runapp.py` does automatically:
1. Creates `HOLOGNN_APP/backend/.venv` and installs the lightweight backend deps.
2. Builds the frontend (`npm run build`) if `dist/` is missing — skipped gracefully if Node/npm is absent.
3. Starts uvicorn and health-checks `/api/health`.
4. Opens the browser.

Useful flags:

| Flag | Effect |
|------|--------|
| `--full` | Also installs the model stack (torch, transformers, etc.) for real inference |
| `--weights PATH` | Use trained weights at PATH; sets `HOLOGNN_WEIGHTS` |
| `--port N` | Listen on port N (default 8000) |
| `--no-browser` | Skip the automatic browser open |
| `--rebuild-frontend` | Force a fresh `npm run build` even if `dist/` already exists |
| `--dev` | Run the Vite hot-reload dev server alongside the backend |

Without `--full` and without weights the app starts in **demo mode** (deterministic biophysical heuristic) — no heavy dependencies required.

---

## Full setup & run

The app lives next to the model folder (`HoloGNN/`) in the same repository — download the whole repo so
both are present.

```bash
# 1) Get the repository
git clone <your-repo-url>
cd <repo>/HOLOGNN_APP

# 2) Build the frontend (one time, ~1-2 min)
cd frontend
npm install            # installs React, Plotly, 3Dmol, etc.
npm run build          # produces frontend/dist/  (the static site)

# 3) Start the backend - it also serves the built frontend
cd ../backend
./run.sh               # macOS / Linux
#   run.bat            # Windows
```

`run.sh` / `run.bat` automatically:
1. create an isolated virtual environment in `backend/.venv` (works on PEP-668 "externally managed" Python),
2. install the lightweight backend deps (`requirements.txt`), and
3. launch the server.

When you see `Uvicorn running on http://127.0.0.1:8000`, open **http://127.0.0.1:8000** in your browser.
You'll land on the dashboard with an amber **demo-mode** banner (expected — no trained weights shipped).

> **Custom port:** `PORT=8080 ./run.sh`.
> **Manual start (instead of run.sh):**
> ```bash
> cd backend && python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> python -m uvicorn app:app --host 127.0.0.1 --port 8000
> ```

### Enabling full Holo-GNN inference (optional)
By default the app is in **demo mode**. To run the real model:
```bash
# from the backend venv:
pip install -r ../../HoloGNN/requirements.txt        # torch, transformers, torch-geometric
# place trained weights where the engine looks (or set HOLOGNN_WEIGHTS):
#   ../HoloGNN/holognn_stability_final.pth
```
Restart the backend — `/api/health` will report `model_loaded: true` and the banner disappears.

### Developing the frontend with hot-reload
```bash
cd HOLOGNN_APP/frontend
npm run dev            # Vite dev server on http://localhost:5173, proxies /api -> :8000
```
(Keep the backend running on port 8000 in another terminal. After UI changes, `npm run build` again so the
backend-served bundle is refreshed.)

---

## Project layout

```
HOLOGNN_APP/
├── backend/
│   ├── app.py            # FastAPI HTTP layer (serves the API + built frontend)
│   ├── inference.py      # model wrapper + deterministic demo fallback
│   ├── requirements.txt  # lightweight deps (fastapi, uvicorn, requests, numpy, pydantic)
│   ├── run.sh / run.bat  # venv bootstrap + launch
│   ├── holognn_history.db  # SQLite prediction history (created on first run, git-ignored)
│   └── .venv/            # created on first run (git-ignored)
└── frontend/
    ├── src/              # React + TypeScript SPA (6 tabs, including History)
    ├── package.json
    ├── vite.config.ts
    └── dist/             # built static site (created by `npm run build`)
```

---

## API reference

All endpoints are under `/api`. Sign convention: **ΔΔG > 0 = stabilizing, ΔΔG < 0 = destabilizing**.

| Method & path | Body | Returns |
|---------------|------|---------|
| `GET  /api/health` | — | `{version, model_loaded, demo_mode, weights_path, load_note}` |
| `POST /api/ddg` | `{wt_sequence, mutation}` | `{mutation, position, ddg, ci_low, ci_high, stabilizing, verdict, demo_mode}` |
| `POST /api/scan` | `{sequence, start, end}` | `{positions, wt_residues, aa_order, matrix (20×N), demo_mode}` |
| `POST /api/idr` | `{sequence}` | `{length, mu, sigma, per_residue, demo_mode}` |
| `POST /api/compare` | `{uniprot_id}` | `{sequence, structure_pdb, plddt, pae, holognn_min_ddg, demo_mode}` |
| `POST /api/export` | `{format, filename?, data}` | downloadable CSV / JSON / PDB file |

---

## Prediction history (SQLite)

Every prediction — ΔΔG, stability scan, IDR ensemble, and AlphaFold compare — is automatically persisted to a local **SQLite** database at:

```
HOLOGNN_APP/backend/holognn_history.db
```

The file is created on first run and is gitignored. It uses the Python standard-library `sqlite3` module with zero external dependencies — the right fit for a single-user local app.

### History tab (6th tab)

The **History** tab lets you:
- Browse all past predictions, filterable by kind (ddg / scan / idr / compare).
- View the full saved result for any past run.
- Re-export any past result as CSV / JSON / PDB.
- Delete individual records or clear the entire history.

The **Export** tab also re-hydrates from history on load, so results are available even after a page refresh or backend restart.

### History API endpoints

| Method & path | Description |
|---------------|-------------|
| `GET  /api/history?kind=&limit=` | List past predictions (optional filter by kind, default limit 100) |
| `GET  /api/history/{id}` | Fetch full saved result for a single record |
| `DELETE /api/history/{id}` | Delete a single record |
| `DELETE /api/history` | Clear all history |

---

## Troubleshooting

- **"Frontend not built yet" JSON at `/`** — run `npm run build` in `frontend/`, then restart the backend.
- **AlphaFold Compare returns an error** — that tab needs internet, and the UniProt ID must have an
  AlphaFold DB entry (try `P69905`, human haemoglobin α).
- **Predictions look heuristic / banner won't go away** — you're in demo mode; add the weights file as
  described above for full Holo-GNN inference.

---

See the model documentation and citation in [`../README.md`](../README.md).
