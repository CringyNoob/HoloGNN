"""
HOLOGNN_APP/backend/app.py
==========================
FastAPI HTTP layer for the Holo-GNN web application.
Inference logic lives entirely in inference.py — this file is the HTTP layer only.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as _requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import get_engine
from db import (
    init_db,
    insert_prediction,
    list_predictions,
    get_prediction,
    delete_prediction,
    clear_history,
)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Holo-GNN API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction history is persisted in a local SQLite file (see db.py) so results
# survive page refreshes and server restarts.
init_db()

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class DdgRequest(BaseModel):
    wt_sequence: str
    mutation: str


class ScanRequest(BaseModel):
    sequence: str
    start: int = 1
    end: Optional[int] = None


class IdrRequest(BaseModel):
    sequence: str


class CompareRequest(BaseModel):
    uniprot_id: str


class ExportRequest(BaseModel):
    format: str          # "csv" | "json" | "pdb"
    filename: Optional[str] = None
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _error(status: int, detail: str) -> Response:
    """Return a JSON error response."""
    return Response(
        content=json.dumps({"detail": detail}),
        status_code=status,
        media_type="application/json",
    )


def _as_dict(model: BaseModel) -> Dict[str, Any]:
    """Pydantic v1/v2-safe dict() for persisting the request payload."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict:
    eng = get_engine()
    return eng.health()


@app.post("/api/ddg")
def ddg(req: DdgRequest):
    eng = get_engine()
    try:
        result = eng.predict_ddg(req.wt_sequence, req.mutation)
    except ValueError as exc:
        return _error(400, str(exc))
    summary = (f"ΔΔG {result['mutation']} = {result['ddg']:+.2f} kcal/mol "
               f"({result['verdict']})")
    insert_prediction("ddg", summary, _as_dict(req), result)
    return result


@app.post("/api/scan")
def scan(req: ScanRequest):
    eng = get_engine()
    try:
        result = eng.mutation_scan(req.sequence, req.start, req.end)
    except ValueError as exc:
        return _error(400, str(exc))
    pos = result["positions"]
    summary = (f"Scan {len(pos)} positions (residues {pos[0]}–{pos[-1]})"
               if pos else "Scan (empty window)")
    insert_prediction("scan", summary, _as_dict(req), result)
    return result


@app.post("/api/idr")
def idr(req: IdrRequest):
    eng = get_engine()
    try:
        result = eng.idr_ensemble(req.sequence)
    except ValueError as exc:
        return _error(400, str(exc))
    summary = (f"IDR ensemble · L={result['length']} · "
               f"Rg μ={result['mu']:.1f} σ={result['sigma']:.2f}")
    insert_prediction("idr", summary, _as_dict(req), result)
    return result


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Fetch AlphaFold2 structure + PAE, run Holo-GNN scan, return combined payload."""
    eng = get_engine()
    uid = req.uniprot_id.strip().upper()

    try:
        # 1. AlphaFold metadata
        af_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
        meta_resp = _requests.get(af_url, timeout=20)
        if meta_resp.status_code == 404:
            return _error(502, f"AlphaFold entry not found for {uid}")
        meta_resp.raise_for_status()
        meta = meta_resp.json()[0]

        pdb_url = meta.get("pdbUrl", "")
        pae_doc_url = meta.get("paeDocUrl", meta.get("paeImageUrl", ""))
        sequence = meta.get("uniprotSequence", "")

        if not pdb_url:
            return _error(502, f"AlphaFold entry for {uid} has no downloadable PDB URL.")

        # 2. Fetch PDB structure text
        pdb_resp = _requests.get(pdb_url, timeout=20)
        pdb_resp.raise_for_status()
        structure_pdb: str = pdb_resp.text

        # 3. Parse pLDDT from B-factor column of CA atoms
        plddt: List[float] = []
        for line in structure_pdb.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    plddt.append(float(line[60:66]))
                except (ValueError, IndexError):
                    pass

        # 4. Fetch PAE document
        pae_matrix: List[List[float]] = []
        pae_downsampled = False
        if pae_doc_url and not pae_doc_url.endswith(".png"):
            pae_resp = _requests.get(pae_doc_url, timeout=20)
            pae_resp.raise_for_status()
            pae_json = pae_resp.json()
            # Handle both flat {"predicted_aligned_error": ...}
            # and list format [{"predicted_aligned_error": ...}]
            if isinstance(pae_json, list):
                raw_pae = pae_json[0]["predicted_aligned_error"]
            else:
                raw_pae = pae_json["predicted_aligned_error"]

            n = len(raw_pae)
            if n > 200:
                stride = max(1, n // 200)
                raw_pae = [row[::stride] for row in raw_pae[::stride]]
                pae_downsampled = True
            pae_matrix = [[float(v) for v in row] for row in raw_pae]

        # 5. Holo-GNN mutation scan for per-position min ΔΔG
        holognn_min_ddg: List[float] = []
        if sequence:
            scan_end = min(len(sequence), 400)
            try:
                scan_result = eng.mutation_scan(sequence, 1, scan_end)
                matrix = scan_result["matrix"]  # 20 rows × N cols
                n_cols = len(scan_result["positions"])
                for col in range(n_cols):
                    col_vals = [matrix[row][col] for row in range(len(matrix))]
                    holognn_min_ddg.append(round(min(col_vals), 4))
            except ValueError:
                holognn_min_ddg = []

        result = {
            "uniprot_id": uid,
            "sequence": sequence,
            "structure_pdb": structure_pdb,
            "plddt": plddt,
            "pae": pae_matrix,
            "pae_downsampled": pae_downsampled,
            "holognn_min_ddg": holognn_min_ddg,
            "demo_mode": eng.demo_mode,
        }
        summary = f"AlphaFold compare · {uid} · {len(sequence)} residues"
        insert_prediction("compare", summary, _as_dict(req), result)
        return result

    except _requests.exceptions.Timeout:
        return _error(502, f"Request to AlphaFold timed out for {uid}")
    except _requests.exceptions.ConnectionError:
        return _error(502, f"Could not connect to AlphaFold API for {uid}")
    except (_requests.exceptions.HTTPError, KeyError, IndexError, ValueError) as exc:
        return _error(502, f"AlphaFold entry not found or parse error for {uid}: {exc}")


@app.post("/api/export")
def export(req: ExportRequest):
    """Return data as a downloadable file (CSV, JSON, or PDB)."""
    fmt = req.format.lower()
    data = req.data

    # ------------------------------------------------------------------ JSON
    if fmt == "json":
        fname = req.filename or "holognn_results.json"
        content = json.dumps(data, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    # ------------------------------------------------------------------ CSV
    elif fmt == "csv":
        fname = req.filename or "holognn_results.csv"
        buf = io.StringIO()

        is_scan = ("matrix" in data and "positions" in data and "aa_order" in data)
        if is_scan:
            positions: List[int] = data["positions"]
            wt_residues: List[str] = data.get("wt_residues", ["?"] * len(positions))
            aa_order: List[str] = data["aa_order"]
            matrix: List[List[float]] = data["matrix"]  # 20 × N

            # Header: position, wt, then one col per amino acid
            header = ["position", "wt"] + aa_order
            buf.write(",".join(header) + "\n")

            for col_idx, pos in enumerate(positions):
                wt = wt_residues[col_idx] if col_idx < len(wt_residues) else "?"
                row_vals = [str(matrix[row_idx][col_idx]) for row_idx in range(len(aa_order))]
                buf.write(",".join([str(pos), wt] + row_vals) + "\n")
        else:
            # Flatten top-level scalar keys into key/value CSV
            buf.write("key,value\n")
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    buf.write(f"{k},{v}\n")

        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    # ------------------------------------------------------------------ PDB
    elif fmt == "pdb":
        fname = req.filename or "holognn_annotated.pdb"

        # Determine sequence
        sequence: str = ""
        if "sequence" in data and isinstance(data["sequence"], str):
            sequence = data["sequence"].upper()
        elif "wt_residues" in data:
            sequence = "".join(data["wt_residues"])

        # Determine B-factor annotation
        if "bfactor" in data and isinstance(data["bfactor"], list):
            bfactors: List[float] = [float(v) for v in data["bfactor"]]
        elif "matrix" in data and "positions" in data:
            # per-position min ΔΔG from scan matrix
            matrix = data["matrix"]
            n_cols = len(data["positions"])
            bfactors = []
            for col in range(n_cols):
                col_vals = [matrix[row][col] for row in range(len(matrix))]
                bfactors.append(min(col_vals))
        else:
            bfactors = [0.0] * len(sequence)

        # Pad / trim bfactors to match sequence length
        n = len(sequence)
        bfactors = (bfactors + [0.0] * n)[:n]

        pdb_lines: List[str] = []
        ca_spacing = 3.8  # Å
        # Lay residues in a boustrophedon grid so the x-coordinate field (%8.3f,
        # max 8 chars) never overflows for long chains (e.g. AlphaFold inputs).
        per_row = 250                      # 250 * 3.8 = 950 Å < 9999.999

        for i, aa1 in enumerate(sequence):
            resname = _AA1_TO_3.get(aa1, "GLY")
            serial = (i + 1) % 100000       # ATOM serial field is 5 chars
            resseq = (i + 1) % 10000        # resSeq field is 4 chars
            bf = bfactors[i] if i < len(bfactors) else 0.0
            row, col = divmod(i, per_row)
            x = col * ca_spacing
            y = row * ca_spacing
            # Standard 80-column PDB ATOM record.
            line = (
                f"ATOM  {serial:>5}  CA  {resname} A{resseq:>4}    "
                f"{x:>8.3f}{y:>8.3f}{0.0:>8.3f}"
                f"{1.00:>6.2f}{bf:>6.2f}          "
                f" C  "
            )
            pdb_lines.append(line)

        pdb_lines.append("END")
        content = "\n".join(pdb_lines) + "\n"

        return Response(
            content=content,
            media_type="chemical/x-pdb",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    else:
        return _error(400, f"Unknown export format '{req.format}'. Use csv, json, or pdb.")


# ---------------------------------------------------------------------------
# Prediction history (SQLite-backed)
# ---------------------------------------------------------------------------

@app.get("/api/history")
def history(kind: Optional[str] = None, limit: int = 50):
    """Most-recent-first prediction history (metadata only). Optional ?kind=."""
    return {"items": list_predictions(kind, limit)}


@app.get("/api/history/{pred_id}")
def history_item(pred_id: int):
    """Full record (request + response payload) for one prediction."""
    item = get_prediction(pred_id)
    if item is None:
        return _error(404, f"No prediction with id {pred_id}.")
    return item


@app.delete("/api/history/{pred_id}")
def history_delete(pred_id: int):
    """Delete a single history entry."""
    if not delete_prediction(pred_id):
        return _error(404, f"No prediction with id {pred_id}.")
    return {"deleted": pred_id}


@app.delete("/api/history")
def history_clear(kind: Optional[str] = None):
    """Clear all history (or just one ?kind=)."""
    return {"cleared": clear_history(kind)}


# ---------------------------------------------------------------------------
# Frontend static files (mount LAST so /api routes are not shadowed)
# ---------------------------------------------------------------------------
_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "note": (
                "Frontend not built yet. "
                "Run `npm run build` inside frontend/ to generate the dist/ directory, "
                "then restart the server."
            )
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
