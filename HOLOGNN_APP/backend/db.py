"""
HOLOGNN_APP/backend/db.py
=========================
Tiny SQLite persistence layer for prediction history.

Why SQLite (and only stdlib ``sqlite3``)?
-----------------------------------------
This app is a **local, single-user, download-and-run** tool.  A file-backed
SQLite database is exactly the right fit: zero configuration, no server, no extra
dependency, and it survives page refreshes / server restarts — fixing the old
behaviour where results lived only in the browser's in-memory store and vanished
on reload.  Each prediction (ΔΔG, scan, IDR, AlphaFold compare) is stored as one
row; the frontend History tab reads it back, can reload a result, and re-export.

One small table keeps it simple; the full request/response payloads are stored as
JSON text so any result can be reconstructed verbatim.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# DB file lives next to this module (gitignored via *.db).  Overridable for tests.
DB_PATH = Path(os.environ.get("HOLOGNN_DB", Path(__file__).resolve().parent / "holognn_history.db"))

# Allowed prediction kinds (also used to validate the ?kind= filter).
KINDS = ("ddg", "scan", "idr", "compare")


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection (one per request → thread-safe under uvicorn)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the predictions table if it does not exist (called on startup)."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                kind          TEXT    NOT NULL,
                created_at    REAL    NOT NULL,
                summary       TEXT    NOT NULL,
                request_json  TEXT    NOT NULL,
                response_json TEXT    NOT NULL,
                demo_mode     INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_kind_time "
                     "ON predictions (kind, created_at DESC)")


def insert_prediction(kind: str, summary: str, request: Dict[str, Any],
                      response: Dict[str, Any]) -> int:
    """Persist one prediction and return its new row id.

    Never raises into the request path — persistence is best-effort; a failure to
    log must not break the actual prediction response.
    """
    try:
        demo = 1 if response.get("demo_mode") else 0
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO predictions "
                "(kind, created_at, summary, request_json, response_json, demo_mode) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (kind, time.time(), summary,
                 json.dumps(request), json.dumps(response), demo),
            )
            return int(cur.lastrowid)
    except Exception:  # pragma: no cover - logging must not break predictions
        return -1


def _row_summary(row: sqlite3.Row) -> Dict[str, Any]:
    """List-view shape: metadata only, no heavy payloads."""
    return {
        "id": row["id"],
        "kind": row["kind"],
        "created_at": row["created_at"],
        "summary": row["summary"],
        "demo_mode": bool(row["demo_mode"]),
    }


def list_predictions(kind: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Most-recent-first history (metadata only). Optional ``kind`` filter."""
    limit = max(1, min(int(limit), 500))
    with _connect() as conn:
        if kind in KINDS:
            rows = conn.execute(
                "SELECT id, kind, created_at, summary, demo_mode FROM predictions "
                "WHERE kind = ? ORDER BY created_at DESC LIMIT ?", (kind, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, kind, created_at, summary, demo_mode FROM predictions "
                "ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_summary(r) for r in rows]


def get_prediction(pred_id: int) -> Optional[Dict[str, Any]]:
    """Full record (incl. request/response payloads) for one id, or None."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    if row is None:
        return None
    item = _row_summary(row)
    item["request"] = json.loads(row["request_json"])
    item["response"] = json.loads(row["response_json"])
    return item


def delete_prediction(pred_id: int) -> bool:
    """Delete one row; True if a row was removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        return cur.rowcount > 0


def clear_history(kind: Optional[str] = None) -> int:
    """Delete all history (or all of one ``kind``); return rows removed."""
    with _connect() as conn:
        if kind in KINDS:
            cur = conn.execute("DELETE FROM predictions WHERE kind = ?", (kind,))
        else:
            cur = conn.execute("DELETE FROM predictions")
        return cur.rowcount


__all__ = [
    "DB_PATH", "KINDS", "init_db", "insert_prediction",
    "list_predictions", "get_prediction", "delete_prediction", "clear_history",
]
