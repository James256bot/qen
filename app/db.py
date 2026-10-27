import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.environ.get("QEN_DB", "qen.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    system_type TEXT NOT NULL,
    status TEXT NOT NULL,
    parameters TEXT NOT NULL,
    result TEXT NOT NULL,
    compute_time_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_type ON runs (system_type);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(SCHEMA)


def insert_run(run_id, system_type, status, parameters, result, compute_time_ms):
    with get_conn() as c:
        c.execute(
            "INSERT INTO runs (id, system_type, status, parameters, result, compute_time_ms) VALUES (?,?,?,?,?,?)",
            (run_id, system_type, status, json.dumps(parameters), json.dumps(result), compute_time_ms),
        )


def get_run(run_id):
    with get_conn() as c:
        row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "system_type": row["system_type"],
        "status": row["status"],
        "parameters": json.loads(row["parameters"]),
        "result": json.loads(row["result"]),
        "compute_time_ms": row["compute_time_ms"],
        "created_at": row["created_at"],
    }


def list_runs(limit=20, system_type=None):
    with get_conn() as c:
        if system_type:
            rows = c.execute(
                "SELECT id, system_type, status, parameters, compute_time_ms, created_at FROM runs WHERE system_type = ? ORDER BY created_at DESC LIMIT ?",
                (system_type, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, system_type, status, parameters, compute_time_ms, created_at FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [
        {
            "id": r["id"],
            "system_type": r["system_type"],
            "status": r["status"],
            "parameters": json.loads(r["parameters"]),
            "compute_time_ms": r["compute_time_ms"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def count_runs():
    with get_conn() as c:
        return c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
