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
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    system_type TEXT NOT NULL,
    parameters TEXT NOT NULL,
    status TEXT NOT NULL,
    progress_current INTEGER DEFAULT 0,
    progress_total INTEGER DEFAULT 0,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
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


# ============================================================
# Job queue helpers
# ============================================================

def create_job(job_id, system_type, parameters):
    with get_conn() as c:
        c.execute(
            "INSERT INTO jobs (id, system_type, parameters, status) VALUES (?,?,?,?)",
            (job_id, system_type, json.dumps(parameters), "queued"),
        )


def get_job(job_id):
    with get_conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "system_type": row["system_type"],
        "parameters": json.loads(row["parameters"]),
        "status": row["status"],
        "progress_current": row["progress_current"],
        "progress_total": row["progress_total"],
        "result": json.loads(row["result"]) if row["result"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
    }


def update_job_status(job_id, status, progress_current=None, progress_total=None):
    with get_conn() as c:
        if status == "running" and progress_current is None:
            c.execute("UPDATE jobs SET status = ?, started_at = datetime('now') WHERE id = ?",
                      (status, job_id))
        elif progress_current is not None:
            c.execute("UPDATE jobs SET status = ?, progress_current = ?, progress_total = ? WHERE id = ?",
                      (status, progress_current, progress_total, job_id))
        else:
            c.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))


def complete_job(job_id, result):
    with get_conn() as c:
        c.execute(
            "UPDATE jobs SET status = 'completed', result = ?, completed_at = datetime('now') WHERE id = ?",
            (json.dumps(result), job_id),
        )


def fail_job(job_id, error):
    with get_conn() as c:
        c.execute(
            "UPDATE jobs SET status = 'failed', error = ?, completed_at = datetime('now') WHERE id = ?",
            (error, job_id),
        )


def list_jobs(limit=20):
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, system_type, status, progress_current, progress_total, created_at FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
