from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
import uuid
from collections import deque

from app.solvers.particle import solve_particle_box
from app.solvers.entropy import von_neumann_entropy
from app.solvers.hydrogen import solve_hydrogen
from app.solvers.n2 import solve_n2

app = FastAPI(title="QEN")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

HISTORY: deque = deque(maxlen=100)


class RunRequest(BaseModel):
    system_type: str
    parameters: dict = {}


def dispatch(system_type, params):
    if system_type == "particle":
        L = params.get("L_nm", 2.0)
        if not (0.01 <= float(L) <= 1000.0):
            raise ValueError("L_nm must be between 0.01 and 1000 nm")
        n = int(params.get("n_states", 5))
        if not (1 <= n <= 32):
            raise ValueError("n_states must be between 1 and 32")
        return solve_particle_box(**params)
    if system_type == "entropy":
        rho = params.get("rho")
        if rho is None:
            raise ValueError("parameters.rho is required")
        return von_neumann_entropy(rho)
    if system_type == "hydrogen":
        Z = int(params.get("Z", 1))
        nmax = int(params.get("n_max", 4))
        if not (1 <= Z <= 20):
            raise ValueError("Z must be between 1 and 20")
        if not (1 <= nmax <= 20):
            raise ValueError("n_max must be between 1 and 20")
        return solve_hydrogen(**params)
    if system_type == "n2":
        n = int(params.get("n_points", 200))
        if not (10 <= n <= 5000):
            raise ValueError("n_points must be between 10 and 5000")
        return solve_n2(**params)
    raise ValueError(f"unknown system_type: {system_type}")


@app.get("/health")
def health():
    return {"status": "ok", "runs_in_memory": len(HISTORY)}


@app.post("/simulations/run")
def run_simulation(req: RunRequest):
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        result = dispatch(req.system_type, req.parameters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    dt_ms = (time.perf_counter() - t0) * 1000
    record = {
        "id": run_id,
        "system_type": req.system_type,
        "status": "completed",
        "parameters": req.parameters,
        "compute_time_ms": dt_ms,
        "result": result,
    }
    HISTORY.append(record)
    return record


@app.get("/simulations/history")
def history(limit: int = Query(20, ge=1, le=100)):
    items = list(HISTORY)[-limit:]
    items.reverse()
    return [
        {
            "id": r["id"],
            "system_type": r["system_type"],
            "status": r["status"],
            "parameters": r["parameters"],
            "compute_time_ms": r["compute_time_ms"],
        }
        for r in items
    ]
