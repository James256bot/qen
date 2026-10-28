from fastapi import FastAPI, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import time
import uuid
from collections import deque
from app.db import init_db, insert_run, get_run, list_runs, count_runs

from app.lessons import list_lessons, get_lesson
from app.i18n import get_strings
from app.solvers.particle import solve_particle_box
from app.solvers.entropy import von_neumann_entropy
from app.solvers.hydrogen import solve_hydrogen
from app.solvers.n2 import solve_n2
from app.solvers.oscillator import solve_oscillator
from app.solvers.finite_well import solve_finite_well
from app.solvers.box2d import solve_box2d
from app.solvers.box3d import solve_box3d
from app.solvers.evolve import solve_time_evolution
from app.solvers.qsim import run_circuit, CANONICAL_CIRCUITS
from app.solvers.noisy_circuit import run_circuit_noisy
from app.solvers.chemistry import run_vqe_h2, h2_exact_ground_energy
from app.solvers.qpe import run_qpe, phase_gate
from app.solvers.battery import vqe_molecule, exact_molecule_energy, estimate_voltage, MOLECULES
from app.solvers.lattice2d import solve_lattice2d
from app.solvers.vmc import vmc_hubbard, exact_hubbard_ground_energy
from app.solvers.dft import solve_dft_1d
from app.solvers.shor import shor
from app.solvers.electrolyte import screen_electrolytes, compute_homo_lumo
from app.solvers.drugs import compute_binding, screen_ligand, screen_library, LIGANDS, RECEPTORS
from app.solvers.carbon import screen_mofs, compute_adsorption, MOFS
from app.solvers.portfolio import optimize_portfolio, list_stocks, STOCKS
from app.solvers.qml import classify, list_datasets
from app.intents import parse_intent
from app.scenarios import list_scenarios, get_scenario
from app.jobs import spawn_job
from app.db import create_job, get_job, list_jobs

app = FastAPI(title="QEN")


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            stype = msg.get("system_type", "")
            params = msg.get("parameters", {})
            t0 = time.perf_counter()
            try:
                result = dispatch(stype, params)
                dt_ms = (time.perf_counter() - t0) * 1000
                await ws.send_json({
                    "status": "completed",
                    "system_type": stype,
                    "parameters": params,
                    "compute_time_ms": dt_ms,
                    "result": result,
                })
            except ValueError as e:
                await ws.send_json({"status": "error", "detail": str(e)})
            except Exception as e:
                await ws.send_json({"status": "error", "detail": f"{type(e).__name__}: {e}"})
    except WebSocketDisconnect:
        return


@app.get("/service-worker.js")
def service_worker():
    # Development: return 404 so browsers unregister any cached SW.
    raise HTTPException(status_code=404, detail="service worker disabled")


@app.get("/manifest.json")
def manifest():
    return FileResponse("app/static/manifest.json", media_type="application/manifest+json")


@app.get("/scenarios")
def scenarios_index():
    return list_scenarios()


@app.get("/scenarios/{scenario_id}")
def scenario_detail(scenario_id: str):
    sc = get_scenario(scenario_id)
    if sc is None:
        raise HTTPException(status_code=404, detail=f"scenario {scenario_id} not found")
    return {"id": scenario_id, **sc}


@app.post("/scenarios/{scenario_id}/run")
def scenario_run(scenario_id: str):
    """Execute all steps in the scenario and return the sequence."""
    sc = get_scenario(scenario_id)
    if sc is None:
        raise HTTPException(status_code=404, detail=f"scenario {scenario_id} not found")

    outputs = []
    t_total = 0.0
    for i, step in enumerate(sc["steps"]):
        if step["kind"] == "note":
            outputs.append({
                "step": i,
                "kind": "note",
                "text": step["text"],
            })
        elif step["kind"] == "solve":
            t0 = time.perf_counter()
            try:
                result = dispatch(step["solver"], step["parameters"])
                dt_ms = (time.perf_counter() - t0) * 1000
                t_total += dt_ms
                outputs.append({
                    "step": i,
                    "kind": "solve",
                    "solver": step["solver"],
                    "parameters": step["parameters"],
                    "compute_time_ms": dt_ms,
                    "result": result,
                })
            except Exception as e:
                outputs.append({
                    "step": i,
                    "kind": "error",
                    "detail": f"{type(e).__name__}: {e}",
                })

    return {
        "scenario_id": scenario_id,
        "title": sc["title"],
        "description": sc["description"],
        "total_compute_ms": t_total,
        "steps": outputs,
    }


@app.get("/circuits")
def circuits_index():
    return [{"id": k, "n_qubits": v["n_qubits"], "description": v["description"], "n_gates": len(v["gates"])} for k, v in CANONICAL_CIRCUITS.items()]


@app.get("/circuits/{name}")
def circuit_detail(name: str):
    if name not in CANONICAL_CIRCUITS:
        raise HTTPException(status_code=404, detail=f"circuit {name} not found")
    return CANONICAL_CIRCUITS[name]


@app.get("/screen/all")
def screen_all():
    """Run all virtual screens and return a unified dashboard.

    Runs drug screening, battery screening, carbon screening, and
    QML classification in parallel where possible.
    """
    import time
    from app.solvers.drugs import screen_library
    from app.solvers.electrolyte import screen_electrolytes
    from app.solvers.carbon import screen_mofs
    from app.solvers.qml import classify as qml_classify
    from app.solvers.portfolio import optimize_portfolio

    t_total = time.perf_counter()
    results = {}

    # Drug screening against COX-2
    try:
        t0 = time.perf_counter()
        drug = screen_library(receptor_id="COX-2")
        results["drug"] = {
            "status": "completed",
            "time_ms": (time.perf_counter() - t0) * 1000,
            "result": drug,
        }
    except Exception as e:
        results["drug"] = {"status": "error", "detail": str(e)}

    # Battery screening
    try:
        t0 = time.perf_counter()
        battery = screen_electrolytes()
        results["battery"] = {
            "status": "completed",
            "time_ms": (time.perf_counter() - t0) * 1000,
            "result": battery,
        }
    except Exception as e:
        results["battery"] = {"status": "error", "detail": str(e)}

    # Carbon screening
    try:
        t0 = time.perf_counter()
        carbon = screen_mofs()
        results["carbon"] = {
            "status": "completed",
            "time_ms": (time.perf_counter() - t0) * 1000,
            "result": carbon,
        }
    except Exception as e:
        results["carbon"] = {"status": "error", "detail": str(e)}

    # QML classification
    try:
        t0 = time.perf_counter()
        qml = qml_classify(dataset="iris", n_qubits=4, test_frac=0.3)
        results["qml"] = {
            "status": "completed",
            "time_ms": (time.perf_counter() - t0) * 1000,
            "result": qml,
        }
    except Exception as e:
        results["qml"] = {"status": "error", "detail": str(e)}

    # Portfolio optimization
    try:
        t0 = time.perf_counter()
        portfolio = optimize_portfolio(
            tickers=["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "GS"],
            risk_tolerance=0.5, budget=2, p=2,
        )
        results["portfolio"] = {
            "status": "completed",
            "time_ms": (time.perf_counter() - t0) * 1000,
            "result": portfolio,
        }
    except Exception as e:
        results["portfolio"] = {"status": "error", "detail": str(e)}

    total_ms = (time.perf_counter() - t_total) * 1000

    # Get the H2 correction status
    h2_corr = None
    try:
        from app.solvers.drugs import get_h2_correlation_correction
        h2_corr = get_h2_correlation_correction()
    except Exception:
        pass

    return {
        "total_compute_ms": total_ms,
        "h2_correction": h2_corr,
        "screens": results,
    }


@app.get("/screen")
def screen_page():
    from fastapi.responses import FileResponse
    resp = FileResponse("app/static/screen.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/lessons")
def lessons_index():
    return list_lessons()


@app.get("/lessons/{lesson_id}")
def lesson_detail(lesson_id: str):
    l = get_lesson(lesson_id)
    if l is None:
        raise HTTPException(status_code=404, detail=f"lesson {lesson_id} not found")
    return l


@app.get("/i18n/{lang}")
def i18n_strings(lang: str):
    return get_strings(lang)


@app.get("/")
def landing():
    """Landing page -- curated scenarios, ask box, links."""
    from fastapi.responses import FileResponse
    resp = FileResponse("app/static/landing.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# ============================================================
# Background jobs
# ============================================================

@app.post("/jobs")
def submit_job(payload: dict):
    """Submit a long-running simulation as a background job.

    Request: {"system_type": "vmc", "parameters": {...}}
    Response: {"job_id": "...", "status": "queued"}
    """
    import uuid
    system_type = payload.get("system_type", "").strip()
    parameters = payload.get("parameters", {})
    if not system_type:
        raise HTTPException(status_code=400, detail="system_type is required")

    job_id = str(uuid.uuid4())
    create_job(job_id, system_type, parameters)

    # Spawn the background thread
    try:
        spawn_job(job_id, system_type, parameters, dispatch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to spawn job: {e}")

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs")
def jobs_index(limit: int = Query(20, ge=1, le=100)):
    """List recent jobs."""
    return list_jobs(limit=limit)


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Get job status, progress, and result."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@app.get("/studio")
def studio():
    """Full technical UI -- all solvers, all parameters."""
    from fastapi.responses import FileResponse
    resp = FileResponse("app/static/index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


init_db()
HISTORY: deque = deque(maxlen=100)  # kept for backward compat; DB is authoritative


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
    if system_type == "oscillator":
        w = float(params.get("omega_eV", 1.0))
        if not (1e-6 <= w <= 100.0):
            raise ValueError("omega_eV must be between 1e-6 and 100")
        n = int(params.get("n_max", 5))
        if not (1 <= n <= 30):
            raise ValueError("n_max must be between 1 and 30")
        return solve_oscillator(**params)

    if system_type == "finite_well":
        V0 = float(params.get("V0_eV", 10.0))
        if not (0.01 <= V0 <= 1000.0):
            raise ValueError("V0_eV must be between 0.01 and 1000")
        return solve_finite_well(**params)

    if system_type == "box2d":
        return solve_box2d(**params)

    if system_type == "box3d":
        return solve_box3d(**params)

    if system_type == "evolve":
        return solve_time_evolution(**params)

    if system_type == "circuit":
        n = int(params.get("n_qubits", 2))
        if not (1 <= n <= 20):
            raise ValueError("n_qubits must be between 1 and 20")
        gates = params.get("gates", [])
        if not isinstance(gates, list):
            raise ValueError("gates must be a list")
        shots = int(params.get("shots", 0))
        if shots < 0 or shots > 100000:
            raise ValueError("shots must be between 0 and 100000")
        ent = params.get("entanglement_cut", None)
        bloch = params.get("bloch_qubits", None)
        noise = params.get("noise", None)
        if noise is None:
            return run_circuit(n, gates, shots=shots, entanglement_cut=ent, bloch_qubits=bloch)
        return run_circuit_noisy(n, gates, shots=shots, entanglement_cut=ent,
                                 bloch_qubits=bloch, noise=noise)

    if system_type == "vqe_h2":
        R = float(params.get("R_angstrom", 0.735))
        if not (0.3 <= R <= 5.0):
            raise ValueError("R_angstrom must be between 0.3 and 5.0")
        n_layers = int(params.get("n_layers", 2))
        if not (1 <= n_layers <= 4):
            raise ValueError("n_layers must be 1..4")
        max_iter = int(params.get("max_iter", 400))
        result = run_vqe_h2(R_angstrom=R, n_layers=n_layers, max_iter=max_iter)
        result["exact_energy_hartree"] = h2_exact_ground_energy(R)
        result["error_kcal_per_mol"] = (result["energy_hartree"] - result["exact_energy_hartree"]) * 627.509
        return result

    if system_type == "qpe":
        phi = float(params.get("phi", 0.25))
        if not (0.0 <= phi < 1.0):
            raise ValueError("phi must be in [0, 1)")
        n = int(params.get("n_ancilla", 5))
        if not (2 <= n <= 12):
            raise ValueError("n_ancilla must be between 2 and 12")
        shots = int(params.get("shots", 0))
        U = phase_gate(phi)
        return run_qpe(U, n_ancilla=n, shots=shots)

    if system_type == "battery":
        molecule = params.get("molecule")
        if molecule is not None:
            if molecule not in MOLECULES:
                raise ValueError(f"unknown molecule {molecule}. valid: {list(MOLECULES.keys())}")
            r = vqe_molecule(molecule, n_layers=int(params.get("n_layers", 2)), max_iter=int(params.get("max_iter", 400)))
            r["exact_energy_hartree"] = exact_molecule_energy(molecule)
            r["error_kcal_per_mol"] = (r["energy_hartree"] - r["exact_energy_hartree"]) * 627.509
            return r
        cathode = params.get("cathode", "LiF")
        anode = params.get("anode", "LiH")
        return estimate_voltage(cathode, anode, n_electrons=int(params.get("n_electrons", 1)))

    if system_type == "lattice2d":
        Nx = int(params.get("Nx", 40))
        Ny = int(params.get("Ny", 40))
        if Nx * Ny > 10000:
            raise ValueError("Nx * Ny must be <= 10000")
        return solve_lattice2d(**params)

    if system_type == "vmc":
        Nx = int(params.get("Nx", 6))
        Ny = int(params.get("Ny", 1))
        N = Nx * Ny
        if N > 40:
            raise ValueError("Nx * Ny must be <= 40 for VMC demo")
        t = float(params.get("t", 1.0))
        fill = float(params.get("fill", 0.5))
        N_elec = max(2, int(round(fill * N)))
        N_up = N_elec // 2
        N_dn = N_elec - N_up
        n_samples = int(params.get("n_samples", 4000))
        do_sweep = bool(int(params.get("sweep", 1)))

        if do_sweep:
            U_values = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0]
            sweep_samples = min(n_samples, 500)
            seeds_per_point = 2
            sweep_data = []
            for U in U_values:
                energies = []
                gs = []
                stderrs = []
                for sd in range(seeds_per_point):
                    r = vmc_hubbard(Nx=Nx, Ny=Ny, N_up=N_up, N_dn=N_dn, U=U, t=t,
                                    n_samples=sweep_samples, max_iter=15, seed=42 + sd)
                    energies.append(r["energy_per_site"])
                    gs.append(r["optimal_g"])
                    stderrs.append(r["energy_stderr"])
                import statistics
                sweep_data.append({
                    "U": U,
                    "energy_per_site": statistics.mean(energies),
                    "optimal_g": statistics.mean(gs),
                    "energy_stderr": statistics.mean(stderrs),
                    "n_seeds": seeds_per_point,
                })
            return {
                "algorithm": "VMC_Hubbard",
                "Nx": Nx, "Ny": Ny, "N": N,
                "N_up": N_up, "N_dn": N_dn,
                "t": t,
                "fill": fill,
                "n_samples": n_samples,
                "sweep": sweep_data,
            }

        # single-point mode
        U = float(params.get("U", 4.0))
        r = vmc_hubbard(Nx=Nx, Ny=Ny, N_up=N_up, N_dn=N_dn, U=U, t=t,
                        n_samples=n_samples, max_iter=int(params.get("max_iter", 30)))
        if Nx * Ny <= 4 and N_up + N_dn <= 2:
            try:
                r["exact_energy"] = exact_hubbard_ground_energy(Nx=Nx, Ny=Ny, N_up=N_up, N_dn=N_dn, U=U, t=t)
            except Exception:
                pass
        return r

    if system_type == "dft":
        N = int(params.get("N", 150))
        L = float(params.get("L", 15.0))
        n_electrons = int(params.get("n_electrons", 2))
        potential = params.get("potential", "harmonic")
        xc_alpha = float(params.get("xc_alpha", 0.5))
        xc_beta = float(params.get("xc_beta", 0.05))
        return solve_dft_1d(N=N, L=L, n_electrons=n_electrons,
                            potential=potential, xc_alpha=xc_alpha, xc_beta=xc_beta)

    if system_type == "shor":
        N = int(params.get("N", 15))
        if not (2 <= N <= 1000):
            raise ValueError("N must be between 2 and 1000")
        return _add_animation_hint("shor", shor(N))

    if system_type == "electrolyte":
        if "candidates" in params:
            return screen_electrolytes(candidates=params["candidates"])
        if "name" in params:
            return compute_homo_lumo(params["name"])
        return screen_electrolytes()

    if system_type == "drug":
        if "ligand" in params and "receptor" in params:
            return compute_binding(params["ligand"], params["receptor"])
        if "ligand" in params:
            return screen_ligand(params["ligand"], receptors=params.get("receptors"))
        if "receptor" in params:
            return screen_library(ligands=params.get("ligands"), receptor_id=params["receptor"])
        # default: screen library against COX-2
        return _add_animation_hint("drug", screen_library(receptor_id="COX-2"))

    if system_type == "ligands":
        return [{"id": k, "name": v["name"], "class": v["class"], "formula": v["formula"]} for k, v in LIGANDS.items()]

    if system_type == "receptors":
        return [{"id": k, "name": v["name"], "description": v["description"]} for k, v in RECEPTORS.items()]

    if system_type == "carbon":
        if "mof" in params:
            return compute_adsorption(params["mof"])
        return _add_animation_hint("carbon", screen_mofs(candidates=params.get("candidates")))

    if system_type == "portfolio":
        tickers = params.get("tickers")
        risk = float(params.get("risk_tolerance", 0.5))
        budget = int(params.get("budget", 2))
        p_layers = int(params.get("p", 2))
        return _add_animation_hint("portfolio", optimize_portfolio(tickers=tickers, risk_tolerance=risk, budget=budget, p=p_layers))

    if system_type == "stocks":
        return list_stocks()

    if system_type == "qml":
        dataset = params.get("dataset", "iris")
        n_qubits = int(params.get("n_qubits", 4))
        test_frac = float(params.get("test_frac", 0.3))
        return _add_animation_hint("qml", classify(dataset=dataset, n_qubits=n_qubits, test_frac=test_frac))

    if system_type == "datasets":
        return list_datasets()

    raise ValueError(f"unknown system_type: {system_type}")


def _add_animation_hint(system_type, result):
    """Attach an animation hint to the result so the UI knows what to render."""
    hints = {
        "shor": {"type": "qpe_period", "duration_ms": 3000},
        "electrolyte": {"type": "energy_levels", "duration_ms": 2000},
        "drug": {"type": "docking", "duration_ms": 3000},
        "carbon": {"type": "adsorption", "duration_ms": 2000},
        "portfolio": {"type": "portfolio_evolution", "duration_ms": 3000},
        "qml": {"type": "decision_boundary", "duration_ms": 2000},
    }
    if system_type in hints:
        result["animation"] = hints[system_type]
    return result


@app.post("/ask")
def ask(payload: dict):
    """Natural-language prompt → solver → result + narration.

    Request: {"prompt": "show me quantum tunneling"}
    Response: {intent, narration, result}
    """
    prompt = payload.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    intent = parse_intent(prompt)
    t0 = time.perf_counter()
    try:
        result = dispatch(intent["system_type"], intent["parameters"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    dt_ms = (time.perf_counter() - t0) * 1000
    return {
        "prompt": prompt,
        "matched": intent["matched"],
        "score": intent["score"],
        "system_type": intent["system_type"],
        "parameters": intent["parameters"],
        "narration": intent["narration"],
        "compute_time_ms": dt_ms,
        "result": result,
    }


@app.get("/health")
def health():
    return {"status": "ok", "runs_in_memory": len(HISTORY), "runs_in_db": count_runs()}


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
    insert_run(run_id, req.system_type, "completed", req.parameters, result, dt_ms)
    return record


@app.get("/simulations/history")
def history(limit: int = Query(20, ge=1, le=100), system: str = Query(None)):
    return list_runs(limit=limit, system_type=system)


@app.get("/runs/{run_id}")
def fetch_run(run_id: str):
    r = get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return r
