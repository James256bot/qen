# QEN — Quantum Equation Nexus

A small quantum-mechanics simulator that runs on a phone.

Four physical systems, one HTTP API, one HTML page. No webpack. No Docker. No cloud. Just Python 3, numpy, and Termux.

## What it does

| system | physics | typical time |
|---|---|---|
| `particle` | 1D infinite square well, N=256 finite-difference diagonalization | 100-170 ms |
| `hydrogen` | Rydberg series, exact analytic | 0.1-0.4 ms |
| `n2` | Morse potential curve, 200-1000 sample points | 2-5 ms |
| `entropy` | von Neumann entropy of an arbitrary density matrix | 1-20 ms |

Runs on ARMv7 (Termux on Android). Exposes a FastAPI server with a single self-contained HTML frontend at `/`.

## Why it exists

Because "quantum simulation on a phone" is usually framed as impossible or absurd, and it turns out to just be a few hundred lines of Python if you pick the right tools.

It is deliberately **not** a research tool. It is an interactive teaching aid for people whose primary computer is a phone.

## Quick start

```bash
pkg install python numpy fastapi uvicorn
git clone <this-repo>
cd qen
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/` in a browser. From another device on the same Wi-Fi, use `http://<phone-ip>:8000/`.

## API

### `POST /simulations/run`

```json
{
  "system_type": "particle",
  "parameters": {"L_nm": 2.0, "n_states": 5}
}
```

Returns the full result plus server-side compute time.

Supported `system_type` values and their parameters:

- `particle` — `L_nm` (0.01-1000), `n_states` (1-32), optional `return_wavefunctions` (bool, default true)
- `hydrogen` — `Z` (1-20), `n_max` (1-20)
- `n2` — `n_points` (10-5000, default 200)
- `entropy` — `rho` (square matrix, trace 1, symmetric)

### `GET /simulations/history?limit=20`

Last N runs, newest first. In-memory, capped at 100.

### `GET /health`

`{"status": "ok", "runs_in_memory": N}`

## The interesting engineering bit

Termuxs OpenBLAS package for ARMv7 ships with `NO_LAPACK=1` and `NO_LAPACKE=1` compiled in. That means `numpy.linalg.eigh` falls back to numpys own reference LAPACK implementation, which is single-threaded and unvectorized. On a 1024-point grid, a dense eigendecomposition takes **5.8 seconds**.

Combined with:

- scipy has no ARMv7 wheel for Python 3.14
- building scipy from source fails during the numpy build dependency step
- numpy 2.5.x wont build from source on this platform either

…the practical path is:

1. Use the pre-installed system numpy (2.4.4)
2. Skip scipy entirely
3. Reduce grid resolution to N=256, which gives 4-5 significant digits on the eigenvalues
4. Use dense `numpy.linalg.eigh` (accept the O(N³), its ~100 ms at N=256)

The `particle` solver is ~200 lines including comments. The other three solvers are pure formulas and run in under 5 ms.

## Testing on a desktop

Same code, faster hardware. On an x86-64 laptop with a proper LAPACK:

- `particle` drops from ~130 ms to ~3 ms
- No code changes required

## Project layout

```
app/
  main.py              FastAPI app + dispatch + routes
  solvers/
    particle.py        1D box, finite-difference eigensolver
    hydrogen.py        Rydberg formula
    n2.py              Morse potential
    entropy.py         von Neumann entropy
  static/
    index.html         the entire UI (~350 lines, no build step)
```

## What it is not

- Not a quantum chemistry package (use Psi4, ORCA, Gaussian for that)
- Not a research simulator — the solvers are textbook formulas
- Not a web product — the API has no auth, no rate limits, no persistence

## License

MIT
