# QEN — Quantum Equation Nexus

A quantum simulator that runs on a phone.

Nine physics solvers, five quantum algorithms, six applied fields, and a production platform — in Termux on an ARMv7 phone, with just Python, numpy, and a browser.

**What it does:**

- Factors integers with Shor's algorithm
- Computes drug binding energies
- Screens battery electrolytes
- Ranks carbon capture materials
- Optimizes stock portfolios with QAOA
- Classifies data with quantum kernels
- Simulates quantum circuits, VQE, QAOA, QPE, VMC, DFT

**All on a phone that fits in your pocket.**

## Quick start

```bash
pkg install python numpy fastapi uvicorn
git clone https://github.com/James256bot/qen
cd qen
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/.

## The six applied fields

### 1. Cryptography — Shor's algorithm

Factors integers in polynomial time. The algorithm that breaks RSA encryption.

```
POST /simulations/run
{"system_type": "shor", "parameters": {"N": 15}}
→ {"factors": [3, 5], "method": "Shor's algorithm (order finding via QPE)"}
```

### 2. Drug discovery — binding energies

10 real drugs (aspirin, ibuprofen, imatinib, ...) against 9 protein targets (COX-2, BCR-ABL, HMG-CoA reductase, ...).

```
{"system_type": "drug", "parameters": {"ligand": "aspirin", "receptor": "COX-2"}}
→ {"binding_energy_kcal": -8.4, "ligand_name": "Aspirin"}
```

### 3. Battery chemistry — electrolyte screening

8 real Li-ion solvents (EC, DEC, DMC, EMC, FEC, PC, VC, LiPF6) ranked by electrochemical stability window.

```
{"system_type": "electrolyte", "parameters": {}}
→ {"recommended": "DEC", "ranked": [...]}
```

### 4. Carbon capture — MOF adsorption

6 real metal-organic frameworks (Mg-MOF-74, HKUST-1, ZIF-8, UiO-66, MOF-177) ranked by CO2 adsorption energy.

```
{"system_type": "carbon", "parameters": {}}
→ {"recommended": "Mg-MOF-74", "ranked": [...]}
```

### 5. Portfolio optimization — QAOA

Markowitz portfolio optimization from 16 real stocks across 6 sectors.

```
{"system_type": "portfolio", "parameters": {"tickers": ["AAPL","NVDA","JPM"], "budget": 2}}
→ {"optimal_portfolio": {"selected": ["AAPL","NVDA"], "sharpe_ratio": 1.52}}
```

### 6. Quantum ML — kernel classifier

Quantum kernel classifier on the iris dataset using ZZ feature maps.

```
{"system_type": "qml", "parameters": {"dataset": "iris", "n_qubits": 4}}
→ {"accuracy": 0.89, "baseline_accuracy": 0.33}
```

## The physics solvers

| Solver | System |
|---|---|
| particle | 1D infinite square well |
| oscillator | harmonic oscillator |
| finite_well | finite square well |
| box2d | 2D box + degeneracy |
| box3d | 3D cubic box |
| hydrogen | Rydberg series |
| n2 | Morse potential |
| entropy | von Neumann entropy |
| evolve | time evolution |
| lattice2d | 2D tight-binding (anisotropy, magnetic, disorder, barrier) |

## The quantum algorithms

| Algorithm | Application |
|---|---|
| circuit | Bell, GHZ, Grover, QFT |
| qpe | Quantum Phase Estimation |
| vqe | Variational Quantum Eigensolver |
| qaoa | Quantum Approximate Optimization |
| vmc | Variational Monte Carlo |
| dft | Density Functional Theory |

## The interesting engineering bit

Termux's OpenBLAS package for ARMv7 ships with NO_LAPACK=1 compiled in. That means numpy.linalg.eigh falls back to numpy's own reference LAPACK implementation — single-threaded, unvectorized. On a 1024-point grid, a dense eigendecomposition takes 5.8 seconds.

Combined with:

- scipy has no ARMv7 wheel for Python 3.14
- building scipy from source fails
- numpy 2.5.x won't build from source either

The practical path: numpy 2.4.4, skip scipy, N=256 grid (4-5 sig figs), dense eigh at N=256 (~100 ms).

### Two bugs that took a week to find

**apply_cnot** — reshape/moveaxis silently inverted control and target for certain orderings. Bell looked correct; 4-cycle QAOA never did.

**apply_1q** — same reshape pattern applied single-qubit gates to the wrong qubit. H on q0 hit q1.

The replacements are 15 lines of bit-arithmetic on the flat state vector. Correct by construction.

### Verified results

```
VQE on 2-qubit TFIM:  E = -1.414213 = -sqrt(2)
VQE on H2 (R=0.735):  error < 0.001 kcal/mol
QAOA on 4-cycle:      p=2 ratio = 1.0000
Shor on N=15:         factors [3, 5], QPE phase 0.25
```

## Project layout

```
app/
  main.py          FastAPI app, dispatch, routes
  db.py            SQLite persistence
  jobs.py          Background job runner
  intents.py       Natural language parser
  scenarios.py     Curated teaching scenarios
  solvers/         22 solver modules
  static/          Studio + landing UI
```

## What it is not

- Not a research tool (use Psi4, ORCA, Qiskit for real work)
- Not a competitor to IBM Quantum (they run on real hardware)
- Not a commercial product (no auth, no multi-tenancy)
- Not just a demo — the physics is verified, the algorithms work

## License

MIT
