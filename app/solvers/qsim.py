import numpy as np
from app.solvers.bloch import bloch_of_qubit

# ============================================================
# State-vector quantum circuit simulator
# Qubits: |q_{n-1} ... q_1 q_0>, MSB first in the state index.
# State is a complex vector of length 2^n.
# ============================================================


def zero_state(n_qubits: int) -> np.ndarray:
    """|0...0>, complex64."""
    if not (1 <= n_qubits <= 20):
        raise ValueError("n_qubits must be between 1 and 20")
    psi = np.zeros(2 ** n_qubits, dtype=np.complex64)
    psi[0] = 1.0
    return psi


# ---------- single-qubit gates ----------

GATES_1Q = {
    "I": np.eye(2, dtype=np.complex64),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex64),
    "T": np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex64),
}


def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex64)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)


def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=np.complex64)


def apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """Apply a single-qubit gate to `target` of an n-qubit state.

    Bit-arithmetic: for each pair of indices (i, i^(1<<tbit)) differing only
    in the target bit, apply the 2x2 gate matrix. MSB-first: qubit 0 is the
    high bit.
    """
    if gate.shape != (2, 2):
        raise ValueError("gate must be 2x2")
    n = len(psi)
    tbit = n_qubits - 1 - target
    idx = np.arange(n, dtype=np.int64)
    target_bit = (idx >> tbit) & 1
    partner = idx ^ (1 << tbit)
    mask0 = (target_bit == 0)
    a = psi[mask0]
    b = psi[partner[mask0]]
    new_a = gate[0, 0] * a + gate[0, 1] * b
    new_b = gate[1, 0] * a + gate[1, 1] * b
    out = psi.astype(np.complex128).copy()
    out[idx[mask0]] = new_a
    out[partner[mask0]] = new_b
    return out.astype(np.complex64)


def apply_cnot(psi: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """CNOT via direct bit-arithmetic on the flat state vector.

    For each basis state index j:
      - if control bit of j is 1, the amplitude at j comes from j ^ (1 << tbit)
      - otherwise, the amplitude at j is psi[j]
    Correct by construction, O(2^n) numpy-vectorized.
    """
    if control == target:
        raise ValueError("control and target must differ")
    n = len(psi)
    cbit = n_qubits - 1 - control
    tbit = n_qubits - 1 - target
    idx = np.arange(n, dtype=np.int64)
    ctrl_is_1 = ((idx >> cbit) & 1) == 1
    src = idx ^ (1 << tbit)
    out = np.where(ctrl_is_1, psi[src], psi)
    return out.astype(np.complex64)


def apply_cz(psi: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """CZ: multiply by -1 the |11> amplitude."""
    psi = psi.reshape([2] * n_qubits)
    caxis = n_qubits - 1 - control
    taxis = n_qubits - 1 - target
    psi = np.moveaxis(psi, [caxis, taxis], [0, 1])
    shape = psi.shape
    psi = psi.reshape(2, 2, -1)
    psi[1, 1, :] *= -1
    psi = psi.reshape(shape)
    psi = np.moveaxis(psi, [0, 1], [caxis, taxis])
    return psi.reshape(-1).astype(np.complex64)


def apply_swap(psi: np.ndarray, a: int, b: int, n_qubits: int) -> np.ndarray:
    """SWAP qubits a and b."""
    if a == b:
        return psi
    psi = psi.reshape([2] * n_qubits)
    aaxis = n_qubits - 1 - a
    baxis = n_qubits - 1 - b
    psi = np.swapaxes(psi, aaxis, baxis)
    return psi.reshape(-1).astype(np.complex64)


# ---------- measurement ----------

def probabilities(psi: np.ndarray) -> np.ndarray:
    return (np.abs(psi) ** 2).astype(np.float64)


def sample_shots(psi: np.ndarray, shots: int, rng=None):
    """Sample bitstrings from |psi|^2. Returns dict {bitstring: count}."""
    if rng is None:
        rng = np.random.default_rng()
    p = probabilities(psi)
    p = p / p.sum()
    n_qubits = int(np.log2(len(psi)))
    idx = rng.choice(len(psi), size=shots, p=p)
    counts = {}
    for i in idx:
        key = format(int(i), f"0{n_qubits}b")
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------- entanglement entropy ----------

def entropy_of_subsystem(psi: np.ndarray, keep: list, n_qubits: int) -> float:
    """Von Neumann entropy of the reduced density matrix on `keep` qubits."""
    psi_t = psi.reshape([2] * n_qubits)
    # axes: axis k corresponds to qubit (n_qubits - 1 - k)
    keep_axes = [n_qubits - 1 - q for q in keep]
    trace_axes = [a for a in range(n_qubits) if a not in keep_axes]
    # move keep to front, trace to back
    perm = keep_axes + trace_axes
    psi_t = np.transpose(psi_t, perm).reshape(2 ** len(keep), -1)
    # reduced density matrix: rho = psi_t @ psi_t^dagger
    rho = psi_t @ psi_t.conj().T
    # eigenvalues (real, nonnegative)
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-10]
    S = float(-np.sum(w * np.log(w)))
    return S


# ---------- circuit execution ----------

def _parse_gate(gate_spec):
    """A gate spec is a dict:
        {"gate": "H", "targets": [0], "params": []}
        {"gate": "CNOT", "targets": [0, 1]}       # control=0, target=1
        {"gate": "CZ", "targets": [0, 1]}
        {"gate": "SWAP", "targets": [0, 1]}
        {"gate": "Rx", "targets": [0], "params": [theta]}
    """
    if not isinstance(gate_spec, dict):
        raise ValueError("gate must be a dict")
    name = gate_spec.get("gate", "").upper()
    targets = gate_spec.get("targets", [])
    params = gate_spec.get("params", [])
    return name, targets, params


def execute_circuit(n_qubits: int, gates: list) -> np.ndarray:
    psi = zero_state(n_qubits)
    for spec in gates:
        name, targets, params = _parse_gate(spec)
        if name in GATES_1Q:
            if len(targets) != 1:
                raise ValueError(f"{name} needs exactly 1 target")
            psi = apply_1q(psi, GATES_1Q[name], targets[0], n_qubits)
        elif name == "RX":
            psi = apply_1q(psi, Rx(float(params[0])), targets[0], n_qubits)
        elif name == "RY":
            psi = apply_1q(psi, Ry(float(params[0])), targets[0], n_qubits)
        elif name == "RZ":
            psi = apply_1q(psi, Rz(float(params[0])), targets[0], n_qubits)
        elif name == "CNOT" or name == "CX":
            if len(targets) != 2:
                raise ValueError("CNOT needs [control, target]")
            psi = apply_cnot(psi, targets[0], targets[1], n_qubits)
        elif name == "CZ":
            psi = apply_cz(psi, targets[0], targets[1], n_qubits)
        elif name == "SWAP":
            psi = apply_swap(psi, targets[0], targets[1], n_qubits)
        else:
            raise ValueError(f"unknown gate: {name}")
    return psi


def run_circuit(n_qubits: int, gates: list, shots: int = 0, entanglement_cut: list = None, bloch_qubits: list = None):
    psi = execute_circuit(n_qubits, gates)
    probs = probabilities(psi)
    result = {
        "n_qubits": n_qubits,
        "n_states": len(psi),
        "probabilities": probs.tolist(),
        "amplitudes_real": psi.real.tolist(),
        "amplitudes_imag": psi.imag.tolist(),
        "norm": float(np.sum(probs)),
    }
    if shots > 0:
        counts = sample_shots(psi, shots)
        result["shots"] = shots
        result["counts"] = counts
    if entanglement_cut is not None:
        S = entropy_of_subsystem(psi, entanglement_cut, n_qubits)
        result["entanglement_cut"] = entanglement_cut
        result["entanglement_entropy_nats"] = S
        result["entanglement_entropy_bits"] = S / float(np.log(2.0))
    if bloch_qubits:
        result["bloch"] = [bloch_of_qubit(psi, n_qubits, q) for q in bloch_qubits]
    return result


# ============================================================
# Canonical circuits — convenient presets
# ============================================================

CANONICAL_CIRCUITS = {
    "bell": {
        "n_qubits": 2,
        "gates": [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [0, 1]},
        ],
        "description": "Bell pair (|00> + |11>)/sqrt(2). Maximal 2-qubit entanglement.",
    },
    "ghz3": {
        "n_qubits": 3,
        "gates": [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [0, 1]},
            {"gate": "CNOT", "targets": [1, 2]},
        ],
        "description": "3-qubit GHZ state (|000> + |111>)/sqrt(2).",
    },
    "ghz5": {
        "n_qubits": 5,
        "gates": [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [0, 1]},
            {"gate": "CNOT", "targets": [1, 2]},
            {"gate": "CNOT", "targets": [2, 3]},
            {"gate": "CNOT", "targets": [3, 4]},
        ],
        "description": "5-qubit GHZ state.",
    },
    "superposition": {
        "n_qubits": 1,
        "gates": [{"gate": "H", "targets": [0]}],
        "description": "Single-qubit equal superposition.",
    },
    "qft2": {
        "n_qubits": 2,
        "gates": [
            {"gate": "H", "targets": [0]},
            {"gate": "S", "targets": [1]},
            {"gate": "CNOT", "targets": [0, 1]},
            {"gate": "H", "targets": [0]},
        ],
        "description": "2-qubit quantum Fourier transform (simplified).",
    },
    "grover2": {
        "n_qubits": 2,
        "gates": [
            {"gate": "H", "targets": [0]},
            {"gate": "H", "targets": [1]},
            {"gate": "CZ", "targets": [0, 1]},
            {"gate": "H", "targets": [0]},
            {"gate": "H", "targets": [1]},
            {"gate": "Z", "targets": [0]},
            {"gate": "Z", "targets": [1]},
            {"gate": "CZ", "targets": [0, 1]},
            {"gate": "H", "targets": [0]},
            {"gate": "H", "targets": [1]},
        ],
        "description": "2-qubit Grover iteration — finds |11>.",
    },
}


def get_canonical(name):
    if name not in CANONICAL_CIRCUITS:
        return None
    return CANONICAL_CIRCUITS[name]
