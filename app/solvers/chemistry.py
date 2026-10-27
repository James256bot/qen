import numpy as np
from app.solvers.qsim import (
    zero_state, apply_1q, apply_cnot, Ry, Rz, probabilities,
)
from app.solvers.variational import nelder_mead


# ============================================================
# H2 STO-3G Jordan-Wigner 2-qubit Hamiltonian
#
#   H = g0 * I + g1 * Z0 + g2 * Z1 + g3 * Z0 Z1 + g4 (X0 X1 + Y0 Y1)
#
# The coefficients depend on the internuclear distance R. Computing them
# exactly requires the two-center overlap, one-electron, and two-electron
# integrals in the STO-3G basis. Here we use approximate values chosen so
# the Hamiltonian has a physically sensible ground-state energy at each R.
#
# IMPORTANT: only R = 0.735 A is a literature value. Other geometries are
# approximate. The purpose of this module is to demonstrate VQE, not to
# reproduce the exact H2 potential energy surface.
# ============================================================


H2_TABLE = {
    0.40: {"constant": -0.7500, "Z0": 0.7500, "Z0Z1": 0.0170, "X0X1": 0.2200},
    0.50: {"constant": -0.9500, "Z0": 0.6000, "Z0Z1": 0.0080, "X0X1": 0.2050},
    0.60: {"constant": -1.0200, "Z0": 0.4800, "Z0Z1": -0.0050, "X0X1": 0.1920},
    0.735:{"constant": -1.0524, "Z0": 0.3979, "Z0Z1": -0.0113, "X0X1": 0.1809},
    0.90: {"constant": -1.0410, "Z0": 0.3450, "Z0Z1": -0.0100, "X0X1": 0.1520},
    1.10: {"constant": -1.0100, "Z0": 0.2950, "Z0Z1": -0.0050, "X0X1": 0.1180},
    1.40: {"constant": -0.9600, "Z0": 0.2450, "Z0Z1": 0.0050, "X0X1": 0.0800},
    1.80: {"constant": -0.9200, "Z0": 0.2050, "Z0Z1": 0.0120, "X0X1": 0.0450},
    2.50: {"constant": -0.8950, "Z0": 0.1800, "Z0Z1": 0.0170, "X0X1": 0.0180},
    3.50: {"constant": -0.8850, "Z0": 0.1700, "Z0Z1": 0.0190, "X0X1": 0.0060},
    5.00: {"constant": -0.8820, "Z0": 0.1650, "Z0Z1": 0.0200, "X0X1": 0.0010},
}


def h2_coefficients(R_angstrom):
    """H2 STO-3G coefficients at the given R, via linear interpolation."""
    R = float(R_angstrom)
    keys = sorted(H2_TABLE.keys())
    if R <= keys[0]:
        entry = H2_TABLE[keys[0]]
    elif R >= keys[-1]:
        entry = H2_TABLE[keys[-1]]
    else:
        entry = H2_TABLE[keys[-1]]
        for i in range(len(keys) - 1):
            r1, r2 = keys[i], keys[i + 1]
            if r1 <= R <= r2:
                t = (R - r1) / (r2 - r1)
                c1 = H2_TABLE[r1]
                c2 = H2_TABLE[r2]
                entry = {k: (1 - t) * c1[k] + t * c2[k] for k in c1}
                break
    z = entry["Z0"]
    return {
        "constant": entry["constant"],
        "Z0": z, "Z1": z,
        "Z0Z1": entry["Z0Z1"],
        "X0X1": entry["X0X1"],
    }


# ============================================================
# Pauli expectation values on a state vector
# ============================================================


def _pauli_z_expectation(psi, qubit, n_qubits):
    probs = probabilities(psi)
    total = 0.0
    for i, p in enumerate(probs):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        total += (1 - 2 * bit) * p
    return float(total)


def _pauli_zz_expectation(psi, q1, q2, n_qubits):
    probs = probabilities(psi)
    total = 0.0
    for i, p in enumerate(probs):
        b1 = (i >> (n_qubits - 1 - q1)) & 1
        b2 = (i >> (n_qubits - 1 - q2)) & 1
        total += (1 - 2 * b1) * (1 - 2 * b2) * p
    return float(total)


def _pauli_xx_plus_yy_expectation(psi, q1, q2, n_qubits):
    n = len(psi)
    total = 0.0
    for i in range(n):
        b1 = (i >> (n_qubits - 1 - q1)) & 1
        b2 = (i >> (n_qubits - 1 - q2)) & 1
        if b1 == b2:
            continue
        j = i ^ (1 << (n_qubits - 1 - q1)) ^ (1 << (n_qubits - 1 - q2))
        total += 2 * float(np.real(psi[i] * np.conj(psi[j])))
    return float(total)


def h2_energy(psi, R_angstrom):
    c = h2_coefficients(R_angstrom)
    n = 2
    E = c["constant"]
    E += c["Z0"] * _pauli_z_expectation(psi, 0, n)
    E += c["Z1"] * _pauli_z_expectation(psi, 1, n)
    E += c["Z0Z1"] * _pauli_zz_expectation(psi, 0, 1, n)
    E += c["X0X1"] * _pauli_xx_plus_yy_expectation(psi, 0, 1, n)
    return float(E)


# ============================================================
# Ansatz and VQE runner
# ============================================================


def _ansatz_state(thetas, n_qubits, n_layers):
    psi = zero_state(n_qubits)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            psi = apply_1q(psi, Rz(float(thetas[idx])), q, n_qubits)
            idx += 1
            psi = apply_1q(psi, Ry(float(thetas[idx])), q, n_qubits)
            idx += 1
        for q in range(n_qubits - 1):
            psi = apply_cnot(psi, q, q + 1, n_qubits)
    return psi


def run_vqe_h2(R_angstrom=0.735, n_layers=2, max_iter=400, seed=42):
    n_qubits = 2
    n_params = 2 * n_qubits * n_layers
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=n_params)

    history = []

    def cost(thetas):
        psi = _ansatz_state(thetas, n_qubits, n_layers)
        E = h2_energy(psi, R_angstrom)
        history.append(E)
        return E

    x_opt, e_opt, iterations = nelder_mead(cost, x0, max_iter=max_iter)
    psi_opt = _ansatz_state(x_opt, n_qubits, n_layers)
    probs = probabilities(psi_opt).tolist()
    return {
        "algorithm": "VQE_H2",
        "molecule": "H2",
        "R_angstrom": float(R_angstrom),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": n_params,
        "optimal_params": x_opt.tolist(),
        "energy_hartree": float(e_opt),
        "energy_history": [float(e) for e in history],
        "iterations": iterations,
        "final_probabilities": probs,
    }


def h2_exact_ground_energy(R_angstrom):
    """Exact ground-state energy of the 2-qubit H2 Hamiltonian."""
    c = h2_coefficients(R_angstrom)
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = c["constant"] * np.kron(I, I)
    H += c["Z0"] * np.kron(Z, I)
    H += c["Z1"] * np.kron(I, Z)
    H += c["Z0Z1"] * np.kron(Z, Z)
    H += c["X0X1"] * (np.kron(X, X) + np.kron(Y, Y))
    return float(np.linalg.eigvalsh(H)[0])
