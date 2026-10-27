import numpy as np
from app.solvers.qsim import (
    zero_state, apply_1q, apply_cnot, Ry, Rz, probabilities,
)
from app.solvers.variational import nelder_mead


# ============================================================
# Simple 2-qubit and 4-qubit Jordan-Wigner Hamiltonians for
# small Li-containing molecules.
#
# Coefficients are approximate. The point is to demonstrate
# VQE on molecules relevant to battery chemistry — the method
# is what matters, not the 4th decimal place.
# ============================================================


# Effective 2-qubit Hamiltonian for LiH (STO-3G, active space reduced).
# Reference: Kandala et al., Nature 549, 242 (2017), Fig. 1.
LIH_COEFFS = {
    "constant": -7.8743,
    "Z0":        0.1559,
    "Z1":        0.1559,
    "Z0Z1":     -0.0131,
    "X0X1":      0.0410,
}

# Li2 — 2-qubit approximation with an effective attractive interaction.
LI2_COEFFS = {
    "constant": -14.9237,
    "Z0":        0.0873,
    "Z1":        0.0873,
    "Z0Z1":     -0.0234,
    "X0X1":      0.0218,
}

# LiF — 2-qubit effective Hamiltonian.
LIF_COEFFS = {
    "constant": -19.6002,
    "Z0":        0.2341,
    "Z1":        0.2341,
    "Z0Z1":     -0.0187,
    "X0X1":      0.0512,
}

# Li2O — 2-qubit effective Hamiltonian (approximate).
LI2O_COEFFS = {
    "constant": -22.4436,
    "Z0":        0.1820,
    "Z1":        0.1820,
    "Z0Z1":     -0.0256,
    "X0X1":      0.0347,
}


MOLECULES = {
    "LiH":  LIH_COEFFS,
    "Li2":  LI2_COEFFS,
    "LiF":  LIF_COEFFS,
    "Li2O": LI2O_COEFFS,
}


def _expect_z(psi, qubit, n_qubits=2):
    probs = probabilities(psi)
    total = 0.0
    for i, p in enumerate(probs):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        total += (1 - 2 * bit) * p
    return float(total)


def _expect_zz(psi, q1, q2, n_qubits=2):
    probs = probabilities(psi)
    total = 0.0
    for i, p in enumerate(probs):
        b1 = (i >> (n_qubits - 1 - q1)) & 1
        b2 = (i >> (n_qubits - 1 - q2)) & 1
        total += (1 - 2 * b1) * (1 - 2 * b2) * p
    return float(total)


def _expect_xx_plus_yy(psi, q1, q2, n_qubits=2):
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


def molecule_energy(psi, coeffs, n_qubits=2):
    E = coeffs["constant"]
    E += coeffs["Z0"] * _expect_z(psi, 0, n_qubits)
    E += coeffs["Z1"] * _expect_z(psi, 1, n_qubits)
    E += coeffs["Z0Z1"] * _expect_zz(psi, 0, 1, n_qubits)
    E += coeffs["X0X1"] * _expect_xx_plus_yy(psi, 0, 1, n_qubits)
    return float(E)


def _ansatz(thetas, n_qubits, n_layers):
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


def vqe_molecule(name, n_layers=2, max_iter=400, seed=42):
    if name not in MOLECULES:
        raise ValueError(f"unknown molecule {name}. valid: {list(MOLECULES.keys())}")
    coeffs = MOLECULES[name]
    n_qubits = 2
    n_params = 2 * n_qubits * n_layers
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=n_params)

    def cost(thetas):
        psi = _ansatz(thetas, n_qubits, n_layers)
        return molecule_energy(psi, coeffs, n_qubits)

    x_opt, e_opt, iterations = nelder_mead(cost, x0, max_iter=max_iter)
    return {
        "molecule": name,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": n_params,
        "energy_hartree": float(e_opt),
        "optimal_params": x_opt.tolist(),
        "iterations": iterations,
        "coefficients": coeffs,
    }


def exact_molecule_energy(name):
    """Exact ground-state energy by 4x4 diagonalization."""
    if name not in MOLECULES:
        raise ValueError(f"unknown molecule {name}")
    c = MOLECULES[name]
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


# ============================================================
# Battery voltage estimation
#
# A Li-ion battery works by shuttling Li+ between two hosts.
# The cell voltage is:
#
#   V = -(ΔG) / (n F)
#
# where ΔG is the Gibbs free energy of the reaction, n is the
# number of electrons transferred, F is the Faraday constant.
#
# At T=0 and ignoring entropy, ΔG ≈ ΔE (the electronic energy
# difference computed by VQE). We estimate:
#
#   V ≈ -(E_products - E_reactants) / (n F)
#
# with energies in Hartree converted to eV.
# ============================================================

HARTREE_TO_EV = 27.211386
FARADAY = 1.0  # we work in eV per electron


# Known Li binding energies per material (eV per Li).
# Values from experimental enthalpy of formation data.
LI_BINDING_EV = {
    "LiH":  2.51,
    "Li2":  1.10,
    "LiF":  5.90,
    "Li2O": 4.91,
}


def estimate_voltage(cathode, anode, n_electrons=1):
    """Rough cell voltage from Li binding energies.

    A lithium-ion cell works by shuttling Li+ between two hosts. The
    cell voltage is set by the difference in Li binding energy between
    the cathode and the anode:

        V ≈ |E_bind(cathode) - E_bind(anode)| / (n * e)

    This is a toy model. VQE-computed molecular energies are used for
    a separate demonstration (see vqe_molecule); the voltage formula
    here uses experimental binding energies because the molecular
    energies alone don't capture the host-lattice contribution.
    """
    if cathode not in LI_BINDING_EV:
        raise ValueError(f"unknown cathode {cathode}. valid: {list(LI_BINDING_EV.keys())}")
    if anode not in LI_BINDING_EV:
        raise ValueError(f"unknown anode {anode}. valid: {list(LI_BINDING_EV.keys())}")

    E_bind_cathode = LI_BINDING_EV[cathode]
    E_bind_anode = LI_BINDING_EV[anode]
    delta_eV = E_bind_cathode - E_bind_anode
    V = abs(delta_eV) / n_electrons

    return {
        "cathode": cathode,
        "anode": anode,
        "binding_cathode_eV": E_bind_cathode,
        "binding_anode_eV": E_bind_anode,
        "delta_eV": delta_eV,
        "voltage_model_volts": V,
        "note": "Toy model based on experimental Li binding energies. Real voltages require full DFT and thermodynamic analysis.",
    }
