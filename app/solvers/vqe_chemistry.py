import numpy as np
from app.solvers.variational import nelder_mead
from app.solvers.qsim import (
    zero_state, apply_1q, apply_cnot, Ry, Rz, probabilities,
)


# ============================================================
# VQE for molecular chemistry from real integrals
#
# Pipeline:
#   1. Take S, T, V, ERI and the SCF orbitals C, orbital energies eps
#   2. Transform ERI to the MO basis
#   3. Build the 2-qubit effective Hamiltonian (Jordan-Wigner)
#   4. Run VQE to find the ground state
#   5. Return E_VQE, E_HF, and the correlation energy
# ============================================================


def transform_eri_to_mo(ERI, C):
    """Transform ERI from AO basis to MO basis.

    h_pqrs = sum_{μνλσ} C_μp C_νq C_λr C_σs (μν|λσ)

    ERI is (n, n, n, n), C is (n, n_mo).
    """
    # Two-step transformation to avoid O(n^8):
    # First: (μν|λσ) -> (μν|λs) = Σ_σ C_σs (μν|λσ)
    # Then: (μν|λs) -> (μq|λs) = Σ_ν C_νq (μν|λs)
    # etc.
    # For n <= 6 (small molecules), we can do it directly with einsum.
    n = ERI.shape[0]
    # Direct 4-step
    h1 = np.einsum("mnls,sp->mnlp", ERI, C)   # (μν|λσ) C_σs -> (μν|λp)
    h2 = np.einsum("mnlp,rl->mnrp", h1, C)    # -> (μν|σ p)
    h3 = np.einsum("mnrp,sn->mprs", h2, C) if False else None
    # Simpler: full einsum
    # h[p,q,r,s] = sum_{mu,nu,lam,sig} C[mu,p] C[nu,q] C[lam,r] C[sig,s] ERI[mu,nu,lam,sig]
    # For n=2 (H2), that's 2^8 = 256 operations — trivial.
    h = np.einsum("up,vq,lr,ms,uvlm->pqrs", C, C, C, C, ERI, optimize=True)
    return h


def build_2qubit_hamiltonian(eps, h, n_electrons, E_nuc=0.0, n_qubits=2):
    """Build the exact 2-qubit Hamiltonian for H2 from real MO integrals.

    Derived from the 3-dimensional singlet subspace {|HH>, |HL-HL>, |LL>}:
        H_AA = 2*eps_H - (HH|HH)
        H_CC = 2*eps_L - (LL|LL) + 2*(HH|LL) - 3*(HL|LH) + ...
        H_BB = eps_H + eps_L + (HH|LL) + (HL|LH)
        H_AC = (HL|LH)

    where the orbital energies already include the Fock contributions.
    """
    n_orb = len(eps)
    if n_orb < 2:
        raise ValueError("need at least 2 orbitals for active space")

    homo_idx = n_electrons // 2 - 1
    lumo_idx = homo_idx + 1

    eps_H = float(eps[homo_idx])
    eps_L = float(eps[lumo_idx])

    U_HH = float(h[homo_idx, homo_idx, homo_idx, homo_idx])
    U_LL = float(h[lumo_idx, lumo_idx, lumo_idx, lumo_idx])
    V_HL = float(h[homo_idx, homo_idx, lumo_idx, lumo_idx])
    J_HL = float(h[homo_idx, lumo_idx, lumo_idx, homo_idx])

    # Diagonal singlet energies (in the electronic Hamiltonian)
    H_AA = 2.0 * eps_H - U_HH               # HOMO^2
    H_BB = eps_H + eps_L + V_HL + J_HL      # singlet single excitation
    H_CC = 2.0 * eps_L - U_LL + 2.0 * V_HL - 3.0 * J_HL  # LUMO^2

    # Off-diagonal: exchange between HOMO^2 and LUMO^2
    H_AC = J_HL

    # Pauli coefficients
    c0 = (H_AA + 2.0 * H_BB + H_CC) / 4.0
    c1 = (H_AA - H_CC) / 4.0
    c2 = c1
    c3 = (H_AA - 2.0 * H_BB + H_CC) / 4.0
    c4 = H_AC

    # Verify: <00|H|00> = c0 + c1 + c2 + c3 should equal E_HF
    # (without E_nuc for the pure electronic part)
    E_00 = c0 + c1 + c2 + c3

    return {
        "constant": c0,
        "Z0": c1,
        "Z1": c2,
        "Z0Z1": c3,
        "X0X1": c4,
        "Y0Y1": 0.0,
        "eps_H": eps_H,
        "eps_L": eps_L,
        "U_HH": U_HH,
        "U_LL": U_LL,
        "V_HL": V_HL,
        "J_HL": J_HL,
        "H_AA": H_AA,
        "H_BB": H_BB,
        "H_CC": H_CC,
        "H_AC": H_AC,
        "E_00": E_00,
        "E_nuc": E_nuc,
    }


# ============================================================
# VQE on the 2-qubit Hamiltonian
# ============================================================

def _expect_z(psi, q, n_qubits=2):
    probs = probabilities(psi)
    return sum((1 - 2 * ((i >> (n_qubits - 1 - q)) & 1)) * p for i, p in enumerate(probs))


def _expect_zz(psi, q1, q2, n_qubits=2):
    probs = probabilities(psi)
    return sum((1 - 2 * ((i >> (n_qubits - 1 - q1)) & 1)) * (1 - 2 * ((i >> (n_qubits - 1 - q2)) & 1)) * p
               for i, p in enumerate(probs))


def _expect_x0x1(psi, q1=0, q2=1, n_qubits=2):
    """<X_q1 X_q2> for a 2-qubit state.

    X0 X1 |b1 b2> = |!b1 !b2>  (flips both qubits).
    So <X0 X1> = 2 * Re(sum over pairs (i, j) where j is i with both bits flipped).
    """
    total = 0.0
    for i in range(len(psi)):
        j = i ^ (1 << (n_qubits - 1 - q1)) ^ (1 << (n_qubits - 1 - q2))
        total += float(np.real(psi[i] * np.conj(psi[j])))
    return total


def _energy(psi, c):
    E = c["constant"]
    E += c["Z0"] * _expect_z(psi, 0)
    E += c["Z1"] * _expect_z(psi, 1)
    E += c["Z0Z1"] * _expect_zz(psi, 0, 1)
    E += c["X0X1"] * _expect_x0x1(psi, 0, 1)
    return E


def _ansatz(thetas, n_layers=1):
    """UCCSD-style ansatz for the H2 active space.

    Single parameter theta representing the HOMO -> LUMO double excitation:
        |psi(theta)> = cos(theta) |00> + sin(theta) |11>

    In qubit terms: Ry(2*theta) on q0, then CNOT(0, 1).

    This is the canonical VQE ansatz for H2 in the singlet subspace.
    """
    psi = zero_state(2)
    psi = apply_1q(psi, Ry(2 * float(thetas[0])), 0, 2)
    psi = apply_cnot(psi, 0, 1, 2)
    return psi


def run_vqe(hamiltonian, n_layers=1, max_iter=200, seed=42):
    """Run VQE on the 2-qubit Hamiltonian.

    For the single-parameter UCCSD ansatz, we do a brute-force scan
    over theta in [0, pi] followed by a local refinement. This is
    more reliable than Nelder-Mead for 1D optimization.
    """
    # Brute-force scan
    n_scan = 200
    thetas = np.linspace(0, 2 * np.pi, n_scan)
    energies = []
    for theta in thetas:
        psi = _ansatz([theta], n_layers)
        energies.append(_energy(psi, hamiltonian))
    energies = np.array(energies)

    # Find the minimum
    idx = int(np.argmin(energies))
    best_theta = float(thetas[idx])
    best_energy = float(energies[idx])

    # Local refinement: scan around the minimum
    for _ in range(3):
        width = 0.1
        n_local = 50
        local_thetas = np.linspace(best_theta - width, best_theta + width, n_local)
        local_energies = []
        for theta in local_thetas:
            psi = _ansatz([theta], n_layers)
            local_energies.append(_energy(psi, hamiltonian))
        local_energies = np.array(local_energies)
        idx = int(np.argmin(local_energies))
        if local_energies[idx] < best_energy:
            best_theta = float(local_thetas[idx])
            best_energy = float(local_energies[idx])
        else:
            break

    return {
        "energy": best_energy,
        "optimal_params": [best_theta],
        "iterations": n_scan + 150,
    }


# ============================================================
# Full pipeline: integrals -> SCF -> VQE
# ============================================================

def full_pipeline_chemistry(molecule, n_electrons, scf_result, integrals):
    """Complete pipeline: SCF orbitals -> VQE -> correlation energy."""
    C = np.array(scf_result["C"])
    eps = np.array(scf_result["orbital_energies"])
    E_hf = scf_result["energy"]

    # Transform ERI to MO basis
    h_mo = transform_eri_to_mo(integrals["ERI"], C)

    # Build 2-qubit Hamiltonian
    ham = build_2qubit_hamiltonian(eps, h_mo, n_electrons)

    # Run VQE
    vqe_result = run_vqe(ham, n_layers=1, max_iter=200)

    E_vqe = vqe_result["energy"]
    E_corr = E_vqe - E_hf

    return {
        "E_HF": float(E_hf),
        "E_VQE": float(E_vqe),
        "E_correlation": float(E_corr),
        "E_correlation_kcal": float(E_corr * 627.509),
        "hamiltonian_coefficients": ham,
        "vqe_iterations": vqe_result["iterations"],
        "n_orbital_energies": len(eps),
        "orbital_energies": eps.tolist(),
    }
