import numpy as np


# ============================================================
# Restricted Hartree-Fock (RHF) SCF
#
# Given S (overlap), T (kinetic), V (nuclear), ERI (electron repulsion),
# iterate the Fock matrix until the density matrix converges.
# ============================================================


def symmetric_orthogonalization(S):
    """Compute S^{-1/2} via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(S)
    # Discard near-zero eigenvalues (numerical noise)
    threshold = 1e-10
    inv_sqrt = np.zeros_like(eigvals)
    mask = eigvals > threshold
    inv_sqrt[mask] = 1.0 / np.sqrt(eigvals[mask])
    return eigvecs @ np.diag(inv_sqrt) @ eigvecs.T


def build_fock(P, H_core, ERI):
    """Build the Fock matrix from the density matrix.

    F_μν = H_μν + Σ_λσ P_λσ [(μν|λσ) - ½(μλ|νσ)]
    """
    n = P.shape[0]
    # Coulomb: J_μν = Σ_λσ P_λσ (μν|λσ)
    J = np.einsum("ls,mnls->mn", P, ERI)
    # Exchange: K_μν = Σ_λσ P_λσ (μλ|νσ)
    # Exchange: K_μν = Σ_λσ P_λσ (μλ|νσ)
    K = np.einsum("ls,mlns->mn", P, ERI)
    return H_core + J - 0.5 * K


def density_matrix(C, n_electrons):
    """P_μν = 2 Σ_i^occ C_μi C_νi (closed shell)."""
    n_occ = n_electrons // 2
    C_occ = C[:, :n_occ]
    return 2.0 * C_occ @ C_occ.T


def total_energy(P, H_core, F, E_nuc):
    """E_total = ½ Σ_μν P_μν (H_μν + F_μν) + E_nuc."""
    return 0.5 * np.sum(P * (H_core + F)) + E_nuc


def nuclear_repulsion(molecule):
    """Sum over pairs of nuclei: Z_A Z_B / R_AB."""
    atoms = molecule["atoms"]
    E_nuc = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            Zi = atoms[i]["Z"]
            Zj = atoms[j]["Z"]
            Ri = np.array(atoms[i]["coords"], dtype=float) * 1.8897259886  # Angstrom to Bohr
            Rj = np.array(atoms[j]["coords"], dtype=float) * 1.8897259886
            R = np.linalg.norm(Ri - Rj)
            if R > 1e-10:
                E_nuc += Zi * Zj / R
    return E_nuc


def run_scf(S, T, V, ERI, n_electrons, E_nuc, max_iter=50, tol=1e-8, verbose=False):
    """Run restricted Hartree-Fock SCF.

    Args:
        S: overlap matrix (n×n)
        T: kinetic energy matrix (n×n)
        V: nuclear attraction matrix (n×n)
        ERI: electron repulsion (n×n×n×n)
        n_electrons: total number of electrons
        E_nuc: nuclear repulsion energy
        max_iter: max SCF iterations
        tol: convergence threshold on density matrix

    Returns:
        dict with: converged, iterations, energy, orbital_energies, C, P, F
    """
    n = S.shape[0]
    H_core = T + V

    # Symmetric orthogonalization
    X = symmetric_orthogonalization(S)

    # Initial guess: diagonalize the core Hamiltonian
    F0 = H_core.copy()
    F0_prime = X.T @ F0 @ X
    eigvals0, C0_prime = np.linalg.eigh(F0_prime)
    C0 = X @ C0_prime

    P = density_matrix(C0, n_electrons)
    energy_history = []

    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration

        # Build Fock matrix
        F = build_fock(P, H_core, ERI)

        # Total energy (using current P and F)
        E_total = total_energy(P, H_core, F, E_nuc)
        energy_history.append(E_total)

        # Transform to orthogonal basis
        F_prime = X.T @ F @ X

        # Diagonalize
        eigvals, C_prime = np.linalg.eigh(F_prime)

        # Back-transform to original basis
        C = X @ C_prime

        # New density matrix
        P_new = density_matrix(C, n_electrons)

        # Convergence check
        dP = np.max(np.abs(P_new - P))
        if verbose:
            print(f"  iter {iteration}: E = {E_total:.8f} Ha, |dP| = {dP:.2e}")

        if dP < tol:
            converged = True
            P = P_new
            break

        P = P_new

    # Final energy
    F = build_fock(P, H_core, ERI)
    E_final = total_energy(P, H_core, F, E_nuc)

    return {
        "converged": converged,
        "iterations": iterations,
        "energy": E_final,
        "orbital_energies": eigvals.tolist(),
        "C": C.tolist(),
        "P": P.tolist(),
        "F": F.tolist(),
        "H_core": H_core.tolist(),
        "E_nuc": E_nuc,
        "energy_history": energy_history,
    }
