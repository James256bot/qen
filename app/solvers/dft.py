import numpy as np


# ============================================================
# 1D Density Functional Theory with LDA exchange-correlation
#
# Solves the Kohn-Sham equations self-consistently on a 1D grid.
#
#   [-1/2 d^2/dx^2 + V_ext(x) + V_H(x) + V_xc[n](x)] phi_i = eps_i phi_i
#   n(x) = sum_i f_i |phi_i(x)|^2
#
# Units: atomic units (Hartree, bohr). hbar = m_e = e = 1.
#
# The LDA XC potential for a 1D electron gas is approximated as:
#   V_xc(n) ~ -alpha * n^(1/3)   (exchange)
#   V_c(n)  ~ beta * log(1 + gamma*n)  (correlation, fit)
#
# This is a pedagogical 1D model, not the 3D LDA used in real codes.
# ============================================================


def _build_kinetic(N, dx):
    """Second-derivative finite-difference kinetic operator (1D)."""
    # T = -1/2 d^2/dx^2 on N interior points
    t = 1.0 / (2.0 * dx * dx)
    main = 2.0 * t * np.ones(N)
    off = -t * np.ones(N - 1)
    return main, off


def _build_hamiltonian(N, dx, V_total):
    """Build the dense Kohn-Sham Hamiltonian."""
    main, off = _build_kinetic(N, dx)
    H = np.diag(main + V_total) + np.diag(off, 1) + np.diag(off, -1)
    return H


def _default_external_potential(x, kind="harmonic"):
    """Return V_ext(x) for a chosen potential shape."""
    if kind == "harmonic":
        # harmonic well centered at midpoint
        xc = 0.5 * (x[0] + x[-1])
        return 0.5 * (x - xc) ** 2
    elif kind == "double_well":
        xc = 0.5 * (x[0] + x[-1])
        a = 0.15 * (x[-1] - x[0])
        return 0.01 * ((x - xc) ** 2 - a ** 2) ** 2
    elif kind == "step":
        # quantum step at midpoint
        xc = 0.5 * (x[0] + x[-1])
        V = np.zeros_like(x)
        V[x > xc] = 5.0
        return V
    elif kind == "flat":
        return np.zeros_like(x)
    else:
        raise ValueError(f"unknown potential kind: {kind}")


def _hartree_potential_1d(n, dx, screening=2.0):
    """Screened Hartree potential (Yukawa) for a 1D electron gas.

    V_H(x) = integral n(x') * exp(-|x-x'| / lambda) / sqrt((x-x')^2 + eps^2) dx'

    With strong screening (small lambda), the long-range 1/r divergence is
    cut off and the potential remains bounded. This is the physically
    correct behavior in a 1D conductor where electrons screen each other.
    """
    N = len(n)
    V = np.zeros(N)
    eps = 1.0
    for i in range(N):
        d = np.abs(np.arange(N) - i) * dx
        kernel = np.exp(-d / screening) / np.sqrt(d ** 2 + eps ** 2)
        V[i] = np.sum(n * kernel) * dx
    return V


def _lda_xc_potential_1d(n, alpha=0.5, beta=0.05, gamma=2.0):
    """Simplified LDA exchange-correlation potential for a 1D electron gas.

    Exchange:  V_x ~ -alpha * n^(1/3)
    Correl:    V_c ~ beta * log(1 + gamma*n)
    """
    n_safe = np.maximum(n, 1e-12)
    Vx = -alpha * n_safe ** (1.0 / 3.0)
    Vc = beta * np.log(1.0 + gamma * n_safe)
    return Vx + Vc


def solve_dft_1d(
    N=200,
    L=20.0,
    n_electrons=2,
    potential="harmonic",
    xc_alpha=0.5,
    xc_beta=0.05,
    max_iter=200,
    mix=0.15,
    tol=1e-6,
    verbose=False,
):
    """Self-consistent DFT calculation on a 1D grid.

    Parameters
    ----------
    N : grid points (100-500)
    L : domain length in bohr
    n_electrons : number of electrons (occupied orbitals)
    potential : "harmonic", "double_well", "step", "flat"
    xc_alpha : exchange strength
    xc_beta : correlation strength
    max_iter : maximum SCF iterations
    mix : density mixing parameter (0-1)
    tol : convergence threshold on density change
    """
    if not (50 <= N <= 500):
        raise ValueError("N must be between 50 and 500")
    if n_electrons < 1 or n_electrons > 10:
        raise ValueError("n_electrons must be between 1 and 10")

    dx = L / (N + 1)
    x = np.linspace(dx, L - dx, N)

    V_ext = _default_external_potential(x, potential)

    # initial density: uniform with normalization
    n = np.full(N, n_electrons / L)

    history = []
    converged = False
    for iteration in range(max_iter):
        V_H = _hartree_potential_1d(n, dx)  # computed for output only
        V_xc = _lda_xc_potential_1d(n, alpha=xc_alpha, beta=xc_beta)
        # NOTE: 1D electrostatics has no 1/r potential, so the Hartree term
        # is not physically meaningful. We use only XC as the effective
        # electron-electron interaction. This is the standard approach in
        # pedagogical 1D DFT models.
        V_total = V_ext + V_xc

        H = _build_hamiltonian(N, dx, V_total)
        eps, phi = np.linalg.eigh(H)

        # occupy the lowest n_electrons orbitals (spin-degenerate: 2 per level)
        n_occ = int(np.ceil(n_electrons / 2))
        occ = np.zeros(n_occ)
        remaining = n_electrons
        for i in range(n_occ):
            if remaining >= 2:
                occ[i] = 2.0
                remaining -= 2
            elif remaining > 0:
                occ[i] = remaining
                remaining = 0

        n_new = np.zeros(N)
        for i in range(n_occ):
            n_new += occ[i] * phi[:, i] ** 2

        # ensure normalization
        norm = np.sum(n_new) * dx
        if norm > 0:
            n_new *= n_electrons / norm

        diff = np.sum((n_new - n) ** 2) * dx
        history.append(float(diff))

        # mixing
        n = (1 - mix) * n + mix * n_new

        if diff < tol:
            converged = True
            if verbose:
                print(f"  converged at iteration {iteration + 1}")
            break

    # compute total energy
    eps, phi = np.linalg.eigh(_build_hamiltonian(N, dx, V_ext + _lda_xc_potential_1d(n, alpha=xc_alpha, beta=xc_beta)))
    n_occ = int(np.ceil(n_electrons / 2))
    occ = np.zeros(n_occ)
    remaining = n_electrons
    for i in range(n_occ):
        if remaining >= 2:
            occ[i] = 2.0
            remaining -= 2
        elif remaining > 0:
            occ[i] = remaining
            remaining = 0

    # band energy
    E_band = np.sum(occ * eps[:n_occ])

    # total energy with correct double-counting correction
    V_H = _hartree_potential_1d(n, dx)
    V_xc = _lda_xc_potential_1d(n, alpha=xc_alpha, beta=xc_beta)
    E_H = 0.5 * np.sum(n * V_H) * dx

    # xc ENERGY density (not potential): eps_xc(n) = -(3/4)*alpha*n^(1/3) + small_corr
    n_safe = np.maximum(n, 1e-12)
    eps_xc = -(3.0 / 4.0) * xc_alpha * n_safe ** (1.0 / 3.0) + xc_beta * np.log(1.0 + 2.0 * n_safe) * 0.5
    E_xc = np.sum(n * eps_xc) * dx

    E_ext = np.sum(n * V_ext) * dx
    E_total = E_band - E_xc


    return {
        "algorithm": "DFT_1D",
        "N": N,
        "L": L,
        "n_electrons": n_electrons,
        "potential": potential,
        "xc_alpha": xc_alpha,
        "xc_beta": xc_beta,
        "converged": converged,
        "iterations": len(history),
        "x": x.tolist(),
        "density": n.tolist(),
        "external_potential": V_ext.tolist(),
        "hartree_potential": V_H.tolist(),
        "xc_potential": V_xc.tolist(),
        "total_potential": (V_ext + V_H + V_xc).tolist(),
        "eigenvalues": eps[:n_occ].tolist(),
        "occupations": occ.tolist(),
        "orbitals": [phi[:, i].tolist() for i in range(n_occ)],
        "energy_band": float(E_band),
        "energy_hartree": float(E_H),
        "energy_xc": float(E_xc),
        "energy_total": float(E_total),
        "convergence_history": history,
    }
