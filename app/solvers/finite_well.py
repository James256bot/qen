import numpy as np

HBAR = 1.054571817e-34
M_E  = 9.1093837015e-31
EV   = 1.602176634e-19


def solve_finite_well(V0_eV: float = 10.0, well_width_nm: float = 1.0, n_states: int = 5, domain_nm: float = 5.0, n_grid: int = 512):
    if V0_eV <= 0:
        raise ValueError("V0_eV must be positive")
    if well_width_nm <= 0 or well_width_nm >= domain_nm:
        raise ValueError("well_width_nm must be in (0, domain_nm)")

    L = domain_nm * 1e-9
    dx = L / (n_grid + 1)
    x = np.linspace(-L/2 + dx, L/2 - dx, n_grid)
    x_nm = x * 1e9

    t = HBAR**2 / (2 * M_E * dx**2)
    a = well_width_nm * 1e-9
    in_well = np.abs(x) < (a / 2.0)
    V = np.where(in_well, -V0_eV * EV, 0.0)

    diag = 2.0 * t + V
    offd = np.full(n_grid - 1, -1.0 * t)
    H = np.diag(diag) + np.diag(offd, 1) + np.diag(offd, -1)
    evals, evecs = np.linalg.eigh(H)

    evals = evals[:n_states]
    evecs = evecs[:, :n_states]
    for k in range(evecs.shape[1]):
        if evecs[np.argmax(np.abs(evecs[:, k])), k] < 0:
            evecs[:, k] *= -1

    n_bound = int(np.sum(evals < 0))
    threshold = (np.pi**2 * HBAR**2) / (8 * M_E * a**2) / EV

    return {
        "V0_eV": V0_eV,
        "well_width_nm": well_width_nm,
        "domain_nm": domain_nm,
        "n_grid": n_grid,
        "n_bound_states": n_bound,
        "bound_state_threshold_eV": float(threshold),
        "energies_eV": (evals / EV).tolist(),
        "x_nm": x_nm.tolist(),
        "wavefunctions": evecs.T.tolist(),
        "potential_eV": (V / EV).tolist(),
    }
