import numpy as np

HBAR = 1.054571817e-34
M_E  = 9.1093837015e-31
EV   = 1.602176634e-19


def solve_particle_box(L_nm: float, n_states: int = 5, n_grid: int = 256, return_wavefunctions: bool = True):
    L = L_nm * 1e-9
    dx = L / (n_grid + 1)

    t = HBAR**2 / (2 * M_E * dx**2)
    diag = np.full(n_grid, 2.0 * t)
    offd = np.full(n_grid - 1, -1.0 * t)

    H = np.diag(diag) + np.diag(offd, 1) + np.diag(offd, -1)
    evals, evecs = np.linalg.eigh(H)
    evals = evals[:n_states]
    evecs = evecs[:, :n_states]

    for k in range(evecs.shape[1]):
        if evecs[np.argmax(np.abs(evecs[:, k])), k] < 0:
            evecs[:, k] *= -1

    x = np.arange(1, n_grid + 1) * dx
    out = {
        "energies_eV": (evals / EV).tolist(),
        "n_grid": n_grid,
    }
    if return_wavefunctions:
        out["x_nm"] = (x * 1e9).tolist()
        out["wavefunctions"] = evecs.T.tolist()
    return out
