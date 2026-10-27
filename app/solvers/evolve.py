import numpy as np

M_E  = 9.1093837015e-31
EV   = 1.602176634e-19


def solve_time_evolution(L_nm: float = 2.0, n_states: int = 20, x0_frac: float = 0.3,
                          sigma_frac: float = 0.03, t_max_fs: float = 200.0,
                          n_frames: int = 20, n_grid: int = 256):
    if not (0.1 <= L_nm <= 100.0):
        raise ValueError("L_nm must be between 0.1 and 100")
    if not (1 <= n_states <= 64):
        raise ValueError("n_states must be between 1 and 64")
    if not (1 <= n_frames <= 200):
        raise ValueError("n_frames must be between 1 and 200")

    L = L_nm * 1e-9
    dx = L / (n_grid + 1)
    x = np.arange(1, n_grid + 1) * dx
    x_nm = x * 1e9

    psi = np.zeros((n_states, n_grid))
    for n in range(1, n_states + 1):
        psi[n-1] = np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

    h = 6.62607015e-34
    E_n = (np.arange(1, n_states + 1)**2 * h**2) / (8 * M_E * L**2) / EV

    x0 = x0_frac * L
    sigma = sigma_frac * L
    psi0 = np.exp(-((x - x0)**2) / (2 * sigma**2))
    psi0 = psi0 / np.sqrt(np.sum(psi0**2) * dx)

    c_n = np.array([np.sum(psi[n] * psi0) * dx for n in range(n_states)])
    total_prob = float(np.sum(c_n**2))

    hbar_eV_s = 6.582119569e-16
    t_array_s = np.linspace(0, t_max_fs * 1e-15, n_frames)

    frames = []
    for t in t_array_s:
        phase = np.exp(-1j * E_n * t / hbar_eV_s)
        cn_t = c_n * phase
        psi_t = np.sum(cn_t[:, None] * psi, axis=0)
        prob = np.abs(psi_t)**2
        frames.append({
            "t_fs": float(t * 1e15),
            "prob_density": prob.tolist(),
        })

    return {
        "L_nm": L_nm,
        "n_states": n_states,
        "x0_frac": x0_frac,
        "sigma_frac": sigma_frac,
        "t_max_fs": t_max_fs,
        "n_frames": n_frames,
        "n_grid": n_grid,
        "total_prob_from_expansion": total_prob,
        "energies_eV": E_n.tolist(),
        "x_nm": x_nm.tolist(),
        "frames": frames,
    }
