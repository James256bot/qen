import numpy as np


# ============================================================
# 2D tight-binding lattice with:
#   - anisotropic hopping (t_x, t_y)
#   - uniform magnetic field via Peierls phase (B)
#   - on-site disorder / Anderson localization (disorder_W)
#   - tunneling barrier (barrier_V0)
#
# H = sum_i V_i |i><i|  -  sum_<ij> [ t_ij e^{i theta_ij} |i><j| + h.c. ]
#
# Time evolution via truncated Taylor series of exp(-i H dtau).
# ============================================================


def _build_onsite(Nx, Ny, disorder_W, seed, barrier_V0):
    """On-site potential: disorder + optional barrier."""
    onsite = np.zeros((Nx, Ny), dtype=np.float64)
    if disorder_W > 0:
        rng = np.random.default_rng(seed)
        onsite += rng.uniform(-disorder_W / 2, disorder_W / 2, size=(Nx, Ny))
    if barrier_V0 > 0:
        mid = Nx // 2
        onsite[mid, :] += barrier_V0
    return onsite


def _apply_H(psi2d, tx, ty, B, onsite):
    """Apply H to psi2d. O(N). Complex in general."""
    Nx, Ny = psi2d.shape
    out = np.zeros_like(psi2d, dtype=np.complex128)

    # on-site
    if onsite is not None:
        out += onsite * psi2d

    # x-direction hops (t_x)
    if B == 0.0:
        # no magnetic field: real hopping
        out[:, :-1] += -tx * psi2d[:, 1:]     # from right
        out[:, 1:]  += -tx * psi2d[:, :-1]    # from left
    else:
        # Peierls phase on x-hops: theta = +B * y / 2
        for j in range(Ny - 1):
            phase = np.exp(1j * B * j / 2)
            out[:, j]     += -tx * np.conj(phase) * psi2d[:, j + 1]
            out[:, j + 1] += -tx * phase * psi2d[:, j]

    # y-direction hops (t_y)
    if B == 0.0:
        out[:-1, :] += -ty * psi2d[1:, :]
        out[1:, :]  += -ty * psi2d[:-1, :]
    else:
        # Peierls phase on y-hops: theta = -B * x / 2
        for i in range(Nx - 1):
            phase = np.exp(-1j * B * i / 2)
            out[i, :]     += -ty * np.conj(phase) * psi2d[i + 1, :]
            out[i + 1, :] += -ty * phase * psi2d[i, :]

    return out


def _expm_apply(psi2d, tx, ty, B, onsite, dtau, order=20):
    """Approximate exp(-i H dtau) psi via truncated Taylor series."""
    psi2d = psi2d.astype(np.complex128)
    result = psi2d.copy()
    term = psi2d.copy()
    coeff = -1j * dtau
    for k in range(1, order + 1):
        term = _apply_H(term, tx, ty, B, onsite) * (coeff / k)
        result = result + term
        if np.max(np.abs(term)) < 1e-15:
            break
    return result


def solve_lattice2d(
    Nx=30, Ny=30,
    x0_frac=0.25, y0_frac=0.5,
    sigma_frac=0.08,
    t_x=1.0, t_y=1.0,
    B=0.0,
    disorder_W=0.0,
    barrier_V0=0.0,
    seed=42,
    t_max_fs=200.0,
    n_frames=40,
    tau_max=15.0,
    order=20,
):
    """2D tight-binding wavepacket evolution with optional anisotropy,
    magnetic field, disorder, and barrier.

    Parameters
    ----------
    Nx, Ny       : lattice dimensions (10-200)
    x0_frac, y0_frac : initial packet center (fraction of lattice)
    sigma_frac   : initial packet width (fraction of Nx)
    t_x, t_y     : hopping amplitudes in x and y (default 1.0)
    B            : magnetic flux per plaquette (Peierls phase). 0 = off.
    disorder_W   : on-site disorder strength. 0 = clean.
    barrier_V0   : potential on the middle row. 0 = no barrier.
    seed         : random seed for disorder reproducibility.
    tau_max      : total dimensionless time.
    n_frames     : number of frames.
    order        : Taylor series truncation order.
    """
    if not (10 <= Nx <= 200 and 10 <= Ny <= 200):
        raise ValueError("Nx, Ny must be between 10 and 200")
    if not (5 <= n_frames <= 400):
        raise ValueError("n_frames must be between 5 and 400")
    if t_x <= 0 or t_y <= 0:
        raise ValueError("t_x, t_y must be positive")
    if disorder_W < 0 or barrier_V0 < 0:
        raise ValueError("disorder_W and barrier_V0 must be non-negative")

    # on-site potential
    onsite = _build_onsite(Nx, Ny, disorder_W, seed, barrier_V0)

    # initial Gaussian wavepacket
    x0 = x0_frac * Nx
    y0 = y0_frac * Ny
    sigma = sigma_frac * Nx
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    psi = (psi / np.linalg.norm(psi)).astype(np.complex128)

    tau_array = np.linspace(0, tau_max, n_frames)
    dtau = tau_array[1] - tau_array[0]

    frames = []
    for k, tau in enumerate(tau_array):
        prob = np.abs(psi)**2
        frames.append({
            "tau": float(tau),
            "prob": prob.real.tolist(),
        })
        if k < len(tau_array) - 1:
            psi = _expm_apply(psi, t_x, t_y, B, onsite, dtau, order=order)

    return {
        "Nx": Nx, "Ny": Ny,
        "t_x": t_x, "t_y": t_y,
        "B": B,
        "disorder_W": disorder_W,
        "barrier_V0": barrier_V0,
        "seed": seed,
        "x0_frac": x0_frac, "y0_frac": y0_frac,
        "sigma_frac": sigma_frac,
        "t_max_fs": t_max_fs,
        "tau_max": tau_max,
        "n_frames": n_frames,
        "onsite": onsite.tolist(),
        "frames": frames,
    }
