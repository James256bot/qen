import numpy as np
from numpy.polynomial.hermite import hermval

HBAR = 1.054571817e-34
M_E  = 9.1093837015e-31
EV   = 1.602176634e-19


def solve_oscillator(omega_eV: float = 1.0, n_max: int = 5, x_range_nm: float = 3.0, n_grid: int = 512):
    """1D quantum harmonic oscillator.

    omega_eV : angular frequency in eV (i.e. ℏω is in eV)
    n_max    : number of levels (0-indexed: 0..n_max-1)
    x_range_nm : half-width of plot window (nm)

    Units: working in eV, nm, and ℏ=1 effective convention.
    """
    if omega_eV <= 0:
        raise ValueError("omega_eV must be positive")
    if not (1 <= n_max <= 30):
        raise ValueError("n_max must be between 1 and 30")

    # x grid in nm
    x_nm = np.linspace(-x_range_nm, x_range_nm, n_grid)

    # Characteristic length scale: sqrt(ℏ / (m ω))
    # In these units, we compute it self-consistently.
    # ℏω is given in eV. Convert to SI for the length scale.
    omega_SI = omega_eV * EV / HBAR
    x0_m = np.sqrt(HBAR / (M_E * omega_SI))
    x0_nm = x0_m * 1e9

    # Dimensionless coordinate ξ = x / x0
    xi = x_nm / x0_nm

    levels = []
    psi_list = []
    for n in range(n_max):
        E_n = omega_eV * (n + 0.5)   # in eV, since ℏω = omega_eV

        # Hermite polynomial coefficients: H_n
        coeffs = [0.0] * n + [1.0]
        Hn = hermval(xi, coeffs)

        # ψ_n(x) ∝ H_n(ξ) exp(-ξ²/2)
        psi_unnorm = Hn * np.exp(-xi**2 / 2.0)

        # Normalize numerically
        dx_nm = x_nm[1] - x_nm[0]
        norm = np.sqrt(np.sum(psi_unnorm**2) * dx_nm)
        psi = psi_unnorm / norm

        levels.append({
            "n": n,
            "energy_eV": float(E_n),
            "degeneracy": 1,
        })
        psi_list.append(psi.tolist())

    # Theoretical expectation values for validation
    # <x> = 0 for all n; <x²> = (n + 1/2) x0²
    return {
        "omega_eV": omega_eV,
        "hbar_omega_eV": omega_eV,
        "x0_nm": float(x0_nm),
        "levels": levels,
        "x_nm": x_nm.tolist(),
        "wavefunctions": psi_list,
        "n_grid": n_grid,
    }
