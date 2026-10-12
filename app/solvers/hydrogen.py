import numpy as np

RYDBERG_EV = 13.605693122994


def solve_hydrogen(Z: int = 1, n_max: int = 4):
    levels = []
    for n in range(1, n_max + 1):
        E_eV = -RYDBERG_EV * Z * Z / (n * n)
        levels.append({
            "n": n,
            "energy_eV": E_eV,
            "energy_J": E_eV * 1.602176634e-19,
            "degeneracy": 2 * n * n,
        })
    return {"Z": Z, "levels": levels}
