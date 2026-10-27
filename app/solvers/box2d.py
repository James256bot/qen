H_EV_NM2 = 0.3760


def _E_nm2(nx, ny, Lx, Ly):
    return H_EV_NM2 * (nx*nx/(Lx*Lx) + ny*ny/(Ly*Ly))


def solve_box2d(Lx_nm: float = 2.0, Ly_nm: float = 2.0, n_max_each: int = 3):
    if Lx_nm <= 0 or Ly_nm <= 0:
        raise ValueError("Lx_nm and Ly_nm must be positive")
    if not (1 <= n_max_each <= 10):
        raise ValueError("n_max_each must be between 1 and 10")

    states = []
    for nx in range(1, n_max_each + 1):
        for ny in range(1, n_max_each + 1):
            E = _E_nm2(nx, ny, Lx_nm, Ly_nm)
            states.append({"nx": nx, "ny": ny, "energy_eV": E})

    states.sort(key=lambda s: s["energy_eV"])

    groups = []
    for s in states:
        if groups and abs(groups[-1]["energy_eV"] - s["energy_eV"]) < 1e-9:
            groups[-1]["states"].append(s)
        else:
            groups.append({"energy_eV": s["energy_eV"], "states": [s]})

    levels = [
        {
            "energy_eV": g["energy_eV"],
            "degeneracy": len(g["states"]),
            "quanta": [{"nx": s["nx"], "ny": s["ny"]} for s in g["states"]],
        }
        for g in groups
    ]

    return {
        "Lx_nm": Lx_nm,
        "Ly_nm": Ly_nm,
        "n_max_each": n_max_each,
        "levels": levels,
    }
