H_EV_NM2 = 0.3760


def _E(nx, ny, nz, Lx, Ly, Lz):
    return H_EV_NM2 * (nx*nx/(Lx*Lx) + ny*ny/(Ly*Ly) + nz*nz/(Lz*Lz))


def solve_box3d(Lx_nm: float = 2.0, Ly_nm: float = 2.0, Lz_nm: float = 2.0, n_max_each: int = 2):
    if Lx_nm <= 0 or Ly_nm <= 0 or Lz_nm <= 0:
        raise ValueError("all box dimensions must be positive")
    if not (1 <= n_max_each <= 6):
        raise ValueError("n_max_each must be between 1 and 6")

    states = []
    for nx in range(1, n_max_each + 1):
        for ny in range(1, n_max_each + 1):
            for nz in range(1, n_max_each + 1):
                E = _E(nx, ny, nz, Lx_nm, Ly_nm, Lz_nm)
                states.append({"nx": nx, "ny": ny, "nz": nz, "energy_eV": E})

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
            "quanta": [{"nx": s["nx"], "ny": s["ny"], "nz": s["nz"]} for s in g["states"]],
        }
        for g in groups
    ]

    return {
        "Lx_nm": Lx_nm,
        "Ly_nm": Ly_nm,
        "Lz_nm": Lz_nm,
        "n_max_each": n_max_each,
        "levels": levels,
    }
