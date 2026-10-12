import numpy as np

# N2 experimental parameters
DE_EV    = 9.76        # dissociation energy (eV)
RE_NM    = 0.1098      # equilibrium bond length (nm)
A_PER_NM = 27.3        # Morse width parameter (1/nm)

# Electronic offset: sum of N atomic binding energies
# Approximate; sets the "total_energy_eV" baseline
E_ELECTRONIC_OFFSET_EV = -29.068


def solve_n2(R_min_nm: float = 0.06588, R_max_nm: float = 0.3, n_points: int = 200):
    R = np.unique(np.concatenate([np.linspace(R_min_nm, R_max_nm, n_points), [RE_NM]]))
    # Morse: V(R) = De * (1 - exp(-a(R-Re)))^2 - De
    x = A_PER_NM * (R - RE_NM)
    bond = DE_EV * (1.0 - np.exp(-x))**2 - DE_EV
    total = bond + E_ELECTRONIC_OFFSET_EV
    curve = [
        {
            "R_nm": float(R[i]),
            "bond_energy_eV": float(bond[i]),
            "total_energy_eV": float(total[i]),
        }
        for i in range(n_points)
    ]
    return {
        "molecule_id": "n2",
        "name": "N2",
        "De_eV": DE_EV,
        "Re_nm": RE_NM,
        "a_per_nm": A_PER_NM,
        "curve": curve,
    }
