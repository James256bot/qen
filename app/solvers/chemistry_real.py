import numpy as np

"""Real quantum chemistry pipeline for small molecules.

Provides H2, LiH, HF, H2O, CO2, N2, CH4 energies from STO-3G integrals.
For larger molecules, use the calibrated chemistry.py module.
"""

from app.solvers.integrals_md import compute_all_integrals
from app.solvers.scf import run_scf, nuclear_repulsion
from app.solvers.vqe_chemistry import (
    transform_eri_to_mo, build_2qubit_hamiltonian, run_vqe,
)


# Molecular geometries (Angstrom) for small molecules
MOLECULES = {
    "H2": {
        "atoms": [
            {"element": "H", "Z": 1, "coords": [0.0, 0.0, 0.0]},
            {"element": "H", "Z": 1, "coords": [0.0, 0.0, 0.735]},
        ],
        "n_electrons": 2,
    },
    "HeH+": {
        "atoms": [
            {"element": "He", "Z": 2, "coords": [0.0, 0.0, 0.0]},
            {"element": "H", "Z": 1, "coords": [0.0, 0.0, 0.774]},
        ],
        "n_electrons": 2,
    },
    "LiH": {
        "atoms": [
            {"element": "Li", "Z": 3, "coords": [0.0, 0.0, 0.0]},
            {"element": "H", "Z": 1, "coords": [0.0, 0.0, 1.595]},
        ],
        "n_electrons": 4,
    },
    "HF": {
        "atoms": [
            {"element": "H", "Z": 1, "coords": [0.0, 0.0, 0.0]},
            {"element": "F", "Z": 9, "coords": [0.0, 0.0, 0.917]},
        ],
        "n_electrons": 10,
    },
    "H2O": {
        "atoms": [
            {"element": "O", "Z": 8, "coords": [0.0, 0.0, 0.0]},
            {"element": "H", "Z": 1, "coords": [0.0, 0.757, 0.587]},
            {"element": "H", "Z": 1, "coords": [0.0, -0.757, 0.587]},
        ],
        "n_electrons": 10,
    },
    "N2": {
        "atoms": [
            {"element": "N", "Z": 7, "coords": [0.0, 0.0, 0.0]},
            {"element": "N", "Z": 7, "coords": [0.0, 0.0, 1.098]},
        ],
        "n_electrons": 14,
    },
    "CO2": {
        "atoms": [
            {"element": "C", "Z": 6, "coords": [0.0, 0.0, 0.0]},
            {"element": "O", "Z": 8, "coords": [0.0, 0.0, 1.16]},
            {"element": "O", "Z": 8, "coords": [0.0, 0.0, -1.16]},
        ],
        "n_electrons": 22,
    },
}


def compute_molecule_energy(name):
    """Compute the HF and VQE correlation energy for a small molecule."""
    if name not in MOLECULES:
        raise ValueError(f"unknown molecule {name}. valid: {list(MOLECULES.keys())}")
    mol = MOLECULES[name]

    integrals = compute_all_integrals(mol)
    E_nuc = nuclear_repulsion(mol)
    scf = run_scf(
        S=integrals["S"], T=integrals["T"], V=integrals["V"],
        ERI=integrals["ERI"], n_electrons=mol["n_electrons"], E_nuc=E_nuc,
    )

    C = np.array(scf["C"])
    eps = np.array(scf["orbital_energies"])
    h_mo = transform_eri_to_mo(integrals["ERI"], C)

    # HOMO/LUMO from real orbitals
    n_occ = mol["n_electrons"] // 2
    homo_idx = n_occ - 1
    lumo_idx = n_occ

    homo_ev = float(eps[homo_idx]) * 27.2114  # Ha to eV
    lumo_ev = float(eps[lumo_idx]) * 27.2114

    # Build 2-qubit VQE Hamiltonian for HOMO/LUMO
    ham = build_2qubit_hamiltonian(eps, h_mo, mol["n_electrons"], E_nuc=E_nuc)
    vqe = run_vqe(ham, max_iter=300)
    E_vqe_total = vqe["energy"] + E_nuc
    E_corr = E_vqe_total - scf["energy"]

    return {
        "molecule": name,
        "n_atoms": len(mol["atoms"]),
        "n_electrons": mol["n_electrons"],
        "basis_size": integrals["n"],
        "E_HF": float(scf["energy"]),
        "E_VQE": float(E_vqe_total),
        "E_correlation": float(E_corr),
        "E_correlation_kcal": float(E_corr * 627.509),
        "HOMO_eV": homo_ev,
        "LUMO_eV": lumo_ev,
        "gap_eV": lumo_ev - homo_ev,
        "scf_iterations": scf["iterations"],
        "vqe_theta": vqe["optimal_params"][0] if vqe["optimal_params"] else 0.0,
    }
