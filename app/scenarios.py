"""Curated scenarios for QEN Studio.

Each scenario is a multi-step workflow: a sequence of solver calls
and narration notes that tell a complete story.
"""

SCENARIOS = {
    "mott_transition": {
        "title": "Metal to Mott insulator",
        "description": "How electron-electron repulsion turns a metal into an insulator.",
        "steps": [
            {
                "kind": "solve",
                "solver": "vmc",
                "parameters": {"Nx": 8, "Ny": 1, "fill": 0.5, "sweep": 1, "n_samples": 1500},
            },
            {
                "kind": "note",
                "text": "As U/t grows, the ground-state energy rises while the Gutzwiller correlation factor g falls. Electrons avoid double occupancy, kinetic energy is suppressed, and the material becomes an insulator.",
            },
        ],
    },
    "h2_bond": {
        "title": "H₂ bond breaking",
        "description": "Watch the H₂ potential energy surface computed by VQE at three bond lengths.",
        "steps": [
            {
                "kind": "solve",
                "solver": "vqe_h2",
                "parameters": {"R_angstrom": 0.5, "n_layers": 2, "max_iter": 300},
            },
            {
                "kind": "note",
                "text": "At R = 0.5 Å, the hydrogens are compressed — high energy, strong repulsion.",
            },
            {
                "kind": "solve",
                "solver": "vqe_h2",
                "parameters": {"R_angstrom": 0.735, "n_layers": 2, "max_iter": 300},
            },
            {
                "kind": "note",
                "text": "At R = 0.735 Å, H₂ is at equilibrium — the lowest energy, a stable bond.",
            },
            {
                "kind": "solve",
                "solver": "vqe_h2",
                "parameters": {"R_angstrom": 2.0, "n_layers": 2, "max_iter": 300},
            },
            {
                "kind": "note",
                "text": "At R = 2.0 Å, the bond is stretched nearly to breaking. The energy approaches the separated-atom limit.",
            },
        ],
    },
    "tunneling": {
        "title": "Quantum tunneling through a barrier",
        "description": "A wavepacket hits a potential barrier and partially transmits — even when classically forbidden.",
        "steps": [
            {
                "kind": "solve",
                "solver": "lattice2d",
                "parameters": {
                    "Nx": 30, "Ny": 30,
                    "x0_frac": 0.2, "y0_frac": 0.5,
                    "sigma_frac": 0.06,
                    "barrier_V0": 2.0,
                    "n_frames": 60, "tau_max": 20.0,
                },
            },
            {
                "kind": "note",
                "text": "The wavepacket hits the barrier row in the middle of the lattice. Part of it reflects — but part transmits, even though the energy is below the barrier height. That's quantum tunneling.",
            },
        ],
    },
    "bell_entanglement": {
        "title": "Bell state entanglement",
        "description": "Two qubits in maximal entanglement. Each is maximally mixed alone; together they are perfectly correlated.",
        "steps": [
            {
                "kind": "solve",
                "solver": "circuit",
                "parameters": {
                    "n_qubits": 2,
                    "gates": [
                        {"gate": "H", "targets": [0]},
                        {"gate": "CNOT", "targets": [0, 1]},
                    ],
                    "shots": 1000,
                    "entanglement_cut": [0],
                    "bloch_qubits": [0, 1],
                },
            },
            {
                "kind": "note",
                "text": "The Bell state has entanglement entropy S(q0) = 1 bit — the maximum possible for a 2-qubit system. Each qubit is at the origin of the Bloch sphere (r = 0), because information is entirely in the correlation, not in the individual qubits.",
            },
        ],
    },
    "quantum_walk": {
        "title": "2D quantum walk spreading",
        "description": "A quantum particle spreads across a lattice as √t — faster than classical diffusion, slower than ballistic motion.",
        "steps": [
            {
                "kind": "solve",
                "solver": "lattice2d",
                "parameters": {
                    "Nx": 40, "Ny": 40,
                    "x0_frac": 0.5, "y0_frac": 0.5,
                    "sigma_frac": 0.06,
                    "n_frames": 60, "tau_max": 20.0,
                },
            },
            {
                "kind": "note",
                "text": "Unlike a classical random walk, the quantum wavepacket shows interference fringes. The bright rings are regions where the wave's amplitude adds; the dark fringes are where it cancels.",
            },
        ],
    },
}


def list_scenarios():
    return [
        {
            "id": k,
            "title": v["title"],
            "description": v["description"],
            "n_steps": len(v["steps"]),
        }
        for k, v in SCENARIOS.items()
    ]


def get_scenario(scenario_id):
    return SCENARIOS.get(scenario_id)
