"""Rule-based intent parser for QEN Studio.

Maps natural-language prompts to solver + parameters. No LLM needed.
"""

INTENTS = [
    # lattice2d with barrier = tunneling
    {
        "keywords": ["tunnel", "tunneling", "tunnelling", "barrier", "penetrate", "penetration"],
        "system_type": "lattice2d",
        "parameters": {
            "Nx": 30, "Ny": 30,
            "x0_frac": 0.2, "y0_frac": 0.5,
            "sigma_frac": 0.06,
            "barrier_V0": 2.0,
            "n_frames": 60, "tau_max": 20.0,
        },
        "narration": "A wavepacket hits a potential barrier. Part reflects, part transmits — even though the particle's energy is lower than the barrier height. That's quantum tunneling.",
    },
    # lattice2d with magnetic field
    {
        "keywords": ["magnetic", "cyclotron", "landau", "hall", "field"],
        "system_type": "lattice2d",
        "parameters": {
            "Nx": 30, "Ny": 30,
            "x0_frac": 0.5, "y0_frac": 0.5,
            "sigma_frac": 0.05,
            "B": 0.3,
            "n_frames": 60, "tau_max": 20.0,
        },
        "narration": "A charged particle in a magnetic field follows a circular trajectory. The radius is set by the field strength — this is the cyclotron motion at the heart of the quantum Hall effect.",
    },
    # lattice2d with disorder
    {
        "keywords": ["anderson", "disorder", "localization", "localized", "disordered"],
        "system_type": "lattice2d",
        "parameters": {
            "Nx": 30, "Ny": 30,
            "x0_frac": 0.5, "y0_frac": 0.5,
            "sigma_frac": 0.06,
            "disorder_W": 3.0,
            "n_frames": 60, "tau_max": 20.0,
        },
        "narration": "A particle in a disordered lattice starts to spread but then stops. That's Anderson localization — disorder can turn a metal into an insulator purely by trapping the quantum wave.",
    },
    # lattice2d plain spreading
    {
        "keywords": ["lattice", "spread", "spreading", "wavepacket", "wave packet", "diffus"],
        "system_type": "lattice2d",
        "parameters": {
            "Nx": 40, "Ny": 40,
            "x0_frac": 0.5, "y0_frac": 0.5,
            "sigma_frac": 0.06,
            "n_frames": 60, "tau_max": 20.0,
        },
        "narration": "A localized quantum particle spreads across a lattice. Unlike a classical ball, it spreads as √t — that's quantum diffusion, driven by the uncertainty principle.",
    },
    # Mott transition
    {
        "keywords": ["mott", "insulator", "hubbard", "correlat"],
        "system_type": "vmc",
        "parameters": {"Nx": 8, "Ny": 1, "fill": 0.5, "sweep": 1, "n_samples": 3000},
        "narration": "The Mott transition: as interaction strength U/t grows, the ground-state energy rises while the correlation factor falls. Electrons avoid each other and the material becomes an insulator.",
    },
    # DFT
    {
        "keywords": ["density functional", "dft", "kohn", "sham", "electron density"],
        "system_type": "dft",
        "parameters": {"N": 150, "L": 15, "n_electrons": 2, "potential": "harmonic"},
        "narration": "Density functional theory: the electron density and the effective potential are computed self-consistently. The blue area is n(x), the green line is V(x), the colored curves are the Kohn-Sham orbitals.",
    },
    # Battery
    {
        "keywords": ["battery", "lithium", "voltage", "cell", "electrode", "cathode", "anode"],
        "system_type": "battery",
        "parameters": {"cathode": "LiF", "anode": "LiH"},
        "narration": "A lithium-ion cell voltage, computed from the difference in Li binding energy between cathode and anode. This is the same physical quantity that determines real battery performance.",
    },
    # H2 molecule
    {
        "keywords": ["h2", "h₂", "hydrogen molecule", "molecule", "bond"],
        "system_type": "vqe_h2",
        "parameters": {"R_angstrom": 0.735, "n_layers": 2, "max_iter": 400},
        "narration": "Variational Quantum Eigensolver for the H₂ molecule. The blue curve is the optimizer's convergence, the red dashed line is the exact FCI energy. VQE reaches chemical accuracy.",
    },
    # QPE
    {
        "keywords": ["phase estimation", "qpe", "phase", "eigenvalue", "shor", "factoring"],
        "system_type": "qpe",
        "parameters": {"phi": 0.25, "n_ancilla": 5},
        "narration": "Quantum Phase Estimation: the ancilla register encodes the phase of a unitary operator. Here φ = 1/4 is recovered exactly, with probability 1.",
    },
    # Bell state / entanglement
    {
        "keywords": ["bell", "entangle", "entanglement", "ghz", "maximal"],
        "system_type": "circuit",
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
        "narration": "The Bell state: two qubits in maximal entanglement. Each qubit alone is maximally mixed (Bloch vector at the origin), yet the pair has perfect correlation.",
    },
    # Grover
    {
        "keywords": ["grover", "search", "amplitude amplification"],
        "system_type": "circuit",
        "parameters": {
            "n_qubits": 2,
            "gates": [
                {"gate": "H", "targets": [0]}, {"gate": "H", "targets": [1]},
                {"gate": "CZ", "targets": [0, 1]},
                {"gate": "H", "targets": [0]}, {"gate": "H", "targets": [1]},
                {"gate": "Z", "targets": [0]}, {"gate": "Z", "targets": [1]},
                {"gate": "CZ", "targets": [0, 1]},
                {"gate": "H", "targets": [0]}, {"gate": "H", "targets": [1]},
            ],
            "shots": 2000,
        },
        "narration": "Grover's algorithm: after one iteration, the target state |11⟩ has probability 1. This is the quadratic speedup that makes quantum search faster than classical.",
    },
    # Hydrogen atom
    {
        "keywords": ["hydrogen atom", "rydberg", "atomic", "spectrum"],
        "system_type": "hydrogen",
        "parameters": {"Z": 1, "n_max": 6},
        "narration": "The hydrogen spectrum: energy levels crowd toward zero as n grows, following the Rydberg formula E = -13.6/n² eV. Every gap is smaller than the last.",
    },
    # Harmonic oscillator
    {
        "keywords": ["harmonic", "oscillator", "equidistant", "ladder"],
        "system_type": "oscillator",
        "parameters": {"omega_eV": 1.0, "n_max": 6},
        "narration": "The quantum harmonic oscillator: energy levels are equally spaced by ℏω. This is the spectrum of a vibrating molecule, a trapped ion, or a photon in a cavity.",
    },
    # Particle in a box
    {
        "keywords": ["particle in a box", "box", "well", "infinite well"],
        "system_type": "particle",
        "parameters": {"L_nm": 2.0, "n_states": 5},
        "narration": "Particle in a box: energies scale as n² and as 1/L². Shrink the box and the energy levels rise.",
    },
]

DEFAULT_INTENT = {
    "system_type": "particle",
    "parameters": {"L_nm": 2.0, "n_states": 5},
    "narration": "Particle in a box. Try one of the preset scenarios below, or type a specific question.",
}


def parse_intent(prompt):
    """Match a prompt to an intent by keyword count."""
    lower = prompt.lower()
    best = None
    best_score = 0
    for intent in INTENTS:
        score = sum(1 for kw in intent["keywords"] if kw in lower)
        if score > best_score:
            best_score = score
            best = intent
    if best is None:
        return {**DEFAULT_INTENT, "matched": False, "score": 0}
    return {**best, "matched": True, "score": best_score}
