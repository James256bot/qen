LESSONS = [
    {
        "id": "energy-scales-1-over-L2",
        "title": "Energy scales as 1/L^2",
        "concept": "When you shrink the box, the ground-state energy grows as the inverse square of the width.",
        "system_type": "particle",
        "parameter_sweep": {
            "L_nm": [4.0, 2.0, 1.0],
            "n_states": 3,
        },
        "expected": "E1 quadruples each time L halves: 4.0 nm -> 0.0235 eV, 2.0 nm -> 0.0940 eV, 1.0 nm -> 0.3760 eV.",
    },
    {
        "id": "harmonic-equidistant",
        "title": "Harmonic oscillator has equidistant levels",
        "concept": "Unlike the box, the harmonic oscillator spectrum is equally spaced by hbar*omega.",
        "system_type": "oscillator",
        "parameter_sweep": {
            "omega_eV": [0.5, 1.0, 2.0],
            "n_max": 4,
        },
        "expected": "Levels are always hbar*omega apart: 0.5, 1.5, 2.5... The spacing is uniform.",
    },
    {
        "id": "degeneracy-from-symmetry",
        "title": "Degeneracy comes from symmetry",
        "concept": "A square 2D box has degenerate levels. Squash it and the degeneracy lifts.",
        "system_type": "box2d",
        "parameter_sweep": [
            {"Lx_nm": 2.0, "Ly_nm": 2.0, "n_max_each": 3},
            {"Lx_nm": 3.0, "Ly_nm": 2.0, "n_max_each": 3},
        ],
        "expected": "Square box: (1,2) and (2,1) are degenerate. Rectangular: they split.",
    },
    {
        "id": "bound-states-need-depth",
        "title": "Shallow wells hold no bound states",
        "concept": "A finite square well must be deep enough (or wide enough) to hold at least one bound state.",
        "system_type": "finite_well",
        "parameter_sweep": [
            {"V0_eV": 10.0, "well_width_nm": 1.0, "n_states": 5},
            {"V0_eV": 2.0, "well_width_nm": 0.3, "n_states": 5},
        ],
        "expected": "First case: multiple bound states. Second: zero. Check n_bound_states.",
    },
    {
        "id": "wavepacket-bounces",
        "title": "A wavepacket bounces off the walls",
        "concept": "Launch a Gaussian packet and watch it reflect, spread, and revive.",
        "system_type": "evolve",
        "parameter_sweep": {
            "L_nm": 2.0, "n_states": 20, "x0_frac": 0.3, "sigma_frac": 0.05,
            "t_max_fs": 500, "n_frames": 60,
        },
        "expected": "Play the animation. The packet moves right, reflects, spreads, and at long times shows quantum revivals.",
    },
    {
        "id": "rydberg-series",
        "title": "The Rydberg series converges to zero",
        "concept": "Hydrogen levels crowd toward E=0 as n grows, with the spacing shrinking as 1/n^3.",
        "system_type": "hydrogen",
        "parameter_sweep": {"Z": 1, "n_max": 6},
        "expected": "-13.6, -3.40, -1.51, -0.85, -0.54, -0.38 eV. Each gap is smaller than the last.",
    },
]


def list_lessons():
    return [{"id": l["id"], "title": l["title"], "system_type": l["system_type"], "concept": l["concept"]} for l in LESSONS]


def get_lesson(lesson_id):
    for l in LESSONS:
        if l["id"] == lesson_id:
            return l
    return None
