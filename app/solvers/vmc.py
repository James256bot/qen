import numpy as np
from app.solvers.variational import nelder_mead


# ============================================================
# Variational Monte Carlo for the 1D and 2D Hubbard model
#
# NOTE ON ACCURACY
# ----------------
# VMC gives a variational UPPER BOUND on the true ground-state
# energy. The quality of the bound depends on the ansatz. The
# single-parameter Gutzwiller ansatz used here is exact for U=0
# on a 2-site lattice, but becomes increasingly approximate for
# larger systems and larger U.
#
# For research-grade accuracy, use a Slater-Jastrow ansatz with
# plane-wave or Gaussian orbitals and a long-range Jastrow factor.
# That's a significant extension, not implemented here.
#
# The purpose of this module is to demonstrate the METHOD:
#   1. Choose a variational wavefunction psi(alpha)
#   2. Sample configurations via Metropolis
#   3. Estimate <H> from local energies
#   4. Optimize alpha
#
# The same machinery scales to hundreds of electrons in real
# research codes, where the ansatz is much richer.
#
# H = -t Σ_<ij>σ (c†_iσ c_jσ + h.c.)
#     + U Σ_i n_i↑ n_i↓
#     + V Σ_i n_i
#
# Variational ansatz: Jastrow + Slater (Gutzwiller-like).
#
#   |ψ(x)⟩ = exp(-Σ_ij J_ij n_i n_j) · |Slater(x)⟩
#
# In practice we work in the real-space occupation basis:
# a configuration is a bit-string of length 2N (spin-up and spin-down).
# ============================================================


def _build_lattice(Nx, Ny):
    """1D: Ny=1. 2D: rectangular grid. Returns list of (site, neighbor) pairs."""
    sites = [(i, j) for i in range(Nx) for j in range(Ny)]
    index_of = {s: k for k, s in enumerate(sites)}
    bonds = []
    for (i, j) in sites:
        for (di, dj) in ((1, 0), (0, 1)):
            ni, nj = i + di, j + dj
            if ni < Nx and nj < Ny:
                bonds.append((index_of[(i, j)], index_of[(ni, nj)]))
    return sites, bonds


def _onsite_energy(config, N, U, V):
    """On-site interaction + potential."""
    E = 0.0
    for i in range(N):
        up = (config >> i) & 1
        dn = (config >> (i + N)) & 1
        n = up + dn
        E += V * n
        E += U * up * dn
    return E


def _hopping_term(config, bonds, N):
    """Kinetic energy via hopping on each bond (return <config'|H|config>)."""
    # For a single configuration, the local hopping contribution is:
    #   -t * Σ_<ij>σ ( sign * [config' with i<->j swapped] )
    # We compute it as: for each bond and spin, if occupations differ,
    # the hopping connects this config to one with occupations swapped.
    return 0.0  # placeholder, filled in by _local_energy


def _local_energy(config, bonds, N, U, V, t, psi_fn):
    """Local energy E_loc(config) = Σ_c' H_{c,c'} ψ(c') / ψ(c)."""
    psi_c = psi_fn(config)
    if psi_c == 0:
        return 0.0
    # diagonal
    E = _onsite_energy(config, N, U, V)
    # off-diagonal: hopping
    for (i, j) in bonds:
        for spin_offset in (0, N):
            bi = (config >> (i + spin_offset)) & 1
            bj = (config >> (j + spin_offset)) & 1
            if bi != bj:
                # hopping connects config to config'
                new_config = config ^ (1 << (i + spin_offset)) ^ (1 << (j + spin_offset))
                psi_new = psi_fn(new_config)
                if psi_new != 0:
                    E += -t * psi_new / psi_c
    return E


# ============================================================
# Variational wavefunction: Jastrow + Gutzwiller
#
# We use a simple form:
#   ψ(c) = g^{D(c)} · exp( Σ_i α_i n_i )
# where D(c) is the number of doubly-occupied sites, g is a
# Gutzwiller factor (0 <= g <= 1), and α_i are site-dependent
# potentials (usually 0).
#
# For this demo we use a single Gutzwiller parameter g:
#   ψ(c) = g^{D(c)}
# ============================================================


def _num_double_occupied(config, N):
    D = 0
    for i in range(N):
        up = (config >> i) & 1
        dn = (config >> (i + N)) & 1
        if up and dn:
            D += 1
    return D


def _make_psi_fn(g, N):
    """Return a function config -> ψ(config)."""
    def psi_fn(config):
        D = _num_double_occupied(config, N)
        return g ** D
    return psi_fn


def _metropolis_sample(psi_fn, initial_config, n_samples, N, bonds, seed=42):
    """Metropolis sampling — propose electron hops along lattice bonds.

    The Hubbard model conserves N_up and N_dn separately. A valid
    proposal moves one electron from an occupied site to an empty
    neighbor, keeping the spin species fixed.
    """
    rng = np.random.default_rng(seed)
    config = initial_config
    psi_c = psi_fn(config)
    samples = []
    for _ in range(n_samples):
        # propose: pick a bond and a spin
        bond_idx = rng.integers(0, len(bonds))
        i, j = bonds[bond_idx]
        spin_offset = 0 if rng.random() < 0.5 else N
        bi = (config >> (i + spin_offset)) & 1
        bj = (config >> (j + spin_offset)) & 1
        if bi == bj:
            # no hop possible (both occupied or both empty); skip
            samples.append(config)
            continue
        new_config = config ^ (1 << (i + spin_offset)) ^ (1 << (j + spin_offset))
        psi_new = psi_fn(new_config)
        ratio = (psi_new / psi_c) ** 2 if psi_c != 0 else 1.0
        if rng.random() < ratio:
            config = new_config
            psi_c = psi_new
        samples.append(config)
    return samples


def vmc_hubbard(
    Nx=4, Ny=1,
    N_up=2, N_dn=2,
    U=2.0, t=1.0, V=0.0,
    g_init=0.5,
    n_samples=15000,
    max_iter=100,
    seed=42,
):
    """VMC for the Hubbard model with a Gutzwiller variational parameter g.

    Small lattices only — this is a demonstration of the *method*.
    """
    N = Nx * Ny
    total_electrons = N_up + N_dn
    if total_electrons > 2 * N:
        raise ValueError("too many electrons for the lattice")

    sites, bonds = _build_lattice(Nx, Ny)

    # initial configuration: fill first N_up sites as up, next N_dn as down
    config0 = 0
    for i in range(N_up):
        config0 |= (1 << i)
    for i in range(N_dn):
        config0 |= (1 << (N_up + i)) if False else (1 << (N + i))

    def cost(g_array):
        g = float(np.clip(g_array[0], 0.0, 1.0))
        psi_fn = _make_psi_fn(g, N)
        samples = _metropolis_sample(psi_fn, config0, n_samples, N, bonds, seed=seed)
        # discard the first half as burn-in
        samples = samples[n_samples // 2:]
        E_total = 0.0
        count = 0
        for c in samples:
            if psi_fn(c) == 0:
                continue
            E_total += _local_energy(c, bonds, N, U, V, t, psi_fn)
            count += 1
        if count == 0:
            return 1e6
        return E_total / count

    x_opt, e_opt, iterations = nelder_mead(
        cost, np.array([g_init]), max_iter=max_iter,
    )

    g_opt = float(np.clip(x_opt[0], 0.0, 1.0))
    psi_opt = _make_psi_fn(g_opt, N)
    samples = _metropolis_sample(psi_opt, config0, n_samples, N, bonds, seed=seed)
    samples = samples[n_samples // 2:]
    energies = []
    for c in samples:
        if psi_opt(c) == 0:
            continue
        energies.append(_local_energy(c, bonds, N, U, V, t, psi_opt))
    energies = np.array(energies)
    # block-averaged error estimate (reduces autocorrelation bias)
    n_blocks = 10
    if len(energies) >= n_blocks * 2:
        block_size = len(energies) // n_blocks
        block_means = np.array([energies[i*block_size:(i+1)*block_size].mean() for i in range(n_blocks)])
        stderr = block_means.std() / np.sqrt(n_blocks)
    else:
        stderr = energies.std() / np.sqrt(len(energies)) if len(energies) > 1 else 0.0

    return {
        "algorithm": "VMC_Hubbard",
        "Nx": Nx, "Ny": Ny, "N": N,
        "N_up": N_up, "N_dn": N_dn,
        "U": U, "t": t, "V": V,
        "optimal_g": g_opt,
        "energy": float(energies.mean()) if len(energies) else 0.0,
        "energy_stderr": float(stderr),
        "iterations": iterations,
        "n_samples": n_samples,
        "n_energy_samples": len(energies),
        "energy_per_site": float(energies.mean() / N) if len(energies) else 0.0,
    }


# ============================================================
# Convenience: exact diagonalization for cross-check
# (only feasible for tiny lattices)
# ============================================================


def exact_hubbard_ground_energy(Nx=2, Ny=1, N_up=1, N_dn=1, U=2.0, t=1.0, V=0.0):
    """Exact ground-state energy of the Hubbard model for tiny lattices."""
    from itertools import combinations
    N = Nx * Ny

    # build the full basis of occupation states
    def configs_with_electrons(N, n):
        for occupied in combinations(range(N), n):
            c = 0
            for i in occupied:
                c |= (1 << i)
            yield c

    up_configs = list(configs_with_electrons(N, N_up))
    dn_configs = list(configs_with_electrons(N, N_dn))
    basis = []
    index = {}
    for cu in up_configs:
        for cd in dn_configs:
            c = cu | (cd << N)
            index[c] = len(basis)
            basis.append(c)

    H = np.zeros((len(basis), len(basis)))
    _, bonds = _build_lattice(Nx, Ny)
    for k, c in enumerate(basis):
        H[k, k] = _onsite_energy(c, N, U, V)
        for (i, j) in bonds:
            for spin in (0, N):
                bi = (c >> (i + spin)) & 1
                bj = (c >> (j + spin)) & 1
                if bi != bj:
                    new_c = c ^ (1 << (i + spin)) ^ (1 << (j + spin))
                    if new_c in index:
                        H[index[new_c], k] += -t

    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def exact_1d_tight_binding_energy(N_up, N_dn, t=1.0):
    """Exact ground-state energy of the 1D tight-binding model (U=0).

    For a half-filled chain of N sites, E/site = -4t/π.
    """
    N = N_up + N_dn
    return -4 * t * N / np.pi
