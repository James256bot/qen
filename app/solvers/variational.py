import numpy as np
from app.solvers.qsim import (
    zero_state, apply_1q, apply_cnot, execute_circuit,
    Rx, Ry, Rz, probabilities,
)


# ============================================================
# Parameterized circuits and expectation values
# ============================================================


def _ry_ansatz_state(thetas, n_qubits, n_layers=1):
    """Hardware-efficient ansatz: Rz-Ry on each qubit, then a CNOT ladder.

    n_params = 2 * n_qubits * n_layers.
    """
    psi = zero_state(n_qubits)
    idx = 0
    for _ in range(n_layers):
        # Rz then Ry on every qubit
        for q in range(n_qubits):
            psi = apply_1q(psi, Rz(float(thetas[idx])), q, n_qubits)
            idx += 1
            psi = apply_1q(psi, Ry(float(thetas[idx])), q, n_qubits)
            idx += 1
        # entangling ladder
        for q in range(n_qubits - 1):
            psi = apply_cnot(psi, q, q + 1, n_qubits)
    return psi


def expectation_pauli_z(psi, n_qubits):
    """<Z_0 Z_1 ... Z_{n-1}> for the given state."""
    probs = probabilities(psi)
    # parity of each basis state: +1 for even number of 1s, -1 for odd
    n_states = len(probs)
    parity = np.zeros(n_states)
    for i in range(n_states):
        parity[i] = 1 - 2 * (bin(i).count("1") % 2)
    return float(np.sum(probs * parity))


def hamiltonian_ising_1d(psi, n_qubits, J=1.0, h=0.0):
    """<H> for H = J sum Z_i Z_{i+1} + h sum X_i.

    Computed by direct measurement of correlators on the state vector.
    """
    probs = probabilities(psi)
    # diagonal part: J * sum_i <Z_i Z_{i+1}>
    total = 0.0
    for i in range(n_qubits - 1):
        corr = 0.0
        for s, p in enumerate(probs):
            # qubit i and i+1 — MSB-first indexing
            bit_i = (s >> (n_qubits - 1 - i)) & 1
            bit_j = (s >> (n_qubits - 1 - (i + 1))) & 1
            z_i = 1 - 2 * bit_i
            z_j = 1 - 2 * bit_j
            corr += z_i * z_j * p
        total += J * corr
    # transverse field: h * sum_i <X_i> — requires rotating basis
    if h != 0.0:
        for i in range(n_qubits):
            # apply H on qubit i, then measure Z_i
            from app.solvers.qsim import GATES_1Q
            psi_rot = apply_1q(psi, GATES_1Q["H"], i, n_qubits)
            probs_rot = probabilities(psi_rot)
            exp_x = 0.0
            for s, p in enumerate(probs_rot):
                bit_i = (s >> (n_qubits - 1 - i)) & 1
                exp_x += (1 - 2 * bit_i) * p
            total += h * exp_x
    return float(total)


def _apply_ry_layer(psi, thetas, offset, n_qubits):
    for q in range(n_qubits):
        psi = apply_1q(psi, Ry(float(thetas[offset + q])), q, n_qubits)
    return psi, offset + n_qubits


def _apply_rz_layer(psi, thetas, offset, n_qubits):
    for q in range(n_qubits):
        psi = apply_1q(psi, Rz(float(thetas[offset + q])), q, n_qubits)
    return psi, offset + n_qubits


# ============================================================
# QAOA for MaxCut on a small graph
# ============================================================


def _qaoa_state(gammas, betas, edges, n_qubits):
    """QAOA ansatz for MaxCut.

    Initial state: |+>^n.
    Each layer: cost unitary exp(-i gamma H_C) then mixer exp(-i beta sum X_i).
    H_C = sum_{(i,j)} (1 - Z_i Z_j)/2.
    """
    from app.solvers.qsim import GATES_1Q
    psi = zero_state(n_qubits)
    for q in range(n_qubits):
        psi = apply_1q(psi, GATES_1Q["H"], q, n_qubits)

    p = len(gammas)
    for layer in range(p):
        gamma = float(gammas[layer])
        beta = float(betas[layer])
        # cost unitary: for each edge, exp(-i gamma (1 - Z_i Z_j)/2)
        # since the constant is a global phase, this equals exp(+i gamma Z_i Z_j / 2)
        # CNOT; Rz(-gamma); CNOT applies exp(+i gamma Z_i Z_j / 2) on the pair
        for (i, j) in edges:
            psi = apply_cnot(psi, i, j, n_qubits)
            psi = apply_1q(psi, Rz(-gamma), j, n_qubits)
            psi = apply_cnot(psi, i, j, n_qubits)
        # mixer: Rx(2 beta) on each qubit — exp(-i beta X)
        for q in range(n_qubits):
            psi = apply_1q(psi, Rx(2 * beta), q, n_qubits)
    return psi


def _maxcut_cost(probs, edges, n_qubits):
    """Expected cut size given probabilities over basis states."""
    total = 0.0
    for s, p in enumerate(probs):
        cut = 0
        for (i, j) in edges:
            bit_i = (s >> (n_qubits - 1 - i)) & 1
            bit_j = (s >> (n_qubits - 1 - j)) & 1
            if bit_i != bit_j:
                cut += 1
        total += cut * p
    return float(total)


# ============================================================
# Classical optimizers (no scipy — hand-rolled)
# ============================================================


def nelder_mead(f, x0, max_iter=200, tol=1e-6):
    """Minimal Nelder-Mead simplex minimizer."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        xi = x0.copy()
        # smaller perturbation to stay within typical bounds
        step = 0.1 if xi[i] == 0 else 0.05 * abs(xi[i]) + 0.05
        xi[i] += step
        simplex.append(xi)
    simplex = np.array(simplex)
    fvals = np.array([f(s) for s in simplex])

    alpha, gamma_c, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for it in range(max_iter):
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        # reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fr < fvals[0]:
            # expansion
            xe = centroid + gamma_c * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            # contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    return simplex[0], float(fvals[0]), it + 1


# ============================================================
# High-level VQE and QAOA runners
# ============================================================


def run_vqe(n_qubits=2, n_layers=1, J=1.0, h=0.5, max_iter=200, seed=42):
    """Find the ground state of a 1D transverse-field Ising model."""
    n_params = 2 * n_qubits * n_layers
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=n_params)

    history = []
    def cost(thetas):
        psi = _ry_ansatz_state(thetas, n_qubits, n_layers)
        e = hamiltonian_ising_1d(psi, n_qubits, J=J, h=h)
        history.append(e)
        return e

    x_opt, e_opt, iterations = nelder_mead(cost, x0, max_iter=max_iter)

    psi_opt = _ry_ansatz_state(x_opt, n_qubits, n_layers)
    probs = probabilities(psi_opt).tolist()
    return {
        "algorithm": "VQE",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "J": J, "h": h,
        "n_params": n_params,
        "optimal_params": x_opt.tolist(),
        "optimal_energy": float(e_opt),
        "iterations": iterations,
        "energy_history": [float(e) for e in history],
        "final_probabilities": probs,
    }


def run_qaoa(edges=None, n_qubits=4, p=1, max_iter=200, seed=42):
    """Solve MaxCut on a small graph using QAOA."""
    if edges is None:
        # default: 4-node cycle
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    # validate
    for (i, j) in edges:
        if not (0 <= i < n_qubits and 0 <= j < n_qubits and i != j):
            raise ValueError(f"invalid edge ({i}, {j}) for n_qubits={n_qubits}")

    n_params = 2 * p
    rng = np.random.default_rng(seed)

    best = None
    for trial in range(30):
        x0 = rng.uniform(-np.pi, np.pi, size=n_params)
        history_local = []
        def cost_local(params, hist=history_local):
            gammas = params[:p]
            betas = params[p:]
            psi = _qaoa_state(gammas, betas, edges, n_qubits)
            probs = probabilities(psi)
            c = _maxcut_cost(probs, edges, n_qubits)
            hist.append(c)
            return -c
        x_opt_t, neg_c_opt_t, iters_t = nelder_mead(cost_local, x0, max_iter=max_iter)
        if best is None or neg_c_opt_t < best[1]:
            best = (x_opt_t, neg_c_opt_t, iters_t, history_local)

    x_opt, neg_c_opt, iterations, history = best

    gammas_opt = x_opt[:p]
    betas_opt = x_opt[p:]
    psi_opt = _qaoa_state(gammas_opt, betas_opt, edges, n_qubits)
    probs_opt = probabilities(psi_opt).tolist()

    # brute-force the optimal cut for comparison
    best_cut = 0
    best_bitstring = ""
    for s in range(2 ** n_qubits):
        cut = 0
        for (i, j) in edges:
            bit_i = (s >> (n_qubits - 1 - i)) & 1
            bit_j = (s >> (n_qubits - 1 - j)) & 1
            if bit_i != bit_j:
                cut += 1
        if cut > best_cut:
            best_cut = cut
            best_bitstring = format(s, f"0{n_qubits}b")

    return {
        "algorithm": "QAOA",
        "n_qubits": n_qubits,
        "edges": [list(e) for e in edges],
        "p": p,
        "n_params": n_params,
        "optimal_params": x_opt.tolist(),
        "optimal_gammas": gammas_opt.tolist(),
        "optimal_betas": betas_opt.tolist(),
        "optimal_cost": float(-neg_c_opt),
        "classical_optimal": best_cut,
        "classical_optimal_bitstring": best_bitstring,
        "approximation_ratio": float(-neg_c_opt) / float(best_cut) if best_cut else 0.0,
        "iterations": iterations,
        "cost_history": [float(c) for c in history],
        "final_probabilities": probs_opt,
    }
