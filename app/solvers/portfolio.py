import numpy as np
from app.solvers.variational import nelder_mead
from app.solvers.qsim import (
    zero_state, apply_1q, apply_cnot, Rx, Rz, probabilities, GATES_1Q,
)


# ============================================================
# Quantum portfolio optimization via QAOA
#
# Maps the Markowitz portfolio problem to a binary Ising
# Hamiltonian, then runs QAOA to find the optimal portfolio.
#
# H = sum_{i<j} J_ij * Z_i Z_j + sum_i h_i * Z_i
#
# The minimum of H corresponds to the optimal binary selection x.
# ============================================================


# Stock universe — real tickers with historical statistics
STOCKS = {
    "AAPL":  {"name": "Apple",       "sector": "tech",     "mu": 0.15, "sigma": 0.25},
    "MSFT":  {"name": "Microsoft",   "sector": "tech",     "mu": 0.13, "sigma": 0.22},
    "GOOGL": {"name": "Alphabet",    "sector": "tech",     "mu": 0.12, "sigma": 0.26},
    "AMZN":  {"name": "Amazon",      "sector": "tech",     "mu": 0.14, "sigma": 0.30},
    "NVDA":  {"name": "NVIDIA",      "sector": "tech",     "mu": 0.22, "sigma": 0.45},
    "JPM":   {"name": "JPMorgan",    "sector": "finance",  "mu": 0.08, "sigma": 0.18},
    "BAC":   {"name": "Bank of America", "sector": "finance", "mu": 0.07, "sigma": 0.20},
    "GS":    {"name": "Goldman Sachs","sector": "finance", "mu": 0.09, "sigma": 0.22},
    "JNJ":   {"name": "J&J",         "sector": "health",   "mu": 0.06, "sigma": 0.15},
    "PFE":   {"name": "Pfizer",      "sector": "health",   "mu": 0.05, "sigma": 0.18},
    "XOM":   {"name": "Exxon",       "sector": "energy",   "mu": 0.07, "sigma": 0.22},
    "CVX":   {"name": "Chevron",     "sector": "energy",   "mu": 0.08, "sigma": 0.20},
    "WMT":   {"name": "Walmart",     "sector": "consumer", "mu": 0.07, "sigma": 0.16},
    "PG":    {"name": "P&G",         "sector": "consumer", "mu": 0.06, "sigma": 0.14},
    "KO":    {"name": "Coca-Cola",   "sector": "consumer", "mu": 0.05, "sigma": 0.13},
    "TSLA":  {"name": "Tesla",       "sector": "auto",     "mu": 0.20, "sigma": 0.55},
}


# Sector correlation matrix (simplified)
SECTOR_CORRELATION = {
    ("tech", "tech"): 0.75,
    ("tech", "finance"): 0.35,
    ("tech", "health"): 0.30,
    ("tech", "energy"): 0.20,
    ("tech", "consumer"): 0.30,
    ("tech", "auto"): 0.55,
    ("finance", "finance"): 0.65,
    ("finance", "health"): 0.30,
    ("finance", "energy"): 0.40,
    ("finance", "consumer"): 0.35,
    ("finance", "auto"): 0.30,
    ("health", "health"): 0.55,
    ("health", "energy"): 0.20,
    ("health", "consumer"): 0.35,
    ("health", "auto"): 0.20,
    ("energy", "energy"): 0.60,
    ("energy", "consumer"): 0.25,
    ("energy", "auto"): 0.25,
    ("consumer", "consumer"): 0.50,
    ("consumer", "auto"): 0.25,
    ("auto", "auto"): 0.65,
}


def _correlation(sector_a, sector_b):
    if sector_a == sector_b:
        return SECTOR_CORRELATION.get((sector_a, sector_b), 0.5)
    key = tuple(sorted([sector_a, sector_b]))
    return SECTOR_CORRELATION.get(key, 0.30)


def build_covariance(tickers):
    """Build the covariance matrix for the selected tickers."""
    n = len(tickers)
    Sigma = np.zeros((n, n))
    for i in range(n):
        si = STOCKS[tickers[i]]["sigma"]
        for j in range(n):
            sj = STOCKS[tickers[j]]["sigma"]
            rho = _correlation(STOCKS[tickers[i]]["sector"], STOCKS[tickers[j]]["sector"])
            Sigma[i, j] = rho * si * sj
    return Sigma


# ============================================================
# QAOA for the Markowitz problem
# ============================================================

def _markowitz_to_ising(tickers, risk_tolerance=1.0, budget=2, penalty=2.0):
    """Map Markowitz to Ising coefficients (J, h).

    x_i = (1 - Z_i) / 2   (spin z in {-1, +1} -> binary x in {0, 1})

    H_Markowitz = lambda * x^T Sigma x - mu^T x
                  + penalty * (sum_i x_i - budget)^2

    Convert to Ising form: H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i
    (dropping constant terms).
    """
    n = len(tickers)
    Sigma = build_covariance(tickers)
    mu = np.array([STOCKS[t]["mu"] for t in tickers])

    # Build the QUBO matrix Q (upper triangular)
    # H = sum_{i,j} Q_ij x_i x_j + sum_i c_i x_i
    Q = risk_tolerance * Sigma.copy()
    # Budget penalty: penalty * (sum x_i - budget)^2
    # = penalty * (sum x_i^2 + 2 sum_{i<j} x_i x_j - 2*budget*sum_i x_i + budget^2)
    # x_i^2 = x_i (binary), so:
    for i in range(n):
        Q[i, i] += penalty  # from x_i^2 term
        for j in range(i + 1, n):
            Q[i, j] += 2 * penalty
    # Linear term
    c = -mu - 2 * penalty * budget * np.ones(n) + penalty * np.ones(n)

    # Convert QUBO to Ising: x_i = (1 - Z_i)/2
    # sum_{ij} Q_ij x_i x_j = sum_{ij} Q_ij (1 - Z_i)(1 - Z_j)/4
    # The Z_i Z_j coefficient is -Q_ij/4 for i != j
    # The Z_i coefficient is -(sum_j Q_ij + sum_j Q_ji)/4 + Q_ii/4 - c_i/2
    J = np.zeros((n, n))
    h = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = -Q[i, j] / 4
    for i in range(n):
        h[i] = -np.sum(Q[i, :]) / 4 - np.sum(Q[:, i]) / 4 + Q[i, i] / 2 - c[i] / 2

    return J, h


def _qaoa_portfolio(tickers, risk_tolerance, budget, p=2, max_iter=300, seed=42):
    """Run QAOA on the portfolio problem."""
    n = len(tickers)
    J, h = _markowitz_to_ising(tickers, risk_tolerance, budget)

    # Initial state: |+>^n
    def initial_state():
        psi = zero_state(n)
        for q in range(n):
            psi = apply_1q(psi, GATES_1Q["H"], q, n)
        return psi

    def cost_phase(psi, gamma):
        # Apply exp(-i gamma H) where H = sum J_ij Z_i Z_j + sum h_i Z_i
        # Implemented as: for each pair, Rz(2*gamma*J_ij) on both after CNOT
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    psi = apply_cnot(psi, i, j, n)
                    psi = apply_1q(psi, Rz(2 * gamma * J[i, j]), j, n)
                    psi = apply_cnot(psi, i, j, n)
        # Linear terms: Rz(2*gamma*h_i) on each qubit
        for i in range(n):
            if abs(h[i]) > 1e-12:
                psi = apply_1q(psi, Rz(2 * gamma * h[i]), i, n)
        return psi

    def mixer(psi, beta):
        for q in range(n):
            psi = apply_1q(psi, Rx(2 * beta), q, n)
        return psi

    def build_state(params):
        psi = initial_state()
        for layer in range(p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            psi = cost_phase(psi, gamma)
            psi = mixer(psi, beta)
        return psi

    def energy_of_probs(probs):
        # <H> = sum over bitstrings of H(x) * p(x)
        total = 0.0
        for state_idx in range(2 ** n):
            p_x = probs[state_idx]
            if p_x < 1e-12:
                continue
            # Convert state_idx to Z eigenvalues: bit k -> Z_k = 1 - 2*bit_k
            Z = np.array([1 - 2 * ((state_idx >> (n - 1 - k)) & 1) for k in range(n)])
            # H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i
            H_val = 0.0
            for i in range(n):
                H_val += h[i] * Z[i]
                for j in range(i + 1, n):
                    H_val += J[i, j] * Z[i] * Z[j]
            total += p_x * H_val
        return total

    def cost(params):
        psi = build_state(params)
        probs = probabilities(psi)
        return energy_of_probs(probs)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=2 * p)

    x_opt, e_opt, iterations = nelder_mead(cost, x0, max_iter=max_iter)

    # Sample the final state and evaluate
    psi = build_state(x_opt)
    probs = probabilities(psi)

    # Find the top bitstrings
    top = np.argsort(-probs)[:5]
    top_portfolios = []
    for idx in top:
        if probs[idx] < 1e-4:
            continue
        selected = []
        for k in range(n):
            if (idx >> (n - 1 - k)) & 1:
                selected.append(tickers[k])
        # Evaluate the Markowitz objective for this portfolio
        if len(selected) > 0:
            Sigma = build_covariance(tickers)
            mu = np.array([STOCKS[t]["mu"] for t in tickers])
            x_vec = np.zeros(n)
            for k in range(n):
                if (idx >> (n - 1 - k)) & 1:
                    x_vec[k] = 1
            expected_return = float(mu @ x_vec)
            expected_risk = float(np.sqrt(x_vec @ Sigma @ x_vec))
            sharpe = expected_return / expected_risk if expected_risk > 0 else 0.0
            top_portfolios.append({
                "bitstring": format(idx, f"0{n}b"),
                "selected": selected,
                "probability": float(probs[idx]),
                "expected_return": expected_return,
                "expected_risk": expected_risk,
                "sharpe_ratio": sharpe,
                "objective": float(risk_tolerance * expected_risk ** 2 - expected_return),
            })
        else:
            top_portfolios.append({
                "bitstring": format(idx, f"0{n}b"),
                "selected": [],
                "probability": float(probs[idx]),
                "expected_return": 0.0,
                "expected_risk": 0.0,
                "sharpe_ratio": 0.0,
                "objective": 0.0,
            })

    # Sort by objective (lower is better)
    top_portfolios.sort(key=lambda x: x["objective"])

    return {
        "n_stocks": n,
        "risk_tolerance": risk_tolerance,
        "budget": budget,
        "p_layers": p,
        "iterations": iterations,
        "qaoa_energy": float(e_opt),
        "top_portfolios": top_portfolios,
        "optimal_portfolio": top_portfolios[0] if top_portfolios else None,
    }


def optimize_portfolio(tickers=None, risk_tolerance=0.5, budget=2, p=2, max_iter=300):
    """Main entry point — optimize a portfolio from a ticker universe."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "GS"]
    for t in tickers:
        if t not in STOCKS:
            raise ValueError(f"unknown ticker {t}. valid: {list(STOCKS.keys())}")
    if len(tickers) < 2 or len(tickers) > 8:
        raise ValueError("portfolio size must be between 2 and 8 stocks")
    if budget < 1 or budget > len(tickers):
        raise ValueError(f"budget must be between 1 and {len(tickers)}")

    result = _qaoa_portfolio(tickers, risk_tolerance, budget, p=p, max_iter=max_iter)
    result["tickers"] = tickers
    result["stock_info"] = [
        {"ticker": t, "name": STOCKS[t]["name"], "sector": STOCKS[t]["sector"],
         "mu": STOCKS[t]["mu"], "sigma": STOCKS[t]["sigma"]}
        for t in tickers
    ]
    return result


def list_stocks():
    return [
        {"ticker": t, "name": v["name"], "sector": v["sector"], "mu": v["mu"], "sigma": v["sigma"]}
        for t, v in STOCKS.items()
    ]
