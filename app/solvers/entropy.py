import numpy as np


def von_neumann_entropy(rho):
    rho = np.asarray(rho, dtype=float)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be square 2D, got shape {rho.shape}")

    if not np.allclose(rho, rho.T, atol=1e-10):
        raise ValueError("rho must be symmetric (Hermitian for real input)")

    tr = float(np.trace(rho))
    if abs(tr - 1.0) > 1e-8:
        raise ValueError(f"trace(rho) = {tr}, expected 1.0")

    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    S = float(-np.sum(w * np.log(w)))
    d = rho.shape[0]

    return {
        "S_nats": S,
        "S_bits": S / float(np.log(2.0)),
        "purity": float(np.sum(w * w)),
        "rank": int(w.size),
        "dim": int(d),
        "S_max_nats": float(np.log(d)),
        "S_max_bits": float(np.log2(d)),
    }
