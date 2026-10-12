import numpy as np

def eigh_tridiagonal_k(diag, offd, k):
    """k smallest eigenpairs of a symmetric tridiagonal matrix.
    Uses Sturm-sequence bisection (evals) + inverse iteration (evecs).
    Pure numpy, no scipy.
    """
    d = diag.astype(float).copy()
    e = offd.astype(float).copy()
    N = len(d)

    # Gershgorin bounds for the spectrum
    e_pad_lo = np.concatenate([[0.0], e])
    e_pad_hi = np.concatenate([e, [0.0]])
    lo = (d - np.abs(e_pad_lo) - np.abs(e_pad_hi)).min()
    hi = (d + np.abs(e_pad_lo) + np.abs(e_pad_hi)).max()

    def sturm_count(x):
        q = d[0] - x
        count = 1 if q < 0 else 0
        for i in range(1, N):
            if q == 0.0:
                q = 1e-300
            q = d[i] - x - e[i-1]**2 / q
            if q < 0:
                count += 1
        return count

    evals = np.empty(k)
    lo_i = lo
    for i in range(k):
        a, b = lo_i, hi
        for _ in range(60):
            m = 0.5 * (a + b)
            if sturm_count(m) >= i + 1:
                b = m
            else:
                a = m
        evals[i] = 0.5 * (a + b)
        lo_i = evals[i]

    # Inverse iteration for eigenvectors
    evecs = np.empty((N, k))
    for i in range(k):
        shift = evals[i] + 1e-12 * max(1.0, abs(evals[i]))
        v = np.ones(N)
        for _ in range(4):
            # Solve (T - shift*I) w = v  via Thomas algorithm
            c = e.copy()
            dd = d - shift
            rhs = v.copy()
            for j in range(1, N):
                m = c[j-1] / dd[j-1]
                dd[j] -= m * c[j-1]
                rhs[j] -= m * rhs[j-1]
            w = np.empty(N)
            w[-1] = rhs[-1] / dd[-1]
            for j in range(N-2, -1, -1):
                w[j] = (rhs[j] - c[j] * w[j+1]) / dd[j]
            v = w / np.linalg.norm(w)
        evecs[:, i] = v

    return evals, evecs
