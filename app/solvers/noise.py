import numpy as np


def depolarizing_kraus(p):
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return [
        np.sqrt(1 - 3*p/4) * I,
        np.sqrt(p/4) * X,
        np.sqrt(p/4) * Y,
        np.sqrt(p/4) * Z,
    ]


def amplitude_damping_kraus(gamma):
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [K0, K1]


def dephasing_kraus(lam):
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=np.complex128)
    K1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=np.complex128)
    return [K0, K1]


def apply_1q_channel(rho, kraus_ops, target, n_qubits):
    """Apply a 1-qubit channel to `target` of an n-qubit density matrix.

    Uses Kronecker products to build the full 2^n x 2^n operator.
    Correct by construction; O(4^n) memory per intermediate, fine for n <= 10.
    """
    dim = 2 ** n_qubits
    if rho.shape != (dim, dim):
        raise ValueError(f"rho shape {rho.shape} does not match n_qubits={n_qubits}")

    I2 = np.eye(2, dtype=np.complex128)
    out = np.zeros((dim, dim), dtype=np.complex128)

    for K in kraus_ops:
        # K_full = I on all qubits except `target` (MSB = qubit 0)
        ops = [(K if q == target else I2) for q in range(n_qubits)]
        K_full = ops[0]
        for op in ops[1:]:
            K_full = np.kron(K_full, op)
        out += K_full @ rho @ K_full.conj().T

    return out


def state_to_density(psi):
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def density_to_probabilities(rho):
    return np.real(np.diag(rho)).astype(np.float64)


def partial_trace(rho, keep, n_qubits):
    """Reduced density matrix on `keep` qubits.

    Uses explicit reshape + einsum on a 4-index tensor:
    reshape rho into (keep_dim, trace_dim, keep_dim, trace_dim), then trace
    over the two trace_dim axes.
    """
    dim = 2 ** n_qubits
    if rho.shape != (dim, dim):
        raise ValueError(f"rho shape {rho.shape} does not match n_qubits={n_qubits}")

    keep_dim = 2 ** len(keep)
    trace_dim = dim // keep_dim

    # Reshape rho into tensor with 2n indices
    T = rho.reshape([2] * (2 * n_qubits))

    # Axis mapping for qubit q: row index = n_qubits - 1 - q, col index = n_qubits + (n_qubits - 1 - q)
    keep_rows = [n_qubits - 1 - q for q in keep]
    keep_cols = [n_qubits + (n_qubits - 1 - q) for q in keep]
    trace_rows = [a for a in range(n_qubits) if a not in keep_rows]
    trace_cols = [a for a in range(n_qubits, 2 * n_qubits) if a not in keep_cols]

    perm = keep_rows + trace_rows + keep_cols + trace_cols
    T = np.transpose(T, perm)
    T = T.reshape(keep_dim, trace_dim, keep_dim, trace_dim)
    # trace over trace_dim (axes 1 and 3)
    # correct: rho_A[a, a'] = sum_r T[a, r, a', r]
    # simplest correct form:
    rho_A = np.einsum("ijkl->il", T.transpose(0, 2, 1, 3))
    # Wait, that's not right either. Let me be explicit:
    # T has shape (keep_dim, trace_dim, keep_dim, trace_dim)
    # We want rho_A[i, k] = sum_j T[i, j, k, j]
    rho_A = np.einsum("ijkj->ik", T)
    return rho_A


def entropy_of_density(rho_A):
    w = np.linalg.eigvalsh(rho_A)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log(w)))


def entanglement_entropy_density(rho, keep, n_qubits):
    rho_A = partial_trace(rho, keep, n_qubits)
    return entropy_of_density(rho_A)
