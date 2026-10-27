import numpy as np


def bloch_vector(psi: np.ndarray) -> tuple:
    """For a 2-element state vector, compute the Bloch vector (x, y, z).

    Returns (x, y, z, r) where r = sqrt(x^2 + y^2 + z^2) is the vector length.
    For a pure state r = 1; for mixed r < 1.
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(2)
    # Normalize
    n = np.sqrt(np.abs(psi[0])**2 + np.abs(psi[1])**2)
    if n < 1e-12:
        raise ValueError("state vector has zero norm")
    psi = psi / n

    a, b = psi[0], psi[1]
    # <X> = 2 Re(a* b)  ; <Y> = 2 Im(a* b)  ; <Z> = |a|^2 - |b|^2
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a)**2 - np.abs(b)**2
    r = float(np.sqrt(x*x + y*y + z*z))

    # spherical coords
    theta = float(np.arccos(np.clip(z, -1, 1)))
    phi = float(np.arctan2(y, x))

    return float(x), float(y), float(z), r, theta, phi


def reduced_density_matrix(psi: np.ndarray, n_qubits: int, keep: int) -> np.ndarray:
    """Reduce an n-qubit state to a single-qubit density matrix (2x2).

    Keeps qubit `keep`; traces out all others.
    """
    psi = np.asarray(psi, dtype=np.complex64)
    psi_t = psi.reshape([2] * n_qubits)
    keep_axis = n_qubits - 1 - keep
    trace_axes = [a for a in range(n_qubits) if a != keep_axis]
    # move keep to front, trace to back
    psi_t = np.moveaxis(psi_t, keep_axis, 0)
    psi_t = psi_t.reshape(2, -1)
    rho = psi_t @ psi_t.conj().T
    return rho


def bloch_from_density(rho: np.ndarray) -> tuple:
    """Bloch vector from a 2x2 density matrix."""
    rho = np.asarray(rho, dtype=np.complex128)
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = float(np.real(rho[0, 0] - rho[1, 1]))
    r = float(np.sqrt(x*x + y*y + z*z))
    theta = float(np.arccos(np.clip(z, -1, 1)))
    phi = float(np.arctan2(y, x))
    return float(x), float(y), float(z), r, theta, phi


def bloch_of_qubit(psi: np.ndarray, n_qubits: int, qubit: int) -> dict:
    """Bloch sphere data for any qubit in any n-qubit circuit."""
    rho = reduced_density_matrix(psi, n_qubits, qubit)
    x, y, z, r, theta, phi = bloch_from_density(rho)
    # Purity = Tr(rho^2)
    purity = float(np.real(np.trace(rho @ rho)))
    return {
        "qubit": qubit,
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "theta_rad": theta,
        "phi_rad": phi,
        "purity": purity,
    }
