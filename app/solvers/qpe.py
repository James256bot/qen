import numpy as np
from app.solvers.qsim import (
    zero_state, apply_1q, apply_cnot, GATES_1Q,
    Rx, Ry, Rz, probabilities, sample_shots,
)
from app.solvers.noise import state_to_density


# ============================================================
# Quantum Phase Estimation
#
# Layout: |ancillas> |state>
#   n_ancilla qubits hold the binary expansion of phi
#   the state register holds the eigenstate |u> of U
#
# We use MSB-first indexing throughout: qubit 0 is the high bit.
# Convention: ancillas are qubits [0, n_ancilla-1],
#             state register is qubits [n_ancilla, n_ancilla+state_qubits-1]
# ============================================================


def _controlled_unitary_power(psi, U, power, control, target_qubits, n_qubits):
    """Apply U^power to target_qubits, controlled on `control` being 1.

    Implemented by: (a) applying the controlled operation via
        decompose U into a sequence of gates
    For a phase-gate or simple unitary, we build U^power explicitly
    as a 2^k x 2^k matrix and apply it via tensor contraction.

    Args:
        psi: state vector
        U: 2^k x 2^k unitary, k = len(target_qubits)
        power: integer power of U to apply
        control: control qubit index
        target_qubits: list of qubit indices the unitary acts on
        n_qubits: total qubit count
    """
    k = len(target_qubits)
    if U.shape != (2**k, 2**k):
        raise ValueError(f"U must be {2**k}x{2**k}")

    # U^power
    Up = np.linalg.matrix_power(U, power)

    # Reshape psi into a tensor
    T = psi.reshape([2] * n_qubits)
    # For the controlled application, work on the flat vector
    # Simpler approach: iterate over all indices and only apply U when
    # control bit is 1.
    dim = 2 ** n_qubits
    out = np.zeros_like(psi, dtype=np.complex128)
    cbit = n_qubits - 1 - control

    # positions of target bits within the flat index
    tbits = [n_qubits - 1 - q for q in target_qubits]

    # collect all unique pairs (i, i') where i' is the result of applying
    # U's basis change on the target subspace, conditioned on control
    for i in range(dim):
        if (i >> cbit) & 1:
            # extract the target subindex
            sub = 0
            for tb in tbits:
                sub = (sub << 1) | ((i >> tb) & 1)
            # apply U: distribute amplitude
            for subp in range(2**k):
                coeff = Up[subp, sub]
                if coeff == 0:
                    continue
                j = i
                for idx, tb in enumerate(tbits):
                    bit = (subp >> (k - 1 - idx)) & 1
                    if bit:
                        j |= (1 << tb)
                    else:
                        j &= ~(1 << tb)
                out[j] += coeff * psi[i]
        else:
            out[i] += psi[i]
    return out.astype(np.complex64)


def _inverse_qft(psi, qubits, n_qubits):
    """Apply the inverse quantum Fourier transform on the given qubits."""
    k = len(qubits)
    # Inverse QFT: reverse the standard QFT
    # Standard QFT: H each, then controlled rotations, then SWAP
    # Inverse: SWAP, then reverse controlled rotations, then H each
    def controlled_phase(control, target, angle):
        # Apply phase e^{i*angle} when both control and target are 1
        # Decompose: Rz on target controlled by control
        # Rz(theta) on target = diag(1, e^{i theta}) up to global phase
        # Actually controlled-Rz(angle) = diag(1, 1, 1, e^{i*angle})
        # We implement it as: apply Rz(angle) on target after CNOT
        # CNOT(control, target); Rz(angle); CNOT(control, target)
        # = controlled-Z-type phase
        # For simplicity, use the CNOT-Rz-CNOT construction:
        pass

    # We use the direct approach: build the IQFT matrix for k qubits
    # and apply it via the same controlled_unitary_power-style contraction
    # but without control.
    # For small k (<= 10), this is fine.
    N = 2 ** k
    omega = np.exp(2j * np.pi / N)
    # QFT matrix (unitary)
    QFT = np.array([[omega ** (j * l) for l in range(N)] for j in range(N)]) / np.sqrt(N)
    IQFT = QFT.conj().T

    # Apply IQFT to the subspace of the given qubits
    dim = 2 ** n_qubits
    tbits = [n_qubits - 1 - q for q in qubits]
    out = np.zeros_like(psi, dtype=np.complex128)
    for i in range(dim):
        sub = 0
        for tb in tbits:
            sub = (sub << 1) | ((i >> tb) & 1)
        for subp in range(N):
            coeff = IQFT[subp, sub]
            if coeff == 0:
                continue
            j = i
            for idx, tb in enumerate(tbits):
                bit = (subp >> (k - 1 - idx)) & 1
                if bit:
                    j |= (1 << tb)
                else:
                    j &= ~(1 << tb)
            out[j] += coeff * psi[i]
    return out.astype(np.complex64)


def run_qpe(U, n_ancilla=5, eigenstate=None, shots=0):
    """Run QPE to estimate the phase of U.

    Args:
        U: 2x2 or 2^k x 2^k unitary matrix (for the state register)
        n_ancilla: number of ancilla qubits (precision bits)
        eigenstate: optional initial state of the state register.
                    If None, uses |1> (for phase gates) or |1>^k (for
                    multi-qubit U acting on |1...1>).
        shots: number of measurement samples (0 = return exact distribution)

    Returns:
        dict with 'phase_estimate' (the most likely phi), 'probabilities'
        (over ancilla outcomes), 'binary_string', and the full state.
    """
    if U.shape[0] != U.shape[1]:
        raise ValueError("U must be square")
    state_dim = U.shape[0]
    state_qubits = int(np.log2(state_dim))
    if 2**state_qubits != state_dim:
        raise ValueError("U dimension must be a power of 2")

    n_qubits = n_ancilla + state_qubits
    if n_qubits > 16:
        raise ValueError(f"QPE supports up to 16 qubits total, got {n_qubits}")

    psi = zero_state(n_qubits)

    # Step 1: initialize the state register (qubits n_ancilla .. n_qubits-1)
    # The state register sits in the LOW-order bits of the flat index
    # (MSB-first indexing: qubit 0 is the high bit).
    if eigenstate is None:
        # Default: |1...1> in the state register
        idx = 0
        for q in range(n_ancilla, n_qubits):
            idx |= (1 << (n_qubits - 1 - q))
        psi[:] = 0
        psi[idx] = 1.0
    else:
        # eigenstate is a 2^state_qubits vector
        # Reshape: psi -> (2^n_ancilla, 2^state_qubits), set state register
        T = psi.reshape(2**n_ancilla, 2**state_qubits)
        T[:] = 0
        T[0, :] = eigenstate
        psi = T.reshape(-1)

    # Step 2: Hadamard on each ancilla qubit (qubits 0 .. n_ancilla-1)
    for q in range(n_ancilla):
        psi = apply_1q(psi, GATES_1Q["H"], q, n_qubits)


    # Apply controlled U^(2^k) from ancilla k
    # Convention: ancilla 0 is the MSB of the phase
    target_qubits = list(range(n_ancilla, n_qubits))
    for k in range(n_ancilla):
        power = 2 ** (n_ancilla - 1 - k)
        psi = _controlled_unitary_power(psi, U, power, control=k,
                                         target_qubits=target_qubits,
                                         n_qubits=n_qubits)

    # Inverse QFT on the ancillas
    psi = _inverse_qft(psi, list(range(n_ancilla)), n_qubits)

    # Extract probability distribution over ancilla outcomes
    probs_full = probabilities(psi)
    # Marginalize over the state register
    probs = np.zeros(2**n_ancilla)
    for i, p in enumerate(probs_full):
        anc = i >> state_qubits
        probs[anc] += p

    # Find most likely ancilla outcome
    best = int(np.argmax(probs))
    phase_est = best / (2 ** n_ancilla)
    binary = format(best, f"0{n_ancilla}b")

    result = {
        "n_ancilla": n_ancilla,
        "n_qubits": n_qubits,
        "state_qubits": state_qubits,
        "phase_estimate": float(phase_est),
        "binary_string": binary,
        "most_likely_outcome": best,
        "probabilities": probs.tolist(),
    }
    if shots > 0:
        rng = np.random.default_rng()
        idx = rng.choice(2**n_ancilla, size=shots, p=probs / probs.sum())
        counts = {}
        for i in idx:
            key = format(int(i), f"0{n_ancilla}b")
            counts[key] = counts.get(key, 0) + 1
        result["shots"] = shots
        result["counts"] = counts
    return result


# ============================================================
# Convenience: QPE for a single-qubit phase gate
# ============================================================


def phase_gate(phi):
    """U = diag(1, e^{2πi φ})."""
    return np.array([[1, 0], [0, np.exp(2j * np.pi * phi)]], dtype=np.complex128)


def T_gate():
    return phase_gate(1/8)


def S_gate():
    return phase_gate(1/4)


def Z_gate():
    return phase_gate(1/2)
