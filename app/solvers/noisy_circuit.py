import numpy as np
from app.solvers.qsim import execute_circuit, sample_shots
from app.solvers.noise import (
    state_to_density,
    apply_1q_channel,
    density_to_probabilities,
    depolarizing_kraus,
    amplitude_damping_kraus,
    dephasing_kraus,
    partial_trace,
    entropy_of_density,
)


NOISE_CHANNELS = {
    "depolarizing":     depolarizing_kraus,
    "amplitude_damping": amplitude_damping_kraus,
    "dephasing":        dephasing_kraus,
}


def _noise_kraus(noise_type, rate):
    if noise_type not in NOISE_CHANNELS:
        raise ValueError(
            f"unknown noise type: {noise_type}. "
            f"valid: {list(NOISE_CHANNELS.keys())}"
        )
    if not (0.0 <= rate <= 1.0):
        raise ValueError("noise rate must be in [0, 1]")
    return NOISE_CHANNELS[noise_type](rate)


def run_circuit_noisy(n_qubits, gates, shots=0, entanglement_cut=None,
                      bloch_qubits=None, noise=None):
    """Run a circuit, apply a single-qubit noise channel to the specified
    qubits, then compute observables from the resulting density matrix.
    """
    # 1. Apply the circuit to a pure state
    psi = execute_circuit(n_qubits, gates)

    # 2. If noise specified, convert to density matrix and apply the channel
    if noise is not None:
        if not isinstance(noise, dict):
            raise ValueError("noise must be a dict")
        ntype = noise.get("type", "dephasing")
        rate = float(noise.get("rate", 0.1))
        qubits = noise.get("qubits", list(range(n_qubits)))
        if not qubits:
            raise ValueError("noise.qubits must be a non-empty list")

        kraus_ops = _noise_kraus(ntype, rate)
        rho = state_to_density(psi)
        for q in qubits:
            if not (0 <= q < n_qubits):
                raise ValueError(f"noise qubit {q} out of range 0..{n_qubits-1}")
            rho = apply_1q_channel(rho, kraus_ops, target=q, n_qubits=n_qubits)

        probs = density_to_probabilities(rho)
        n_states = len(probs)
        # Purity and diagonal - full density matrix info not returned (too big)
        purity = float(np.real(np.trace(rho @ rho)))
        result = {
            "n_qubits": n_qubits,
            "n_states": n_states,
            "probabilities": probs.tolist(),
            "norm": float(np.sum(probs)),
            "purity": purity,
            "noise": {"type": ntype, "rate": rate, "qubits": qubits},
        }
        # Shots: sample from the diagonal
        if shots > 0:
            p = probs / probs.sum()
            rng = np.random.default_rng()
            idx = rng.choice(n_states, size=shots, p=p)
            counts = {}
            for i in idx:
                key = format(int(i), f"0{n_qubits}b")
                counts[key] = counts.get(key, 0) + 1
            result["shots"] = shots
            result["counts"] = counts
        # Entanglement entropy of a subsystem
        if entanglement_cut is not None:
            rho_A = partial_trace(rho, entanglement_cut, n_qubits)
            S = entropy_of_density(rho_A)
            result["entanglement_cut"] = entanglement_cut
            result["entanglement_entropy_nats"] = S
            result["entanglement_entropy_bits"] = S / float(np.log(2.0))
        # Bloch vectors (from the reduced density matrix)
        if bloch_qubits:
            bloch_data = []
            for q in bloch_qubits:
                rho_q = partial_trace(rho, [q], n_qubits)
                x = 2 * np.real(rho_q[0, 1])
                y = 2 * np.imag(rho_q[1, 0])
                z = float(np.real(rho_q[0, 0] - rho_q[1, 1]))
                r_len = float(np.sqrt(x*x + y*y + z*z))
                purity_q = float(np.real(np.trace(rho_q @ rho_q)))
                bloch_data.append({
                    "qubit": q,
                    "x": float(x), "y": float(y), "z": float(z), "r": r_len,
                    "purity": purity_q,
                })
            result["bloch"] = bloch_data
        return result

    # No noise: fall back to the ideal-state path
    from app.solvers.qsim import run_circuit
    return run_circuit(n_qubits, gates, shots=shots,
                       entanglement_cut=entanglement_cut,
                       bloch_qubits=bloch_qubits)
