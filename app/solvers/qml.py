import numpy as np
from app.solvers.qsim import zero_state, apply_1q, apply_cnot, Ry, Rz, GATES_1Q


# ============================================================
# Quantum kernel classifier
#
# 1. Encode classical features into quantum states via a feature map
# 2. Compute the quantum kernel K(x, x') = |<psi(x)|psi(x')>|^2
# 3. Train a classical SVM on the kernel
# 4. Report accuracy on the test set
# ============================================================


# ============================================================
# Datasets
# ============================================================

def load_iris():
    """Classic iris dataset — 3 classes, 4 features, 150 samples."""
    from sklearn.datasets import load_iris as _load
    data = _load()
    X = data.data.tolist()
    y = data.target.tolist()
    return X, y, ["setosa", "versicolor", "virginica"]


# Fallback: hand-coded small dataset if sklearn isn't available
IRIS_MANUAL = {
    "X": [
        # Setosa (20 samples)
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2],
        [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
        [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
        # Versicolor (20 samples)
        [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5],
        [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3],
        [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0],
        [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
        # Virginica (20 samples)
        [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1],
        [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8], [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9],
        [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4],
        [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2],
        [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5],
    ],
    "y": [0]*20 + [1]*20 + [2]*20,
    "names": ["setosa", "versicolor", "virginica"],
}


def load_dataset(name):
    if name == "iris":
        try:
            return load_iris()
        except Exception:
            return IRIS_MANUAL["X"], IRIS_MANUAL["y"], IRIS_MANUAL["names"]
    raise ValueError(f"unknown dataset {name}. valid: iris")


# ============================================================
# Quantum feature map
# ============================================================

def feature_map_zz(x, n_qubits):
    """ZZ feature map: encode classical features into a quantum state.

    Reference: Havlicek et al., Nature 567, 209 (2019).

    For each feature x_i:
        apply Ry(x_i) to qubit i
    Then entangle with ZZ rotations:
        CNOT(i, i+1); Rz(2*x_i*x_{i+1}); CNOT(i, i+1)
    """
    psi = zero_state(n_qubits)
    # Encode features via Ry rotations
    for q in range(n_qubits):
        if q < len(x):
            psi = apply_1q(psi, Ry(x[q]), q, n_qubits)
    # Entangle via ZZ
    for q in range(n_qubits - 1):
        if q < len(x) and q + 1 < len(x):
            psi = apply_cnot(psi, q, q + 1, n_qubits)
            psi = apply_1q(psi, Rz(2 * x[q] * x[q + 1]), q + 1, n_qubits)
            psi = apply_cnot(psi, q, q + 1, n_qubits)
    return psi


def quantum_kernel(x1, x2, n_qubits, feature_map="zz"):
    """K(x1, x2) = |<psi(x1)|psi(x2)>|^2."""
    psi1 = feature_map_zz(x1, n_qubits)
    psi2 = feature_map_zz(x2, n_qubits)
    overlap = abs(np.vdot(psi1, psi2)) ** 2
    return float(overlap)


def kernel_matrix(X, n_qubits):
    """Compute the full kernel matrix."""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k = quantum_kernel(X[i], X[j], n_qubits)
            K[i, j] = k
            K[j, i] = k
    return K


# ============================================================
# Simplified SVM (no sklearn — hand-rolled)
# ============================================================

def train_test_split(X, y, test_frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


def _kernel_svm_predict(K_train, y_train, K_test, n_classes):
    """Kernel SVM via nearest-centroid-in-kernel-space approximation.

    For each test point, compute kernel similarities to each class's
    training points, and assign the class with the highest average
    similarity. This is a "kernel nearest class centroid" classifier —
    simpler than full SVM but works for small datasets.
    """
    predictions = []
    for i in range(K_test.shape[0]):
        scores = np.zeros(n_classes)
        counts = np.zeros(n_classes)
        for j in range(K_train.shape[0]):
            scores[y_train[j]] += K_test[i, j]
            counts[y_train[j]] += 1
        # Average kernel similarity per class
        for c in range(n_classes):
            if counts[c] > 0:
                scores[c] /= counts[c]
        predictions.append(int(np.argmax(scores)))
    return predictions


def _accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if y_true else 0.0


def classify(dataset="iris", n_qubits=4, test_frac=0.3, seed=42):
    """Full pipeline: load → encode → kernel → classify → accuracy."""
    X, y, class_names = load_dataset(dataset)
    n_classes = len(set(y))

    if n_qubits < 2 or n_qubits > 8:
        raise ValueError("n_qubits must be between 2 and 8")

    # Truncate features to n_qubits (or pad with zeros)
    X_enc = []
    for x in X:
        if len(x) >= n_qubits:
            X_enc.append(x[:n_qubits])
        else:
            padded = list(x) + [0.0] * (n_qubits - len(x))
            X_enc.append(padded)

    X_train, y_train, X_test, y_test = train_test_split(X_enc, y, test_frac, seed)

    # Compute kernels
    K_train = kernel_matrix(X_train, n_qubits)
    K_test = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            K_test[i, j] = quantum_kernel(X_test[i], X_train[j], n_qubits)

    # Classify
    y_pred = _kernel_svm_predict(K_train, y_train, K_test, n_classes)

    accuracy = _accuracy(y_test, y_pred)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y_test, y_pred):
        cm[a, b] += 1

    # Baseline: majority class
    from collections import Counter
    majority = Counter(y_train).most_common(1)[0][0]
    baseline_pred = [majority] * len(y_test)
    baseline_acc = _accuracy(y_test, baseline_pred)

    return {
        "algorithm": "quantum_kernel_classifier",
        "dataset": dataset,
        "n_qubits": n_qubits,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": n_classes,
        "class_names": class_names,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_acc,
        "improvement_over_baseline": accuracy - baseline_acc,
        "confusion_matrix": cm.tolist(),
        "predictions": [
            {"true": int(a), "predicted": int(b), "correct": bool(a == b)}
            for a, b in zip(y_test, y_pred)
        ],
        "note": "Quantum kernel K(x,x') = |<psi(x)|psi(x')>|^2 with a "
                "ZZ feature map. Classifier: kernel nearest-class-centroid.",
    }


def list_datasets():
    return [
        {"name": "iris", "classes": 3, "features": 4, "samples": 150,
         "description": "Classic iris dataset — 3 species of iris flowers"},
    ]
