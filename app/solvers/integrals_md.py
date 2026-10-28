import numpy as np
from math import factorial, pi, sqrt, erfc
from app.solvers.integrals_core import (
    boys, gaussian_product_center, overlap_1d, overlap_3d, kinetic_3d, nuclear_3d,
)


# ============================================================
# Electron repulsion integrals via McMurchie-Davidson
#
# Reference: McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978)
# ============================================================


def hermite_coeffs(i, j, t, Qx, a, b):
    """Hermite expansion coefficient E_t^{ij} for one Cartesian axis.

    E_t^{ij} = sum over combinations of the two Gaussians' angular momenta.
    """
    p = a + b
    q = a * b / p

    # Base cases
    if i == 0 and j == 0 and t == 0:
        return 1.0
    if i == 0 and j == 0:
        return 0.0
    if i == 0 and j == 0:
        return 0.0

    # Recurrence relations from McMurchie-Davidson:
    # E_t^{i+1,j} = (1/(2p)) E_{t-1}^{ij} + X_PA E_t^{ij} + (t+1) E_{t+1}^{ij}
    # E_t^{i,j+1} = (1/(2p)) E_{t-1}^{ij} + X_PB E_t^{ij} + (t+1) E_{t+1}^{ij}

    # Use memoized dynamic programming
    # Build a dict of (i, j, t) -> value for the given i, j
    # Only need t in the range that contributes

    # For simplicity, implement the base recurrence
    # E_0^{00} = 1
    # E_t^{ij} is nonzero only for 0 <= t <= i+j

    def E(ii, jj, tt):
        if ii == 0 and jj == 0:
            return 1.0 if tt == 0 else 0.0
        if tt > ii + jj or tt < 0:
            return 0.0
        if ii > 0:
            # E_t^{i,j} = (1/(2p)) E_{t-1}^{i-1,j} + X_PA E_t^{i-1,j} + (t+1) E_{t+1}^{i-1,j}
            return (E(ii - 1, jj, tt - 1) / (2 * p)
                    + Qx[0] * E(ii - 1, jj, tt)
                    + (tt + 1) * E(ii - 1, jj, tt + 1))
        else:
            # E_t^{i,j} = (1/(2p)) E_{t-1}^{i,j-1} + X_PB E_t^{i,j-1} + (t+1) E_{t+1}^{i,j-1}
            return (E(ii, jj - 1, tt - 1) / (2 * p)
                    + Qx[1] * E(ii, jj - 1, tt)
                    + (tt + 1) * E(ii, jj - 1, tt + 1))

    return E(i, j, t)


def R_tuv(t, u, v, p, alpha, PC):
    """2-center integral R_{tuv} = (-2α)^(t+u+v) * F_{t+u+v}(α |PC|²)."""
    n = t + u + v
    PC2 = np.dot(PC, PC)
    return (-2 * alpha) ** n * boys(n, alpha * PC2)


def eri_primitive(a, lmn1, A, b, lmn2, B, c, lmn3, C, d, lmn4, D):
    """4-center electron repulsion integral (μν|λσ) for primitive Gaussians."""
    p, P = gaussian_product_center(a, A, b, B)
    q, Q = gaussian_product_center(c, C, d, D)
    alpha = p * q / (p + q)
    PC = P - Q

    # Gaussian product factors
    AB2 = np.dot(A - B, A - B)
    CD2 = np.dot(C - D, C - D)
    gpf = np.exp(-a * b / p * AB2) * np.exp(-c * d / q * CD2)

    # Prefactor for the ERI
    prefactor = 2 * pi ** 2.5 / (p * q * sqrt(p + q)) * gpf

    # PA and PB vectors
    Qx = (P[0] - A[0], P[0] - B[0])
    Qy = (P[1] - A[1], P[1] - B[1])
    Qz = (P[2] - A[2], P[2] - B[2])
    Qx2 = (Q[0] - C[0], Q[0] - D[0])
    Qy2 = (Q[1] - C[1], Q[1] - D[1])
    Qz2 = (Q[2] - C[2], Q[2] - D[2])

    # Sum over Hermite indices
    total = 0.0
    for tx in range(lmn1[0] + lmn2[0] + 1):
        for ty in range(lmn1[1] + lmn2[1] + 1):
            for tz in range(lmn1[2] + lmn2[2] + 1):
                Ex = hermite_coeffs(lmn1[0], lmn2[0], tx, Qx, a, b)
                if abs(Ex) < 1e-15:
                    continue
                Ey = hermite_coeffs(lmn1[1], lmn2[1], ty, Qy, a, b)
                if abs(Ey) < 1e-15:
                    continue
                Ez = hermite_coeffs(lmn1[2], lmn2[2], tz, Qz, a, b)
                if abs(Ez) < 1e-15:
                    continue
                E1 = Ex * Ey * Ez

                for ux in range(lmn3[0] + lmn4[0] + 1):
                    Fx = hermite_coeffs(lmn3[0], lmn4[0], ux, Qx2, c, d)
                    if abs(Fx) < 1e-15:
                        continue
                    for uy in range(lmn3[1] + lmn4[1] + 1):
                        Fy = hermite_coeffs(lmn3[1], lmn4[1], uy, Qy2, c, d)
                        if abs(Fy) < 1e-15:
                            continue
                        for uz in range(lmn3[2] + lmn4[2] + 1):
                            Fz = hermite_coeffs(lmn3[2], lmn4[2], uz, Qz2, c, d)
                            if abs(Fz) < 1e-15:
                                continue
                            E2 = Fx * Fy * Fz

                            # Sign: (-1)^(u+v+w)
                            sign = (-1) ** (ux + uy + uz)
                            R = R_tuv(tx + ux, ty + uy, tz + uz, p, alpha, PC)
                            total += sign * E1 * E2 * R

    return prefactor * total


# ============================================================
# Contracted integrals — combine over STO-3G primitives
# ============================================================


STO3G_PARAMS = {
    "H":  [("s", [(3.42525091, 0.15432897), (0.62391373, 0.53532814), (0.16885540, 0.44463454)])],
    "He": [("s", [(6.36242139, 0.15432897), (1.15892299, 0.53532814), (0.31364979, 0.44463454)])],
    "Li": [
        ("s", [(16.11957477, 0.15432897), (2.93619880, 0.53532814), (0.79465049, 0.44463454)]),
        ("s", [(0.63628976, -0.09996723), (0.17979342, 0.39951283), (0.06965671, 0.70011547)]),
        ("p", [(0.63628976, 0.15591627), (0.17979342, 0.60768372), (0.06965671, 0.39195739)]),
    ],
    "C":  [
        ("s", [(71.6168373, 0.15432897), (13.0450963, 0.53532814), (3.5305122, 0.44463454)]),
        ("s", [(2.9412494, -0.09996723), (0.6834831, 0.39951283), (0.2222899, 0.70011547)]),
        ("p", [(2.9412494, 0.15591627), (0.6834831, 0.60768372), (0.2222899, 0.39195739)]),
    ],
    "N":  [
        ("s", [(99.1061690, 0.15432897), (18.0523120, 0.53532814), (4.8856602, 0.44463454)]),
        ("s", [(3.7804559, -0.09996723), (0.8784966, 0.39951283), (0.2857144, 0.70011547)]),
        ("p", [(3.7804559, 0.15591627), (0.8784966, 0.60768372), (0.2857144, 0.39195739)]),
    ],
    "O":  [
        ("s", [(130.7093200, 0.15432897), (23.8088610, 0.53532814), (6.4436083, 0.44463454)]),
        ("s", [(5.0331513, -0.09996723), (1.1695961, 0.39951283), (0.3803890, 0.70011547)]),
        ("p", [(5.0331513, 0.15591627), (1.1695961, 0.60768372), (0.3803890, 0.39195739)]),
    ],
    "F":  [
        ("s", [(166.6791300, 0.15432897), (30.3608120, 0.53532814), (8.2168207, 0.44463454)]),
        ("s", [(6.4648032, -0.09996723), (1.5022812, 0.39951283), (0.4885885, 0.70011547)]),
        ("p", [(6.4648032, 0.15591627), (1.5022812, 0.60768372), (0.4885885, 0.39195739)]),
    ],
}

ANG_TO_BOHR = 1.8897259886


def _make_basis(molecule):
    """Build the contracted basis set for a molecule."""
    basis = []
    for atom in molecule["atoms"]:
        element = atom["element"]
        center = np.array(atom["coords"], dtype=float) * ANG_TO_BOHR
        if element not in STO3G_PARAMS:
            raise ValueError(f"element {element} not in STO-3G basis")
        for shell_type, primitives in STO3G_PARAMS[element]:
            if shell_type == "s":
                basis.append({"center": center, "lmn": (0, 0, 0),
                              "primitives": primitives, "element": element})
            elif shell_type == "p":
                for axis in [(1,0,0), (0,1,0), (0,0,1)]:
                    basis.append({"center": center, "lmn": axis,
                                  "primitives": primitives, "element": element})
    return basis


def _add_primitive_normalization(basis):
    """Multiply each primitive's coefficient by its s-orbital normalization.

    N(alpha) = (2*alpha/pi)^(3/4) for s orbitals.
    For p orbitals, the normalization is different (N_p = (2*alpha/pi)^(3/4) * sqrt(4*alpha)).
    """
    for bf in basis:
        prims = bf["primitives"]
        lmn = bf["lmn"]
        is_p = any(l > 0 for l in lmn)
        normalized = []
        for (alpha, c) in prims:
            if is_p:
                # p orbital normalization: (2a/pi)^(3/4) * sqrt(2a)
                N = (2 * alpha / np.pi) ** 0.75 * np.sqrt(2 * alpha)
            else:
                # s orbital normalization: (2a/pi)^(3/4)
                N = (2 * alpha / np.pi) ** 0.75
            normalized.append((alpha, c * N))
        bf["primitives"] = normalized


def _renormalize_contraction(basis, S):
    """Divide each basis function by sqrt(S_ii) to make the diagonal 1."""
    for i, bf in enumerate(basis):
        norm = np.sqrt(S[i, i])
        if norm > 1e-15:
            bf["primitives"] = [(a, c / norm) for (a, c) in bf["primitives"]]


def compute_all_integrals(molecule, verbose=False):
    """Compute S, T, V, ERI for a molecule.

    Two-pass: build basis, compute S with primitive normalization,
    renormalize the contraction, recompute all integrals.
    """
    basis = _make_basis(molecule)
    n = len(basis)
    if verbose:
        print(f"  basis size: {n} orbitals")

    # Add primitive normalization to each basis function
    _add_primitive_normalization(basis)

    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            for (a, ca) in basis[i]["primitives"]:
                for (b, cb) in basis[j]["primitives"]:
                    norm = ca * cb
                    S[i, j] += norm * overlap_3d(a, basis[i]["lmn"], basis[i]["center"],
                                                  b, basis[j]["lmn"], basis[j]["center"])
                    T[i, j] += norm * kinetic_3d(a, basis[i]["lmn"], basis[i]["center"],
                                                  b, basis[j]["lmn"], basis[j]["center"])
                    for atom in molecule["atoms"]:
                        C = np.array(atom["coords"], dtype=float) * ANG_TO_BOHR
                        V[i, j] += norm * nuclear_3d(
                            a, basis[i]["lmn"], basis[i]["center"],
                            b, basis[j]["lmn"], basis[j]["center"],
                            C, atom["Z"]
                        )

    # ERI: O(n^4)
    ERI = np.zeros((n, n, n, n))
    count = 0
    total = n ** 4
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    total_v = 0.0
                    for (a, ca) in basis[i]["primitives"]:
                        for (b, cb) in basis[j]["primitives"]:
                            for (c, cc) in basis[k]["primitives"]:
                                for (d, cd) in basis[l]["primitives"]:
                                    total_v += ca * cb * cc * cd * eri_primitive(
                                        a, basis[i]["lmn"], basis[i]["center"],
                                        b, basis[j]["lmn"], basis[j]["center"],
                                        c, basis[k]["lmn"], basis[k]["center"],
                                        d, basis[l]["lmn"], basis[l]["center"],
                                    )
                    ERI[i, j, k, l] = total_v
                    count += 1
                    if verbose and count % (total // 10 + 1) == 0:
                        print(f"    ERI progress: {100*count//total}%")

    # Renormalize each contracted function
    _renormalize_contraction(basis, S)

    # Recompute with renormalized basis (second pass)
    for i in range(n):
        for j in range(n):
            S[i, j] = 0.0
            T[i, j] = 0.0
            V[i, j] = 0.0
            for (a, ca) in basis[i]["primitives"]:
                for (b, cb) in basis[j]["primitives"]:
                    norm = ca * cb
                    S[i, j] += norm * overlap_3d(a, basis[i]["lmn"], basis[i]["center"],
                                                  b, basis[j]["lmn"], basis[j]["center"])
                    T[i, j] += norm * kinetic_3d(a, basis[i]["lmn"], basis[i]["center"],
                                                  b, basis[j]["lmn"], basis[j]["center"])
                    for atom in molecule["atoms"]:
                        C = np.array(atom["coords"], dtype=float) * ANG_TO_BOHR
                        V[i, j] += norm * nuclear_3d(
                            a, basis[i]["lmn"], basis[i]["center"],
                            b, basis[j]["lmn"], basis[j]["center"],
                            C, atom["Z"]
                        )

    # Recompute ERI with renormalized basis
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    total_v = 0.0
                    for (a, ca) in basis[i]["primitives"]:
                        for (b, cb) in basis[j]["primitives"]:
                            for (c, cc) in basis[k]["primitives"]:
                                for (d, cd) in basis[l]["primitives"]:
                                    total_v += ca * cb * cc * cd * eri_primitive(
                                        a, basis[i]["lmn"], basis[i]["center"],
                                        b, basis[j]["lmn"], basis[j]["center"],
                                        c, basis[k]["lmn"], basis[k]["center"],
                                        d, basis[l]["lmn"], basis[l]["center"],
                                    )
                    ERI[i, j, k, l] = total_v

    # Renormalize each contracted function
    _renormalize_contraction(basis, S)

    # Recompute with renormalized basis (second pass)
    for i in range(n):
        for j in range(n):
            S[i, j] = 0.0
            T[i, j] = 0.0
            V[i, j] = 0.0
            for (a, ca) in basis[i]["primitives"]:
                for (b, cb) in basis[j]["primitives"]:
                    norm = ca * cb
                    S[i, j] += norm * overlap_3d(a, basis[i]["lmn"], basis[i]["center"],
                                                  b, basis[j]["lmn"], basis[j]["center"])
                    T[i, j] += norm * kinetic_3d(a, basis[i]["lmn"], basis[i]["center"],
                                                  b, basis[j]["lmn"], basis[j]["center"])
                    for atom in molecule["atoms"]:
                        C = np.array(atom["coords"], dtype=float) * ANG_TO_BOHR
                        V[i, j] += norm * nuclear_3d(
                            a, basis[i]["lmn"], basis[i]["center"],
                            b, basis[j]["lmn"], basis[j]["center"],
                            C, atom["Z"]
                        )

    # Recompute ERI with renormalized basis
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    total_v = 0.0
                    for (a, ca) in basis[i]["primitives"]:
                        for (b, cb) in basis[j]["primitives"]:
                            for (c, cc) in basis[k]["primitives"]:
                                for (d, cd) in basis[l]["primitives"]:
                                    total_v += ca * cb * cc * cd * eri_primitive(
                                        a, basis[i]["lmn"], basis[i]["center"],
                                        b, basis[j]["lmn"], basis[j]["center"],
                                        c, basis[k]["lmn"], basis[k]["center"],
                                        d, basis[l]["lmn"], basis[l]["center"],
                                    )
                    ERI[i, j, k, l] = total_v

    return {"S": S, "T": T, "V": V, "ERI": ERI, "basis": basis, "n": n}
