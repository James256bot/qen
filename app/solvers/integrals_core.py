import numpy as np
from math import factorial, pi, sqrt, erfc


# ============================================================
# STO-3G integral foundation — corrected version
# ============================================================


def boys(n, x):
    """Boys function F_n(x) = ∫₀¹ t^(2n) exp(-x t²) dt.

    Taylor for small x, asymptotic for large x.
    """
    if x < 1e-10:
        return 1.0 / (2 * n + 1)
    if x < 30.0:
        # Taylor series (accurate to ~30)
        term = 1.0 / (2 * n + 1)
        total = term
        for k in range(1, 200):
            term *= 2 * x / (2 * n + 2 * k + 1)
            total += term
            if abs(term) < 1e-15 * abs(total):
                break
        return total * np.exp(-x)
    else:
        # Asymptotic for large x
        F0 = 0.5 * sqrt(pi / x) * erfc(sqrt(x))
        F = [F0]
        for k in range(1, n + 1):
            F.append(((2 * k - 1) * F[-1] - np.exp(-x)) / (2 * x))
        return F[n]


def gaussian_product_center(a, A, b, B):
    """Center P and exponent p of the Gaussian product."""
    p = a + b
    P = (a * A + b * B) / p
    return p, P


def _binomial(a, b):
    from math import comb
    return comb(a, b)


def overlap_1d(l1, l2, PA, PB, p):
    """1D moment: ∫ (x-A)^l1 (x-B)^l2 exp(-p x²) dx.

    This is the "unnormalized" overlap along one axis. Returns a finite value.
    The full 3D overlap requires multiplying by the prefactor (π/p)^1.5
    AND a Gaussian product distance factor.
    """
    total = 0.0
    for i in range(l1 + 1):
        for j in range(l2 + 1):
            coef = (factorial(l1) // (factorial(i) * factorial(l1 - i))
                    * factorial(l2) // (factorial(j) * factorial(l2 - j))
                    * PA ** (l1 - i) * PB ** (l2 - j))
            k = i + j
            if k % 2 == 0:
                if k == 0:
                    val = sqrt(pi / p)
                else:
                    double_fact = 1
                    for m in range(1, k // 2 + 1):
                        double_fact *= (2 * m - 1)
                    val = double_fact * sqrt(pi) / (2 ** (k // 2) * p ** (k // 2 + 0.5))
                total += coef * val
    return total


def overlap_3d(a, lmn1, A, b, lmn2, B):
    """3D overlap of two primitive Gaussians."""
    p, P = gaussian_product_center(a, A, b, B)
    AB2 = np.dot(A - B, A - B)
    # Gaussian product factor
    gpf = np.exp(-a * b / p * AB2)
    Sx = overlap_1d(lmn1[0], lmn2[0], P[0] - A[0], P[0] - B[0], p)
    Sy = overlap_1d(lmn1[1], lmn2[1], P[1] - A[1], P[1] - B[1], p)
    Sz = overlap_1d(lmn1[2], lmn2[2], P[2] - A[2], P[2] - B[2], p)
    return gpf * Sx * Sy * Sz


def kinetic_3d(a, lmn1, A, b, lmn2, B):
    """3D kinetic energy of two primitive Gaussians.

    For s orbitals (lmn1 = lmn2 = (0,0,0)):
        T = S · (ab/p) · (3 − 2·ab/p·|A-B|²)

    where S = overlap integral.

    For p and higher, we fall back to the general component formula.
    """
    # s-s case: use the closed form
    if lmn1 == (0, 0, 0) and lmn2 == (0, 0, 0):
        p, P = gaussian_product_center(a, A, b, B)
        AB2 = np.dot(A - B, A - B)
        S = overlap_3d(a, lmn1, A, b, lmn2, B)
        return S * (a * b / p) * (3.0 - 2.0 * a * b / p * AB2)

    # For p orbitals and beyond, use the component form (fallback)
    p, P = gaussian_product_center(a, A, b, B)
    AB2 = np.dot(A - B, A - B)
    gpf = np.exp(-a * b / p * AB2)
    PA = P - A
    PB = P - B

    def T_1d(l1, l2, PA_comp, PB_comp):
        T = 0.0
        T += a * b / p * overlap_1d(l1, l2, PA_comp, PB_comp, p)
        if l1 >= 1:
            T -= a * l1 / (2 * p) * overlap_1d(l1 - 1, l2, PA_comp, PB_comp, p)
        if l2 >= 1:
            T -= b * l2 / (2 * p) * overlap_1d(l1, l2 - 1, PA_comp, PB_comp, p)
        T += a * a / 2 * overlap_1d(l1 + 2, l2, PA_comp, PB_comp, p)
        T += b * b / 2 * overlap_1d(l1, l2 + 2, PA_comp, PB_comp, p)
        if l1 >= 2:
            T -= a * a * l1 * (l1 - 1) / 2 * overlap_1d(l1 - 2, l2, PA_comp, PB_comp, p)
        if l2 >= 2:
            T -= b * b * l2 * (l2 - 1) / 2 * overlap_1d(l1, l2 - 2, PA_comp, PB_comp, p)
        return T

    Tx = T_1d(lmn1[0], lmn2[0], PA[0], PB[0])
    Ty = T_1d(lmn1[1], lmn2[1], PA[1], PB[1])
    Tz = T_1d(lmn1[2], lmn2[2], PA[2], PB[2])
    return gpf * (Tx + Ty + Tz)


def nuclear_3d(a, lmn1, A, b, lmn2, B, C, Z):
    """Nuclear attraction integral. Only handles s orbitals for now."""
    p, P = gaussian_product_center(a, A, b, B)
    AB2 = np.dot(A - B, A - B)
    PC2 = np.dot(P - C, P - C)
    return -Z * 2 * pi / p * np.exp(-a * b / p * AB2) * boys(0, p * PC2)
