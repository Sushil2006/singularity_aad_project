"""Baseline matrix multiplication algorithms.

Provides wrappers for exact GEMM (NumPy matmul) and a simple Strassen
implementation for square matrices.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def gemm_baseline(a: FloatArray, b: FloatArray) -> FloatArray:
    """Compute the exact product ``C = A @ B`` using NumPy's optimized BLAS.

    Parameters
    ----------
    a, b:
        Input arrays with shapes ``(m, n)`` and ``(n, p)``.

    Returns
    -------
    ndarray
        Exact product with shape ``(m, p)``.
    
    Note
    ----
    This uses NumPy's highly optimized BLAS-backed matmul. For fair comparison
    with from-scratch RMM, use ``naive_matmul`` instead.
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError("inner dimensions of a and b are incompatible")
    return (a @ b).astype(np.float64)


def naive_matmul(a: FloatArray, b: FloatArray) -> FloatArray:
    """Compute the exact product ``C = A @ B`` using naive triple-loop.

    This is a from-scratch implementation for fair comparison with RMM.
    Uses outer-product formulation (same as RMM but with all n terms).

    Parameters
    ----------
    a, b:
        Input arrays with shapes ``(m, n)`` and ``(n, p)``.

    Returns
    -------
    ndarray
        Exact product with shape ``(m, p)``.
    
    Complexity
    ----------
    O(m * n * p) - same as standard matrix multiplication.
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError("inner dimensions of a and b are incompatible")
    
    m, n = a.shape
    n_b, p = b.shape
    
    # Use outer-product formulation: C = sum_{k=1}^{n} a_k * b_k^T
    # This is the same formulation as RMM, but with all n terms instead of s samples
    c = np.zeros((m, p), dtype=np.float64)
    for k in range(n):
        # Outer product: a[:, k] (column) * b[k, :] (row)
        c += np.outer(a[:, k], b[k, :])
    
    return c


def _pad_to_power_of_two(mat: FloatArray) -> Tuple[FloatArray, Tuple[int, int]]:
    """Pad a square matrix to the next power-of-two dimension.

    Parameters
    ----------
    mat:
        Square matrix ``(n, n)``.

    Returns
    -------
    padded, (orig_n, new_n)
        Padded matrix and a tuple containing original and padded sizes.
    """

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square for Strassen padding")

    n = mat.shape[0]
    if n == 0:
        return mat.copy(), (0, 0)

    new_n = 1
    while new_n < n:
        new_n *= 2

    if new_n == n:
        return mat.copy(), (n, n)

    padded = np.zeros((new_n, new_n), dtype=np.float64)
    padded[:n, :n] = mat
    return padded, (n, new_n)


def _strassen_recursive(a: FloatArray, b: FloatArray, threshold: int) -> FloatArray:
    """Internal recursive Strassen implementation.

    Assumes that ``a`` and ``b`` are square matrices with power-of-two size
    and identical shapes.
    """

    n = a.shape[0]
    if n <= threshold:
        return a @ b

    mid = n // 2
    a11 = a[:mid, :mid]
    a12 = a[:mid, mid:]
    a21 = a[mid:, :mid]
    a22 = a[mid:, mid:]

    b11 = b[:mid, :mid]
    b12 = b[:mid, mid:]
    b21 = b[mid:, :mid]
    b22 = b[mid:, mid:]

    m1 = _strassen_recursive(a11 + a22, b11 + b22, threshold)
    m2 = _strassen_recursive(a21 + a22, b11, threshold)
    m3 = _strassen_recursive(a11, b12 - b22, threshold)
    m4 = _strassen_recursive(a22, b21 - b11, threshold)
    m5 = _strassen_recursive(a11 + a12, b22, threshold)
    m6 = _strassen_recursive(a21 - a11, b11 + b12, threshold)
    m7 = _strassen_recursive(a12 - a22, b21 + b22, threshold)

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6

    c = np.empty_like(a)
    c[:mid, :mid] = c11
    c[:mid, mid:] = c12
    c[mid:, :mid] = c21
    c[mid:, mid:] = c22
    return c


def strassen(a: FloatArray, b: FloatArray, threshold: int = 64) -> FloatArray:
    """Compute ``C = A @ B`` using a basic Strassen algorithm.

    Parameters
    ----------
    a, b:
        Square matrices with shapes ``(n, n)``. Non-square inputs are not
        supported by this implementation.
    threshold:
        Size below which the recursion falls back to standard GEMM.

    Returns
    -------
    ndarray
        Product matrix with shape ``(n, n)``.
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D arrays")
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Strassen implementation currently supports only square A, B of the same size")

    padded_a, (orig_n, padded_n) = _pad_to_power_of_two(a.astype(np.float64))
    padded_b, (_, _) = _pad_to_power_of_two(b.astype(np.float64))

    c_padded = _strassen_recursive(padded_a, padded_b, threshold)
    return c_padded[:orig_n, :orig_n]


def _self_test() -> None:
    """Run a small internal self-test for the baselines.

    Raises ``AssertionError`` if any check fails.
    """

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8))
    b = rng.normal(size=(8, 8))

    c_gemm = gemm_baseline(a, b)
    c_strassen = strassen(a, b, threshold=2)
    diff = np.linalg.norm(c_gemm - c_strassen, ord="fro")
    assert diff <= 1e-8, f"Strassen mismatch: Frobenius diff={diff!r}"


if __name__ == "__main__":  # pragma: no cover - manual quick check
    _self_test()
