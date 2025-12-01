"""Two-Sided Low-Rank GEMM core algorithms.

Implements RSVD-based two-sided low-rank GEMM following low_rank_approx_matrix_mul/plan.md:

Given A ∈ R^{m×n} and B ∈ R^{n×p}, we approximate C = AB by:
1. Offline: RSVD(A) → (U_A, Σ_A, V_A^T), RSVD(B) → (U_B, Σ_B, V_B^T)
2. Online: C ≈ U_A · Σ_A · (V_A^T · U_B) · Σ_B · V_B^T

This replaces O(mnp) GEMM with O(N²r + Nr²) operations when r << N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from randomized_matrix_algorithms.rsvd.core import rsvd, truncated_svd, RsvdResult

FloatArray = NDArray[np.floating]


@dataclass
class LowRankFactors:
    """Container for low-rank factorization of a matrix.
    
    Stores U, S, Vt such that A ≈ U @ diag(S) @ Vt.
    """
    u: FloatArray      # (m, r)
    s: FloatArray      # (r,)
    vt: FloatArray     # (r, n)
    
    @property
    def rank(self) -> int:
        """Rank of the factorization."""
        return int(self.s.shape[0])
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the reconstructed matrix."""
        return (self.u.shape[0], self.vt.shape[1])
    
    def as_matrix(self) -> FloatArray:
        """Reconstruct the low-rank approximation U @ diag(S) @ Vt."""
        return (self.u * self.s[np.newaxis, :]) @ self.vt


@dataclass
class LowRankGemmResult:
    """Container for low-rank GEMM outputs."""
    
    estimate: FloatArray          # The approximate product C ≈ AB
    rank: int                     # Rank used for approximation
    factors_a: LowRankFactors     # Low-rank factors of A
    factors_b: LowRankFactors     # Low-rank factors of B
    
    # Timing breakdown (optional, filled by experiments)
    offline_time_a: float = 0.0   # Time to factorize A
    offline_time_b: float = 0.0   # Time to factorize B
    online_time: float = 0.0      # Time for factorized multiply


def _validate_inputs(a: FloatArray, b: FloatArray, r: int) -> Tuple[int, int, int]:
    """Validate inputs and return dimensions (m, n, p)."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Inner dimensions must match: A is {a.shape}, B is {b.shape}")
    
    m, n = a.shape
    _, p = b.shape
    
    if r <= 0:
        raise ValueError("Target rank r must be positive")
    if r > min(m, n, p):
        raise ValueError(f"Target rank r={r} cannot exceed min(m,n,p)={min(m,n,p)}")
    
    return m, n, p


def low_rank_gemm_rsvd(
    a: FloatArray,
    b: FloatArray,
    r: int,
    p: int = 10,
    q: int = 0,
    seed: Optional[int] = None,
) -> LowRankGemmResult:
    """Compute approximate C = AB using RSVD-based two-sided low-rank GEMM.
    
    Parameters
    ----------
    a : FloatArray
        Left matrix of shape (m, n).
    b : FloatArray
        Right matrix of shape (n, p).
    r : int
        Target rank for low-rank approximations.
    p : int
        Oversampling parameter for RSVD (default 10).
    q : int
        Number of power iterations for RSVD (default 0).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    LowRankGemmResult
        Contains the approximate product and factorization details.
    
    Algorithm
    ---------
    Offline:
        1. RSVD(A) → U_A, Σ_A, V_A^T
        2. RSVD(B) → U_B, Σ_B, V_B^T
    
    Online:
        3. M_1 = V_A^T @ U_B  (r × r)
        4. M_2 = Σ_A @ M_1 @ Σ_B  (r × r, diagonal scaling)
        5. T = U_A @ M_2  (m × r)
        6. C̃ = T @ V_B^T  (m × p)
    
    Complexity
    ----------
    Offline: O(mnr + npr) for RSVD
    Online: O(nr² + mr² + mpr) vs O(mnp) for full GEMM
    """
    m, n, p_dim = _validate_inputs(a, b, r)
    
    rng = np.random.default_rng(seed)
    seed_a = int(rng.integers(0, 2**31))
    seed_b = int(rng.integers(0, 2**31))
    
    # Offline: RSVD factorizations
    rsvd_a = rsvd(a, k=r, p=p, q=q, seed=seed_a)
    rsvd_b = rsvd(b, k=r, p=p, q=q, seed=seed_b)
    
    factors_a = LowRankFactors(u=rsvd_a.u, s=rsvd_a.s, vt=rsvd_a.vt)
    factors_b = LowRankFactors(u=rsvd_b.u, s=rsvd_b.s, vt=rsvd_b.vt)
    
    # Online: Factorized multiply
    estimate = _factorized_multiply(factors_a, factors_b)
    
    return LowRankGemmResult(
        estimate=estimate,
        rank=r,
        factors_a=factors_a,
        factors_b=factors_b,
    )


def low_rank_gemm_deterministic(
    a: FloatArray,
    b: FloatArray,
    r: int,
) -> LowRankGemmResult:
    """Compute approximate C = AB using deterministic truncated SVD.
    
    This is a baseline that uses exact truncated SVD instead of RSVD.
    Useful for comparing RSVD-based approach against optimal low-rank.
    
    Parameters
    ----------
    a : FloatArray
        Left matrix of shape (m, n).
    b : FloatArray
        Right matrix of shape (n, p).
    r : int
        Target rank for low-rank approximations.
    
    Returns
    -------
    LowRankGemmResult
        Contains the approximate product and factorization details.
    """
    m, n, p_dim = _validate_inputs(a, b, r)
    
    # Deterministic truncated SVD
    svd_a = truncated_svd(a, k=r)
    svd_b = truncated_svd(b, k=r)
    
    factors_a = LowRankFactors(u=svd_a.u, s=svd_a.s, vt=svd_a.vt)
    factors_b = LowRankFactors(u=svd_b.u, s=svd_b.s, vt=svd_b.vt)
    
    # Factorized multiply
    estimate = _factorized_multiply(factors_a, factors_b)
    
    return LowRankGemmResult(
        estimate=estimate,
        rank=r,
        factors_a=factors_a,
        factors_b=factors_b,
    )


def _factorized_multiply(
    factors_a: LowRankFactors,
    factors_b: LowRankFactors,
) -> FloatArray:
    """Perform the factorized multiply: U_A @ Σ_A @ (V_A^T @ U_B) @ Σ_B @ V_B^T.
    
    This is the "online" computation that exploits low-rank structure.
    
    Parameters
    ----------
    factors_a : LowRankFactors
        Low-rank factors of A: U_A (m×r), S_A (r,), V_A^T (r×n)
    factors_b : LowRankFactors
        Low-rank factors of B: U_B (n×r), S_B (r,), V_B^T (r×p)
    
    Returns
    -------
    FloatArray
        Approximate product C̃ of shape (m, p).
    
    Algorithm Steps
    ---------------
    1. M_1 = V_A^T @ U_B  → (r × r)
    2. M_2 = diag(S_A) @ M_1 @ diag(S_B)  → (r × r)
    3. T = U_A @ M_2  → (m × r)
    4. C̃ = T @ V_B^T  → (m × p)
    """
    # Step 1: Core coupling matrix (r × r)
    m1 = factors_a.vt @ factors_b.u  # (r, n) @ (n, r) = (r, r)
    
    # Step 2: Diagonal scaling (efficient element-wise)
    # diag(S_A) @ M_1 @ diag(S_B) = S_A[:, None] * M_1 * S_B[None, :]
    m2 = factors_a.s[:, np.newaxis] * m1 * factors_b.s[np.newaxis, :]  # (r, r)
    
    # Step 3: Left multiply (m × r)
    t = factors_a.u @ m2  # (m, r) @ (r, r) = (m, r)
    
    # Step 4: Final product (m × p)
    c_approx = t @ factors_b.vt  # (m, r) @ (r, p) = (m, p)
    
    return c_approx.astype(np.float64)


def factorized_multiply_from_factors(
    factors_a: LowRankFactors,
    factors_b: LowRankFactors,
) -> FloatArray:
    """Public wrapper for factorized multiply (for timing experiments).
    
    Use this when you already have pre-computed factors and want to
    measure only the online multiplication time.
    """
    return _factorized_multiply(factors_a, factors_b)


def compute_factors_rsvd(
    mat: FloatArray,
    r: int,
    p: int = 10,
    q: int = 0,
    seed: Optional[int] = None,
) -> LowRankFactors:
    """Compute low-rank factors of a matrix using RSVD.
    
    Parameters
    ----------
    mat : FloatArray
        Input matrix of shape (m, n).
    r : int
        Target rank.
    p : int
        Oversampling parameter.
    q : int
        Power iterations.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    LowRankFactors
        Low-rank factorization U, S, Vt.
    """
    if mat.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if r <= 0 or r > min(mat.shape):
        raise ValueError(f"Rank r={r} must be in [1, min(m,n)={min(mat.shape)}]")
    
    result = rsvd(mat, k=r, p=p, q=q, seed=seed)
    return LowRankFactors(u=result.u, s=result.s, vt=result.vt)


def compute_factors_deterministic(
    mat: FloatArray,
    r: int,
) -> LowRankFactors:
    """Compute low-rank factors of a matrix using deterministic truncated SVD.
    
    Parameters
    ----------
    mat : FloatArray
        Input matrix of shape (m, n).
    r : int
        Target rank.
    
    Returns
    -------
    LowRankFactors
        Low-rank factorization U, S, Vt.
    """
    if mat.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if r <= 0 or r > min(mat.shape):
        raise ValueError(f"Rank r={r} must be in [1, min(m,n)={min(mat.shape)}]")
    
    result = truncated_svd(mat, k=r)
    return LowRankFactors(u=result.u, s=result.s, vt=result.vt)


def _self_test(seed: int = 42) -> None:
    """Lightweight correctness check for low-rank GEMM."""
    rng = np.random.default_rng(seed)
    
    # Create low-rank matrices
    m, n, p = 100, 80, 90
    true_rank = 10
    
    # A = U_A @ S_A @ V_A^T (rank 10)
    u_a, _ = np.linalg.qr(rng.normal(size=(m, true_rank)))
    v_a, _ = np.linalg.qr(rng.normal(size=(n, true_rank)))
    s_a = np.linspace(10.0, 1.0, true_rank)
    a = (u_a * s_a) @ v_a.T
    
    # B = U_B @ S_B @ V_B^T (rank 10)
    u_b, _ = np.linalg.qr(rng.normal(size=(n, true_rank)))
    v_b, _ = np.linalg.qr(rng.normal(size=(p, true_rank)))
    s_b = np.linspace(8.0, 0.5, true_rank)
    b = (u_b * s_b) @ v_b.T
    
    # Exact product
    c_exact = a @ b
    
    # Low-rank GEMM with r = true_rank
    result = low_rank_gemm_rsvd(a, b, r=true_rank, p=5, q=1, seed=seed)
    
    # Check relative error
    rel_error = np.linalg.norm(c_exact - result.estimate, 'fro') / np.linalg.norm(c_exact, 'fro')
    
    assert rel_error < 0.1, f"Low-rank GEMM self-test failed: relative error {rel_error:.4f}"
    print(f"Self-test passed: relative error = {rel_error:.6f}")


if __name__ == "__main__":
    _self_test()
