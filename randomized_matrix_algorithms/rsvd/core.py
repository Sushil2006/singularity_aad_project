"""Randomized SVD (RSVD) core algorithms.

Implements from-scratch RSVD following randomized_SVD/plan.md:
- Random test matrix
- Optional power iterations
- QR to form an orthonormal basis
- Small SVD on the projected matrix

All logic is written against NumPy arrays with explicit validation and
reproducibility via optional seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass
class RsvdResult:
    """Container for RSVD outputs."""

    u: FloatArray
    s: FloatArray
    vt: FloatArray

    @property
    def rank(self) -> int:
        """Rank of the returned approximation."""

        return int(self.s.shape[0])

    def as_matrix(self) -> FloatArray:
        """Reconstruct the low-rank approximation ``U diag(S) Vt``."""

        return (self.u * self.s[np.newaxis, :]) @ self.vt


def _validate_inputs(a: FloatArray, k: int, p: int, q: int) -> Tuple[int, int]:
    """Validate inputs and return useful dimensions (m, n)."""

    if a.ndim != 2:
        raise ValueError("RSVD expects a 2D array")
    m, n = a.shape
    if k <= 0:
        raise ValueError("target rank k must be positive")
    if k > min(m, n):
        raise ValueError("target rank k cannot exceed min(m, n)")
    if p < 0:
        raise ValueError("oversampling p must be non-negative")
    if q < 0:
        raise ValueError("power iterations q must be non-negative")
    return m, n


def rsvd(
    a: FloatArray,
    k: int,
    p: int = 10,
    q: int = 0,
    seed: Optional[int] = None,
) -> RsvdResult:
    """Compute a randomized rank-``k`` SVD approximation of ``a``.

    Parameters
    ----------
    a:
        Input matrix of shape ``(m, n)``.
    k:
        Target rank (``1 <= k <= min(m, n)``).
    p:
        Oversampling parameter (default 10). The sketch size is ``k + p``.
    q:
        Number of power iterations to improve accuracy for slow singular
        value decay. Each power step applies ``A`` and ``A.T`` once.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    RsvdResult
        Low-rank factors ``(U_k, S_k, Vt_k)`` where ``U_k`` has shape
        ``(m, k)``, ``S_k`` has shape ``(k,)``, and ``Vt_k`` has shape
        ``(k, n)``.
    """

    m, n = _validate_inputs(a, k, p, q)
    ell = k + p
    rng = np.random.default_rng(seed)

    omega = rng.normal(size=(n, ell))
    y = a @ omega  # (m, ell)

    # Optional power iterations to accentuate dominant singular directions.
    for _ in range(q):
        y = a @ (a.T @ y)

    q_mat, _ = np.linalg.qr(y, mode="reduced")  # (m, ell)

    b_small = q_mat.T @ a  # (ell, n)
    u_tilde, s_vals, vt = np.linalg.svd(b_small, full_matrices=False)

    u_full = q_mat @ u_tilde  # (m, ell)

    u_k = u_full[:, :k].astype(np.float64, copy=False)
    s_k = s_vals[:k].astype(np.float64, copy=False)
    vt_k = vt[:k, :].astype(np.float64, copy=False)

    return RsvdResult(u=u_k, s=s_k, vt=vt_k)


def truncated_svd(a: FloatArray, k: int) -> RsvdResult:
    """Deterministic truncated SVD wrapper for comparison baselines."""

    if a.ndim != 2:
        raise ValueError("truncated_svd expects a 2D array")
    if k <= 0 or k > min(a.shape):
        raise ValueError("k must be in [1, min(m, n)]")

    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return RsvdResult(u=u[:, :k].astype(np.float64), s=s[:k].astype(np.float64), vt=vt[:k, :].astype(np.float64))


def _self_test(seed: int = 0) -> None:
    """Lightweight correctness check comparing RSVD to exact SVD."""

    rng = np.random.default_rng(seed)
    m, n = 80, 50
    # Construct a numerically low-rank matrix with known spectrum.
    u_true, _ = np.linalg.qr(rng.normal(size=(m, 10)))
    v_true, _ = np.linalg.qr(rng.normal(size=(n, 10)))
    singulars = np.linspace(10.0, 1.0, num=10)
    a = (u_true[:, :10] * singulars[np.newaxis, :]) @ v_true[:, :10].T
    a = a + 0.01 * rng.normal(size=a.shape)

    exact = truncated_svd(a, k=10)
    approx = rsvd(a, k=10, p=5, q=1, seed=seed + 1)

    # Relative Frobenius error should be small.
    diff = np.linalg.norm(exact.as_matrix() - approx.as_matrix(), ord="fro") / np.linalg.norm(
        exact.as_matrix(), ord="fro"
    )
    assert diff < 0.05, f"RSVD self-test failed: relative error {diff:.4f}"


if __name__ == "__main__":  # pragma: no cover - manual quick check
    _self_test()
