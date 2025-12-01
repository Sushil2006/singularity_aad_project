"""Core randomized matrix multiplication (RMM) algorithms.

Implements outer-product sampling estimators for ``C = A @ B`` with:

- Uniform sampling over columns of ``A`` / rows of ``B``.
- Importance sampling with probabilities proportional to ``||a_k|| * ||b_k||``.

All routines operate on NumPy arrays and raise explicit errors on
shape/dtype mismatches or degenerate probability distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass
class RmmResult:
    """Container for the result of a randomized matrix multiplication.

    Attributes
    ----------
    estimate : ndarray
        The approximate product ``\tilde{C}``.
    indices : ndarray
        One-dimensional array of sampled column/row indices ``k_i`` of
        length ``s``.
    probabilities : ndarray
        One-dimensional array of probabilities ``p_k`` used for sampling,
        of length ``n``.
    """

    estimate: FloatArray
    indices: np.ndarray
    probabilities: np.ndarray


def _validate_inputs(a: FloatArray, b: FloatArray) -> None:
    """Validate that ``a`` and ``b`` can be multiplied.

    Parameters
    ----------
    a, b:
        Two-dimensional arrays with shapes ``(m, n)`` and ``(n, p)``.

    Raises
    ------
    ValueError
        If shapes are incompatible or inputs are not 2D.
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("RMM expects 2D arrays for a and b")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"incompatible shapes for matmul: a{a.shape}, b{b.shape} (a.shape[1] != b.shape[0])"
        )


def rmm_uniform(
    a: FloatArray,
    b: FloatArray,
    num_samples: int,
    seed: Optional[int] = None,
) -> RmmResult:
    r"""Randomized matrix multiplication with **uniform** sampling.

    This implements the estimator

    .. math::

        \tilde{C} = \frac{1}{s} \sum_{i=1}^s \frac{1}{p_{k_i}} a_{k_i} b_{k_i}^T,

    where ``p_k = 1 / n`` for all columns/rows and ``a_k`` is the ``k``-th
    column of ``A`` and ``b_k^T`` is the ``k``-th row of ``B``.

    Parameters
    ----------
    a, b:
        Input matrices with shapes ``(m, n)`` and ``(n, p)``.
    num_samples:
        Number of outer products to sample (``s``). Must be positive.
    seed:
        Optional random seed.

    Returns
    -------
    RmmResult
        Approximate product and sampling metadata.
    """

    _validate_inputs(a, b)
    m, n = a.shape
    n_b, p = b.shape
    assert n == n_b

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    rng = np.random.default_rng(seed)

    # Sample indices k_i uniformly from {0, ..., n-1} with replacement.
    indices = rng.integers(low=0, high=n, size=num_samples, endpoint=False)

    # p_k = 1/n for all k, so the estimator reduces to:
    #  C_hat = (n/s) * sum_i a_{k_i} b_{k_i}^T
    # Note that sum_i a_{k_i} b_{k_i}^T = A_sub @ B_sub where
    #  A_sub = A[:, ks], B_sub = B[ks, :].
    a_sub = a[:, indices]  # (m, s)
    b_sub = b[indices, :]  # (s, p)

    factor = float(n) / float(num_samples)
    estimate = (factor * (a_sub @ b_sub)).astype(np.float64)

    probabilities = np.full(shape=(n,), fill_value=1.0 / float(n), dtype=np.float64)

    return RmmResult(estimate=estimate, indices=indices, probabilities=probabilities)


def rmm_importance(
    a: FloatArray,
    b: FloatArray,
    num_samples: int,
    seed: Optional[int] = None,
) -> RmmResult:
    r"""Randomized matrix multiplication with **importance sampling**.

    Probabilities are chosen as

    .. math::

        p_k = \frac{\|a_k\| \; \|b_k\|}{\sum_j \|a_j\| \; \|b_j\|},

    which (in theory) minimizes the variance of the estimator among all
    choices that depend only on column/row norms.

    Parameters
    ----------
    a, b:
        Input matrices with shapes ``(m, n)`` and ``(n, p)``.
    num_samples:
        Number of outer products to sample (``s``). Must be positive.
    seed:
        Optional random seed.

    Returns
    -------
    RmmResult
        Approximate product and sampling metadata.

    Raises
    ------
    ValueError
        If all columns/rows are zero (making the importance distribution
        undefined) or shapes are incompatible.
    """

    _validate_inputs(a, b)
    m, n = a.shape
    n_b, p = b.shape
    assert n == n_b

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    # Compute norms for columns of A and rows of B.
    col_norms_a = np.linalg.norm(a, axis=0)
    row_norms_b = np.linalg.norm(b, axis=1)

    weights = col_norms_a * row_norms_b
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "cannot construct importance distribution: all column/row norms are zero or non-finite",
        )

    probabilities = (weights / total_weight).astype(np.float64)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_samples, replace=True, p=probabilities)

    # Estimator: (1/s) * sum_i (1 / p_{k_i}) a_{k_i} b_{k_i}^T.
    # In matrix form, let A_sub = A[:, ks], B_sub = B[ks, :], and
    # diag_s = diag(1 / p_{k_i}). Then
    #   sum_i (1 / p_{k_i}) a_{k_i} b_{k_i}^T = A_sub @ diag_s @ B_sub.
    # We implement this by scaling the columns of A_sub.
    a_sub = a[:, indices]  # (m, s)
    b_sub = b[indices, :]  # (s, p)

    inv_p = 1.0 / probabilities[indices]
    a_weighted = a_sub * inv_p[np.newaxis, :]  # scale each column of A_sub

    estimate = ((a_weighted @ b_sub) / float(num_samples)).astype(np.float64)

    return RmmResult(estimate=estimate, indices=indices, probabilities=probabilities)


def _self_test(num_trials: int = 200, seed: int = 0) -> None:
    """Run a small Monte Carlo self-test for the RMM estimators.

    The test checks empirically that both uniform and importance-sampled
    estimators are approximately unbiased for a small problem.

    Parameters
    ----------
    num_trials:
        Number of Monte Carlo runs for each estimator.
    seed:
        Random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)
    m, n, p = 6, 5, 4
    a = rng.normal(size=(m, n))
    b = rng.normal(size=(n, p))
    exact = (a @ b).astype(np.float64)

    s = max(2, n // 2)

    # Uniform estimator
    uniform_sum = np.zeros_like(exact)
    for t in range(num_trials):
        res = rmm_uniform(a, b, num_samples=s, seed=int(rng.integers(0, 1_000_000)))
        uniform_sum += res.estimate
    uniform_mean = uniform_sum / float(num_trials)
    uniform_err = np.linalg.norm(uniform_mean - exact, ord="fro") / np.linalg.norm(
        exact, ord="fro"
    )

    # Importance estimator
    importance_sum = np.zeros_like(exact)
    for t in range(num_trials):
        res = rmm_importance(a, b, num_samples=s, seed=int(rng.integers(0, 1_000_000)))
        importance_sum += res.estimate
    importance_mean = importance_sum / float(num_trials)
    importance_err = np.linalg.norm(
        importance_mean - exact, ord="fro"
    ) / np.linalg.norm(exact, ord="fro")

    assert uniform_err < 0.1, f"uniform RMM mean error too large: {uniform_err!r}"
    assert (
        importance_err < 0.1
    ), f"importance RMM mean error too large: {importance_err!r}"


if __name__ == "__main__":  # pragma: no cover - manual quick check
    _self_test()
