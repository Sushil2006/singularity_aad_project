"""Metric utilities for evaluating randomized matrix algorithms.

All functions here work purely on NumPy arrays and are side-effect free.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def relative_frobenius_error(true: FloatArray, approx: FloatArray) -> float:
    """Compute relative Frobenius norm error ``||true - approx||_F / ||true||_F``.

    Parameters
    ----------
    true:
        Ground-truth matrix.
    approx:
        Approximate matrix with the same shape as ``true``.

    Returns
    -------
    float
        Relative Frobenius norm error.
    """

    if true.shape != approx.shape:
        raise ValueError("shapes of true and approx must match")

    diff = true - approx
    num = np.linalg.norm(diff, ord="fro")
    denom = np.linalg.norm(true, ord="fro")
    if denom == 0.0:
        if num == 0.0:
            return 0.0
        raise ValueError("cannot compute relative error: true has zero Frobenius norm")
    return float(num / denom)


def spectral_norm_error_power_iteration(
    true: FloatArray,
    approx: FloatArray,
    max_iters: int = 50,
    tol: float = 1e-6,
    seed: int | None = None,
) -> float:
    """Approximate spectral-norm error ``||true - approx||_2`` by power iteration.

    Parameters
    ----------
    true, approx:
        Matrices with identical shapes.
    max_iters:
        Maximum number of power-iteration steps.
    tol:
        Convergence tolerance on successive Rayleigh quotient changes.
    seed:
        Random seed for initial vector.

    Returns
    -------
    float
        Approximate spectral norm of ``true - approx``.
    """

    if true.shape != approx.shape:
        raise ValueError("shapes of true and approx must match")

    rng = np.random.default_rng(seed)
    diff = (true - approx).astype(np.float64, copy=False)
    m, n = diff.shape

    # Power iteration on diff^T diff.
    v = rng.normal(size=(n,))
    v /= np.linalg.norm(v)

    prev_val = 0.0
    for _ in range(max_iters):
        w = diff.T @ (diff @ v)
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            return 0.0
        v = w / norm_w
        rayleigh = float(v @ (diff.T @ (diff @ v)))
        if abs(rayleigh - prev_val) <= tol * max(1.0, abs(prev_val)):
            break
        prev_val = rayleigh

    return float(np.sqrt(max(prev_val, 0.0)))


def topk_overlap(
    baseline_scores: FloatArray,
    approx_scores: FloatArray,
    k: int,
) -> float:
    """Compute average top-k overlap between baseline and approximate scores.

    This metric is appropriate for recommender-style workloads where rows
    correspond to entities (e.g., users) and columns to items.

    Parameters
    ----------
    baseline_scores:
        Matrix of shape ``(num_entities, num_items)`` containing baseline
        scores.
    approx_scores:
        Matrix of the same shape containing approximate scores.
    k:
        Top-k cutoff (``1 <= k <= num_items``).

    Returns
    -------
    float
        Average fraction of overlap between the baseline and approximate
        top-k sets across all entities.
    """

    if baseline_scores.shape != approx_scores.shape:
        raise ValueError("score matrices must have the same shape")

    num_entities, num_items = baseline_scores.shape
    if not 1 <= k <= num_items:
        raise ValueError("k must be between 1 and num_items inclusive")

    overlap_sum = 0.0
    for u in range(num_entities):
        base_row = baseline_scores[u]
        approx_row = approx_scores[u]

        base_topk_idx = np.argpartition(-base_row, kth=k - 1)[:k]
        approx_topk_idx = np.argpartition(-approx_row, kth=k - 1)[:k]

        base_set = set(int(i) for i in base_topk_idx)
        approx_set = set(int(i) for i in approx_topk_idx)
        inter = base_set.intersection(approx_set)
        overlap_sum += len(inter) / float(k)

    return overlap_sum / float(num_entities)


def accuracy_delta(baseline_acc: float, approx_acc: float) -> float:
    """Compute ``approx_acc - baseline_acc`` for convenience.

    Parameters
    ----------
    baseline_acc:
        Baseline accuracy (e.g., with exact GEMM).
    approx_acc:
        Accuracy achieved when approximate methods are used.

    Returns
    -------
    float
        Difference ``approx_acc - baseline_acc``.
    """

    return float(approx_acc - baseline_acc)
