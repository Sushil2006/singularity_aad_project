"""Lightweight RSVD tests / sanity checks.

These are intentionally small so they can run quickly inside the project
without external test frameworks. They complement the self-test in
``core.py`` and mirror Phase 5 requirements.
"""

from __future__ import annotations

import numpy as np

from randomized_matrix_algorithms.rsvd.core import rsvd, truncated_svd


def test_low_rank_recovery() -> None:
    """RSVD should closely match truncated SVD on a numerically low-rank matrix."""

    rng = np.random.default_rng(0)
    m, n, r_true = 80, 60, 10
    u_true, _ = np.linalg.qr(rng.normal(size=(m, r_true)))
    v_true, _ = np.linalg.qr(rng.normal(size=(n, r_true)))
    singulars = np.linspace(5.0, 1.0, num=r_true)
    a = (u_true * singulars[np.newaxis, :]) @ v_true.T
    a = a + 0.01 * rng.normal(size=a.shape)

    exact = truncated_svd(a, k=r_true)
    approx = rsvd(a, k=r_true, p=5, q=1, seed=1)

    err = np.linalg.norm(exact.as_matrix() - approx.as_matrix(), ord="fro") / np.linalg.norm(
        exact.as_matrix(), ord="fro"
    )
    assert err < 0.05, f"RSVD error too large on low-rank matrix: {err:.4f}"


def test_power_iteration_improves_error() -> None:
    """Adding power iterations should not worsen error on slow-decay spectra."""

    rng = np.random.default_rng(1)
    m, n = 120, 80
    # Slow decay spectrum to emphasize power-iteration benefit.
    u_true, _ = np.linalg.qr(rng.normal(size=(m, n)))
    v_true, _ = np.linalg.qr(rng.normal(size=(n, n)))
    singulars = np.linspace(5.0, 0.5, num=n)
    a = (u_true * singulars[np.newaxis, :]) @ v_true.T

    exact = truncated_svd(a, k=20)
    base = rsvd(a, k=20, p=5, q=0, seed=2)
    boosted = rsvd(a, k=20, p=5, q=1, seed=2)

    err_base = np.linalg.norm(exact.as_matrix() - base.as_matrix(), ord="fro") / np.linalg.norm(
        exact.as_matrix(), ord="fro"
    )
    err_boosted = np.linalg.norm(exact.as_matrix() - boosted.as_matrix(), ord="fro") / np.linalg.norm(
        exact.as_matrix(), ord="fro"
    )

    assert err_boosted <= err_base + 1e-8, f"Power iteration failed to improve error: base={err_base:.4f}, boosted={err_boosted:.4f}"


if __name__ == "__main__":  # pragma: no cover - manual execution
    test_low_rank_recovery()
    test_power_iteration_improves_error()
    print("RSVD tests passed.")
