"""Two-Sided Low-Rank GEMM module.

Provides RSVD-based two-sided low-rank matrix multiplication for
accelerating GEMM when both operands are approximately low-rank.
"""

from randomized_matrix_algorithms.low_rank_gemm.core import (
    LowRankFactors,
    LowRankGemmResult,
    low_rank_gemm_rsvd,
    low_rank_gemm_deterministic,
    compute_factors_rsvd,
    compute_factors_deterministic,
    factorized_multiply_from_factors,
)

__all__ = [
    "LowRankFactors",
    "LowRankGemmResult",
    "low_rank_gemm_rsvd",
    "low_rank_gemm_deterministic",
    "compute_factors_rsvd",
    "compute_factors_deterministic",
    "factorized_multiply_from_factors",
]
