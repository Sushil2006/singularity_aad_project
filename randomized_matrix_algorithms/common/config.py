"""Configuration dataclasses for experiments.

These provide typed containers for experiment parameters so that runners and
plotting scripts share a common schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class MatrixFamily(str, Enum):
    """Supported synthetic / real matrix families."""

    GAUSSIAN = "gaussian"
    LOW_RANK = "low_rank"
    SPARSE = "sparse"
    NN_LIKE_SYNTHETIC = "nn_like_synthetic"
    NN_LIKE_REAL = "nn_like_real"
    RECSYS_SYNTHETIC = "recsys_synthetic"
    RECSYS_REAL = "recsys_real"


class MethodKind(str, Enum):
    """High-level method categories for comparisons."""

    GEMM = "gemm"
    STRASSEN = "strassen"
    RMM_UNIFORM = "rmm_uniform"
    RMM_IMPORTANCE = "rmm_importance"
    RSVD = "rsvd"
    LOW_RANK_GEMM_RSVD = "low_rank_gemm_rsvd"
    LOW_RANK_GEMM_DET = "low_rank_gemm_det"


@dataclass
class MatrixSize:
    """Matrix size triple (m, n, p) for products A(m×n) @ B(n×p)."""

    m: int
    n: int
    p: int


@dataclass
class RmmConfig:
    """Configuration parameters for an RMM experiment sweep."""

    sampling_ratios: List[float]
    num_trials: int
    seed: int


@dataclass
class LowRankGemmConfig:
    """Configuration parameters for low-rank GEMM experiments."""

    ranks: List[int]
    num_trials: int
    seed: int


@dataclass
class RsvdConfig:
    """Configuration parameters for RSVD experiments."""

    ranks: List[int]
    oversampling: int
    power_iters: List[int]
    num_trials: int
    seed: int


@dataclass
class ExperimentConfig:
    """Top-level configuration for a single experiment suite.

    This ties together matrix sizes, matrix families, and method-specific
    parameter sweeps.
    """

    sizes: List[MatrixSize]
    families: List[MatrixFamily]
    rmm: RmmConfig
    low_rank_gemm: LowRankGemmConfig
    rsvd: RsvdConfig
