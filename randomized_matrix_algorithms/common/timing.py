"""Timing utilities for experiments.

These helpers provide simple, explicit control over offline vs online timing.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Generator, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class TimerResult:
    """Result of a timed execution block.

    Attributes
    ----------
    seconds : float
        Elapsed wall-clock time in seconds.
    """

    seconds: float


@contextlib.contextmanager
def timer() -> Generator[TimerResult, None, None]:
    """Context manager for wall-clock timing.

    Example
    -------
    >>> with timer() as t:
    ...     do_work()
    >>> print(t.seconds)
    """

    start = time.perf_counter()
    result = TimerResult(seconds=0.0)
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.seconds = float(end - start)


def time_function(func: Callable[[], T]) -> Tuple[T, TimerResult]:
    """Time a zero-argument function and return its result and timing.

    Parameters
    ----------
    func:
        Callable with no arguments.

    Returns
    -------
    (result, TimerResult)
        The function's return value and a ``TimerResult`` structure.
    """

    with timer() as t:
        value = func()
    return value, t
