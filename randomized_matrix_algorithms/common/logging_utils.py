"""Logging helpers for randomized matrix experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str = "randomized_matrix_algorithms") -> logging.Logger:
    """Return a configured logger for the project.

    Parameters
    ----------
    name:
        Logger name; defaults to a shared project logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL (one-JSON-per-line) file.

    Parameters
    ----------
    path:
        Destination file path.
    record:
        Mapping to be serialized as JSON on a single line.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")
