"""
Centralized Loguru configuration for the tv package.

- Use Loguru for developer-facing logs (debugging, tracing, errors).
- Keep logs structured and avoid printing directly from core logic.
"""

from __future__ import annotations

import os
import sys
from loguru import logger


def setup_logging() -> None:
    """Configure Loguru once for CLI/script runs."""
    logger.remove()

    level = os.getenv("TV_LOG_LEVEL", "INFO").upper()

    # Compact, readable by default; includes time/level/module:line.
    logger.add(
        sys.stderr,
        level=level,
        backtrace=True,
        diagnose=False,
        colorize=True,
        enqueue=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

