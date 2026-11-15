"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys


class _DynamicStderrHandler(logging.StreamHandler):
    """StreamHandler resolving sys.stderr dynamically on each emit."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root package logger with consistent formatting."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logger = logging.getLogger("chicago_crimes")
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        handler = _DynamicStderrHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(formatter)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the `chicago_crimes` namespace."""
    return logging.getLogger(f"chicago_crimes.{name}")
