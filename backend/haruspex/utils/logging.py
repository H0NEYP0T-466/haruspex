"""HARUSPEX logging configuration."""

from __future__ import annotations

import logging
import sys


_LOG_FORMAT = "[HARUSPEX] %(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED:
        logger.setLevel(level)

        # Console handler
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(handler)

        logger.propagate = False
        _CONFIGURED.add(name)

    return logger
