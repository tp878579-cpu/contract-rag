# src/utils/logger.py
"""Shared logger. Usage in any module:

    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Ingested %d chunks", n)
"""

import logging
import sys

from src.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called many times
    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(handler)

    # Don't pass messages up to the root logger (prevents double printing)
    logger.propagate = False

    return logger