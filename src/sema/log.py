"""Logging setup using loguru.

All modules should import logger from here:
    from sema.log import logger
"""
import sys

from loguru import logger

__all__ = ["logger"]

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)
