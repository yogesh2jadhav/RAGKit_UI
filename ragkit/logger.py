"""
Purpose
-------
Provide the central logger used throughout RAGKit and a helper
to configure logging for applications (API server, CLI, examples).

Responsibilities
----------------
- Create the framework logger.
- Provide a shared logger instance.
- Optionally configure console + rotating file handlers.

Does NOT
--------
- Configure logging automatically on import (libraries must not).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "ragkit"

logger = logging.getLogger(
    LOGGER_NAME,
)

#
# Prevent "No handler found" warnings
# if the application has not configured logging.
#
logger.addHandler(
    logging.NullHandler(),
)


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def _project_root() -> Path:
    """
    Return the repository root directory.
    """

    return Path(__file__).resolve().parents[1]


def configure_logging(
    *,
    level: int | str | None = None,
    log_file: str | os.PathLike[str] | None = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Configure the RAGKit logger with console and rotating file output.

    This is safe to call multiple times; existing RAGKit handlers are
    replaced so logs are not duplicated.

    Parameters
    ----------
    level
        Log level. Defaults to the ``RAGKIT_LOG_LEVEL`` environment
        variable, or ``INFO``.
    log_file
        Path to the log file. Defaults to the ``RAGKIT_LOG_FILE``
        environment variable, or ``<project_root>/logs/ragkit.log``.
    log_to_console
        Whether to also emit logs to stderr.

    Returns
    -------
    logging.Logger
        The configured ``ragkit`` logger.
    """

    if level is None:
        level = os.environ.get("RAGKIT_LOG_LEVEL", "INFO")

    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(
            level.upper(),
            logging.INFO,
        )

    if log_file is None:
        log_file = os.environ.get("RAGKIT_LOG_FILE") or (
            _project_root() / "logs" / "ragkit.log"
        )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    #
    # Remove previously configured RAGKit handlers so repeated
    # calls (e.g. uvicorn reload) do not duplicate log lines.
    #
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.setLevel(level)
    logger.propagate = False

    logger.info(
        "Logging configured (level=%s, file=%s)",
        logging.getLevelName(level),
        log_path,
    )

    return logger
