"""
Logging utility for AI Hand Gesture Mouse Controller.
Configures console and file loggers.
"""

import logging
import os
from typing import Optional


def get_logger(
    name: str = "HandGestureApp",
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: str = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """
    Get or create a configured logger instance.

    Args:
        name: Name of the logger.
        log_dir: Directory where log file is stored.
        log_file: Name of the log file.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console_output: Whether to also stream logs to stdout/stderr.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

