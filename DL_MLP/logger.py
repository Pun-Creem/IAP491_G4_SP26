"""
Logging System.

Logs everything to BOTH:
    - Console (real-time display)
    - File (permanent record for thesis report)

Log files are saved to logs/ with timestamp in filename.
"""

import os
import sys
import logging
from datetime import datetime

import config


def setup_logger(name="malware_dl", mode="train"):
    """
    Setup logger that writes to both console and file.

    Args:
        name: Logger name
        mode: "train" or "predict" (used in log filename)

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Log format
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (shows on screen)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler (saves permanently)
    log_dir = os.path.join(config.BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{mode}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # File gets ALL details
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_path}")
    return logger


def get_logger():
    """Get existing logger (call setup_logger first)."""
    return logging.getLogger("malware_dl")
