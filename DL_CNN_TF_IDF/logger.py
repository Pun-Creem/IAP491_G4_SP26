"""
Logging utility for CNN Malware Action Recommendation.
"""

import logging
import os
import sys
from datetime import datetime

_logger = None


def setup_logger(name="malware_cnn", log_dir=None, mode=None):
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{mode}_" if mode else "cnn_"
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{prefix}{timestamp}.log"), encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _logger = logger
    return logger


def get_logger():
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
