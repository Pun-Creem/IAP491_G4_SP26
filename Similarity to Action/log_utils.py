#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log Utilities - Tee stdout to both console and .log log file
=============================================================
Import and call setup_logging() at the start of any script to
automatically capture all print() output into a timestamped .log file.

Usage:
    from log_utils import setup_logging
    setup_logging()                       # log to logs/ in script dir
    setup_logging(log_dir="path/to/dir")  # log to custom dir
    setup_logging(log_name="my_run")      # custom log filename prefix
"""

import sys
import os
import datetime
from pathlib import Path


class TeeWriter:
    """Writes to both the original stdout/stderr and a log file."""

    def __init__(self, log_file, original_stream):
        self._log_file = log_file
        self._original = original_stream

    def write(self, message):
        self._original.write(message)
        self._original.flush()
        try:
            self._log_file.write(message)
            self._log_file.flush()
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except Exception:
            pass

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    @property
    def encoding(self):
        return self._original.encoding


_log_file_handle = None


def setup_logging(log_dir: str = "", log_name: str = "", script_path: str = ""):
    """
    Redirect stdout and stderr to both console and a .log log file.

    Args:
        log_dir:     Directory for log files. Default: logs/ next to the calling script.
        log_name:    Custom prefix for the log filename. Default: calling script's name.
        script_path: Path of the calling script (auto-detected if omitted).

    Returns:
        Path to the created log file.
    """
    global _log_file_handle

    # Avoid double setup
    if _log_file_handle is not None:
        return _log_file_handle.name

    # Auto-detect calling script
    if not script_path:
        import inspect
        frame = inspect.stack()[1]
        script_path = frame.filename

    script_file = Path(script_path)

    # Determine log directory
    if log_dir:
        log_directory = Path(log_dir)
    else:
        log_directory = script_file.parent / "logs"

    log_directory.mkdir(parents=True, exist_ok=True)

    # Determine log filename
    if not log_name:
        log_name = script_file.stem

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_name}_{timestamp}.log"
    log_path = log_directory / log_filename

    # Open log file
    _log_file_handle = open(log_path, "w", encoding="utf-8")

    # Write header
    _log_file_handle.write(f"{'='*70}\n")
    _log_file_handle.write(f" Log file: {log_filename}\n")
    _log_file_handle.write(f" Script : {script_file.name}\n")
    _log_file_handle.write(f" Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file_handle.write(f" CWD    : {os.getcwd()}\n")
    _log_file_handle.write(f"{'='*70}\n\n")
    _log_file_handle.flush()

    # Tee stdout and stderr
    sys.stdout = TeeWriter(_log_file_handle, sys.__stdout__)
    sys.stderr = TeeWriter(_log_file_handle, sys.__stderr__)

    print(f"[LOG] Logging to: {log_path}")

    return str(log_path)


def close_logging():
    """Restore original stdout/stderr and close the log file."""
    global _log_file_handle
    if _log_file_handle is not None:
        # Write footer
        try:
            _log_file_handle.write(f"\n{'='*70}\n")
            _log_file_handle.write(f" Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _log_file_handle.write(f"{'='*70}\n")
            _log_file_handle.flush()
        except Exception:
            pass

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        try:
            _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None


def get_log_path() -> str:
    """Return path to current log file, or empty string if not logging."""
    global _log_file_handle
    if _log_file_handle is not None:
        return _log_file_handle.name
    return ""


import atexit
atexit.register(close_logging)
