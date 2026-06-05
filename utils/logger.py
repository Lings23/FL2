"""
utils/logger.py
---------------
Centralised logging setup for all experiment scripts.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: str = "logs/", level: str = "INFO",
                  name: Optional[str] = None) -> None:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_file = log_dir_path / f"{name or 'experiment'}.log"
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]

    logging.basicConfig(level=numeric_level, format=fmt,
                        datefmt=datefmt, handlers=handlers, force=True)

    # Quiet noisy libraries
    for noisy in ("flwr", "urllib3", "PIL", "torchvision"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
