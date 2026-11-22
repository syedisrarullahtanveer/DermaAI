"""System Utilities: Reproducibility, Device Diagnostics, and Structured Logging."""

import logging
import os
import random
import sys
from typing import Dict, Any
import numpy as np


def setup_logger(
    name: str = "DermaAI",
    log_level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """Initialize a standardized application logger with stream and optional file output.

    Args:
        name: Name of the logger instance.
        log_level: Logging severity level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to output log file on disk.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic scientific reproducibility.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_device_info() -> Dict[str, Any]:
    """Detect available compute hardware (CPU, GPU, MPS) and environment versions.

    Returns:
        Dict[str, Any]: Device capabilities and framework runtime versions.
    """
    info = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "tensorflow_available": False,
        "gpu_available": False,
        "gpu_devices": [],
    }

    try:
        import tensorflow as tf
        info["tensorflow_available"] = True
        info["tensorflow_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["gpu_available"] = True
            info["gpu_devices"] = [gpu.name for gpu in gpus]
    except ImportError:
        pass

    return info
