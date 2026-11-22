"""DermaAI: Core Package Initialization and System Version Metadata."""

__version__ = "1.0.0"
__author__ = "DermaAI Engineering & Research Team"
__license__ = "MIT"
__description__ = "Intelligent Skin Disease Detection with Attention and XAI"

from src.config import load_config
from src.utils import set_seed, get_device_info, setup_logger

__all__ = ["__version__", "load_config", "set_seed", "get_device_info", "setup_logger"]
