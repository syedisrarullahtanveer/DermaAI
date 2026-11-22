"""Configuration Management Module for DermaAI.

Loads, validates, and resolves configuration files from YAML formats.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class Config:
    """Configuration container providing attribute-style and dict-style access."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely retrieve configuration key with fallback default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration hierarchy back to standard dictionary."""
        return self._config

    def __getitem__(self, item: str) -> Any:
        return self._config[item]

    def __contains__(self, item: str) -> bool:
        return item in self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"


def find_project_root() -> Path:
    """Locate the project root directory containing the configs folder."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
    return Path.cwd()


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file from disk safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(
    base_config_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
) -> Config:
    """Load unified application and model configurations.

    Args:
        base_config_path: Optional path to master config.yaml.
        model_config_path: Optional path to model_configs.yaml.

    Returns:
        Config: Structured configuration object.
    """
    root = find_project_root()
    base_path = Path(base_config_path) if base_config_path else (root / "configs" / "config.yaml")
    model_path = Path(model_config_path) if model_config_path else (root / "configs" / "model_configs.yaml")

    base_dict = load_yaml(base_path)
    if model_path.exists():
        model_dict = load_yaml(model_path)
        base_dict["models"] = model_dict

    # Store project root in config for absolute path resolution
    base_dict["project_root"] = str(root)

    return Config(base_dict)
