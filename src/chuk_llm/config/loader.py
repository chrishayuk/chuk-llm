"""
Configuration Loader
====================

Loads and validates configuration from YAML files using Pydantic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from chuk_llm.core import get_performance_info

from .models import ChukLLMConfig

logger = logging.getLogger(__name__)

# Try to import YAML library
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Try to import dotenv
try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


class ConfigLoader:
    """
    Configuration loader with Pydantic validation.

    Loads configuration from YAML and validates it against Pydantic models.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize config loader.

        Args:
            config_path: Optional path to config file. If not provided,
                        searches standard locations.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: ChukLLMConfig | None = None
        self._load_env()

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        if not _DOTENV_AVAILABLE:
            return

        env_candidates: list[Path] = [
            Path(".env"),
            Path(".env.local"),
            Path.home() / ".chuk_llm" / ".env",
        ]

        for env_path in env_candidates:
            if env_path.exists():
                logger.info(f"Loading environment from {env_path}")
                load_dotenv(env_path, override=False)
                break

    def _find_config_file(self) -> Path | None:
        """Find configuration file in standard locations."""
        if self.config_path and self.config_path.exists():
            return self.config_path

        # Check environment variable
        env_path_str = os.getenv("CHUK_LLM_CONFIG")
        if env_path_str:
            path = Path(env_path_str)
            if path.exists():
                return path

        # Check standard locations
        candidates = [
            Path("chuk_llm.yaml"),
            Path("config/chuk_llm.yaml"),
            Path.home() / ".chuk_llm" / "config.yaml",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def load(self) -> ChukLLMConfig:
        """
        Load and validate configuration.

        Returns:
            Validated configuration

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file not found
        """
        if self._config:
            return self._config

        # Find config file
        config_file = self._find_config_file()

        if not config_file:
            logger.info("No config file found, using defaults")
            self._config = ChukLLMConfig()
            return self._config

        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load configuration")

        # Load YAML
        logger.info(f"Loading configuration from {config_file}")
        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            logger.warning(f"Empty config file: {config_file}, using defaults")
            self._config = ChukLLMConfig()
            return self._config

        # Validate with Pydantic
        try:
            self._config = ChukLLMConfig.model_validate(config_data)
            logger.info(
                f"Configuration loaded successfully: "
                f"{len(self._config.providers)} providers configured"
            )

            # Log performance info
            perf = get_performance_info()
            logger.debug(
                f"Using {perf['library']} for JSON (speedup: {perf['speedup']})"
            )

            return self._config

        except Exception as e:
            raise ValueError(f"Invalid configuration in {config_file}: {e}") from e

    def reload(self) -> ChukLLMConfig:
        """Reload configuration from file."""
        self._config = None
        return self.load()

    def get(self) -> ChukLLMConfig:
        """Get current configuration (loads if not loaded)."""
        if not self._config:
            return self.load()
        return self._config


# Global config loader instance
_global_loader: ConfigLoader | None = None


def load_config(config_path: str | Path | None = None) -> ChukLLMConfig:
    """
    Load global configuration.

    Args:
        config_path: Optional path to config file

    Returns:
        Validated configuration
    """
    global _global_loader

    if _global_loader is None or config_path:
        _global_loader = ConfigLoader(config_path)

    return _global_loader.load()


def get_config() -> ChukLLMConfig:
    """
    Get current global configuration.

    Returns:
        Current configuration (loads if needed)
    """
    global _global_loader

    if _global_loader is None:
        _global_loader = ConfigLoader()

    return _global_loader.get()


def reload_config() -> ChukLLMConfig:
    """
    Reload configuration from file.

    Returns:
        Reloaded configuration
    """
    global _global_loader

    if _global_loader is None:
        _global_loader = ConfigLoader()

    return _global_loader.reload()
