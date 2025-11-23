# chuk_llm/configuration/__init__.py
"""
Configuration module for ChukLLM - Clean Forward-Looking Version
===============================================================

Unified configuration system using Pydantic + Registry.
"""

# Import models
from .models import (
    DiscoveryConfig,
    Feature,
    GlobalConfig,
    ModelCapabilities,
    ProviderConfig,
)

# Import config manager
from .unified_config import (
    CapabilityChecker,
    ConfigManager,
    UnifiedConfigManager,
    get_config,
    reset_config,
    reset_unified_config,
)

# Import validator
from .validator import ConfigValidator

# Clean exports
__all__ = [
    # Models
    "Feature",
    "ModelCapabilities",
    "ProviderConfig",
    "GlobalConfig",
    "DiscoveryConfig",
    # Config Manager
    "UnifiedConfigManager",
    "ConfigManager",
    "get_config",
    "reset_config",
    "reset_unified_config",
    # Helpers
    "ConfigValidator",
    "CapabilityChecker",
]
