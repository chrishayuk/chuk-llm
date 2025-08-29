# chuk_llm/configuration/__init__.py
"""
Configuration module for ChukLLM - Clean Forward-Looking Version
===============================================================

Unified configuration system - get_config now returns UnifiedConfigManager.
"""

from .unified_config import (
    Feature,
    ModelCapabilities,
    ProviderConfig,
    UnifiedConfigManager,
    CapabilityChecker,
    get_config,
    reset_config,
    reset_unified_config,
    ConfigManager
)
from .validator import ConfigValidator

# Clean exports
__all__ = [
    "Feature",
    "ModelCapabilities", 
    "ProviderConfig",
    "UnifiedConfigManager",
    "ConfigValidator",
    "CapabilityChecker",
    "get_config",
    "reset_config", 
    "reset_unified_config",
    "ConfigManager",
]