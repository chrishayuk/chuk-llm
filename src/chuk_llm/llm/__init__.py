# src/chuk_llm/llm/__init__.py
"""
ChukLLM LLM Module
=================

Core LLM functionality including clients and providers.
"""

import logging
import warnings
from typing import Dict, Any, Optional, List

# Set up logging
logger = logging.getLogger(__name__)

# Version info
__version__ = "0.1.0"
__author__ = "Chris Hay"
__email__ = "chris@chrishay.co.uk"

# Core configuration - FIXED: Use correct relative imports
try:
    from ..configuration.config import get_config, reset_config, ConfigManager
    _config_available = True
except ImportError as e:
    logger.debug(f"Configuration module import failed: {e}")
    _config_available = False
    
    # Provide fallback implementations
    def get_config():
        raise ImportError("Configuration module not available")
    
    def reset_config():
        raise ImportError("Configuration module not available")
    
    class ConfigManager:
        def __init__(self):
            raise ImportError("Configuration module not available")

# Core LLM functionality - FIXED: Remove redundant .llm
try:
    from .client import get_client
    from .core.base import BaseLLMClient
    _llm_available = True
except ImportError as e:
    logger.debug(f"LLM client import failed: {e}")
    _llm_available = False
    
    def get_client(*args, **kwargs):
        raise ImportError("LLM client module not available")
    
    class BaseLLMClient:
        pass

# API layer - FIXED: Use correct relative imports
try:
    from ..api import (
        ask, stream, ask_sync, stream_sync,
        configure, get_current_config, get_client as api_get_client,
        reset_config as api_reset_config, compare_providers, quick_question
    )
    _api_available = True
except ImportError as e:
    logger.debug(f"API module import failed: {e}")
    _api_available = False
    
    # Provide fallback implementations
    async def ask(*args, **kwargs):
        raise ImportError("API module not available")
    
    async def stream(*args, **kwargs):
        raise ImportError("API module not available")
    
    def ask_sync(*args, **kwargs):
        raise ImportError("API module not available")
    
    def stream_sync(*args, **kwargs):
        raise ImportError("API module not available")
    
    def configure(*args, **kwargs):
        raise ImportError("API module not available")
    
    def get_current_config():
        raise ImportError("API module not available")
    
    def api_get_client(*args, **kwargs):
        raise ImportError("API module not available")
    
    def api_reset_config():
        raise ImportError("API module not available")
    
    def compare_providers(*args, **kwargs):
        raise ImportError("API module not available")
    
    def quick_question(*args, **kwargs):
        raise ImportError("API module not available")

# Conversation management - FIXED: Use correct relative imports
try:
    from ..api.conversation import conversation, ConversationContext
    _conversation_available = True
except ImportError as e:
    logger.debug(f"Conversation module import failed: {e}")
    _conversation_available = False
    
    async def conversation(*args, **kwargs):
        raise ImportError("Conversation module not available")
    
    class ConversationContext:
        def __init__(self, *args, **kwargs):
            raise ImportError("Conversation module not available")

# Utilities - FIXED: Use correct relative imports
try:
    from ..api.utils import health_check, test_connection, print_diagnostics
    _utils_available = True
except ImportError as e:
    logger.debug(f"Utils module import failed: {e}")
    _utils_available = False
    
    async def health_check():
        raise ImportError("Utils module not available")
    
    async def test_connection(*args, **kwargs):
        raise ImportError("Utils module not available")
    
    def print_diagnostics():
        raise ImportError("Utils module not available")


def get_version() -> str:
    """Get the current version of ChukLLM."""
    return __version__


def get_available_modules() -> Dict[str, bool]:
    """Get information about which modules are available."""
    return {
        "config": _config_available,
        "llm": _llm_available,
        "api": _api_available,
        "conversation": _conversation_available,
        "utils": _utils_available,
    }


def check_installation() -> Dict[str, Any]:
    """Check the installation status and provide diagnostic information."""
    modules = get_available_modules()
    
    issues = []
    if not modules["config"]:
        issues.append("Configuration module failed to import")
    if not modules["llm"]:
        issues.append("LLM client module failed to import")
    if not modules["api"]:
        issues.append("API module failed to import")
    
    return {
        "version": __version__,
        "modules": modules,
        "issues": issues,
        "status": "healthy" if not issues else "degraded"
    }


# Export main API functions if available
__all__ = [
    "__version__",
    "get_version",
    "get_available_modules", 
    "check_installation",
    "get_config",
    "reset_config",
    "ConfigManager",
    "get_client",
    "BaseLLMClient",
]

# Add API functions if available
if _api_available:
    __all__.extend([
        "ask", "stream", "ask_sync", "stream_sync",
        "configure", "get_current_config", "api_get_client",
        "api_reset_config", "compare_providers", "quick_question"
    ])

if _conversation_available:
    __all__.extend(["conversation", "ConversationContext"])

if _utils_available:
    __all__.extend(["health_check", "test_connection", "print_diagnostics"])