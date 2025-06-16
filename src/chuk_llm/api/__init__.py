# chuk_llm/api/__init__.py
"""
Clean API for ChukLLM - Dynamic provider functions
==================================================

Simple, clean API with dynamic function generation from YAML configuration.
"""

# Core functions
from .core import ask, stream
from .config import configure, get_current_config, get_client, reset as reset_config
from .sync import ask_sync, stream_sync, compare_providers, quick_question

# Import providers module to trigger dynamic generation
# This must be done after core imports to avoid circular dependencies
try:
    from . import providers
    # Import all dynamically generated functions
    from .providers import *
    
    # Export everything
    __all__ = [
        # Core functions
        "ask", "stream",
        # Config functions  
        "configure", "get_current_config", "get_client", "reset_config",
        # Conversation Functions
        "conversation",
        # Sync functions
        "ask_sync", "stream_sync", "compare_providers", "quick_question",
    ] + getattr(providers, '__all__', [])

except ImportError as e:
    # Fallback if providers module fails to load
    import logging
    logging.warning(f"Failed to load dynamic provider functions: {e}")
    
    __all__ = [
        "ask", "stream",
        "configure", "get_current_config", "get_client", "reset_config",
        "ask_sync", "stream_sync", "compare_providers", "quick_question",
    ]

# Note: get_config is from the configuration module, not the API module
# Import it here for convenience
from ..configuration.config import get_config