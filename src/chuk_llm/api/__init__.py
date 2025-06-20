# chuk_llm/api/__init__.py
"""
ChukLLM API Module - Clean Direct Imports
=========================================

Modern API interface with automatic session tracking when available.
"""

# Core async API
from .core import (
    ask,
    stream, 
    ask_with_tools,
    ask_json,
    quick_ask,
    multi_provider_ask,
    validate_request,
    # Session management functions
    get_session_stats,
    get_session_history,
    get_current_session_id,
    reset_session,
    disable_sessions,
    enable_sessions,
)

# Configuration management
from .config import (
    configure,
    get_current_config,
    reset,
    debug_config_state,
    quick_setup,
    switch_provider,
    auto_configure,
    validate_config,
    get_capabilities,
    supports_feature
)

# Client factory
from ..llm.client import (
    get_client,
    list_available_providers,
    validate_provider_setup
)

# Sync wrappers
from .sync import (
    ask_sync,
    stream_sync,
    stream_sync_iter,
    compare_providers,
    quick_question
)

# Import all provider functions
from .providers import *

# Export clean API
__all__ = [
    # Core async API
    "ask", 
    "stream",
    "ask_with_tools",
    "ask_json", 
    "quick_ask",
    "multi_provider_ask",
    "validate_request",
    
    # Session management
    "get_session_stats",
    "get_session_history",
    "get_current_session_id",
    "reset_session",
    "disable_sessions",
    "enable_sessions",
    
    # Sync wrappers
    "ask_sync",
    "stream_sync",
    "stream_sync_iter",
    "compare_providers",
    "quick_question",
    
    # Configuration
    "configure",
    "get_current_config", 
    "reset",
    "debug_config_state",
    "quick_setup",
    "switch_provider", 
    "auto_configure",
    "validate_config",
    "get_capabilities",
    "supports_feature",
    
    # Client management
    "get_client",
    "list_available_providers",
    "validate_provider_setup",
]

# Add provider functions to __all__
try:
    from .providers import __all__ as provider_all
    __all__.extend(provider_all)
except ImportError:
    pass  # providers may not have generated functions yet