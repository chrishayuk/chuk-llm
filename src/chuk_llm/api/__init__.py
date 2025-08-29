# chuk_llm/api/__init__.py
"""
ChukLLM API Module - Clean Direct Imports
=========================================

Modern API interface with automatic session tracking when available.
"""

# Core async API
# Client factory
from ..llm.client import get_client, list_available_providers, validate_provider_setup

# Configuration management
from .config import (
    auto_configure,
    configure,
    debug_config_state,
    get_capabilities,
    get_current_config,
    quick_setup,
    reset,
    supports_feature,
    switch_provider,
    validate_config,
)
from .core import (
    ask,
    ask_json,
    ask_with_tools,
    disable_sessions,
    enable_sessions,
    get_current_session_id,
    get_session_history,
    # Session management functions
    get_session_stats,
    multi_provider_ask,
    quick_ask,
    reset_session,
    stream,
    validate_request,
)

# Dynamic provider registration
from .dynamic_providers import (
    get_provider_config,
    list_dynamic_providers,
    provider_exists,
    register_openai_compatible,
    register_provider,
    unregister_provider,
    update_provider,
)

# Import all provider functions
from .providers import *  # noqa: F403

# Sync wrappers
from .sync import (
    ask_sync,
    compare_providers,
    quick_question,
    stream_sync,
    stream_sync_iter,
)

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
    # Dynamic provider registration
    "register_provider",
    "update_provider",
    "unregister_provider",
    "list_dynamic_providers",
    "get_provider_config",
    "provider_exists",
    "register_openai_compatible",
]

# Add provider functions to __all__
try:
    from .providers import __all__ as provider_all

    __all__.extend(provider_all)
except ImportError:
    pass  # providers may not have generated functions yet
