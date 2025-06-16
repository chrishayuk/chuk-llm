# chuk_llm/api/__init__.py
"""
ChukLLM API Module - Clean Direct Imports
=========================================

Modern API interface using unified configuration system.
"""

# Core async API
from .core import (
    ask,
    stream, 
    ask_with_tools,
    ask_json,
    quick_ask,
    multi_provider_ask,
    validate_request
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