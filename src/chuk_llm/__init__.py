# chuk_llm/__init__.py
"""
ChukLLM - A clean, intuitive Python library for LLM interactions
================================================================

Main package initialization with provider function exports.
"""

# Version
__version__ = "0.1.0"

# Core API imports
from .api import (
    # Core async functions
    ask,
    stream,
    ask_with_tools,
    ask_json,
    quick_ask,
    multi_provider_ask,
    validate_request,
    
    # Sync wrappers
    ask_sync,
    stream_sync,
    stream_sync_iter,
    compare_providers,
    quick_question,
    
    # Configuration
    configure,
    get_current_config,
    reset,
    debug_config_state,
    quick_setup,
    switch_provider,
    auto_configure,
    validate_config,
    get_capabilities,
    supports_feature,
    
    # Client management
    get_client,
    list_available_providers,
    validate_provider_setup,
)

# Import all from api (which includes provider functions)
from .api import *

# Configuration utilities
from .configuration import (
    Feature,
    ModelCapabilities,
    ProviderConfig,
    UnifiedConfigManager,
    ConfigValidator,
    CapabilityChecker,
    get_config,
    reset_config,
)

# Conversation management
from .api.conversation import (
    conversation,
    ConversationContext,
)

# Utilities
from .api.utils import (
    get_metrics,
    health_check,
    health_check_sync,
    get_current_client_info,
    test_connection,
    test_connection_sync,
    test_all_providers,
    test_all_providers_sync,
    print_diagnostics,
    cleanup,
    cleanup_sync,
)

# Get all API exports including provider functions
from .api import __all__ as api_exports

# Define what's exported
__all__ = [
    # Version
    "__version__",
] + api_exports + [
    # Configuration types not in api
    "Feature",
    "ModelCapabilities",
    "ProviderConfig",
    "UnifiedConfigManager",
    "ConfigValidator",
    "CapabilityChecker",
    "get_config",
    "reset_config",
    
    # Conversation
    "conversation",
    "ConversationContext",
    
    # Utilities
    "get_metrics",
    "health_check",
    "health_check_sync",
    "get_current_client_info",
    "test_connection",
    "test_connection_sync",
    "test_all_providers",
    "test_all_providers_sync",
    "print_diagnostics",
    "cleanup",
    "cleanup_sync",
]