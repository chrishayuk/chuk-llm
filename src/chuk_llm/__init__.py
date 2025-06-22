# chuk_llm/__init__.py
"""
ChukLLM - A clean, intuitive Python library for LLM interactions
================================================================

Main package initialization with automatic session tracking support.

Installation Options:
    pip install chuk_llm                    # Core with session tracking (memory)
    pip install chuk_llm[redis]             # Production (Redis sessions)  
    pip install chuk_llm[cli]               # Enhanced CLI
    pip install chuk_llm[all]               # All features

Session Storage:
    Session tracking included by default with chuk-ai-session-manager
    Memory (default): Fast, no persistence, no extra dependencies
    Redis: Persistent, requires [redis] extra
    Configure with SESSION_PROVIDER environment variable
"""

# Version
__version__ = "0.7"

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
    
    # Session management
    get_session_stats,
    get_session_history,
    get_current_session_id,
    reset_session,
    disable_sessions,
    enable_sessions,
    
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

# Show functions
from .api.show_info import (
    show_providers,
    show_functions,
    show_model_aliases,
    show_capabilities,
    show_config,
)

# Session utilities
try:
    from .api.session_utils import (
        check_session_backend_availability,
        validate_session_configuration,
        get_session_recommendations,
        auto_configure_sessions,
        print_session_diagnostics,
    )
    SESSION_UTILS_AVAILABLE = True
except ImportError:
    SESSION_UTILS_AVAILABLE = False
    # Create stub functions
    def check_session_backend_availability():
        return {"error": "Session utilities not available"}
    def validate_session_configuration():
        return False
    def get_session_recommendations():
        return ["Session utilities not available"]
    def auto_configure_sessions():
        return False
    def print_session_diagnostics():
        print("Session diagnostics not available")

# Get all API exports including provider functions
from .api import __all__ as api_exports

# Enhanced diagnostics function
def print_full_diagnostics():
    """Print comprehensive ChukLLM diagnostics including session info."""
    print_diagnostics()  # Existing function
    print_session_diagnostics()  # Session-specific diagnostics

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
    "print_full_diagnostics",
    "cleanup",
    "cleanup_sync",
    
    # Session utilities
    "check_session_backend_availability",
    "validate_session_configuration", 
    "get_session_recommendations",
    "auto_configure_sessions",
    "print_session_diagnostics",
    
    # Show functions
    "show_providers",
    "show_functions",
    "show_model_aliases",
    "show_capabilities",
    "show_config",
]

# Auto-configure sessions on import (optional)
try:
    if SESSION_UTILS_AVAILABLE:
        auto_configure_sessions()
except Exception:
    pass  # Silent fail for auto-configuration