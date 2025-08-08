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

# Configure clean logging on import
import logging
import os

def _configure_clean_logging():
    """Configure clean logging with suppressed third-party noise and verbose ChukLLM internals"""
    # Suppress noisy third-party loggers by default
    third_party_loggers = [
        'httpx',
        'httpx._client', 
        'urllib3',
        'requests',
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Suppress verbose ChukLLM internal logs (make them DEBUG level)
    verbose_chuk_loggers = [
        'chuk_llm.api.providers',  # Provider generation noise
        'chuk_llm.configuration.unified_config',  # Config loading details
        'chuk_llm.llm.discovery.ollama_discoverer',  # Discovery details
        'chuk_llm.llm.discovery.openai_discoverer',  # Discovery details
        'chuk_llm.llm.discovery.engine',  # Engine details
        'chuk_llm.configuration.discovery',  # Discovery updates
    ]
    
    for logger_name in verbose_chuk_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Allow environment overrides for debugging
    if os.getenv('CHUK_LLM_DEBUG_HTTP'):
        logging.getLogger('httpx').setLevel(logging.DEBUG)
    
    if os.getenv('CHUK_LLM_DEBUG_PROVIDERS'):
        logging.getLogger('chuk_llm.api.providers').setLevel(logging.DEBUG)
    
    if os.getenv('CHUK_LLM_DEBUG_DISCOVERY'):
        for logger_name in ['chuk_llm.llm.discovery.ollama_discoverer', 
                           'chuk_llm.llm.discovery.openai_discoverer',
                           'chuk_llm.llm.discovery.engine',
                           'chuk_llm.configuration.discovery']:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    if os.getenv('CHUK_LLM_DEBUG_CONFIG'):
        logging.getLogger('chuk_llm.configuration.unified_config').setLevel(logging.DEBUG)
    
    # Allow full debug mode
    if os.getenv('CHUK_LLM_DEBUG_ALL'):
        logging.getLogger('chuk_llm').setLevel(logging.DEBUG)

# Configure logging on import
_configure_clean_logging()

# Version - get from package metadata instead of hardcoding
try:
    from importlib.metadata import version
    __version__ = version("chuk-llm")
except ImportError:
    # Fallback for Python < 3.8
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("chuk-llm").version
    except Exception:
        # Last resort fallback
        __version__ = "0.8.1"

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