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
        "httpx",
        "httpx._client",
        "urllib3",
        "requests",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Suppress verbose ChukLLM internal logs (make them DEBUG level)
    verbose_chuk_loggers = [
        "chuk_llm.api.providers",  # Provider generation noise
        "chuk_llm.configuration.unified_config",  # Config loading details
        "chuk_llm.llm.discovery.ollama_discoverer",  # Discovery details
        "chuk_llm.llm.discovery.openai_discoverer",  # Discovery details
        "chuk_llm.llm.discovery.engine",  # Engine details
        "chuk_llm.configuration.discovery",  # Discovery updates
    ]

    for logger_name in verbose_chuk_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Allow environment overrides for debugging
    if os.getenv("CHUK_LLM_DEBUG_HTTP"):
        logging.getLogger("httpx").setLevel(logging.DEBUG)

    if os.getenv("CHUK_LLM_DEBUG_PROVIDERS"):
        logging.getLogger("chuk_llm.api.providers").setLevel(logging.DEBUG)

    if os.getenv("CHUK_LLM_DEBUG_DISCOVERY"):
        for logger_name in [
            "chuk_llm.llm.discovery.ollama_discoverer",
            "chuk_llm.llm.discovery.openai_discoverer",
            "chuk_llm.llm.discovery.engine",
            "chuk_llm.configuration.discovery",
        ]:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)

    if os.getenv("CHUK_LLM_DEBUG_CONFIG"):
        logging.getLogger("chuk_llm.configuration.unified_config").setLevel(
            logging.DEBUG
        )

    # Allow full debug mode
    if os.getenv("CHUK_LLM_DEBUG_ALL"):
        logging.getLogger("chuk_llm").setLevel(logging.DEBUG)


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
# Import all from api (which includes provider functions)
from .api import *  # noqa: F403, E402
from .api import (  # noqa: E402
    # Core async functions
    ask,  # noqa: F401
    ask_json,  # noqa: F401
    # Sync wrappers
    ask_sync,  # noqa: F401
    ask_with_tools,  # noqa: F401
    auto_configure,  # noqa: F401
    compare_providers,  # noqa: F401
    # Configuration
    configure,  # noqa: F401
    debug_config_state,  # noqa: F401
    disable_sessions,  # noqa: F401
    enable_sessions,  # noqa: F401
    get_capabilities,  # noqa: F401
    # Client management
    get_client,  # noqa: F401
    get_current_config,  # noqa: F401
    get_current_session_id,  # noqa: F401
    get_session_history,  # noqa: F401
    # Session management
    get_session_stats,  # noqa: F401
    list_available_providers,  # noqa: F401
    multi_provider_ask,  # noqa: F401
    quick_ask,  # noqa: F401
    quick_question,  # noqa: F401
    quick_setup,  # noqa: F401
    reset,  # noqa: F401
    reset_session,  # noqa: F401
    stream,  # noqa: F401
    stream_sync,  # noqa: F401
    stream_sync_iter,  # noqa: F401
    supports_feature,  # noqa: F401
    switch_provider,  # noqa: F401
    validate_config,  # noqa: F401
    validate_provider_setup,  # noqa: F401
    validate_request,  # noqa: F401
)

# Conversation management
from .api.conversation import (  # noqa: E402
    ConversationContext,
    conversation,
)
from .api.conversation_sync import (  # noqa: E402
    ConversationContextSync,
    conversation_sync,
)

# Show functions
from .api.show_info import (  # noqa: E402
    show_capabilities,
    show_config,
    show_functions,
    show_model_aliases,
    show_providers,
)

# Utilities
from .api.utils import (  # noqa: E402
    cleanup,
    cleanup_sync,
    get_current_client_info,
    get_metrics,
    health_check,
    health_check_sync,
    print_diagnostics,
    test_all_providers,
    test_all_providers_sync,
    test_connection,
    test_connection_sync,
)

# Configuration utilities
from .configuration import (  # noqa: E402
    CapabilityChecker,
    ConfigValidator,
    Feature,
    ModelCapabilities,
    ProviderConfig,
    UnifiedConfigManager,
    get_config,
    reset_config,
)

# Session utilities
try:
    from .api.session_utils import (
        auto_configure_sessions,
        check_session_backend_availability,
        get_session_recommendations,
        print_session_diagnostics,
        validate_session_configuration,
    )

    SESSION_UTILS_AVAILABLE = True
except ImportError:
    SESSION_UTILS_AVAILABLE = False

    # Create stub functions
    def check_session_backend_availability():  # type: ignore[misc]
        return {"error": "Session utilities not available"}

    def validate_session_configuration():  # type: ignore[misc]
        return False

    def get_session_recommendations():  # type: ignore[misc]
        return ["Session utilities not available"]

    def auto_configure_sessions():  # type: ignore[misc]
        return False

    def print_session_diagnostics():  # type: ignore[misc]
        print("Session diagnostics not available")


# Get all API exports including provider functions
from .api import __all__ as api_exports  # noqa: E402


# Enhanced diagnostics function
def print_full_diagnostics():
    """Print comprehensive ChukLLM diagnostics including session info."""
    print_diagnostics()  # Existing function
    print_session_diagnostics()  # Session-specific diagnostics


# Define what's exported
__all__ = (
    [
        # Version
        "__version__",
    ]
    + api_exports
    + [
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
        "conversation_sync",
        "ConversationContextSync",
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
)

# Auto-configure sessions on import (optional)
try:
    if SESSION_UTILS_AVAILABLE:
        auto_configure_sessions()
except Exception:
    pass  # Silent fail for auto-configuration
