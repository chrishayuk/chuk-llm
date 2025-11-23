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
from typing import Any


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

# Lazy imports using __getattr__ for fast startup
# This delays loading heavy modules until they're actually used
_LAZY_IMPORTS = {
    # Core API functions
    "ask": ".api",
    "ask_json": ".api",
    "ask_sync": ".api",
    "stream": ".api",
    "stream_sync": ".api",
    "stream_sync_iter": ".api",
    "quick_ask": ".api",
    "quick_question": ".api",
    "multi_provider_ask": ".api",
    "compare_providers": ".api",
    "validate_request": ".api",
    # Configuration
    "configure": ".api",
    "auto_configure": ".api",
    "quick_setup": ".api",
    "reset": ".api",
    "switch_provider": ".api",
    "get_current_config": ".api",
    "get_capabilities": ".api",
    "supports_feature": ".api",
    "validate_config": ".api",
    "debug_config_state": ".api",
    # Client management
    "get_client": ".api",
    "list_available_providers": ".api",
    "validate_provider_setup": ".api",
    # Session management
    "enable_sessions": ".api",
    "disable_sessions": ".api",
    "reset_session": ".api",
    "get_current_session_id": ".api",
    "get_session_history": ".api",
    "get_session_stats": ".api",
    # Conversation
    "conversation": ".api.conversation",
    "ConversationContext": ".api.conversation",
    "conversation_sync": ".api.conversation_sync",
    "ConversationContextSync": ".api.conversation_sync",
    # Show functions
    "show_providers": ".api.show_info",
    "show_functions": ".api.show_info",
    "show_model_aliases": ".api.show_info",
    "show_capabilities": ".api.show_info",
    "show_config": ".api.show_info",
    # Tools
    "Tool": ".api.tools",
    "ToolKit": ".api.tools",
    "Tools": ".api.tools",
    "tool": ".api.tools",
    "create_tool": ".api.tools",
    "tools_from_functions": ".api.tools",
    # Utilities
    "get_metrics": ".api.utils",
    "health_check": ".api.utils",
    "health_check_sync": ".api.utils",
    "get_current_client_info": ".api.utils",
    "test_connection": ".api.utils",
    "test_connection_sync": ".api.utils",
    "test_all_providers": ".api.utils",
    "test_all_providers_sync": ".api.utils",
    "print_diagnostics": ".api.utils",
    "cleanup": ".api.utils",
    "cleanup_sync": ".api.utils",
    # Configuration utilities
    "get_config": ".configuration",
    "reset_config": ".configuration",
    "Feature": ".configuration",
    "ModelCapabilities": ".configuration",
    "ProviderConfig": ".configuration",
    "UnifiedConfigManager": ".configuration",
    "ConfigValidator": ".configuration",
    "CapabilityChecker": ".configuration",
    # Registry
    "get_registry": ".registry",
    "ModelRegistry": ".registry",
    "ModelSpec": ".registry",
    "ModelWithCapabilities": ".registry",
    "ModelQuery": ".registry",
    "QualityTier": ".registry",
}

# Cache for already-imported modules
_imported_attrs: dict[str, Any] = {}


def __getattr__(name):
    """Lazy import attributes on first access"""
    # Check if already imported
    if name in _imported_attrs:
        return _imported_attrs[name]

    # Check if it's a lazy import
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]

        # Import the module
        if module_path == ".api":
            # Special handling for .api - use wildcard import to get all provider functions
            from . import api

            # Get all exports from api module
            for attr_name in dir(api):
                if not attr_name.startswith("_"):
                    _imported_attrs[attr_name] = getattr(api, attr_name)

            # Return the requested attribute
            if name in _imported_attrs:
                return _imported_attrs[name]
            raise AttributeError(f"Module 'chuk_llm.api' has no attribute '{name}'")
        else:
            # Import specific submodule
            from importlib import import_module

            module = import_module(module_path, package="chuk_llm")
            attr = getattr(module, name)
            _imported_attrs[name] = attr
            return attr

    # Not found
    raise AttributeError(f"module 'chuk_llm' has no attribute '{name}'")


# Add session utilities to lazy imports
_LAZY_IMPORTS.update(
    {
        "check_session_backend_availability": ".api.session_utils",
        "validate_session_configuration": ".api.session_utils",
        "get_session_recommendations": ".api.session_utils",
        "auto_configure_sessions": ".api.session_utils",
        "print_session_diagnostics": ".api.session_utils",
    }
)

# Assume session utilities are available - will fail gracefully if not
SESSION_UTILS_AVAILABLE = True


# Enhanced diagnostics function (also lazy)
def print_full_diagnostics():
    """Print comprehensive ChukLLM diagnostics including session info."""
    # Lazy import to avoid loading modules unnecessarily
    from chuk_llm.api.session_utils import print_session_diagnostics
    from chuk_llm.api.utils import print_diagnostics

    print_diagnostics()
    print_session_diagnostics()


# Define what's exported - all lazy imports plus version
__all__ = ["__version__", "SESSION_UTILS_AVAILABLE", "print_full_diagnostics"] + list(
    _LAZY_IMPORTS.keys()
)

# DON'T auto-configure sessions on import - it forces imports!
# Let it happen lazily when session functions are first used
