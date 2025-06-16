# chuk_llm/__init__.py
"""
ChukLLM - Clean Forward-Looking API
==================================

Modern LLM client library with unified configuration.
"""

# Version
__version__ = "0.2.0"

def get_version():
    """Get ChukLLM version"""
    return __version__

def __getattr__(name):
    """Lazy loading of modules and dynamic function generation"""
    
    # Core API functions
    if name in ["ask", "stream", "ask_sync", "stream_sync"]:
        try:
            from chuk_llm.api import ask, stream, ask_sync, stream_sync
            globals().update({
                "ask": ask, "stream": stream, 
                "ask_sync": ask_sync, "stream_sync": stream_sync
            })
            return globals()[name]
        except ImportError as e:
            raise ImportError(f"Core API functions not available: {e}")
    
    # Configuration functions
    elif name in ["configure", "get_current_config", "reset"]:
        try:
            from chuk_llm.api import configure, get_current_config, reset
            globals().update({
                "configure": configure, 
                "get_current_config": get_current_config, 
                "reset": reset
            })
            return globals()[name]
        except ImportError as e:
            raise ImportError(f"Configuration functions not available: {e}")
    
    # Client factory
    elif name == "get_client":
        try:
            from chuk_llm.llm.client import get_client
            globals()["get_client"] = get_client
            return get_client
        except ImportError as e:
            raise ImportError(f"Client factory not available: {e}")
    
    # Configuration system
    elif name in ["get_config", "Feature"]:
        try:
            from chuk_llm.configuration import get_config, Feature
            globals().update({
                "get_config": get_config, 
                "Feature": Feature
            })
            return globals()[name]
        except ImportError as e:
            raise ImportError(f"Configuration system not available: {e}")
    
    # Diagnostics
    elif name == "quick_diagnostic":
        try:
            from chuk_llm.llm import quick_diagnostic
            globals()["quick_diagnostic"] = quick_diagnostic
            return quick_diagnostic
        except ImportError as e:
            raise ImportError(f"Diagnostics not available: {e}")
    
    # Provider-specific functions - delegate to providers.py
    elif name.startswith(("ask_", "stream_")) and ("_" in name[4:] or "_" in name[7:]):
        try:
            # Import the providers module which generates all the functions
            from chuk_llm.api import providers
            
            # Check if the function exists in providers module
            if hasattr(providers, name):
                func = getattr(providers, name)
                globals()[name] = func  # Cache it
                return func
            else:
                # Function doesn't exist in providers
                raise AttributeError(f"Function '{name}' not found in providers")
                
        except ImportError as e:
            raise ImportError(f"Provider functions not available: {e}")
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Core exports for IDE/static analysis
__all__ = [
    "__version__", "get_version",
    # Core API
    "ask", "stream", "ask_sync", "stream_sync",
    # Configuration  
    "configure", "get_current_config", "reset",
    # Client
    "get_client", 
    # Configuration system
    "get_config", "Feature",
    # Utilities
    "quick_diagnostic"
    # Note: Dynamic functions like ask_openai_sync are created on demand via providers.py
]