# src/chuk_llm/__init__.py
"""ChukLLM - Universal LLM Interface with Dynamic Provider Support.

A comprehensive Python library for interacting with multiple LLM providers
through a unified, intuitive API. Features automatic provider discovery,
smart model resolution, and both sync/async interfaces.

Examples:
    Simple usage:
        from chuk_llm import ask, quick_question
        
        response = ask("Hello!", provider="openai")
        answer = quick_question("What is 2+2?")
    
    Provider-specific functions:
        from chuk_llm import ask_openai, ask_anthropic, ask_groq
        
        openai_response = ask_openai("Hello!")
        anthropic_response = ask_anthropic("Hello!")
        groq_response = ask_groq("Hello!")
    
    Global alias functions:
        from chuk_llm import ask_gpt4, ask_claude4, ask_llama70b
        
        gpt_response = ask_gpt4("Explain quantum computing")
        claude_response = ask_claude4("Explain quantum computing")
        llama_response = ask_llama70b("Explain quantum computing")
    
    Conversations with memory:
        from chuk_llm import conversation
        
        async with conversation(provider="anthropic") as chat:
            response1 = await chat.say("I'm working on a Python project")
            response2 = await chat.say("Can you help me optimize it?")
    
    Configuration:
        from chuk_llm import configure
        
        configure(provider="anthropic", temperature=0.7)
        # All subsequent calls use these defaults
"""

# Import all API functions from the api module
# This makes them available at the top level: from chuk_llm import ask
from .api import *

# Explicitly expose key functions for better IDE support and documentation
from .api import (
    # Core functions
    ask, stream, 
    
    # Configuration
    configure, get_config, reset_config,
    
    # Conversation management  
    conversation,
    
    # Utility functions
    quick_question, compare_providers,
)

# Import sync functions from the sync module specifically
from .api.sync import ask_sync, stream_sync

# Import utility functions that don't depend on providers
from .api.utils import (
    get_metrics, health_check, health_check_sync,
    get_current_client_info, test_connection, test_connection_sync,
    test_all_providers, test_all_providers_sync,
    print_diagnostics, cleanup, cleanup_sync
)

# Version information
__version__ = "1.0.0"
__author__ = "ChukLLM Team"
__email__ = "support@chukllm.com"
__description__ = "Universal LLM Interface with Dynamic Provider Support"

# Package metadata
__all__ = [
    # Core API (imported from .api)
    "ask", "stream", 
    "configure", "get_config", "reset_config",
    "conversation",
    "quick_question", "compare_providers",
    
    # Sync functions (imported from .api.sync)
    "ask_sync", "stream_sync",
    
    # Utilities
    "get_metrics", "health_check", "health_check_sync",
    "get_current_client_info", "test_connection", "test_connection_sync", 
    "test_all_providers", "test_all_providers_sync",
    "print_diagnostics", "cleanup", "cleanup_sync",
    
    # All dynamically generated provider functions are automatically included
    # via the `from .api import *` statement above
]

# Module-level convenience functions
def get_available_providers():
    """Get list of available providers from configuration.
    
    Returns:
        List of provider names
        
    Examples:
        providers = chuk_llm.get_available_providers()
        print(f"Available: {providers}")
    """
    from .api.provider_utils import get_all_providers
    return get_all_providers()

def get_available_functions():
    """Get list of all available API functions.
    
    Returns:
        List of function names
        
    Examples:
        functions = chuk_llm.get_available_functions()
        openai_functions = [f for f in functions if 'openai' in f]
    """
    from .api import providers
    return providers.__all__

def show_providers():
    """Print information about available providers and their models.
    
    Examples:
        chuk_llm.show_providers()
    """
    from .api.provider_utils import get_provider_config, find_providers_yaml_path
    
    print("🔍 ChukLLM Available Providers")
    print("=" * 50)
    
    yaml_path = find_providers_yaml_path()
    if yaml_path:
        print(f"📁 Config: {yaml_path}")
    else:
        print("📁 Config: Using fallback providers")
    
    providers = get_available_providers()
    print(f"📊 Total providers: {len(providers)}")
    print()
    
    for provider in providers:
        config = get_provider_config(provider)
        default_model = config.get('default_model', 'Not set')
        api_key_env = config.get('api_key_env', 'Not specified')
        
        print(f"🔹 {provider}")
        print(f"   Default model: {default_model}")
        print(f"   API key env: {api_key_env}")
        
        if 'inherits' in config:
            print(f"   Inherits from: {config['inherits']}")
        
        print()

def show_functions():
    """Print information about available API functions.
    
    Examples:
        chuk_llm.show_functions()
    """
    functions = get_available_functions()
    
    # Categorize functions
    base_functions = []
    model_functions = []
    global_functions = []
    utility_functions = []
    
    for func in functions:
        if func in ['ask', 'stream', 'configure', 'conversation', 'quick_question', 'compare_providers']:
            utility_functions.append(func)
        elif any(provider in func for provider in get_available_providers()):
            # Check if it's a model-specific function (has extra underscores)
            parts = func.replace('ask_', '').replace('stream_', '').replace('_sync', '')
            provider_part = parts.split('_')[0]
            if provider_part in get_available_providers() and len(parts.split('_')) > 1:
                model_functions.append(func)
            else:
                base_functions.append(func)
        else:
            # Global aliases (no provider name in function)
            global_functions.append(func)
    
    print("🎯 ChukLLM Available Functions")
    print("=" * 50)
    print(f"📊 Total functions: {len(functions)}")
    print()
    
    print(f"🔹 Core functions ({len(utility_functions)}):")
    for func in sorted(utility_functions):
        print(f"   {func}()")
    print()
    
    if global_functions:
        print(f"🔹 Global alias functions ({len(global_functions)}):")
        for func in sorted(global_functions)[:15]:  # Show first 15
            print(f"   {func}()")
        if len(global_functions) > 15:
            print(f"   ... and {len(global_functions) - 15} more")
        print()
    
    print(f"🔹 Base provider functions ({len(base_functions)}):")
    for func in sorted(base_functions)[:10]:  # Show first 10
        print(f"   {func}()")
    if len(base_functions) > 10:
        print(f"   ... and {len(base_functions) - 10} more")
    print()
    
    print(f"🔹 Model-specific functions ({len(model_functions)}):")
    for func in sorted(model_functions)[:10]:  # Show first 10
        print(f"   {func}()")
    if len(model_functions) > 10:
        print(f"   ... and {len(model_functions) - 10} more")
    print()
    
    print("💡 Use help(chuk_llm.function_name) for detailed documentation")

# Add convenience functions to __all__
__all__.extend([
    "get_available_providers", "get_available_functions", 
    "show_providers", "show_functions",
    "__version__", "__author__", "__description__"
])

# Optional: Print welcome message on import (can be disabled with env var)
import os
if not os.environ.get('CHUK_LLM_QUIET'):
    try:
        providers = get_available_providers()
        functions = get_available_functions()
        print(f"🚀 ChukLLM v{__version__} loaded with {len(providers)} providers and {len(functions)} functions")
        print("💡 Use chuk_llm.show_providers() or chuk_llm.show_functions() for details")
    except Exception:
        # Silently fail if there are import issues during package initialization
        pass