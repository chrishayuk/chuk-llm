# chuk_llm/api/show_info.py
"""
Utility functions for displaying provider and function information
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def show_providers():
    """Display all available providers and their models."""
    try:
        from ..configuration import get_config
        config = get_config()
        
        print("\n🚀 Available LLM Providers")
        print("=" * 60)
        
        providers = config.get_all_providers()
        
        for i, provider_name in enumerate(providers, 1):
            try:
                provider = config.get_provider(provider_name)
                has_key = "✅" if config.get_api_key(provider_name) else "❌"
                
                print(f"\n{i}. {provider_name.upper()}")
                print(f"   Status: {has_key} {'Configured' if has_key == '✅' else 'No API key'}")
                print(f"   Models: {len(provider.models)}")
                print(f"   Default: {provider.default_model}")
                
                # Show first few models
                if provider.models:
                    print("   Available models:")
                    for model in provider.models[:5]:
                        print(f"     - {model}")
                    if len(provider.models) > 5:
                        print(f"     ... and {len(provider.models) - 5} more")
                
                # Show features
                if provider.features:
                    features = [f.value for f in provider.features]
                    print(f"   Features: {', '.join(features[:5])}")
                    if len(features) > 5:
                        print(f"            ... and {len(features) - 5} more")
                        
            except Exception as e:
                print(f"{i}. {provider_name} - Error: {e}")
        
        print(f"\n📊 Total: {len(providers)} providers available")
        
    except Exception as e:
        print(f"❌ Error loading providers: {e}")
        print("Make sure you have a valid providers.yaml configuration")


def show_functions():
    """Display all auto-generated functions."""
    try:
        from .providers import list_provider_functions
        
        functions = list_provider_functions()
        
        if not functions:
            print("❌ No provider functions found")
            return
        
        print(f"\n🔧 Auto-Generated Functions ({len(functions)} total)")
        print("=" * 60)
        
        # Categorize functions
        async_funcs = [f for f in functions if not f.endswith('_sync') and not f.startswith('stream_')]
        sync_funcs = [f for f in functions if f.endswith('_sync')]
        stream_funcs = [f for f in functions if f.startswith('stream_')]
        
        # Show async functions
        if async_funcs:
            print(f"\n📌 Async Functions ({len(async_funcs)}):")
            for i, func in enumerate(sorted(async_funcs)[:20], 1):
                print(f"   {i:2d}. {func}()")
            if len(async_funcs) > 20:
                print(f"   ... and {len(async_funcs) - 20} more")
        
        # Show sync functions
        if sync_funcs:
            print(f"\n📌 Sync Functions ({len(sync_funcs)}):")
            for i, func in enumerate(sorted(sync_funcs)[:20], 1):
                print(f"   {i:2d}. {func}()")
            if len(sync_funcs) > 20:
                print(f"   ... and {len(sync_funcs) - 20} more")
        
        # Show stream functions
        if stream_funcs:
            print(f"\n📌 Streaming Functions ({len(stream_funcs)}):")
            for i, func in enumerate(sorted(stream_funcs)[:20], 1):
                print(f"   {i:2d}. {func}()")
            if len(stream_funcs) > 20:
                print(f"   ... and {len(stream_funcs) - 20} more")
        
        # Show examples
        print("\n💡 Example Usage:")
        print("   from chuk_llm import ask_openai, ask_claude_sync, stream_groq")
        print("   ")
        print("   # Async")
        print("   response = await ask_openai('Hello!')")
        print("   ")
        print("   # Sync")
        print("   response = ask_claude_sync('Hello!')")
        print("   ")
        print("   # Streaming")
        print("   async for chunk in stream_groq('Tell me a story'):")
        print("       print(chunk, end='')")
        
    except Exception as e:
        print(f"❌ Error listing functions: {e}")


def show_model_aliases():
    """Display model aliases for each provider."""
    try:
        from ..configuration import get_config
        config = get_config()
        
        print("\n🏷️  Model Aliases")
        print("=" * 60)
        
        providers = config.get_all_providers()
        
        for provider_name in providers:
            try:
                provider = config.get_provider(provider_name)
                
                if provider.model_aliases:
                    print(f"\n{provider_name.upper()}:")
                    for alias, actual in provider.model_aliases.items():
                        print(f"   {alias} → {actual}")
                        
            except Exception:
                pass
        
        # Global aliases
        global_aliases = config.get_global_aliases()
        if global_aliases:
            print("\nGLOBAL ALIASES:")
            for alias, target in global_aliases.items():
                print(f"   ask_{alias}() → {target}")
                
    except Exception as e:
        print(f"❌ Error loading aliases: {e}")


def show_capabilities(provider: str = None):
    """Show capabilities for a specific provider or all providers."""
    try:
        from ..configuration import get_config, Feature
        config = get_config()
        
        if provider:
            # Show specific provider
            try:
                provider_config = config.get_provider(provider)
                print(f"\n🎯 {provider.upper()} Capabilities")
                print("=" * 40)
                
                # Features
                features = [f.value for f in provider_config.features]
                print(f"Features: {', '.join(features)}")
                
                # Models with capabilities
                print("\nModels:")
                for model in provider_config.models[:5]:
                    caps = provider_config.get_model_capabilities(model)
                    print(f"  {model}:")
                    print(f"    Max context: {caps.max_context_length:,} tokens")
                    print(f"    Max output: {caps.max_output_tokens:,} tokens")
                    model_features = [f.value for f in caps.features]
                    print(f"    Features: {', '.join(model_features[:5])}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            # Show all providers
            print("\n🎯 Provider Capabilities Overview")
            print("=" * 80)
            
            # Create comparison table
            print(f"{'Provider':<15} {'Vision':<8} {'Tools':<8} {'Stream':<8} {'JSON':<8} {'API Key':<8}")
            print("-" * 70)
            
            for provider_name in config.get_all_providers():
                try:
                    provider = config.get_provider(provider_name)
                    has_key = "✅" if config.get_api_key(provider_name) else "❌"
                    
                    print(f"{provider_name:<15} "
                          f"{'✅' if Feature.VISION in provider.features else '❌':<8} "
                          f"{'✅' if Feature.TOOLS in provider.features else '❌':<8} "
                          f"{'✅' if Feature.STREAMING in provider.features else '❌':<8} "
                          f"{'✅' if Feature.JSON_MODE in provider.features else '❌':<8} "
                          f"{has_key:<8}")
                except:
                    pass
                
    except Exception as e:
        print(f"❌ Error showing capabilities: {e}")


def show_config():
    """Show current configuration."""
    try:
        from .config import get_current_config
        from ..configuration import get_config
        
        config = get_current_config()
        config_manager = get_config()
        
        print("\n⚙️  Current Configuration")
        print("=" * 50)
        
        print(f"Active Provider: {config.get('provider', 'not set')}")
        print(f"Model: {config.get('model', 'default')}")
        print(f"Temperature: {config.get('temperature', 'default')}")
        print(f"Max Tokens: {config.get('max_tokens', 'default')}")
        
        # Session tracking status
        try:
            from .core import _SESSIONS_ENABLED
            print(f"Session Tracking: {'✅ Enabled' if _SESSIONS_ENABLED else '❌ Disabled'}")
        except:
            print("Session Tracking: Unknown")
        
        # Show available providers with API keys
        print("\nConfigured Providers:")
        for provider in config_manager.get_all_providers():
            has_key = "✅" if config_manager.get_api_key(provider) else "❌"
            print(f"  {has_key} {provider}")
            
    except Exception as e:
        print(f"❌ Error showing config: {e}")


# Add to __init__.py exports
def _add_show_functions_to_module():
    """Add show functions to module namespace."""
    import sys
    module = sys.modules['chuk_llm']
    
    module.show_providers = show_providers
    module.show_functions = show_functions
    module.show_model_aliases = show_model_aliases
    module.show_capabilities = show_capabilities
    module.show_config = show_config


# Call this when module loads
_add_show_functions_to_module()