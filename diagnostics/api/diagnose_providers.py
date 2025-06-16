#!/usr/bin/env python3
"""
Show all available providers and functions in ChukLLM
"""

from dotenv import load_dotenv
load_dotenv()

import chuk_llm
from chuk_llm.configuration.config import get_config

def show_providers():
    """Display all configured providers with their details."""
    print("\n🔧 Available Providers")
    print("=" * 60)
    
    config = get_config()
    providers = config.get_all_providers()
    
    for provider_name in sorted(providers):
        try:
            provider = config.get_provider(provider_name)
            api_key = config.get_api_key(provider_name)
            has_key = "✅" if api_key else "❌"
            
            print(f"\n📦 {provider_name}")
            print(f"   Status: {has_key} {'API key set' if api_key else 'No API key'}")
            print(f"   Default model: {provider.default_model}")
            print(f"   Models: {len(provider.models)}")
            if provider.models:
                # Show first 3 models
                for model in provider.models[:3]:
                    print(f"     - {model}")
                if len(provider.models) > 3:
                    print(f"     ... and {len(provider.models) - 3} more")
            
            if provider.model_aliases:
                print(f"   Aliases: {len(provider.model_aliases)}")
                # Show first 3 aliases
                for alias, target in list(provider.model_aliases.items())[:3]:
                    print(f"     - {alias} → {target}")
                if len(provider.model_aliases) > 3:
                    print(f"     ... and {len(provider.model_aliases) - 3} more")
                    
        except Exception as e:
            print(f"\n📦 {provider_name}")
            print(f"   ❌ Error: {e}")

def show_functions():
    """Display all dynamically generated functions."""
    print("\n🚀 Generated Functions")
    print("=" * 60)
    
    # Get all functions from chuk_llm module
    all_functions = []
    
    # Categorize functions
    ask_functions = []
    stream_functions = []
    sync_functions = []
    utility_functions = []
    
    for name in dir(chuk_llm):
        if name.startswith('_'):
            continue
            
        obj = getattr(chuk_llm, name)
        if callable(obj):
            if name.startswith('ask_') and name.endswith('_sync'):
                sync_functions.append(name)
            elif name.startswith('ask_'):
                ask_functions.append(name)
            elif name.startswith('stream_'):
                stream_functions.append(name)
            elif name in ['show_config', 'quick_question', 'compare_providers']:
                utility_functions.append(name)
    
    # Display by category
    print(f"\n📨 Ask Functions (async): {len(ask_functions)}")
    for func in sorted(ask_functions)[:10]:
        print(f"   - {func}()")
    if len(ask_functions) > 10:
        print(f"   ... and {len(ask_functions) - 10} more")
    
    print(f"\n🔄 Stream Functions: {len(stream_functions)}")
    for func in sorted(stream_functions)[:10]:
        print(f"   - {func}()")
    if len(stream_functions) > 10:
        print(f"   ... and {len(stream_functions) - 10} more")
    
    print(f"\n⚡ Sync Functions: {len(sync_functions)}")
    for func in sorted(sync_functions)[:10]:
        print(f"   - {func}()")
    if len(sync_functions) > 10:
        print(f"   ... and {len(sync_functions) - 10} more")
    
    print(f"\n🛠️  Utility Functions: {len(utility_functions)}")
    for func in sorted(utility_functions):
        print(f"   - {func}()")
    
    # Total count
    total = len(ask_functions) + len(stream_functions) + len(sync_functions) + len(utility_functions)
    print(f"\n📊 Total Functions: {total}")

def show_model_specific_functions(provider='openai'):
    """Show model-specific functions for a provider."""
    print(f"\n🎯 Model-Specific Functions for {provider}")
    print("=" * 60)
    
    functions = []
    for name in dir(chuk_llm):
        if name.startswith(f'ask_{provider}_') and not name == f'ask_{provider}_sync':
            functions.append(name)
    
    # Group by model
    model_functions = {}
    for func in sorted(functions):
        # Extract model part
        parts = func.split('_', 2)
        if len(parts) >= 3:
            model_part = parts[2].replace('_sync', '')
            if model_part not in model_functions:
                model_functions[model_part] = []
            model_functions[model_part].append(func)
    
    for model, funcs in sorted(model_functions.items()):
        print(f"\n   {model}:")
        for func in funcs:
            print(f"     - {func}()")

def debug_gpt4o_mini():
    """Debug why gpt-4o-mini function might not exist."""
    print("\n🔍 Debugging gpt-4o-mini Functions")
    print("=" * 60)
    
    # Check configuration
    config = get_config()
    try:
        openai_config = config.get_provider('openai')
        print("\nOpenAI Models:")
        for model in openai_config.models:
            print(f"  - {model}")
            
        print("\nOpenAI Aliases:")
        for alias, target in openai_config.model_aliases.items():
            print(f"  - {alias} → {target}")
            
        # Check what functions were actually generated
        print("\nGenerated Functions containing 'gpt' and '4o':")
        for name in sorted(dir(chuk_llm)):
            if 'gpt' in name and '4o' in name and callable(getattr(chuk_llm, name)):
                print(f"  - {name}")
                
        # Check specific patterns
        print("\nGenerated Functions containing 'mini':")
        for name in sorted(dir(chuk_llm)):
            if 'mini' in name and callable(getattr(chuk_llm, name)):
                print(f"  - {name}")
                
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all diagnostic functions."""
    show_providers()
    show_functions()
    show_model_specific_functions('openai')
    show_model_specific_functions('anthropic')
    debug_gpt4o_mini()
    
    # Try to use show_config if it exists
    if hasattr(chuk_llm, 'show_config'):
        print("\n" + "=" * 60)
        print("Using built-in show_config():")
        print("=" * 60)
        chuk_llm.show_config()

if __name__ == "__main__":
    main()