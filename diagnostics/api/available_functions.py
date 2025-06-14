#!/usr/bin/env python3
"""
Script to discover what functions are actually available.
"""

import chuk_llm

def main():
    print("🔍 Discovering Available Functions")
    print("=" * 50)
    
    # Get all available functions
    functions = chuk_llm.get_available_functions()
    
    # Filter for global aliases (functions without provider names)
    providers = chuk_llm.get_available_providers()
    
    global_aliases = []
    provider_functions = []
    
    for func in functions:
        # Check if it's a global alias (doesn't contain any provider name)
        is_global = True
        for provider in providers:
            if provider in func:
                is_global = False
                break
        
        if is_global and func.endswith('_sync'):
            global_aliases.append(func)
        elif any(provider in func for provider in providers) and func.endswith('_sync'):
            provider_functions.append(func)
    
    print(f"📊 Global Alias Functions ({len(global_aliases)}):")
    for func in sorted(global_aliases):
        print(f"   {func}")
    print()
    
    print(f"📊 Provider-Specific Sync Functions (first 20):")
    for func in sorted(provider_functions)[:20]:
        print(f"   {func}")
    if len(provider_functions) > 20:
        print(f"   ... and {len(provider_functions) - 20} more")
    print()
    
    # Look for specific patterns
    print("🔍 Looking for specific patterns:")
    claude_functions = [f for f in functions if 'claude' in f and f.endswith('_sync')]
    gpt_functions = [f for f in functions if 'gpt4' in f and f.endswith('_sync')]
    llama_functions = [f for f in functions if 'llama' in f and f.endswith('_sync')]
    
    print(f"Claude functions: {claude_functions[:5]}")
    print(f"GPT functions: {gpt_functions[:5]}")
    print(f"Llama functions: {llama_functions[:5]}")
    
    # Test a few basic imports
    print("\n🧪 Testing Basic Imports:")
    
    basic_functions = [
        'ask_sync', 'ask_openai_sync', 'ask_anthropic_sync', 
        'quick_question', 'compare_providers'
    ]
    
    for func_name in basic_functions:
        try:
            func = getattr(chuk_llm, func_name)
            print(f"   ✅ {func_name} - Available")
        except AttributeError:
            print(f"   ❌ {func_name} - Not found")
    
    print(f"\n💡 Total functions available: {len(functions)}")

if __name__ == "__main__":
    main()