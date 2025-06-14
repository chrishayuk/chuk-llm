#!/usr/bin/env python3
"""
Working sync sample script that only imports functions that actually exist.
"""

# Import only the guaranteed basic functions first
from chuk_llm import (
    # Core functions (guaranteed to exist)
    ask_sync,
    quick_question,
    compare_providers,
    
    # Provider base functions (should exist)
    ask_openai_sync,
    ask_anthropic_sync, 
    ask_groq_sync,
    ask_gemini_sync,
    ask_mistral_sync,
    ask_ollama_sync,
    ask_deepseek_sync,
)

# Try to import additional functions with fallbacks
additional_functions = {}

# Try to import global aliases if they exist
def safe_import(module, func_name):
    """Safely import a function, returning None if it doesn't exist."""
    try:
        return getattr(module, func_name)
    except AttributeError:
        return None

import chuk_llm

# Try to get global aliases
additional_functions['ask_gpt4_sync'] = safe_import(chuk_llm, 'ask_gpt4_sync')
additional_functions['ask_claude4_sync'] = safe_import(chuk_llm, 'ask_claude4_sync')
additional_functions['ask_llama70b_sync'] = safe_import(chuk_llm, 'ask_llama70b_sync')
additional_functions['ask_fastest_sync'] = safe_import(chuk_llm, 'ask_fastest_sync')
additional_functions['ask_smartest_sync'] = safe_import(chuk_llm, 'ask_smartest_sync')
additional_functions['ask_creative_sync'] = safe_import(chuk_llm, 'ask_creative_sync')
additional_functions['ask_coding_sync'] = safe_import(chuk_llm, 'ask_coding_sync')
additional_functions['ask_cheapest_sync'] = safe_import(chuk_llm, 'ask_cheapest_sync')

# Try to get some provider-specific functions
additional_functions['ask_openai_gpt4o_sync'] = safe_import(chuk_llm, 'ask_openai_gpt4o_sync')
additional_functions['ask_anthropic_opus_sync'] = safe_import(chuk_llm, 'ask_anthropic_opus_sync')
additional_functions['ask_groq_instant_sync'] = safe_import(chuk_llm, 'ask_groq_instant_sync')

def test_provider(name, func, question="What's 2+2? Answer briefly."):
    """Helper to test a provider function with error handling."""
    if func is None:
        print(f"🤖 {name}")
        print(f"   ⚠️  Function not available")
        print()
        return False
        
    print(f"🤖 {name}")
    try:
        response = func(question)
        print(f"   {response}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    print()

def main():
    question = "What's the capital of France? Answer in one word."
    
    print("=" * 70)
    print("🚀 ChukLLM Working Demo Script")
    print("=" * 70)
    print()
    
    # Show available functions
    print("📊 Function Availability Check:")
    functions = chuk_llm.get_available_functions()
    print(f"   Total functions available: {len(functions)}")
    
    # Count global aliases
    providers = chuk_llm.get_available_providers()
    global_aliases = [f for f in functions if not any(p in f for p in providers) and f.endswith('_sync')]
    print(f"   Global alias functions: {len(global_aliases)}")
    
    if global_aliases:
        print(f"   Examples: {', '.join(global_aliases[:5])}")
    print()
    
    # Test default provider
    print("🌟 DEFAULT PROVIDER")
    test_provider("Default (ask_sync)", ask_sync, question)
    
    # Test global aliases if available
    if any(additional_functions.values()):
        print("⚡ GLOBAL ALIASES (if available)")
        
        if additional_functions['ask_gpt4_sync']:
            test_provider("GPT-4 Global", additional_functions['ask_gpt4_sync'], question)
        
        if additional_functions['ask_claude4_sync']:
            test_provider("Claude 4 Global", additional_functions['ask_claude4_sync'], question)
        
        if additional_functions['ask_llama70b_sync']:
            test_provider("Llama 70B Global", additional_functions['ask_llama70b_sync'], question)
    
    # Test capability-based functions
    if any(f for f in ['ask_fastest_sync', 'ask_smartest_sync', 'ask_creative_sync'] if additional_functions.get(f)):
        print("🎯 CAPABILITY-BASED FUNCTIONS")
        
        if additional_functions['ask_fastest_sync']:
            test_provider("Fastest Model", additional_functions['ask_fastest_sync'], question)
        
        if additional_functions['ask_smartest_sync']:
            test_provider("Smartest Model", additional_functions['ask_smartest_sync'], question)
        
        if additional_functions['ask_creative_sync']:
            test_provider("Most Creative", additional_functions['ask_creative_sync'], question)
    
    # Test major providers (guaranteed to work)
    print("☁️  MAJOR PROVIDERS")
    test_provider("OpenAI", ask_openai_sync, question)
    test_provider("Anthropic", ask_anthropic_sync, question)
    test_provider("Groq", ask_groq_sync, question)
    test_provider("Google Gemini", ask_gemini_sync, question)
    test_provider("Mistral", ask_mistral_sync, question)
    test_provider("Ollama", ask_ollama_sync, question)
    test_provider("DeepSeek", ask_deepseek_sync, question)
    
    # Test some provider-specific functions if available
    print("🔧 PROVIDER-SPECIFIC FUNCTIONS")
    
    if additional_functions['ask_openai_gpt4o_sync']:
        test_provider("OpenAI GPT-4o", additional_functions['ask_openai_gpt4o_sync'], question)
    
    if additional_functions['ask_anthropic_opus_sync']:
        test_provider("Anthropic Opus", additional_functions['ask_anthropic_opus_sync'], question)
    
    if additional_functions['ask_groq_instant_sync']:
        test_provider("Groq Instant", additional_functions['ask_groq_instant_sync'], question)
    
    # Test utility functions
    print("🛠️  UTILITY FUNCTIONS")
    
    print("🤖 Quick Question")
    try:
        response = quick_question("What's 1+1?")
        print(f"   {response}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    print("🤖 Compare Providers")
    try:
        results = compare_providers("What's the square root of 16?", ["openai", "anthropic"])
        for provider, response in results.items():
            print(f"   {provider}: {response}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test different task types with available functions
    print("📝 TASK DEMONSTRATIONS")
    
    # Creative task
    creative_question = "Write a haiku about coding. Be creative!"
    print("🎨 CREATIVE TASK")
    if additional_functions['ask_creative_sync']:
        test_provider("Creative Model", additional_functions['ask_creative_sync'], creative_question)
    else:
        test_provider("Anthropic (Creative)", ask_anthropic_sync, creative_question)
    
    # Speed test
    speed_question = "Name 3 colors. Be brief."
    print("⚡ SPEED TEST")
    if additional_functions['ask_fastest_sync']:
        test_provider("Fastest Model", additional_functions['ask_fastest_sync'], speed_question)
    else:
        test_provider("Groq (Fast)", ask_groq_sync, speed_question)
    
    # Cost test
    if additional_functions['ask_cheapest_sync']:
        cost_question = "What's 2*3? Be brief."
        print("💰 COST-EFFECTIVE TEST")
        test_provider("Cheapest Model", additional_functions['ask_cheapest_sync'], cost_question)
    
    print("=" * 70)
    print("✅ Demo complete!")
    print()
    print("💡 WHAT WE DEMONSTRATED:")
    print("   ✅ Core functions (ask_sync, quick_question, compare_providers)")
    print("   ✅ Provider functions (ask_openai_sync, ask_anthropic_sync, etc.)")
    
    if any(additional_functions.values()):
        available_globals = [k for k, v in additional_functions.items() if v is not None]
        print(f"   ✅ Global aliases: {len(available_globals)} available")
        if available_globals:
            print(f"      Examples: {', '.join(available_globals[:3])}")
    else:
        print("   ⚠️  No global aliases found (check your providers.yaml)")
    
    print(f"   📊 Total functions available: {len(chuk_llm.get_available_functions())}")
    print()
    print("💻 To see all available functions:")
    print("   chuk_llm.show_functions()")
    print("=" * 70)

if __name__ == "__main__":
    main()