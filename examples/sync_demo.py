#!/usr/bin/env python3
"""Synchronous API demo using the dynamic functions."""

from chuk_llm import (
    ask_sync, configure, 
    ask_openai_sync, ask_anthropic_sync,  # Use the actual generated functions
    quick_question, compare_providers
)

def main():
    """Demo sync functions - no async needed!"""
    
    print("🔄 ChukLLM Dynamic API Demo")
    print("=" * 50)
    print("No async/await needed!")
    
    # 1. Simple question
    print("\n1. Simple question:")
    try:
        answer = ask_sync("What's 2+2?")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Configure and use
    print("\n2. Configure once, use everywhere:")
    try:
        configure(provider="openai", model="gpt-4o-mini")
        response = ask_sync("Tell me a very short joke")
        print(f"Joke: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Provider-specific (using actual generated functions)
    print("\n3. Provider-specific calls:")
    try:
        openai_response = ask_openai_sync("Hello from OpenAI!")
        print(f"OpenAI: {openai_response[:100]}...")
    except Exception as e:
        print(f"OpenAI Error: {e}")
    
    try:
        # Use the actual function name that was generated
        anthropic_response = ask_anthropic_sync("Hello from Claude!")
        print(f"Anthropic: {anthropic_response[:100]}...")
    except Exception as e:
        print(f"Anthropic Error: {e}")
    
    # 4. Quick question helper
    print("\n4. Quick question helper:")
    try:
        answer = quick_question("What's the capital of France?")
        print(f"Capital: {answer}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Compare providers
    print("\n5. Compare providers:")
    try:
        responses = compare_providers("What is AI?", ["openai", "anthropic"])
        for provider, response in responses.items():
            print(f"{provider}: {response[:80]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Dynamic sync demo complete!")
    print("\n💡 Perfect for:")
    print("   • Simple scripts")
    print("   • Command-line tools") 
    print("   • Quick prototypes")
    print("   • No async complexity!")
    print("\n🎯 All functions generated dynamically from config!")

if __name__ == "__main__":
    main()