#!/usr/bin/env python3
"""
ChukLLM QuickStart Demo - Basic Usage
=====================================

This demo shows the simplest ways to get started with ChukLLM.
Perfect for beginners and quick prototypes!
"""

# Basic imports - everything you need from one place
from chuk_llm import ask_sync, quick_question, configure

def demo_basic_usage():
    """Demonstrate the absolute basics."""
    print("🚀 ChukLLM Basic Usage Demo")
    print("=" * 40)
    
    # 1. Simplest possible usage
    print("\n1️⃣ Ultra-simple one-liner:")
    answer = quick_question("What is 2 + 2?")
    print(f"   Q: What is 2 + 2?")
    print(f"   A: {answer}")
    
    # 2. Basic ask function
    print("\n2️⃣ Basic ask function:")
    response = ask_sync("Tell me a dad joke")
    print(f"   Q: Tell me a dad joke")
    print(f"   A: {response}")
    
    # 3. Configuration
    print("\n3️⃣ Configuration:")
    configure(temperature=0.9)  # Make responses more creative
    creative_response = ask_sync("Write a creative opening line for a story")
    print(f"   Q: Write a creative opening line for a story")
    print(f"   A: {creative_response}")
    
    print("\n✅ Basic demo complete!")

def demo_provider_selection():
    """Show how to use different providers."""
    print("\n🔄 Provider Selection Demo")
    print("=" * 40)
    
    question = "What's the capital of Japan?"
    
    # Try different providers
    providers_to_test = ["openai", "anthropic"]
    
    for provider in providers_to_test:
        try:
            response = ask_sync(question, provider=provider)
            print(f"\n🔹 {provider.title()}:")
            print(f"   {response[:100]}...")
        except Exception as e:
            print(f"\n❌ {provider.title()}: {str(e)[:50]}...")
    
    print("\n✅ Provider selection demo complete!")

def demo_model_selection():
    """Show how to specify models explicitly."""
    print("\n🎯 Model Selection Demo")
    print("=" * 40)
    
    question = "Explain quantum computing in one sentence."
    
    # Test different models
    model_tests = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-sonnet-20240229"),
    ]
    
    for provider, model in model_tests:
        try:
            response = ask_sync(question, provider=provider, model=model)
            print(f"\n🔹 {provider} ({model}):")
            print(f"   {response}")
        except Exception as e:
            print(f"\n❌ {provider} ({model}): {str(e)[:50]}...")
    
    print("\n✅ Model selection demo complete!")

def demo_configuration_options():
    """Show various configuration options."""
    print("\n⚙️ Configuration Options Demo")
    print("=" * 40)
    
    question = "Write three words about the weather"
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 0.9]
    
    for temp in temperatures:
        response = ask_sync(question, temperature=temp, max_tokens=20)
        print(f"\n🌡️ Temperature {temp}:")
        print(f"   {response}")
    
    print("\n✅ Configuration demo complete!")

if __name__ == "__main__":
    print("🎯 ChukLLM QuickStart - Basic Usage")
    print("This demo shows the fundamentals of ChukLLM")
    print("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables")
    print()
    
    try:
        demo_basic_usage()
        demo_provider_selection() 
        demo_model_selection()
        demo_configuration_options()
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Next steps:")
        print("   • Try the advanced demo: python advanced_demo.py")
        print("   • Try the conversation demo: python conversation_demo.py") 
        print("   • Explore provider-specific functions: python provider_demo.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\n💡 Make sure you have API keys configured:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'")