#!/usr/bin/env python3
"""
Multi-provider pirate examples showing how different AI models
interpret the same pirate personality.
"""

from chuk_llm import ask_sync
import os

def compare_pirate_providers():
    """Compare how different providers handle pirate personalities"""
    
    # Same pirate prompt for all providers
    pirate_prompt = """You are Captain Jack Sparrow from Pirates of the Caribbean.
    You're witty, unpredictable, and always speak with pirate slang. You often
    reference rum, the Black Pearl, and your various adventures. Keep responses brief."""
    
    # Question to ask all providers
    question = "What's the secret to writing good code?"
    
    # Provider configurations
    # Each tuple: (provider, model, description)
    providers = [
        ("ollama", "granite3.3:latest", "Ollama Granite 3.3 (Local)"),
        ("ollama", "llama3.2", "Ollama Llama 3.2 (Local)"),
        ("ollama", "phi3", "Ollama Phi-3 (Local)"),
    ]
    
    # Add cloud providers if API keys are available
    if os.getenv("OPENAI_API_KEY"):
        providers.extend([
            ("openai", "gpt-4o-mini", "OpenAI GPT-4o Mini"),
            ("openai", "gpt-3.5-turbo", "OpenAI GPT-3.5 Turbo"),
        ])
    
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.extend([
            ("anthropic", "claude-3-haiku", "Anthropic Claude 3 Haiku"),
            ("anthropic", "claude-3-sonnet", "Anthropic Claude 3 Sonnet"),
        ])
    
    if os.getenv("GROQ_API_KEY"):
        providers.append(("groq", "llama-3.1-8b", "Groq Llama 3.1 8B"))
    
    if os.getenv("TOGETHER_API_KEY"):
        providers.append(("togetherai", "llama-3-8b", "Together AI Llama 3 8B"))
    
    print("🏴‍☠️ Multi-Provider Pirate Comparison 🏴‍☠️")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"\nAll pirates are Captain Jack Sparrow\n")
    print("=" * 60 + "\n")
    
    successful_responses = []
    failed_providers = []
    
    for provider, model, description in providers:
        print(f"### {description} ###")
        print(f"Provider: {provider}, Model: {model}\n")
        
        try:
            response = ask_sync(
                question,
                provider=provider,
                model=model,
                system_prompt=pirate_prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            print(f"Captain Jack says:\n{response}\n")
            successful_responses.append((description, response))
            
        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg or "not found" in error_msg:
                print(f"⚠️  {provider} not configured or model not available\n")
            else:
                print(f"❌ Error: {error_msg}\n")
            failed_providers.append(description)
        
        print("-" * 60 + "\n")
    
    # Summary
    print("=" * 60)
    print("📊 Summary")
    print(f"✅ Successfully got responses from {len(successful_responses)} providers")
    if failed_providers:
        print(f"⚠️  Could not get responses from: {', '.join(failed_providers)}")
    print("=" * 60)

def pirate_battle_providers():
    """Different providers compete as different pirate captains"""
    
    print("\n🗡️  Pirate Captain Battle 🗡️")
    print("=" * 60)
    print("Different AI providers play different famous pirates!\n")
    
    # Different pirate personalities for different providers
    pirate_captains = [
        {
            "provider": "ollama",
            "model": "granite3.3:latest",
            "name": "Blackbeard",
            "prompt": """You are Edward Teach, known as Blackbeard, the most feared pirate.
            You're intimidating, fierce, and dramatic. You light fuses in your beard during
            battle and speak with authority and menace."""
        },
        {
            "provider": "ollama",
            "model": "llama3.2",
            "name": "Anne Bonny",
            "prompt": """You are Anne Bonny, the famous female pirate. You're fierce,
            independent, and brave. You prove that women can be just as fearsome as
            any male pirate. Speak with confidence and defiance."""
        },
        {
            "provider": "ollama",
            "model": "phi3",
            "name": "Captain Kidd",
            "prompt": """You are Captain William Kidd. You started as a privateer but
            became a pirate. You're sophisticated, well-spoken, but with a dangerous edge.
            You often mention your buried treasure."""
        }
    ]
    
    # Topic for the pirate debate
    topic = "What's the most important quality for a software engineer?"
    
    print(f"Topic: {topic}\n")
    print("=" * 60 + "\n")
    
    for captain in pirate_captains:
        print(f"⚓ Captain {captain['name']} speaks:")
        print(f"   (Played by {captain['provider']}/{captain['model']})\n")
        
        try:
            response = ask_sync(
                topic,
                provider=captain['provider'],
                model=captain['model'],
                system_prompt=captain['prompt'],
                max_tokens=150
            )
            
            print(f"{response}\n")
            
        except Exception as e:
            print(f"   *{captain['name']} is not available: {e}*\n")
        
        print("-" * 40 + "\n")
    
    print("=" * 60)
    print("🏴‍☠️ The pirate council has spoken! 🏴‍☠️")

def pirate_temperature_test():
    """Test how temperature affects pirate creativity"""
    
    print("\n🌡️  Pirate Temperature Test 🌡️")
    print("=" * 60)
    print("How does temperature affect pirate creativity?\n")
    
    pirate_prompt = """You are a pirate storyteller. Tell a very brief tale
    about finding treasure. Be creative and use pirate language."""
    
    temperatures = [0.1, 0.5, 0.9, 1.5]
    
    for temp in temperatures:
        print(f"Temperature: {temp}")
        print("-" * 30)
        
        try:
            response = ask_sync(
                "Tell me a two-sentence story about finding treasure",
                provider="ollama",
                model="granite",
                system_prompt=pirate_prompt,
                temperature=temp,
                max_tokens=100
            )
            
            print(f"{response}\n")
            
        except Exception as e:
            print(f"Error at temperature {temp}: {e}\n")

def pirate_model_sizes():
    """Compare different model sizes with pirate prompts"""
    
    print("\n📏 Pirate Model Size Comparison 📏")
    print("=" * 60)
    print("How do different model sizes handle pirate personas?\n")
    
    pirate_prompt = """You are a pirate. Answer in one sentence with heavy pirate accent."""
    
    # Ollama models of different sizes (if available)
    models_to_test = [
        ("phi3:mini", "Phi-3 Mini (3.8B)"),
        ("llama3.2:1b", "Llama 3.2 1B"),
        ("llama3.2:3b", "Llama 3.2 3B"),
        ("granite", "Granite (3B)"),
        ("llama3.1:8b", "Llama 3.1 8B"),
    ]
    
    question = "What is machine learning?"
    
    print(f"Question: {question}\n")
    
    for model, description in models_to_test:
        print(f"{description}:")
        
        try:
            response = ask_sync(
                question,
                provider="ollama",
                model=model,
                system_prompt=pirate_prompt,
                max_tokens=50
            )
            
            print(f"  {response}\n")
            
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"  Model not installed\n")
            else:
                print(f"  Error: {e}\n")

def main():
    """Run all multi-provider pirate examples"""
    
    print("\n" + "=" * 60)
    print("🏴‍☠️ ChukLLM Multi-Provider Pirate Examples 🏴‍☠️")
    print("=" * 60 + "\n")
    
    # Run comparisons
    compare_pirate_providers()
    pirate_battle_providers()
    pirate_temperature_test()
    pirate_model_sizes()
    
    print("\n" + "=" * 60)
    print("🦜 All pirate provider examples complete! Fair winds! 🦜")
    print("=" * 60)

if __name__ == "__main__":
    main()