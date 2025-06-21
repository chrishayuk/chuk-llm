#!/usr/bin/env python3
# diagnostics/configuration_e2e.py
"""
ChukLLM End-to-End Configuration Demo
====================================

Comprehensive test of all configuration features:
- Provider configuration and inheritance
- Dynamic function generation
- API key resolution
- Model aliases and global aliases
- Configuration state management
- Error handling and diagnostics
"""

from dotenv import load_dotenv
load_dotenv()

import chuk_llm
import os

def header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def section(title: str):
    """Print a formatted section"""
    print(f"\n{'-'*40}")
    print(f"📋 {title}")
    print(f"{'-'*40}")

def test_basic_configuration():
    """Test basic configuration functionality"""
    header("Basic Configuration")
    
    # Show initial state
    section("Initial Configuration State")
    chuk_llm.print_diagnostics()
    
    # Show available providers
    section("Available Providers")
    chuk_llm.show_providers()
    
    # Show available functions (first 20)
    section("Generated Functions (Sample)")
    functions = chuk_llm.get_available_functions()
    print(f"Total functions generated: {len(functions)}")
    print("Sample functions:")
    for i, func in enumerate(sorted(functions)[:20]):
        print(f"  {func}()")
    print(f"  ... and {len(functions) - 20} more")

def test_global_configuration():
    """Test global configuration management"""
    header("Global Configuration Management")
    
    section("Default Configuration")
    config = chuk_llm.get_current_config()
    print(f"Default provider: {config['provider']}")
    print(f"Default model: {config['model']}")
    print(f"API key present: {'✅' if config['api_key'] else '❌'}")
    
    section("Configure Different Providers")
    providers_to_test = ["anthropic", "groq", "openai"]
    
    for provider in providers_to_test:
        chuk_llm.configure(provider=provider)
        config = chuk_llm.get_current_config()
        print(f"\nAfter configure({provider}):")
        print(f"  Provider: {config['provider']}")
        print(f"  Model: {config['model']}")
        print(f"  API key: {config['api_key'][:10] if config['api_key'] else 'None'}...")
        print(f"  API base: {config['api_base'] or 'Default'}")
    
    section("Configure with Parameters")
    chuk_llm.configure(
        provider="anthropic",
        temperature=0.7,
        max_tokens=100,
        system_prompt="You are a helpful assistant that gives brief answers."
    )
    config = chuk_llm.get_current_config()
    print(f"Configured with parameters:")
    print(f"  Temperature: {config.get('temperature')}")
    print(f"  Max tokens: {config.get('max_tokens')}")
    print(f"  System prompt: {config.get('system_prompt')[:50]}...")

def test_provider_functions():
    """Test provider-specific functions"""
    header("Provider-Specific Functions")
    
    question = "What's 5+5? Answer with just the number."
    
    section("Base Provider Functions")
    providers = ["openai", "anthropic", "groq"]
    
    for provider in providers:
        try:
            # Test base provider function
            func = getattr(chuk_llm, f"ask_{provider}_sync")
            response = func(question)
            print(f"✅ ask_{provider}_sync(): {response}")
        except AttributeError:
            print(f"❌ ask_{provider}_sync(): Function not found")
        except Exception as e:
            print(f"❌ ask_{provider}_sync(): {str(e)[:50]}...")
    
    section("Model-Specific Functions")
    model_functions = [
        "ask_openai_gpt4o_sync",
        "ask_anthropic_sonnet4_sync", 
        "ask_groq_llama_sync",
    ]
    
    for func_name in model_functions:
        try:
            func = getattr(chuk_llm, func_name)
            response = func(question)
            print(f"✅ {func_name}(): {response}")
        except AttributeError:
            print(f"❌ {func_name}(): Function not found")
        except Exception as e:
            print(f"❌ {func_name}(): {str(e)[:50]}...")

def test_global_aliases():
    """Test global alias functions"""
    header("Global Alias Functions")
    
    question = "What's 7+3? Just the number please."
    
    section("Semantic Aliases")
    aliases = [
        ("ask_smartest_sync", "→ Claude Opus (best reasoning)"),
        ("ask_fastest_sync", "→ Groq Llama (ultra fast)"),
        ("ask_gpt4_sync", "→ OpenAI GPT-4o"),
        ("ask_claude_sync", "→ Anthropic Claude Sonnet 4"),
    ]
    
    for alias_name, description in aliases:
        try:
            func = getattr(chuk_llm, alias_name)
            response = func(question)
            print(f"✅ {alias_name}() {description}: {response}")
        except AttributeError:
            print(f"❌ {alias_name}(): Function not found")
        except Exception as e:
            print(f"❌ {alias_name}(): {str(e)[:50]}...")

def test_inheritance_and_features():
    """Test configuration inheritance and feature detection"""
    header("Configuration Inheritance & Features")
    
    section("Provider Inheritance")
    from chuk_llm.configuration.unified_config import get_config
    config_manager = get_config()
    
    inheriting_providers = ["deepseek", "groq", "perplexity"]
    
    for provider in inheriting_providers:
        try:
            provider_config = config_manager.get_provider(provider)
            inherits = getattr(provider_config, 'inherits', None)
            client_class = getattr(provider_config, 'client_class', 'Unknown')
            api_base = getattr(provider_config, 'api_base', None)
            
            print(f"\n{provider}:")
            print(f"  Inherits from: {inherits or 'None'}")
            print(f"  Client class: {client_class.split('.')[-1]}")
            print(f"  API base: {api_base or 'Default'}")
            
        except Exception as e:
            print(f"\n{provider}: ❌ Error - {e}")
    
    section("Feature Support")
    providers = ["openai", "anthropic", "groq", "ollama"]
    features = ["streaming", "tools", "vision", "json_mode"]
    
    print(f"{'Provider':<12} | {'Streaming':<9} | {'Tools':<5} | {'Vision':<6} | {'JSON':<4}")
    print("-" * 50)
    
    for provider in providers:
        feature_status = []
        for feature in features:
            try:
                supported = config_manager.supports_feature(provider, feature)
                feature_status.append("✅" if supported else "❌")
            except:
                feature_status.append("❓")
        
        print(f"{provider:<12} | {feature_status[0]:<9} | {feature_status[1]:<5} | {feature_status[2]:<6} | {feature_status[3]:<4}")

def test_advanced_features():
    """Test advanced configuration features"""
    header("Advanced Features")
    
    section("Parameter Override")
    question = "Describe AI in one word."
    
    # Test parameter overrides
    try:
        # Different temperatures
        response1 = chuk_llm.ask_sync(question, provider="openai", temperature=0.1)
        response2 = chuk_llm.ask_sync(question, provider="openai", temperature=0.9)
        
        print(f"Temperature 0.1 (conservative): {response1}")
        print(f"Temperature 0.9 (creative): {response2}")
        
        # Different max tokens
        response3 = chuk_llm.ask_sync("Explain quantum computing", provider="openai", max_tokens=20)
        print(f"Max tokens 20: {response3}")
        
    except Exception as e:
        print(f"❌ Parameter override test failed: {e}")
    
    section("Provider Comparison")
    try:
        comparison = chuk_llm.compare_providers(
            "What's the capital of France?",
            providers=["openai", "anthropic", "groq"]
        )
        
        for provider, response in comparison.items():
            status = "✅" if not response.startswith("Error") else "❌"
            print(f"{status} {provider}: {response}")
            
    except Exception as e:
        print(f"❌ Provider comparison failed: {e}")
    
    section("Quick Question Utility")
    try:
        response = chuk_llm.quick_question("What's 2+2?")
        print(f"Quick question result: {response}")
    except Exception as e:
        print(f"❌ Quick question failed: {e}")

def test_streaming():
    """Test streaming functionality"""
    header("Streaming Functionality")
    
    section("Basic Streaming")
    try:
        print("Streaming response: ", end="", flush=True)
        chunk_count = 0
        for chunk in chuk_llm.stream_sync("Count from 1 to 5", provider="openai", max_tokens=50):
            print(chunk, end="", flush=True)
            chunk_count += 1
        print(f"\n(Received {chunk_count} chunks)")
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")

def test_error_handling():
    """Test error handling and diagnostics"""
    header("Error Handling & Diagnostics")
    
    section("Invalid Provider")
    try:
        response = chuk_llm.ask_sync("Hello", provider="nonexistent")
        print(f"❌ Should have failed: {response}")
    except Exception as e:
        print(f"✅ Correctly caught invalid provider: {type(e).__name__}")
    
    section("Health Check")
    try:
        health = chuk_llm.health_check_sync()
        print(f"Health check status: {health.get('status', 'Unknown')}")
        print(f"Total clients: {health.get('total_clients', 0)}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    section("Metrics")
    metrics = chuk_llm.get_metrics()
    if metrics:
        print(f"Total requests: {metrics.get('total_requests', 0)}")
        print(f"Average duration: {metrics.get('average_duration', 0):.2f}s")
        print(f"Error rate: {metrics.get('error_rate', 0):.1%}")
    else:
        print("No metrics available (middleware not enabled)")

def test_conversation_management():
    """Test conversation management"""
    header("Conversation Management")
    
    section("Conversation Context")
    try:
        import asyncio
        
        async def test_conversation():
            async with chuk_llm.conversation(provider="openai") as chat:
                response1 = await chat.say("I'm thinking of a number between 1 and 10.")
                print(f"User: I'm thinking of a number between 1 and 10.")
                print(f"AI: {response1}")
                
                response2 = await chat.say("It's 7. Did you guess correctly?")
                print(f"User: It's 7. Did you guess correctly?")
                print(f"AI: {response2}")
                
                stats = chat.get_stats()
                print(f"Conversation stats: {stats['total_messages']} messages, ~{stats['estimated_tokens']:.0f} tokens")
        
        asyncio.run(test_conversation())
        
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")

def run_comprehensive_demo():
    """Run the complete configuration demo"""
    print("🚀 ChukLLM End-to-End Configuration Demo")
    print("🚀 " + "="*58)
    
    try:
        test_basic_configuration()
        test_global_configuration()
        test_provider_functions()
        test_global_aliases()
        test_inheritance_and_features()
        test_advanced_features()
        test_streaming()
        test_conversation_management()
        test_error_handling()
        
        print(f"\n🎉 Demo Complete!")
        print("=" * 60)
        print("✅ If most tests show ✅, ChukLLM configuration is working correctly!")
        print("💡 Any ❌ errors indicate missing API keys or provider issues.")
        print("🔧 Use chuk_llm.print_diagnostics() for detailed troubleshooting.")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    run_comprehensive_demo()