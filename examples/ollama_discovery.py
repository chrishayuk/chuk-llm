#!/usr/bin/env python3
"""
Practical ChukLLM Ollama Discovery Example

This script demonstrates discovery working with your real chuk_llm.yaml config
and the actual models you have installed. No test configs, just real usage!

What this shows:
1. Uses your existing chuk_llm.yaml configuration
2. Shows static vs discovered models
3. Tests with real models you have (like devstral, qwen3:32b, etc.)
4. Demonstrates transparent model resolution
5. Shows how the system prompt generator works with discovered models

Prerequisites:
- Your chuk_llm.yaml config file
- Ollama running with models installed
"""

import asyncio
import sys
from pathlib import Path

async def check_ollama_status():
    """Check Ollama status and available models"""
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = [model_data["name"] for model_data in data.get("models", [])]
            return True, models
            
    except Exception as e:
        return False, str(e)

async def demonstrate_discovery():
    """Demonstrate discovery with real config and models"""
    
    print("🚀 ChukLLM Ollama Discovery - Real World Demo")
    print("=" * 55)
    
    # Check Ollama first
    print("🔍 Checking Ollama status...")
    ollama_running, ollama_data = await check_ollama_status()
    
    if not ollama_running:
        print(f"❌ Ollama not available: {ollama_data}")
        print("\nMake sure Ollama is running: ollama serve")
        return False
    
    available_models = ollama_data
    print(f"✅ Ollama running with {len(available_models)} models")
    print(f"   📋 Your models: {', '.join(available_models[:5])}")
    if len(available_models) > 5:
        print(f"       ... and {len(available_models) - 5} more")
    
    # Load ChukLLM config
    print("\n📄 Loading ChukLLM configuration...")
    
    try:
        from chuk_llm.configuration import get_config
        from chuk_llm.llm.client import get_client, get_provider_info
        
        config_manager = get_config()
        ollama_provider = config_manager.get_provider("ollama")
        
        print(f"✅ Config loaded successfully")
        print(f"   🏠 Default model: {ollama_provider.default_model}")
        print(f"   📋 Static models: {ollama_provider.models}")
        print(f"   🔧 Discovery enabled: {ollama_provider.extra.get('dynamic_discovery', {}).get('enabled', False)}")
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        print("\nMake sure your chuk_llm.yaml file is in the right location:")
        print("  • Working directory: ./chuk_llm.yaml")
        print("  • ChukLLM package: chuk_llm/chuk_llm.yaml")
        print("  • Environment: CHUK_LLM_CONFIG=/path/to/config.yaml")
        return False
    
    # Test static model access
    print("\n🔧 Testing Static Model Access...")
    
    static_models = set(ollama_provider.models)
    for model_name in list(static_models)[:2]:  # Test first 2 static models
        try:
            client = get_client("ollama", model=model_name)
            model_info = client.get_model_info()
            features = model_info.get("features", [])
            
            print(f"   ✅ {model_name}: {len(features)} features")
            print(f"      🎯 Capabilities: {', '.join(features[:4])}")
            
        except Exception as e:
            print(f"   ❌ {model_name}: {e}")
    
    # Test model aliases
    print("\n🏷️  Testing Model Aliases...")
    
    aliases = ollama_provider.model_aliases
    for alias, target in list(aliases.items())[:3]:
        try:
            client = get_client("ollama", model=alias)
            print(f"   ✅ {alias} → {client.model}")
        except Exception as e:
            print(f"   ❌ {alias}: {e}")
    
    # Find models for discovery testing
    print("\n🔍 Testing Dynamic Discovery...")
    
    # Find models that exist in Ollama but not in static config
    discovery_candidates = []
    for model in available_models:
        # Remove :latest suffix for comparison
        base_name = model.replace(":latest", "")
        if base_name not in static_models and model not in static_models:
            discovery_candidates.append(model)
    
    if discovery_candidates:
        print(f"   🎯 Found {len(discovery_candidates)} models for discovery testing")
        
        # Test discovery with a few models
        for test_model in discovery_candidates[:3]:
            print(f"\n   🧪 Testing discovery: {test_model}")
            
            try:
                # This should trigger discovery if enabled
                client = get_client("ollama", model=test_model)
                model_info = client.get_model_info()
                
                print(f"      ✅ Discovery successful!")
                print(f"      📋 Resolved to: {client.model}")
                
                # Check inferred capabilities
                features = model_info.get("features", [])
                if features:
                    print(f"      🎯 Inferred features: {', '.join(features[:4])}")
                
                context_length = model_info.get("max_context_length")
                if context_length:
                    print(f"      📏 Context length: {context_length:,}")
                
            except Exception as e:
                print(f"      ❌ Discovery failed: {e}")
                
    else:
        print("   ℹ️  All Ollama models are already in static config")
        print("      (This means discovery is working - static models take precedence)")
    
    # Test ChukLLM API integration
    print("\n🤖 Testing ChukLLM API Integration...")
    
    try:
        from chuk_llm import ask
        
        # Use a model that should be available
        test_models = []
        
        # Try static models first
        for model in ollama_provider.models:
            if any(base in model for base in ["llama", "qwen", "granite"]):
                test_models.append(model)
                break
        
        # Try discovered models if no static model worked
        if not test_models and discovery_candidates:
            for model in discovery_candidates:
                if any(base in model.lower() for base in ["llama", "qwen", "granite", "phi"]):
                    test_models.append(model)
                    break
        
        if test_models:
            test_model = test_models[0]
            print(f"   🧪 Testing with model: {test_model}")
            
            response = await ask(
                "What is the capital of France? Answer in one sentence.",
                provider="ollama",
                model=test_model
            )
            
            print(f"   ✅ Response: {response[:100]}...")
            
        else:
            print("   ⚠️  No suitable models found for testing")
            
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        print("      (This might be normal if models aren't loaded)")
    
    # Test System Prompt Generator
    print("\n✨ Testing System Prompt Generator...")
    
    try:
        from chuk_llm.llm.system_prompt_generator import generate_system_prompt
        
        # Create example tools
        example_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        # Test with Ollama provider using convenience function
        prompt = generate_system_prompt(
            tools=example_tools,
            user_prompt="You are a helpful weather assistant.",
            provider="ollama",
            model=ollama_provider.default_model
        )
        
        print("   ✅ Generated system prompt:")
        print("   " + "─" * 50)
        # Show first few lines
        lines = prompt.split('\n')
        for line in lines[:8]:
            print(f"   {line}")
        if len(lines) > 8:
            print(f"   ... ({len(lines) - 8} more lines)")
        print("   " + "─" * 50)
        
    except Exception as e:
        print(f"   ❌ System prompt generation failed: {e}")
        import traceback
        print(f"   📋 Error details: {traceback.format_exc()}")
    
    # Show provider info summary
    print("\n📊 Provider Information Summary...")
    
    try:
        provider_info = get_provider_info("ollama")
        
        if not provider_info.get("error"):
            print(f"   📋 Total models: {len(provider_info.get('available_models', []))}")
            print(f"   🔧 Discovery enabled: {provider_info.get('discovery_enabled', False)}")
            print(f"   ⚙️  Default model: {provider_info.get('model')}")
            
            supports = provider_info.get('supports', {})
            capabilities = [k for k, v in supports.items() if v]
            print(f"   🎯 Supported features: {', '.join(capabilities)}")
            
            if 'discovery_stats' in provider_info:
                stats = provider_info['discovery_stats']
                if stats:
                    print(f"   📈 Discovery stats: {stats}")
        else:
            print(f"   ❌ Provider info error: {provider_info['error']}")
            
    except Exception as e:
        print(f"   ❌ Provider info failed: {e}")
    
    print("\n" + "=" * 55)
    print("✨ Demo completed successfully!")
    
    print("\n💡 Key Takeaways:")
    print("   • Static models from config are always available")
    print("   • Discovery adds models transparently when requested")
    print("   • Model aliases work for both static and discovered models")
    print("   • Capabilities are inferred automatically for discovered models")
    print("   • Existing ChukLLM APIs work unchanged")
    print("   • System prompt generator adapts to provider capabilities")
    
    return True

async def show_config_location():
    """Show where ChukLLM is looking for config"""
    print("\n🔍 Config File Location Detection...")
    
    try:
        from chuk_llm.configuration.unified_config import UnifiedConfigManager
        
        # Create a config manager to see where it looks
        manager = UnifiedConfigManager()
        config_file = manager._find_config_file()
        
        if config_file:
            print(f"   ✅ Found config: {config_file}")
            print(f"   📏 Size: {config_file.stat().st_size:,} bytes")
        else:
            print("   ❌ No config file found")
            print("\n   📍 ChukLLM looks in these locations (in order):")
            print("      1. CHUK_LLM_CONFIG environment variable")
            print("      2. ./chuk_llm.yaml (working directory)")
            print("      3. chuk_llm/chuk_llm.yaml (package directory)")
            print("      4. ./providers.yaml (fallback)")
            print("      5. ~/.chuk_llm/config.yaml (user config)")
            
    except Exception as e:
        print(f"   ❌ Could not detect config location: {e}")

async def main():
    """Main demo function"""
    print("🎯 ChukLLM Ollama Discovery - Real World Demo")
    print("This demonstrates discovery with your actual config and models\n")
    
    # Show config location
    await show_config_location()
    
    # Run main demo
    success = await demonstrate_discovery()
    
    if success:
        print("\n🎉 Demo completed! Discovery is working with your real setup.")
    else:
        print("\n💥 Demo encountered issues. Check the output above.")
    
    print("\n📚 Next Steps:")
    print("   • Try asking questions with different models")
    print("   • Use model aliases for convenience")
    print("   • Experiment with the system prompt generator")
    print("   • Add your own models to the static config for faster access")

if __name__ == "__main__":
    asyncio.run(main())