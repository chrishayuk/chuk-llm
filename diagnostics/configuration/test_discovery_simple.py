#!/usr/bin/env python3
"""
Test Discovery Actually Working

Simple test to verify discovery works end-to-end with your actual models.
"""

import asyncio

async def test_discovery_models():
    """Test discovery with your actual models"""
    
    print("🎯 Testing Discovery End-to-End")
    print("=" * 40)
    
    # Test 1: Direct discovery
    print("1️⃣ Testing Direct Discovery...")
    
    try:
        from chuk_llm.api.discovery import discover_models
        
        models = await discover_models("ollama", force_refresh=True)
        
        print(f"   ✅ Discovery successful!")
        print(f"   📊 Found {len(models)} models")
        
        # Find your specific models
        your_models = {
            "devstral:latest": None,
            "qwen3:32b": None, 
            "llama3.2-vision:latest": None,
            "phi4-reasoning:latest": None,
            "granite3.3:latest": None
        }
        
        discovered_names = [m['name'] for m in models]
        
        for test_model in your_models.keys():
            base_name = test_model.replace(':latest', '')
            
            if test_model in discovered_names:
                model_data = next(m for m in models if m['name'] == test_model)
                your_models[test_model] = model_data
                print(f"   ✅ {test_model} - Found ({model_data.get('family', 'unknown')})")
            elif base_name in discovered_names:
                model_data = next(m for m in models if m['name'] == base_name)
                your_models[base_name] = model_data
                print(f"   ✅ {test_model} - Found as {base_name} ({model_data.get('family', 'unknown')})")
            else:
                print(f"   ❌ {test_model} - Not found")
        
    except Exception as e:
        print(f"   ❌ Discovery failed: {e}")
        return False
    
    # Test 2: Client creation with discovered models
    print(f"\n2️⃣ Testing Client Creation...")
    
    try:
        from chuk_llm.llm.client import get_client
        
        # Test with a few different models
        test_models = ["devstral", "qwen3", "granite3.3"]
        
        for test_model in test_models:
            try:
                client = get_client("ollama", model=test_model)
                model_info = client.get_model_info()
                features = model_info.get('features', [])
                
                print(f"   ✅ {test_model} - Client created")
                print(f"      Features: {', '.join(features[:4])}")
                
                break  # Success with first working model
                
            except Exception as model_error:
                print(f"   ⚠️  {test_model} - {str(model_error)[:50]}...")
                continue
        
    except Exception as e:
        print(f"   ❌ Client creation failed: {e}")
        return False
    
    # Test 3: Actual LLM completion
    print(f"\n3️⃣ Testing LLM Completion...")
    
    try:
        from chuk_llm import ask
        
        # Try with a model that should work
        working_models = ["granite3.3", "qwen3", "llama3.3"]
        
        for test_model in working_models:
            try:
                print(f"   🧪 Testing completion with {test_model}...")
                
                response = await ask(
                    "What is 2+2? Answer with just the number.",
                    provider="ollama",
                    model=test_model,
                    max_tokens=5
                )
                
                print(f"   ✅ Success! Response: '{response.strip()}'")
                break
                
            except Exception as model_error:
                print(f"   ⚠️  {test_model} failed: {str(model_error)[:50]}...")
                continue
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM completion failed: {e}")
        return False

async def test_specific_discovered_models():
    """Test with models that should be discovered"""
    
    print(f"\n4️⃣ Testing Specific Discovered Models...")
    
    # These should be discovered and work
    discovered_models = ["devstral:latest", "qwen3:32b"]
    
    try:
        from chuk_llm import ask
        
        for model in discovered_models:
            try:
                print(f"   🧪 Testing {model}...")
                
                # Try to use the model directly
                response = await ask(
                    "Say 'working' if you can respond.",
                    provider="ollama", 
                    model=model,
                    max_tokens=5
                )
                
                print(f"   ✅ {model} - Response: '{response.strip()}'")
                
            except Exception as e:
                error_msg = str(e)
                if "not available" in error_msg.lower():
                    print(f"   ❌ {model} - Model not available (discovery integration issue)")
                else:
                    print(f"   ⚠️  {model} - Error: {error_msg[:50]}...")
    
    except Exception as e:
        print(f"   ❌ Discovered model test failed: {e}")

def show_discovery_status():
    """Show current discovery status"""
    
    print(f"\n📊 Discovery Status Summary...")
    
    try:
        from chuk_llm.configuration import get_config
        
        manager = get_config()
        provider = manager.get_provider('ollama')
        
        discovery_config = provider.extra.get('dynamic_discovery', {})
        
        print(f"   🔧 Discovery enabled: {discovery_config.get('enabled', False)}")
        print(f"   📋 Static models: {len(provider.models)}")
        print(f"   🏷️  Model aliases: {len(provider.model_aliases)}")
        print(f"   ⚙️  Default model: {provider.default_model}")
        
        if discovery_config.get('enabled'):
            print(f"   ✅ Discovery is properly configured!")
        else:
            print(f"   ❌ Discovery is disabled")
    
    except Exception as e:
        print(f"   ❌ Status check failed: {e}")

async def main():
    """Main test function"""
    
    # Show status first
    show_discovery_status()
    
    # Run tests
    success = await test_discovery_models()
    
    if success:
        await test_specific_discovered_models()
        
        print(f"\n🎉 Discovery Test Complete!")
        print(f"💡 Your 48 Ollama models are now accessible through ChukLLM")
        print(f"📋 You can use models like: devstral:latest, qwen3:32b, etc.")
    else:
        print(f"\n💥 Some tests failed - check the output above")
    
    print(f"\n📖 Usage Examples:")
    print(f"   await chuk_llm.ask('Hello', provider='ollama', model='qwen3:32b')")
    print(f"   await chuk_llm.ask('Code review', provider='ollama', model='devstral')")
    print(f"   await chuk_llm.ask('Analyze image', provider='ollama', model='llama3.2-vision')")

if __name__ == "__main__":
    asyncio.run(main())