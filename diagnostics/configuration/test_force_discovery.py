#!/usr/bin/env python3
"""
Force discovery test - actually trigger the discovery process
"""

import asyncio
import sys

async def force_ollama_discovery():
    """Force Ollama discovery to actually happen"""
    
    print("🔍 Force Ollama Discovery")
    print("=" * 40)
    
    try:
        # Import the configuration system
        from chuk_llm.configuration import get_config
        
        config = get_config()
        
        print("📋 Step 1: Check current Ollama models...")
        ollama_provider = config.get_provider('ollama')
        initial_models = ollama_provider.models.copy()
        print(f"   Initial models: {len(initial_models)}")
        
        # Show some examples
        for model in initial_models[:5]:
            print(f"     • {model}")
        if len(initial_models) > 5:
            print(f"     ... and {len(initial_models) - 5} more")
        
        print(f"\n🔍 Step 2: Manually trigger discovery...")
        
        # Method 1: Try using the discovery API directly
        try:
            from chuk_llm.api.discovery import discover_models_sync
            
            print("   Attempting discovery via API...")
            discovered = discover_models_sync('ollama', force_refresh=True)
            
            print(f"   ✅ Discovery API returned {len(discovered)} models")
            
            # Show discovered models
            if discovered:
                print("   📋 Discovered models:")
                for model in discovered[:10]:
                    name = model.get('name', 'unknown')
                    features = model.get('features', [])
                    print(f"     • {name} - {features}")
                if len(discovered) > 10:
                    print(f"     ... and {len(discovered) - 10} more")
            
        except ImportError as e:
            print(f"   ⚠️  Discovery API not available: {e}")
        except Exception as e:
            print(f"   ⚠️  Discovery API failed: {e}")
        
        # Method 2: Try triggering via configuration manager
        print(f"\n🔧 Step 3: Trigger via configuration manager...")
        
        try:
            # This should trigger discovery internally
            test_model = config._ensure_model_available('ollama', 'qwen3-32b')
            if test_model:
                print(f"   ✅ Model resolution worked: {test_model}")
            else:
                print(f"   ❌ Model resolution failed")
                
        except Exception as e:
            print(f"   ⚠️  Configuration manager discovery failed: {e}")
        
        # Method 3: Direct discovery manager approach
        print(f"\n⚙️  Step 4: Direct discovery manager...")
        
        try:
            from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
            from chuk_llm.llm.discovery.providers import OllamaModelDiscoverer
            
            print("   Creating Ollama discoverer...")
            
            # Create discoverer with localhost config
            discoverer = OllamaModelDiscoverer(api_base="http://localhost:11434")
            
            # Create discovery manager
            manager = UniversalModelDiscoveryManager(
                provider_name="ollama",
                discoverer=discoverer
            )
            
            print("   Running discovery...")
            models = await manager.discover_models(force_refresh=True)
            
            print(f"   ✅ Direct discovery found {len(models)} models")
            
            # Show what was found
            if models:
                print("   📋 Direct discovery results:")
                for model in models[:10]:
                    print(f"     • {model.name} - {[f.value for f in model.capabilities]}")
                if len(models) > 10:
                    print(f"     ... and {len(models) - 10} more")
                
                # Check if our target models are there
                target_models = ['devstral', 'qwen3-32b', 'phi4-reasoning']
                found_targets = []
                
                for target in target_models:
                    for model in models:
                        if target.lower() in model.name.lower():
                            found_targets.append((target, model.name))
                            break
                
                if found_targets:
                    print("   🎯 Target models found:")
                    for target, actual in found_targets:
                        print(f"     • {target} -> {actual}")
                else:
                    print("   ❌ Target models not found in discovery")
            
        except ImportError as e:
            print(f"   ⚠️  Discovery engine not available: {e}")
        except Exception as e:
            print(f"   ⚠️  Direct discovery failed: {e}")
        
        # Method 4: Check what models are actually available in Ollama
        print(f"\n🌐 Step 5: Check Ollama directly...")
        
        try:
            import httpx
            
            print("   Querying Ollama API directly...")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                
                if response.status_code == 200:
                    data = response.json()
                    ollama_models = data.get('models', [])
                    
                    print(f"   ✅ Ollama has {len(ollama_models)} models available")
                    
                    if ollama_models:
                        print("   📋 Available in Ollama:")
                        for model_data in ollama_models[:10]:
                            name = model_data.get('name', 'unknown')
                            size = model_data.get('size', 0)
                            size_gb = f"{size / (1024**3):.1f}GB" if size else "unknown"
                            print(f"     • {name} ({size_gb})")
                        if len(ollama_models) > 10:
                            print(f"     ... and {len(ollama_models) - 10} more")
                        
                        # Check for our targets
                        target_models = ['devstral', 'qwen3', 'phi4']
                        found_in_ollama = []
                        
                        for target in target_models:
                            for model_data in ollama_models:
                                name = model_data.get('name', '')
                                if target.lower() in name.lower():
                                    found_in_ollama.append((target, name))
                        
                        if found_in_ollama:
                            print("   🎯 Target models in Ollama:")
                            for target, actual in found_in_ollama:
                                print(f"     • {target} -> {actual}")
                            return found_in_ollama
                        else:
                            print("   ❌ Target models not found in Ollama")
                            print("   💡 You may need to pull them first:")
                            print("      ollama pull devstral")
                            print("      ollama pull qwen3:32b")
                            print("      ollama pull phi4:reasoning")
                    
                else:
                    print(f"   ❌ Ollama API error: {response.status_code}")
                    
        except Exception as e:
            print(f"   ⚠️  Ollama API check failed: {e}")
            print("   💡 Make sure Ollama is running: ollama serve")
        
        return []
        
    except Exception as e:
        print(f"❌ Force discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_function_generation_with_discovered():
    """Test function generation after discovery"""
    
    print(f"\n🔧 Testing Function Generation")
    print("=" * 40)
    
    try:
        from chuk_llm.api.providers import _generate_functions_for_models
        from chuk_llm.configuration import get_config
        
        config = get_config()
        ollama_provider = config.get_provider('ollama')
        
        # Test with a known model first
        print("   Testing with known model 'llama3.3'...")
        
        test_functions = _generate_functions_for_models('ollama', ollama_provider, ['llama3.3'])
        
        if test_functions:
            print(f"   ✅ Generated {len(test_functions)} functions for test model")
            
            # Show examples
            for name in sorted(test_functions.keys())[:3]:
                print(f"     • {name}")
        else:
            print("   ❌ No functions generated for test model")
        
        # Now test with potential discovered models
        test_discovered = ['devstral:latest', 'qwen3:32b', 'phi4:reasoning']
        
        print(f"\n   Testing with potential discovered models...")
        
        for model_name in test_discovered:
            print(f"   Generating functions for '{model_name}'...")
            
            functions = _generate_functions_for_models('ollama', ollama_provider, [model_name])
            
            if functions:
                print(f"     ✅ Generated {len(functions)} functions")
                
                # Show what was generated
                for func_name in sorted(functions.keys())[:2]:
                    print(f"       • {func_name}")
            else:
                print(f"     ❌ No functions generated")
        
    except Exception as e:
        print(f"   ❌ Function generation test failed: {e}")


def main():
    """Main test function"""
    
    print("🚀 Force Discovery Test")
    print("=" * 50)
    
    # Run async discovery
    found_models = asyncio.run(force_ollama_discovery())
    
    # Test function generation
    asyncio.run(test_function_generation_with_discovered())
    
    print(f"\n📊 Summary")
    print("=" * 20)
    
    if found_models:
        print(f"✅ Found {len(found_models)} target models in Ollama")
        print("💡 Next steps:")
        print("   1. Update configuration to include these models")
        print("   2. Regenerate functions")
        print("   3. Test function calls")
    else:
        print("❌ No target models found")
        print("💡 Solutions:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Pull the models you want:")
        print("      ollama pull devstral")
        print("      ollama pull qwen3:32b") 
        print("      ollama pull phi4:reasoning")
        print("   3. Run this test again")


if __name__ == "__main__":
    main()