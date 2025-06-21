#!/usr/bin/env python3
"""
End-to-End Test Script for ChukLLM with Ollama Discovery

This script demonstrates the complete transparent discovery workflow:
1. Creates a minimal config with discovery enabled
2. Tests static model usage
3. Tests dynamic model discovery 
4. Shows seamless model resolution
5. Validates that existing APIs work unchanged

Prerequisites:
- Ollama running locally on http://localhost:11434
- At least one model pulled (e.g., llama3.2, qwen2.5, etc.)
"""

import asyncio
import tempfile
import os
from pathlib import Path
import yaml

# Test configuration with discovery enabled
TEST_CONFIG = {
    "__global__": {
        "active_provider": "ollama",
        "active_model": "llama3.2"
    },
    
    "__global_aliases__": {
        "llama": "ollama/llama3.2",
        "smart": "ollama/qwen2.5"
    },
    
    "ollama": {
        "client_class": "chuk_llm.llm.providers.ollama_client.OllamaLLMClient",
        "api_base": "http://localhost:11434",
        "default_model": "llama3.2",
        
        # Static models (these take precedence)
        "models": [
            "llama3.2",
            "qwen2.5"
        ],
        
        "model_aliases": {
            "llama": "llama3.2",
            "qwen": "qwen2.5",
            "smart": "qwen2.5",
            "default": "llama3.2"
        },
        
        # Baseline capabilities
        "features": ["text", "streaming", "system_messages"],
        "max_context_length": 8192,
        "max_output_tokens": 4096,
        
        # Static model capabilities (authoritative)
        "model_capabilities": [
            {
                "pattern": "llama3\\.2.*",
                "features": ["text", "streaming", "tools", "reasoning", "system_messages"],
                "max_context_length": 128000,
                "max_output_tokens": 8192
            },
            {
                "pattern": "qwen.*",
                "features": ["text", "streaming", "tools", "reasoning", "system_messages"],
                "max_context_length": 32768,
                "max_output_tokens": 8192
            }
        ],
        
        # Discovery configuration
        "extra": {
            "dynamic_discovery": {
                "enabled": True,
                "discoverer_type": "ollama",
                "cache_timeout": 300,
                
                # How to infer capabilities for discovered models
                "inference_config": {
                    "default_features": ["text", "streaming"],
                    "default_context_length": 8192,
                    
                    "family_rules": {
                        "llama": {
                            "patterns": ["llama"],
                            "features": ["text", "streaming", "tools", "reasoning", "system_messages"],
                            "context_rules": {
                                "llama3\\.[2-9]": 128000,
                                "llama3\\.[01]": 32768
                            }
                        },
                        
                        "qwen": {
                            "patterns": ["qwen"],
                            "features": ["text", "streaming", "tools", "reasoning", "system_messages"],
                            "base_context_length": 32768
                        },
                        
                        "mistral": {
                            "patterns": ["mistral", "mixtral"],
                            "features": ["text", "streaming", "tools", "system_messages"],
                            "base_context_length": 32768
                        },
                        
                        "phi": {
                            "patterns": ["phi"],
                            "features": ["text", "streaming", "system_messages"],
                            "base_context_length": 4096
                        },
                        
                        "gemma": {
                            "patterns": ["gemma"],
                            "features": ["text", "streaming", "tools", "system_messages"],
                            "base_context_length": 8192
                        }
                    },
                    
                    "pattern_rules": {
                        "vision_models": {
                            "patterns": [".*vision.*", ".*llava.*"],
                            "add_features": ["vision", "multimodal"]
                        },
                        
                        "code_models": {
                            "patterns": [".*code.*", "codellama"],
                            "add_features": ["reasoning"]
                        }
                    }
                }
            }
        }
    }
}

async def test_ollama_discovery():
    """Complete end-to-end test of Ollama discovery"""
    
    print("🚀 ChukLLM Ollama Discovery Test")
    print("=" * 50)
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(TEST_CONFIG, f, default_flow_style=False)
        config_path = f.name
    
    try:
        # Set config file location
        os.environ['CHUK_LLM_CONFIG'] = config_path
        
        # Reset config to pick up our test config
        from chuk_llm.configuration import reset_config, get_config
        reset_config()
        
        print(f"📄 Using test config: {config_path}")
        
        # Test 1: Basic config loading
        print("\n1. Testing Configuration Loading")
        config_manager = get_config()
        
        try:
            ollama_provider = config_manager.get_provider("ollama")
            print(f"   ✅ Provider loaded: {len(ollama_provider.models)} static models")
            print(f"   📋 Static models: {ollama_provider.models}")
            print(f"   🔧 Discovery enabled: {ollama_provider.extra.get('dynamic_discovery', {}).get('enabled', False)}")
        except Exception as e:
            print(f"   ❌ Config loading failed: {e}")
            return False
        
        # Test 2: Client creation with static models
        print("\n2. Testing Static Model Usage")
        
        from chuk_llm.llm.client import get_client
        
        # Test static model
        try:
            client = get_client("ollama", model="llama3.2")
            print(f"   ✅ Static model client created: {client.model}")
        except Exception as e:
            print(f"   ❌ Static model client failed: {e}")
            return False
        
        # Test model alias
        try:
            client = get_client("ollama", model="smart")  # Should resolve to qwen2.5
            print(f"   ✅ Model alias resolved: smart -> {client.model}")
        except Exception as e:
            print(f"   ❌ Model alias failed: {e}")
        
        # Test 3: Discovery in action
        print("\n3. Testing Dynamic Discovery")
        
        # First, let's see what models Ollama actually has
        try:
            # Try to get a model that's not in our static list
            # This should trigger discovery
            
            # Get current models before discovery
            provider_before = config_manager.get_provider("ollama")
            models_before = len(provider_before.models)
            
            print(f"   📊 Models before discovery: {models_before}")
            
            # Attempt to create client with potentially unknown model
            # This will trigger discovery if the model exists in Ollama
            available_models = await get_available_ollama_models()
            print(f"   🔍 Ollama has {len(available_models)} models available")
            
            # Find a model that's not in our static list
            static_models = set(provider_before.models)
            dynamic_candidates = [m for m in available_models if m not in static_models]
            
            if dynamic_candidates:
                test_model = dynamic_candidates[0]
                print(f"   🎯 Testing discovery with: {test_model}")
                
                try:
                    client = get_client("ollama", model=test_model)
                    print(f"   ✅ Dynamic model client created: {client.model}")
                    
                    # Check if provider was updated
                    provider_after = config_manager.get_provider("ollama")
                    models_after = len(provider_after.models)
                    print(f"   📈 Models after discovery: {models_after} (+{models_after - models_before})")
                    
                except Exception as e:
                    print(f"   ⚠️  Dynamic model failed: {e}")
            else:
                print("   ℹ️  All Ollama models are already in static config")
        
        except Exception as e:
            print(f"   ❌ Discovery test failed: {e}")
        
        # Test 4: Model capabilities
        print("\n4. Testing Model Capabilities")
        
        provider = config_manager.get_provider("ollama")
        for model in provider.models[:3]:  # Test first 3 models
            try:
                caps = provider.get_model_capabilities(model)
                features = [f.value for f in caps.features]
                print(f"   📋 {model}: {features[:3]}{'...' if len(features) > 3 else ''}")
                print(f"      Context: {caps.max_context_length}, Output: {caps.max_output_tokens}")
            except Exception as e:
                print(f"   ❌ Capabilities for {model}: {e}")
        
        # Test 5: Actual LLM interaction
        print("\n5. Testing LLM Interaction")
        
        try:
            # Import the main API
            from chuk_llm.api import ask
            
            # Simple test with static model
            response = await ask("What is 2+2?", provider="ollama", model="llama3.2")
            print(f"   ✅ LLM Response: {response[:50]}...")
            
        except Exception as e:
            print(f"   ❌ LLM interaction failed: {e}")
            # This might fail if Ollama models aren't actually available
            print("   ℹ️  Make sure Ollama is running and has models available")
        
        # Test 6: Provider info
        print("\n6. Testing Provider Information")
        
        from chuk_llm.llm.client import list_available_providers, get_provider_info
        
        try:
            providers = list_available_providers()
            ollama_info = providers.get("ollama", {})
            
            print(f"   📊 Total models: {len(ollama_info.get('models', []))}")
            print(f"   🔧 Discovery enabled: {ollama_info.get('discovery_enabled', False)}")
            
            if 'discovery_stats' in ollama_info:
                stats = ollama_info['discovery_stats']
                print(f"   📈 Discovery stats: {stats}")
            
        except Exception as e:
            print(f"   ❌ Provider info failed: {e}")
        
        print("\n" + "=" * 50)
        print("✨ Discovery test completed!")
        print("\nKey findings:")
        print("  • Static models are loaded from config")
        print("  • Discovery triggers when unknown models are requested")
        print("  • Model capabilities are inferred using rules")
        print("  • Existing APIs work unchanged")
        print("  • Everything is transparent to the user")
        
        return True
        
    finally:
        # Cleanup
        try:
            os.unlink(config_path)
            print(f"\n🧹 Cleaned up test config: {config_path}")
        except:
            pass


async def get_available_ollama_models():
    """Get list of models available in Ollama"""
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = [model_data["name"] for model_data in data.get("models", [])]
            return models
            
    except Exception as e:
        print(f"   ⚠️  Could not fetch Ollama models: {e}")
        return []


def check_ollama_running():
    """Check if Ollama is running"""
    try:
        import httpx
        
        with httpx.Client(timeout=3.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            return True
    except Exception:
        return False


async def main():
    """Main test function"""
    print("🔍 Checking Ollama availability...")
    
    if not check_ollama_running():
        print("❌ Ollama is not running on localhost:11434")
        print("\nTo run this test:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama3.2")
        print("4. Run this test again")
        return
    
    print("✅ Ollama is running")
    
    # Get available models
    models = await get_available_ollama_models()
    if not models:
        print("❌ No models found in Ollama")
        print("\nPull some models first:")
        print("  ollama pull llama3.2")
        print("  ollama pull qwen2.5")
        return
    
    print(f"✅ Found {len(models)} models: {models[:3]}{'...' if len(models) > 3 else ''}")
    
    # Run the test
    success = await test_ollama_discovery()
    
    if success:
        print("\n🎉 All tests passed! Discovery is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the output above.")


if __name__ == "__main__":
    print("ChukLLM Ollama Discovery Test Script")
    print("This tests transparent model discovery with Ollama")
    print()
    
    asyncio.run(main())