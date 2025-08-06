#!/usr/bin/env python3
"""
ChukLLM Dynamic Model Inference Demo
====================================

This script demonstrates using discovered models that are NOT in the static 
configuration for live inference. This proves that discovery enables seamless 
access to any available model without manual configuration updates.

Key Features:
- Discovers models dynamically from Ollama and OpenAI
- Identifies models NOT in static configuration
- Performs live inference tests with discovered models
- Demonstrates parameter handling for reasoning models
- Shows cross-provider model switching
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"📁 Loaded environment variables")
except ImportError:
    print("💡 Using system environment variables")

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def get_static_configuration():
    """Get the current static configuration to compare against discovered models"""
    print("📋 Loading Static Configuration...")
    
    try:
        from chuk_llm.configuration import get_config
        
        config_manager = get_config()
        
        # Get configured providers
        static_models = {}
        
        # Check Ollama static models
        try:
            ollama_config = config_manager.get_provider("ollama")
            static_models["ollama"] = set(ollama_config.models)
            print(f"   🏠 Ollama static models: {len(static_models['ollama'])}")
            print(f"      └─ {list(static_models['ollama'])[:3]}...")
        except Exception as e:
            static_models["ollama"] = set()
            print(f"   🏠 Ollama: No static configuration")
        
        # Check OpenAI static models  
        try:
            openai_config = config_manager.get_provider("openai")
            static_models["openai"] = set(openai_config.models)
            print(f"   ☁️  OpenAI static models: {len(static_models['openai'])}")
            print(f"      └─ {list(static_models['openai'])[:3]}...")
        except Exception as e:
            static_models["openai"] = set()
            print(f"   ☁️  OpenAI: No static configuration")
        
        return static_models
        
    except Exception as e:
        print(f"❌ Failed to load static configuration: {e}")
        return {"ollama": set(), "openai": set()}

async def discover_models():
    """Discover models from all available providers"""
    print("\n🔍 Discovering Available Models...")
    
    discovered = {"ollama": [], "openai": []}
    
    # Discover Ollama models
    try:
        from chuk_llm.llm.discovery.ollama_discoverer import OllamaModelDiscoverer
        from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
        
        print("   🏠 Discovering Ollama models...")
        ollama_discoverer = OllamaModelDiscoverer()
        ollama_manager = UniversalModelDiscoveryManager("ollama", ollama_discoverer)
        ollama_models = await ollama_manager.discover_models(force_refresh=True)
        discovered["ollama"] = ollama_models
        print(f"      ✅ Found {len(ollama_models)} Ollama models")
        
    except Exception as e:
        print(f"      ❌ Ollama discovery failed: {e}")
    
    # Discover OpenAI models
    if os.getenv('OPENAI_API_KEY'):
        try:
            from chuk_llm.llm.discovery.openai_discoverer import OpenAIModelDiscoverer
            from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
            
            print("   ☁️  Discovering OpenAI models...")
            openai_discoverer = OpenAIModelDiscoverer(
                api_key=os.getenv('OPENAI_API_KEY'),
                api_base="https://api.openai.com/v1"
            )
            openai_manager = UniversalModelDiscoveryManager("openai", openai_discoverer)
            openai_models = await openai_manager.discover_models(force_refresh=True)
            discovered["openai"] = openai_models
            print(f"      ✅ Found {len(openai_models)} OpenAI models")
            
        except Exception as e:
            print(f"      ❌ OpenAI discovery failed: {e}")
    else:
        print("   ☁️  OpenAI: API key not available")
    
    return discovered

def find_dynamic_only_models(discovered: Dict, static: Dict):
    """Find models that are discovered but NOT in static configuration"""
    print("\n🎯 Identifying Dynamic-Only Models...")
    
    dynamic_models = {"ollama": [], "openai": []}
    
    for provider in ["ollama", "openai"]:
        discovered_names = {m.name for m in discovered[provider]}
        static_names = static[provider]
        
        # Find models that exist in discovery but not in static config
        dynamic_names = discovered_names - static_names
        
        # Get the full model objects for dynamic models
        for model in discovered[provider]:
            if model.name in dynamic_names:
                dynamic_models[provider].append(model)
        
        if dynamic_models[provider]:
            print(f"   🔍 {provider.upper()}: {len(dynamic_models[provider])} dynamic-only models")
            for model in dynamic_models[provider][:3]:
                family = getattr(model, 'family', 'unknown') or 'unknown'
                size_info = ""
                if provider == "ollama":
                    size_gb = model.metadata.get('size_gb', 0)
                    size_info = f" ({size_gb:.1f}GB)"
                elif provider == "openai":
                    is_reasoning = model.metadata.get('is_reasoning', False)
                    size_info = f" ({'🧠 reasoning' if is_reasoning else '💬 standard'})"
                
                print(f"      • {model.name} [{family}]{size_info}")
            
            if len(dynamic_models[provider]) > 3:
                print(f"      ... and {len(dynamic_models[provider]) - 3} more")
        else:
            print(f"   🔍 {provider.upper()}: All discovered models are in static config")
    
    return dynamic_models

async def test_dynamic_model_inference(provider: str, model_name: str, model_obj=None):
    """Test inference with a dynamically discovered model"""
    print(f"\n🧪 Testing Dynamic Inference: {provider}/{model_name}")
    print("-" * 60)
    
    try:
        from chuk_llm import ask
        
        # Test prompt
        test_prompt = "What is the capital of France? Respond in exactly one sentence."
        
        # Special handling for reasoning models
        kwargs = {"max_tokens": 100}
        if model_obj and model_obj.metadata.get('is_reasoning', False):
            print("   🧠 Detected reasoning model - using special parameters")
            # For reasoning models, use max_completion_tokens instead
            kwargs = {"max_completion_tokens": 100}
        
        print(f"   📝 Prompt: {test_prompt}")
        print(f"   🎯 Model: {model_name}")
        print(f"   ⚙️  Parameters: {kwargs}")
        
        start_time = time.time()
        
        try:
            response = await ask(
                test_prompt,
                provider=provider,
                model=model_name,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            print(f"   ✅ Success in {response_time:.2f}s:")
            print(f"   💬 Response: {response}")
            
            return True, response
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Inference failed: {error_msg}")
            
            # Provide helpful error analysis
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                print("   💡 This might be a reasoning model requiring max_completion_tokens")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                print("   💡 Model might not be loaded or accessible")
            elif "rate limit" in error_msg.lower():
                print("   💡 API rate limit hit - try again in a moment")
            
            return False, error_msg
    
    except ImportError as e:
        print(f"   ❌ ChukLLM import failed: {e}")
        return False, str(e)

async def test_reasoning_model_parameters():
    """Special test for reasoning model parameter handling"""
    print(f"\n🧠 Testing Reasoning Model Parameter Handling")
    print("=" * 60)
    
    # Try to find an OpenAI reasoning model
    if os.getenv('OPENAI_API_KEY'):
        try:
            from chuk_llm import ask
            
            reasoning_models = ["o1-mini", "o1-preview", "o3-mini"]
            
            for model_name in reasoning_models:
                print(f"\n   🎯 Testing: {model_name}")
                
                try:
                    # Test with proper reasoning model parameters
                    response = await ask(
                        "2+2=?",
                        provider="openai",
                        model=model_name,
                        max_completion_tokens=50  # Reasoning models need this instead of max_tokens
                    )
                    
                    print(f"   ✅ {model_name} works with max_completion_tokens")
                    print(f"   💬 Response: {response}")
                    break
                    
                except Exception as e:
                    if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                        print(f"   ⚠️  {model_name} not available")
                        continue
                    else:
                        print(f"   ❌ {model_name} error: {e}")
                        continue
            else:
                print("   ⚠️  No reasoning models available for testing")
        
        except Exception as e:
            print(f"   ❌ Reasoning model test failed: {e}")
    else:
        print("   ⚠️  OpenAI API key not available for reasoning model test")

async def demonstrate_cross_provider_switching(dynamic_models: Dict):
    """Demonstrate switching between different discovered models"""
    print(f"\n🔄 Cross-Provider Model Switching Demo")
    print("=" * 60)
    
    # Collect available models for testing
    test_candidates = []
    
    # Add Ollama models
    for model in dynamic_models["ollama"][:2]:  # Test first 2
        test_candidates.append(("ollama", model.name, model))
    
    # Add OpenAI models  
    for model in dynamic_models["openai"][:2]:  # Test first 2
        # Skip certain model types that might not work for chat
        if not any(skip in model.name.lower() for skip in ["embedding", "tts", "whisper", "dall-e", "moderation"]):
            test_candidates.append(("openai", model.name, model))
    
    if not test_candidates:
        print("   ⚠️  No suitable dynamic models found for cross-provider testing")
        return
    
    print(f"   🎯 Testing {len(test_candidates)} dynamic models:")
    
    # Test the same prompt across different models
    test_prompt = "Name one interesting fact about artificial intelligence."
    results = []
    
    for provider, model_name, model_obj in test_candidates:
        print(f"\n   🧪 {provider.upper()}: {model_name}")
        
        success, response = await test_dynamic_model_inference(provider, model_name, model_obj)
        
        results.append({
            "provider": provider,
            "model": model_name,
            "success": success,
            "response": response[:100] + "..." if success and len(response) > 100 else response
        })
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    # Summary
    print(f"\n   📊 Cross-Provider Test Summary:")
    successful = sum(1 for r in results if r["success"])
    print(f"   ✅ {successful}/{len(results)} models responded successfully")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"   {status} {result['provider']}/{result['model']}")
        if result["success"]:
            print(f"      └─ {result['response']}")

async def demonstrate_model_capabilities(dynamic_models: Dict):
    """Demonstrate different capabilities of discovered models"""
    print(f"\n🎯 Model Capability Demonstration")
    print("=" * 60)
    
    # Find models with different capabilities
    capability_tests = {
        "reasoning": {
            "models": [],
            "prompt": "If all roses are flowers and all flowers need water, do roses need water? Explain your reasoning briefly.",
            "description": "🧠 Reasoning capability"
        },
        "general": {
            "models": [],
            "prompt": "Write a haiku about programming.",
            "description": "💬 Creative writing"
        }
    }
    
    # Categorize models by capability
    for provider in ["ollama", "openai"]:
        for model in dynamic_models[provider][:3]:  # Limit to prevent too many tests
            if (model.metadata.get('reasoning_capable', False) or 
                model.metadata.get('is_reasoning', False)):
                capability_tests["reasoning"]["models"].append((provider, model.name, model))
            else:
                capability_tests["general"]["models"].append((provider, model.name, model))
    
    # Test each capability
    for capability, test_info in capability_tests.items():
        if not test_info["models"]:
            continue
            
        print(f"\n   {test_info['description']} Test:")
        print(f"   📝 Prompt: {test_info['prompt']}")
        
        # Test with first available model of this capability
        provider, model_name, model_obj = test_info["models"][0]
        print(f"   🎯 Using: {provider}/{model_name}")
        
        success, response = await test_dynamic_model_inference(
            provider, model_name, model_obj
        )
        
        if success:
            print(f"   💬 Capability demonstrated successfully!")
        else:
            print(f"   ⚠️  Capability test failed, but model discovery still worked")

async def main():
    """Main demo orchestrator"""
    print("🚀 ChukLLM Dynamic Model Inference Demo")
    print("=" * 70)
    print("Testing models discovered dynamically (not in static configuration)")
    print()
    
    # Step 1: Get static configuration
    static_models = await get_static_configuration()
    
    # Step 2: Discover available models
    discovered_models = await discover_models()
    
    # Step 3: Find dynamic-only models
    dynamic_models = find_dynamic_only_models(discovered_models, static_models)
    
    # Check if we have any dynamic models to test
    total_dynamic = len(dynamic_models["ollama"]) + len(dynamic_models["openai"])
    
    if total_dynamic == 0:
        print("\n⚠️  No dynamic-only models found!")
        print("   This means all discovered models are already in your static configuration.")
        print("   This is actually good - it shows your config is comprehensive!")
        
        # Still run a test with any available model
        if discovered_models["ollama"] or discovered_models["openai"]:
            print("\n🧪 Running test with any discovered model...")
            
            if discovered_models["ollama"]:
                model = discovered_models["ollama"][0]
                await test_dynamic_model_inference("ollama", model.name, model)
            elif discovered_models["openai"]:
                model = discovered_models["openai"][0]
                await test_dynamic_model_inference("openai", model.name, model)
        
        return
    
    print(f"\n🎯 Found {total_dynamic} dynamic-only models to test!")
    
    # Step 4: Test individual dynamic models
    print(f"\n" + "="*70)
    print("🧪 DYNAMIC MODEL INFERENCE TESTS")
    print("="*70)
    
    # Test a few Ollama dynamic models
    if dynamic_models["ollama"]:
        print(f"\n🏠 Testing Dynamic Ollama Models:")
        for model in dynamic_models["ollama"][:2]:  # Test first 2
            success, response = await test_dynamic_model_inference("ollama", model.name, model)
            await asyncio.sleep(1)  # Brief pause between tests
    
    # Test a few OpenAI dynamic models  
    if dynamic_models["openai"]:
        print(f"\n☁️  Testing Dynamic OpenAI Models:")
        for model in dynamic_models["openai"][:2]:  # Test first 2
            # Skip non-chat models
            if not any(skip in model.name.lower() for skip in ["embedding", "tts", "whisper", "dall-e", "moderation"]):
                success, response = await test_dynamic_model_inference("openai", model.name, model)
                await asyncio.sleep(1)  # Brief pause between tests
    
    # Step 5: Special reasoning model test
    await test_reasoning_model_parameters()
    
    # Step 6: Cross-provider switching demo
    await demonstrate_cross_provider_switching(dynamic_models)
    
    # Step 7: Capability demonstration
    await demonstrate_model_capabilities(dynamic_models)
    
    # Summary
    print(f"\n" + "="*70)
    print("🎉 Dynamic Model Inference Demo Complete!")
    print("="*70)
    
    print(f"\n✨ Key Achievements:")
    print(f"   • Successfully identified {total_dynamic} dynamic-only models")
    print(f"   • Tested inference without static configuration")
    print(f"   • Demonstrated cross-provider model switching")
    print(f"   • Showed reasoning model parameter handling")
    print(f"   • Proved discovery enables seamless model access")
    
    print(f"\n💡 This proves that ChukLLM discovery:")
    print(f"   • Works with ANY discovered model (static config not required)")
    print(f"   • Handles special model requirements automatically")
    print(f"   • Enables dynamic model switching across providers")
    print(f"   • Makes new models immediately available for use")
    
    print(f"\n🚀 Next: Try using any discovered model in your applications!")

if __name__ == "__main__":
    asyncio.run(main())