#!/usr/bin/env python3
"""
Test Discovery-Inference Integration
====================================

This script tests the discovery integration that's already implemented in 
your configuration system. It uses the existing _ensure_model_available()
method and environment controls.
"""

import asyncio
import os
import logging
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def test_discovery_integration():
    """Test the discovery integration that's already implemented"""
    print("🧪 Testing Discovery-Inference Integration")
    print("=" * 60)
    
    # Test environment variable controls
    print("\n🔧 Current Discovery Settings:")
    env_vars = {
        'CHUK_LLM_DISCOVERY_ENABLED': os.getenv('CHUK_LLM_DISCOVERY_ENABLED', 'not set'),
        'CHUK_LLM_AUTO_DISCOVER': os.getenv('CHUK_LLM_AUTO_DISCOVER', 'not set'),
        'CHUK_LLM_OLLAMA_DISCOVERY': os.getenv('CHUK_LLM_OLLAMA_DISCOVERY', 'not set'),
        'CHUK_LLM_OPENAI_DISCOVERY': os.getenv('CHUK_LLM_OPENAI_DISCOVERY', 'not set'),
        'CHUK_LLM_DISCOVERY_TIMEOUT': os.getenv('CHUK_LLM_DISCOVERY_TIMEOUT', 'not set'),
    }
    
    for var, value in env_vars.items():
        print(f"   {var}: {value}")
    
    # Enable discovery for testing
    if os.getenv('CHUK_LLM_DISCOVERY_ENABLED') != 'true':
        print(f"\n💡 Setting CHUK_LLM_DISCOVERY_ENABLED=true for this test")
        os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'true'
    
    if os.getenv('CHUK_LLM_AUTO_DISCOVER') != 'true':
        print(f"💡 Setting CHUK_LLM_AUTO_DISCOVER=true for this test")
        os.environ['CHUK_LLM_AUTO_DISCOVER'] = 'true'
    
    try:
        from chuk_llm.configuration import get_config
        
        # Get configuration manager (this should have discovery integrated)
        print(f"\n📋 Loading Configuration Manager with Discovery...")
        config_manager = get_config()
        
        # Test discovery settings
        if hasattr(config_manager, 'get_discovery_settings'):
            settings = config_manager.get_discovery_settings()
            print(f"   Discovery enabled: {settings.get('enabled', 'unknown')}")
            print(f"   Auto discover: {settings.get('auto_discover', 'unknown')}")
            print(f"   Cache timeout: {settings.get('cache_timeout', 'unknown')}s")
        else:
            print("   ⚠️  Discovery settings method not available")
        
        # Test providers with discovery
        print(f"\n🔍 Testing Provider Discovery Integration...")
        
        providers_to_test = ['ollama', 'openai']
        
        for provider_name in providers_to_test:
            print(f"\n   🎯 Testing {provider_name.upper()}:")
            
            try:
                provider = config_manager.get_provider(provider_name)
                print(f"      ✅ Provider loaded: {provider.name}")
                print(f"      📋 Static models: {len(provider.models)}")
                
                if provider.models:
                    print(f"         └─ Examples: {provider.models[:3]}")
                
                # Check if discovery is configured
                discovery_config = provider.extra.get('dynamic_discovery', {})
                if discovery_config.get('enabled'):
                    print(f"      🔍 Discovery: enabled")
                    print(f"         └─ Cache timeout: {discovery_config.get('cache_timeout', 300)}s")
                else:
                    print(f"      🔍 Discovery: not configured in YAML")
                
                # Test model availability check (this should trigger discovery)
                test_models = [
                    'llama3.1:latest',
                    'qwen3:latest', 
                    'granite3.3:latest',
                    'o1-mini-2024-09-12',
                    'gpt-4o-2024-11-20'
                ]
                
                print(f"      🧪 Testing model availability (triggers discovery):")
                
                for model_name in test_models[:2]:  # Test first 2 to avoid timeout
                    # Use the _ensure_model_available method that should trigger discovery
                    if hasattr(config_manager, '_ensure_model_available'):
                        try:
                            resolved = config_manager._ensure_model_available(provider_name, model_name)
                            if resolved:
                                print(f"         ✅ {model_name} → {resolved}")
                            else:
                                print(f"         ❌ {model_name} → not found")
                        except Exception as e:
                            print(f"         ❌ {model_name} → error: {e}")
                    else:
                        print(f"         ⚠️  _ensure_model_available method not available")
                        
                        # Fallback: check static availability
                        if model_name in provider.models:
                            print(f"         ✅ {model_name} → available (static)")
                        else:
                            print(f"         ❌ {model_name} → not in static config")
                
            except Exception as e:
                print(f"      ❌ Error testing {provider_name}: {e}")
        
        # Test with ChukLLM ask function
        print(f"\n🎯 Testing ChukLLM ask() with discovered models...")
        
        try:
            from chuk_llm import ask
            
            # Test cases - try models that might be discovered
            test_cases = [
                ('ollama', 'qwen3:latest'),
                ('openai', 'gpt-4o-mini'),  # Should be available
            ]
            
            for provider, model in test_cases:
                print(f"\n   🧪 Testing {provider}/{model}:")
                
                try:
                    response = await ask(
                        "What is 2+2? Answer briefly.",
                        provider=provider,
                        model=model,
                        max_tokens=20
                    )
                    
                    print(f"      ✅ Success: {response[:50]}...")
                    
                except Exception as e:
                    error_msg = str(e)
                    if "not available" in error_msg.lower():
                        print(f"      ❌ Model not available: {e}")
                        
                        # Check if this is a discovery issue
                        if hasattr(config_manager, '_ensure_model_available'):
                            resolved = config_manager._ensure_model_available(provider, model)
                            if resolved:
                                print(f"      💡 Discovery found it as: {resolved}")
                                print(f"      💡 Try rerunning - discovery may have updated configuration")
                            else:
                                print(f"      💡 Model truly not available through discovery")
                        
                    else:
                        print(f"      ❌ Error: {e}")
        
        except ImportError:
            print(f"      ⚠️  ChukLLM ask function not available for testing")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("🎉 Discovery Integration Test Complete!")
        print("=" * 60)
        
        print(f"\n✨ What was tested:")
        print(f"   🔧 Environment variable discovery controls")
        print(f"   📋 Configuration manager with discovery mixin")
        print(f"   🔍 Provider discovery configuration")
        print(f"   🧪 Model availability checking (_ensure_model_available)")
        print(f"   🎯 Integration with ChukLLM ask() function")
        
        print(f"\n💡 Key findings:")
        
        discovery_working = False
        if hasattr(config_manager, '_ensure_model_available'):
            print(f"   ✅ Discovery integration is implemented")
            discovery_working = True
        else:
            print(f"   ❌ Discovery integration method missing")
        
        if hasattr(config_manager, 'get_discovery_settings'):
            print(f"   ✅ Discovery settings accessible")
        else:
            print(f"   ❌ Discovery settings not accessible")
        
        # Check if discovery is actually enabled
        settings = getattr(config_manager, '_discovery_settings', {})
        if settings.get('enabled', False):
            print(f"   ✅ Discovery is enabled")
        else:
            print(f"   ⚠️  Discovery may not be enabled - check environment variables")
        
        if discovery_working:
            print(f"\n🚀 Discovery integration is ready!")
            print(f"   Set CHUK_LLM_DISCOVERY_ENABLED=true to enable")
            print(f"   Configure dynamic_discovery in provider YAML")
            print(f"   Models will be discovered automatically when requested")
        
    except ImportError as e:
        print(f"❌ Failed to import ChukLLM configuration: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_environment_controls():
    """Test environment variable controls for discovery"""
    print(f"\n🔧 Testing Environment Variable Controls")
    print("-" * 40)
    
    # Test different environment settings
    test_settings = [
        ('CHUK_LLM_DISCOVERY_ENABLED', ['true', 'false']),
        ('CHUK_LLM_AUTO_DISCOVER', ['true', 'false']),
        ('CHUK_LLM_DISCOVERY_TIMEOUT', ['2', '10']),
        ('CHUK_LLM_OLLAMA_DISCOVERY', ['true', 'false']),
    ]
    
    for var_name, test_values in test_settings:
        print(f"\n   Testing {var_name}:")
        
        for value in test_values:
            original = os.getenv(var_name)
            os.environ[var_name] = value
            
            try:
                # Reset config to pick up new environment
                from chuk_llm.configuration import reset_config
                reset_config()
                
                from chuk_llm.configuration import get_config
                config = get_config()
                
                if hasattr(config, 'get_discovery_settings'):
                    settings = config.get_discovery_settings()
                    print(f"      {var_name}={value} → enabled: {settings.get('enabled', 'unknown')}")
                
            except Exception as e:
                print(f"      {var_name}={value} → error: {e}")
            
            # Restore original value
            if original is not None:
                os.environ[var_name] = original
            elif var_name in os.environ:
                del os.environ[var_name]

async def main():
    """Run all tests"""
    print("🚀 Discovery-Inference Integration Test Suite")
    print("=" * 70)
    
    await test_discovery_integration()
    await test_environment_controls()
    
    print(f"\n" + "=" * 70)
    print("✅ Test Suite Complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())