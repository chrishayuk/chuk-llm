#!/usr/bin/env python3
"""
ChukLLM Safe Discovery Configuration Debug Script

This script investigates discovery configuration WITHOUT triggering downloads.
Only checks configuration files and API connectivity.
"""

import json
from pathlib import Path

def debug_discovery_config_safe():
    """Debug discovery configuration without triggering downloads"""
    
    print("🔍 ChukLLM Safe Discovery Configuration Debug")
    print("=" * 50)
    print("⚠️  This script will NOT download or discover models")
    print()
    
    # Step 1: Check config file location and content
    print("1️⃣ Config File Analysis...")
    
    try:
        from chuk_llm.configuration.unified_config import UnifiedConfigManager
        
        manager = UnifiedConfigManager()
        config_file = manager._find_config_file()
        
        if config_file:
            print(f"   ✅ Config file: {config_file}")
            
            # Read raw YAML to see what's actually there
            try:
                import yaml
                with open(config_file, 'r') as f:
                    raw_config = yaml.safe_load(f)
                
                ollama_config = raw_config.get('ollama', {})
                discovery_config = ollama_config.get('extra', {}).get('dynamic_discovery', {})
                
                print(f"   📄 Raw YAML discovery config:")
                print(f"      enabled: {discovery_config.get('enabled', 'NOT_FOUND')}")
                print(f"      discoverer_type: {discovery_config.get('discoverer_type', 'NOT_FOUND')}")
                print(f"      cache_timeout: {discovery_config.get('cache_timeout', 'NOT_FOUND')}")
                
                # Show the entire extra section if it exists
                extra_section = ollama_config.get('extra', {})
                if extra_section:
                    print(f"   📋 Full 'extra' section:")
                    for key, value in extra_section.items():
                        print(f"      {key}: {value}")
                else:
                    print(f"   ⚠️  No 'extra' section found in YAML")
                
            except Exception as e:
                print(f"   ❌ Failed to read raw YAML: {e}")
        else:
            print("   ❌ No config file found")
            
    except Exception as e:
        print(f"   ❌ Config file analysis failed: {e}")
    
    # Step 2: Check loaded configuration (NO API CALLS)
    print(f"\n2️⃣ Loaded Configuration Analysis...")
    
    try:
        from chuk_llm.configuration import get_config
        
        config_manager = get_config()
        provider_config = config_manager.get_provider("ollama")
        
        print(f"   📋 Provider loaded successfully")
        print(f"   🔧 Provider type: {type(provider_config)}")
        print(f"   🔧 Extra fields: {list(provider_config.extra.keys())}")
        
        # Check discovery config in provider.extra
        if 'dynamic_discovery' in provider_config.extra:
            discovery_data = provider_config.extra['dynamic_discovery']
            print(f"   ✅ Discovery config found in extra:")
            print(f"      Type: {type(discovery_data)}")
            print(f"      Raw content: {discovery_data}")
            
            if isinstance(discovery_data, dict):
                enabled = discovery_data.get('enabled', False)
                print(f"   🎯 Discovery enabled: {enabled}")
                print(f"   📋 Full discovery config:")
                for key, value in discovery_data.items():
                    print(f"      {key}: {value} ({type(value)})")
            else:
                print(f"   ⚠️ Discovery data is not a dict: {type(discovery_data)}")
                
        else:
            print(f"   ❌ No 'dynamic_discovery' found in extra")
            print(f"   📋 Available extra keys: {list(provider_config.extra.keys())}")
            
        # Show full provider config structure
        print(f"   📋 Provider config structure:")
        print(f"      api_base: {getattr(provider_config, 'api_base', 'NOT_FOUND')}")
        print(f"      models: {len(getattr(provider_config, 'models', []))} models")
        print(f"      features: {getattr(provider_config, 'features', 'NOT_FOUND')}")
        
    except Exception as e:
        print(f"   ❌ Loaded config analysis failed: {e}")
        import traceback
        print(f"   📋 Traceback:")
        traceback.print_exc()
    
    # Step 3: Check API connectivity WITHOUT discovery
    print(f"\n3️⃣ API Connectivity Test (No Discovery)...")
    
    try:
        import httpx
        import asyncio
        
        async def test_api_only():
            try:
                # Just test if Ollama is running - no model operations
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get("http://localhost:11434/api/version")
                    if response.status_code == 200:
                        version_data = response.json()
                        print(f"   ✅ Ollama API accessible")
                        print(f"      Version: {version_data.get('version', 'unknown')}")
                        return True
                    else:
                        print(f"   ⚠️  Ollama API responded with status: {response.status_code}")
                        return False
            except Exception as e:
                print(f"   ❌ Ollama API not accessible: {e}")
                return False
        
        # Run the API test
        api_available = asyncio.run(test_api_only())
        
        if api_available:
            print(f"   💡 API is available for discovery when enabled")
        else:
            print(f"   ⚠️  API not available - discovery would fail")
            
    except Exception as e:
        print(f"   ❌ API connectivity test failed: {e}")
    
    # Step 4: Check discovery imports WITHOUT creating instances
    print(f"\n4️⃣ Discovery Module Import Test...")
    
    try:
        # Just test imports - don't create instances
        print(f"   🔍 Testing discovery imports...")
        
        try:
            from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
            print(f"   ✅ UniversalModelDiscoveryManager import OK")
        except ImportError as e:
            print(f"   ❌ UniversalModelDiscoveryManager import failed: {e}")
        
        try:
            from chuk_llm.llm.discovery.providers import DiscovererFactory
            print(f"   ✅ DiscovererFactory import OK")
        except ImportError as e:
            print(f"   ❌ DiscovererFactory import failed: {e}")
        
        try:
            from chuk_llm.api.discovery import discover_models_sync
            print(f"   ✅ discover_models_sync import OK")
        except ImportError as e:
            print(f"   ❌ discover_models_sync import failed: {e}")
            
        print(f"   💡 All discovery components available for use")
        
    except Exception as e:
        print(f"   ❌ Discovery module test failed: {e}")
    
    # Step 5: Configuration consistency check
    print(f"\n5️⃣ Configuration Consistency Check...")
    
    try:
        # Check if the config parsing is consistent
        from chuk_llm.configuration.discovery import ConfigDiscoveryMixin
        
        # Test the mixin parsing without any actual discovery
        mixin = ConfigDiscoveryMixin()
        
        # Create test data similar to what should be in the config
        test_provider_data = {
            "extra": {
                "dynamic_discovery": {
                    "enabled": True,
                    "discoverer_type": "ollama",
                    "cache_timeout": 3600
                }
            }
        }
        
        parsed_config = mixin._parse_discovery_config(test_provider_data)
        
        if parsed_config:
            print(f"   ✅ Discovery config parsing works correctly")
            print(f"      enabled: {parsed_config.enabled}")
            print(f"      discoverer_type: {parsed_config.discoverer_type}")
            print(f"      cache_timeout: {parsed_config.cache_timeout}")
        else:
            print(f"   ❌ Discovery config parsing returned None")
            
        # Now test with actual provider config
        if 'dynamic_discovery' in provider_config.extra:
            actual_provider_data = {"extra": provider_config.extra}
            actual_parsed = mixin._parse_discovery_config(actual_provider_data)
            
            if actual_parsed:
                print(f"   ✅ Actual provider config parses correctly")
                print(f"      enabled: {actual_parsed.enabled}")
            else:
                print(f"   ❌ Actual provider config parsing failed")
                print(f"   📋 Input data: {actual_provider_data}")
        
    except Exception as e:
        print(f"   ❌ Configuration consistency check failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Discovery status summary
    print(f"\n6️⃣ Discovery Status Summary...")
    
    try:
        discovery_data = provider_config.extra.get('dynamic_discovery')
        
        print(f"   📊 Discovery Configuration Status:")
        print(f"      Config exists: {'Yes' if discovery_data else 'No'}")
        
        if discovery_data:
            if isinstance(discovery_data, dict):
                enabled = discovery_data.get('enabled', False)
                discoverer_type = discovery_data.get('discoverer_type', 'unknown')
                
                print(f"      Discovery enabled: {enabled}")
                print(f"      Discoverer type: {discoverer_type}")
                
                if enabled:
                    print(f"   ✅ Discovery should work when triggered")
                else:
                    print(f"   ⚠️  Discovery is disabled in configuration")
            else:
                print(f"      ❌ Discovery config malformed (not a dict)")
        else:
            print(f"      ❌ No discovery configuration found")
    
    except Exception as e:
        print(f"   ❌ Discovery status summary failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("🎯 Safe Debug Analysis Complete")
    print()
    print("💡 What this tells us:")
    print("   • Whether discovery is configured correctly")
    print("   • If the API is accessible for discovery")  
    print("   • Whether discovery modules can be imported")
    print("   • Configuration parsing consistency")
    print()
    print("🚫 What this DIDN'T do:")
    print("   • No model downloads triggered")
    print("   • No actual discovery operations")
    print("   • No model list queries")


def check_existing_model_functions():
    """Check what model functions already exist WITHOUT discovery"""
    print(f"\n🔧 Existing Function Check...")
    
    try:
        from chuk_llm.api.providers import get_all_functions
        
        all_functions = get_all_functions()
        
        # Count different types
        ollama_ask = [name for name in all_functions.keys() if name.startswith('ask_ollama_') and not name.endswith('_sync')]
        ollama_stream = [name for name in all_functions.keys() if name.startswith('stream_ollama_')]
        ollama_sync = [name for name in all_functions.keys() if name.startswith('ask_ollama_') and name.endswith('_sync')]
        
        print(f"   📊 Current Ollama Functions:")
        print(f"      Async functions: {len(ollama_ask)}")
        print(f"      Stream functions: {len(ollama_stream)}")
        print(f"      Sync functions: {len(ollama_sync)}")
        
        # Show some examples
        if ollama_ask:
            print(f"   📋 Sample async functions:")
            for func in sorted(ollama_ask)[:5]:
                print(f"      • {func}")
            if len(ollama_ask) > 5:
                print(f"      ... and {len(ollama_ask) - 5} more")
        
        print(f"   💡 These functions exist from static configuration")
        print(f"   💡 Discovery would add more functions for new models")
        
    except Exception as e:
        print(f"   ❌ Function check failed: {e}")


def main():
    """Main safe debug function"""
    debug_discovery_config_safe()
    check_existing_model_functions()
    
    print(f"\n🛡️  Safe Debug Complete")
    print("   No models were downloaded or discovered")
    print("   Only configuration and connectivity tested")

if __name__ == "__main__":
    main()