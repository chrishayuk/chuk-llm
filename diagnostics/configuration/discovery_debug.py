#!/usr/bin/env python3
"""
ChukLLM Discovery Configuration Debug Script

This script investigates why discovery appears disabled when it should be enabled.
"""

import asyncio
import json
from pathlib import Path

async def debug_discovery_config():
    """Debug discovery configuration loading"""
    
    print("🔍 ChukLLM Discovery Configuration Debug")
    print("=" * 50)
    
    # Step 1: Check config file location and content
    print("\n1️⃣ Config File Analysis...")
    
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
                
            except Exception as e:
                print(f"   ❌ Failed to read raw YAML: {e}")
        else:
            print("   ❌ No config file found")
            
    except Exception as e:
        print(f"   ❌ Config file analysis failed: {e}")
    
    # Step 2: Check loaded configuration
    print("\n2️⃣ Loaded Configuration Analysis...")
    
    try:
        from chuk_llm.configuration import get_config
        
        config_manager = get_config()
        provider_config = config_manager.get_provider("ollama")
        
        print(f"   📋 Provider loaded successfully")
        print(f"   🔧 Extra fields: {list(provider_config.extra.keys())}")
        
        # Check discovery config in provider.extra
        if 'dynamic_discovery' in provider_config.extra:
            discovery_data = provider_config.extra['dynamic_discovery']
            print(f"   ✅ Discovery config found in extra:")
            print(f"      Type: {type(discovery_data)}")
            print(f"      Content: {discovery_data}")
            
            if isinstance(discovery_data, dict):
                enabled = discovery_data.get('enabled', False)
                print(f"   🎯 Discovery enabled: {enabled}")
                print(f"   📋 Full discovery config:")
                for key, value in discovery_data.items():
                    print(f"      {key}: {value}")
            else:
                print(f"   ⚠️ Discovery data is not a dict: {type(discovery_data)}")
                
        else:
            print(f"   ❌ No 'dynamic_discovery' found in extra")
            print(f"   📋 Available extra keys: {list(provider_config.extra.keys())}")
            
    except Exception as e:
        print(f"   ❌ Loaded config analysis failed: {e}")
        import traceback
        print(f"   📋 Traceback: {traceback.format_exc()}")
    
    # Step 3: Test the discovery mixin directly
    print("\n3️⃣ Discovery Mixin Test...")
    
    try:
        from chuk_llm.configuration.discovery import ConfigDiscoveryMixin
        from chuk_llm.configuration.models import DiscoveryConfig
        
        # Create a test discovery mixin
        mixin = ConfigDiscoveryMixin()
        mixin.providers = {"ollama": provider_config}  # Add provider manually
        
        # Test parsing discovery config
        provider_data = {"extra": provider_config.extra}
        discovery_config = mixin._parse_discovery_config(provider_data)
        
        if discovery_config:
            print(f"   ✅ Discovery config parsed successfully:")
            print(f"      enabled: {discovery_config.enabled}")
            print(f"      discoverer_type: {discovery_config.discoverer_type}")
            print(f"      cache_timeout: {discovery_config.cache_timeout}")
        else:
            print(f"   ❌ Discovery config parsing returned None")
            print(f"   📋 Input data: {provider_data}")
            
    except Exception as e:
        print(f"   ❌ Discovery mixin test failed: {e}")
        import traceback
        print(f"   📋 Traceback: {traceback.format_exc()}")
    
    # Step 4: Test client get_model_info method
    print("\n4️⃣ Client Model Info Test...")
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client("ollama", model="granite3.3")
        model_info = client.get_model_info()
        
        print(f"   ✅ Client created successfully")
        print(f"   📋 Model info keys: {list(model_info.keys())}")
        
        # Check if client has discovery-related info
        if hasattr(client, '_get_provider_config'):
            provider_config = client._get_provider_config()
            if provider_config:
                discovery_data = provider_config.extra.get('dynamic_discovery')
                print(f"   🔧 Client sees discovery config: {discovery_data is not None}")
                if discovery_data:
                    print(f"      enabled: {discovery_data.get('enabled', 'NOT_FOUND')}")
            else:
                print(f"   ❌ Client cannot get provider config")
        else:
            print(f"   ⚠️ Client doesn't have _get_provider_config method")
            
    except Exception as e:
        print(f"   ❌ Client test failed: {e}")
        import traceback
        print(f"   📋 Traceback: {traceback.format_exc()}")
    
    # Step 5: Check discovery manager creation
    print("\n5️⃣ Discovery Manager Creation Test...")
    
    try:
        # Try to create discovery manager
        if 'dynamic_discovery' in provider_config.extra:
            discovery_data = provider_config.extra['dynamic_discovery']
            
            if discovery_data.get('enabled'):
                print(f"   🎯 Attempting to create discovery manager...")
                
                # Import discovery components
                try:
                    from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
                    from chuk_llm.llm.discovery.providers import DiscovererFactory
                    
                    print(f"   ✅ Discovery imports successful")
                    
                    # Try to create discoverer
                    discoverer_config = {
                        **discovery_data.get('discoverer_config', {}),
                        "api_base": provider_config.api_base,
                    }
                    
                    discoverer_type = discovery_data.get('discoverer_type', 'ollama')
                    print(f"   🔧 Creating discoverer: {discoverer_type}")
                    print(f"   📋 Discoverer config: {discoverer_config}")
                    
                    discoverer = DiscovererFactory.create_discoverer(discoverer_type, **discoverer_config)
                    print(f"   ✅ Discoverer created: {type(discoverer)}")
                    
                    # Create manager
                    manager = UniversalModelDiscoveryManager(
                        provider_name="ollama",
                        discoverer=discoverer,
                        inference_config=discovery_data.get('inference_config', {})
                    )
                    print(f"   ✅ Discovery manager created successfully")
                    
                    # Test discovery
                    print(f"   🧪 Testing model discovery...")
                    models = await manager.discover_models()
                    print(f"   🎯 Discovered {len(models)} models")
                    
                    if models:
                        print(f"   📋 Sample models:")
                        for model in models[:5]:
                            print(f"      • {model.name} ({model.family})")
                        if len(models) > 5:
                            print(f"      ... and {len(models) - 5} more")
                    
                except Exception as discovery_error:
                    print(f"   ❌ Discovery manager creation failed: {discovery_error}")
                    import traceback
                    print(f"   📋 Traceback: {traceback.format_exc()}")
                    
            else:
                print(f"   ⚠️ Discovery disabled in config")
        else:
            print(f"   ❌ No discovery config found")
            
    except Exception as e:
        print(f"   ❌ Discovery manager test failed: {e}")
        import traceback
        print(f"   📋 Traceback: {traceback.format_exc()}")
    
    # Step 6: Check the specific method that showed discovery as disabled
    print("\n6️⃣ Discovery Status Check...")
    
    try:
        # This is what the original script was checking
        discovery_enabled = False
        discovery_data = provider_config.extra.get('dynamic_discovery')
        
        print(f"   📋 Raw discovery data: {discovery_data}")
        print(f"   📋 Type: {type(discovery_data)}")
        
        if discovery_data:
            if isinstance(discovery_data, dict):
                discovery_enabled = discovery_data.get('enabled', False)
                print(f"   🎯 Discovery enabled (dict access): {discovery_enabled}")
            else:
                print(f"   ⚠️ Discovery data is not a dict!")
                
        print(f"   📊 Final discovery status: {discovery_enabled}")
        
    except Exception as e:
        print(f"   ❌ Discovery status check failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Debug Analysis Complete")
    
    return True

async def test_manual_discovery():
    """Test discovery manually"""
    print("\n🧪 Manual Discovery Test...")
    
    try:
        from chuk_llm.api.discovery import discover_models_sync
        
        print("   🔍 Testing manual discovery...")
        models = discover_models_sync("ollama")
        
        print(f"   ✅ Manual discovery successful!")
        print(f"   📊 Found {len(models)} models")
        
        if models:
            print("   📋 Sample discovered models:")
            for model in models[:5]:
                print(f"      • {model['name']} - {model.get('features', [])}")
        
    except Exception as e:
        print(f"   ❌ Manual discovery failed: {e}")
        import traceback
        print(f"   📋 Traceback: {traceback.format_exc()}")

async def main():
    """Main debug function"""
    await debug_discovery_config()
    await test_manual_discovery()
    
    print("\n💡 Next Steps:")
    print("   1. Check if discovery is actually working despite the status")
    print("   2. Try manual discovery using the API")
    print("   3. Check for any config inheritance issues")
    print("   4. Verify all discovery components are imported correctly")

if __name__ == "__main__":
    asyncio.run(main())