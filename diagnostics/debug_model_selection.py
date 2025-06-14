#!/usr/bin/env python3
"""Debug model selection for different providers."""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

def debug_model_selection():
    """Debug how models are being selected for each provider."""
    
    print("🔍 Debugging Model Selection")
    print("=" * 50)
    
    # 1. Check global configuration
    print("\n1. Global Configuration:")
    from chuk_llm.api.config import get_current_config
    config = get_current_config()
    print(f"   Global provider: {config.get('provider')}")
    print(f"   Global model: {config.get('model')}")
    print()
    
    # 2. Check YAML configuration
    print("2. YAML Configuration:")
    try:
        import yaml
        yaml_path = "src/chuk_llm/providers.yaml"
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        for provider in ['openai', 'anthropic', 'groq']:
            if provider in yaml_config:
                default_model = yaml_config[provider].get('default_model', 'NOT SET')
                print(f"   {provider}: {default_model}")
            else:
                print(f"   {provider}: NOT IN YAML")
    except Exception as e:
        print(f"   Error reading YAML: {e}")
    print()
    
    # 3. Test provider-specific configs
    print("3. Provider-Specific Model Resolution:")
    
    # Test what happens when we call each provider
    test_providers = ['openai', 'anthropic', 'groq']
    
    for provider in test_providers:
        print(f"\n   Testing {provider}:")
        
        try:
            # Get config for this provider
            from chuk_llm.api.config import get_client_for_config
            
            test_config = config.copy()
            test_config['provider'] = provider
            
            print(f"     Config before client: provider={test_config.get('provider')}, model={test_config.get('model')}")
            
            # Try to get client (this might reveal model resolution)
            client = get_client_for_config(test_config)
            print(f"     Client created: {type(client).__name__}")
            
        except Exception as e:
            print(f"     Error: {e}")

def test_model_resolution():
    """Test if the new model resolution logic is working."""
    
    print("\n4. Testing Model Resolution Logic:")
    print("-" * 40)
    
    try:
        from chuk_llm.api.core import _get_provider_default_model
        
        test_providers = ['openai', 'anthropic', 'groq', 'deepseek']
        
        for provider in test_providers:
            default_model = _get_provider_default_model(provider)
            print(f"   {provider}: {default_model or 'NO DEFAULT FOUND'}")
            
    except ImportError:
        print("   ❌ New model resolution function not found - apply the core.py fix first")
    except Exception as e:
        print(f"   ❌ Error testing model resolution: {e}")

def test_direct_calls():
    """Test direct provider calls to see what models are used."""
    
    print("\n5. Direct Provider Call Testing:")
    print("-" * 30)
    
    # Test without specifying models to see if auto-resolution works
    test_providers = ['openai', 'anthropic', 'groq']
    
    for provider in test_providers:
        print(f"\n   {provider} (auto-resolving model):")
        
        try:
            from chuk_llm.api.core import ask
            import asyncio
            
            # Test the model resolution without specifying a model
            async def test_provider():
                return await ask(
                    "Say 'test' and nothing else",
                    provider=provider,
                    max_tokens=10
                )
            
            response = asyncio.run(test_provider())
            print(f"     ✅ Success: {response[:50]}...")
            
        except Exception as e:
            error_msg = str(e)
            if "gpt-4o-mini" in error_msg and provider == "anthropic":
                print(f"     ❌ Still using wrong model: {error_msg[:100]}...")
                print("     💡 Apply the core.py fix to resolve this")
            else:
                print(f"     ❌ Error: {error_msg[:100]}...")

if __name__ == "__main__":
    debug_model_selection()
    test_model_resolution()
    
    # Only test actual calls if the user wants to
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-calls":
        print("\n" + "=" * 50)
        test_direct_calls()
    else:
        print("\n💡 Run with --test-calls to test actual API calls")
        print("   (requires API keys to be configured)")