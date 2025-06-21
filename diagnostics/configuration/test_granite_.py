#!/usr/bin/env python3
"""
Debug why granite3_3_latest function is missing
"""

def debug_granite_function():
    """Debug the missing granite function"""
    
    print("🔍 Granite Function Debug")
    print("=" * 30)
    
    # Test the sanitization logic
    print("1️⃣ Sanitization Test...")
    
    try:
        from chuk_llm.api.providers import _sanitize_name
        
        test_names = [
            "granite3.3",
            "granite3.3:latest",
            "devstral:latest",  # This works
            "qwen3:32b"         # This works
        ]
        
        for name in test_names:
            sanitized = _sanitize_name(name)
            expected_func = f"ask_ollama_{sanitized}"
            print(f"   {name} → {sanitized} → {expected_func}")
            
    except Exception as e:
        print(f"   ❌ Sanitization test failed: {e}")
    
    # Check what granite functions actually exist
    print(f"\n2️⃣ Existing Granite Functions...")
    
    try:
        import chuk_llm
        
        all_attrs = dir(chuk_llm)
        granite_functions = [attr for attr in all_attrs if 'granite' in attr.lower()]
        
        print(f"   Found {len(granite_functions)} granite-related functions:")
        for func in sorted(granite_functions):
            print(f"   ✅ {func}")
            
    except Exception as e:
        print(f"   ❌ Function check failed: {e}")
    
    # Check configuration models
    print(f"\n3️⃣ Configuration Model Check...")
    
    try:
        from chuk_llm.configuration import get_config
        
        config = get_config()
        provider = config.get_provider("ollama")
        
        granite_models = [model for model in provider.models if 'granite' in model.lower()]
        
        print(f"   Granite models in config:")
        for model in granite_models:
            print(f"   📦 {model}")
            
    except Exception as e:
        print(f"   ❌ Config check failed: {e}")
    
    # Test function generation logic
    print(f"\n4️⃣ Function Generation Logic Test...")
    
    try:
        from chuk_llm.api.providers import _generate_functions_for_models
        from chuk_llm.configuration import get_config
        
        config = get_config()
        provider = config.get_provider("ollama")
        
        # Test with just granite models
        granite_models = ["granite3.3:latest"]
        
        print(f"   Testing function generation for: {granite_models}")
        
        new_functions = _generate_functions_for_models("ollama", provider, granite_models)
        
        granite_funcs = {name: func for name, func in new_functions.items() if 'granite' in name}
        
        print(f"   Generated granite functions:")
        for name in sorted(granite_funcs.keys()):
            print(f"   🔧 {name}")
            
        # Check if the missing function was generated
        if "ask_ollama_granite3_3_latest" in granite_funcs:
            print(f"   ✅ ask_ollama_granite3_3_latest was generated!")
        else:
            print(f"   ❌ ask_ollama_granite3_3_latest was NOT generated")
            
    except Exception as e:
        print(f"   ❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_granite_function()