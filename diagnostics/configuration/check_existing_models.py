#!/usr/bin/env python3
"""
Quick test with just a few key models to avoid hanging
"""

import sys

def test_key_models():
    """Test just a few key models from your collection"""
    
    print("🎯 Quick Model Test")
    print("=" * 30)
    
    # Based on your output, test these key models
    key_models = [
        "devstral:latest",
        "qwen3:32b", 
        "phi4-reasoning:latest",
        "llama3.3:latest",
        "granite3.3:latest"
    ]
    
    print(f"Testing {len(key_models)} key models from your collection:")
    for model in key_models:
        print(f"  • {model}")
    
    print(f"\n🔧 Step 1: Test Function Name Generation")
    print("-" * 40)
    
    try:
        from chuk_llm.api.providers import _sanitize_name
        
        function_mapping = {}
        
        for model in key_models:
            print(f"Processing {model}...")
            
            sanitized = _sanitize_name(model)
            
            if sanitized:
                main_func = f"ask_ollama_{sanitized}"
                function_mapping[model] = {
                    'sanitized': sanitized,
                    'main_function': main_func
                }
                print(f"  ✅ {model} -> {main_func}")
            else:
                print(f"  ❌ {model} -> Cannot sanitize")
        
        return function_mapping
        
    except Exception as e:
        print(f"❌ Function name generation failed: {e}")
        return {}


def test_function_existence(function_mapping):
    """Test if functions exist for key models"""
    
    print(f"\n🔍 Step 2: Check Function Existence")
    print("-" * 40)
    
    try:
        import chuk_llm
        
        found = []
        missing = []
        
        for model, info in function_mapping.items():
            func_name = info['main_function']
            
            print(f"Checking {func_name}...")
            
            if hasattr(chuk_llm, func_name):
                print(f"  ✅ Found in chuk_llm")
                found.append((model, func_name))
            else:
                print(f"  ❌ Not found")
                missing.append((model, func_name))
        
        return found, missing
        
    except Exception as e:
        print(f"❌ Function existence check failed: {e}")
        return [], []


def try_manual_generation(missing_functions):
    """Try to generate missing functions manually"""
    
    if not missing_functions:
        print(f"\n✅ All functions found - no generation needed")
        return
    
    print(f"\n🔧 Step 3: Generate Missing Functions")
    print("-" * 40)
    
    try:
        from chuk_llm.api.providers import _generate_functions_for_models
        from chuk_llm.configuration import get_config
        
        # Get just the model names we're missing
        missing_models = [model for model, func in missing_functions]
        
        print(f"Attempting to generate functions for {len(missing_models)} models:")
        for model in missing_models:
            print(f"  • {model}")
        
        config = get_config()
        ollama_provider = config.get_provider('ollama')
        
        # Generate functions for just these models
        new_functions = _generate_functions_for_models('ollama', ollama_provider, missing_models)
        
        if new_functions:
            print(f"✅ Generated {len(new_functions)} functions")
            
            # Add them to the module
            import chuk_llm.api.providers as providers_module
            
            for name, func in new_functions.items():
                setattr(providers_module, name, func)
                setattr(chuk_llm, name, func)
            
            print("✅ Functions added to modules")
            
            # Test if they're now accessible
            print(f"\n🧪 Testing generated functions...")
            for model, func_name in missing_functions:
                if hasattr(chuk_llm, func_name):
                    print(f"  ✅ {func_name} - now available")
                else:
                    print(f"  ❌ {func_name} - still missing")
        else:
            print(f"❌ No functions generated")
        
    except Exception as e:
        print(f"❌ Manual generation failed: {e}")


def test_a_function(found_functions):
    """Test calling one of the found functions"""
    
    if not found_functions:
        print(f"\n❌ No functions to test")
        return
    
    print(f"\n🧪 Step 4: Test Function Call")
    print("-" * 40)
    
    try:
        import chuk_llm
        import asyncio
        
        # Test the first available function
        test_model, test_func = found_functions[0]
        
        print(f"Testing {test_func} (model: {test_model})")
        
        func = getattr(chuk_llm, test_func)
        
        async def test_call():
            try:
                response = await func("Say 'Hello from ChukLLM!'", max_tokens=20)
                return response
            except Exception as e:
                return f"Error: {e}"
        
        result = asyncio.run(test_call())
        
        if "Error:" in str(result):
            print(f"  ⚠️  Call failed: {result}")
        else:
            print(f"  ✅ Success: '{result[:100]}{'...' if len(str(result)) > 100 else ''}'")
        
    except Exception as e:
        print(f"❌ Function test failed: {e}")


def main():
    """Main test function"""
    
    print("🚀 ChukLLM Key Models Test")
    print("=" * 40)
    print("Testing dynamic functions for your key models")
    print()
    
    # Step 1: Generate function names for key models
    function_mapping = test_key_models()
    
    if not function_mapping:
        print("❌ Cannot proceed - function mapping failed")
        return
    
    # Step 2: Check if functions exist
    found, missing = test_function_existence(function_mapping)
    
    # Step 3: Generate missing functions
    if missing:
        try_manual_generation(missing)
        
        # Re-check after generation
        found_after, still_missing = test_function_existence(function_mapping)
        print(f"\n📊 After generation:")
        print(f"  Found: {len(found_after)}")
        print(f"  Still missing: {len(still_missing)}")
        found = found_after
    
    # Step 4: Test a function
    if found:
        test_a_function(found)
    
    # Final summary
    print(f"\n📊 Final Results")
    print("=" * 30)
    
    if found:
        print(f"✅ Working functions:")
        for model, func in found:
            print(f"  await chuk_llm.{func}('your prompt')")
    else:
        print(f"❌ No working functions found")
        print(f"💡 Try manually refreshing:")
        print(f"  from chuk_llm.api.providers import refresh_provider_functions")
        print(f"  refresh_provider_functions('ollama')")


if __name__ == "__main__":
    main()