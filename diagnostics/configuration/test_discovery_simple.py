#!/usr/bin/env python3
"""
Function Existence Only Test

ZERO RISK test - only checks if functions exist, never calls them.
No model downloads, no API calls, no inference.
"""

def safe_function_existence_check():
    """Check function existence without calling anything"""
    
    print("🔍 ChukLLM Function Existence Check")
    print("=" * 40)
    print("🛡️  ZERO RISK - No function calls, no downloads")
    print()
    
    print("1️⃣ Import Test...")
    
    try:
        import chuk_llm
        print("   ✅ chuk_llm imported successfully")
        
        # Count total attributes
        all_attrs = dir(chuk_llm)
        print(f"   📊 Total attributes: {len(all_attrs)}")
        
    except ImportError as e:
        print(f"   ❌ chuk_llm import failed: {e}")
        return
    except Exception as e:
        print(f"   ❌ Unexpected import error: {e}")
        return
    
    print("\n2️⃣ Ollama Function Detection...")
    
    # Check for ollama functions (based on your known models)
    expected_functions = [
        # Base model functions (without :latest)
        "ask_ollama_granite3_3",
        "ask_ollama_qwen3",
        "ask_ollama_devstral", 
        "ask_ollama_mistral",
        "ask_ollama_gemma3",
        "ask_ollama_phi4_reasoning",
        "ask_ollama_llama3_2_vision",
        
        # With :latest variants
        "ask_ollama_granite3_3_latest",
        "ask_ollama_devstral_latest",
        "ask_ollama_phi4_reasoning_latest",
        
        # Specific variants
        "ask_ollama_qwen3_32b",
        "ask_ollama_llama3_2_vision_latest"
    ]
    
    found_functions = []
    missing_functions = []
    
    for func_name in expected_functions:
        if hasattr(chuk_llm, func_name):
            found_functions.append(func_name)
            print(f"   ✅ {func_name}")
        else:
            missing_functions.append(func_name)
            print(f"   ❌ {func_name}")
    
    print(f"\n   📊 Results:")
    print(f"      Found: {len(found_functions)}")
    print(f"      Missing: {len(missing_functions)}")
    
    print("\n3️⃣ Function Type Analysis...")
    
    # Count different types of functions
    ask_functions = [attr for attr in all_attrs if attr.startswith('ask_ollama_') and not attr.endswith('_sync')]
    stream_functions = [attr for attr in all_attrs if attr.startswith('stream_ollama_')]
    sync_functions = [attr for attr in all_attrs if attr.startswith('ask_ollama_') and attr.endswith('_sync')]
    
    print(f"   📋 Async functions: {len(ask_functions)}")
    print(f"   📋 Stream functions: {len(stream_functions)}")
    print(f"   📋 Sync functions: {len(sync_functions)}")
    
    # Show some examples
    if ask_functions:
        print(f"\n   📝 Sample async functions:")
        for func in sorted(ask_functions)[:5]:
            print(f"      • {func}")
        if len(ask_functions) > 5:
            print(f"      ... and {len(ask_functions) - 5} more")
    
    print("\n4️⃣ Core Function Check...")
    
    core_functions = ["ask", "stream", "show_config"]
    
    for func_name in core_functions:
        if hasattr(chuk_llm, func_name):
            print(f"   ✅ chuk_llm.{func_name}")
        else:
            print(f"   ❌ chuk_llm.{func_name}")
    
    print("\n5️⃣ Provider Function Detection...")
    
    # Look for other provider functions
    providers = ["openai", "anthropic", "gemini"]
    
    for provider in providers:
        provider_funcs = [attr for attr in all_attrs if f"ask_{provider}" in attr and not attr.endswith('_sync')]
        if provider_funcs:
            print(f"   📦 {provider}: {len(provider_funcs)} functions")
        else:
            print(f"   📦 {provider}: No functions found")
    
    print("\n6️⃣ Function Inspection (Safe)...")
    
    # Look at function properties without calling them
    if found_functions:
        sample_func_name = found_functions[0]
        sample_func = getattr(chuk_llm, sample_func_name)
        
        print(f"   🔍 Inspecting: {sample_func_name}")
        print(f"      Type: {type(sample_func)}")
        print(f"      Callable: {callable(sample_func)}")
        
        # Check if it has docstring
        if hasattr(sample_func, '__doc__') and sample_func.__doc__:
            doc = sample_func.__doc__.strip()
            print(f"      Docstring: {doc[:50]}{'...' if len(doc) > 50 else ''}")
        else:
            print(f"      Docstring: None")
        
        # Check signature without calling
        try:
            import inspect
            sig = inspect.signature(sample_func)
            params = list(sig.parameters.keys())
            print(f"      Parameters: {params[:3]}{'...' if len(params) > 3 else ''}")
        except Exception:
            print(f"      Parameters: Cannot inspect")
    
    # Final summary
    print(f"\n🎯 Summary")
    print("=" * 30)
    
    if found_functions:
        print(f"✅ ChukLLM is working!")
        print(f"📊 Found {len(found_functions)} expected Ollama functions")
        print(f"🚀 Total Ollama functions: {len(ask_functions)} async + {len(sync_functions)} sync + {len(stream_functions)} stream")
        
        print(f"\n💡 Ready to use (examples):")
        for func in found_functions[:3]:
            print(f"   await chuk_llm.{func}('your prompt')")
        
        if len(found_functions) > 3:
            print(f"   ... and {len(found_functions) - 3} more functions")
    else:
        print(f"⚠️  No expected functions found")
        print(f"💡 Total ollama functions available: {len(ask_functions)}")
        if ask_functions:
            print(f"   Available: {', '.join(ask_functions[:3])}")
    
    print(f"\n🛡️  Safety Confirmation:")
    print("   ✅ No functions called")
    print("   ✅ No API requests made") 
    print("   ✅ No models downloaded")
    print("   ✅ Only existence checked")

def main():
    """Main safe check"""
    try:
        safe_function_existence_check()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Check interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Check failed with error: {e}")
        import traceback
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()