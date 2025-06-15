#!/usr/bin/env python3
"""
Quick parameter and kwargs compatibility test
"""
import asyncio
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

async def quick_parameter_test():
    """Quick test of key parameters and kwargs handling"""
    
    from chuk_llm.llm.client import get_client
    
    providers = ["openai", "anthropic", "gemini", "groq", "ollama", "mistral"]
    
    # Test parameters
    test_params = {
        "temperature": [0.0, 0.5, 1.0, 2.0],
        "max_tokens": [10, 50, 100],
        "top_p": [0.1, 0.9, 1.0],
        "stop": [None, ".", [".", "!"]]
    }
    
    # Unknown kwargs to test
    unknown_kwargs = {
        "unknown_param": "test",
        "made_up_setting": 42,
        "fictional_option": True
    }
    
    print("🧪 QUICK PARAMETER & KWARGS COMPATIBILITY TEST")
    print("="*60)
    
    for provider in providers:
        print(f"\n🔧 Testing {provider.upper()}...")
        
        try:
            client = get_client(provider)
            
            # Test 1: Basic parameters
            print("  📋 Testing standard parameters...")
            
            param_results = {}
            
            for param_name, values in test_params.items():
                param_results[param_name] = []
                
                for value in values[:2]:  # Test first 2 values for speed
                    try:
                        kwargs = {param_name: value}
                        if provider == "anthropic" and param_name != "max_tokens":
                            kwargs["max_tokens"] = 50
                        
                        response = await client.create_completion(
                            [{"role": "user", "content": "Say hello"}],
                            **kwargs
                        )
                        
                        if response and response.get("response"):
                            param_results[param_name].append(f"✅ {value}")
                        else:
                            param_results[param_name].append(f"⚠️ {value} (no response)")
                            
                    except Exception as e:
                        param_results[param_name].append(f"❌ {value} ({str(e)[:30]}...)")
            
            # Print parameter results
            for param, results in param_results.items():
                print(f"    {param}: {', '.join(results)}")
            
            # Test 2: Unknown kwargs handling
            print("  📦 Testing unknown kwargs...")
            
            try:
                response = await client.create_completion(
                    [{"role": "user", "content": "Test kwargs"}],
                    temperature=0.5,
                    max_tokens=20,
                    **unknown_kwargs
                )
                
                if response and response.get("response"):
                    print("    ✅ Unknown kwargs handled gracefully")
                else:
                    print("    ⚠️ Unknown kwargs caused no response")
                    
            except Exception as e:
                error_str = str(e).lower()
                if any(kwarg in error_str for kwarg in unknown_kwargs.keys()):
                    print(f"    ⚠️ Unknown kwargs rejected: {str(e)[:50]}...")
                else:
                    print(f"    ❌ Unexpected error: {str(e)[:50]}...")
            
            # Test 3: Mixed valid/invalid parameters
            print("  🔀 Testing mixed parameters...")
            
            try:
                response = await client.create_completion(
                    [{"role": "user", "content": "Mixed test"}],
                    temperature=0.7,          # Valid
                    max_tokens=30,            # Valid
                    invalid_param="ignore",   # Should be ignored
                    fake_setting=999          # Should be ignored
                )
                
                if response and response.get("response"):
                    print("    ✅ Mixed parameters handled correctly")
                else:
                    print("    ⚠️ Mixed parameters caused issues")
                    
            except Exception as e:
                print(f"    ❌ Mixed parameters failed: {str(e)[:50]}...")
            
            # Test 4: Edge cases
            print("  ⚠️  Testing edge cases...")
            
            edge_cases = [
                ("temperature", -1.0, "negative temp"),
                ("max_tokens", 0, "zero tokens"),
                ("top_p", 2.0, "top_p > 1"),
            ]
            
            edge_results = []
            
            for param, value, description in edge_cases:
                try:
                    kwargs = {param: value}
                    if provider == "anthropic" and param != "max_tokens":
                        kwargs["max_tokens"] = 50
                    
                    response = await client.create_completion(
                        [{"role": "user", "content": "Edge test"}],
                        **kwargs
                    )
                    
                    edge_results.append(f"⚠️ {description} accepted")
                    
                except Exception:
                    edge_results.append(f"✅ {description} rejected")
            
            print(f"    {', '.join(edge_results)}")
            
        except Exception as e:
            print(f"  ❌ Provider {provider} failed: {str(e)[:50]}...")
    
    print("\n" + "="*60)
    print("🎯 SUMMARY:")
    print("✅ = Parameter works correctly")
    print("⚠️ = Parameter has issues or is ignored") 
    print("❌ = Parameter causes errors")
    print("\n💡 TIP: Run the full parameter test for detailed analysis:")
    print("uv run diagnostics/compatibility/parameter_compatibility_test.py")

if __name__ == "__main__":
    asyncio.run(quick_parameter_test())