#!/usr/bin/env python3
"""
Test Generated Functions for Discovered Models

This checks if ChukLLM automatically generated convenience functions
for your discovered Ollama models like ask_ollama_devstral.
"""

import asyncio


def check_generated_functions():
    """Check what functions were generated"""

    print("🔍 Checking Generated Functions")
    print("=" * 40)

    try:
        import chuk_llm

        # Get all functions that start with ask_ollama_
        ollama_functions = []

        for attr_name in dir(chuk_llm):
            if attr_name.startswith("ask_ollama_"):
                attr = getattr(chuk_llm, attr_name)
                if callable(attr):
                    ollama_functions.append(attr_name)

        print(f"📊 Found {len(ollama_functions)} generated Ollama functions:")

        # Group by type
        standard_models = []
        discovered_models = []

        # Expected discovered models based on your setup

        for func_name in sorted(ollama_functions):
            if any(
                expected in func_name
                for expected in [
                    "devstral",
                    "qwen3_32b",
                    "phi4_reasoning",
                    "llama3_2_vision",
                ]
            ):
                discovered_models.append(func_name)
                print(f"   🔍 {func_name} (discovered)")
            else:
                standard_models.append(func_name)
                print(f"   📋 {func_name} (static)")

        print("\n📈 Summary:")
        print(f"   Static model functions: {len(standard_models)}")
        print(f"   Discovered model functions: {len(discovered_models)}")

        # Check for key discovered functions
        key_functions = ["ask_ollama_devstral", "ask_ollama_qwen3_32b"]

        print("\n🎯 Key Discovered Functions:")
        for func_name in key_functions:
            if func_name in ollama_functions:
                print(f"   ✅ {func_name} - AVAILABLE")
            else:
                print(f"   ❌ {func_name} - NOT FOUND")

        return ollama_functions

    except Exception as e:
        print(f"❌ Function check failed: {e}")
        return []


async def test_discovered_functions():
    """Test the discovered model functions"""

    print("\n🧪 Testing Discovered Functions...")

    try:
        import chuk_llm

        # Test functions for your key discovered models
        test_functions = [
            ("ask_ollama_devstral", "Write hello world in Python"),
            ("ask_ollama_qwen3_32b", "What is 7 * 8?"),
            ("ask_ollama_phi4_reasoning", "Is 17 a prime number?"),
        ]

        for func_name, test_prompt in test_functions:
            if hasattr(chuk_llm, func_name):
                try:
                    print(f"   🧪 Testing {func_name}...")

                    func = getattr(chuk_llm, func_name)
                    response = await func(test_prompt, max_tokens=50)

                    print(f"      ✅ Success: '{response[:60]}...'")

                except Exception as func_error:
                    print(f"      ❌ Failed: {str(func_error)[:50]}...")
            else:
                print(f"   ❌ {func_name} - Function not found")

    except Exception as e:
        print(f"❌ Function testing failed: {e}")


def inspect_function_generation():
    """Inspect how functions are generated"""

    print("\n🔧 Inspecting Function Generation...")

    try:
        # Check the providers module where functions are generated
        import chuk_llm.api.providers as providers_module

        print("📋 Providers module attributes:")
        provider_attrs = [
            attr for attr in dir(providers_module) if not attr.startswith("_")
        ]

        ollama_attrs = [attr for attr in provider_attrs if "ollama" in attr.lower()]
        print(f"   Ollama-related: {len(ollama_attrs)} attributes")

        for attr in sorted(ollama_attrs)[:10]:  # Show first 10
            print(f"      • {attr}")
        if len(ollama_attrs) > 10:
            print(f"      ... and {len(ollama_attrs) - 10} more")

        # Check if dynamic generation is working
        if hasattr(providers_module, "_generate_provider_functions"):
            print("   ✅ Dynamic generation function found")
        else:
            print("   ⚠️  Dynamic generation function not found")

        # Check for discovered models in __all__
        if hasattr(providers_module, "__all__"):
            all_exports = providers_module.__all__
            discovered_exports = [
                exp
                for exp in all_exports
                if any(disc in exp for disc in ["devstral", "qwen3_32b", "phi4"])
            ]
            print(f"   📊 Discovered model exports: {len(discovered_exports)}")
            for exp in discovered_exports[:5]:
                print(f"      • {exp}")

    except Exception as e:
        print(f"❌ Generation inspection failed: {e}")


def check_function_signatures():
    """Check the signatures of generated functions"""

    print("\n📝 Checking Function Signatures...")

    try:
        import inspect

        import chuk_llm

        test_functions = ["ask_ollama_devstral", "ask_ollama_qwen3_32b"]

        for func_name in test_functions:
            if hasattr(chuk_llm, func_name):
                func = getattr(chuk_llm, func_name)
                signature = inspect.signature(func)

                print(f"   📋 {func_name}:")
                print(f"      Signature: {signature}")
                print(f"      Parameters: {list(signature.parameters.keys())}")

                # Check docstring
                if func.__doc__:
                    print(f"      Docstring: {func.__doc__[:100]}...")
                else:
                    print("      Docstring: None")
            else:
                print(f"   ❌ {func_name} - Not found")

    except Exception as e:
        print(f"❌ Signature check failed: {e}")


async def main():
    """Main function"""

    # Check what functions exist
    functions = check_generated_functions()

    if functions:
        # Test the functions
        await test_discovered_functions()

        # Inspect the generation process
        inspect_function_generation()

        # Check function signatures
        check_function_signatures()

        print("\n🎉 Function Generation Analysis Complete!")

        if any("devstral" in f for f in functions):
            print("✅ Dynamic function generation is working!")
            print("💡 You can use: await chuk_llm.ask_ollama_devstral('Write code')")
        else:
            print("⚠️  Dynamic function generation may not be working")
            print("🔧 Functions might need manual refresh or different trigger")
    else:
        print("\n💥 No generated functions found")
        print("🔧 Dynamic function generation may need debugging")

    print("\n📖 Expected Usage:")
    print("   await chuk_llm.ask_ollama_devstral('Write Python code')")
    print("   await chuk_llm.ask_ollama_qwen3_32b('Solve math problem')")
    print("   await chuk_llm.ask_ollama_phi4_reasoning('Logical reasoning')")


if __name__ == "__main__":
    asyncio.run(main())
