#!/usr/bin/env python3
"""
Debug script to see what functions are being generated and why the lookup might be failing
"""


def debug_function_generation():
    """Debug the function generation process"""

    print("=== Debugging Function Generation ===\n")

    # Step 1: Check configuration loading
    print("1. Testing configuration loading:")
    try:
        from chuk_llm.configuration import get_config

        config = get_config()
        providers = config.get_all_providers()
        print(f"✅ Found {len(providers)} providers: {providers}")

        # Check OpenAI specifically
        openai_config = config.get_provider("openai")
        print(f"✅ OpenAI models: {openai_config.models[:5]}...")  # Show first 5
        print(
            f"✅ OpenAI aliases: {list(openai_config.model_aliases.keys())[:5]}..."
        )  # Show first 5

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return

    print()

    # Step 2: Test sanitization function
    print("2. Testing name sanitization:")
    try:
        from chuk_llm.api.providers import _sanitize_name

        test_names = [
            "gpt-4o-mini",
            "gpt4o",
            "claude-3-sonnet-20240229",
            "llama-3.3-70b-versatile",
        ]
        for name in test_names:
            sanitized = _sanitize_name(name)
            print(f"   '{name}' -> '{sanitized}'")

    except Exception as e:
        print(f"❌ Sanitization testing failed: {e}")

    print()

    # Step 3: Check what functions are actually generated
    print("3. Testing function generation:")
    try:
        from chuk_llm.api.providers import list_provider_functions

        functions = list_provider_functions()
        print(f"✅ Generated {len(functions)} total functions")

        # Look for OpenAI functions specifically
        openai_functions = [
            f for f in functions if f.startswith("ask_openai_") and f.endswith("_sync")
        ]
        print(f"✅ OpenAI sync functions ({len(openai_functions)}):")
        for func in openai_functions[:10]:  # Show first 10
            print(f"   {func}")
        if len(openai_functions) > 10:
            print(f"   ... and {len(openai_functions) - 10} more")

        # Look for the specific function we want
        target_function = "ask_openai_gpt4o_mini_sync"
        if target_function in functions:
            print(f"✅ Found target function: {target_function}")
        else:
            print(f"❌ Target function not found: {target_function}")

            # Check what openai gpt4o functions exist
            gpt4o_functions = [f for f in functions if "openai" in f and "gpt4o" in f]
            print(f"   Available GPT-4o functions: {gpt4o_functions}")

    except Exception as e:
        print(f"❌ Function generation testing failed: {e}")

    print()

    # Step 4: Test direct access
    print("4. Testing direct function access:")
    try:
        from chuk_llm.api import providers

        # Try to access the function directly
        target_function = "ask_openai_gpt4o_mini_sync"
        if hasattr(providers, target_function):
            func = getattr(providers, target_function)
            print(f"✅ Direct access works: {func}")
        else:
            print(f"❌ Direct access failed for: {target_function}")

            # Check what attributes the module actually has
            attrs = [attr for attr in dir(providers) if attr.startswith("ask_openai_")]
            print(f"   Available ask_openai_ attributes: {attrs[:10]}")

    except Exception as e:
        print(f"❌ Direct access testing failed: {e}")

    print()

    # Step 5: Test main module access
    print("5. Testing main module access:")
    try:
        import chuk_llm

        target_function = "ask_openai_gpt4o_mini_sync"
        func = getattr(chuk_llm, target_function)
        print(f"✅ Main module access works: {func}")

    except AttributeError as e:
        print(f"❌ Main module access failed: {e}")
    except Exception as e:
        print(f"❌ Main module access error: {e}")

    print()

    # Step 6: Check specific model name resolution
    print("6. Testing model name resolution:")
    try:
        from chuk_llm.configuration import get_config

        config = get_config()
        openai_config = config.get_provider("openai")

        # Check if gpt4o_mini resolves to a real model
        aliases = openai_config.model_aliases
        models = openai_config.models

        print(f"   Looking for 'gpt4o_mini' in aliases: {'gpt4o_mini' in aliases}")
        if "gpt4o_mini" in aliases:
            print(f"   Alias 'gpt4o_mini' -> '{aliases['gpt4o_mini']}'")

        print(f"   Looking for 'gpt-4o-mini' in models: {'gpt-4o-mini' in models}")

        # Test the reverse sanitization
        from chuk_llm.api.providers import _sanitize_name

        for model in models:
            if _sanitize_name(model) == "gpt4o_mini":
                print(f"   Model '{model}' sanitizes to 'gpt4o_mini'")
                break
        else:
            print("   No model sanitizes to 'gpt4o_mini'")

    except Exception as e:
        print(f"❌ Model name resolution testing failed: {e}")


if __name__ == "__main__":
    debug_function_generation()
