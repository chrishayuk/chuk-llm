#!/usr/bin/env python3
"""
Script to discover what functions are actually available.
"""


def main():
    print("🔍 Discovering Available Functions")
    print("=" * 50)

    try:
        # Get functions from the providers module
        from chuk_llm.api.providers import list_provider_functions
        from chuk_llm.configuration import get_config

        all_functions = list_provider_functions()
        config = get_config()
        providers = config.get_all_providers()

        print(f"📊 Total functions found: {len(all_functions)}")
        print()

        # Categorize functions
        sync_functions = [f for f in all_functions if f.endswith("_sync")]
        async_functions = [
            f for f in all_functions if f.startswith("ask_") and not f.endswith("_sync")
        ]
        stream_functions = [f for f in all_functions if f.startswith("stream_")]
        utility_functions = [
            f
            for f in all_functions
            if f in ["quick_question", "compare_providers", "show_config"]
        ]

        print("Function breakdown:")
        print(f"   Sync functions: {len(sync_functions)}")
        print(f"   Async functions: {len(async_functions)}")
        print(f"   Stream functions: {len(stream_functions)}")
        print(f"   Utility functions: {len(utility_functions)}")
        print()

        # Look for global aliases (functions that don't start with ask_<provider>)
        global_aliases = []
        provider_functions = []

        for func in sync_functions:
            # Check if it starts with ask_<provider>_
            is_provider_function = False
            for provider in providers:
                if func.startswith(f"ask_{provider}_"):
                    is_provider_function = True
                    provider_functions.append(func)
                    break

            if not is_provider_function and func != "ask_sync":
                global_aliases.append(func)

        print(f"📊 Global Alias Functions ({len(global_aliases)}):")
        for func in sorted(global_aliases)[:10]:
            print(f"   {func}")
        if len(global_aliases) > 10:
            print(f"   ... and {len(global_aliases) - 10} more")
        print()

        # Show provider breakdown
        print("📊 Functions by Provider:")
        for provider in sorted(providers):
            provider_funcs = [
                f for f in sync_functions if f.startswith(f"ask_{provider}_")
            ]
            print(f"   {provider}: {len(provider_funcs)} functions")
        print()

        # Look for specific popular functions
        print("🔍 Popular Function Examples:")

        examples = [
            "ask_openai_gpt4o_mini_sync",
            "ask_anthropic_sonnet_sync",
            "ask_groq_llama_sync",
            "ask_gpt4_sync",
            "ask_claude_sync",
            "ask_llama_sync",
        ]

        for func_name in examples:
            if func_name in all_functions:
                print(f"   ✅ {func_name}")
            else:
                print(f"   ❌ {func_name}")

        print()

        # Test actual imports
        print("🧪 Testing Actual Imports:")

        import chuk_llm

        test_functions = [
            "ask_sync",
            "ask_openai_sync",
            "ask_anthropic_sync",
            "ask_openai_gpt4o_mini_sync",
        ]

        for func_name in test_functions:
            try:
                func = getattr(chuk_llm, func_name)
                print(f"   ✅ {func_name} - {type(func).__name__}")
            except AttributeError as e:
                print(f"   ❌ {func_name} - {e}")

        # Show a sample of what's actually available
        print("\n📋 Sample of Available Functions:")
        sample_functions = sorted(sync_functions)[:15]
        for func in sample_functions:
            print(f"   {func}")
        if len(sync_functions) > 15:
            print(f"   ... and {len(sync_functions) - 15} more sync functions")

    except Exception as e:
        print(f"❌ Error discovering functions: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
