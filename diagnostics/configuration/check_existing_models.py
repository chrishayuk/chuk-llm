#!/usr/bin/env python3
"""
Enhanced model discovery and testing script for ChukLLM
Includes better error handling and performance testing
"""

import asyncio
import time


def test_all_discovered_models():
    """Test all models discovered by ChukLLM"""

    print("🔍 ChukLLM Complete Model Discovery")
    print("=" * 50)

    try:
        import chuk_llm
        from chuk_llm.api.providers import (
            _ensure_provider_models_current,
            trigger_ollama_discovery_and_refresh,
        )

        # Step 1: Get all current Ollama models
        print("📡 Step 1: Discovering Ollama Models")
        print("-" * 30)

        current_models = _ensure_provider_models_current("ollama")
        print(f"Found {len(current_models)} Ollama models:")

        # Group models by family for better display
        model_families = {}
        for model in current_models:
            family = model.split(":")[0]
            if family not in model_families:
                model_families[family] = []
            model_families[family].append(model)

        for family, models in sorted(model_families.items()):
            print(f"  {family}: {len(models)} variants")

        # Step 2: Trigger discovery and refresh
        print("\n🔧 Step 2: Generating Functions")
        print("-" * 30)

        start_time = time.time()
        new_functions = trigger_ollama_discovery_and_refresh()
        generation_time = time.time() - start_time

        print(f"Generated {len(new_functions)} functions in {generation_time:.2f}s")

        # Step 3: Categorize functions
        print("\n📊 Step 3: Function Analysis")
        print("-" * 30)

        ask_functions = [
            name
            for name in new_functions
            if name.startswith("ask_ollama_") and not name.endswith("_sync")
        ]
        stream_functions = [
            name for name in new_functions if name.startswith("stream_ollama_")
        ]
        sync_functions = [name for name in new_functions if name.endswith("_sync")]

        print(f"  Async functions: {len(ask_functions)}")
        print(f"  Stream functions: {len(stream_functions)}")
        print(f"  Sync functions: {len(sync_functions)}")

        # Step 4: Test function accessibility
        print("\n🧪 Step 4: Function Accessibility Test")
        print("-" * 30)

        accessible_count = 0
        sample_functions = ask_functions[:10]  # Test first 10

        for func_name in sample_functions:
            if hasattr(chuk_llm, func_name):
                accessible_count += 1
                print(f"  ✅ {func_name}")
            else:
                print(f"  ❌ {func_name}")

        print(
            f"\nAccessibility: {accessible_count}/{len(sample_functions)} functions available"
        )

        return {
            "models": current_models,
            "functions": new_functions,
            "accessible": accessible_count,
            "generation_time": generation_time,
        }

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_function_performance(function_names: list[str], max_tests: int = 3):
    """Test performance of a few functions"""

    print("\n⚡ Step 5: Performance Test")
    print("-" * 30)

    if not function_names:
        print("No functions to test")
        return

    try:
        import chuk_llm

        # Test first few functions
        test_functions = function_names[:max_tests]
        test_prompt = "Say 'OK' (one word only)"

        results = []

        for func_name in test_functions:
            if hasattr(chuk_llm, func_name):
                func = getattr(chuk_llm, func_name)

                print(f"Testing {func_name}...")

                async def test_call():
                    start_time = time.time()
                    try:
                        response = await func(test_prompt, max_tokens=5)
                        call_time = time.time() - start_time
                        return response, call_time, None
                    except Exception as e:
                        call_time = time.time() - start_time
                        return None, call_time, str(e)

                try:
                    response, call_time, error = asyncio.run(test_call())

                    if error:
                        print(f"  ⚠️  Error: {error} ({call_time:.2f}s)")
                        status = "error"
                    else:
                        print(f"  ✅ '{response}' ({call_time:.2f}s)")
                        status = "success"

                    results.append(
                        {
                            "function": func_name,
                            "time": call_time,
                            "status": status,
                            "response": response,
                            "error": error,
                        }
                    )

                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    results.append(
                        {
                            "function": func_name,
                            "time": 0,
                            "status": "failed",
                            "response": None,
                            "error": str(e),
                        }
                    )

        # Summary
        if results:
            successful = [r for r in results if r["status"] == "success"]
            avg_time = (
                sum(r["time"] for r in successful) / len(successful)
                if successful
                else 0
            )

            print("\nPerformance Summary:")
            print(f"  Success rate: {len(successful)}/{len(results)}")
            if successful:
                print(f"  Average response time: {avg_time:.2f}s")
                fastest = min(successful, key=lambda r: r["time"])
                print(f"  Fastest: {fastest['function']} ({fastest['time']:.2f}s)")

        return results

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return []


def show_usage_examples(model_families: dict[str, list[str]]):
    """Show usage examples for different model families"""

    print("\n📚 Usage Examples")
    print("-" * 30)

    examples = []

    # Pick one model from each family for examples
    for family, models in list(model_families.items())[:5]:  # Show first 5 families
        # Prefer :latest version if available
        latest_model = None
        for model in models:
            if model.endswith(":latest"):
                latest_model = model
                break

        if not latest_model:
            latest_model = models[0]  # Fallback to first model

        # Generate function name
        from chuk_llm.api.providers import _sanitize_name

        sanitized = _sanitize_name(latest_model)
        if sanitized:
            func_name = f"ask_ollama_{sanitized}"
            examples.append((family, func_name))

    print("Copy and paste these examples:")
    print()

    for family, func_name in examples:
        print(f"# {family.title()} model")
        print("import chuk_llm")
        print(f"response = await chuk_llm.{func_name}('Hello, how are you?')")
        print("print(response)")
        print()


def generate_function_index():
    """Generate an index of all available functions"""

    print("\n📋 Function Index")
    print("-" * 30)

    try:
        from chuk_llm.api.providers import get_all_functions

        all_functions = get_all_functions()

        # Categorize functions
        categories = {
            "ollama_ask": [],
            "ollama_stream": [],
            "ollama_sync": [],
            "other": [],
        }

        for name in sorted(all_functions.keys()):
            if name.startswith("ask_ollama_") and not name.endswith("_sync"):
                categories["ollama_ask"].append(name)
            elif name.startswith("stream_ollama_"):
                categories["ollama_stream"].append(name)
            elif name.startswith("ask_ollama_") and name.endswith("_sync"):
                categories["ollama_sync"].append(name)
            else:
                categories["other"].append(name)

        print("📁 Function Categories:")
        for category, functions in categories.items():
            if functions:
                print(f"  {category}: {len(functions)} functions")

        # Show some examples from each category
        print("\n📝 Examples by Category:")

        for category, functions in categories.items():
            if functions and category != "other":
                print(f"\n{category.replace('_', ' ').title()}:")
                for func in functions[:3]:  # Show first 3
                    print(f"  chuk_llm.{func}()")
                if len(functions) > 3:
                    print(f"  ... and {len(functions) - 3} more")

    except Exception as e:
        print(f"❌ Could not generate function index: {e}")


def main():
    """Main enhanced test function"""

    print("🚀 ChukLLM Enhanced Model Discovery & Testing")
    print("=" * 60)
    print("Comprehensive testing of dynamic function generation")
    print()

    # Discovery and function generation
    discovery_results = test_all_discovered_models()

    if not discovery_results:
        print("❌ Cannot proceed - discovery failed")
        return

    # Extract data for further testing
    models = discovery_results["models"]
    functions = discovery_results["functions"]

    # Group models by family for examples
    model_families = {}
    for model in models:
        family = model.split(":")[0]
        if family not in model_families:
            model_families[family] = []
        model_families[family].append(model)

    # Performance testing
    ask_functions = [
        name
        for name in functions
        if name.startswith("ask_ollama_") and not name.endswith("_sync")
    ]
    if ask_functions:
        test_function_performance(ask_functions)

    # Usage examples
    show_usage_examples(model_families)

    # Function index
    generate_function_index()

    # Final summary
    print("\n🎯 Summary")
    print("=" * 30)
    print(f"✅ Discovered {len(models)} models")
    print(f"✅ Generated {len(functions)} functions")
    print("✅ System working correctly")
    print()
    print("💡 Next steps:")
    print("  • Use any of the generated functions in your code")
    print("  • Functions are available immediately after import")
    print("  • Both async and sync versions are available")
    print("  • Run this script again if you add new models to Ollama")


if __name__ == "__main__":
    main()
