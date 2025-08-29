#!/usr/bin/env python3
"""
Working sync sample script that demonstrates ChukLLM functionality.
Fixed version with proper error handling and import safety.
"""

import os
import sys
import warnings
from collections.abc import Callable

from dotenv import load_dotenv

load_dotenv()

# Suppress async warnings for cleaner demo output
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*Event loop is closed.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*"
)


def safe_import_chuk_llm():
    """Safely import chuk_llm with fallback handling."""
    try:
        import chuk_llm

        return chuk_llm
    except Exception as e:
        print(f"❌ Failed to import chuk_llm: {e}")
        sys.exit(1)


# Import ChukLLM
chuk_llm = safe_import_chuk_llm()


def safe_get_function(name: str) -> Callable | None:
    """Safely get a function from chuk_llm module."""
    try:
        return getattr(chuk_llm, name)
    except AttributeError:
        return None


def test_provider(
    name: str, func: Callable | None, question: str = "What's 2+2? Answer briefly."
) -> bool:
    """Helper to test a provider function with error handling."""
    if func is None:
        print(f"🤖 {name}")
        print("   ⚠️  Function not available")
        print()
    print("🔑 API KEY STATUS:")

    # Check which API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
    }

    for key, value in api_keys.items():
        if value:
            # Show first 8 and last 4 characters for security
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***set***"
            print(f"   ✅ {key}: {masked}")
        else:
            print(f"   ❌ {key}: Not set")

    if not any(api_keys.values()):
        print("   💡 Create a .env file with your API keys to test actual responses")
        print()
        return False

    print(f"🤖 {name}")
    try:
        response = func(question)
        # Truncate very long responses for demo
        if len(str(response)) > 100:
            response = str(response)[:97] + "..."
        print(f"   {response}")
        print()
        return True
    except Exception as e:
        error_msg = str(e)
        # Clean up common error messages for readability
        if "api_key" in error_msg.lower():
            if "openai" in error_msg.lower():
                error_msg = (
                    "No OpenAI API key set (OPENAI_API_KEY environment variable)"
                )
            elif "anthropic" in error_msg.lower():
                error_msg = (
                    "No Anthropic API key set (ANTHROPIC_API_KEY environment variable)"
                )
            elif "groq" in error_msg.lower():
                error_msg = "No Groq API key set (GROQ_API_KEY environment variable)"
            elif "google" in error_msg.lower() or "gemini" in error_msg.lower():
                error_msg = (
                    "No Google API key set (GEMINI_API_KEY environment variable)"
                )
        elif "authentication" in error_msg.lower():
            error_msg = "Invalid API key or authentication failed"
        elif "401" in error_msg:
            error_msg = "Invalid API key (401 Unauthorized)"
        elif "404" in error_msg and "ollama" in error_msg.lower():
            error_msg = "Ollama model not found - try: ollama pull <model-name>"

        print(f"   ❌ Error: {error_msg}")
        print()
        return False


def get_available_functions() -> dict[str, Callable]:
    """Get all available functions from chuk_llm."""
    functions = {}

    # Core functions (guaranteed to exist)
    core_functions = ["ask_sync", "quick_question", "compare_providers", "show_config"]

    for func_name in core_functions:
        func = safe_get_function(func_name)
        if func:
            functions[func_name] = func

    # Provider base functions
    providers = [
        "openai",
        "anthropic",
        "groq",
        "gemini",
        "mistral",
        "ollama",
        "deepseek",
    ]
    for provider in providers:
        func_name = f"ask_{provider}_sync"
        func = safe_get_function(func_name)
        if func:
            functions[func_name] = func

    # Global aliases (try common ones)
    global_aliases = [
        "ask_gpt4_sync",
        "ask_gpt4_mini_sync",
        "ask_claude_sync",
        "ask_claude4_sync",
        "ask_llama_sync",
        "ask_fastest_sync",
        "ask_smartest_sync",
        "ask_creative_sync",
        "ask_coding_sync",
        "ask_cheapest_sync",
    ]

    for alias in global_aliases:
        func = safe_get_function(alias)
        if func:
            functions[alias] = func

    # Provider-specific functions (try common ones)
    specific_functions = [
        "ask_openai_gpt4o_sync",
        "ask_openai_gpt4o_mini_sync",
        "ask_anthropic_opus_sync",
        "ask_anthropic_sonnet_sync",
        "ask_groq_instant_sync",
        "ask_groq_llama_sync",
    ]

    for func_name in specific_functions:
        func = safe_get_function(func_name)
        if func:
            functions[func_name] = func

    return functions


def main():
    question = "What's the capital of France? Answer in one word."

    print("=" * 70)
    print("🚀 ChukLLM Working Demo Script")
    print("=" * 70)
    print()

    # Get available functions
    functions = get_available_functions()

    # Show configuration info if available
    show_config = safe_get_function("show_config")
    if show_config:
        try:
            print("📊 Configuration Status:")
            show_config()
            print()
        except Exception as e:
            print(f"📊 Configuration check failed: {e}")
            print()

    print("📊 Function Availability Check:")
    print(f"   Total functions found: {len(functions)}")

    # Categorize functions
    core_funcs = [
        f for f in functions if f in ["ask_sync", "quick_question", "compare_providers"]
    ]
    provider_funcs = [
        f
        for f in functions
        if f.startswith("ask_") and f.endswith("_sync") and "_" in f[4:-5]
    ]
    global_funcs = [
        f
        for f in functions
        if f.startswith("ask_") and f.endswith("_sync") and "_" not in f[4:-5]
    ]

    print(f"   Core functions: {len(core_funcs)}")
    print(f"   Provider functions: {len(provider_funcs)}")
    print(f"   Global aliases: {len(global_funcs)}")

    if global_funcs:
        print(f"   Examples: {', '.join(global_funcs[:5])}")

    # Show API key status once
    print()
    print("🔑 API KEY STATUS:")

    # Check which API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
    }

    for key, value in api_keys.items():
        if value:
            # Show first 8 and last 4 characters for security
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***set***"
            print(f"   ✅ {key}: {masked}")
        else:
            print(f"   ❌ {key}: Not set")

    if not any(api_keys.values()):
        print("   💡 Create a .env file with your API keys to test actual responses")
    print()

    # Test default provider
    print("🌟 DEFAULT PROVIDER")
    ask_sync = functions.get("ask_sync")
    test_provider("Default (ask_sync)", ask_sync, question)

    # Test global aliases if available
    global_aliases_to_test = [
        ("ask_gpt4_sync", "GPT-4 Global"),
        ("ask_claude4_sync", "Claude 4 Global"),
        ("ask_llama_sync", "Llama Global"),
    ]

    available_globals = [
        (name, desc) for name, desc in global_aliases_to_test if name in functions
    ]

    if available_globals:
        print("⚡ GLOBAL ALIASES")
        for func_name, description in available_globals:
            test_provider(description, functions[func_name], question)

    # Test capability-based functions
    capability_functions = [
        ("ask_fastest_sync", "Fastest Model"),
        ("ask_smartest_sync", "Smartest Model"),
        ("ask_creative_sync", "Most Creative"),
        ("ask_cheapest_sync", "Most Cost-Effective"),
    ]

    available_capabilities = [
        (name, desc) for name, desc in capability_functions if name in functions
    ]

    if available_capabilities:
        print("🎯 CAPABILITY-BASED FUNCTIONS")
        for func_name, description in available_capabilities:
            test_provider(description, functions[func_name], question)

    # Test major providers
    providers_to_test = [
        ("ask_openai_sync", "OpenAI"),
        ("ask_anthropic_sync", "Anthropic"),
        ("ask_groq_sync", "Groq"),
        ("ask_gemini_sync", "Google Gemini"),
        ("ask_mistral_sync", "Mistral"),
        ("ask_ollama_sync", "Ollama"),
        ("ask_deepseek_sync", "DeepSeek"),
    ]

    print("☁️  MAJOR PROVIDERS")
    for func_name, description in providers_to_test:
        func = functions.get(func_name)
        test_provider(description, func, question)

    # Test some provider-specific functions if available
    specific_to_test = [
        ("ask_openai_gpt4o_sync", "OpenAI GPT-4o"),
        ("ask_anthropic_opus_sync", "Anthropic Opus"),
        ("ask_groq_instant_sync", "Groq Instant"),
    ]

    available_specific = [
        (name, desc) for name, desc in specific_to_test if name in functions
    ]

    if available_specific:
        print("🔧 PROVIDER-SPECIFIC FUNCTIONS")
        for func_name, description in available_specific:
            test_provider(description, functions[func_name], question)

    # Test utility functions
    print("🛠️  UTILITY FUNCTIONS")

    quick_question = functions.get("quick_question")
    if quick_question:
        print("🤖 Quick Question")
        try:
            response = quick_question("What's 1+1?")
            print(f"   {response}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()

    compare_providers = functions.get("compare_providers")
    if compare_providers:
        print("🤖 Compare Providers")
        try:
            # Use a simpler question for comparison
            results = compare_providers(
                "What's the square root of 16?", ["openai", "anthropic"]
            )
            for provider, response in results.items():
                # Truncate long responses
                if len(str(response)) > 80:
                    response = str(response)[:77] + "..."
                print(f"   {provider}: {response}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()

    # Test different task types
    print("📝 TASK DEMONSTRATIONS")

    # Creative task
    creative_question = "Write a haiku about coding. Be creative!"
    print("🎨 CREATIVE TASK")
    creative_func = functions.get("ask_creative_sync") or functions.get(
        "ask_anthropic_sync"
    )
    test_provider("Creative Model", creative_func, creative_question)

    # Speed test
    speed_question = "Name 3 colors. Be brief."
    print("⚡ SPEED TEST")
    speed_func = functions.get("ask_fastest_sync") or functions.get("ask_groq_sync")
    test_provider("Fastest Model", speed_func, speed_question)

    # Cost test
    cost_question = "What's 2*3? Answer briefly."
    print("💰 COST-EFFECTIVE TEST")
    cost_func = functions.get("ask_cheapest_sync") or functions.get(
        "ask_openai_gpt4o_mini_sync"
    )
    test_provider("Cheapest Model", cost_func, cost_question)

    print("=" * 70)
    print("✅ Demo complete!")
    print()
    print("💡 WHAT WE DEMONSTRATED:")
    print(f"   ✅ Core functions: {len(core_funcs)} available")
    print(f"   ✅ Provider functions: {len(provider_funcs)} available")
    print(f"   ✅ Global aliases: {len(global_funcs)} available")
    print(f"   📊 Total functions tested: {len(functions)}")
    print()
    print("🔑 TO USE WITH YOUR API KEYS:")
    print("   1. Create a .env file in your project root:")
    print("      OPENAI_API_KEY=your-openai-key")
    print("      ANTHROPIC_API_KEY=your-anthropic-key")
    print("      GROQ_API_KEY=your-groq-key")
    print("      GEMINI_API_KEY=your-google-key")
    print("   2. Or export them as environment variables")
    print("   3. Re-run this demo to see actual AI responses!")
    print()
    print("💻 To see all available functions:")
    print(
        '   python -c "import chuk_llm; print(len(chuk_llm.__all__)); print(chuk_llm.__all__[:10])"'
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
