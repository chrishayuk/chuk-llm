#!/usr/bin/env python3
"""Debug script to check what's being imported"""

import sys

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\n--- Checking chuk_llm imports ---")

# First check what's in chuk_llm
try:
    import chuk_llm

    print(f"chuk_llm imported from: {chuk_llm.__file__}")
    print(
        f"chuk_llm attributes: {[attr for attr in dir(chuk_llm) if not attr.startswith('_')][:20]}..."
    )
except Exception as e:
    print(f"Error importing chuk_llm: {e}")

print("\n--- Checking api.providers ---")

# Check what's in providers
try:
    from chuk_llm.api import providers

    print(f"providers imported from: {providers.__file__}")
    provider_funcs = [
        attr for attr in dir(providers) if attr.startswith(("ask_", "stream_"))
    ]
    print(f"Provider functions found: {len(provider_funcs)}")
    print(f"First 10: {provider_funcs[:10]}")

    # Check specifically for ask_claude
    if "ask_claude" in dir(providers):
        print("\n✓ ask_claude found in providers module")
    else:
        print("\n✗ ask_claude NOT found in providers module")

    # Check __all__
    if hasattr(providers, "__all__"):
        print(f"\nproviders.__all__ has {len(providers.__all__)} items")
        claude_funcs = [f for f in providers.__all__ if "claude" in f]
        print(f"Claude-related functions: {claude_funcs}")

except Exception as e:
    print(f"Error checking providers: {e}")
    import traceback

    traceback.print_exc()

print("\n--- Checking configuration ---")

# Check if config loads properly
try:
    from chuk_llm.configuration import get_config

    config = get_config()
    aliases = config.get_global_aliases()
    print(f"Global aliases found: {len(aliases)}")
    if "claude" in aliases:
        print(f"  claude -> {aliases['claude']}")
    if "opus" in aliases:
        print(f"  opus -> {aliases['opus']}")
except Exception as e:
    print(f"Error loading config: {e}")

print("\n--- Direct import test ---")

# Try to import ask_claude directly
try:
    from chuk_llm.api.providers import ask_claude

    print("✓ Direct import of ask_claude from providers works!")
except ImportError as e:
    print(f"✗ Direct import of ask_claude failed: {e}")

# Try from main package
try:
    from chuk_llm import ask_claude  # noqa: F401

    print("✓ Import of ask_claude from main package works!")
except ImportError as e:
    print(f"✗ Import of ask_claude from main package failed: {e}")
