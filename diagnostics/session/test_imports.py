#!/usr/bin/env python3
# test_imports.py
"""Test basic imports to debug the issue."""

import sys

print(f"Python path: {sys.path}")

try:
    print("\n1. Testing chuk_llm import...")
    import chuk_llm

    print(f"   ✓ chuk_llm imported from: {chuk_llm.__file__}")
    print(f"   ✓ Version: {chuk_llm.__version__}")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("\n2. Testing chuk_llm.api import...")
    import chuk_llm.api

    print("   ✓ chuk_llm.api imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("\n3. Testing core functions import...")

    print("   ✓ Core functions imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("\n4. Testing session functions import...")

    print("   ✓ Session functions imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

try:
    print("\n5. Testing session manager import...")
    import chuk_ai_session_manager

    print("   ✓ chuk_ai_session_manager imported")
    print(f"   ✓ Version: {chuk_ai_session_manager.__version__}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n✅ Import test complete!")
