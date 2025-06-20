#!/usr/bin/env python3
# verify_setup.py
"""
Verify CHUK LLM setup and session integration
"""

import os
import sys

print("🔍 CHUK LLM Setup Verification")
print("=" * 40)

# Check Python version
print(f"\n📌 Python: {sys.version}")

# Check if we're in a virtual environment
print(f"\n📌 Virtual env: {sys.prefix}")

# Check installed packages
print("\n📌 Checking installed packages:")
try:
    import chuk_llm
    print(f"  ✅ chuk_llm: {chuk_llm.__version__}")
except ImportError as e:
    print(f"  ❌ chuk_llm: Not installed or import error: {e}")

try:
    import chuk_ai_session_manager
    print(f"  ✅ chuk_ai_session_manager: {chuk_ai_session_manager.__version__}")
except ImportError:
    print(f"  ❌ chuk_ai_session_manager: Not installed")

# Check environment variables
print("\n📌 Environment variables:")
api_keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
}

for key, value in api_keys.items():
    if value:
        print(f"  ✅ {key}: {'*' * 8}...")
    else:
        print(f"  ❌ {key}: Not set")

# Check session settings
sessions_disabled = os.getenv("CHUK_LLM_DISABLE_SESSIONS", "").lower() in ("true", "1", "yes")
print(f"\n📌 Session tracking: {'Disabled' if sessions_disabled else 'Enabled'}")

# Try importing core components
print("\n📌 Testing imports:")
try:
    from chuk_llm.api.core import ask, get_session_stats
    print("  ✅ Core functions imported")
except ImportError as e:
    print(f"  ❌ Core import failed: {e}")

try:
    from chuk_llm.api.conversation import conversation
    print("  ✅ Conversation imported")
except ImportError as e:
    print(f"  ❌ Conversation import failed: {e}")

print("\n✅ Setup verification complete!")

# Run instructions
print("\n📋 Next steps:")
print("1. If imports failed, check your project structure")
print("2. Make sure you're running from the project root")
print("3. Try: uv run python examples/test_session_integration.py")
print("4. Or: cd examples && uv run python test_session_integration.py")