#!/usr/bin/env python3
"""
Comprehensive Ollama version and capability checker
"""

import asyncio
import inspect
import sys
from pathlib import Path


def check_ollama_capabilities():
    """Check what's available in the installed Ollama package"""
    print("🔍 OLLAMA CAPABILITY CHECK")
    print("=" * 60)

    try:
        import ollama

        # Check version
        version = getattr(ollama, "__version__", "Version not found")
        print(f"📦 Ollama package version: {version}")

        # Check module path
        print(f"📁 Module location: {ollama.__file__}")

        # Check available functions and classes
        print("\n📚 Available top-level items:")
        items = dir(ollama)

        # Filter for likely API functions
        functions = []
        classes = []
        for item in items:
            if not item.startswith("_"):
                obj = getattr(ollama, item)
                if inspect.isfunction(obj):
                    functions.append(item)
                elif inspect.isclass(obj):
                    classes.append(item)

        print(f"   Functions: {', '.join(functions)}")
        print(f"   Classes: {', '.join(classes)}")

        # Check chat function signature
        if hasattr(ollama, "chat"):
            print("\n📝 ollama.chat() signature:")
            sig = inspect.signature(ollama.chat)
            print(f"   Parameters: {list(sig.parameters.keys())}")

            # Check if 'think' is a parameter
            if "think" in sig.parameters:
                print("   ✅ 'think' parameter is supported in chat()")
            else:
                print("   ❌ 'think' parameter NOT found in chat()")

        # Check AsyncClient
        if hasattr(ollama, "AsyncClient"):
            print("\n📝 AsyncClient.chat() signature:")
            client = ollama.AsyncClient()
            if hasattr(client, "chat"):
                sig = inspect.signature(client.chat)
                print(f"   Parameters: {list(sig.parameters.keys())}")

                if "think" in sig.parameters:
                    print("   ✅ 'think' parameter is supported in AsyncClient.chat()")
                else:
                    print("   ❌ 'think' parameter NOT found in AsyncClient.chat()")

        # Check Client (sync)
        if hasattr(ollama, "Client"):
            print("\n📝 Client.chat() signature:")
            client = ollama.Client()
            if hasattr(client, "chat"):
                sig = inspect.signature(client.chat)
                print(f"   Parameters: {list(sig.parameters.keys())}")

                if "think" in sig.parameters:
                    print("   ✅ 'think' parameter is supported in Client.chat()")
                else:
                    print("   ❌ 'think' parameter NOT found in Client.chat()")

        # Check for _types module
        if hasattr(ollama, "_types"):
            print("\n📝 Types module found")
            types_items = dir(ollama._types)
            response_types = [item for item in types_items if "Response" in item]
            print(f"   Response types: {', '.join(response_types)}")

            # Check ChatResponse
            if hasattr(ollama._types, "ChatResponse"):
                print("\n📝 ChatResponse attributes:")
                # Try to inspect the class
                chat_response = ollama._types.ChatResponse
                if hasattr(chat_response, "__annotations__"):
                    print(f"   Fields: {list(chat_response.__annotations__.keys())}")

        return True

    except ImportError as e:
        print(f"❌ Ollama not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_thinking_field():
    """Test if the thinking field is returned in responses"""
    print("\n" + "=" * 60)
    print("🧪 TESTING THINKING FIELD IN RESPONSES")
    print("=" * 60)

    try:
        import ollama

        # Test with a simple prompt
        print("\n Testing response structure...")

        client = ollama.AsyncClient()
        response = await client.chat(
            model="gpt-oss:latest",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            stream=False,
        )

        print(f"\n📊 Response type: {type(response)}")
        print(f"📊 Response attributes: {dir(response)}")

        if hasattr(response, "message"):
            message = response.message
            print(f"\n📊 Message type: {type(message)}")
            print(f"📊 Message attributes: {dir(message)}")

            # Check for thinking field
            if hasattr(message, "thinking"):
                thinking = message.thinking
                if thinking:
                    print(
                        f"✅ Thinking field exists and has content: {len(thinking)} chars"
                    )
                else:
                    print("ℹ️  Thinking field exists but is empty/None")
            else:
                print("❌ No 'thinking' attribute in message")

            # Check for content field
            if hasattr(message, "content"):
                content = message.content
                print(f"✅ Content field: {content}")

        return True

    except Exception as e:
        print(f"❌ Error testing thinking field: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_with_thinking_prompt():
    """Test with a prompt that should trigger thinking"""
    print("\n" + "=" * 60)
    print("🧪 TESTING WITH THINKING-TRIGGERING PROMPT")
    print("=" * 60)

    try:
        import ollama

        client = ollama.AsyncClient()

        # Try different prompts that might trigger thinking
        prompts = [
            "Think carefully: What is the square root of 144?",
            "Let me think step by step: If I have 5 apples and eat 2, how many are left?",
            "<thinking>What is 10 divided by 2?</thinking>",
        ]

        for prompt in prompts:
            print(f"\n📝 Testing prompt: {prompt[:50]}...")

            response = await client.chat(
                model="gpt-oss:latest",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )

            if hasattr(response, "message") and hasattr(response.message, "thinking"):
                thinking = response.message.thinking
                if thinking:
                    print(f"   ✅ Thinking populated: {len(thinking)} chars")
                    print(f"      Preview: {thinking[:100]}...")
                else:
                    print("   ℹ️  No thinking content")

            # Small delay between requests
            await asyncio.sleep(1)

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all checks"""
    # Check capabilities
    check_ollama_capabilities()

    # Run async tests
    asyncio.run(test_thinking_field())
    asyncio.run(test_with_thinking_prompt())

    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print("""
Based on the tests:
1. Check if 'think' parameter is available in your Ollama version
2. Check if 'thinking' field is returned in responses
3. Try different prompts to trigger thinking

If 'think' parameter is not available:
- Your Ollama version doesn't support it yet
- Use prompt engineering instead ("Think step by step")
- The model will still reason, just without explicit thinking field

To upgrade Ollama Python client:
  pip install --upgrade ollama
  or
  uv pip install --upgrade ollama
""")


if __name__ == "__main__":
    # Add project root to path if needed
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
