"""
Test Modern Client Integration
===============================

Quick test to verify the main API now uses modern clients internally.
"""

import asyncio
import logging
import os

logging.basicConfig(level=logging.DEBUG)


async def test_modern_ask():
    """Test that ask() uses modern client for OpenAI."""
    from chuk_llm.api import ask

    # Check if we have API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping test - OPENAI_API_KEY not set")
        return

    print("\n" + "=" * 60)
    print("Testing ask() with OpenAI (should use modern client)")
    print("=" * 60)

    response = await ask(
        prompt="Say 'hello' in one word",
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=10,
    )

    print(f"\nResponse: {response}")
    print(f"Type: {type(response)}")

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"

    print("\n✅ Test passed! ask() works with modern client integration")


async def test_anthropic():
    """Test Anthropic modern client."""
    from chuk_llm.api import ask

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Skipping Anthropic test - ANTHROPIC_API_KEY not set")
        return

    print("\n" + "=" * 60)
    print("Testing ask() with Anthropic (should use modern client)")
    print("=" * 60)

    response = await ask(
        prompt="Say 'hello' in one word",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
    )

    print(f"\nResponse: {response}")
    print(f"Type: {type(response)}")

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"

    print("\n✅ Test passed! Anthropic works with modern client")


async def main():
    """Run all tests."""
    try:
        await test_modern_ask()
    except Exception as e:
        print(f"\n❌ OpenAI test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        await test_anthropic()
    except Exception as e:
        print(f"\n❌ Anthropic test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
