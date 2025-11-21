#!/usr/bin/env python3
import asyncio

# Import sync functions but handle them carefully
from chuk_llm import (
    ask,
    ask_claude,
    ask_openai,
    ask_openai_sync,
    ask_sync,
    configure,
    stream,
)


async def basic_examples():
    """Basic usage patterns with the functions we have."""

    print("🚀 Basic ChukLLM Examples")
    print("=" * 50)

    # 1. Simple ask
    print("\n1. Simple ask:")
    try:
        response = await ask("What is Python?")
        print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Configure and use
    print("\n2. Configure and use:")
    try:
        configure(provider="openai", model="gpt-4o-mini", temperature=0.7)
        response = await ask("Tell me a very short joke")
        print(f"Joke: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Provider shortcuts
    print("\n3. Provider shortcuts:")
    try:
        openai_response = await ask_openai("Hello from OpenAI!", model="gpt-4o-mini")
        print(f"OpenAI: {openai_response[:100]}...")
    except Exception as e:
        print(f"OpenAI Error: {e}")

    try:
        # Use the correct Claude model name
        claude_response = await ask_claude(
            "Hello from Claude!", model="claude-3-5-sonnet-20241022"
        )
        print(f"Claude: {claude_response[:100]}...")
    except Exception as e:
        print(f"Claude Error: {e}")

    # 4. Streaming (simplified)
    print("\n4. Streaming response:")
    try:
        print("Story: ", end="", flush=True)
        chunk_count = 0
        async for chunk in stream("Tell me a very short story about a robot"):
            print(chunk, end="", flush=True)
            chunk_count += 1
            if chunk_count > 50:  # Prevent infinite loop
                break
        print(f"\n(Received {chunk_count} chunks)")
    except Exception as e:
        print(f"Streaming Error: {e}")


def sync_examples_separate():
    """Run sync examples in a separate process to avoid event loop issues."""

    print("\n🔄 Synchronous Examples (separate script)")
    print("=" * 50)

    print("\n1. Simple sync ask:")
    try:
        response = ask_sync("What's 2+2?")
        print(f"Answer: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n2. Provider-specific sync:")
    try:
        response = ask_openai_sync("Hello from sync OpenAI!")
        print(f"OpenAI sync: {response[:100]}...")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run async examples."""

    print("🎯 ChukLLM Basic API Examples")
    print("=" * 70)

    try:
        await basic_examples()

        print("\n" + "=" * 70)
        print("✅ Async examples completed successfully!")
        print("\n💡 Available functions:")
        print("   • ask() - basic questions")
        print("   • stream() - streaming responses")
        print("   • configure() - set defaults")
        print("   • ask_openai(), ask_claude() - provider shortcuts")
        print("   • ask_sync() - synchronous version (use in separate script)")

        print("\n🔍 Note about sync functions:")
        print("   Sync functions can't be called from async context.")
        print("   Run them in a separate script or use the async versions.")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\n💡 Make sure you have API keys set in your environment:")
        print("   export OPENAI_API_KEY=your_key_here")
        print("   export ANTHROPIC_API_KEY=your_key_here")


def sync_main():
    """Entry point for sync-only examples."""
    sync_examples_separate()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        # Run sync examples only
        sync_main()
    else:
        # Run async examples
        asyncio.run(main())
