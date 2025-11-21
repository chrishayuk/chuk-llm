# examples/modular_api_usage.py
"""Examples showing the clean modular API in action."""

import asyncio

from chuk_llm import (
    ask,
    ask_claude,
    ask_openai,
    ask_sync,
    compare_providers,
    configure,
    conversation,
    conversation_sync,
    get_metrics,
    print_diagnostics,
    quick_question,
    stream,
    test_all_providers,
)


async def basic_examples():
    """Basic usage patterns with the modular API."""

    print("🚀 Basic Examples")
    print("=" * 50)

    # 1. Simplest possible usage
    print("\n1. Simple ask:")
    response = await ask("What is Python?")
    print(f"Response: {response[:100]}...")

    # 2. Configure once, use everywhere
    print("\n2. Configure and use:")
    configure(provider="openai", model="gpt-4o-mini", temperature=0.7)

    response1 = await ask("Question 1")
    response2 = await ask("Question 2")
    print(f"Configured responses received: {len(response1)} and {len(response2)} chars")

    # 3. Provider shortcuts
    print("\n3. Provider shortcuts:")
    openai_response = await ask_openai("Hello from OpenAI!")
    claude_response = await ask_claude("Hello from Claude!")
    print(f"OpenAI: {openai_response[:50]}...")
    print(f"Claude: {claude_response[:50]}...")

    # 4. Streaming
    print("\n4. Streaming response:")
    print("Story: ", end="", flush=True)
    async for chunk in stream("Tell me a very short story"):
        print(chunk, end="", flush=True)
    print("\n")


async def conversation_examples():
    """Multi-turn conversation examples."""

    print("\n💬 Conversation Examples")
    print("=" * 50)

    # Basic conversation
    print("\n1. Basic conversation:")
    async with conversation() as chat:
        response1 = await chat.ask("Hi, I'm working on a Python project")
        print(f"Assistant: {response1[:100]}...")

        response2 = await chat.ask("Can you help me optimize it?")
        print(f"Assistant: {response2[:100]}...")

        # Check conversation stats
        stats = chat.get_stats()
        print(
            f"Conversation stats: {stats['total_messages']} messages, ~{stats['estimated_tokens']:.0f} tokens"
        )

    # Conversation with specific provider
    print("\n2. Conversation with Claude:")
    async with conversation(provider="anthropic", model="claude-3-sonnet") as chat:
        response = await chat.ask("What makes you different from other AI assistants?")
        print(f"Claude: {response[:150]}...")


def sync_examples():
    """Synchronous examples for simple scripts."""

    print("\n🔄 Synchronous Examples")
    print("=" * 50)

    # 1. Simple sync usage - no async needed!
    print("\n1. Simple sync ask:")
    response = ask_sync("What's 2+2?")
    print(f"Answer: {response}")

    # 2. Quick one-liner
    print("\n2. Quick question:")
    answer = quick_question("What's the capital of France?")
    print(f"Capital: {answer}")

    # 3. Compare providers
    print("\n3. Compare providers:")
    responses = compare_providers("What is AI?", ["openai", "anthropic"])
    for provider, response in responses.items():
        print(f"{provider}: {response[:100]}...")

    # 4. Sync conversation
    print("\n4. Sync conversation:")
    with conversation_sync() as chat:
        chat.ask("Hi!")
        response2 = chat.ask("Tell me a joke")
        print(f"Joke: {response2[:100]}...")


async def advanced_examples():
    """Advanced usage with middleware and monitoring."""

    print("\n⚙️  Advanced Examples")
    print("=" * 50)

    # 1. Configure with middleware
    print("\n1. Configure with monitoring:")
    configure(
        provider="openai",
        model="gpt-4o-mini",
        enable_metrics=True,
        enable_logging=True,
        enable_caching=True,
    )

    # Make some requests
    await ask("Question 1")
    await ask("Question 2")
    await ask("Question 1")  # Should hit cache

    # Check metrics
    metrics = get_metrics()
    print(f"Metrics: {metrics}")

    # 2. Test connections
    print("\n2. Test all providers:")
    test_results = await test_all_providers()
    for provider, result in test_results.items():
        status = "✅" if result.get("success") else "❌"
        duration = result.get("duration", 0)
        print(f"{status} {provider}: {duration:.2f}s")

    # 3. Custom system prompt
    print("\n3. Custom system prompt:")
    response = await ask(
        "How do I use list comprehensions?",
        system_prompt="You are a helpful Python tutor. Always include examples.",
        temperature=0.1,
    )
    print(f"Tutor response: {response[:150]}...")


def production_example():
    """How you might structure this in a production app."""

    print("\n🏭 Production Example")
    print("=" * 50)

    # Configure at app startup
    configure(
        provider="openai",
        model="gpt-4o-mini",  # Fast and cost-effective
        temperature=0.1,  # Consistent responses
        max_tokens=1000,
        enable_metrics=True,
        enable_logging=True,
    )

    # Business logic functions
    async def summarize_document(text: str) -> str:
        return await ask(f"Summarize this document:\n\n{text}")

    async def generate_title(content: str) -> str:
        return await ask(f"Generate a concise title for:\n\n{content}", max_tokens=50)

    async def chat_with_user(user_message: str) -> str:
        # Use better model for user interactions
        return await ask(
            user_message,
            model="gpt-4o",  # Override to better model
            temperature=0.7,  # More creative for chat
        )

    # Example usage
    print("\nProduction functions configured with global settings")
    print("Business logic can just call ask() with overrides as needed")


def debug_and_monitoring():
    """Debug and monitoring capabilities."""

    print("\n🔍 Debug & Monitoring")
    print("=" * 50)

    # Print comprehensive diagnostics
    print("\nDiagnostics:")
    print_diagnostics()


async def main():
    """Run all examples."""

    print("🎯 ChukLLM Modular API Examples")
    print("=" * 70)

    try:
        await basic_examples()
        await conversation_examples()
        sync_examples()
        await advanced_examples()
        production_example()
        debug_and_monitoring()

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("\n💡 Try these patterns in your own code:")
        print("   • Use ask() for simple questions")
        print("   • Use conversation() for multi-turn chats")
        print("   • Use ask_sync() in simple scripts")
        print("   • Use configure() to set global defaults")
        print("   • Use provider shortcuts like ask_openai()")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\n💡 Make sure you have API keys set in your environment:")
        print("   export OPENAI_API_KEY=your_key_here")
        print("   export ANTHROPIC_API_KEY=your_key_here")

        # Show diagnostics on error
        print("\nDiagnostics:")
        print_diagnostics()


if __name__ == "__main__":
    asyncio.run(main())
