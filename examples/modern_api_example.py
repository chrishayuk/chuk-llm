"""
Modern API Example
==================

Demonstrates the new type-safe API using Pydantic models.
"""

import asyncio
import os

from chuk_llm.api.modern import modern_ask, modern_stream
from chuk_llm.core import Message, MessageRole


async def main():
    """Demo the modern type-safe API."""

    print("=" * 60)
    print("Modern Type-Safe API Demo")
    print("=" * 60)

    # Example 1: Simple ask with Pydantic response
    print("\n1. Simple Ask (returns Pydantic CompletionResponse)")
    print("-" * 60)

    response = await modern_ask(
        prompt="What is 2+2?",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Type: {type(response)}")
    print(f"Content: {response.content}")
    print(f"Finish Reason: {response.finish_reason}")
    print(f"Usage: {response.usage}")

    # Example 2: Streaming with type safety
    print("\n2. Streaming (yields string chunks)")
    print("-" * 60)
    print("Response: ", end="", flush=True)

    async for chunk in modern_stream(
        prompt="Count from 1 to 5",
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=50,
    ):
        print(chunk, end="", flush=True)

    print("\n")

    # Example 3: With system message
    print("\n3. With System Message")
    print("-" * 60)

    response = await modern_ask(
        prompt="What's your name?",
        system="You are a helpful pirate assistant. Always respond like a pirate.",
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=50,
    )

    print(f"Response: {response.content}")

    print("\n" + "=" * 60)
    print("All examples use Pydantic models internally!")
    print("- Zero magic strings")
    print("- Full type safety")
    print("- IDE autocomplete")
    print("- Fast JSON with orjson/ujson")
    print("=" * 60)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable to run this example")
        exit(1)

    asyncio.run(main())
