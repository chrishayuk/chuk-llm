#!/usr/bin/env python3
"""
Basic Ask - Simple Synchronous Usage
=====================================

Demonstrates the simplest synchronous usage patterns.
Perfect for scripts and simple applications.
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import ask, ask_sync

# Method 1: Async (recommended for performance)
async def async_example():
    print("=== Async Example ===")
    answer = await ask("What is Python? Answer in one sentence.")
    print(f"Answer: {answer}\n")

# Method 2: Sync (convenience wrapper)
def sync_example():
    print("=== Sync Example ===")
    answer = ask_sync("What is JavaScript? Answer in one sentence.")
    print(f"Answer: {answer}\n")

# Method 3: Specify provider and model
async def specific_provider_example():
    print("=== Specific Provider Example ===")

    # Use OpenAI specifically
    answer = await ask(
        "What is machine learning?",
        provider="openai",
        model="gpt-4o-mini"
    )
    print(f"OpenAI: {answer}\n")

# Method 4: Use auto-generated provider functions
def convenience_functions():
    print("=== Convenience Functions ===")

    # These are auto-generated based on available providers
    from chuk_llm import ask_openai_sync, ask_anthropic_sync

    # Each provider has convenience functions
    answer = ask_openai_sync("Tell me a programming joke")
    print(f"OpenAI: {answer}\n")

if __name__ == "__main__":
    # Run async examples
    asyncio.run(async_example())
    asyncio.run(specific_provider_example())

    # Run sync examples
    sync_example()
    convenience_functions()
