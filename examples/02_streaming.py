#!/usr/bin/env python3
"""
Streaming Responses
===================

Demonstrates real-time streaming responses for better UX.
Great for chatbots and interactive applications.
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import stream, stream_sync

# Method 1: Async streaming (recommended)
async def async_streaming():
    print("=== Async Streaming ===")
    print("Response: ", end="", flush=True)

    async for chunk in stream("Write a haiku about coding"):
        print(chunk, end="", flush=True)

    print("\n")

# Method 2: Async streaming with specific provider
async def provider_streaming():
    print("=== Provider-Specific Streaming ===")
    print("Groq (ultra-fast): ", end="", flush=True)

    async for chunk in stream(
        "Count from 1 to 5",
        provider="groq",
        model="llama-3.3-70b-versatile"
    ):
        print(chunk, end="", flush=True)

    print("\n")

# Method 3: Sync streaming
def sync_streaming():
    print("=== Sync Streaming ===")
    print("Response: ", end="", flush=True)

    # sync streaming returns a list of chunks
    chunks = stream_sync("Explain Python in one sentence")
    for chunk in chunks:
        print(chunk, end="", flush=True)

    print("\n")

# Method 4: Streaming with parameters
async def streaming_with_params():
    print("=== Streaming with Parameters ===")
    print("Response: ", end="", flush=True)

    async for chunk in stream(
        "Tell me about quantum computing",
        temperature=0.7,
        max_tokens=100
    ):
        print(chunk, end="", flush=True)

    print("\n")

# Method 5: Using convenience stream functions
async def convenience_streaming():
    print("=== Convenience Streaming ===")

    # Auto-generated streaming functions
    from chuk_llm import stream_openai, stream_anthropic

    print("OpenAI: ", end="", flush=True)
    async for chunk in stream_openai("What is AI?"):
        print(chunk, end="", flush=True)

    print("\n")

if __name__ == "__main__":
    # Run examples
    asyncio.run(async_streaming())
    asyncio.run(provider_streaming())
    sync_streaming()
    asyncio.run(streaming_with_params())
    asyncio.run(convenience_streaming())
