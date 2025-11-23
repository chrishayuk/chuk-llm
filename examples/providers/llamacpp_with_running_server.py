#!/usr/bin/env python3
"""
llama.cpp with Running Server Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates using chuk-llm with an already-running llama-server.

This example:
1. Connects to a running llama-server instance
2. Discovers available models via API
3. Runs chat completions
4. Shows streaming

Prerequisites:
- llama-server already running (e.g., `llama-server -m model.gguf`)

To start llama-server manually:
  llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
"""

import asyncio

import httpx

from chuk_llm.core import Message
from chuk_llm.core.enums import MessageRole
from chuk_llm.llm.providers.openai_client import OpenAILLMClient


async def discover_models(api_base: str = "http://localhost:8080") -> list[dict]:
    """Discover models via llama.cpp API."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{api_base}/v1/models")
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except (httpx.ConnectError, httpx.HTTPError) as e:
        print(f"❌ Could not connect to llama-server: {e}")
        return []


async def main():
    """Run examples with running llama-server."""
    print("=" * 70)
    print("llama.cpp with Running Server Example")
    print("=" * 70)

    api_base = "http://localhost:8033"

    # Discover models
    print(f"\n🔍 Discovering models from {api_base}...")
    models = await discover_models(api_base)

    if not models:
        print("\n❌ No models found. Is llama-server running?")
        print("\nTo start llama-server:")
        print("  llama-server -m /path/to/model.gguf --port 8080")
        return

    print(f"\n✓ Found {len(models)} model(s):")
    for model in models:
        model_id = model.get("id", "unknown")
        meta = model.get("meta", {})
        size_gb = meta.get("size", 0) / (1024**3)
        vocab_size = meta.get("n_vocab", "unknown")
        print(f"\n  Model: {model_id}")
        print(f"    Size: {size_gb:.2f} GB")
        print(f"    Vocab: {vocab_size:,}")
        print(f"    Context: {meta.get('n_ctx_train', 'unknown'):,}")

    # Use first model
    model_id = models[0]["id"]
    print(f"\n📦 Using model: {model_id}")

    # Create OpenAI-compatible client
    client = OpenAILLMClient(
        model=model_id,
        api_base=api_base,
    )

    # Example 1: Simple completion
    print("\n" + "=" * 70)
    print("Example 1: Simple Completion")
    print("=" * 70)

    messages = [
        Message(role=MessageRole.USER, content="Say 'Hello from llama.cpp!' and nothing else."),
    ]

    print("\n[User] Say 'Hello from llama.cpp!' and nothing else.")
    result = await client.create_completion(
        messages=messages,
        max_tokens=50,
        temperature=0.7,
    )
    print(f"[Assistant] {result['response']}")

    # Example 2: Streaming
    print("\n" + "=" * 70)
    print("Example 2: Streaming Response")
    print("=" * 70)

    messages = [
        Message(role=MessageRole.USER, content="Count from 1 to 5, one number per line."),
    ]

    print("\n[User] Count from 1 to 5, one number per line.")
    print("[Assistant] ", end="", flush=True)

    stream = client.create_completion(
        messages=messages,
        stream=True,
        max_tokens=50,
    )

    async for chunk in stream:
        if content := chunk.get("response"):
            print(content, end="", flush=True)

    print()  # Newline after streaming

    # Example 3: Multi-turn conversation
    print("\n" + "=" * 70)
    print("Example 3: Multi-turn Conversation")
    print("=" * 70)

    conversation = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful math tutor."),
        Message(role=MessageRole.USER, content="What is 7 times 8?"),
    ]

    print("\n[User] What is 7 times 8?")
    result = await client.create_completion(
        messages=conversation,
        max_tokens=30,
    )
    print(f"[Assistant] {result['response']}")

    # Add to conversation
    conversation.append(Message(role=MessageRole.ASSISTANT, content=result["response"]))
    conversation.append(Message(role=MessageRole.USER, content="And what is that divided by 2?"))

    print("\n[User] And what is that divided by 2?")
    result = await client.create_completion(
        messages=conversation,
        max_tokens=30,
    )
    print(f"[Assistant] {result['response']}")

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
