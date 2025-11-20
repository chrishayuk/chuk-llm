#!/usr/bin/env python3
"""
OpenAI-Compatible API Example
==============================

Example of using any OpenAI-compatible API endpoint with chuk-llm.
Works with vLLM, LM Studio, Ollama (with OpenAI adapter), LocalAI, etc.

Requirements:
- Set OPENAI_API_KEY (or your custom API key)
- Set OPENAI_API_BASE (your custom endpoint)

Usage:
    # For vLLM
    export OPENAI_API_BASE="http://localhost:8000/v1"
    export OPENAI_API_KEY="EMPTY"
    python openai_compatible_example.py

    # For Ollama with OpenAI compatibility
    export OPENAI_API_BASE="http://localhost:11434/v1"
    export OPENAI_API_KEY="ollama"
    python openai_compatible_example.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Default to local vLLM setup
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
    from chuk_llm.core.models import Message
    from chuk_llm.core.enums import MessageRole
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


async def main():
    """Example using OpenAI-compatible API"""
    print("\n🌐 OpenAI-Compatible API Example")
    print("=" * 50)
    print(f"   Endpoint: {api_base}")
    print(f"   Model: {model}")
    print("=" * 50)

    # Create client with custom endpoint
    client = OpenAILLMClient(
        model=model,
        api_key=api_key,
        api_base=api_base
    )

    # Optional: Suppress warnings for custom endpoints
    client.detected_provider = "openai"
    client.provider_name = "openai"

    # Test 1: Simple completion
    print("\n1️⃣ Simple Completion:")
    try:
        messages = [
            Message(role=MessageRole.USER, content="What is 2+2?")
        ]

        response = await client.create_completion(messages, max_tokens=100)
        print(f"   {response['response']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Streaming
    print("\n2️⃣ Streaming:")
    try:
        messages = [
            Message(role=MessageRole.USER, content="Count from 1 to 3")
        ]

        print("   ", end="", flush=True)
        async for chunk in client.create_completion(messages, stream=True, max_tokens=50):
            if chunk.get("response"):
                print(chunk["response"], end="", flush=True)
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Conversation
    print("\n3️⃣ Conversation:")
    try:
        conversation = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello!"),
        ]

        response = await client.create_completion(conversation, max_tokens=100)
        print(f"   User: Hello!")
        print(f"   AI: {response['response']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n✅ Done!")
    print("\nℹ️  Tips:")
    print("   - Make sure your API endpoint is running")
    print("   - Check that the model name matches your server")
    print("   - Adjust API key if your server requires authentication")


if __name__ == "__main__":
    asyncio.run(main())
