#!/usr/bin/env python3
"""
llama.cpp with Ollama Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run llama-server using GGUF models downloaded by Ollama.

This example shows how to:
1. Discover Ollama models (GGUF blobs in ~/.ollama/models/blobs/)
2. Start llama-server with an Ollama model
3. Run chat completions

Prerequisites:
- Ollama installed with models downloaded (e.g., `ollama pull llama3.2`)
- llama-server binary in PATH

This avoids re-downloading models if you already use Ollama!
"""

import asyncio

from chuk_llm.core import Message
from chuk_llm.core.enums import MessageRole
from chuk_llm.registry.resolvers.llamacpp_ollama import discover_ollama_models
from chuk_llm.llm.providers.llamacpp_server import (
    LlamaCppServerConfig,
    LlamaCppServerManager,
)
from chuk_llm.llm.providers.openai_client import OpenAILLMClient


async def main():
    """Run llama.cpp with Ollama models."""
    print("=" * 70)
    print("llama.cpp with Ollama Models")
    print("=" * 70)

    # Discover Ollama models
    print("\n🔍 Discovering Ollama models...")
    models = discover_ollama_models()

    if not models:
        print("\n❌ No Ollama models found!")
        print("\nTo download models with Ollama:")
        print("  ollama pull llama3.2")
        print("  ollama pull qwen2.5:3b")
        return

    print(f"\n✓ Found {len(models)} Ollama model(s):\n")
    for i, model in enumerate(models, 1):
        size_gb = model.size_bytes / (1024**3)
        print(f"  {i}. {model.name}")
        print(f"     Size: {size_gb:.2f} GB")
        print(f"     Path: {model.gguf_path}")
        print()

    # Use the smallest model
    model = models[0]
    print(f"📦 Using: {model.name}\n")

    # Configure llama-server with Ollama's GGUF
    config = LlamaCppServerConfig(
        model_path=model.gguf_path,
        port=8081,  # Different from default to avoid conflicts
        ctx_size=4096,
        n_gpu_layers=-1,
    )

    print(f"🚀 Starting llama-server on port {config.port}...")

    try:
        async with LlamaCppServerManager(config) as server:
            print(f"✓ Server ready at {server.base_url}\n")

            # Create OpenAI-compatible client
            client = OpenAILLMClient(
                model=model.name,
                api_base=server.base_url,
            )

            # Example 1: Simple completion
            print("=" * 70)
            print("Example 1: Simple Completion")
            print("=" * 70)

            messages = [
                Message(
                    role=MessageRole.USER,
                    content="What is the capital of France? Answer in 5 words or less.",
                ),
            ]

            print("\n[User] What is the capital of France? Answer in 5 words or less.")
            result = await client.create_completion(
                messages=messages,
                max_tokens=20,
                temperature=0.7,
            )
            print(f"[Assistant] {result['response']}")

            # Example 2: Streaming
            print("\n" + "=" * 70)
            print("Example 2: Streaming Response")
            print("=" * 70)

            messages = [
                Message(
                    role=MessageRole.USER,
                    content="List 3 programming languages.",
                ),
            ]

            print("\n[User] List 3 programming languages.")
            print("[Assistant] ", end="", flush=True)

            stream = client.create_completion(
                messages=messages,
                stream=True,
                max_tokens=50,
            )

            async for chunk in stream:
                if content := chunk.get("response"):
                    print(content, end="", flush=True)

            print("\n")

            # Example 3: Multi-turn conversation
            print("=" * 70)
            print("Example 3: Multi-turn Conversation")
            print("=" * 70)

            conversation = [
                Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                Message(role=MessageRole.USER, content="Hi! What's 2+2?"),
            ]

            print("\n[User] Hi! What's 2+2?")
            result = await client.create_completion(
                messages=conversation,
                max_tokens=30,
            )
            print(f"[Assistant] {result['response']}")

            # Continue conversation
            conversation.append(
                Message(role=MessageRole.ASSISTANT, content=result["response"])
            )
            conversation.append(
                Message(role=MessageRole.USER, content="And what about 3+3?")
            )

            print("\n[User] And what about 3+3?")
            result = await client.create_completion(
                messages=conversation,
                max_tokens=30,
            )
            print(f"[Assistant] {result['response']}")

            print("\n" + "=" * 70)
            print("✓ All examples completed successfully!")
            print("=" * 70)
            print("\n💡 Tip: You reused Ollama's downloaded models - no re-download needed!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure llama-server is installed:")
        print("  - macOS: brew install llama.cpp")
        print("  - Linux: Build from https://github.com/ggerganov/llama.cpp")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
