#!/usr/bin/env python3
"""
Simple llama.cpp Example - Auto-discovering Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates llama.cpp with automatic model discovery.

This example:
1. Discovers GGUF models in common locations
2. Starts llama-server with the first found model
3. Runs basic chat completions
4. Cleans up automatically

Prerequisites:
- llama-server binary in PATH (brew install llama.cpp or build from source)
- At least one GGUF model file

To download a model:
  huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ~/models
"""

import asyncio
from pathlib import Path

from chuk_llm.core import Message
from chuk_llm.core.enums import MessageRole
from chuk_llm.registry.resolvers.llamacpp_ollama import discover_ollama_models
from chuk_llm.llm.providers.llamacpp_server import (
    LlamaCppServerConfig,
    LlamaCppServerManager,
)
from chuk_llm.llm.providers.openai_client import OpenAILLMClient


def find_gguf_models() -> list[tuple[str, Path, int]]:
    """
    Find GGUF models in common locations, including Ollama models.

    Returns:
        List of (name, path, size) tuples
    """
    models = []

    # First, try Ollama models (most common)
    print("  Checking Ollama models...")
    ollama_models = discover_ollama_models()
    if ollama_models:
        print(f"    → Found {len(ollama_models)} Ollama model(s)")
        for m in ollama_models:
            models.append((m.name, m.gguf_path, m.size_bytes))

    # Then search for standalone GGUF files
    search_paths = [
        Path.home() / "models",
        Path.home() / "llama.cpp" / "models",
        Path.home() / ".cache" / "huggingface",
        Path("/opt/models"),
        Path("/usr/local/models"),
        Path.cwd() / "models",
    ]

    for search_path in search_paths:
        if search_path.exists():
            print(f"  Searching: {search_path}")
            found = list(search_path.rglob("*.gguf"))
            if found:
                print(f"    → Found {len(found)} GGUF file(s)")
                for f in found:
                    models.append((f.name, f, f.stat().st_size))

    # Sort by size and return smallest 10
    models.sort(key=lambda x: x[2])
    return models[:10]


async def main():
    """Run simple llama.cpp example with auto-discovery."""
    print("=" * 70)
    print("llama.cpp Simple Example - Auto-discovering Models")
    print("=" * 70)

    # Find available models
    print("\n🔍 Searching for GGUF models...")
    models = find_gguf_models()

    if not models:
        print("\n❌ No GGUF models found!")
        print("\nTo download a model, run:")
        print("  pip install huggingface-hub")
        print("  mkdir -p ~/models")
        print("  huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \\")
        print("    llama-2-7b-chat.Q4_K_M.gguf --local-dir ~/models")
        return

    print(f"\n✓ Found {len(models)} GGUF model(s):")
    for i, (name, path, size) in enumerate(models, 1):
        size_gb = size / (1024 * 1024 * 1024)
        print(f"  {i}. {name} ({size_gb:.2f} GB)")

    # Use the first (smallest) model
    model_name, model_path, model_size = models[0]
    print(f"\n📦 Using: {model_name}")
    print(f"   Path: {model_path}")
    print(f"   Size: {model_size / (1024**3):.2f} GB")

    # Configure llama-server
    config = LlamaCppServerConfig(
        model_path=model_path,
        port=8080,
        ctx_size=2048,  # Smaller context for faster startup
        n_gpu_layers=-1,  # Use all available GPU layers
    )

    print(f"\n🚀 Starting llama-server on port {config.port}...")

    # Start server and run examples
    try:
        async with LlamaCppServerManager(config) as server:
            print(f"✓ Server ready at {server.base_url}")

            # Create OpenAI-compatible client
            client = OpenAILLMClient(
                model=model_name,
                api_base=server.base_url,
            )

            # Example 1: Simple completion
            print("\n" + "=" * 70)
            print("Example 1: Simple Completion")
            print("=" * 70)

            messages = [
                Message(role=MessageRole.USER, content="Say hello in 5 words or less."),
            ]

            print("\n[User] Say hello in 5 words or less.")
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
                Message(role=MessageRole.USER, content="Count from 1 to 5."),
            ]

            print("\n[User] Count from 1 to 5.")
            print("[Assistant] ", end="", flush=True)

            stream = client.create_completion(
                messages=messages,
                stream=True,
                max_tokens=30,
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
                Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                Message(role=MessageRole.USER, content="What is 2+2?"),
            ]

            print("\n[User] What is 2+2?")
            result = await client.create_completion(
                messages=conversation,
                max_tokens=20,
            )
            print(f"[Assistant] {result['response']}")

            # Add to conversation
            conversation.append(
                Message(role=MessageRole.ASSISTANT, content=result["response"])
            )
            conversation.append(
                Message(role=MessageRole.USER, content="What about 3+3?")
            )

            print("\n[User] What about 3+3?")
            result = await client.create_completion(
                messages=conversation,
                max_tokens=20,
            )
            print(f"[Assistant] {result['response']}")

            print("\n" + "=" * 70)
            print("✓ All examples completed successfully!")
            print("=" * 70)

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
