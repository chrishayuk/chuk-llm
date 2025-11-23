#!/usr/bin/env python3
"""
llama.cpp Server Usage Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates how to use chuk-llm with llama.cpp server.

This example shows:
1. Starting a llama-server process
2. Using OpenAI-compatible client with llama.cpp
3. Chat completions with streaming
4. Tool/function calling (if supported by model)
5. Proper cleanup

Prerequisites:
- llama.cpp compiled with llama-server binary in PATH
- A GGUF model file (e.g., llama-3.1-8b-instruct.gguf)
"""

import asyncio
from pathlib import Path

from chuk_llm.core import Message, Tool, ToolFunction, ToolParameter
from chuk_llm.core.enums import MessageRole, Provider
from chuk_llm.llm.providers.llamacpp_server import (
    LlamaCppServerConfig,
    LlamaCppServerManager,
)
from chuk_llm.llm.providers.openai_client import OpenAILLMClient


async def example_basic_chat():
    """Example: Basic chat completion with llama.cpp."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Chat Completion")
    print("=" * 60)

    # Configure llama-server
    config = LlamaCppServerConfig(
        model_path=Path("~/models/llama-3.1-8b-instruct-q4_k_m.gguf").expanduser(),
        host="127.0.0.1",
        port=8080,
        ctx_size=8192,
        n_gpu_layers=-1,  # Use all GPU layers (or 0 for CPU-only)
    )

    # Start server (automatically cleaned up on exit)
    async with LlamaCppServerManager(config) as server:
        print(f"✓ llama-server running at {server.base_url}")

        # Create OpenAI-compatible client pointing to llama.cpp server
        client = OpenAILLMClient(
            model="llama-3.1-8b-instruct",  # Model name (for logging)
            api_base=server.base_url,
            api_key=None,  # No API key needed for local server
        )

        # Create messages
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
            Message(role=MessageRole.USER, content="Explain quantum computing in one sentence."),
        ]

        # Get completion
        print("\nSending request...")
        result = await client.create_completion(
            messages=messages,
            stream=False,
            max_tokens=100,
            temperature=0.7,
        )

        print(f"\nResponse: {result['response']}")
        print(f"Tool calls: {result.get('tool_calls', [])}")


async def example_streaming_chat():
    """Example: Streaming chat completion."""
    print("\n" + "=" * 60)
    print("Example 2: Streaming Chat")
    print("=" * 60)

    config = LlamaCppServerConfig(
        model_path=Path("~/models/llama-3.1-8b-instruct-q4_k_m.gguf").expanduser(),
        port=8080,
    )

    async with LlamaCppServerManager(config) as server:
        print(f"✓ Server ready at {server.base_url}")

        client = OpenAILLMClient(
            model="llama-3.1-8b-instruct",
            api_base=server.base_url,
        )

        messages = [
            Message(role=MessageRole.USER, content="Write a haiku about programming."),
        ]

        print("\nStreaming response:")
        print("-" * 40)

        async for chunk in await client.create_completion(
            messages=messages,
            stream=True,
            max_tokens=50,
        ):
            if content := chunk.get("response"):
                print(content, end="", flush=True)

        print("\n" + "-" * 40)


async def example_tool_calling():
    """Example: Tool/function calling with llama.cpp."""
    print("\n" + "=" * 60)
    print("Example 3: Tool Calling")
    print("=" * 60)

    # NOTE: Tool calling requires a model with tool support
    # (e.g., Llama 3.1+, Mistral with function calling, etc.)
    config = LlamaCppServerConfig(
        model_path=Path("~/models/llama-3.1-8b-instruct-q4_k_m.gguf").expanduser(),
        port=8080,
    )

    async with LlamaCppServerManager(config) as server:
        print(f"✓ Server ready at {server.base_url}")

        client = OpenAILLMClient(
            model="llama-3.1-8b-instruct",
            api_base=server.base_url,
        )

        # Define a tool
        get_weather_tool = Tool(
            type="function",
            function=ToolFunction(
                name="get_current_weather",
                description="Get the current weather for a location",
                parameters=ToolParameter(
                    type="object",
                    properties={
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g., San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    required=["location"],
                ),
            ),
        )

        messages = [
            Message(
                role=MessageRole.USER,
                content="What's the weather like in Tokyo?",
            ),
        ]

        print("\nSending request with tool...")
        result = await client.create_completion(
            messages=messages,
            tools=[get_weather_tool],
            stream=False,
        )

        print(f"\nResponse: {result.get('response')}")

        if tool_calls := result.get("tool_calls"):
            print(f"\nTool calls ({len(tool_calls)}):")
            for tc in tool_calls:
                print(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
        else:
            print("\nNo tool calls (model may not support tools)")


async def example_multi_turn_conversation():
    """Example: Multi-turn conversation with context."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Turn Conversation")
    print("=" * 60)

    config = LlamaCppServerConfig(
        model_path=Path("~/models/llama-3.1-8b-instruct-q4_k_m.gguf").expanduser(),
        port=8080,
        ctx_size=8192,  # Ensure enough context for conversation
    )

    async with LlamaCppServerManager(config) as server:
        print(f"✓ Server ready at {server.base_url}")

        client = OpenAILLMClient(
            model="llama-3.1-8b-instruct",
            api_base=server.base_url,
        )

        # Build conversation history
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="What is Python?"),
        ]

        # First turn
        print("\n[User] What is Python?")
        result = await client.create_completion(messages=messages, max_tokens=100)
        assistant_response = result["response"]
        print(f"[Assistant] {assistant_response}")

        # Add assistant response to history
        messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_response))

        # Second turn
        messages.append(
            Message(role=MessageRole.USER, content="What are its main advantages?")
        )
        print("\n[User] What are its main advantages?")

        result = await client.create_completion(messages=messages, max_tokens=100)
        print(f"[Assistant] {result['response']}")


async def example_custom_server_args():
    """Example: Custom llama-server arguments."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Server Configuration")
    print("=" * 60)

    config = LlamaCppServerConfig(
        model_path=Path("~/models/llama-3.1-8b-instruct-q4_k_m.gguf").expanduser(),
        port=8080,
        ctx_size=16384,  # Larger context
        n_gpu_layers=32,  # Only offload 32 layers to GPU
        extra_args=[
            "--n-parallel", "4",  # Process 4 requests in parallel
            "--cont-batching",  # Enable continuous batching
            "--flash-attn",  # Use flash attention if available
        ],
    )

    async with LlamaCppServerManager(config) as server:
        print(f"✓ Server started with custom configuration")
        print(f"  Context size: {config.ctx_size}")
        print(f"  GPU layers: {config.n_gpu_layers}")
        print(f"  Extra args: {' '.join(config.extra_args)}")

        is_healthy = await server.is_healthy()
        print(f"  Health status: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("llama.cpp Server Usage Examples")
    print("=" * 60)
    print("\nNOTE: Update model paths to match your local GGUF files")
    print("=" * 60)

    try:
        # Run examples (comment out any you don't want to run)
        await example_basic_chat()
        await example_streaming_chat()
        await example_tool_calling()
        await example_multi_turn_conversation()
        await example_custom_server_args()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease:")
        print("1. Install llama.cpp and ensure llama-server is in PATH")
        print("2. Download a GGUF model file")
        print("3. Update the model_path in examples to match your model location")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
