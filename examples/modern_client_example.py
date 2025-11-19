"""
Modern Client Example
=====================

Demonstrates the new type-safe, async-native client system.
"""

import asyncio
import os

from chuk_llm.clients import OpenAIClient
from chuk_llm.core import (
    CompletionRequest,
    Message,
    MessageRole,
    Tool,
    ToolFunction,
    get_json_library,
    get_performance_info,
)


async def simple_completion():
    """Simple completion example."""
    print("=== Simple Completion ===\n")

    # Create client with connection pooling
    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_connections=10,  # Connection pool size
    )

    try:
        # Create type-safe request
        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                Message(role=MessageRole.USER, content="What is Python?"),
            ],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100,
        )

        # Get response (fully validated)
        response = await client.complete(request)

        print(f"Response: {response.content}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Tokens: {response.usage}")
        print()

    finally:
        await client.close()


async def streaming_example():
    """Zero-copy streaming example."""
    print("=== Streaming Example ===\n")

    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    try:
        request = CompletionRequest(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="Write a haiku about programming",
                )
            ],
            model="gpt-4o-mini",
        )

        # Stream with zero-copy (no accumulation until needed)
        print("Streaming response: ", end="", flush=True)

        async for chunk in client.stream(request):
            if chunk.content:
                print(chunk.content, end="", flush=True)

            if chunk.finish_reason:
                print(f"\n\nFinish reason: {chunk.finish_reason}")

        print()

    finally:
        await client.close()


async def tool_calling_example():
    """Tool calling example with type-safe tools."""
    print("=== Tool Calling Example ===\n")

    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    try:
        # Define tools with type-safe Pydantic models
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_weather",
                    description="Get the current weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            )
        ]

        request = CompletionRequest(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="What's the weather in San Francisco?",
                )
            ],
            model="gpt-4o-mini",
            tools=tools,
        )

        response = await client.complete(request)

        if response.tool_calls:
            print(f"Tool calls: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                print(f"  - {tc.function.name}: {tc.function.arguments}")
        else:
            print(f"Response: {response.content}")

        print()

    finally:
        await client.close()


async def performance_info():
    """Show performance optimizations."""
    print("=== Performance Info ===\n")

    perf_info = get_performance_info()
    print(f"JSON library: {perf_info['library']}")
    print(f"Speedup: {perf_info['speedup']}")
    print(f"orjson available: {perf_info['orjson_available']}")
    print(f"ujson available: {perf_info['ujson_available']}")
    print()


async def main():
    """Run all examples."""
    # Show performance info
    await performance_info()

    # Run examples
    await simple_completion()
    await streaming_example()
    await tool_calling_example()


if __name__ == "__main__":
    asyncio.run(main())
