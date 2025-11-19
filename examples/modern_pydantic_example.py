#!/usr/bin/env python3
"""
Modern Pydantic API Example
============================

Demonstrates the new modern Pydantic-based API with 100% type safety.

New Features:
- ✅ Type-safe Pydantic V2 models
- ✅ Zero magic strings (all enums)
- ✅ Fast JSON with orjson
- ✅ Clean, modern DX

Requirements:
- Set OPENAI_API_KEY or other provider API key

Usage:
    python modern_pydantic_example.py
    python modern_pydantic_example.py --provider anthropic
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    # Import modern Pydantic clients from internal path
    import sys
    from pathlib import Path

    # Add src to path for development
    src_path = Path(__file__).parent.parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    # Modern Pydantic clients
    from chuk_llm.clients.openai import OpenAIClient
    from chuk_llm.clients.anthropic import AnthropicClient
    from chuk_llm.clients.gemini import GeminiClient

    # Modern Pydantic models (no magic strings!)
    from chuk_llm.core.models import (
        CompletionRequest,
        CompletionResponse,
        Message,
        MessageRole,
        ContentPart,
        ContentType,
        ToolCall,
        ToolType,
        FunctionCall,
        Tool,
        ToolFunction,
    )
    from chuk_llm.core.enums import FinishReason, ErrorType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install -e .")
    import traceback
    traceback.print_exc()
    sys.exit(1)


async def example_basic_completion():
    """Example 1: Basic completion with type-safe Pydantic models."""
    print("\n" + "="*60)
    print("Example 1: Basic Completion (Type-Safe)")
    print("="*60)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping")
        return

    # Create modern client
    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    # Create request using Pydantic models (NO MAGIC STRINGS!)
    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.SYSTEM,  # Enum, not "system"!
                content="You are a helpful AI assistant.",
            ),
            Message(
                role=MessageRole.USER,  # Enum, not "user"!
                content="Explain quantum computing in one sentence.",
            ),
        ],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100,
    )

    # Type-safe completion
    response: CompletionResponse = await client.complete(request)

    # Type-safe response access
    print(f"✅ Response: {response.content}")
    print(f"   Finish reason: {response.finish_reason.value}")  # Enum!
    print(f"   Tool calls: {len(response.tool_calls)}")

    await client.close()


async def example_streaming():
    """Example 2: Streaming with type-safe models."""
    print("\n" + "="*60)
    print("Example 2: Streaming (Type-Safe)")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping")
        return

    client = OpenAIClient(model="gpt-4o-mini", api_key=api_key)

    # Type-safe request
    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Write a haiku about modern code.",
            )
        ],
        model="gpt-4o-mini",
        temperature=0.8,
    )

    print("🌊 Streaming response:")
    print("   ", end="", flush=True)

    # Type-safe streaming
    async for chunk in client.stream(request):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n✅ Streaming complete")

    await client.close()


async def example_tool_calling():
    """Example 3: Tool calling with type-safe models."""
    print("\n" + "="*60)
    print("Example 3: Tool Calling (Type-Safe)")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping")
        return

    client = OpenAIClient(model="gpt-4o-mini", api_key=api_key)

    # Define tools using Pydantic models (NO DICTS!)
    tools = [
        Tool(
            type=ToolType.FUNCTION,  # Enum!
            function=Function(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            ),
        )
    ]

    # Type-safe request
    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="What's the weather in Paris?",
            )
        ],
        model="gpt-4o-mini",
        tools=tools,
        temperature=0.0,
    )

    # Get response
    response = await client.complete(request)

    if response.tool_calls:
        print(f"✅ Tool calls requested: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            # Type-safe access to tool call fields
            print(f"   Function: {tc.function.name}")
            print(f"   Arguments: {tc.function.arguments}")
            print(f"   Type: {tc.type.value}")  # Enum!
    else:
        print(f"ℹ️  No tool calls: {response.content}")

    await client.close()


async def example_multimodal():
    """Example 4: Multi-modal (vision) with type-safe models."""
    print("\n" + "="*60)
    print("Example 4: Multi-modal/Vision (Type-Safe)")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping")
        return

    client = OpenAIClient(model="gpt-4o", api_key=api_key)

    # Multi-modal message with typed content parts (NO DICTS!)
    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content=[
                    ContentPart(
                        type=ContentType.TEXT,  # Enum!
                        text="What's in this image?",
                    ),
                    ContentPart(
                        type=ContentType.IMAGE_URL,  # Enum!
                        image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/240px-Cat03.jpg",
                    ),
                ],
            )
        ],
        model="gpt-4o",
        max_tokens=100,
    )

    response = await client.complete(request)

    print(f"✅ Vision response: {response.content}")

    await client.close()


async def example_multiple_providers():
    """Example 5: Multiple providers with same type-safe API."""
    print("\n" + "="*60)
    print("Example 5: Multiple Providers (Same Type-Safe API)")
    print("="*60)

    providers = [
        ("OpenAI", OpenAIClient, "gpt-4o-mini", "OPENAI_API_KEY"),
        ("Anthropic", AnthropicClient, "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
        ("Gemini", GeminiClient, "gemini-2.0-flash-exp", "GEMINI_API_KEY"),
    ]

    # Same request for all providers (type-safe!)
    base_request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Say 'Hello from {provider}' in exactly 5 words",
            )
        ],
        temperature=0.0,
    )

    for name, ClientClass, model, env_var in providers:
        api_key = os.getenv(env_var)
        if not api_key:
            print(f"⚠️  {env_var} not set, skipping {name}")
            continue

        # Create client
        client = ClientClass(model=model, api_key=api_key)

        # Update model in request
        request = CompletionRequest(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=f"Say 'Hello from {name}' in exactly 5 words",
                )
            ],
            model=model,
            temperature=0.0,
        )

        # Same API for all providers!
        response = await client.complete(request)

        print(f"✅ {name}: {response.content}")

        await client.close()


async def example_error_handling():
    """Example 6: Type-safe error handling."""
    print("\n" + "="*60)
    print("Example 6: Type-Safe Error Handling")
    print("="*60)

    from chuk_llm.core import LLMError, ErrorType

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping")
        return

    client = OpenAIClient(model="gpt-4o-mini", api_key=api_key)

    # Invalid request (negative max_tokens)
    try:
        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="Hello")
            ],
            model="gpt-4o-mini",
            max_tokens=-1,  # Invalid!
        )

        response = await client.complete(request)
        print(f"Response: {response.content}")

    except ValueError as e:
        print(f"✅ Validation error caught: {e}")

    # Test with invalid API key
    bad_client = OpenAIClient(model="gpt-4o-mini", api_key="sk-invalid")

    try:
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="gpt-4o-mini",
        )

        response = await bad_client.complete(request)

    except LLMError as e:
        # Type-safe error handling
        print(f"✅ LLM error caught:")
        print(f"   Error type: {e.error_type}")  # Enum!
        print(f"   Message: {e.error_message}")

    await client.close()
    await bad_client.close()


async def main():
    """Run all examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Modern Pydantic API Examples")
    parser.add_argument("--example", type=int, help="Run specific example (1-6)")

    args = parser.parse_args()

    print("🚀 Modern Pydantic API Examples")
    print("="*60)
    print("Demonstrating:")
    print("  ✅ Type-safe Pydantic V2 models")
    print("  ✅ Zero magic strings (all enums)")
    print("  ✅ Fast JSON with orjson")
    print("  ✅ Clean, modern DX")
    print("="*60)

    examples = [
        example_basic_completion,
        example_streaming,
        example_tool_calling,
        example_multimodal,
        example_multiple_providers,
        example_error_handling,
    ]

    if args.example:
        if 1 <= args.example <= len(examples):
            await examples[args.example - 1]()
        else:
            print(f"❌ Invalid example number. Choose 1-{len(examples)}")
            sys.exit(1)
    else:
        # Run all examples
        for example_func in examples:
            try:
                await example_func()
            except Exception as e:
                print(f"❌ Error in {example_func.__name__}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 Modern Pydantic API Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
