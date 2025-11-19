#!/usr/bin/env python3
"""
OpenAI Chat Completions API - Comprehensive Example
====================================================

Complete demonstration of the traditional Chat Completions API (/v1/chat/completions).
This is the standard OpenAI-compatible API used by most providers.

Features Demonstrated:
- ✅ Basic completion
- ✅ Streaming
- ✅ Function/tool calling
- ✅ Vision (multimodal)
- ✅ JSON mode
- ✅ GPT-5, GPT-5-mini support
- ✅ Model comparison
- ✅ Parameters testing
- ✅ Error handling
- ✅ Multi-turn conversations (manual history)
- ✅ Structured outputs
- ✅ Zero magic strings (all enums)
- ✅ Type-safe Pydantic V2

Requirements:
- Set OPENAI_API_KEY environment variable

Usage:
    python openai_chat_completions_example.py
    python openai_chat_completions_example.py --model gpt-5
    python openai_chat_completions_example.py --demo 1  # Run specific demo
    python openai_chat_completions_example.py --quick  # Skip slow demos
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add src to path for development
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from chuk_llm.clients.openai import OpenAIClient
    from chuk_llm.core.models import (
        CompletionRequest,
        CompletionResponse,
        Message,
        Tool,
        ToolFunction,
        TextContent,
        ImageUrlContent,
    )
    from chuk_llm.core.enums import MessageRole, ContentType, ToolType
    from chuk_llm.core import LLMError
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


async def demo_basic_completion(model: str):
    """Demo 1: Basic completion with Chat Completions API."""
    print(f"\n{'='*60}")
    print(f"Demo 1: Basic Completion - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Endpoint: POST /v1/chat/completions")
    print(f"Note: Manual conversation history management")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Standard chat completions request
    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful AI assistant.",
            ),
            Message(
                role=MessageRole.USER,
                content="Explain the Chat Completions API in one sentence.",
            ),
        ],
        model=model,
        temperature=0.7,
        max_tokens=150,
    )

    response: CompletionResponse = await client.complete(request)

    print(f"\n✅ Response: {response.content}")
    print(f"   Finish: {response.finish_reason.value}")
    print(f"   Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")

    await client.close()


async def demo_streaming(model: str):
    """Demo 2: Streaming with Chat Completions API."""
    print(f"\n{'='*60}")
    print(f"Demo 2: Streaming - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Parameter: stream=true")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Write a haiku about traditional APIs.",
            )
        ],
        model=model,
    )

    print("\n🌊 Streaming:")
    print("   ", end="", flush=True)

    async for chunk in client.stream(request):
        # Stream chunks contain 'content' field with incremental text
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n✅ Complete")

    await client.close()


async def demo_function_calling(model: str):
    """Demo 3: Function calling with Chat Completions API."""
    print(f"\n{'='*60}")
    print(f"Demo 3: Function/Tool Calling - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Feature: Custom function definitions")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Define tools using Pydantic models (NO DICTS!)
    tools = [
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="get_weather",
                description="Get current weather for a city",
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
                            "description": "Temperature units",
                        },
                    },
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="search_web",
                description="Search the web for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]

    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="What's the weather in Paris and search for Eiffel Tower history?",
            )
        ],
        model=model,
        tools=tools,
        temperature=0.0,
    )

    response = await client.complete(request)

    if response.tool_calls:
        print(f"✅ Tool calls: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"   📞 {tc.function.name}({tc.function.arguments})")
            print(f"      Type: {tc.type.value}")
    else:
        print(f"ℹ️  No tools called: {response.content}")

    await client.close()


async def demo_vision(model: str = "gpt-4o"):
    """Demo 4: Vision/multimodal with Chat Completions API."""
    print(f"\n{'='*60}")
    print(f"Demo 4: Vision/Multi-modal - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Feature: Image understanding")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Use a base64 encoded simple image (more reliable than external URLs)
    # This is a 1x1 red pixel PNG - replace with actual base64 image in production
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    image_data_url = f"data:image/png;base64,{base64_image}"

    try:
        # Multi-modal message using typed content parts
        request = CompletionRequest(
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[
                        TextContent(
                            type=ContentType.TEXT,
                            text="What color is this image? (One word)",
                        ),
                        ImageUrlContent(
                            type=ContentType.IMAGE_URL,
                            image_url={"url": image_data_url},
                        ),
                    ],
                )
            ],
            model=model,
            max_tokens=100,
        )

        response = await client.complete(request)

        print(f"✅ Vision: {response.content}")

    except (Exception, LLMError) as e:
        print(f"⚠️  Vision demo skipped: {str(e)[:100]}")
        print(f"   Note: Vision requires a valid image URL that OpenAI can access")
    finally:
        await client.close()


async def demo_json_mode(model: str):
    """Demo 5: JSON mode with Chat Completions API."""
    print(f"\n{'='*60}")
    print(f"Demo 5: JSON Mode - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Parameter: response_format={{\"type\": \"json_object\"}}")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant that outputs JSON.",
            ),
            Message(
                role=MessageRole.USER,
                content='Generate a JSON object with "name": "Alice" and "age": 30',
            ),
        ],
        model=model,
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    response = await client.complete(request)

    print(f"✅ JSON: {response.content}")

    await client.close()


async def demo_model_discovery():
    """Demo 6: Model Discovery - Discover available OpenAI models."""
    print(f"\n{'='*60}")
    print(f"Demo 6: Model Discovery - Chat Completions API")
    print(f"{'='*60}")
    print(f"Using OpenAI discoverer to fetch available models")

    try:
        from chuk_llm.llm.discovery.openai_discoverer import OpenAIModelDiscoverer

        discoverer = OpenAIModelDiscoverer(
            provider_name="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        print("\n🔍 Discovering models from OpenAI API...")
        models_data = await discoverer.discover_models()

        print(f"\n📊 Found {len(models_data)} models")

        # Group by family
        families = {}
        for model in models_data:
            family = model.get("provider_specific", {}).get("model_family", "unknown")
            if family not in families:
                families[family] = []
            families[family].append(model.get("name"))

        for family, models in sorted(families.items()):
            print(f"\n📦 {family.upper()} family:")
            for model_name in sorted(models)[:5]:  # Show first 5 of each family
                # Check if it's a reasoning model
                provider_spec = next(
                    (
                        m.get("provider_specific", {})
                        for m in models_data
                        if m.get("name") == model_name
                    ),
                    {},
                )
                if provider_spec.get("reasoning_capable"):
                    print(f"   • {model_name} [🧠 reasoning]")
                elif provider_spec.get("supports_vision"):
                    print(f"   • {model_name} [👁️ vision]")
                else:
                    print(f"   • {model_name}")

            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")

        # Test the first available model
        if models_data:
            test_model = models_data[0].get("name")
            print(f"\n🧪 Testing model: {test_model}")
            try:
                client = OpenAIClient(
                    model=test_model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
                request = CompletionRequest(
                    messages=[
                        Message(role=MessageRole.USER, content="Say hello in one word")
                    ],
                    model=test_model,
                    max_tokens=10,
                )
                response = await client.complete(request)
                print(f"   ✅ Model works: {response.content}")
                await client.close()
            except Exception as e:
                print(f"   ⚠️ Test failed: {e}")

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback

        traceback.print_exc()


async def demo_gpt5_models():
    """Demo 7: GPT-5 and GPT-5-mini models."""
    print(f"\n{'='*60}")
    print(f"Demo 7: GPT-5 Models - Chat Completions API")
    print(f"{'='*60}")
    print(f"Testing: gpt-5, gpt-5-mini")

    models = ["gpt-5", "gpt-5-mini"]
    prompt = "What is the Chat Completions API? (One sentence)"

    for model_name in models:
        try:
            client = OpenAIClient(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            request = CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=model_name,
                temperature=0.0,
            )

            start = time.time()
            response = await client.complete(request)
            duration = time.time() - start

            print(f"\n✅ {model_name} ({duration:.2f}s):")
            print(f"   {response.content[:100]}...")

            await client.close()

        except Exception as e:
            print(f"\n❌ {model_name}: {e}")


async def demo_model_comparison():
    """Demo 8: Compare multiple models."""
    print(f"\n{'='*60}")
    print(f"Demo 8: Model Comparison - Chat Completions API")
    print(f"{'='*60}")

    models = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    prompt = "What is machine learning? (One sentence)"

    for model_name in models:
        try:
            client = OpenAIClient(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            request = CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=model_name,
                temperature=0.0,
            )

            start = time.time()
            response = await client.complete(request)
            duration = time.time() - start

            print(f"\n✅ {model_name} ({duration:.2f}s):")
            print(f"   {response.content[:80]}...")

            await client.close()

        except Exception as e:
            print(f"\n❌ {model_name}: {e}")


async def demo_parameters(model: str):
    """Demo 8: Test various parameters."""
    print(f"\n{'='*60}")
    print(f"Demo 8: Parameters Testing - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Test temperature variations
    temperatures = [0.0, 0.5, 1.0]
    prompt = "Say hello"

    for temp in temperatures:
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content=prompt)],
            model=model,
            temperature=temp,
            max_tokens=20,
        )

        response = await client.complete(request)
        print(f"Temperature {temp}: {response.content}")

    await client.close()


async def demo_error_handling(model: str):
    """Demo 10: Error handling."""
    print(f"\n{'='*60}")
    print(f"Demo 10: Error Handling - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")

    # Test with invalid API key
    bad_client = OpenAIClient(model=model, api_key="sk-invalid")

    try:
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model=model,
        )

        await bad_client.complete(request)

    except LLMError as e:
        print(f"✅ LLM error caught:")
        print(f"   Type: {e.error_type}")
        print(f"   Message: {e.error_message}")

    await bad_client.close()


async def demo_multi_turn(model: str):
    """Demo 10: Multi-turn conversation (manual history)."""
    print(f"\n{'='*60}")
    print(f"Demo 10: Multi-Turn Conversation - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Note: YOU manage conversation history manually")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Manual conversation history management
    conversation = [
        Message(
            role=MessageRole.USER,
            content="My name is Alice.",
        )
    ]

    # Turn 1
    print("\n👤 User: My name is Alice.")
    request = CompletionRequest(messages=conversation, model=model, temperature=0.7)
    response = await client.complete(request)
    print(f"🤖 Assistant: {response.content}")

    # Add assistant response to history
    conversation.append(
        Message(
            role=MessageRole.ASSISTANT,
            content=response.content or "",
        )
    )

    # Turn 2
    conversation.append(
        Message(
            role=MessageRole.USER,
            content="What's my name?",
        )
    )

    print("\n👤 User: What's my name?")
    request = CompletionRequest(messages=conversation, model=model, temperature=0.7)
    response = await client.complete(request)
    print(f"🤖 Assistant: {response.content}")

    print(
        "\n✅ You manually appended messages to maintain conversation history"
    )

    await client.close()


async def demo_structured_outputs(model: str):
    """Demo 12: Structured outputs with JSON mode."""
    print(f"\n{'='*60}")
    print(f"Demo 12: Structured Outputs - Chat Completions API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Note: Uses JSON mode with schema in prompt")

    client = OpenAIClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    schema_instruction = """
Generate a JSON object matching this schema:
{
  "person": {
    "name": "string",
    "age": "number",
    "email": "string"
  }
}
"""

    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant that outputs valid JSON.",
            ),
            Message(
                role=MessageRole.USER,
                content=f"{schema_instruction}\nCreate a person object for Bob, age 25, email bob@example.com",
            ),
        ],
        model=model,
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    response = await client.complete(request)

    print(f"✅ Structured output: {response.content}")

    await client.close()


async def demo_reasoning_models():
    """Demo 12: Reasoning models with token display."""
    print(f"\n{'='*60}")
    print(f"Demo 12: Reasoning Models - Chat Completions API")
    print(f"{'='*60}")
    print(f"Testing: o1, gpt-5-mini (reasoning models)")

    models = ["o1", "gpt-5-mini"]
    prompt = "I have a 3-gallon jug and a 5-gallon jug. How can I measure exactly 4 gallons of water?"

    for model_name in models:
        try:
            client = OpenAIClient(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            print(f"\n🧠 Testing {model_name}...")
            request = CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=model_name,
            )

            start = time.time()
            response = await client.complete(request)
            duration = time.time() - start

            print(f"\n✅ {model_name} ({duration:.2f}s):")
            print(f"   {response.content[:200]}...")

            # Display token usage with reasoning tokens
            if response.usage:
                print(f"\n📊 Token Usage:")
                print(f"   Prompt: {response.usage.prompt_tokens} tokens")
                print(f"   Completion: {response.usage.completion_tokens} tokens")
                if response.usage.reasoning_tokens:
                    print(f"   🧠 Reasoning: {response.usage.reasoning_tokens} tokens")
                    output_tokens = response.usage.completion_tokens - response.usage.reasoning_tokens
                    print(f"   📝 Output: {output_tokens} tokens")
                print(f"   Total: {response.usage.total_tokens} tokens")

            await client.close()

        except Exception as e:
            print(f"\n❌ {model_name}: {e}")


async def main():
    """Run all demos."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--demo", type=int, help="Run specific demo (1-13)")
    parser.add_argument("--quick", action="store_true", help="Skip slow demos")

    args = parser.parse_args()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        print("   export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("🚀 OpenAI Chat Completions API - Comprehensive Examples")
    print("=" * 60)
    print("API: POST /v1/chat/completions")
    print("Features:")
    print("  ✅ Standard OpenAI-compatible endpoint")
    print("  ✅ Manual conversation history")
    print("  ✅ Type-safe Pydantic V2 models")
    print("  ✅ Zero magic strings (all enums)")
    print("  ✅ Works with GPT-5, GPT-4, GPT-3.5")
    print("=" * 60)

    demos = [
        lambda: demo_basic_completion(args.model),
        lambda: demo_streaming(args.model),
        lambda: demo_function_calling(args.model),
        lambda: demo_vision("gpt-4o"),
        lambda: demo_json_mode(args.model),
        demo_model_discovery,
        demo_gpt5_models,
        demo_model_comparison,
        lambda: demo_parameters(args.model),
        lambda: demo_error_handling(args.model),
        lambda: demo_multi_turn(args.model),
        lambda: demo_structured_outputs(args.model),
        demo_reasoning_models,
    ]

    if args.demo:
        if 1 <= args.demo <= len(demos):
            try:
                await demos[args.demo - 1]()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print(f"\n❌ Demo failed: {str(e)[:200]}")
                sys.exit(1)
        else:
            print(f"❌ Invalid demo. Choose 1-{len(demos)}")
            sys.exit(1)
    else:
        # Skip slow demos if --quick
        skip_demos = {6, 7} if args.quick else set()

        for i, demo in enumerate(demos, 1):
            if i in skip_demos:
                print(f"\n⏩ Skipping Demo {i} (--quick mode)")
                continue

            try:
                await demo()
            except KeyboardInterrupt:
                print(f"\n⏸️  Demo {i} interrupted by user")
                break
            except (Exception, LLMError) as e:
                print(f"\n❌ Error in demo {i}: {str(e)[:200]}")
                # Continue with next demo instead of crashing

    print(f"\n{'='*60}")
    print("🎉 All Demos Complete!")
    print("=" * 60)
    print("\nKey Takeaway:")
    print("  Chat Completions API = Manual conversation history management")
    print("  You build and pass the full messages array on each request")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
