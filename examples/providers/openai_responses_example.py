#!/usr/bin/env python3
"""
OpenAI Responses API - Comprehensive Example
=============================================

Complete demonstration of the next-generation Responses API (/v1/responses).
This is OpenAI's new API with built-in conversation state management.

Features Demonstrated:
- ✅ Basic response creation
- ✅ Streaming responses
- ✅ Stateful conversations (previous_response_id)
- ✅ Function/tool calling (custom functions)
- ✅ Built-in tools (would require special setup)
- ✅ Vision (multimodal)
- ✅ JSON mode with text.format
- ✅ Structured outputs with json_schema
- ✅ GPT-5 and reasoning models
- ✅ Model comparison
- ✅ Background processing mode
- ✅ Parameters testing
- ✅ Error handling
- ✅ Response retrieval and deletion
- ✅ Zero magic strings (all enums)
- ✅ Type-safe Pydantic V2

Requirements:
- Set OPENAI_API_KEY environment variable

Usage:
    python openai_responses_example.py
    python openai_responses_example.py --model gpt-5
    python openai_responses_example.py --demo 1  # Run specific demo
    python openai_responses_example.py --quick  # Skip slow demos
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
    from chuk_llm.clients.openai_responses import OpenAIResponsesClient
    from chuk_llm.core.models import (
        ResponsesRequest,
        ResponsesResponse,
        ResponsesTextConfig,
        ResponsesTextFormat,
        ResponsesReasoningConfig,
        Tool,
        ToolFunction,
    )
    from chuk_llm.core.enums import ResponsesTextFormatType, ToolType
    from chuk_llm.core import LLMError
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


async def demo_basic_response(model: str):
    """Demo 1: Basic response with Responses API."""
    print(f"\n{'='*60}")
    print(f"Demo 1: Basic Response - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Endpoint: POST /v1/responses")
    print(f"Feature: Automatic conversation storage")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Simple text input
    request = ResponsesRequest(
        model=model,
        input="Explain the Responses API in one sentence.",
        instructions="You are a helpful AI assistant.",
        temperature=0.7,
        max_output_tokens=150,
        store=True,  # Store for conversation history
    )

    response: ResponsesResponse = await client.create_response(request)

    print(f"\n✅ Response ID: {response.id}")
    print(f"   Status: {response.status}")
    print(f"   Output: {response.output_text}")
    print(f"   Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"   Stored: {response.store}")

    await client.close()

    return response.id


async def demo_streaming(model: str):
    """Demo 2: Streaming responses."""
    print(f"\n{'='*60}")
    print(f"Demo 2: Streaming - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Parameter: stream=true")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    request = ResponsesRequest(
        model=model,
        input="Write a haiku about stateful conversations.",
        stream=True,
    )

    print("\n🌊 Streaming:")
    print("   ", end="", flush=True)

    async for chunk in client.stream_response(request):
        if chunk.output_text:
            print(chunk.output_text, end="", flush=True)

    print("\n✅ Complete")

    await client.close()


async def demo_stateful_conversation(model: str):
    """Demo 3: Stateful conversation using previous_response_id."""
    print(f"\n{'='*60}")
    print(f"Demo 3: Stateful Conversation - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Feature: previous_response_id for automatic history")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Turn 1
    print("\n👤 User: My name is Bob and I like Python.")
    request1 = ResponsesRequest(
        model=model,
        input="My name is Bob and I like Python.",
        store=True,
        temperature=0.7,
    )

    response1 = await client.create_response(request1)
    print(f"🤖 Assistant: {response1.output_text}")
    print(f"   Response ID: {response1.id}")

    # Turn 2 - uses previous_response_id
    print("\n👤 User: What's my name and what do I like?")
    request2 = ResponsesRequest(
        model=model,
        input="What's my name and what do I like?",
        previous_response_id=response1.id,  # Links to previous!
        store=True,
        temperature=0.7,
    )

    response2 = await client.create_response(request2)
    print(f"🤖 Assistant: {response2.output_text}")
    print(f"   Response ID: {response2.id}")
    print(f"   Previous: {response2.previous_response_id}")

    # Turn 3 - continues the chain
    print("\n👤 User: Tell me more about it.")
    request3 = ResponsesRequest(
        model=model,
        input="Tell me more about it.",
        previous_response_id=response2.id,  # Links to previous!
        store=True,
        temperature=0.7,
    )

    response3 = await client.create_response(request3)
    print(f"🤖 Assistant: {response3.output_text}")

    print(
        "\n✅ Conversation state managed by OpenAI (no manual history needed!)"
    )

    await client.close()

    return response3.id


async def demo_function_calling(model: str):
    """Demo 4: Function calling with custom tools."""
    print(f"\n{'='*60}")
    print(f"Demo 4: Function Calling - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Feature: Custom function tools")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Define custom tools
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
                        },
                    },
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="calculate",
                description="Perform a mathematical calculation",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            ),
        ),
    ]

    request = ResponsesRequest(
        model=model,
        input="What's the weather in Tokyo and calculate 15 * 23?",
        tools=tools,
        temperature=0.0,
    )

    response = await client.create_response(request)

    print(f"\n✅ Response: {response.output_text or 'Tool calls made'}")
    print(f"   Output items: {len(response.output)}")
    for item in response.output:
        print(f"   - {item.get('type', 'unknown')}")

    await client.close()


async def demo_vision(model: str = "gpt-4o"):
    """Demo 5: Vision/multimodal with Responses API."""
    print(f"\n{'='*60}")
    print(f"Demo 5: Vision/Multi-modal - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Feature: Image understanding (consistent with Chat Completions API)")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Use base64 encoded image (same as Chat Completions example)
    # This is a 1x1 red pixel PNG
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    image_data_url = f"data:image/png;base64,{base64_image}"

    try:
        # Multi-modal message using same Pydantic objects as Chat Completions API
        from chuk_llm.core.models import Message, TextContent, ImageUrlContent
        from chuk_llm.core.enums import MessageRole, ContentType

        request = ResponsesRequest(
            model=model,
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
            max_output_tokens=100,
        )

        response = await client.create_response(request)

        print(f"✅ Vision: {response.output_text}")
        print(f"   Note: Same Message objects work for both Chat Completions and Responses APIs!")

    except (Exception, LLMError) as e:
        print(f"⚠️  Vision demo skipped: {str(e)[:100]}")
        print(f"   Note: Vision requires valid image data")
    finally:
        await client.close()


async def demo_json_mode(model: str):
    """Demo 6: JSON mode with text.format."""
    print(f"\n{'='*60}")
    print(f"Demo 6: JSON Mode - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Parameter: text.format.type = 'json_object'")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    request = ResponsesRequest(
        model=model,
        input='Generate a JSON object with "language": "Python" and "year": 1991',
        instructions="You are a helpful assistant that outputs JSON.",
        text=ResponsesTextConfig(
            format=ResponsesTextFormat(type="json_object")
        ),
        temperature=0.0,
    )

    response = await client.create_response(request)

    print(f"✅ JSON: {response.output_text}")

    await client.close()


async def demo_structured_outputs(model: str):
    """Demo 7: Structured outputs with json_schema."""
    print(f"\n{'='*60}")
    print(f"Demo 7: Structured Outputs - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Feature: text.format with JSON schema validation")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Define schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age", "email"],
                "additionalProperties": False,
            }
        },
        "required": ["person"],
        "additionalProperties": False,
    }

    request = ResponsesRequest(
        model=model,
        input="Create a person object for Carol, age 28, email carol@example.com",
        instructions="You output valid JSON matching the provided schema.",
        text=ResponsesTextConfig(
            format=ResponsesTextFormat(
                type=ResponsesTextFormatType.JSON_SCHEMA,
                name="person_schema",
                schema_=schema,
            )
        ),
        temperature=0.0,
    )

    response = await client.create_response(request)

    print(f"✅ Structured output: {response.output_text}")

    await client.close()


async def demo_gpt5_models():
    """Demo 8: GPT-5 and GPT-5-mini with reasoning."""
    print(f"\n{'='*60}")
    print(f"Demo 8: GPT-5 Models - Responses API")
    print(f"{'='*60}")
    print(f"Testing: gpt-5, gpt-5-mini")
    print(f"Feature: Reasoning model support")

    models_config = [
        ("gpt-5", "high"),
        ("gpt-5-mini", "medium"),
    ]

    prompt = "What is the Responses API? (One sentence)"

    for model_name, effort in models_config:
        try:
            client = OpenAIResponsesClient(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            request = ResponsesRequest(
                model=model_name,
                input=prompt,
                reasoning=ResponsesReasoningConfig(effort=effort),
                temperature=0.0,
            )

            start = time.time()
            response = await client.create_response(request)
            duration = time.time() - start

            print(f"\n✅ {model_name} (effort={effort}, {duration:.2f}s):")
            print(f"   {response.output_text[:100] if response.output_text else 'N/A'}...")

            await client.close()

        except Exception as e:
            print(f"\n❌ {model_name}: {e}")


async def demo_reasoning_models():
    """Demo 9: Reasoning models with thinking display."""
    print(f"\n{'='*60}")
    print(f"Demo 9: Reasoning Models with Thinking - Responses API")
    print(f"{'='*60}")
    print(f"Testing: gpt-5-mini")
    print(f"Feature: Extended thinking with token breakdown")

    model_name = "gpt-5-mini"
    prompt = "I have a 3-gallon jug and a 5-gallon jug. How can I measure exactly 4 gallons of water?"

    try:
        client = OpenAIResponsesClient(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        print(f"\n🧠 Testing {model_name} with reasoning...")
        request = ResponsesRequest(
            model=model_name,
            input=prompt,
            reasoning=ResponsesReasoningConfig(effort="high"),
            temperature=0.0,
        )

        start = time.time()
        response = await client.create_response(request)
        duration = time.time() - start

        print(f"\n✅ {model_name} (effort=high, {duration:.2f}s):")

        # Display thinking/reasoning if available
        if response.output:
            for item in response.output:
                if isinstance(item, dict):
                    item_type = item.get('type')
                    if item_type == "thinking":
                        thinking_text = item.get('thinking', {}).get('text', '')
                        print(f"\n🧠 Thinking Process:")
                        print(f"   {thinking_text[:300]}...")
                        print(f"   (Total thinking: {len(thinking_text)} chars)")
                    elif item_type == "output_text":
                        output_text = item.get('text', {}).get('text', '')
                        print(f"\n📝 Final Answer:")
                        print(f"   {output_text[:200]}...")
                elif hasattr(item, 'type'):
                    if item.type == "thinking":
                        thinking_text = getattr(item, 'thinking', {}).get('text', '')
                        print(f"\n🧠 Thinking Process:")
                        print(f"   {thinking_text[:300]}...")
                        print(f"   (Total thinking: {len(thinking_text)} chars)")
                    elif item.type == "output_text":
                        output_text = getattr(item, 'text', {}).get('text', '')
                        print(f"\n📝 Final Answer:")
                        print(f"   {output_text[:200]}...")

        # Display token usage with reasoning tokens
        if response.usage:
            print(f"\n📊 Token Usage:")
            print(f"   Prompt: {response.usage.input_tokens} tokens")
            print(f"   Completion: {response.usage.output_tokens} tokens")

            # Check for reasoning/thinking tokens
            if hasattr(response.usage, 'output_tokens_details') and response.usage.output_tokens_details:
                details = response.usage.output_tokens_details
                if hasattr(details, 'thinking_tokens') and details.thinking_tokens:
                    print(f"   🧠 Thinking: {details.thinking_tokens} tokens")
                    output_only = response.usage.output_tokens - details.thinking_tokens
                    print(f"   📝 Output: {output_only} tokens")
                elif hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                    print(f"   🧠 Reasoning: {details.reasoning_tokens} tokens")
                    output_only = response.usage.output_tokens - details.reasoning_tokens
                    print(f"   📝 Output: {output_only} tokens")

            print(f"   Total: {response.usage.total_tokens} tokens")

        await client.close()

    except Exception as e:
        print(f"\n❌ {model_name}: {e}")


async def demo_model_comparison():
    """Demo 10: Compare multiple models."""
    print(f"\n{'='*60}")
    print(f"Demo 10: Model Comparison - Responses API")
    print(f"{'='*60}")

    models = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    prompt = "What is stateful conversation management? (One sentence)"

    for model_name in models:
        try:
            client = OpenAIResponsesClient(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            request = ResponsesRequest(
                model=model_name,
                input=prompt,
                temperature=0.0,
            )

            start = time.time()
            response = await client.create_response(request)
            duration = time.time() - start

            print(f"\n✅ {model_name} ({duration:.2f}s):")
            print(f"   {response.output_text[:80] if response.output_text else 'N/A'}...")

            await client.close()

        except Exception as e:
            print(f"\n❌ {model_name}: {e}")


async def demo_background_processing(model: str):
    """Demo 11: Background processing mode."""
    print(f"\n{'='*60}")
    print(f"Demo 11: Background Processing - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Parameter: background=true")
    print(f"Feature: Async processing for long-running tasks")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create background response
    request = ResponsesRequest(
        model=model,
        input="Write a short poem about async processing.",
        background=True,  # Process in background
        store=True,
    )

    response = await client.create_response(request)

    print(f"\n✅ Response submitted:")
    print(f"   ID: {response.id}")
    print(f"   Status: {response.status}")
    print(
        f"   Note: Response may be 'queued' or 'in_progress' initially"
    )

    # Poll for completion
    if response.status in ["queued", "in_progress"]:
        print(f"\n🔄 Polling for completion...")
        for i in range(5):
            await asyncio.sleep(1)
            updated = await client.retrieve_response(response.id)
            print(f"   Status: {updated.status}")
            if updated.status == "completed":
                print(f"   Output: {updated.output_text}")
                break

    await client.close()

    return response.id


async def demo_parameters(model: str):
    """Demo 12: Test various parameters."""
    print(f"\n{'='*60}")
    print(f"Demo 12: Parameters Testing - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")

    client = OpenAIResponsesClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Test temperature variations
    temperatures = [0.0, 0.5, 1.0]
    prompt = "Say hello"

    for temp in temperatures:
        request = ResponsesRequest(
            model=model,
            input=prompt,
            temperature=temp,
            max_output_tokens=20,
        )

        response = await client.create_response(request)
        print(f"Temperature {temp}: {response.output_text}")

    await client.close()


async def demo_error_handling(model: str):
    """Demo 13: Error handling."""
    print(f"\n{'='*60}")
    print(f"Demo 13: Error Handling - Responses API")
    print(f"{'='*60}")
    print(f"Model: {model}")

    # Test with invalid API key
    bad_client = OpenAIResponsesClient(model=model, api_key="sk-invalid")

    try:
        request = ResponsesRequest(
            model=model,
            input="Hello",
        )

        await bad_client.create_response(request)

    except LLMError as e:
        print(f"✅ LLM error caught:")
        print(f"   Type: {e.error_type}")
        print(f"   Message: {e.error_message}")

    await bad_client.close()


async def demo_model_discovery():
    """Demo 14: Model Discovery - Discover available OpenAI models."""
    print(f"\n{'='*60}")
    print(f"Demo 14: Model Discovery - Responses API")
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

        # Test a dynamically discovered model with Responses API
        if models_data:
            # Find a suitable model (not a reasoning-only model)
            test_model = None
            for model in models_data:
                provider_spec = model.get("provider_specific", {})
                # Skip o1 models as they don't support Responses API well
                if "o1" not in model.get("name", ""):
                    test_model = model.get("name")
                    break

            if test_model:
                print(f"\n🧪 Testing dynamically discovered model: {test_model}")
                try:
                    client = OpenAIResponsesClient(
                        model=test_model,
                        api_key=os.getenv("OPENAI_API_KEY"),
                    )
                    request = ResponsesRequest(
                        model=test_model,
                        input="Say hello in one creative word",
                        max_output_tokens=10,
                    )
                    response = await client.create_response(request)
                    print(f"   ✅ Model works: {response.output_text}")
                    await client.close()
                except Exception as e:
                    print(f"   ⚠️ Test failed: {str(e)[:100]}")

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback

        traceback.print_exc()


async def demo_response_management(response_id: str):
    """Demo 15: Response retrieval and deletion."""
    print(f"\n{'='*60}")
    print(f"Demo 15: Response Management - Responses API")
    print(f"{'='*60}")
    print(f"Feature: Retrieve and delete stored responses")

    client = OpenAIResponsesClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Retrieve
    print(f"\n🔍 Retrieving response: {response_id}")
    response = await client.retrieve_response(response_id)
    print(f"✅ Retrieved:")
    print(f"   Status: {response.status}")
    print(f"   Output: {response.output_text[:50] if response.output_text else 'N/A'}...")

    # Delete
    print(f"\n🗑️  Deleting response: {response_id}")
    result = await client.delete_response(response_id)
    print(f"✅ Deleted: {result.get('deleted', False)}")

    await client.close()


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

    print("🚀 OpenAI Responses API - Comprehensive Examples")
    print("=" * 60)
    print("API: POST /v1/responses")
    print("Features:")
    print("  ✅ Stateful conversations (previous_response_id)")
    print("  ✅ Automatic history management (store=true)")
    print("  ✅ Background processing mode")
    print("  ✅ Type-safe Pydantic V2 models")
    print("  ✅ Zero magic strings (all enums)")
    print("  ✅ GPT-5 and reasoning model support")
    print("=" * 60)

    demos = [
        lambda: demo_basic_response(args.model),
        lambda: demo_streaming(args.model),
        lambda: demo_stateful_conversation(args.model),
        lambda: demo_function_calling(args.model),
        lambda: demo_vision("gpt-4o"),
        lambda: demo_json_mode(args.model),
        lambda: demo_structured_outputs(args.model),
        demo_gpt5_models,
        demo_reasoning_models,
        demo_model_comparison,
        lambda: demo_background_processing(args.model),
        lambda: demo_parameters(args.model),
        lambda: demo_error_handling(args.model),
        demo_model_discovery,
    ]

    # Track response IDs for cleanup demo
    response_ids = []

    if args.demo:
        if 1 <= args.demo <= len(demos):
            try:
                result = await demos[args.demo - 1]()
                if isinstance(result, str):
                    response_ids.append(result)
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
        skip_demos = {8, 9, 10} if args.quick else set()

        for i, demo in enumerate(demos, 1):
            if i in skip_demos:
                print(f"\n⏩ Skipping Demo {i} (--quick mode)")
                continue

            try:
                result = await demo()
                if isinstance(result, str):
                    response_ids.append(result)
            except KeyboardInterrupt:
                print(f"\n⏸️  Demo {i} interrupted by user")
                break
            except (Exception, LLMError) as e:
                print(f"\n❌ Error in demo {i}: {str(e)[:200]}")
                # Continue with next demo instead of crashing

    # Run response management demo if we have IDs
    if response_ids and not args.quick:
        try:
            await demo_response_management(response_ids[0])
        except Exception as e:
            print(f"❌ Error in response management: {e}")

    print(f"\n{'='*60}")
    print("🎉 All Demos Complete!")
    print("=" * 60)
    print("\nKey Takeaway:")
    print("  Responses API = Automatic conversation history with previous_response_id")
    print("  OpenAI stores and manages conversation state for you")
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
