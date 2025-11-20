#!/usr/bin/env python3
# examples/providers/openai_usage_examples.py
"""
OpenAI Provider Example Usage Script
=====================================

Demonstrates all features of the OpenAI provider including GPT-4, GPT-5,
o1/o3 reasoning models, vision, function calling, and streaming.

Requirements:
- pip install chuk-llm
- Set OPENAI_API_KEY environment variable

Usage:
    python openai_usage_examples.py
    python openai_usage_examples.py --model gpt-4o
    python openai_usage_examples.py --model o1-mini  # Reasoning model
    python openai_usage_examples.py --skip-tools
    python openai_usage_examples.py --skip-vision
"""

import argparse
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# Ensure we have the required environment
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    print("   Get your key at: https://platform.openai.com/api-keys")
    sys.exit(1)

try:
    from chuk_llm.configuration import Feature, get_config
    from chuk_llm.llm.client import get_client, get_provider_info
    from chuk_llm.core.models import Message, Tool, ToolFunction, TextContent, ImageUrlContent
    from chuk_llm.core.enums import MessageRole, ContentType, ToolType

    # Import common demos
    import sys
    from pathlib import Path
    examples_dir = Path(__file__).parent.parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from common_demos import (
        demo_basic_completion,
        demo_streaming,
        demo_function_calling,
        demo_vision,
        demo_json_mode,
        demo_reasoning,
        demo_conversation,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please install chuk-llm: pip install chuk-llm")
    sys.exit(1)


# =============================================================================
# Example 1: Basic Text Completion (using common demo)
# =============================================================================

async def basic_text_example(model: str = "gpt-4o-mini"):
    """Use common demo for basic completion"""
    client = get_client("openai", model=model)
    return await demo_basic_completion(client, "openai", model)


# =============================================================================
# Example 1b: Basic Text Completion (original)
# =============================================================================

async def basic_text_example_original(model: str = "gpt-4o-mini"):
    """Basic text completion with GPT models"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_client("openai", model=model)

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        Message(
            role=MessageRole.USER,
            content="Explain quantum computing in simple terms (2-3 sentences).",
        ),
    ]

    response = await client.create_completion(messages)
    print(f"✅ Response:\n   {response['response']}")

    return response


# =============================================================================
# Example 2: Streaming Response
# =============================================================================

async def streaming_example(model: str = "gpt-4o-mini"):
    """Real-time streaming with OpenAI"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    client = get_client("openai", model=model)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a short haiku about artificial intelligence.",
        )
    ]

    print("🌊 Streaming response:")
    print("   ", end="", flush=True)

    full_response = ""
    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            content = chunk["response"]
            print(content, end="", flush=True)
            full_response += content

    print()
    return full_response


# =============================================================================
# Example 3: Function Calling / Tool Use
# =============================================================================

async def function_calling_example(model: str = "gpt-4o-mini"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling Example with {model}")
    print("=" * 60)

    # Check if model supports tools
    config = get_config()
    if not config.supports_feature("openai", Feature.TOOLS, model):
        print(f"⚠️  Model {model} doesn't support function calling")
        return None

    client = get_client("openai", model=model)

    # Define tools
    tools = [
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="calculate",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            ),
        ),
    ]

    messages = [
        Message(
            role=MessageRole.USER,
            content="What's the weather in London and calculate 25 * 4?",
        )
    ]

    response = await client.create_completion(messages, tools=tools)

    if response.get("tool_calls"):
        print(f"✅ Model wants to call {len(response['tool_calls'])} function(s):")
        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])
            print(f"   📞 {func_name}({func_args})")

            # Simulate function execution
            if func_name == "get_weather":
                result = f"Weather in {func_args.get('location')}: 22°C, partly cloudy"
            elif func_name == "calculate":
                try:
                    result = str(eval(func_args.get("expression", "0")))
                except:
                    result = "Error calculating"
            else:
                result = "Unknown function"

            print(f"   ✓ Result: {result}")

        # Continue conversation with function results
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=response.get("response"),
                tool_calls=response.get("tool_calls")
            )
        )

        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            if func_name == "get_weather":
                result = f"Weather in {func_args.get('location')}: 22°C, partly cloudy"
            elif func_name == "calculate":
                result = str(eval(func_args.get("expression", "0")))

            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=result,
                    tool_call_id=tool_call["id"],
                    name=func_name
                )
            )

        # Get final response with function results
        final_response = await client.create_completion(messages, tools=tools)
        print(f"\n✅ Final response:\n   {final_response['response']}")

    else:
        print(f"✅ Response (no function calls):\n   {response['response']}")

    return response


# =============================================================================
# Example 4: Vision / Multimodal
# =============================================================================

async def vision_example(model: str = "gpt-4o"):
    """Vision/multimodal example with GPT-4o"""
    print(f"\n👁️ Vision Example with {model}")
    print("=" * 60)

    # Check if model supports vision
    config = get_config()
    if not config.supports_feature("openai", Feature.VISION, model):
        print(f"⚠️  Model {model} doesn't support vision")
        return None

    client = get_client("openai", model=model)

    # Create a simple 1x1 red pixel image (base64 encoded)
    red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                TextContent(type=ContentType.TEXT, text="What color is this image?"),
                ImageUrlContent(
                    type=ContentType.IMAGE_URL,
                    image_url={"url": f"data:image/png;base64,{red_pixel_base64}"},
                ),
            ],
        )
    ]

    response = await client.create_completion(messages)
    print(f"✅ Response:\n   {response['response']}")

    return response


# =============================================================================
# Example 5: Reasoning Models (o1, o3) - WITH THINKING VISIBILITY
# =============================================================================

async def reasoning_model_example(model: str = "o1-mini"):
    """Reasoning model with extended thinking - shows full thinking process"""
    client = get_client("openai", model=model)
    return await demo_reasoning(client, "openai", model)


# =============================================================================
# Example 6: JSON Mode
# =============================================================================

async def json_mode_example(model: str = "gpt-4o-mini"):
    """Structured JSON output"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)

    client = get_client("openai", model=model)

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that outputs JSON.",
        ),
        Message(
            role=MessageRole.USER,
            content="Generate a person with name, age, and hobbies (array).",
        ),
    ]

    response = await client.create_completion(
        messages,
        response_format={"type": "json_object"}
    )

    print(f"✅ JSON Response:")
    try:
        parsed = json.loads(response['response'])
        print(json.dumps(parsed, indent=2))
    except:
        print(response['response'])

    return response


# =============================================================================
# Example 7: Multi-turn Conversation
# =============================================================================

async def conversation_example(model: str = "gpt-4o-mini"):
    """Multi-turn conversation with context"""
    print(f"\n💬 Conversation Example with {model}")
    print("=" * 60)

    client = get_client("openai", model=model)

    conversation = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="My name is Alice."),
    ]

    # First turn
    response1 = await client.create_completion(conversation)
    print(f"User: My name is Alice.")
    print(f"Assistant: {response1['response']}")

    # Add to conversation
    conversation.append(Message(role=MessageRole.ASSISTANT, content=response1['response']))
    conversation.append(Message(role=MessageRole.USER, content="What's my name?"))

    # Second turn (tests memory)
    response2 = await client.create_completion(conversation)
    print(f"\nUser: What's my name?")
    print(f"Assistant: {response2['response']}")

    return response2


# =============================================================================
# Example 8: Model Information
# =============================================================================

async def model_info_example(model: str = "gpt-4o-mini"):
    """Get detailed model information"""
    print(f"\n📊 Model Information for {model}")
    print("=" * 60)

    client = get_client("openai", model=model)
    info = client.get_model_info()

    print(f"Provider: {info.get('provider')}")
    print(f"Model: {info.get('model')}")
    print(f"Max context: {info.get('max_context_length', 'unknown')} tokens")
    print(f"Max output: {info.get('max_output_tokens', 'unknown')} tokens")

    print(f"\nFeatures:")
    features = [
        "supports_text",
        "supports_streaming",
        "supports_tools",
        "supports_vision",
        "supports_json_mode",
    ]
    for feature in features:
        supported = info.get(feature, False)
        icon = "✅" if supported else "❌"
        print(f"  {icon} {feature.replace('supports_', '')}")

    # Check for reasoning model
    if info.get("openai_specific", {}).get("is_reasoning_model"):
        print(f"\n🧠 This is a reasoning model (o1/o3 series)")
        print(f"  - Extended thinking before responding")
        print(f"  - No system messages")
        print(f"  - No temperature control")

    return info


# =============================================================================
# Main Runner
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="OpenAI Provider Examples")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip-tools", action="store_true", help="Skip function calling example"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision example"
    )
    parser.add_argument(
        "--skip-reasoning", action="store_true", help="Skip reasoning model example"
    )
    parser.add_argument(
        "--example",
        type=int,
        help="Run specific example number (1-8)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🚀 OpenAI Provider Examples")
    print("=" * 60)

    examples = []

    if args.example is None or args.example == 1:
        examples.append(("Basic Text", basic_text_example(args.model)))

    if args.example is None or args.example == 2:
        examples.append(("Streaming", streaming_example(args.model)))

    if (args.example is None or args.example == 3) and not args.skip_tools:
        examples.append(("Function Calling", function_calling_example(args.model)))

    if (args.example is None or args.example == 4) and not args.skip_vision:
        # Use GPT-4o for vision
        vision_model = "gpt-4o" if "gpt-4o" in args.model else "gpt-4o"
        examples.append(("Vision", vision_example(vision_model)))

    if (args.example is None or args.example == 5) and not args.skip_reasoning:
        examples.append(("Reasoning Model", reasoning_model_example("o1-mini")))

    if args.example is None or args.example == 6:
        examples.append(("JSON Mode", json_mode_example(args.model)))

    if args.example is None or args.example == 7:
        examples.append(("Conversation", conversation_example(args.model)))

    if args.example is None or args.example == 8:
        examples.append(("Model Info", model_info_example(args.model)))

    # Run all examples
    for name, example_coro in examples:
        try:
            await example_coro
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
