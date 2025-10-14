#!/usr/bin/env python3
# examples/llm_provider_examples/advantage_usage_examples.py
"""
Advantage Provider Example Usage Script
=======================================

Demonstrates all the features of the Advantage provider in the chuk-llm library.
Advantage is an OpenAI-compatible provider with enhanced function calling support.

This script focuses on testing the global/gpt-5-chat model specifically, as requested.

Requirements:
- uv sync  # or pip install chuk-llm
- Set ADVANTAGE_API_KEY environment variable (for Advantage API key)
- Set ADVANTAGE_API_BASE environment variable (for Advantage endpoint)

Usage:
    uv run python advantage_usage_examples.py
    uv run python advantage_usage_examples.py --model global/gpt-5-chat
    uv run python advantage_usage_examples.py --skip-vision

Note: The Advantage provider uses a custom client (AdvantageClient) that:
- Automatically injects system prompts to guide function calling
- Parses function calls from content and converts to tool_calls format
- Handles the strict parameter requirement for tools
"""

import argparse
import asyncio
import base64
import os
import sys
import time

# dotenv
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Ensure we have the required environment
# Support both ADVANTAGE_* and OPENAI_COMPATIBLE_* environment variables
if not (os.getenv("ADVANTAGE_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY")):
    print("❌ Please set ADVANTAGE_API_KEY environment variable (for Advantage API key)")
    print("   export ADVANTAGE_API_KEY='your_advantage_api_key_here'")
    print("   (or OPENAI_COMPATIBLE_API_KEY for backward compatibility)")
    sys.exit(1)

if not (os.getenv("ADVANTAGE_API_BASE") or os.getenv("OPENAI_COMPATIBLE_API_BASE")):
    print("❌ Please set ADVANTAGE_API_BASE environment variable (for Advantage endpoint)")
    print("   export ADVANTAGE_API_BASE='https://your-advantage-endpoint.com/v1'")
    print("   (or OPENAI_COMPATIBLE_API_BASE for backward compatibility)")
    sys.exit(1)

try:
    from chuk_llm.configuration import Feature, get_config
    from chuk_llm.llm.client import get_client, get_provider_info
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)


def create_test_image(color: str = "red", size: int = 15) -> str:
    """Create a test image as base64 - tries PIL first, fallback to hardcoded"""
    try:
        import io

        from PIL import Image

        # Create a colored square
        img = Image.new("RGB", (size, size), color)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_data
    except ImportError:
        print("⚠️  PIL not available, using fallback image")
        # Fallback: 15x15 red square (valid PNG)
        return "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVCiRY2RgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBgZGBgYGAAAgAANgAOAUUe1wAAAABJRU5ErkJggg=="


# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================


async def basic_text_example(model: str = "global/gpt-5-chat"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_client("advantage", model=model)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Explain neural networks in simple terms (2-3 sentences).",
        },
    ]

    start_time = time.time()
    # Advantage API requires modelId in the request body
    # The standard OpenAI client doesn't support this parameter
    response = await client.create_completion(messages)
    duration = time.time() - start_time

    print(f"✅ Response ({duration:.2f}s):")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 2: Streaming Response
# =============================================================================


async def streaming_example(model: str = "global/gpt-5-chat"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    # Check streaming support
    config = get_config()
    if not config.supports_feature("advantage", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

    client = get_client("advantage", model=model)

    messages = [
        {
            "role": "user",
            "content": "Write a short haiku about artificial intelligence.",
        }
    ]

    print("🌊 Streaming response:")
    print("   ", end="", flush=True)

    start_time = time.time()
    full_response = ""

    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            content = chunk["response"]
            print(content, end="", flush=True)
            full_response += content

    duration = time.time() - start_time
    print(f"\n✅ Streaming completed ({duration:.2f}s)")

    return full_response


# =============================================================================
# Example 3: Function Calling
# =============================================================================


async def function_calling_example(model: str = "global/gpt-5-chat"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    # The advantage provider supports function calling with automatic workarounds
    client = get_client("advantage", model=model)

    # Define tools - includes both separate and combined tools to handle model's preference
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_math",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Number of decimal places",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_and_calculate",
                "description": "Search the web and perform a calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "calculation": {
                            "type": "string",
                            "description": "Mathematical expression to calculate",
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Number of decimal places for calculation result",
                        },
                    },
                    "required": ["query", "calculation"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a function-calling assistant. "
                "AVAILABLE FUNCTIONS: search_web, calculate_math, search_and_calculate. "
                "You MUST use these EXACT function names only. "
                "For tasks requiring both search and calculation, use search_and_calculate."
            )
        },
        {
            "role": "user",
            "content": "Call search_and_calculate to search for 'latest AI research' and calculate 25.5 * 14.2 with 3 decimal places."
        }
    ]

    print("🔄 Making function calling request...")
    # The AdvantageClient automatically handles:
    # - Adding the strict parameter to tools
    # - Injecting system prompt to guide function calling
    # - Parsing function calls from content field
    # - Converting to standard tool_calls format

    response = await client.create_completion(
        messages,
        tools=tools
    )

    if response.get("tool_calls"):
        print(f"✅ Tool calls requested: {len(response['tool_calls'])}")
        for i, tool_call in enumerate(response["tool_calls"], 1):
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            print(f"   {i}. {func_name}({func_args})")

        # Simulate tool execution
        messages.append(
            {"role": "assistant", "content": "", "tool_calls": response["tool_calls"]}
        )

        # Add mock tool results
        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]

            if func_name == "search_web":
                result = '{"results": ["Latest breakthrough in transformer models", "New multimodal AI research", "Advances in reasoning capabilities"]}'
            elif func_name == "calculate_math":
                result = (
                    '{"result": 361.100, "expression": "25.5 * 14.2", "precision": 3}'
                )
            elif func_name == "search_and_calculate":
                # Handle combined function - provide both search results and calculation
                result = '{"search_results": ["Latest breakthrough in transformer models", "New multimodal AI research", "Advances in reasoning capabilities"], "calculation_result": 362.100, "expression": "25.5 * 14.2", "precision": 3}'
            else:
                result = '{"status": "success"}'

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": func_name,
                    "content": result,
                }
            )

        # Get final response
        print("🔄 Getting final response...")
        final_response = await client.create_completion(messages)
        print("✅ Final response:")
        print(f"   {final_response['response']}")

        return final_response
    else:
        print("ℹ️  No tool calls were made")
        print(f"   Response: {response['response']}")
        return response


# =============================================================================
# Example 4: Streaming Function Calling
# =============================================================================


async def streaming_function_calling_example(model: str = "global/gpt-5-chat"):
    """Streaming function calling with tools - demonstrates real-time tool call detection"""
    print(f"\n⚡🔧 Streaming Function Calling with {model}")
    print("=" * 60)

    client = get_client("advantage", model=model)

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_tip",
                "description": "Calculate tip amount for a bill",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bill_amount": {
                            "type": "number",
                            "description": "Total bill amount",
                        },
                        "tip_percentage": {
                            "type": "number",
                            "description": "Tip percentage (e.g., 15, 18, 20)",
                        },
                    },
                    "required": ["bill_amount", "tip_percentage"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a function-calling assistant. "
                "AVAILABLE FUNCTIONS: get_weather, calculate_tip. "
                "You MUST use these EXACT function names only."
            )
        },
        {
            "role": "user",
            "content": "Call get_weather for San Francisco and call calculate_tip for a $85.50 bill with 20% tip."
        }
    ]

    print("🌊 Streaming function calling request...")
    print("   ", end="", flush=True)

    start_time = time.time()
    full_response = ""
    tool_calls = []

    try:
        async for chunk in client.create_completion(messages, tools=tools, stream=True):
            # Handle text response chunks
            if chunk.get("response"):
                content = chunk["response"]
                print(content, end="", flush=True)
                full_response += content

            # Handle tool call chunks
            if chunk.get("tool_calls"):
                if not tool_calls:
                    print("\n\n🔧 Tool calls detected:")
                tool_calls = chunk["tool_calls"]

        duration = time.time() - start_time

        if tool_calls:
            print(f"\n✅ Tool calls generated ({len(tool_calls)} calls):")
            for i, tool_call in enumerate(tool_calls, 1):
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                print(f"   {i}. {func_name}({func_args})")

            # Simulate tool execution
            messages.append(
                {"role": "assistant", "content": full_response or "", "tool_calls": tool_calls}
            )

            # Add mock tool results
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]

                if func_name == "get_weather":
                    result = '{"location": "San Francisco", "temperature": 68, "units": "fahrenheit", "conditions": "Partly cloudy", "humidity": 65}'
                elif func_name == "calculate_tip":
                    result = '{"bill_amount": 85.50, "tip_percentage": 20, "tip_amount": 17.10, "total": 102.60}'
                else:
                    result = '{"status": "success"}'

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": func_name,
                        "content": result,
                    }
                )

            # Get final response with streaming
            print("\n🌊 Streaming final response:")
            print("   ", end="", flush=True)

            final_response = ""
            async for chunk in client.create_completion(messages, stream=True):
                if chunk.get("response"):
                    content = chunk["response"]
                    print(content, end="", flush=True)
                    final_response += content

            print(f"\n✅ Streaming function calling completed ({duration:.2f}s)")
            return {"tool_calls": tool_calls, "final_response": final_response}

        else:
            print(f"\n✅ Direct response (no tool calls) ({duration:.2f}s)")
            return {"response": full_response}

    except Exception as e:
        print(f"\n❌ Streaming function calling failed: {e}")
        print("ℹ️  Falling back to non-streaming function calling...")
        return await function_calling_example(model)


# =============================================================================
# Example 5: Vision Capabilities
# =============================================================================


async def vision_example(model: str = "global/gpt-5-chat"):
    """Vision capabilities with vision-enabled models"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)

    # Check if model supports vision
    config = get_config()
    if not config.supports_feature("advantage", Feature.VISION, model):
        print(f"⚠️  Skipping vision: Model {model} doesn't support vision")
        print("💡 Try a vision-capable model like: global/gpt-5-chat, global/chat-gpt-4o, global/chat-gpt-4-turbo")
        return None

    client = get_client("advantage", model=model)

    # Create a proper test image (larger size for better visibility)
    print("🖼️  Creating test image...")
    test_image = create_test_image("blue", 100)

    # Verify image was created
    if not test_image or len(test_image) < 100:
        print("⚠️  Warning: Test image may be invalid or too small")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{test_image}",
                        "detail": "auto"
                    },
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image. What color is the square?",
                },
            ],
        }
    ]

    print("👀 Analyzing image...")
    try:
        response = await client.create_completion(messages, max_tokens=100)
        print("✅ Vision response:")
        print(f"   {response['response']}")

        # Check if vision actually worked
        if "can't" in response['response'].lower() or "cannot" in response['response'].lower() or "don't see" in response['response'].lower():
            print("\n⚠️  Note: The model indicates it cannot process the image.")
        elif "blue" in response['response'].lower():
            print("\n✅ Vision is working! The model correctly identified the blue square.")
    except Exception as e:
        print(f"❌ Vision request failed: {e}")
        print("ℹ️  Vision support may require specific configuration with this provider.")
        return None

    return response


# =============================================================================
# Example 6: JSON Mode
# =============================================================================


async def json_mode_example(model: str = "global/gpt-5-chat"):
    """JSON mode example with structured output"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)

    # Check JSON mode support
    config = get_config()
    if not config.supports_feature("advantage", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None

    client = get_client("advantage", model=model)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant designed to output JSON. Generate a JSON object with information about a programming language.",
        },
        {
            "role": "user",
            "content": "Tell me about Python programming language in JSON format with fields: name, year_created, creator, main_features (array), and popularity_score (1-10).",
        },
    ]

    print("📝 Requesting JSON output...")

    try:
        response = await client.create_completion(
            messages, response_format={"type": "json_object"}, temperature=0.7
        )

        print("✅ JSON response:")
        print(f"   {response['response']}")

        # Try to parse as JSON to verify
        import json
        import re

        try:
            response_text = response["response"]

            # Strip markdown code fences if present
            json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
            elif response_text.startswith('```') and response_text.endswith('```'):
                response_text = response_text.strip('`').strip()

            parsed = json.loads(response_text)
            print(f"✅ Valid JSON structure with keys: {list(parsed.keys())}")
        except json.JSONDecodeError:
            print("⚠️  Response is not valid JSON (but may be wrapped in markdown)")

    except Exception as e:
        print(f"❌ JSON mode failed: {e}")
        # Fallback to regular request
        response = await client.create_completion(messages)
        print(f"📝 Fallback response: {response['response'][:200]}...")

    return response


# =============================================================================
# Example 7: Model Comparison
# =============================================================================


async def model_comparison_example():
    """Test single model (global/gpt-5-chat) with different prompts"""
    print("\n📊 Single Model Test (global/gpt-5-chat)")
    print("=" * 60)

    # Test different types of prompts with the single model
    test_prompts = [
        ("Simple Question", "What is machine learning? (One sentence)"),
        ("Creative Task", "Write a haiku about AI"),
        ("Technical Query", "Explain recursion briefly"),
        ("Math Problem", "What is 15 * 23?"),
    ]

    model = "global/gpt-5-chat"
    results = {}

    for prompt_name, prompt_text in test_prompts:
        try:
            print(f"🔄 Testing {prompt_name}...")
            client = get_client("advantage", model=model)
            messages = [{"role": "user", "content": prompt_text}]

            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time

            results[prompt_name] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "success": True,
            }

        except Exception as e:
            results[prompt_name] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "success": False,
            }

    print(f"\n📈 Results for {model}:")
    for prompt_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"   {status} {prompt_name}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()

    return results


# =============================================================================
# Example 8: Feature Detection
# =============================================================================


async def feature_detection_example(model: str = "global/gpt-5-chat"):
    """Detect and display model features"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)

    # Get model info
    try:
        model_info = get_provider_info("advantage", model)

        print("📋 Model Information:")
        print(f"   Provider: {model_info['provider']}")
        print(f"   Model: {model_info['model']}")

        # Handle potentially None values
        max_context = model_info.get('max_context_length')
        max_output = model_info.get('max_output_tokens')

        if max_context is not None:
            print(f"   Max Context: {max_context:,} tokens")
        else:
            print(f"   Max Context: Not specified")

        if max_output is not None:
            print(f"   Max Output: {max_output:,} tokens")
        else:
            print(f"   Max Output: Not specified")

        print("\n🎯 Supported Features:")
        for feature, supported in model_info["supports"].items():
            status = "✅" if supported else "❌"
            print(f"   {status} {feature}")

        print("\n📊 Rate Limits:")
        for tier, limit in model_info["rate_limits"].items():
            print(f"   {tier}: {limit} requests/min")

    except Exception as e:
        print(f"⚠️  Could not get model info: {e}")

    # Test actual client info
    try:
        client = get_client("advantage", model=model)
        client_info = client.get_model_info()

        print("\n🔧 Client Features:")
        print(
            f"   Streaming: {'✅' if client_info.get('supports_streaming') else '❌'}"
        )
        print(f"   Tools: {'✅' if client_info.get('supports_tools') else '❌'}")
        print(f"   Vision: {'✅' if client_info.get('supports_vision') else '❌'}")
        print(
            f"   JSON Mode: {'✅' if client_info.get('supports_json_mode') else '❌'}"
        )
        print(
            f"   System Messages: {'✅' if client_info.get('supports_system_messages') else '❌'}"
        )

    except Exception as e:
        print(f"⚠️  Could not get client info: {e}")

    return model_info if "model_info" in locals() else None


# =============================================================================
# Example 9: Simple Chat Interface
# =============================================================================


async def simple_chat_example(model: str = "global/gpt-5-chat"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface with {model}")
    print("=" * 60)

    client = get_client("advantage", model=model)

    # Simulate a simple conversation
    conversation = [
        "Hello! What's the weather like?",
        "What's the most exciting development in AI recently?",
        "Can you help me write a JavaScript function to sort an array?",
    ]

    messages = [
        {"role": "system", "content": "You are a helpful and friendly AI assistant."}
    ]

    for user_input in conversation:
        print(f"👤 User: {user_input}")

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Get response
        response = await client.create_completion(messages, max_tokens=200)
        assistant_response = response.get("response", "No response")

        print(f"🤖 Assistant: {assistant_response}")
        print()

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})

    return messages


# =============================================================================
# Example 10: Temperature and Parameters Test
# =============================================================================


async def parameters_example(model: str = "global/gpt-5-chat"):
    """Test different parameters and settings"""
    print(f"\n🎛️  Parameters Test with {model}")
    print("=" * 60)

    client = get_client("advantage", model=model)

    # Test different temperatures
    temperatures = [0.1, 0.7, 1.2]
    prompt = "Write a creative opening line for a science fiction story."

    for temp in temperatures:
        print(f"\n🌡️  Temperature {temp}:")

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await client.create_completion(
                messages, temperature=temp, max_tokens=50
            )
            print(f"   {response['response']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Test with system message
    print("\n🎭 With System Message:")
    messages = [
        {"role": "system", "content": "You are a poetic AI that speaks in rhymes."},
        {"role": "user", "content": "Tell me about the ocean."},
    ]

    response = await client.create_completion(messages, temperature=0.8, max_tokens=100)
    print(f"   {response['response']}")

    return True


# =============================================================================
# Example 11: Comprehensive Feature Test
# =============================================================================


async def comprehensive_test(model: str = "global/gpt-5-chat"):
    """Test multiple features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)

    # Check what features this model supports
    config = get_config()
    supports_tools = config.supports_feature("advantage", Feature.TOOLS, model)
    supports_vision = config.supports_feature("advantage", Feature.VISION, model)

    print(f"Model capabilities: Tools={supports_tools}, Vision={supports_vision}")

    if not supports_tools and not supports_vision:
        print("⚠️  Model doesn't support tools or vision - using text-only test")
        return await simple_chat_example(model)

    client = get_client("advantage", model=model)

    # Define tools if supported
    tools = None
    if supports_tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_content",
                    "description": "Analyze and categorize text or image content",
                    "strict": False,  # Add required strict parameter for Advantage API
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content_type": {
                                "type": "string",
                                "description": "Type of content being analyzed (e.g., 'text', 'image', 'multimodal')"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text content to analyze (if applicable)"
                            },
                            "image_description": {
                                "type": "string",
                                "description": "Description of image content (if applicable)"
                            },
                            "main_topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Main topics or themes found in the content"
                            },
                            "complexity": {
                                "type": "string",
                                "enum": ["simple", "medium", "complex"],
                                "description": "Complexity level of the content"
                            },
                        },
                        "required": ["content_type", "main_topics"],
                    },
                },
            }
        ]

    # Create content based on capabilities
    if supports_vision:
        print("🖼️  Creating test image...")
        test_image = create_test_image("green", 25)

        messages = [
            {
                "role": "system",
                "content": "You are an expert content analyst. Use the provided function when analyzing content.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{test_image}"},
                    },
                    {
                        "type": "text",
                        "text": "Please analyze this image and the following text using the analyze_content function: 'This is a test of multimodal AI capabilities.'",
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are an expert content analyst. Use the provided function when analyzing content.",
            },
            {
                "role": "user",
                "content": "Please analyze this text using the analyze_content function: 'Artificial intelligence is transforming how we interact with technology through natural language processing and machine learning algorithms.'",
            },
        ]

    print("🔄 Testing comprehensive capabilities...")

    response = await client.create_completion(messages, tools=tools)

    if response.get("tool_calls"):
        print(f"✅ Tool calls generated: {len(response['tool_calls'])}")
        for tc in response["tool_calls"]:
            print(
                f"   🔧 {tc['function']['name']}: {tc['function']['arguments'][:100]}..."
            )
    else:
        print(f"ℹ️  Direct response: {response['response'][:150]}...")

    print("✅ Comprehensive test completed!")
    return response


# =============================================================================
# Main Function
# =============================================================================


async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Advantage Provider Example Script")
    parser.add_argument(
        "--model", default="global/gpt-5-chat", help="Model to use (default: global/gpt-5-chat)"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision examples"
    )
    parser.add_argument(
        "--skip-functions", action="store_true", help="Skip function calling"
    )
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")

    args = parser.parse_args()

    print("🚀 Advantage Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    api_key = os.getenv('ADVANTAGE_API_KEY') or os.getenv('OPENAI_COMPATIBLE_API_KEY')
    api_base = os.getenv('ADVANTAGE_API_BASE') or os.getenv('OPENAI_COMPATIBLE_API_BASE')
    print(f"API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"API Base: {api_base or '❌ Missing'}")

    # Show model capabilities
    try:
        config = get_config()
        supports_tools = config.supports_feature("advantage", Feature.TOOLS, args.model)
        supports_vision = config.supports_feature("advantage", Feature.VISION, args.model)
        supports_streaming = config.supports_feature("advantage", Feature.STREAMING, args.model)
        supports_json = config.supports_feature("advantage", Feature.JSON_MODE, args.model)

        print("Model capabilities:")
        print(f"  Tools: {'✅' if supports_tools else '❌'}")
        print(f"  Vision: {'✅' if supports_vision else '❌'}")
        print(f"  Streaming: {'✅' if supports_streaming else '❌'}")
        print(f"  JSON Mode: {'✅' if supports_json else '❌'}")

    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("JSON Mode", lambda: json_mode_example(args.model)),
    ]

    if not args.quick:
        if not args.skip_functions:
            examples.extend([
                ("Function Calling", lambda: function_calling_example(args.model)),
                ("Streaming Function Calling", lambda: streaming_function_calling_example(args.model)),
            ])

        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example("global/gpt-5-chat")))

        examples.extend(
            [
                ("Single Model Test", model_comparison_example),
                ("Simple Chat", lambda: simple_chat_example(args.model)),
                ("Parameters Test", lambda: parameters_example(args.model)),
                ("Comprehensive Test", lambda: comprehensive_test("global/gpt-5-chat")),
            ]
        )

    # Run examples
    results = {}
    for name, example_func in examples:
        try:
            print("\n" + "=" * 60)
            start_time = time.time()
            result = await example_func()
            duration = time.time() - start_time
            results[name] = {"success": True, "result": result, "time": duration}
            print(f"✅ {name} completed in {duration:.2f}s")
        except Exception as e:
            results[name] = {"success": False, "error": str(e), "time": 0}
            print(f"❌ {name} failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    total_time = sum(r["time"] for r in results.values())

    print(f"✅ Successful: {successful}/{total}")
    print(f"⏱️  Total time: {total_time:.2f}s")

    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['time']:.2f}s" if result["success"] else "failed"
        print(f"   {status} {name}: {time_str}")

    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("🔗 Advantage provider is working perfectly with chuk-llm!")
        print(f"✨ Features tested: {args.model} capabilities")
    else:
        print("\n⚠️  Some examples failed. Check your API key and endpoint configuration.")

        # Show configuration recommendations
        print("\n💡 Configuration Tips:")
        print("   • Set ADVANTAGE_API_KEY to your Advantage API key")
        print("   • Set ADVANTAGE_API_BASE to your Advantage endpoint URL")
        print("   • Or use OPENAI_COMPATIBLE_API_KEY and OPENAI_COMPATIBLE_API_BASE for backward compatibility")
        print("   • The Advantage provider includes automatic workarounds for:")
        print("     - System prompt injection for function calling")
        print("     - Parsing function calls from content field")
        print("     - Adding required strict parameter to tools")
        print("     - Converting to standard tool_calls format")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)