#!/usr/bin/env python3
# examples/mistral_usage_examples.py
"""
Mistral Provider Example Usage Script
====================================

Demonstrates all the features of the Mistral provider in the chuk-llm library.
Run this script to see Mistral in action with various capabilities.

Requirements:
- pip install mistralai chuk-llm
- Set MISTRAL_API_KEY environment variable

Usage:
    python mistral_example.py
    python mistral_example.py --model mistral-medium-2505
    python mistral_example.py --skip-vision
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
if not os.getenv("MISTRAL_API_KEY"):
    print("❌ Please set MISTRAL_API_KEY environment variable")
    print("   export MISTRAL_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    import httpx

    from chuk_llm.configuration import Feature, get_config
    from chuk_llm.llm.client import get_client, get_provider_info
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)


def get_available_models():
    """Get available models from configuration and optionally from API"""
    config = get_config()
    configured_models = []
    discovered_models = []

    # Get configured models
    if "mistral" in config.providers:
        provider = config.providers["mistral"]
        if hasattr(provider, "models"):
            configured_models = list(provider.models)

    # Try to get models from Mistral API
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        try:
            response = httpx.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                api_models = response.json()
                discovered_models = [
                    m.get("id") for m in api_models.get("data", []) if m.get("id")
                ]
        except Exception as e:
            print(f"⚠️  Could not fetch models from API: {e}")

    # Combine models (configured first, then discovered)
    all_models = list(configured_models)
    for model in discovered_models:
        if model not in all_models:
            all_models.append(model)

    return {
        "configured": configured_models,
        "discovered": discovered_models,
        "all": all_models,
    }


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


async def basic_text_example(model: str = "mistral-medium-2505"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_client("mistral", model=model)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms (2-3 sentences).",
        },
    ]

    start_time = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start_time

    print(f"✅ Response ({duration:.2f}s):")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 2: Streaming Response
# =============================================================================


async def streaming_example(model: str = "mistral-medium-2505"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    # Check streaming support
    config = get_config()
    if not config.supports_feature("mistral", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

    client = get_client("mistral", model=model)

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


async def function_calling_example(model: str = "mistral-medium-2505"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    # Check if model supports tools
    config = get_config()
    if not config.supports_feature("mistral", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        print(
            "💡 Try a tools-capable model like: mistral-medium-2505, mistral-large-2411, pixtral-large-2411"
        )
        return None

    client = get_client("mistral", model=model)

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_tip",
                "description": "Calculate tip amount and total bill",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bill_amount": {
                            "type": "number",
                            "description": "The bill amount in dollars",
                        },
                        "tip_percentage": {
                            "type": "number",
                            "description": "Tip percentage (default: 18)",
                        },
                    },
                    "required": ["bill_amount"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
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
    ]

    messages = [
        {
            "role": "user",
            "content": "Calculate a 20% tip on a $85 bill and tell me the weather in Paris.",
        }
    ]

    print("🔄 Making function calling request...")
    response = await client.create_completion(messages, tools=tools)

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

            if func_name == "calculate_tip":
                result = '{"tip_amount": 17.0, "total_amount": 102.0}'
            elif func_name == "get_weather":
                result = (
                    '{"temperature": "22°C", "condition": "Sunny", "humidity": "65%"}'
                )
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
# Example 4: Vision Capabilities
# =============================================================================


async def vision_example(model: str = "mistral-medium-2505"):
    """Vision capabilities with multimodal models"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)

    # Check if model supports vision
    config = get_config()
    if not config.supports_feature("mistral", Feature.VISION, model):
        print(f"⚠️  Skipping vision: Model {model} doesn't support vision")
        print(
            "💡 Try a vision-capable model like: mistral-medium-2505, pixtral-large-2411, pixtral-12b-2409"
        )
        return None

    client = get_client("mistral", model=model)

    # Create a proper test image
    print("🖼️  Creating test image...")
    test_image = create_test_image("blue", 20)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What color is this square? Answer with just the color name.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{test_image}"},
                },
            ],
        }
    ]

    print("👀 Analyzing image...")
    response = await client.create_completion(messages, max_tokens=50)

    print("✅ Vision response:")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 5: Model Discovery
# =============================================================================


async def model_discovery_example():
    """Discover available models from Mistral API"""
    print("\n🔍 Model Discovery")
    print("=" * 60)

    model_info = get_available_models()

    print(f"📦 Configured models ({len(model_info['configured'])}):")
    for model in model_info["configured"][:10]:  # Show first 10
        # Check if it's a reasoning model
        if "magistral" in model.lower():
            print(f"   • {model} [🧠 reasoning]")
        elif "codestral" in model.lower() or "devstral" in model.lower():
            print(f"   • {model} [💻 coding]")
        elif "pixtral" in model.lower():
            print(f"   • {model} [👁️ vision]")
        else:
            print(f"   • {model}")

    if len(model_info["discovered"]) > 0:
        print(f"\n🌐 Discovered from API ({len(model_info['discovered'])}):")
        # Show models that are not in config
        new_models = [
            m for m in model_info["discovered"] if m not in model_info["configured"]
        ]
        if new_models:
            print("   New models not in config:")
            for model in new_models[:5]:  # Show first 5
                print(f"   ✨ {model}")
        else:
            print("   All API models are already configured")

    print(f"\n📊 Total available: {len(model_info['all'])} models")

    # Test a discovered model if available
    if model_info["discovered"] and len(model_info["discovered"]) > 0:
        test_model = model_info["discovered"][0]
        print(f"\n🧪 Testing discovered model: {test_model}")
        try:
            client = get_client("mistral", model=test_model)
            messages = [{"role": "user", "content": "Say hello"}]
            response = await client.create_completion(messages, max_tokens=20)
            print(f"   ✅ Model works: {response['response'][:50]}...")
        except Exception as e:
            print(f"   ⚠️ Model test failed: {e}")

    return model_info


# =============================================================================
# Example 6: Reasoning Models Test
# =============================================================================


async def reasoning_example(model: str = "magistral-medium-2506"):
    """Test reasoning capabilities with Magistral models"""
    print(f"\n🧠 Reasoning Example with {model}")
    print("=" * 60)

    # Check if it's a magistral reasoning model
    if "magistral" not in model.lower():
        print(f"💡 Note: {model} is not a Magistral reasoning model")
        print("   Switching to: magistral-medium-2506 for reasoning demonstration")
        model = "magistral-medium-2506"
    else:
        print(f"✅ Using Magistral model: {model}")
        print("   This model generates <think> tags for step-by-step reasoning")

    client = get_client("mistral", model=model)

    messages = [
        {
            "role": "user",
            "content": "I have a 3-gallon jug and a 5-gallon jug. How can I measure exactly 4 gallons of water? Think step by step.",
        }
    ]

    print("🧠 Processing reasoning task...")
    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=500)
    duration = time.time() - start_time

    print(f"✅ Reasoning response ({duration:.2f}s):")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 6: Model Comparison
# =============================================================================


async def model_comparison_example():
    """Compare different Mistral models"""
    print("\n📊 Model Comparison")
    print("=" * 60)

    # Updated model list based on config
    models = [
        "mistral-medium-2505",  # Flagship multimodal model
        "mistral-large-2411",  # Top-tier reasoning
        "magistral-medium-2506",  # Reasoning specialist
        "codestral-2501",  # Latest coding model
        "ministral-8b-2410",  # Edge model
    ]

    prompt = "What is machine learning? (One sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("mistral", model=model)
            messages = [{"role": "user", "content": prompt}]

            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time

            results[model] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "success": True,
            }

        except Exception as e:
            results[model] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "success": False,
            }

    print("\n📈 Results:")
    for model, result in results.items():
        status = "✅" if result["success"] else "❌"
        model_short = (
            model.replace("mistral-", "")
            .replace("-2505", "")
            .replace("-2411", "")
            .replace("-2506", "")
            .replace("-2501", "")
            .replace("-2410", "")
        )
        print(f"   {status} {model_short}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()

    return results


# =============================================================================
# Example 7: Feature Detection
# =============================================================================


async def feature_detection_example(model: str = "mistral-medium-2505"):
    """Detect and display model features"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)

    # Get model info
    try:
        model_info = get_provider_info("mistral", model)

        print("📋 Model Information:")
        print(f"   Provider: {model_info['provider']}")
        print(f"   Model: {model_info['model']}")
        print(f"   Max Context: {model_info['max_context_length']:,} tokens")
        print(f"   Max Output: {model_info['max_output_tokens']:,} tokens")

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
        client = get_client("mistral", model=model)
        client_info = client.get_model_info()

        print("\n🔧 Client Features:")
        # Mistral supports function calling through tools
        supports_tools = (
            model_info["supports"].get("tools", False)
            if "model_info" in locals()
            else False
        )
        print(f"   Function Calling: {'✅' if supports_tools else '❌'}")
        print(f"   Vision: {'✅' if client_info.get('supports_vision') else '❌'}")
        print(
            f"   Streaming: {'✅' if client_info.get('supports_streaming') else '❌'}"
        )
        print(
            f"   System Messages: {'✅' if client_info.get('supports_system_messages') else '❌'}"
        )
        # Magistral models support reasoning
        print(
            f"   Reasoning: {'✅ (Magistral thinking model)' if 'magistral' in model.lower() else '❌'}"
        )

    except Exception as e:
        print(f"⚠️  Could not get client info: {e}")

    return model_info if "model_info" in locals() else None


# =============================================================================
# Example 8: Simple Chat Interface
# =============================================================================


async def simple_chat_example(model: str = "mistral-medium-2505"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface with {model}")
    print("=" * 60)

    client = get_client("mistral", model=model)

    # Simulate a simple conversation
    conversation = [
        "Hello! How are you?",
        "What's your favorite programming language?",
        "Can you write a simple Python function?",
    ]

    messages = [
        {"role": "system", "content": "You are a helpful and friendly AI assistant."}
    ]

    for user_input in conversation:
        print(f"👤 User: {user_input}")

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Get response
        response = await client.create_completion(messages, max_tokens=150)
        assistant_response = response.get("response", "No response")

        print(f"🤖 Assistant: {assistant_response}")
        print()

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})

    return messages


# =============================================================================
# Example 9: Comprehensive Feature Test
# =============================================================================


async def comprehensive_test(model: str = "mistral-medium-2505"):
    """Test multiple features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)

    # Check what features this model supports
    config = get_config()
    supports_tools = config.supports_feature("mistral", Feature.TOOLS, model)
    supports_vision = config.supports_feature("mistral", Feature.VISION, model)

    print(f"Model capabilities: Tools={supports_tools}, Vision={supports_vision}")

    if not supports_tools and not supports_vision:
        print("⚠️  Model doesn't support tools or vision - using text-only test")
        return await simple_chat_example(model)

    client = get_client("mistral", model=model)

    # Define tools if supported
    tools = None
    if supports_tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_content",
                    "description": "Analyze and categorize content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content_type": {"type": "string"},
                            "main_topics": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "complexity": {
                                "type": "string",
                                "enum": ["simple", "medium", "complex"],
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
                        "type": "text",
                        "text": "Please analyze this image and the following text using the analyze_content function: 'This is a test of multimodal AI capabilities.'",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{test_image}"},
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
    parser = argparse.ArgumentParser(description="Mistral Provider Example Script")
    parser.add_argument(
        "--model",
        default="mistral-medium-2505",
        help="Model to use (default: mistral-medium-2505)",
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision examples"
    )
    parser.add_argument(
        "--skip-functions", action="store_true", help="Skip function calling"
    )
    parser.add_argument(
        "--test-reasoning", action="store_true", help="Test reasoning models"
    )
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")

    args = parser.parse_args()

    print("🚀 Mistral Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('MISTRAL_API_KEY') else '❌ Missing'}")

    # Show model capabilities
    try:
        config = get_config()
        supports_tools = config.supports_feature("mistral", Feature.TOOLS, args.model)
        supports_vision = config.supports_feature("mistral", Feature.VISION, args.model)
        # Magistral models support reasoning (they generate <think> tags)
        supports_reasoning = "magistral" in args.model.lower()

        print("Model capabilities:")
        print(f"  Tools: {'✅' if supports_tools else '❌'}")
        print(f"  Vision: {'✅' if supports_vision else '❌'}")
        print(
            f"  Reasoning: {'✅ (Magistral thinking model)' if supports_reasoning else '❌'}"
        )

    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Model Discovery", model_discovery_example),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
    ]

    if not args.quick:
        if not args.skip_functions:
            examples.append(
                ("Function Calling", lambda: function_calling_example(args.model))
            )

        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example(args.model)))

        if args.test_reasoning:
            examples.append(
                ("Reasoning", lambda: reasoning_example("magistral-medium-2506"))
            )

        examples.extend(
            [
                ("Model Comparison", model_comparison_example),
                ("Simple Chat", lambda: simple_chat_example(args.model)),
                ("Comprehensive Test", lambda: comprehensive_test(args.model)),
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
        print("🔗 Mistral provider is working perfectly with chuk-llm!")
        print(f"✨ Features tested: {args.model} capabilities")
    else:
        print("\n⚠️  Some examples failed. Check your API key and model access.")

        # Show model recommendations
        print("\n💡 Model Recommendations:")
        print("   • For tools + vision: mistral-medium-2505, pixtral-large-2411")
        print("   • For reasoning: magistral-medium-2506, magistral-small-2506")
        print("   • For coding: codestral-2501, devstral-small-2505")
        print("   • For general use: mistral-large-2411")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
