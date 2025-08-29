#!/usr/bin/env python3
"""
Anthropic/Claude Provider Example Usage Script - Universal Config Version
=========================================================================

Demonstrates all the features of the Anthropic provider using the unified config system.
Tests universal vision format, JSON mode support, and system parameter handling.

Requirements:
- pip install anthropic chuk-llm pillow
- Set ANTHROPIC_API_KEY environment variable

Usage:
    python anthropic_example.py
    python anthropic_example.py --model claude-sonnet-4-20250514
    python anthropic_example.py --skip-vision
    python anthropic_example.py --test-json-mode
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time

# dotenv
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Ensure we have the required environment
if not os.getenv("ANTHROPIC_API_KEY"):
    print("❌ Please set ANTHROPIC_API_KEY environment variable")
    print("   export ANTHROPIC_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    import httpx

    from chuk_llm.configuration import CapabilityChecker, Feature, get_config
    from chuk_llm.llm.client import (
        get_client,
        get_provider_info,
        validate_provider_setup,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)


def get_available_models():
    """Get available Anthropic models from configuration and optionally from API"""
    config = get_config()
    configured_models = []
    discovered_models = []

    # Get configured models
    if "anthropic" in config.providers:
        provider = config.providers["anthropic"]
        if hasattr(provider, "models"):
            configured_models = list(provider.models)

    # Note: Anthropic doesn't have a public models endpoint like OpenAI
    # But we can check which models are available by attempting to use them
    # For now, we'll use a predefined list of known models
    known_models = [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3.5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    # Models that might be available but not configured
    for model in known_models:
        if model not in configured_models:
            discovered_models.append(model)

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


def create_test_image(color="red", size=20):
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
        # Fallback: 20x20 red square (valid PNG)
        return "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVDiN7dMxDQAhDAVQPlYgASuwAhuwAiuwAiuwAiuwAiuwgv8FJpBMJnfJfc0TDaVLkiRJkiRJkmQpY621zjl775xzSimllFJKKaWUUkoppZRSSimllFJKKe8AK0wGkZ6oONkAAAAASUVORK5CYII="


# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================


async def basic_text_example(model: str = "claude-sonnet-4-20250514"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_client("anthropic", model=model)

    messages = [
        {
            "role": "user",
            "content": "Explain the concept of recursion in programming in simple terms (2-3 sentences).",
        }
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


async def streaming_example(model: str = "claude-sonnet-4-20250514"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    # Check streaming support using unified config
    config = get_config()
    if not config.supports_feature("anthropic", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

    client = get_client("anthropic", model=model)

    messages = [
        {"role": "user", "content": "Write a short poem about the beauty of code."}
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


async def function_calling_example(model: str = "claude-sonnet-4-20250514"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    # Check if model supports tools using unified config
    config = get_config()
    if not config.supports_feature("anthropic", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        return None

    client = get_client("anthropic", model=model)

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_sentiment",
                "description": "Analyze the sentiment of a given text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze",
                        },
                        "include_confidence": {
                            "type": "boolean",
                            "description": "Whether to include confidence score",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_summary",
                "description": "Generate a summary of given text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to summarize",
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum length of summary in words",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "user",
            "content": "Please analyze the sentiment of this text: 'I absolutely love this new feature!' Then separately, please generate a summary of that same text in exactly 10 words using the generate_summary function.",
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

            if func_name == "analyze_sentiment":
                result = '{"sentiment": "positive", "confidence": 0.95, "score": 0.8}'
            elif func_name == "generate_summary":
                result = '{"summary": "User expresses strong positive sentiment about new feature."}'
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
# Example 4: Universal Vision Format
# =============================================================================


async def universal_vision_example(model: str = "claude-sonnet-4-20250514"):
    """Vision capabilities using universal image_url format"""
    print(f"\n👁️  Universal Vision Format Example with {model}")
    print("=" * 60)

    # Check if model supports vision using unified config
    config = get_config()
    if not config.supports_feature("anthropic", Feature.VISION, model):
        print(f"⚠️  Skipping vision: Model {model} doesn't support vision")
        print(
            "💡 Vision-capable models: Claude 4.x (sonnet-4, opus-4) and Claude 3.7.x"
        )

        # Suggest a vision-capable model
        vision_models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
        ]
        for suggested_model in vision_models:
            if config.supports_feature("anthropic", Feature.VISION, suggested_model):
                print(f"💡 Try: --model {suggested_model}")
                break

        return None

    client = get_client("anthropic", model=model)

    # Create a proper test image
    print("🖼️  Creating test image...")
    test_image_b64 = create_test_image("blue", 30)

    # Test universal image_url format (this should work with all providers)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What color is this square? Please describe it in one sentence.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{test_image_b64}"},
                },
            ],
        }
    ]

    print("👀 Analyzing image using universal format...")
    response = await client.create_completion(messages, max_tokens=100)

    print("✅ Vision response:")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 5: System Parameter Support
# =============================================================================


async def system_parameter_example(model: str = "claude-sonnet-4-20250514"):
    """System parameter example with different personas"""
    print(f"\n🎭 System Parameter Example with {model}")
    print("=" * 60)

    # Check system message support
    config = get_config()
    if not config.supports_feature("anthropic", Feature.SYSTEM_MESSAGES, model):
        print(f"⚠️  Model {model} doesn't support system messages")
        return None

    client = get_client("anthropic", model=model)

    # Test different personas using the system parameter
    personas = [
        {
            "name": "Helpful Assistant",
            "system": "You are a helpful, harmless, and honest AI assistant.",
            "query": "How do I bake a cake?",
        },
        {
            "name": "Pirate Captain",
            "system": "You are a friendly pirate captain. Speak like a pirate and use nautical terms.",
            "query": "How do I bake a cake?",
        },
        {
            "name": "Technical Expert",
            "system": "You are a senior software engineer with expertise in Python and system design.",
            "query": "How do I optimize a slow database query?",
        },
    ]

    for persona in personas:
        print(f"\n🎭 Testing {persona['name']} persona:")

        messages = [{"role": "user", "content": persona["query"]}]

        # Use the system parameter properly
        response = await client.create_completion(
            messages, system=persona["system"], max_tokens=150
        )
        print(f"   {response['response'][:200]}...")

    return True


# =============================================================================
# Example 6: JSON Mode Support
# =============================================================================


async def json_mode_example(model: str = "claude-sonnet-4-20250514"):
    """JSON mode example using response_format"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)

    # Check JSON mode support using unified config
    config = get_config()
    if not config.supports_feature("anthropic", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None

    client = get_client("anthropic", model=model)

    # Test JSON mode with different requests - more specific prompts
    json_tasks = [
        {
            "name": "User Profile",
            "prompt": """Create a JSON user profile with exactly these fields: name, profession, skills (array), work_style.
            User: A software developer named Alice who loves Python and works remotely.""",
            "expected_keys": ["name", "profession", "skills", "work_style"],
        },
        {
            "name": "Product Analysis",
            "prompt": """Analyze this product and return JSON with exactly these fields: product_type, features (array), price, pros (array), cons (array).
            Product: A wireless headphone with 30-hour battery life, noise cancellation, and premium sound quality for $200.""",
            "expected_keys": ["product_type", "features", "price", "pros", "cons"],
        },
        {
            "name": "Weather API",
            "prompt": """Generate a weather API response JSON with exactly these fields: location, temperature, conditions, humidity, wind.
            Location: San Francisco, Current conditions: Sunny, 72°F, 45% humidity, 5 mph wind.""",
            "expected_keys": [
                "location",
                "temperature",
                "conditions",
                "humidity",
                "wind",
            ],
        },
    ]

    for task in json_tasks:
        print(f"\n📋 {task['name']} JSON Generation:")

        messages = [{"role": "user", "content": task["prompt"]}]

        # Test using OpenAI-style response_format with explicit system instruction
        response = await client.create_completion(
            messages,
            response_format={"type": "json_object"},
            system="You must respond with valid JSON only. No markdown, no code blocks, no explanations. Just pure JSON.",
            max_tokens=300,
        )

        if response.get("response"):
            try:
                # Clean the response - remove any markdown formatting
                clean_response = response["response"].strip()
                if clean_response.startswith("```json"):
                    clean_response = (
                        clean_response.replace("```json", "").replace("```", "").strip()
                    )

                json_data = json.loads(clean_response)
                print(f"   ✅ Valid JSON with keys: {list(json_data.keys())}")

                # Check if expected keys are present
                found_keys = set(json_data.keys())
                expected_keys = set(task["expected_keys"])
                missing_keys = expected_keys - found_keys

                if missing_keys:
                    print(f"   ⚠️  Missing expected keys: {missing_keys}")
                else:
                    print("   ✅ All expected keys found")

                # Pretty print a sample
                sample_json = json.dumps(json_data, indent=2)
                if len(sample_json) > 200:
                    sample_json = sample_json[:200] + "..."
                print(f"   📄 Sample: {sample_json}")

            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON: {e}")
                print(f"   📄 Raw response: {response['response'][:200]}...")
        else:
            print("   ❌ No response received")

    return True


# =============================================================================
# Example 7: Model Comparison using Unified Config
# =============================================================================


async def model_comparison_example():
    """Compare different Claude models using unified config"""
    print("\n📊 Model Comparison")
    print("=" * 60)

    # Get all Anthropic models from unified config
    config = get_config()
    provider_config = config.get_provider("anthropic")
    models = provider_config.models[:4]  # Test top 4 models

    prompt = "What is artificial intelligence? (One sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")

            # Get model capabilities
            model_caps = provider_config.get_model_capabilities(model)
            features = [f.value for f in model_caps.features]

            client = get_client("anthropic", model=model)
            messages = [{"role": "user", "content": prompt}]

            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time

            results[model] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "features": features,
                "success": True,
            }

        except Exception as e:
            results[model] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "features": [],
                "success": False,
            }

    print("\n📈 Results:")
    for model, result in results.items():
        status = "✅" if result["success"] else "❌"
        model_short = (
            model.replace("claude-", "")
            .replace("-20241022", "")
            .replace("-20240229", "")
            .replace("-20250514", "")
            .replace("-20250219", "")
        )
        print(f"   {status} {model_short}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Features: {', '.join(result['features'][:3])}...")
        print(f"      Response: {result['response'][:80]}...")
        print()

    return results


# =============================================================================
# Example 8: Feature Detection with Universal Config
# =============================================================================


async def feature_detection_example(model: str = "claude-sonnet-4-20250514"):
    """Detect and display model features using unified config"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)

    # Get model info
    model_info = get_provider_info("anthropic", model)

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

    # Test actual client info
    client = get_client("anthropic", model=model)
    client_info = client.get_model_info()
    print("\n🔧 Client Features:")
    print(f"   Vision Format: {client_info.get('vision_format', 'standard')}")
    print(f"   JSON Mode: {'✅' if client_info.get('supports_json_mode') else '❌'}")
    print(
        f"   System Messages: {'✅' if client_info.get('supports_system_messages') else '❌'}"
    )

    return model_info


# =============================================================================
# Example 9: Model Discovery
# =============================================================================


async def model_discovery_example():
    """Discover available Anthropic models"""
    print("\n🔍 Model Discovery")
    print("=" * 60)

    model_info = get_available_models()

    print(f"📦 Configured models ({len(model_info['configured'])}):")
    for model in model_info["configured"][:10]:  # Show first 10
        # Identify model capabilities
        if "opus-4-1" in model:
            print(f"   • {model} [🎭 Opus 4.1 - Latest flagship]")
        elif "opus-4" in model:
            print(f"   • {model} [🎭 Opus 4 - Flagship model]")
        elif "sonnet-4" in model:
            print(f"   • {model} [🎵 Sonnet 4 - Balanced performance]")
        elif "3-7-sonnet" in model:
            print(f"   • {model} [🎵 Sonnet 3.7 - Fast & capable]")
        elif "haiku" in model:
            print(f"   • {model} [🍃 Haiku - Fast & efficient]")
        elif "instant" in model:
            print(f"   • {model} [⚡ Instant - Ultra-fast]")
        else:
            print(f"   • {model}")

    if len(model_info["discovered"]) > 0:
        print(f"\n🌐 Known models not in config ({len(model_info['discovered'])}):")
        # Show models that are not in config
        for model in model_info["discovered"][:5]:  # Show first 5
            if "opus" in model.lower() or "sonnet" in model.lower():
                print(f"   ○ {model} [may require API access]")
            else:
                print(f"   ○ {model}")

    print(f"\n📊 Total known: {len(model_info['all'])} models")

    # Show model families
    print("\n🌟 Model Families:")
    opus_models = [m for m in model_info["all"] if "opus" in m.lower()]
    sonnet_models = [m for m in model_info["all"] if "sonnet" in m.lower()]
    haiku_models = [m for m in model_info["all"] if "haiku" in m.lower()]

    if opus_models:
        print(f"   🎭 Opus (Flagship): {len(opus_models)} models")
    if sonnet_models:
        print(f"   🎵 Sonnet (Balanced): {len(sonnet_models)} models")
    if haiku_models:
        print(f"   🍃 Haiku (Fast): {len(haiku_models)} models")

    # Test a configured model if available
    if model_info["configured"]:
        test_model = model_info["configured"][0]
        print(f"\n🧪 Testing model: {test_model}")
        try:
            client = get_client("anthropic", model=test_model)
            messages = [{"role": "user", "content": "Say hello in 3 words"}]
            response = await client.create_completion(messages, max_tokens=20)
            content = response.get("response", "")
            if content:
                print(f"   ✅ Model works: {content[:50]}...")
            else:
                print("   ⚠️ Empty response")
        except Exception as e:
            print(f"   ⚠️ Model test failed: {e}")

    return model_info


# =============================================================================
# Example 10: Comprehensive Feature Test
# =============================================================================


async def comprehensive_feature_test(model: str = "claude-sonnet-4-20250514"):
    """Test all features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)

    # Check if model supports vision first
    config = get_config()
    if not config.supports_feature("anthropic", Feature.VISION, model):
        print(
            f"⚠️  Model {model} doesn't support vision - using text-only comprehensive test"
        )
        return await comprehensive_text_only_test(model)

    client = get_client("anthropic", model=model)

    # Create a test image
    print("🖼️  Creating test image...")
    test_image_b64 = create_test_image("green", 25)

    # Test: System message + Vision + Tools + JSON mode
    tools = [
        {
            "type": "function",
            "function": {
                "name": "image_analysis_result",
                "description": "Store the structured result of image analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_type": {"type": "string"},
                        "dominant_colors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "dimensions": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["image_type", "dominant_colors", "description"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze this image and use the image_analysis_result function to store your findings in a structured format.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{test_image_b64}"},
                },
            ],
        }
    ]

    print("🔄 Testing: System + Vision + Tools...")

    # Test with all features combined
    response = await client.create_completion(
        messages,
        tools=tools,
        system="You are an expert image analyst. Always use the provided function to structure your results.",
        max_tokens=300,
    )

    if response.get("tool_calls"):
        print(f"✅ Tool calls generated: {len(response['tool_calls'])}")
        for tc in response["tool_calls"]:
            print(
                f"   🔧 {tc['function']['name']}: {tc['function']['arguments'][:100]}..."
            )

        # Simulate tool execution
        messages.append({"role": "assistant", "tool_calls": response["tool_calls"]})

        # Add tool result
        for tc in response["tool_calls"]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": '{"status": "stored", "analysis_id": "test_123"}',
                }
            )

        # Get final response
        final_response = await client.create_completion(messages)
        print(f"✅ Final analysis: {final_response['response'][:150]}...")

    else:
        print(f"ℹ️  Direct response: {response['response'][:150]}...")

    print("✅ Comprehensive test completed!")
    return response


async def comprehensive_text_only_test(model: str):
    """Comprehensive test without vision for non-vision models"""
    client = get_client("anthropic", model=model)

    # Test: System message + Tools + JSON mode
    tools = [
        {
            "type": "function",
            "function": {
                "name": "text_analysis_result",
                "description": "Store the structured result of text analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string"},
                        "key_topics": {"type": "array", "items": {"type": "string"}},
                        "word_count": {"type": "number"},
                        "summary": {"type": "string"},
                    },
                    "required": ["sentiment", "key_topics", "summary"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "Please analyze this text and use the text_analysis_result function: 'I absolutely love working with Python and machine learning. The ecosystem is fantastic!'",
        }
    ]

    print("🔄 Testing: System + Tools + JSON (text-only)...")

    response = await client.create_completion(
        messages,
        tools=tools,
        system="You are an expert text analyst. Always use the provided function to structure your results.",
        max_tokens=300,
    )

    if response.get("tool_calls"):
        print(f"✅ Tool calls generated: {len(response['tool_calls'])}")
        for tc in response["tool_calls"]:
            print(
                f"   🔧 {tc['function']['name']}: {tc['function']['arguments'][:100]}..."
            )
    else:
        print(f"ℹ️  Direct response: {response['response'][:150]}...")

    return response


# =============================================================================
# Main Function
# =============================================================================


async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(
        description="Anthropic/Claude Provider Example Script - Universal Config"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision examples"
    )
    parser.add_argument(
        "--skip-functions", action="store_true", help="Skip function calling"
    )
    parser.add_argument(
        "--test-json-mode", action="store_true", help="Focus on JSON mode testing"
    )
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    parser.add_argument(
        "--comprehensive", action="store_true", help="Run comprehensive feature test"
    )

    args = parser.parse_args()

    print("🚀 Anthropic/Claude Provider Examples (Universal Config v3)")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Missing'}")

    # Show config info
    try:
        config = get_config()
        provider_config = config.get_provider("anthropic")
        print(f"Available models: {len(provider_config.models)}")
        print(
            f"Baseline features: {', '.join(f.value for f in provider_config.features)}"
        )

        # Check if the selected model supports vision
        if config.supports_feature("anthropic", Feature.VISION, args.model):
            print(f"✅ Model {args.model} supports vision")
        else:
            print(
                f"⚠️  Model {args.model} doesn't support vision - vision tests will be skipped"
            )
            if not args.skip_vision:
                vision_models = [
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-3-7-sonnet-20250219",
                ]
                print(f"💡 For vision tests, try: {', '.join(vision_models)}")

    except Exception as e:
        print(f"⚠️  Config warning: {e}")

    # Run comprehensive test if requested
    if args.comprehensive:
        await comprehensive_feature_test(args.model)
        return

    # Focus on JSON mode if requested
    if args.test_json_mode:
        await json_mode_example(args.model)
        return

    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Model Discovery", model_discovery_example),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("System Parameter", lambda: system_parameter_example(args.model)),
        ("JSON Mode", lambda: json_mode_example(args.model)),
    ]

    if not args.quick:
        if not args.skip_functions:
            examples.append(
                ("Function Calling", lambda: function_calling_example(args.model))
            )

        if not args.skip_vision:
            examples.append(
                ("Universal Vision", lambda: universal_vision_example(args.model))
            )

        examples.extend(
            [
                ("Model Comparison", model_comparison_example),
                ("Comprehensive Test", lambda: comprehensive_feature_test(args.model)),
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
        print(
            "🔗 Anthropic/Claude provider is working perfectly with universal config!"
        )
        print(
            "✨ Features tested: System params, JSON mode, Universal vision, Tools, Streaming"
        )
    else:
        print("\n⚠️  Some examples failed. Check your API key and model access.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
