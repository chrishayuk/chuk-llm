#!/usr/bin/env python3
# examples/openrouter_usage_examples.py
"""
OpenRouter Provider Example Usage Script
========================================

Demonstrates all the features of the OpenRouter provider in the chuk-llm library.
Run this script to see various models from different providers in action.

Requirements:
- pip install chuk-llm
- Set OPENROUTER_API_KEY environment variable

Usage:
    python openrouter_usage_examples.py
    python openrouter_usage_examples.py --model anthropic/claude-3-sonnet
    python openrouter_usage_examples.py --skip-vision
    python openrouter_usage_examples.py --list-models
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
if not os.getenv("OPENROUTER_API_KEY"):
    print("❌ Please set OPENROUTER_API_KEY environment variable")
    print("   export OPENROUTER_API_KEY='your_api_key_here'")
    print("   Get your key at: https://openrouter.ai/keys")
    sys.exit(1)

try:
    import httpx

    from chuk_llm.configuration import Feature, get_config
    from chuk_llm.llm.client import get_client, get_provider_info
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)


async def get_available_models():
    """Get available models using discovery system"""
    config = get_config()
    configured_models = []
    discovered_models = []

    # Get configured models
    if "openrouter" in config.providers:
        provider = config.providers["openrouter"]
        if hasattr(provider, "models"):
            configured_models = [m for m in provider.models if m != "*"]

    # Use discovery system to get models from API
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            from chuk_llm.llm.discovery.general_discoverers import (
                OpenAICompatibleDiscoverer,
            )

            discoverer = OpenAICompatibleDiscoverer(
                provider_name="openrouter",
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
            )
            models_data = await discoverer.discover_models()
            discovered_models = [m.get("name") for m in models_data]
        except Exception as e:
            print(f"Warning: Could not fetch models from API: {e}")

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
        return "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVCiRY2RgYGBkYGBgZGBgYGRkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGAAAgAANgAOAUUe1wAAAABJRU5ErkJggg=="


# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================


async def basic_text_example(model: str = "openai/gpt-3.5-turbo"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_client("openrouter", model=model)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Explain neural networks in simple terms (2-3 sentences).",
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


async def streaming_example(model: str = "openai/gpt-3.5-turbo"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    # Check streaming support
    config = get_config()
    if not config.supports_feature("openrouter", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

    client = get_client("openrouter", model=model)

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


async def function_calling_example(model: str = "openai/gpt-4"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    # Check if model supports tools
    config = get_config()
    if not config.supports_feature("openrouter", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        return None

    client = get_client("openrouter", model=model)

    # Define tools
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
    ]

    messages = [
        {
            "role": "user",
            "content": "Search for 'latest AI research' and calculate 25.5 * 14.2 with 3 decimal places.",
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

            if func_name == "search_web":
                result = '{"results": ["Latest breakthrough in transformer models", "New multimodal AI research", "Advances in reasoning capabilities"]}'
            elif func_name == "calculate_math":
                result = (
                    '{"result": 361.100, "expression": "25.5 * 14.2", "precision": 3}'
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


async def vision_example(model: str = "openai/gpt-4-turbo"):
    """Vision capabilities with models that support it"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)

    # Check if model supports vision
    config = get_config()
    if not config.supports_feature("openrouter", Feature.VISION, model):
        print(f"⚠️  Skipping vision: Model {model} doesn't support vision")
        print(
            "💡 Try a vision-capable model like: anthropic/claude-3.5-sonnet, openai/gpt-4o"
        )
        return None

    client = get_client("openrouter", model=model)

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
# Example 5: JSON Mode
# =============================================================================


async def json_mode_example(model: str = "openai/gpt-4-turbo"):
    """JSON mode example with structured output"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)

    # Check JSON mode support
    config = get_config()
    if not config.supports_feature("openrouter", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None

    client = get_client("openrouter", model=model)

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

        try:
            parsed = json.loads(response["response"])
            print(f"✅ Valid JSON structure with keys: {list(parsed.keys())}")
        except json.JSONDecodeError:
            print("⚠️  Response is not valid JSON")

    except Exception as e:
        print(f"❌ JSON mode failed: {e}")
        # Fallback to regular request
        response = await client.create_completion(messages)
        print(f"📝 Fallback response: {response['response'][:200]}...")

    return response


# =============================================================================
# Example 6: Multi-Provider Comparison
# =============================================================================


async def multi_provider_comparison():
    """Compare different models from different providers via OpenRouter"""
    print("\n🌐 Multi-Provider Comparison via OpenRouter")
    print("=" * 60)

    # Sample of models from different providers - using more available models
    models = [
        "openai/gpt-3.5-turbo",  # OpenAI
        "openai/gpt-4-turbo",  # OpenAI Premium
        "meta-llama/llama-3.1-8b-instruct",  # Meta (often available)
        "mistralai/mistral-7b-instruct:free",  # Mistral Free
        "google/gemini-flash-1.5",  # Google
    ]

    prompt = "What is machine learning? (One sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("openrouter", model=model)
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
        provider = model.split("/")[0]
        model_name = model.split("/")[1]
        print(f"   {status} {provider}/{model_name[:20]}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()

    return results


# =============================================================================
# Example 7: Discovered Models Test
# =============================================================================


async def discovered_models_example():
    """Test models that are discovered but not in configuration"""
    print("\n🌐 Testing Discovered Models (Not in Config)")
    print("=" * 60)

    model_info = await get_available_models()

    # Find models that are only discovered, not configured
    only_discovered = [
        m for m in model_info["discovered"] if m not in model_info["configured"]
    ][:5]  # Test first 5

    if not only_discovered:
        print("⚠️  No discovered-only models found")
        # Try with any discovered model
        if model_info["discovered"]:
            print("Testing with first discovered model instead...")
            only_discovered = model_info["discovered"][:1]
        else:
            return None

    print(f"Found {len(model_info['discovered'])} total discovered models")
    print(f"Testing {len(only_discovered)} models:\n")

    for model in only_discovered:
        try:
            print(f"🔄 Testing {model}:")
            client = get_client("openrouter", model=model)

            messages = [{"role": "user", "content": "Say hello in 5 words or less"}]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=20)
            duration = time.time() - start_time

            print(f"   ✅ Response: {response['response'][:50]}")
            print(f"   Time: {duration:.2f}s\n")

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print("   ❌ Model not available for this account\n")
            elif "402" in error_msg:
                print("   💳 Requires payment/credits\n")
            else:
                print(f"   ❌ Error: {error_msg[:100]}...\n")

    return True


# =============================================================================
# Example 8: Cost-Aware Routing
# =============================================================================


async def cost_aware_example():
    """Demonstrate cost-aware model selection"""
    print("\n💰 Cost-Aware Model Selection")
    print("=" * 60)

    # Models with different cost tiers - using more available options
    models = [
        ("openai/gpt-4-turbo", "Premium"),
        ("openai/gpt-3.5-turbo", "Balanced"),
        ("mistralai/mistral-7b-instruct:free", "Free Tier"),
        ("meta-llama/llama-3.1-8b-instruct", "Cost Effective"),
    ]

    prompt = (
        "Generate a creative business name for a tech startup (one suggestion only)"
    )

    for model, tier in models:
        try:
            print(f"\n💎 {tier} - {model}:")
            client = get_client("openrouter", model=model)

            messages = [{"role": "user", "content": prompt}]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=50)
            duration = time.time() - start_time

            print(f"   Response: {response['response']}")
            print(f"   Time: {duration:.2f}s")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return True


# =============================================================================
# Example 9: Feature Detection
# =============================================================================


async def feature_detection_example(model: str = "openai/gpt-3.5-turbo"):
    """Detect and display model features"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)

    # Get model info
    try:
        model_info = get_provider_info("openrouter", model)

        print("📋 Model Information:")
        print("   Provider: OpenRouter")
        print(f"   Model: {model_info['model']}")
        print(f"   Max Context: {model_info.get('max_context_length', 'N/A'):,} tokens")
        print(f"   Max Output: {model_info.get('max_output_tokens', 'N/A'):,} tokens")

        print("\n🎯 Supported Features:")
        supports = model_info.get("supports", {})
        for feature, supported in supports.items():
            status = "✅" if supported else "❌"
            print(f"   {status} {feature}")

    except Exception as e:
        print(f"⚠️  Could not get model info: {e}")

    # Test actual client info
    try:
        client = get_client("openrouter", model=model)
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
# Example 10: Simple Chat Interface
# =============================================================================


async def simple_chat_example(model: str = "openai/gpt-3.5-turbo"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface with {model}")
    print("=" * 60)

    client = get_client("openrouter", model=model)

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
# Example 11: Temperature and Parameters Test
# =============================================================================


async def parameters_example(model: str = "openai/gpt-3.5-turbo"):
    """Test different parameters and settings"""
    print(f"\n🎛️  Parameters Test with {model}")
    print("=" * 60)

    client = get_client("openrouter", model=model)

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
# Example 12: Specialized Models Test
# =============================================================================


async def specialized_models_example():
    """Test specialized models available on OpenRouter"""
    print("\n🎯 Specialized Models Test")
    print("=" * 60)

    # Test different specialized models
    tests = [
        {
            "model": "openai/gpt-4-turbo",
            "type": "Vision & Tools",
            "prompt": "Describe the capabilities of modern AI assistants.",
        },
        {
            "model": "openai/gpt-3.5-turbo",
            "type": "Fast Response",
            "prompt": "Explain the difference between correlation and causation with an example.",
        },
        {
            "model": "mistralai/mistral-7b-instruct:free",
            "type": "Free Tier",
            "prompt": "Write a Python function to calculate fibonacci numbers efficiently.",
        },
        {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "type": "Open Source",
            "prompt": "List three advantages of using open source AI models.",
        },
    ]

    for test in tests:
        print(f"\n🔍 Testing {test['type']} - {test['model']}:")
        try:
            client = get_client("openrouter", model=test["model"])
            messages = [{"role": "user", "content": test["prompt"]}]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=150)
            duration = time.time() - start_time

            print(f"   Response: {response['response'][:200]}...")
            print(f"   Time: {duration:.2f}s")

        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")

    return True


# =============================================================================
# Example 13: Model Comparison
# =============================================================================


async def model_comparison_example():
    """Compare different OpenRouter models"""
    print("\n📊 Model Comparison")
    print("=" * 60)

    # Compare popular models across providers
    models = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.3-70b-instruct",
    ]

    prompt = "What is the future of AI? (One sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("openrouter", model=model)
            messages = [Message(role=MessageRole.USER, content=prompt)]

            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=100)
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
        model_short = model.split("/")[-1]
        print(f"   {status} {model_short}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()

    return results


# =============================================================================
# Example 14: Context Window Test
# =============================================================================


async def context_window_test(model: str = "meta-llama/llama-3.3-70b-instruct"):
    """Test OpenRouter's large context window support"""
    print(f"\n📏 Context Window Test with {model}")
    print("=" * 60)

    client = get_client("openrouter", model=model)

    # Create a long context (~4500 words)
    long_text = "The quick brown fox jumps over the lazy dog. " * 500

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=f"You have been given a long text. Here it is:\n\n{long_text}\n\nPlease analyze this text.",
        ),
        Message(
            role=MessageRole.USER,
            content="How many times does the word 'fox' appear in the text? Also tell me the total word count.",
        ),
    ]

    print(f"📝 Testing with ~{len(long_text.split())} words of context...")

    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=150)
    duration = time.time() - start_time

    print(f"✅ Response ({duration:.2f}s):")
    print(f"   {response.get('response', '')}")

    return response


# =============================================================================
# Example 15: Parallel Processing Test
# =============================================================================


async def parallel_processing_test(model: str = "meta-llama/llama-3.3-70b-instruct"):
    """Test parallel request processing with OpenRouter"""
    print("\n🔀 Parallel Processing Test")
    print("=" * 60)

    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing.",
        "What is machine learning?",
        "Define neural networks.",
        "What is deep learning?",
    ]

    print(f"📊 Testing {len(prompts)} parallel requests with {model}...")

    # Sequential processing
    print("\n📝 Sequential processing:")
    sequential_start = time.time()

    for prompt in prompts:
        client = get_client("openrouter", model=model)
        await client.create_completion(
            [Message(role=MessageRole.USER, content=prompt)], max_tokens=50
        )

    sequential_time = time.time() - sequential_start
    print(f"   ✅ Completed in {sequential_time:.2f}s")

    # Parallel processing
    print("\n⚡ Parallel processing:")
    parallel_start = time.time()

    async def process_prompt(prompt):
        client = get_client("openrouter", model=model)
        response = await client.create_completion(
            [Message(role=MessageRole.USER, content=prompt)], max_tokens=50
        )
        return response.get("response", "")[:50]

    await asyncio.gather(*[process_prompt(p) for p in prompts])
    parallel_time = time.time() - parallel_start
    print(f"   ✅ Completed in {parallel_time:.2f}s")

    # Results
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print("\n📈 Results:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Parallel: {parallel_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x")

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
    }


# =============================================================================
# Example 16: Dynamic Model Test
# =============================================================================


async def dynamic_model_test():
    """Test a non-configured model to prove library flexibility"""
    print("\n🔄 Dynamic Model Test")
    print("=" * 60)
    print("Testing a model NOT in chuk_llm.yaml config")

    # Use a model specific to this provider that might not be in config
    dynamic_model = "meta-llama/llama-3.3-70b-instruct"

    print(f"\n🧪 Testing dynamic model: {dynamic_model}")
    print("   This model may not be in the config file")

    try:
        client = get_client("openrouter", model=dynamic_model)
        messages = [
            {
                "role": "user",
                "content": "Say hello in exactly one creative word"
            }
        ]

        response = await client.create_completion(messages, max_tokens=10)
        print(f"   ✅ Dynamic model works: {response['response']}")

        return response

    except Exception as e:
        print(f"   ⚠️ Test failed: {str(e)[:100]}")
        return None


# =============================================================================
# Example 17: Comprehensive Feature Test
# =============================================================================


async def comprehensive_test(model: str = "openai/gpt-4"):
    """Test multiple features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)

    # Check what features this model supports
    config = get_config()
    supports_tools = config.supports_feature("openrouter", Feature.TOOLS, model)
    supports_vision = config.supports_feature("openrouter", Feature.VISION, model)

    print(f"Model capabilities: Tools={supports_tools}, Vision={supports_vision}")

    if not supports_tools and not supports_vision:
        print("⚠️  Model doesn't support tools or vision - using text-only test")
        return await simple_chat_example(model)

    client = get_client("openrouter", model=model)

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
    parser = argparse.ArgumentParser(description="OpenRouter Provider Example Script")
    parser.add_argument(
        "--model",
        default="openai/gpt-3.5-turbo",
        help="Model to use (default: openai/gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision examples"
    )
    parser.add_argument(
        "--skip-functions", action="store_true", help="Skip function calling"
    )
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("📋 Available OpenRouter Models")
        print("=" * 60)
        model_info = await get_available_models()

        print(f"\n📦 Configured Models ({len(model_info['configured'])}):")
        for model in model_info["configured"][:10]:
            print(f"  - {model}")
        if len(model_info["configured"]) > 10:
            print(f"  ... and {len(model_info['configured']) - 10} more")

        print(f"\n🌐 Discovered Models ({len(model_info['discovered'])}):")
        # Show some popular discovered models
        popular_discovered = [
            m
            for m in model_info["discovered"]
            if any(p in m for p in ["gpt", "claude", "llama", "mistral", "gemini"])
        ][:10]
        for model in popular_discovered:
            status = "✅" if model not in model_info["configured"] else "🔄"
            print(f"  {status} {model}")
        if len(model_info["discovered"]) > 10:
            print(f"  ... and {len(model_info['discovered']) - 10} more")

        print(f"\n💡 Total Available: {len(model_info['all'])} models")
        print("\n✅ = Only in discovery, 🔄 = Also in config")
        return

    print("🚀 OpenRouter Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('OPENROUTER_API_KEY') else '❌ Missing'}")
    print("Website: https://openrouter.ai")

    # Show model capabilities
    try:
        config = get_config()
        supports_tools = config.supports_feature(
            "openrouter", Feature.TOOLS, args.model
        )
        supports_vision = config.supports_feature(
            "openrouter", Feature.VISION, args.model
        )
        supports_streaming = config.supports_feature(
            "openrouter", Feature.STREAMING, args.model
        )
        supports_json = config.supports_feature(
            "openrouter", Feature.JSON_MODE, args.model
        )

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
    ]

    if not args.quick:
        if not args.skip_functions:
            examples.append(
                ("Function Calling", lambda: function_calling_example("openai/gpt-4"))
            )

        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example("openai/gpt-4-turbo")))

        examples.extend(
            [
                ("JSON Mode", lambda: json_mode_example("openai/gpt-4-turbo")),
                ("Discovered Models", discovered_models_example),
                ("Multi-Provider Comparison", multi_provider_comparison),
                ("Model Comparison", model_comparison_example),
                ("Context Window Test", lambda: context_window_test("meta-llama/llama-3.3-70b-instruct")),
                ("Parallel Processing", lambda: parallel_processing_test("meta-llama/llama-3.3-70b-instruct")),
                ("Dynamic Model Test", dynamic_model_test),
                ("Cost-Aware Routing", cost_aware_example),
                ("Simple Chat", lambda: simple_chat_example(args.model)),
                ("Parameters Test", lambda: parameters_example(args.model)),
                ("Specialized Models", specialized_models_example),
                ("Comprehensive Test", lambda: comprehensive_test("openai/gpt-4")),
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
        print("🔗 OpenRouter provider is working perfectly with chuk-llm!")
        print(
            "✨ You have access to models from: OpenAI, Anthropic, Google, Meta, and more!"
        )
    else:
        print("\n⚠️  Some examples failed. Check your API key and model access.")

        # Show model recommendations
        print("\n💡 Popular OpenRouter Models:")
        print("   • For general use: openai/gpt-3.5-turbo, openai/gpt-4-turbo")
        print("   • For vision: openai/gpt-4-turbo, google/gemini-flash-1.5")
        print(
            "   • For free tier: mistralai/mistral-7b-instruct:free, meta-llama/llama-3.1-8b-instruct:free"
        )
        print("   • For coding: openai/gpt-4-turbo, deepseek/deepseek-coder")
        print("   • Check available models with: --list-models")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
