#!/usr/bin/env python3
# examples/groq_usage_examples.py
"""
Groq Provider Example Usage Script
===================================

Demonstrates all the features of the Groq provider in the chuk-llm library.
Groq is known for ultra-fast inference speeds with their LPU (Language Processing Unit) technology.

Requirements:
- pip install chuk-llm
- Set GROQ_API_KEY environment variable

Usage:
    python groq_usage_examples.py
    python groq_usage_examples.py --model llama-3.1-8b-instant
    python groq_usage_examples.py --benchmark
    python groq_usage_examples.py --skip-tools
"""

import argparse
import asyncio
import json
import os
import sys
import time

# dotenv
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Ensure we have the required environment
if not os.getenv("GROQ_API_KEY"):
    print("❌ Please set GROQ_API_KEY environment variable")
    print("   export GROQ_API_KEY='your_api_key_here'")
    print("   Get your key at: https://console.groq.com/keys")
    sys.exit(1)

try:
    import httpx

    from chuk_llm.configuration import Feature, get_config
    from chuk_llm.llm.client import get_client, get_provider_info
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)


def get_available_groq_models():
    """Get available Groq models from config and API"""
    config = get_config()
    configured_models = []
    api_models = []

    # Get configured models
    if "groq" in config.providers:
        provider = config.providers["groq"]
        if hasattr(provider, "models"):
            configured_models = provider.models

    # Try to get models from API
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            response = httpx.get(
                "https://api.groq.com/openai/v1/models", headers=headers, timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                api_models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception:
            pass  # Silently ignore API errors

    return {
        "configured": configured_models,
        "api": api_models,
        "all": list(set(configured_models + api_models)),
    }


# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================


async def basic_text_example(model: str = "llama-3.3-70b-versatile"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_client("groq", model=model)

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

    # Calculate tokens per second (approximate)
    response_length = len(response["response"].split())
    tokens_per_sec = response_length / duration if duration > 0 else 0
    print(f"⚡ Speed: ~{tokens_per_sec:.1f} tokens/sec")

    return response


# =============================================================================
# Example 2: Streaming Response
# =============================================================================


async def streaming_example(model: str = "llama-3.1-8b-instant"):
    """Real-time streaming example - Groq streams VERY fast!"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    # Check streaming support
    config = get_config()
    if not config.supports_feature("groq", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

    client = get_client("groq", model=model)

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
    chunk_count = 0

    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            content = chunk["response"]
            print(content, end="", flush=True)
            full_response += content
            chunk_count += 1

    duration = time.time() - start_time
    print(f"\n✅ Streaming completed ({duration:.2f}s)")
    print(
        f"⚡ Received {chunk_count} chunks at {chunk_count / duration:.1f} chunks/sec"
    )

    return full_response


# =============================================================================
# Example 3: Speed Benchmark
# =============================================================================


async def speed_benchmark():
    """Benchmark Groq's ultra-fast inference speed"""
    print("\n🏎️  Speed Benchmark - Groq's LPU Technology")
    print("=" * 60)

    models = [
        "llama-3.1-8b-instant",  # Fastest model
        "llama-3.3-70b-versatile",  # Larger model
    ]

    prompt = (
        "Generate a list of 5 creative uses for artificial intelligence in education."
    )

    results = {}

    for model in models:
        try:
            print(f"\n🔄 Testing {model}...")
            client = get_client("groq", model=model)
            messages = [{"role": "user", "content": prompt}]

            # Warm-up request
            await client.create_completion(messages, max_tokens=10)

            # Timed requests
            times = []
            token_counts = []

            for i in range(3):
                start_time = time.time()
                response = await client.create_completion(messages, max_tokens=200)
                duration = time.time() - start_time
                times.append(duration)

                # Approximate token count
                tokens = len(response.get("response", "").split())
                token_counts.append(tokens)

                print(f"   Run {i + 1}: {duration:.3f}s, ~{tokens} tokens")

            avg_time = sum(times) / len(times)
            avg_tokens = sum(token_counts) / len(token_counts)
            tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

            results[model] = {
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "tokens_per_sec": tokens_per_sec,
                "success": True,
            }

            print(f"   ✅ Average: {avg_time:.3f}s, ~{tokens_per_sec:.1f} tokens/sec")

        except Exception as e:
            results[model] = {"error": str(e), "success": False}
            print(f"   ❌ Error: {e}")

    # Summary
    print("\n📊 Benchmark Summary:")
    for model, result in results.items():
        if result["success"]:
            print(f"   {model}: {result['tokens_per_sec']:.1f} tokens/sec")
        else:
            print(f"   {model}: Failed")

    return results


# =============================================================================
# Example 4: Function Calling / Tools
# =============================================================================


async def function_calling_example(model: str = "llama-3.3-70b-versatile"):
    """Function calling with tools - Groq supports OpenAI-compatible tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    # Check if model supports tools
    config = get_config()
    if not config.supports_feature("groq", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        return None

    client = get_client("groq", model=model)

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_speed",
                "description": "Calculate speed given distance and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "distance": {
                            "type": "number",
                            "description": "Distance traveled (in meters)",
                        },
                        "time": {
                            "type": "number",
                            "description": "Time taken (in seconds)",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["m/s", "km/h", "mph"],
                            "description": "Desired output unit",
                        },
                    },
                    "required": ["distance", "time"],
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
                        "location": {
                            "type": "string",
                            "description": "City name or coordinates",
                        },
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
            "content": "If I travel 1000 meters in 50 seconds, what's my speed in km/h? Also, what's the weather like in Tokyo?",
        }
    ]

    print("🔄 Making function calling request...")
    start_time = time.time()
    response = await client.create_completion(messages, tools=tools)
    duration = time.time() - start_time

    if response.get("tool_calls"):
        print(f"✅ Tool calls requested in {duration:.2f}s:")
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

            if func_name == "calculate_speed":
                result = '{"speed": 72, "unit": "km/h", "calculation": "1000m / 50s = 20 m/s = 72 km/h"}'
            elif func_name == "get_weather":
                result = '{"location": "Tokyo", "temperature": 22, "unit": "celsius", "condition": "partly cloudy"}'
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
# Example 5: JSON Mode
# =============================================================================


async def json_mode_example(model: str = "llama-3.3-70b-versatile"):
    """JSON mode example with structured output"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)

    # Check JSON mode support
    config = get_config()
    if not config.supports_feature("groq", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None

    client = get_client("groq", model=model)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant designed to output JSON. Generate a JSON object with information about Groq's LPU technology.",
        },
        {
            "role": "user",
            "content": "Tell me about Groq's Language Processing Unit (LPU) in JSON format with fields: name, type, main_advantage, speed_comparison, use_cases (array), and innovation_score (1-10).",
        },
    ]

    print("📝 Requesting JSON output...")

    try:
        start_time = time.time()
        response = await client.create_completion(
            messages, response_format={"type": "json_object"}, temperature=0.7
        )
        duration = time.time() - start_time

        print(f"✅ JSON response ({duration:.2f}s):")
        print(f"   {response['response']}")

        # Try to parse as JSON to verify
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
# Example 6: Model Comparison
# =============================================================================


async def model_comparison_example():
    """Compare different Groq models"""
    print("\n📊 Groq Model Comparison")
    print("=" * 60)

    # Get available models from config
    from chuk_llm.configuration import get_config

    config = get_config()
    available_models = []

    if "groq" in config.providers:
        provider = config.providers["groq"]
        if hasattr(provider, "models"):
            available_models = provider.models

    # Use configured models or defaults
    if len(available_models) >= 2:
        models = available_models[:3]  # Use first 3 configured models
    else:
        models = [
            "llama-3.1-8b-instant",  # Fastest
            "llama-3.3-70b-versatile",  # Most capable
        ]

    print(f"Testing {len(models)} models: {', '.join(models)}")

    prompt = "What is the future of AI? (One sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("groq", model=model)
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
        print(f"   {status} {model}:")
        print(f"      Time: {result['time']:.2f}s")
        print(
            f"      Speed: {result['length'] / result['time']:.1f} chars/sec"
            if result["time"] > 0
            else "      Speed: N/A"
        )
        print(f"      Response: {result['response'][:100]}...")
        print()

    return results


# =============================================================================
# Example 7: Feature Detection
# =============================================================================


async def feature_detection_example(model: str = "llama-3.3-70b-versatile"):
    """Detect and display model features"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)

    # Get model info
    try:
        model_info = get_provider_info("groq", model)

        print("📋 Model Information:")
        print("   Provider: Groq")
        print(f"   Model: {model_info['model']}")
        print(
            f"   Max Context: {model_info.get('max_context_length', 131072):,} tokens"
        )
        print(f"   Max Output: {model_info.get('max_output_tokens', 32768):,} tokens")

        print("\n🎯 Supported Features:")
        supports = model_info.get("supports", {})
        for feature, supported in supports.items():
            status = "✅" if supported else "❌"
            print(f"   {status} {feature}")

    except Exception as e:
        print(f"⚠️  Could not get model info: {e}")

    # Test actual client info
    try:
        client = get_client("groq", model=model)
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
# Example 8: Context Window Test
# =============================================================================


async def context_window_test(model: str = "llama-3.3-70b-versatile"):
    """Test Groq's large context window (131k tokens!)"""
    print(f"\n📏 Context Window Test with {model}")
    print("=" * 60)

    client = get_client("groq", model=model)

    # Create a long context
    long_text = "The quick brown fox jumps over the lazy dog. " * 500  # ~4500 words

    messages = [
        {
            "role": "system",
            "content": f"You have been given a long text. Here it is:\n\n{long_text}\n\nPlease analyze this text.",
        },
        {
            "role": "user",
            "content": "How many times does the word 'fox' appear in the text you were given? Please also tell me the total word count.",
        },
    ]

    print(f"📝 Testing with ~{len(long_text.split())} words of context...")

    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=100)
    duration = time.time() - start_time

    print(f"✅ Response ({duration:.2f}s):")
    print(f"   {response['response']}")

    return response


# =============================================================================
# Example 9: Simple Chat Interface
# =============================================================================


async def simple_chat_example(model: str = "llama-3.1-8b-instant"):
    """Simple chat interface simulation - ultra-fast responses"""
    print(f"\n💬 Simple Chat Interface with {model}")
    print("=" * 60)

    client = get_client("groq", model=model)

    # Simulate a simple conversation
    conversation = [
        "Hello! What makes Groq special?",
        "Can you explain LPU vs GPU?",
        "Write a Python function to calculate factorial.",
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and knowledgeable AI assistant.",
        }
    ]

    total_time = 0

    for user_input in conversation:
        print(f"👤 User: {user_input}")

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Get response
        start_time = time.time()
        response = await client.create_completion(messages, max_tokens=200)
        duration = time.time() - start_time
        total_time += duration

        assistant_response = response.get("response", "No response")

        print(f"🤖 Groq ({duration:.2f}s): {assistant_response}")
        print()

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})

    print(f"⚡ Total conversation time: {total_time:.2f}s")
    print(f"⚡ Average response time: {total_time / len(conversation):.2f}s")

    return messages


# =============================================================================
# Example 10: Temperature and Parameters Test
# =============================================================================


async def parameters_example(model: str = "llama-3.3-70b-versatile"):
    """Test different parameters and settings"""
    print(f"\n🎛️  Parameters Test with {model}")
    print("=" * 60)

    client = get_client("groq", model=model)

    # Test different temperatures
    temperatures = [0.1, 0.7, 1.2]
    prompt = "Write a creative opening line for a science fiction story."

    for temp in temperatures:
        print(f"\n🌡️  Temperature {temp}:")

        messages = [{"role": "user", "content": prompt}]

        try:
            start_time = time.time()
            response = await client.create_completion(
                messages, temperature=temp, max_tokens=50
            )
            duration = time.time() - start_time
            print(f"   ({duration:.2f}s) {response['response']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Test with system message
    print("\n🎭 With System Message:")
    messages = [
        {"role": "system", "content": "You are a poetic AI that speaks in rhymes."},
        {"role": "user", "content": "Tell me about the speed of light."},
    ]

    start_time = time.time()
    response = await client.create_completion(messages, temperature=0.8, max_tokens=100)
    duration = time.time() - start_time
    print(f"   ({duration:.2f}s) {response['response']}")

    return True


# =============================================================================
# Example 11: Parallel Processing Test
# =============================================================================


async def parallel_processing_test():
    """Test Groq's ability to handle parallel requests efficiently"""
    print("\n🔀 Parallel Processing Test")
    print("=" * 60)

    model = "llama-3.1-8b-instant"
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
    sequential_responses = []

    for prompt in prompts:
        client = get_client("groq", model=model)
        response = await client.create_completion(
            [{"role": "user", "content": prompt}], max_tokens=50
        )
        sequential_responses.append(response["response"][:50])

    sequential_time = time.time() - sequential_start
    print(f"   ✅ Completed in {sequential_time:.2f}s")

    # Parallel processing
    print("\n⚡ Parallel processing:")
    parallel_start = time.time()

    async def process_prompt(prompt):
        client = get_client("groq", model=model)
        response = await client.create_completion(
            [{"role": "user", "content": prompt}], max_tokens=50
        )
        return response["response"][:50]

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
# Example 12: Comprehensive Feature Test
# =============================================================================


async def comprehensive_test(model: str = "llama-3.3-70b-versatile"):
    """Test multiple features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)

    # Check what features this model supports
    config = get_config()
    supports_tools = config.supports_feature("groq", Feature.TOOLS, model)
    supports_json = config.supports_feature("groq", Feature.JSON_MODE, model)

    print(f"Model capabilities: Tools={supports_tools}, JSON={supports_json}")

    client = get_client("groq", model=model)

    # Test 1: Speed test
    print("\n⚡ Speed Test:")
    messages = [{"role": "user", "content": "Count from 1 to 10."}]
    start_time = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start_time
    print(f"   Response in {duration:.3f}s: {response['response'][:50]}...")

    # Test 2: Tool calling (if supported)
    if supports_tools:
        print("\n🔧 Tool Calling Test:")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_performance",
                    "description": "Analyze system performance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "value": {"type": "number"},
                        },
                        "required": ["metric", "value"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "user",
                "content": "My system is processing 1000 requests per second. Analyze this performance.",
            }
        ]
        response = await client.create_completion(messages, tools=tools)

        if response.get("tool_calls"):
            print(f"   ✅ Tool calls generated: {len(response['tool_calls'])}")
        else:
            print(f"   ℹ️  Direct response: {response['response'][:100]}...")

    # Test 3: JSON mode (if supported)
    if supports_json:
        print("\n📋 JSON Mode Test:")
        messages = [
            {"role": "system", "content": "Output JSON only."},
            {
                "role": "user",
                "content": "Create a JSON object with fields: status, speed, provider.",
            },
        ]

        try:
            response = await client.create_completion(
                messages, response_format={"type": "json_object"}
            )
            parsed = json.loads(response["response"])
            print(f"   ✅ Valid JSON: {parsed}")
        except Exception as e:
            print(f"   ⚠️  JSON test failed: {e}")

    print("\n✅ Comprehensive test completed!")
    return True


# =============================================================================
# Main Function
# =============================================================================


async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Groq Provider Example Script")
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Model to use (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--skip-tools", action="store_true", help="Skip function calling examples"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("📋 Available Groq Models")
        print("=" * 60)
        model_info = get_available_groq_models()

        print(f"\n📦 Configured Models ({len(model_info['configured'])}):")
        for model in model_info["configured"]:
            print(f"  - {model}")

        if model_info["api"]:
            print(f"\n🌐 Available via API ({len(model_info['api'])}):")
            # Group by family
            llama_models = [m for m in model_info["api"] if "llama" in m.lower()]
            other_models = [m for m in model_info["api"] if "llama" not in m.lower()]

            if llama_models:
                print("  Llama Models:")
                for model in sorted(llama_models)[:10]:
                    print(f"    - {model}")

            if other_models:
                print("  Other Models:")
                for model in sorted(other_models)[:10]:
                    print(f"    - {model}")

            if len(model_info["api"]) > 20:
                print(f"  ... and {len(model_info['api']) - 20} more")

        print(f"\n💡 Total Available: {len(model_info['all'])} models")
        print("\n⚡ Groq's LPU delivers ultra-fast inference for all models!")
        return

    print("🚀 Groq Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
    print("Website: https://groq.com")
    print("Console: https://console.groq.com")

    # Show model capabilities
    try:
        config = get_config()
        supports_tools = config.supports_feature("groq", Feature.TOOLS, args.model)
        supports_streaming = config.supports_feature(
            "groq", Feature.STREAMING, args.model
        )
        supports_json = config.supports_feature("groq", Feature.JSON_MODE, args.model)

        print("Model capabilities:")
        print(f"  Tools: {'✅' if supports_tools else '❌'}")
        print(f"  Streaming: {'✅' if supports_streaming else '❌'}")
        print(f"  JSON Mode: {'✅' if supports_json else '❌'}")
        print("  System Messages: ✅")  # Groq supports system messages

    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    # Focus on benchmark if requested
    if args.benchmark:
        await speed_benchmark()
        await parallel_processing_test()
        return

    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
    ]

    if not args.quick:
        if not args.skip_tools:
            examples.append(
                ("Function Calling", lambda: function_calling_example(args.model))
            )

        examples.extend(
            [
                ("JSON Mode", lambda: json_mode_example(args.model)),
                ("Speed Benchmark", speed_benchmark),
                ("Model Comparison", model_comparison_example),
                ("Context Window Test", lambda: context_window_test(args.model)),
                ("Simple Chat", lambda: simple_chat_example(args.model)),
                ("Parameters Test", lambda: parameters_example(args.model)),
                ("Parallel Processing", parallel_processing_test),
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
        print("⚡ Groq's LPU technology delivers ultra-fast inference!")
        print("🚀 Experience the speed difference with chuk-llm!")
    else:
        print("\n⚠️  Some examples failed. Check your API key and model access.")

        # Show model recommendations
        print("\n💡 Groq Model Recommendations:")
        print("   • For speed: llama-3.1-8b-instant")
        print("   • For capability: llama-3.3-70b-versatile")
        print("   • For reasoning: openai/gpt-oss-120b (if available)")
        print("   • Context window: Up to 131k tokens!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
