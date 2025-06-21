#!/usr/bin/env python3
# examples/openai_usage_examples.py
"""
OpenAI/GPT Provider Example Usage Script
========================================

Demonstrates all the features of the OpenAI provider in the chuk-llm library.
Run this script to see GPT models in action with various capabilities.

Requirements:
- pip install openai chuk-llm
- Set OPENAI_API_KEY environment variable

Usage:
    python openai_example.py
    python openai_example.py --model gpt-4o
    python openai_example.py --skip-vision
"""

import asyncio
import argparse
import os
import sys
import time
import base64
from typing import Dict, Any, List

# dotenv
from dotenv import load_dotenv

# load environment variables
load_dotenv() 

# Ensure we have the required environment
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.client import get_client, get_provider_info
    from chuk_llm.configuration import get_config, Feature
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)

def create_test_image(color: str = "red", size: int = 15) -> str:
    """Create a test image as base64 - tries PIL first, fallback to hardcoded"""
    try:
        from PIL import Image
        import io
        
        # Create a colored square
        img = Image.new('RGB', (size, size), color)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_data
    except ImportError:
        print("⚠️  PIL not available, using fallback image")
        # Fallback: 15x15 red square (valid PNG)
        return "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAPCAYAAAA71pVKAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABYSURBVCiRY2RgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBkYGBgZGBgYGRgYGBgZGBgYGAAAgAANgAOAUUe1wAAAABJRU5ErkJggg=="

# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================

async def basic_text_example(model: str = "gpt-4o-mini"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_client("openai", model=model)
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain neural networks in simple terms (2-3 sentences)."}
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

async def streaming_example(model: str = "gpt-4o-mini"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)
    
    # Check streaming support
    config = get_config()
    if not config.supports_feature("openai", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None
    
    client = get_client("openai", model=model)
    
    messages = [
        {"role": "user", "content": "Write a short haiku about artificial intelligence."}
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

async def function_calling_example(model: str = "gpt-4o-mini"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)
    
    # Check if model supports tools
    config = get_config()
    if not config.supports_feature("openai", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        return None
    
    client = get_client("openai", model=model)
    
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
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "max_results": {
                            "type": "integer", 
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
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
                            "description": "Mathematical expression to evaluate"
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Number of decimal places"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Search for 'latest AI research' and calculate 25.5 * 14.2 with 3 decimal places."}
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
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": response["tool_calls"]
        })
        
        # Add mock tool results
        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]
            
            if func_name == "search_web":
                result = '{"results": ["Latest breakthrough in transformer models", "New multimodal AI research", "Advances in reasoning capabilities"]}'
            elif func_name == "calculate_math":
                result = '{"result": 361.100, "expression": "25.5 * 14.2", "precision": 3}'
            else:
                result = '{"status": "success"}'
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": func_name,
                "content": result
            })
        
        # Get final response
        print("🔄 Getting final response...")
        final_response = await client.create_completion(messages)
        print(f"✅ Final response:")
        print(f"   {final_response['response']}")
        
        return final_response
    else:
        print("ℹ️  No tool calls were made")
        print(f"   Response: {response['response']}")
        return response

# =============================================================================
# Example 4: Vision Capabilities
# =============================================================================

async def vision_example(model: str = "gpt-4o"):
    """Vision capabilities with GPT-4 Vision models"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)
    
    # Check if model supports vision
    config = get_config()
    if not config.supports_feature("openai", Feature.VISION, model):
        print(f"⚠️  Skipping vision: Model {model} doesn't support vision")
        print(f"💡 Try a vision-capable model like: gpt-4o, gpt-4-turbo, gpt-4")
        return None
    
    client = get_client("openai", model=model)
    
    # Create a proper test image
    print("🖼️  Creating test image...")
    test_image = create_test_image("blue", 20)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What color is this square? Answer with just the color name."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{test_image}"
                    }
                }
            ]
        }
    ]
    
    print("👀 Analyzing image...")
    response = await client.create_completion(messages, max_tokens=50)
    
    print(f"✅ Vision response:")
    print(f"   {response['response']}")
    
    return response

# =============================================================================
# Example 5: JSON Mode
# =============================================================================

async def json_mode_example(model: str = "gpt-4o-mini"):
    """JSON mode example with structured output"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)
    
    # Check JSON mode support
    config = get_config()
    if not config.supports_feature("openai", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None
    
    client = get_client("openai", model=model)
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant designed to output JSON. Generate a JSON object with information about a programming language."
        },
        {
            "role": "user", 
            "content": "Tell me about Python programming language in JSON format with fields: name, year_created, creator, main_features (array), and popularity_score (1-10)."
        }
    ]
    
    print("📝 Requesting JSON output...")
    
    try:
        response = await client.create_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        print(f"✅ JSON response:")
        print(f"   {response['response']}")
        
        # Try to parse as JSON to verify
        import json
        try:
            parsed = json.loads(response['response'])
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
# Example 6: Reasoning Models Test
# =============================================================================

async def reasoning_example(model: str = "o3-mini"):
    """Test reasoning capabilities with O-series models"""
    print(f"\n🧠 Reasoning Example with {model}")
    print("=" * 60)
    
    # Check if model supports reasoning
    config = get_config()
    if not config.supports_feature("openai", Feature.REASONING, model):
        print(f"⚠️  Model {model} doesn't have enhanced reasoning")
        print(f"💡 Try a reasoning model like: o3, o3-mini, o1, o1-mini")
        return None
    
    client = get_client("openai", model=model)
    
    messages = [
        {
            "role": "user", 
            "content": "I have a 3-gallon jug and a 5-gallon jug. I need to measure exactly 4 gallons of water. How can I do this? Think through this step-by-step."
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
# Example 7: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different GPT models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    # Updated model list based on current config
    models = [
        "gpt-4.1-mini",         # Latest GPT-4.1 Mini
        "gpt-4o-mini",          # GPT-4o Mini
        "gpt-4o",               # GPT-4o
        "gpt-4-turbo",          # GPT-4 Turbo
        "gpt-3.5-turbo",        # GPT-3.5 Turbo
    ]
    
    prompt = "What is machine learning? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("openai", model=model)
            messages = [{"role": "user", "content": prompt}]
            
            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time
            
            results[model] = {
                "response": response.get("response", ""),
                "time": duration,
                "length": len(response.get("response", "")),
                "success": True
            }
            
        except Exception as e:
            results[model] = {
                "response": f"Error: {str(e)}",
                "time": 0,
                "length": 0,
                "success": False
            }
    
    print("\n📈 Results:")
    for model, result in results.items():
        status = "✅" if result["success"] else "❌"
        model_short = model.replace("gpt-", "").replace("-turbo", "")
        print(f"   {status} {model_short}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()
    
    return results

# =============================================================================
# Example 8: Feature Detection
# =============================================================================

async def feature_detection_example(model: str = "gpt-4o-mini"):
    """Detect and display model features"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)
    
    # Get model info
    try:
        model_info = get_provider_info("openai", model)
        
        print("📋 Model Information:")
        print(f"   Provider: {model_info['provider']}")
        print(f"   Model: {model_info['model']}")
        print(f"   Max Context: {model_info['max_context_length']:,} tokens")
        print(f"   Max Output: {model_info['max_output_tokens']:,} tokens")
        
        print("\n🎯 Supported Features:")
        for feature, supported in model_info['supports'].items():
            status = "✅" if supported else "❌"
            print(f"   {status} {feature}")
        
        print("\n📊 Rate Limits:")
        for tier, limit in model_info['rate_limits'].items():
            print(f"   {tier}: {limit} requests/min")
        
    except Exception as e:
        print(f"⚠️  Could not get model info: {e}")
    
    # Test actual client info
    try:
        client = get_client("openai", model=model)
        client_info = client.get_model_info()
        
        print(f"\n🔧 Client Features:")
        print(f"   Streaming: {'✅' if client_info.get('supports_streaming') else '❌'}")
        print(f"   Tools: {'✅' if client_info.get('supports_tools') else '❌'}")
        print(f"   Vision: {'✅' if client_info.get('supports_vision') else '❌'}")
        print(f"   JSON Mode: {'✅' if client_info.get('supports_json_mode') else '❌'}")
        print(f"   System Messages: {'✅' if client_info.get('supports_system_messages') else '❌'}")
        
    except Exception as e:
        print(f"⚠️  Could not get client info: {e}")
    
    return model_info if 'model_info' in locals() else None

# =============================================================================
# Example 9: Simple Chat Interface
# =============================================================================

async def simple_chat_example(model: str = "gpt-4o-mini"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface with {model}")
    print("=" * 60)
    
    client = get_client("openai", model=model)
    
    # Simulate a simple conversation
    conversation = [
        "Hello! What's the weather like?",
        "What's the most exciting development in AI recently?", 
        "Can you help me write a JavaScript function to sort an array?"
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
        
        print(f"🤖 GPT: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 10: Temperature and Parameters Test
# =============================================================================

async def parameters_example(model: str = "gpt-4o-mini"):
    """Test different parameters and settings"""
    print(f"\n🎛️  Parameters Test with {model}")
    print("=" * 60)
    
    client = get_client("openai", model=model)
    
    # Test different temperatures
    temperatures = [0.1, 0.7, 1.2]
    prompt = "Write a creative opening line for a science fiction story."
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature {temp}:")
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await client.create_completion(
                messages,
                temperature=temp,
                max_tokens=50
            )
            print(f"   {response['response']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test with system message
    print(f"\n🎭 With System Message:")
    messages = [
        {"role": "system", "content": "You are a poetic AI that speaks in rhymes."},
        {"role": "user", "content": "Tell me about the ocean."}
    ]
    
    response = await client.create_completion(messages, temperature=0.8, max_tokens=100)
    print(f"   {response['response']}")
    
    return True

# =============================================================================
# Example 11: Comprehensive Feature Test
# =============================================================================

async def comprehensive_test(model: str = "gpt-4o"):
    """Test multiple features in one comprehensive example"""
    print(f"\n🚀 Comprehensive Feature Test with {model}")
    print("=" * 60)
    
    # Check what features this model supports
    config = get_config()
    supports_tools = config.supports_feature("openai", Feature.TOOLS, model)
    supports_vision = config.supports_feature("openai", Feature.VISION, model)
    
    print(f"Model capabilities: Tools={supports_tools}, Vision={supports_vision}")
    
    if not supports_tools and not supports_vision:
        print("⚠️  Model doesn't support tools or vision - using text-only test")
        return await simple_chat_example(model)
    
    client = get_client("openai", model=model)
    
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
                            "main_topics": {"type": "array", "items": {"type": "string"}},
                            "complexity": {"type": "string", "enum": ["simple", "medium", "complex"]}
                        },
                        "required": ["content_type", "main_topics"]
                    }
                }
            }
        ]
    
    # Create content based on capabilities
    if supports_vision:
        print("🖼️  Creating test image...")
        test_image = create_test_image("green", 25)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert content analyst. Use the provided function when analyzing content."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this image and the following text using the analyze_content function: 'This is a test of multimodal AI capabilities.'"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{test_image}"
                        }
                    }
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are an expert content analyst. Use the provided function when analyzing content."
            },
            {
                "role": "user",
                "content": "Please analyze this text using the analyze_content function: 'Artificial intelligence is transforming how we interact with technology through natural language processing and machine learning algorithms.'"
            }
        ]
    
    print("🔄 Testing comprehensive capabilities...")
    
    response = await client.create_completion(messages, tools=tools)
    
    if response.get("tool_calls"):
        print(f"✅ Tool calls generated: {len(response['tool_calls'])}")
        for tc in response["tool_calls"]:
            print(f"   🔧 {tc['function']['name']}: {tc['function']['arguments'][:100]}...")
    else:
        print(f"ℹ️  Direct response: {response['response'][:150]}...")
    
    print("✅ Comprehensive test completed!")
    return response

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="OpenAI/GPT Provider Example Script")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--test-reasoning", action="store_true", help="Test reasoning models")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 OpenAI/GPT Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    
    # Show model capabilities
    try:
        config = get_config()
        supports_tools = config.supports_feature("openai", Feature.TOOLS, args.model)
        supports_vision = config.supports_feature("openai", Feature.VISION, args.model)
        supports_reasoning = config.supports_feature("openai", Feature.REASONING, args.model)
        supports_streaming = config.supports_feature("openai", Feature.STREAMING, args.model)
        supports_json = config.supports_feature("openai", Feature.JSON_MODE, args.model)
        
        print(f"Model capabilities:")
        print(f"  Tools: {'✅' if supports_tools else '❌'}")
        print(f"  Vision: {'✅' if supports_vision else '❌'}")
        print(f"  Reasoning: {'✅' if supports_reasoning else '❌'}")
        print(f"  Streaming: {'✅' if supports_streaming else '❌'}")
        print(f"  JSON Mode: {'✅' if supports_json else '❌'}")
        
    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    # Focus on reasoning if requested
    if args.test_reasoning:
        await reasoning_example("o3-mini")
        return

    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("JSON Mode", lambda: json_mode_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_functions:
            examples.append(("Function Calling", lambda: function_calling_example(args.model)))
        
        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example("gpt-4o")))
        
        examples.extend([
            ("Model Comparison", model_comparison_example),
            ("Simple Chat", lambda: simple_chat_example(args.model)),
            ("Parameters Test", lambda: parameters_example(args.model)),
            ("Comprehensive Test", lambda: comprehensive_test("gpt-4o")),
        ])
    
    # Run examples
    results = {}
    for name, example_func in examples:
        try:
            print(f"\n" + "="*60)
            start_time = time.time()
            result = await example_func()
            duration = time.time() - start_time
            results[name] = {"success": True, "result": result, "time": duration}
            print(f"✅ {name} completed in {duration:.2f}s")
        except Exception as e:
            results[name] = {"success": False, "error": str(e), "time": 0}
            print(f"❌ {name} failed: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
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
        print(f"\n🎉 All examples completed successfully!")
        print(f"🔗 OpenAI/GPT provider is working perfectly with chuk-llm!")
        print(f"✨ Features tested: {args.model} capabilities")
    else:
        print(f"\n⚠️  Some examples failed. Check your API key and model access.")
        
        # Show model recommendations
        print(f"\n💡 Model Recommendations:")
        print(f"   • For general use: gpt-4o-mini, gpt-4o")
        print(f"   • For vision: gpt-4o, gpt-4-turbo")
        print(f"   • For reasoning: o3-mini, o3, o1-mini")
        print(f"   • For cost-effective: gpt-4o-mini, gpt-3.5-turbo")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)