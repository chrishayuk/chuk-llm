#!/usr/bin/env python3
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
    python openai_example.py --model gpt-4.1
    python openai_example.py --skip-vision
"""

import asyncio
import argparse
import os
import sys
import time
from typing import Dict, Any, List

# dotenv
from dotenv import load_dotenv

# load environment variables
load_dotenv() 

# Ensure we have the required environment
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.llm_client import get_llm_client
    from chuk_llm.llm.configuration.capabilities import CapabilityChecker
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)

# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================

async def basic_text_example(model: str = "gpt-4.1"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_llm_client("openai", model=model)
    
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

async def streaming_example(model: str = "gpt-4.1"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("openai", model=model)
    
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

async def function_calling_example(model: str = "gpt-4.1"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)
    
    # Check if model supports tools first
    can_handle, issues = CapabilityChecker.can_handle_request(
        "openai", model, has_tools=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping function calling: {', '.join(issues)}")
        return None
    
    client = get_llm_client("openai", model=model)
    
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
    can_handle, issues = CapabilityChecker.can_handle_request(
        "openai", model, has_vision=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping vision: {', '.join(issues)}")
        return None
    
    client = get_llm_client("openai", model=model)
    
    # Simple test image (1x1 red pixel)
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this image? Please describe it in detail."
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
    response = await client.create_completion(messages)
    
    print(f"✅ Vision response:")
    print(f"   {response['response']}")
    
    return response

# =============================================================================
# Example 5: JSON Mode
# =============================================================================

async def json_mode_example(model: str = "gpt-4.1"):
    """JSON mode example with structured output"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("openai", model=model)
    
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
        print(f"❌ JSON mode not supported or failed: {e}")
        # Fallback to regular request
        response = await client.create_completion(messages)
        print(f"📝 Fallback response: {response['response'][:200]}...")
    
    return response

# =============================================================================
# Example 6: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different GPT models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    models = [
        "gpt-4.1",              # Latest GPT-4.1
        "gpt-4.1-mini",         # GPT-4.1 Mini
        "gpt-4.1-nano",         # GPT-4.1 Nano
        "gpt-4o-mini",          # GPT-4o Mini
        "gpt-4o",               # GPT-4o
        "gpt-4-turbo",          # GPT-4 Turbo
        "gpt-3.5-turbo",        # GPT-3.5 Turbo
        "gpt-4"                 # GPT-4
    ]
    
    prompt = "What is machine learning? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_llm_client("openai", model=model)
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
        print(f"   {status} {model}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()
    
    return results

# =============================================================================
# Example 7: Multiple Models Test
# =============================================================================

async def multiple_models_example():
    """Test multiple GPT models"""
    print(f"\n🔄 Multiple Models Test")
    print("=" * 60)
    
    models_to_test = [
        "gpt-4.1",              # Latest GPT-4.1
        "gpt-4.1-mini",         # GPT-4.1 Mini
        "gpt-4o",               # GPT-4o
        "gpt-4o-mini",          # GPT-4o Mini
        "gpt-4-turbo",          # GPT-4 Turbo
        "gpt-3.5-turbo"         # GPT-3.5 Turbo
    ]
    
    prompt = "Write a one-line explanation of neural networks."
    
    for model in models_to_test:
        try:
            print(f"🔄 Testing {model}...")
            client = get_llm_client("openai", model=model)
            messages = [{"role": "user", "content": prompt}]
            
            start_time = time.time()
            response = await client.create_completion(messages)
            duration = time.time() - start_time
            
            print(f"✅ {model} ({duration:.2f}s):")
            print(f"   {response['response'][:100]}...")
            
        except Exception as e:
            print(f"❌ {model}: {str(e)}")
    
    return True

# =============================================================================
# Example 8: Simple Chat Interface
# =============================================================================

async def simple_chat_example(model: str = "gpt-4.1"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface")
    print("=" * 60)
    
    client = get_llm_client("openai", model=model)
    
    # Simulate a simple conversation
    conversation = [
        "Hello! What's the weather like?",
        "What's the most exciting development in AI recently?", 
        "Can you help me write a JavaScript function to sort an array?"
    ]
    
    messages = []
    
    for user_input in conversation:
        print(f"👤 User: {user_input}")
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = await client.create_completion(messages)
        assistant_response = response.get("response", "No response")
        
        print(f"🤖 GPT: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 9: Model Information
# =============================================================================

async def model_info_example(model: str = "gpt-4.1"):
    """Get detailed model information"""
    print(f"\n📋 Model Information for {model}")
    print("=" * 60)
    
    client = get_llm_client("openai", model=model)
    
    # Get model info from client
    if hasattr(client, 'get_model_info'):
        info = client.get_model_info()
        print("🔍 Model details:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    # Get capability info
    model_info = CapabilityChecker.get_model_info("openai", model)
    print(f"\n🎯 Capabilities:")
    for key, value in model_info.items():
        if key != "error":
            print(f"   {key}: {value}")
    
    return model_info

# =============================================================================
# Example 10: Temperature and Parameters Test
# =============================================================================

async def parameters_example(model: str = "gpt-4.1"):
    """Test different parameters and settings"""
    print(f"\n🎛️  Parameters Test with {model}")
    print("=" * 60)
    
    client = get_llm_client("openai", model=model)
    
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
    
    response = await client.create_completion(messages, temperature=0.8)
    print(f"   {response['response']}")
    
    return True

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="OpenAI/GPT Provider Example Script")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 OpenAI/GPT Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    
    examples = [
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("Model Info", lambda: model_info_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_functions:
            examples.append(("Function Calling", lambda: function_calling_example(args.model)))
        
        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example("gpt-4o")))
        
        examples.extend([
            ("JSON Mode", lambda: json_mode_example(args.model)),
            ("Model Comparison", model_comparison_example),
            ("Multiple Models", multiple_models_example),
            ("Simple Chat", lambda: simple_chat_example(args.model)),
            ("Parameters Test", lambda: parameters_example(args.model)),
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
    else:
        print(f"\n⚠️  Some examples failed. Check your API key and model access.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)