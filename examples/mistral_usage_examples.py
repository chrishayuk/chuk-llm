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
    python mistral_example.py --model mistral-small-latest
    python mistral_example.py --skip-vision
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
if not os.getenv("MISTRAL_API_KEY"):
    print("❌ Please set MISTRAL_API_KEY environment variable")
    print("   export MISTRAL_API_KEY='your_api_key_here'")
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

async def basic_text_example(model: str = "mistral-large-latest"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_llm_client("mistral", model=model)
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms (2-3 sentences)."}
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

async def streaming_example(model: str = "mistral-large-latest"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("mistral", model=model)
    
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

async def function_calling_example(model: str = "mistral-large-latest"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)
    
    # Check if model supports tools first
    can_handle, issues = CapabilityChecker.can_handle_request(
        "mistral", model, has_tools=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping function calling: {', '.join(issues)}")
        return None
    
    client = get_llm_client("mistral", model=model)
    
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
                            "description": "The bill amount in dollars"
                        },
                        "tip_percentage": {
                            "type": "number", 
                            "description": "Tip percentage (default: 18)"
                        }
                    },
                    "required": ["bill_amount"]
                }
            }
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
                            "description": "City name"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Calculate a 20% tip on a $85 bill and tell me the weather in Paris."}
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
            
            if func_name == "calculate_tip":
                result = '{"tip_amount": 17.0, "total_amount": 102.0}'
            elif func_name == "get_weather":
                result = '{"temperature": "22°C", "condition": "Sunny", "humidity": "65%"}'
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

async def vision_example(model: str = "pixtral-12b-latest"):
    """Vision capabilities with multimodal models"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)
    
    # Check if model supports vision
    can_handle, issues = CapabilityChecker.can_handle_request(
        "mistral", model, has_vision=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping vision: {', '.join(issues)}")
        return None
    
    client = get_llm_client("mistral", model=model)
    
    # Simple test image (1x1 red pixel)
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this image? Describe it briefly."
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{test_image}"
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
# Example 5: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different Mistral models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    models = [
        "mistral-large-latest",
        "mistral-small-latest", 
        "ministral-8b-latest"
    ]
    
    prompt = "What is machine learning? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_llm_client("mistral", model=model)
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
# Example 6: Multiple Models Test
# =============================================================================

async def multiple_models_example():
    """Test multiple Mistral models"""
    print(f"\n🔄 Multiple Models Test")
    print("=" * 60)
    
    models_to_test = [
        "mistral-large-latest",
        "mistral-small-latest",
        "codestral-latest",
        "ministral-8b-latest"
    ]
    
    prompt = "Write a one-line summary of machine learning."
    
    for model in models_to_test:
        try:
            print(f"🔄 Testing {model}...")
            client = get_llm_client("mistral", model=model)
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
# Example 7: Simple Chat Interface
# =============================================================================

async def simple_chat_example(model: str = "mistral-large-latest"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface")
    print("=" * 60)
    
    client = get_llm_client("mistral", model=model)
    
    # Simulate a simple conversation
    conversation = [
        "Hello! How are you?",
        "What's your favorite programming language?", 
        "Can you write a simple Python function?"
    ]
    
    messages = []
    
    for user_input in conversation:
        print(f"👤 User: {user_input}")
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = await client.create_completion(messages)
        assistant_response = response.get("response", "No response")
        
        print(f"🤖 Assistant: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 8: Model Information
# =============================================================================

async def model_info_example(model: str = "mistral-large-latest"):
    """Get detailed model information"""
    print(f"\n📋 Model Information for {model}")
    print("=" * 60)
    
    client = get_llm_client("mistral", model=model)
    
    # Get model info from client
    if hasattr(client, 'get_model_info'):
        info = client.get_model_info()
        print("🔍 Model details:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    # Get capability info
    model_info = CapabilityChecker.get_model_info("mistral", model)
    print(f"\n🎯 Capabilities:")
    for key, value in model_info.items():
        if key != "error":
            print(f"   {key}: {value}")
    
    return model_info

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Mistral Provider Example Script")
    parser.add_argument("--model", default="mistral-large-latest", help="Model to use")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 Mistral Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('MISTRAL_API_KEY') else '❌ Missing'}")
    
    examples = [
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("Model Info", lambda: model_info_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_functions:
            examples.append(("Function Calling", lambda: function_calling_example(args.model)))
        
        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example("pixtral-12b-latest")))
        
        examples.extend([
            ("Model Comparison", model_comparison_example),
            ("Multiple Models", multiple_models_example),
            ("Simple Chat", lambda: simple_chat_example(args.model)),
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
        print(f"🔗 Mistral provider is working perfectly with chuk-llm!")
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