#!/usr/bin/env python3
"""
Google Gemini Provider Example Usage Script
===========================================

Demonstrates all the features of the Gemini provider in the chuk-llm library.
Run this script to see Gemini models in action with various capabilities.

Requirements:
- pip install google-genai chuk-llm
- Set GEMINI_API_KEY environment variable

Usage:
    python gemini_example.py
    python gemini_example.py --model gemini-2.5-flash-preview-05-20
    python gemini_example.py --skip-vision
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
if not os.getenv("GEMINI_API_KEY"):
    print("❌ Please set GEMINI_API_KEY environment variable")
    print("   export GEMINI_API_KEY='your_api_key_here'")
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

async def basic_text_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_llm_client("gemini", model=model)
    
    messages = [
        {"role": "user", "content": "Explain large language models in simple terms (2-3 sentences)."}
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

async def streaming_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("gemini", model=model)
    
    messages = [
        {"role": "user", "content": "Write a short poem about the future of technology."}
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

async def function_calling_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)
    
    # Check if model supports tools first
    can_handle, issues = CapabilityChecker.can_handle_request(
        "gemini", model, has_tools=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping function calling: {', '.join(issues)}")
        return None
    
    client = get_llm_client("gemini", model=model)
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_location_info",
                "description": "Get information about a location including coordinates and timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location name (city, country)"
                        },
                        "include_weather": {
                            "type": "boolean", 
                            "description": "Whether to include weather information"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "unit_converter",
                "description": "Convert between different units of measurement",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The value to convert"
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "The unit to convert from"
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "The unit to convert to"
                        }
                    },
                    "required": ["value", "from_unit", "to_unit"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Get information about Tokyo, Japan and convert 100 kilometers to miles."}
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
            
            if func_name == "get_location_info":
                result = '{"location": "Tokyo, Japan", "coordinates": {"lat": 35.6762, "lng": 139.6503}, "timezone": "Asia/Tokyo", "population": 37400068}'
            elif func_name == "unit_converter":
                result = '{"original_value": 100, "from_unit": "kilometers", "to_unit": "miles", "converted_value": 62.137, "formula": "km * 0.621371"}'
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

async def vision_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """Vision capabilities with Gemini models"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)
    
    # Check if model supports vision
    can_handle, issues = CapabilityChecker.can_handle_request(
        "gemini", model, has_vision=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping vision: {', '.join(issues)}")
        return None
    
    client = get_llm_client("gemini", model=model)
    
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
    
    try:
        # Add timeout to prevent hanging
        response = await asyncio.wait_for(
            client.create_completion(messages),
            timeout=30.0  # 30 second timeout
        )
        
        print(f"✅ Vision response:")
        print(f"   {response['response']}")
        
        return response
        
    except asyncio.TimeoutError:
        print("❌ Vision request timed out after 30 seconds")
        return {"response": "Timeout error", "tool_calls": [], "error": True}
    except Exception as e:
        print(f"❌ Vision request failed: {e}")
        return {"response": f"Error: {str(e)}", "tool_calls": [], "error": True}

# =============================================================================
# Example 5: System Instructions
# =============================================================================

async def system_instructions_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """System instructions example with different personas"""
    print(f"\n🎭 System Instructions Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("gemini", model=model)
    
    # Test different system instructions
    instructions = [
        {
            "name": "Creative Writer",
            "system": "You are a creative writer who loves to tell engaging stories with vivid descriptions.",
            "query": "Describe a sunset."
        },
        {
            "name": "Technical Expert",
            "system": "You are a technical expert who explains complex concepts clearly and precisely.",
            "query": "Explain how neural networks work."
        },
        {
            "name": "Friendly Teacher",
            "system": "You are a patient and encouraging teacher who makes learning fun for students.",
            "query": "Teach me about photosynthesis."
        }
    ]
    
    for instruction in instructions:
        print(f"\n🎭 Testing {instruction['name']} persona:")
        
        messages = [
            {"role": "user", "content": instruction["query"]}
        ]
        
        try:
            # Try with system parameter if supported
            response = await client.create_completion(
                messages, 
                system_instruction=instruction["system"],
                max_tokens=150
            )
            print(f"   {response['response'][:200]}...")
        except Exception as e:
            # Fallback: add system instruction to user message
            system_messages = [
                {"role": "user", "content": f"System: {instruction['system']}\n\nUser: {instruction['query']}"}
            ]
            response = await client.create_completion(system_messages, max_tokens=150)
            print(f"   {response['response'][:200]}...")
    
    return True

# =============================================================================
# Example 6: JSON Mode
# =============================================================================

async def json_mode_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """JSON mode example with structured output"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("gemini", model=model)
    
    messages = [
        {
            "role": "user", 
            "content": "Generate information about JavaScript in JSON format with fields: name, year_created, creator, main_features (array), and difficulty_level (1-10)."
        }
    ]
    
    print("📝 Requesting JSON output...")
    
    try:
        response = await client.create_completion(
            messages,
            generation_config={
                "response_mime_type": "application/json"
            },
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
        # Fallback to regular request with JSON instruction
        json_messages = [
            {"role": "user", "content": messages[0]["content"] + " Please respond only with valid JSON."}
        ]
        response = await client.create_completion(json_messages)
        print(f"📝 Fallback response: {response['response'][:200]}...")
    
    return response

# =============================================================================
# Example 7: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different Gemini models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    models = [
        "gemini-2.5-flash-preview-04-17",   # Gemini 2.5 Flash Preview (April)
        "gemini-2.5-flash-preview-05-20",   # Gemini 2.5 Flash Preview (May)
        "gemini-2.5-pro-preview-05-06",     # Gemini 2.5 Pro Preview
        "gemini-2.0-flash",                 # Gemini 2.0 Flash
        "gemini-2.0-flash-lite",            # Gemini 2.0 Flash Lite
        "gemini-1.5-pro",                   # Gemini 1.5 Pro
        "gemini-1.5-flash",                 # Gemini 1.5 Flash
        "gemini-1.5-flash-8b"               # Gemini 1.5 Flash 8B
    ]
    
    prompt = "What is artificial intelligence? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_llm_client("gemini", model=model)
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
# Example 8: Multiple Models Test
# =============================================================================

async def multiple_models_example():
    """Test multiple Gemini models"""
    print(f"\n🔄 Multiple Models Test")
    print("=" * 60)
    
    models_to_test = [
        "gemini-2.5-flash-preview-05-20",   # Latest Gemini 2.5 Flash Preview
        "gemini-2.5-pro-preview-05-06",     # Gemini 2.5 Pro Preview
        "gemini-2.0-flash",                 # Gemini 2.0 Flash
        "gemini-1.5-pro",                   # Gemini 1.5 Pro
        "gemini-1.5-flash",                 # Gemini 1.5 Flash
        "gemini-1.5-flash-8b"               # Gemini 1.5 Flash 8B
    ]
    
    prompt = "Write a one-line explanation of deep learning."
    
    for model in models_to_test:
        try:
            print(f"🔄 Testing {model}...")
            client = get_llm_client("gemini", model=model)
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
# Example 9: Simple Chat Interface
# =============================================================================

async def simple_chat_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface")
    print("=" * 60)
    
    client = get_llm_client("gemini", model=model)
    
    # Simulate a simple conversation
    conversation = [
        "Hello! How can you help me today?",
        "What's the most interesting thing about Google's AI research?", 
        "Can you help me write a Python function to calculate fibonacci numbers?"
    ]
    
    messages = []
    
    for user_input in conversation:
        print(f"👤 User: {user_input}")
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = await client.create_completion(messages)
        assistant_response = response.get("response", "No response")
        
        print(f"🤖 Gemini: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 10: Model Information
# =============================================================================

async def model_info_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """Get detailed model information"""
    print(f"\n📋 Model Information for {model}")
    print("=" * 60)
    
    client = get_llm_client("gemini", model=model)
    
    # Get model info from client
    if hasattr(client, 'get_model_info'):
        info = client.get_model_info()
        print("🔍 Model details:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    # Get capability info
    model_info = CapabilityChecker.get_model_info("gemini", model)
    print(f"\n🎯 Capabilities:")
    for key, value in model_info.items():
        if key != "error":
            print(f"   {key}: {value}")
    
    return model_info

# =============================================================================
# Example 11: Large Context Test
# =============================================================================

async def large_context_example(model: str = "gemini-2.5-flash-preview-05-20"):
    """Test large context capabilities"""
    print(f"\n📚 Large Context Test with {model}")
    print("=" * 60)
    
    client = get_llm_client("gemini", model=model)
    
    # Create a longer context to test Gemini's large context window
    long_text = """
    Google's Gemini is a family of multimodal large language models developed by Google DeepMind. 
    The models are designed to be highly capable across text, images, audio, and video understanding.
    Gemini models come in different sizes: Ultra, Pro, Flash, and Nano variants.
    They feature advanced reasoning capabilities, multimodal understanding, and efficient processing.
    """ * 100  # Repeat to create longer context
    
    messages = [
        {"role": "user", "content": f"Here's information about Gemini:\n\n{long_text}\n\nPlease summarize the key points about Gemini in 2-3 sentences."}
    ]
    
    print("📝 Processing large context...")
    start_time = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start_time
    
    print(f"✅ Large context response ({duration:.2f}s):")
    print(f"   Input length: ~{len(long_text)} characters")
    print(f"   Summary: {response['response']}")
    
    return response

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Google Gemini Provider Example Script")
    parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20", help="Model to use")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 Google Gemini Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    
    examples = [
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("Model Info", lambda: model_info_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_functions:
            examples.append(("Function Calling", lambda: function_calling_example(args.model)))
        
        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example(args.model)))
        
        examples.extend([
            ("System Instructions", lambda: system_instructions_example(args.model)),
            ("JSON Mode", lambda: json_mode_example(args.model)),
            ("Model Comparison", model_comparison_example),
            ("Multiple Models", multiple_models_example),
            ("Simple Chat", lambda: simple_chat_example(args.model)),
            ("Large Context", lambda: large_context_example(args.model)),
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
        print(f"🔗 Google Gemini provider is working perfectly with chuk-llm!")
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