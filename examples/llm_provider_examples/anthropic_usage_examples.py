#!/usr/bin/env python3
"""
Anthropic/Claude Provider Example Usage Script
==============================================

Demonstrates all the features of the Anthropic provider in the chuk-llm library.
Run this script to see Claude in action with various capabilities.

Requirements:
- pip install anthropic chuk-llm
- Set ANTHROPIC_API_KEY environment variable

Usage:
    python anthropic_example.py
    python anthropic_example.py --model claude-3-5-sonnet-20241022
    python anthropic_example.py --skip-vision
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
if not os.getenv("ANTHROPIC_API_KEY"):
    print("❌ Please set ANTHROPIC_API_KEY environment variable")
    print("   export ANTHROPIC_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.client import get_client
    from chuk_llm.configuration.capabilities import CapabilityChecker
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please make sure you're running from the chuk-llm directory")
    sys.exit(1)

# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================

async def basic_text_example(model: str = "claude-sonnet-4-20250514"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_client("anthropic", model=model)
    
    messages = [
        {"role": "user", "content": "Explain the concept of recursion in programming in simple terms (2-3 sentences)."}
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
    
    # Check if model supports tools first
    can_handle, issues = CapabilityChecker.can_handle_request(
        "anthropic", model, has_tools=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping function calling: {', '.join(issues)}")
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
                            "description": "The text to analyze"
                        },
                        "include_confidence": {
                            "type": "boolean", 
                            "description": "Whether to include confidence score"
                        }
                    },
                    "required": ["text"]
                }
            }
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
                            "description": "The text to summarize"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum length of summary in words"
                        }
                    },
                    "required": ["text"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Analyze the sentiment of this text: 'I absolutely love this new feature!' and then summarize it in 10 words."}
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
            
            if func_name == "analyze_sentiment":
                result = '{"sentiment": "positive", "confidence": 0.95, "score": 0.8}'
            elif func_name == "generate_summary":
                result = '{"summary": "User expresses strong positive sentiment about new feature."}'
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

async def vision_example(model: str = "claude-sonnet-4-20250514"):
    """Vision capabilities with Claude models"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)
    
    # Check if model supports vision
    can_handle, issues = CapabilityChecker.can_handle_request(
        "anthropic", model, has_vision=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping vision: {', '.join(issues)}")
        return None
    
    client = get_client("anthropic", model=model)
    
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
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": test_image
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
# Example 5: System Messages
# =============================================================================

async def system_message_example(model: str = "claude-sonnet-4-20250514"):
    """System message example with different personas"""
    print(f"\n🎭 System Message Example with {model}")
    print("=" * 60)
    
    client = get_client("anthropic", model=model)
    
    # Test different personas
    personas = [
        {
            "name": "Helpful Assistant",
            "system": "You are a helpful, harmless, and honest AI assistant.",
            "query": "How do I bake a cake?"
        },
        {
            "name": "Pirate",
            "system": "You are a friendly pirate captain. Speak like a pirate and use nautical terms.",
            "query": "How do I bake a cake?"
        },
        {
            "name": "Technical Expert",
            "system": "You are a senior software engineer with expertise in Python and system design.",
            "query": "How do I optimize a slow database query?"
        }
    ]
    
    for persona in personas:
        print(f"\n🎭 Testing {persona['name']} persona:")
        
        # For Anthropic, we need to pass system message separately
        # Note: This uses the client's create_completion with system parameter
        messages = [
            {"role": "user", "content": persona["query"]}
        ]
        
        try:
            response = await client.create_completion(
                messages, 
                system=persona["system"],
                max_tokens=150
            )
            print(f"   {response['response'][:200]}...")
        except Exception as e:
            # Fallback: add system message to conversation
            system_messages = [
                {"role": "user", "content": f"System: {persona['system']}\n\nUser: {persona['query']}"}
            ]
            response = await client.create_completion(system_messages, max_tokens=150)
            print(f"   {response['response'][:200]}...")
    
    return True

# =============================================================================
# Example 6: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different Claude models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    models = [
        "claude-opus-4-20250514",          # Latest Claude 4 Opus
        "claude-sonnet-4-20250514",        # Latest Claude 4 Sonnet
        "claude-3-7-sonnet-20250219",      # Claude 3.7 Sonnet
        "claude-3-5-sonnet-20241022",      # Claude 3.5 Sonnet
        "claude-3-5-haiku-20241022",       # Claude 3.5 Haiku
        "claude-3-opus-20240229"           # Claude 3 Opus
    ]
    
    prompt = "What is artificial intelligence? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("anthropic", model=model)
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
    """Test multiple Claude models"""
    print(f"\n🔄 Multiple Models Test")
    print("=" * 60)
    
    models_to_test = [
        "claude-opus-4-20250514",          # Latest Claude 4 Opus  
        "claude-sonnet-4-20250514",        # Latest Claude 4 Sonnet
        "claude-3-7-sonnet-20250219",      # Claude 3.7 Sonnet
        "claude-3-5-sonnet-20241022",      # Claude 3.5 Sonnet
        "claude-3-5-haiku-20241022",       # Claude 3.5 Haiku
        "claude-3-opus-20240229"           # Claude 3 Opus
    ]
    
    prompt = "Write a one-line explanation of machine learning."
    
    for model in models_to_test:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("anthropic", model=model)
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

async def simple_chat_example(model: str = "claude-sonnet-4-20250514"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface")
    print("=" * 60)
    
    client = get_client("anthropic", model=model)
    
    # Simulate a simple conversation
    conversation = [
        "Hello Claude! How are you today?",
        "What's the most interesting thing about AI?", 
        "Can you help me write a Python function to reverse a string?"
    ]
    
    messages = []
    
    for user_input in conversation:
        print(f"👤 User: {user_input}")
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = await client.create_completion(messages)
        assistant_response = response.get("response", "No response")
        
        print(f"🤖 Claude: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 9: Model Information
# =============================================================================

async def model_info_example(model: str = "claude-sonnet-4-20250514"):
    """Get detailed model information"""
    print(f"\n📋 Model Information for {model}")
    print("=" * 60)
    
    client = get_client("anthropic", model=model)
    
    # Get model info from client
    if hasattr(client, 'get_model_info'):
        info = client.get_model_info()
        print("🔍 Model details:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    # Get capability info
    model_info = CapabilityChecker.get_model_info("anthropic", model)
    print(f"\n🎯 Capabilities:")
    for key, value in model_info.items():
        if key != "error":
            print(f"   {key}: {value}")
    
    return model_info

# =============================================================================
# Example 11: Comprehensive Model Benchmark (matches your benchmark script)
# =============================================================================

async def comprehensive_benchmark():
    """Comprehensive benchmark across all Claude models"""
    print(f"\n🏁 Comprehensive Model Benchmark")
    print("=" * 60)
    
    # All models from your benchmark command
    models = [
        "claude-opus-4-20250514",          # Claude 4 Opus
        "claude-sonnet-4-20250514",        # Claude 4 Sonnet  
        "claude-3-7-sonnet-20250219",      # Claude 3.7 Sonnet
        "claude-3-5-sonnet-20241022",      # Claude 3.5 Sonnet
        "claude-3-5-haiku-20241022",       # Claude 3.5 Haiku
        "claude-3-opus-20240229"           # Claude 3 Opus
    ]
    
    # Quick benchmark tasks
    tasks = [
        {
            "name": "Reasoning",
            "prompt": "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?"
        },
        {
            "name": "Creative Writing", 
            "prompt": "Write a creative two-sentence story about a robot discovering emotions."
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function that finds the second largest number in a list."
        }
    ]
    
    results = {}
    
    print("🔄 Running benchmark across all models...")
    print(f"📋 Models: {len(models)}, Tasks: {len(tasks)}")
    
    for model in models:
        print(f"\n🤖 Testing {model}...")
        model_results = {}
        
        try:
            client = get_client("anthropic", model=model)
            
            for task in tasks:
                print(f"   📝 {task['name']}...", end="", flush=True)
                
                messages = [{"role": "user", "content": task["prompt"]}]
                
                start_time = time.time()
                response = await client.create_completion(messages, max_tokens=200)
                duration = time.time() - start_time
                
                model_results[task["name"]] = {
                    "success": True,
                    "time": duration,
                    "length": len(response.get("response", "")),
                    "response": response.get("response", "")[:100] + "..."
                }
                
                print(f" ✅ ({duration:.2f}s)")
            
            results[model] = model_results
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results[model] = {"error": str(e)}
    
    # Display comprehensive results
    print(f"\n📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    # Performance summary table
    print(f"{'Model':<30} {'Reasoning':<12} {'Creative':<12} {'Coding':<12} {'Avg Time':<10}")
    print("-" * 80)
    
    for model, model_results in results.items():
        if "error" in model_results:
            print(f"{model:<30} {'❌ Failed':<40}")
            continue
            
        times = []
        status_icons = []
        
        for task in tasks:
            task_result = model_results.get(task["name"], {})
            if task_result.get("success"):
                times.append(task_result["time"])
                status_icons.append(f"{task_result['time']:.2f}s")
            else:
                status_icons.append("❌")
        
        avg_time = sum(times) / len(times) if times else 0
        
        print(f"{model:<30} {status_icons[0]:<12} {status_icons[1]:<12} {status_icons[2]:<12} {avg_time:.2f}s")
    
    # Speed ranking
    speed_ranking = []
    for model, model_results in results.items():
        if "error" not in model_results:
            times = [r["time"] for r in model_results.values() if r.get("success")]
            if times:
                avg_time = sum(times) / len(times)
                speed_ranking.append((model, avg_time))
    
    speed_ranking.sort(key=lambda x: x[1])
    
    print(f"\n🏃 Speed Ranking (Average Response Time):")
    for i, (model, avg_time) in enumerate(speed_ranking, 1):
        model_short = model.replace("claude-", "").replace("-20250514", "").replace("-20250219", "").replace("-20241022", "").replace("-20240229", "")
        print(f"   {i}. {model_short:<20} {avg_time:.2f}s")
    
    # Show sample responses
    print(f"\n📝 Sample Responses (Reasoning Task):")
    for model, model_results in results.items():
        if "error" not in model_results and "Reasoning" in model_results:
            reasoning = model_results["Reasoning"]
            model_short = model.replace("claude-", "").replace("-20250514", "").replace("-20250219", "").replace("-20241022", "").replace("-20240229", "")
            print(f"\n{model_short}:")
            print(f"   {reasoning['response']}")
    
    return results

async def long_context_example(model: str = "claude-sonnet-4-20250514"):
    """Test long context capabilities"""
    print(f"\n📚 Long Context Test with {model}")
    print("=" * 60)
    
    client = get_client("anthropic", model=model)
    
    # Create a longer context
    long_text = """
    Claude is a family of large language models developed by Anthropic. 
    The models are designed to be helpful, harmless, and honest. 
    Claude can engage in conversations, answer questions, help with analysis and math, 
    create content, help with coding, and much more. 
    """ * 50  # Repeat to create longer context
    
    messages = [
        {"role": "user", "content": f"Here's some text about Claude:\n\n{long_text}\n\nPlease summarize this text in 2-3 sentences."}
    ]
    
    print("📝 Processing long context...")
    start_time = time.time()
    response = await client.create_completion(messages)
    duration = time.time() - start_time
    
    print(f"✅ Long context response ({duration:.2f}s):")
    print(f"   Input length: ~{len(long_text)} characters")
    print(f"   Summary: {response['response']}")
    
    return response

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Anthropic/Claude Provider Example Script")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 Anthropic/Claude Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Missing'}")
    
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
            ("System Messages", lambda: system_message_example(args.model)),
            ("Model Comparison", model_comparison_example),
            ("Multiple Models", multiple_models_example),
            ("Simple Chat", lambda: simple_chat_example(args.model)),
            ("Long Context", lambda: long_context_example(args.model)),
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
        print(f"🔗 Anthropic/Claude provider is working perfectly with chuk-llm!")
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