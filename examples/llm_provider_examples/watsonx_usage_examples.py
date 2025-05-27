#!/usr/bin/env python3
# examples/watsonx_usage_examples.py
"""
Watson X Provider Example Usage Script
=====================================

Demonstrates all the features of the Watson X provider in the chuk-llm library.
Run this script to see Watson X in action with various capabilities.

Requirements:
- pip install ibm-watsonx-ai chuk-llm
- Set WATSONX_API_KEY (or IBM_CLOUD_API_KEY) environment variable
- Set WATSONX_PROJECT_ID environment variable
- Set WATSONX_AI_URL environment variable (optional, defaults to us-south)

Usage:
    python watsonx_example.py
    python watsonx_example.py --model ibm/granite-3-8b-instruct
    python watsonx_example.py --skip-tools
"""

import asyncio
import argparse
import os
import sys
import time
from typing import Dict, Any, List

# dotenv
from dotenv import load_dotenv

# load environment variables
load_dotenv() 

# Ensure we have the required environment
if not (os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")):
    print("❌ Please set WATSONX_API_KEY or IBM_CLOUD_API_KEY environment variable")
    print("   export WATSONX_API_KEY='your_api_key_here'")
    print("   # or")
    print("   export IBM_CLOUD_API_KEY='your_ibm_cloud_api_key_here'")
    sys.exit(1)

if not os.getenv("WATSONX_PROJECT_ID"):
    print("❌ Please set WATSONX_PROJECT_ID environment variable")
    print("   export WATSONX_PROJECT_ID='your_project_id_here'")
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

async def basic_text_example(model: str = "ibm/granite-3-8b-instruct"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)
    
    client = get_llm_client("watsonx", model=model)
    
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

async def streaming_example(model: str = "ibm/granite-3-8b-instruct"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("watsonx", model=model)
    
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

async def function_calling_example(model: str = "ibm/granite-3-8b-instruct"):
    """Function calling with tools"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)
    
    # Check if model supports tools first
    can_handle, issues = CapabilityChecker.can_handle_request(
        "watsonx", model, has_tools=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping function calling: {', '.join(issues)}")
        return None
    
    client = get_llm_client("watsonx", model=model)
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Adds the values a and b to get a sum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "description": "A number value",
                            "type": "number"
                        },
                        "b": {
                            "description": "A number value", 
                            "type": "number"
                        }
                    },
                    "required": ["a", "b"]
                }
            }
        },
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
        }
    ]
    
    messages = [
        {"role": "user", "content": "What is 2 plus 4? And also analyze the sentiment of this text: 'I love Watson X!'"}
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
            
            if func_name == "add":
                result = '{"sum": 6}'
            elif func_name == "analyze_sentiment":
                result = '{"sentiment": "positive", "confidence": 0.95, "score": 0.9}'
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
# Example 4: System Messages
# =============================================================================

async def system_message_example(model: str = "ibm/granite-3-8b-instruct"):
    """System message example with different personas"""
    print(f"\n🎭 System Message Example with {model}")
    print("=" * 60)
    
    client = get_llm_client("watsonx", model=model)
    
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
        
        messages = [
            {"role": "system", "content": persona["system"]},
            {"role": "user", "content": persona["query"]}
        ]
        
        try:
            response = await client.create_completion(messages, max_tokens=150)
            print(f"   {response['response'][:200]}...")
        except Exception as e:
            print(f"   Error: {e}")
    
    return True

# =============================================================================
# Example 5: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different Watson X models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)
    
    models = [
        "ibm/granite-3-8b-instruct",               # IBM Granite - very reliable
        "meta-llama/llama-3-2-1b-instruct",        # Llama 3.2 1B - fast
        "meta-llama/llama-3-2-3b-instruct",        # Llama 3.2 3B - good balance  
        "meta-llama/llama-3-3-70b-instruct",       # Llama 3.3 70B - most capable
        "mistralai/mistral-large",                 # Mistral Large - enterprise
    ]
    
    prompt = "What is artificial intelligence? (One sentence)"
    results = {}
    
    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_llm_client("watsonx", model=model)
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
# Example 6: Simple Chat Interface
# =============================================================================

async def simple_chat_example(model: str = "ibm/granite-3-8b-instruct"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface")
    print("=" * 60)
    
    client = get_llm_client("watsonx", model=model)
    
    # Simulate a simple conversation
    conversation = [
        "Hello Watson X! How are you today?",
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
        
        print(f"🤖 Watson X: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 7: Model Information
# =============================================================================

async def model_info_example(model: str = "ibm/granite-3-8b-instruct"):
    """Get detailed model information"""
    print(f"\n📋 Model Information for {model}")
    print("=" * 60)
    
    client = get_llm_client("watsonx", model=model)
    
    # Get model info from client
    if hasattr(client, 'get_model_info'):
        info = client.get_model_info()
        print("🔍 Model details:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    # Get capability info
    model_info = CapabilityChecker.get_model_info("watsonx", model)
    print(f"\n🎯 Capabilities:")
    for key, value in model_info.items():
        if key != "error":
            print(f"   {key}: {value}")
    
    return model_info

# =============================================================================
# Example 8: Comprehensive Model Benchmark
# =============================================================================

async def comprehensive_benchmark():
    """Comprehensive benchmark across all Watson X models"""
    print(f"\n🏁 Comprehensive Model Benchmark")
    print("=" * 60)
    
    # All major Watson X models (using non-deprecated, reliable versions)
    models = [
        "ibm/granite-3-8b-instruct",               # IBM Granite - very reliable
        "meta-llama/llama-3-2-1b-instruct",        # Llama 3.2 1B - fast and reliable
        "meta-llama/llama-3-2-3b-instruct",        # Llama 3.2 3B - good performance
        "meta-llama/llama-3-3-70b-instruct",       # Llama 3.3 70B - latest, powerful
        "mistralai/mistral-large",                 # Mistral Large - reliable
        # Note: Excluding vision model due to timeout issues
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
            client = get_llm_client("watsonx", model=model)
            
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
    print(f"{'Model':<35} {'Reasoning':<12} {'Creative':<12} {'Coding':<12} {'Avg Time':<10}")
    print("-" * 85)
    
    for model, model_results in results.items():
        if "error" in model_results:
            print(f"{model:<35} {'❌ Failed':<40}")
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
        
        print(f"{model:<35} {status_icons[0]:<12} {status_icons[1]:<12} {status_icons[2]:<12} {avg_time:.2f}s")
    
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
        model_short = model.replace("meta-llama/", "").replace("ibm/", "").replace("mistralai/", "")
        print(f"   {i}. {model_short:<25} {avg_time:.2f}s")
    
    # Show sample responses
    print(f"\n📝 Sample Responses (Reasoning Task):")
    for model, model_results in results.items():
        if "error" not in model_results and "Reasoning" in model_results:
            reasoning = model_results["Reasoning"]
            model_short = model.replace("meta-llama/", "").replace("ibm/", "").replace("mistralai/", "")
            print(f"\n{model_short}:")
            print(f"   {reasoning['response']}")
    
    return results

# =============================================================================
# Example 9: Long Context Test
# =============================================================================

async def long_context_example(model: str = "ibm/granite-3-8b-instruct"):
    """Test long context capabilities"""
    print(f"\n📚 Long Context Test with {model}")
    print("=" * 60)
    
    client = get_llm_client("watsonx", model=model)
    
    # Create a longer context
    long_text = """
    Watson X is IBM's AI and data platform that brings together new generative AI capabilities
    powered by foundation models and traditional machine learning into a powerful studio
    spanning the AI lifecycle. Watson X enables organizations to scale and accelerate the
    impact of AI with trusted data across the business. The platform includes three core
    components: the watsonx.ai studio for new foundation models, generative AI and machine
    learning; watsonx.data, a fit-for-purpose data store built on an open lake house
    architecture; and watsonx.governance, a toolkit to enable AI workflows that are built
    with responsibility, transparency and explainability.
    """ * 20  # Repeat to create longer context
    
    messages = [
        {"role": "user", "content": f"Here's some text about Watson X:\n\n{long_text}\n\nPlease summarize this text in 2-3 sentences."}
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
# Example 10: Vision Test (if supported)
# =============================================================================

async def vision_example(model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    """Vision capabilities test with Watson X models"""
    print(f"\n👁️  Vision Example with {model}")
    print("=" * 60)
    
    # Check if model supports vision
    can_handle, issues = CapabilityChecker.can_handle_request(
        "watsonx", model, has_vision=True
    )
    
    if not can_handle:
        print(f"⚠️  Skipping vision: {', '.join(issues)}")
        return None
    
    client = get_llm_client("watsonx", model=model)
    
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
    try:
        response = await client.create_completion(messages)
        print(f"✅ Vision response:")
        print(f"   {response['response']}")
        return response
    except Exception as e:
        print(f"❌ Vision test failed: {e}")
        return None

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Watson X Provider Example Script")
    parser.add_argument("--model", default="ibm/granite-3-8b-instruct", help="Model to use")
    parser.add_argument("--skip-tools", action="store_true", help="Skip function calling")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision examples")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 Watson X Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if (os.getenv('WATSONX_API_KEY') or os.getenv('IBM_CLOUD_API_KEY')) else '❌ Missing'}")
    print(f"Project ID: {'✅ Set' if os.getenv('WATSONX_PROJECT_ID') else '❌ Missing'}")
    print(f"Watson X URL: {os.getenv('WATSONX_AI_URL', 'https://us-south.ml.cloud.ibm.com (default)')}")
    
    examples = [
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("Model Info", lambda: model_info_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_tools:
            examples.append(("Function Calling", lambda: function_calling_example("ibm/granite-3-8b-instruct")))
        
        if not args.skip_vision:
            examples.append(("Vision", lambda: vision_example("meta-llama/llama-4-scout-17b-16e-instruct")))
        
        examples.extend([
            ("System Messages", lambda: system_message_example(args.model)),
            ("Model Comparison", model_comparison_example),
            ("Simple Chat", lambda: simple_chat_example(args.model)),
            ("Long Context", lambda: long_context_example(args.model)),
            ("Comprehensive Benchmark", comprehensive_benchmark),
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
        print(f"🔗 Watson X provider is working perfectly with chuk-llm!")
    else:
        print(f"\n⚠️  Some examples failed. Check your API key, project ID, and model access.")
        
    # Tips for common issues
    print(f"\n💡 Tips:")
    print(f"   • Make sure you have access to the models you're trying to use")
    print(f"   • Some models may not support all features (tools, vision, etc.)")
    print(f"   • Check your Watson X project permissions and quotas")
    print(f"   • For function calling, try ibm/granite-3-8b-instruct or mistralai/mistral-large")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)