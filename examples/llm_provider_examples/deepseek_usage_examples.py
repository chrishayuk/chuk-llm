#!/usr/bin/env python3
# examples/deepseek_usage_examples.py
"""
DeepSeek Provider Example Usage Script
======================================

Demonstrates all features of the DeepSeek provider including reasoning capabilities.
DeepSeek uses OpenAI-compatible API but with enhanced reasoning models.

Prerequisites
-------------
1.  `pip install openai chuk-llm python-dotenv`
2.  Export your DeepSeek API key:

        export DEEPSEEK_API_KEY="sk-…"

Usage
-----
    python deepseek_usage_examples.py
    python deepseek_usage_examples.py --model deepseek-reasoner
    python deepseek_usage_examples.py --test-reasoning
"""

import asyncio
import argparse
import os
import sys
import time
from typing import Dict, Any, List

from dotenv import load_dotenv

load_dotenv()

# Environment sanity check
if not os.getenv("DEEPSEEK_API_KEY"):
    print("❌ Please set DEEPSEEK_API_KEY environment variable")
    print("   export DEEPSEEK_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.client import get_client, get_provider_info
    from chuk_llm.configuration import get_config, Feature
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you installed chuk-llm and are running from the repo root")
    sys.exit(1)

# =============================================================================
# Example 1: Basic Text Completion
# =============================================================================

async def basic_text_example(model: str = "deepseek-reasoner"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    client = get_client("deepseek", model=model)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain neural networks in simple terms (2-3 sentences)."},
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

async def streaming_example(model: str = "deepseek-reasoner"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    # Check streaming support
    config = get_config()
    if not config.supports_feature("deepseek", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

    client = get_client("deepseek", model=model)
    
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
# Example 3: Reasoning Capabilities
# =============================================================================

async def reasoning_example(model: str = "deepseek-reasoner"):
    """Test enhanced reasoning capabilities"""
    print(f"\n🧠 Reasoning Example with {model}")
    print("=" * 60)
    
    # Check if model supports reasoning
    config = get_config()
    if not config.supports_feature("deepseek", Feature.REASONING, model):
        print(f"⚠️  Model {model} doesn't have enhanced reasoning")
        return None

    client = get_client("deepseek", model=model)

    # Use a complex reasoning task that the model should respond to
    messages = [
        {
            "role": "user", 
            "content": "I have a 3-gallon jug and a 5-gallon jug. I need to measure exactly 4 gallons of water. How can I do this? Please think through this step-by-step and explain your reasoning process."
        }
    ]

    print("🧠 Processing complex reasoning task...")
    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=500)
    duration = time.time() - start_time

    # Handle empty responses
    response_text = response.get('response', '').strip()
    if not response_text:
        print(f"⚠️  Empty response received ({duration:.2f}s)")
        print(f"   Note: deepseek-reasoner may only respond to complex reasoning tasks")
        return response
    
    print(f"✅ Reasoning response ({duration:.2f}s):")
    print(f"   {response_text}")
    
    return response

# =============================================================================
# Example 4: Function Calling
# =============================================================================

async def function_calling_example(model: str = "deepseek-chat"):
    """Function calling with tools (if supported)"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    # Check if model supports tools
    config = get_config()
    if not config.supports_feature("deepseek", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        print(f"💡 DeepSeek models may not support function calling yet")
        return None

    client = get_client("deepseek", model=model)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_math",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "precision": {"type": "integer"},
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
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": response["tool_calls"]
        })
        
        # Add mock tool results
        for tool_call in response["tool_calls"]:
            func_name = tool_call["function"]["name"]
            
            if func_name == "search_web":
                result = '{"results": ["Paper A: Neural Architecture Search", "Paper B: Transformer Improvements", "Paper C: Multimodal Learning"]}'
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
# Example 5: JSON Mode
# =============================================================================

async def json_mode_example(model: str = "deepseek-chat"):
    """JSON mode example using response_format"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)

    # Check JSON mode support
    config = get_config()
    if not config.supports_feature("deepseek", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None

    client = get_client("deepseek", model=model)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs JSON only.",
        },
        {
            "role": "user",
            "content": "Give me a JSON object with fields name, year_created, creator "
            "and a features array describing Python programming language.",
        },
    ]

    try:
        response = await client.create_completion(
            messages, 
            response_format={"type": "json_object"}, 
            temperature=0.7
        )
        print("✅ JSON response:")
        print(f"   {response['response']}")
        
        # Try to validate JSON
        import json
        try:
            json_data = json.loads(response['response'])
            print("✅ Valid JSON structure confirmed")
            print(f"   Keys: {list(json_data.keys())}")
        except json.JSONDecodeError:
            print("⚠️  Response is not valid JSON")
            
    except Exception as e:
        print(f"❌ JSON mode failed: {e}")
        # Fallback to regular completion
        print("📝 Trying fallback without JSON mode...")
        response = await client.create_completion(messages)
        print(f"   Fallback response: {response['response'][:200]}...")
    
    return response

# =============================================================================
# Example 6: Model Comparison
# =============================================================================

async def model_comparison_example():
    """Compare different DeepSeek models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)

    models = ["deepseek-chat", "deepseek-reasoner"]
    prompt = "What is the most significant challenge in developing artificial general intelligence? (one sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("deepseek", model=model)
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
        model_short = model.replace("deepseek-", "")
        print(f"   {status} {model_short}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()
    
    return results

# =============================================================================
# Example 7: Feature Detection
# =============================================================================

async def feature_detection_example(model: str = "deepseek-reasoner"):
    """Detect and display model features"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)
    
    # Get model info
    try:
        model_info = get_provider_info("deepseek", model)
        
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
        client = get_client("deepseek", model=model)
        client_info = client.get_model_info()
        
        print(f"\n🔧 Client Features:")
        print(f"   Streaming: {'✅' if client_info.get('supports_streaming') else '❌'}")
        print(f"   JSON Mode: {'✅' if client_info.get('supports_json_mode') else '❌'}")
        print(f"   System Messages: {'✅' if client_info.get('supports_system_messages') else '❌'}")
        
    except Exception as e:
        print(f"⚠️  Could not get client info: {e}")
    
    return model_info if 'model_info' in locals() else None

# =============================================================================
# Example 8: Simple Chat Interface
# =============================================================================

async def simple_chat_example(model: str = "deepseek-chat"):
    """Simple chat interface simulation - use chat model for conversations"""
    print(f"\n💬 Simple Chat Interface with {model}")
    print("=" * 60)

    client = get_client("deepseek", model=model)
    
    # Simulate a conversation - use more complex questions for reasoner model
    if "reasoner" in model:
        conversation = [
            "What are the philosophical implications of artificial intelligence becoming more capable than humans?",
            "How would you solve the problem of ensuring AI systems remain aligned with human values as they become more powerful?",
            "What are the key differences between reasoning-based AI and traditional language models in terms of their cognitive processes?",
        ]
    else:
        conversation = [
            "Hello! What makes DeepSeek special?",
            "What's the most exciting development in AI recently?",
            "Can you help me understand the difference between reasoning and regular language models?",
        ]

    messages = [
        {"role": "system", "content": "You are a helpful and knowledgeable AI assistant."}
    ]
    
    for user_input in conversation:
        print(f"👤 User: {user_input}")
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = await client.create_completion(messages, max_tokens=300)
        assistant_response = response.get("response", "").strip()
        
        # Handle empty responses
        if not assistant_response:
            assistant_response = f"[No response - {model} may need more complex questions]"
            print(f"⚠️  Empty response for: {user_input[:50]}...")
        
        print(f"🤖 Assistant: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 9: Temperature Sweep
# =============================================================================

async def parameters_example(model: str = "deepseek-chat"):
    """Test different temperature settings - use chat model for simple prompts"""
    print(f"\n🎛️  Temperature Sweep with {model}")
    print("=" * 60)

    client = get_client("deepseek", model=model)
    
    # Use different prompts based on model type
    if "reasoner" in model:
        prompt = "Analyze the philosophical implications of this scenario: What would happen if we could perfectly simulate human consciousness in a computer? Consider the ethical, existential, and practical aspects."
    else:
        prompt = "Write a creative opening line for a science-fiction story."
    
    for temp in [0.1, 0.7, 1.2]:
        print(f"\n🌡️  Temperature {temp}:")
        response = await client.create_completion(
            [{"role": "user", "content": prompt}], 
            temperature=temp, 
            max_tokens=150
        )
        
        response_text = response.get("response", "").strip()
        if not response_text:
            response_text = f"[No response - try more complex prompt for {model}]"
            print(f"   ⚠️  Empty response at temperature {temp}")
        
        print(f"   {response_text}")
    
    return True

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="DeepSeek Provider Example Script")
    parser.add_argument("--model", default="deepseek-reasoner", help="Model to use (default: deepseek-reasoner)")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--test-reasoning", action="store_true", help="Focus on reasoning capabilities")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 DeepSeek Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('DEEPSEEK_API_KEY') else '❌ Missing'}")
    
    # Show model capabilities
    try:
        config = get_config()
        supports_reasoning = config.supports_feature("deepseek", Feature.REASONING, args.model)
        supports_streaming = config.supports_feature("deepseek", Feature.STREAMING, args.model)
        supports_json = config.supports_feature("deepseek", Feature.JSON_MODE, args.model)
        
        print(f"Model capabilities:")
        print(f"  Reasoning: {'✅' if supports_reasoning else '❌'}")
        print(f"  Streaming: {'✅' if supports_streaming else '❌'}")
        print(f"  JSON Mode: {'✅' if supports_json else '❌'}")
        
    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    # Focus on reasoning if requested
    if args.test_reasoning:
        await reasoning_example(args.model)
        return

    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("Reasoning", lambda: reasoning_example(args.model)),
        ("JSON Mode", lambda: json_mode_example(args.model)),
    ]
    
    if not args.quick:
        if not args.skip_functions:
            examples.append(("Function Calling", lambda: function_calling_example(args.model)))
        
        examples.extend([
            ("Model Comparison", model_comparison_example),
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
        print(f"🔗 DeepSeek provider is working perfectly with chuk-llm!")
        print(f"✨ Features tested: {args.model} capabilities")
    else:
        print(f"\n⚠️  Some examples failed. Check your API key and model access.")
        
        # Show model recommendations
        print(f"\n💡 Model Recommendations:")
        print(f"   • For reasoning: deepseek-reasoner (enhanced thinking)")
        print(f"   • For general use: deepseek-chat (faster, cost-effective)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)