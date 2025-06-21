#!/usr/bin/env python3
# examples/perplexity_usage_examples.py
"""
Perplexity Provider Example Usage Script
========================================

Demonstrates all the features of the Perplexity provider in the chuk-llm library.
Perplexity uses OpenAI-compatible API but with enhanced search and reasoning capabilities.

Prerequisites:
- pip install openai chuk-llm python-dotenv
- Set PERPLEXITY_API_KEY environment variable

Usage:
    python perplexity_example.py
    python perplexity_example.py --model sonar-pro
    python perplexity_example.py --skip-functions
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
if not os.getenv("PERPLEXITY_API_KEY"):
    print("❌ Please set PERPLEXITY_API_KEY environment variable")
    print("   export PERPLEXITY_API_KEY='your_api_key_here'")
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

async def basic_text_example(model: str = "sonar-pro"):
    """Basic text completion example"""
    print(f"\n🤖 Basic Text Completion with {model}")
    print("=" * 60)

    # Use OpenAI client with Perplexity API base as fallback
    try:
        client = get_client("perplexity", model=model)
    except Exception as e:
        print(f"⚠️  Perplexity provider not configured, using OpenAI client with Perplexity API base")
        from chuk_llm.llm.providers.openai_client import OpenAILLMClient
        client = OpenAILLMClient(
            model=model,
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            api_base="https://api.perplexity.ai"
        )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with access to current information."},
        {"role": "user", "content": "Explain transformers in simple terms (2-3 sentences)."},
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

async def streaming_example(model: str = "sonar-pro"):
    """Real-time streaming example"""
    print(f"\n⚡ Streaming Example with {model}")
    print("=" * 60)

    # Check streaming support
    config = get_config()
    if not config.supports_feature("perplexity", Feature.STREAMING, model):
        print(f"⚠️  Model {model} doesn't support streaming")
        return None

    client = get_client("perplexity", model=model)
    
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
# Example 3: Current Information Search
# =============================================================================

async def current_info_example(model: str = "sonar-pro"):
    """Test Perplexity's ability to access current information"""
    print(f"\n🔍 Current Information Search with {model}")
    print("=" * 60)
    
    client = get_client("perplexity", model=model)
    
    # Test current information capabilities
    messages = [
        {
            "role": "user", 
            "content": "What are the latest developments in AI this week? Please provide recent, specific examples."
        }
    ]
    
    print("🔄 Searching for current information...")
    start_time = time.time()
    response = await client.create_completion(messages, max_tokens=300)
    duration = time.time() - start_time
    
    print(f"✅ Current info response ({duration:.2f}s):")
    print(f"   {response['response']}")
    
    return response

# =============================================================================
# Example 4: Function Calling
# =============================================================================

async def function_calling_example(model: str = "sonar-pro"):
    """Function calling with tools (if supported)"""
    print(f"\n🔧 Function Calling with {model}")
    print("=" * 60)

    # Check if model supports tools
    config = get_config()
    if not config.supports_feature("perplexity", Feature.TOOLS, model):
        print(f"⚠️  Skipping function calling: Model {model} doesn't support tools")
        print(f"💡 Perplexity models may not support function calling")
        return None

    client = get_client("perplexity", model=model)

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
            "content": "Search for 'LLM eval benchmarks 2025' and calculate 3.14159 * 42 with 2 decimal places.",
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
                result = '{"results": ["MMLU Benchmark 2025", "HellaSwag Updated", "GSM8K Advanced"]}'
            elif func_name == "calculate_math":
                result = '{"result": 131.95, "expression": "3.14159 * 42", "precision": 2}'
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

async def json_mode_example(model: str = "sonar-pro"):
    """JSON mode example using response_format"""
    print(f"\n📋 JSON Mode Example with {model}")
    print("=" * 60)

    # Check JSON mode support
    config = get_config()
    if not config.supports_feature("perplexity", Feature.JSON_MODE, model):
        print(f"⚠️  Model {model} doesn't support JSON mode")
        return None

    client = get_client("perplexity", model=model)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs JSON only.",
        },
        {
            "role": "user",
            "content": "Give me a JSON object with information about the latest AI company that raised funding. Include fields: company_name, funding_amount, funding_round, investors (array), and description.",
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
    """Compare different Perplexity models"""
    print(f"\n📊 Model Comparison")
    print("=" * 60)

    # Current Perplexity models (based on official tier info)
    models = [
        "sonar-pro",                # 50 RPM, full features
        "sonar-reasoning",          # 50 RPM, reasoning + full features  
        "sonar-reasoning-pro",      # 50 RPM, premium reasoning + full features
        "r1-1776"                   # 50 RPM, offline chat (no search)
        # Note: sonar-deep-research has only 5 RPM, so skipping for comparison
    ]
    
    prompt = "What is the current state of autonomous vehicles? (One sentence)"
    results = {}

    for model in models:
        try:
            print(f"🔄 Testing {model}...")
            client = get_client("perplexity", model=model)
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
        model_short = model.replace("sonar-", "").replace("reasoning-", "r-")
        print(f"   {status} {model_short}:")
        print(f"      Time: {result['time']:.2f}s")
        print(f"      Length: {result['length']} chars")
        print(f"      Response: {result['response'][:80]}...")
        print()
    
    return results

# =============================================================================
# Example 7: Feature Detection
# =============================================================================

async def feature_detection_example(model: str = "sonar-pro"):
    """Detect and display model features"""
    print(f"\n🔬 Feature Detection for {model}")
    print("=" * 60)
    
    # Get model info
    try:
        model_info = get_provider_info("perplexity", model)
        
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
        client = get_client("perplexity", model=model)
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

async def simple_chat_example(model: str = "sonar-pro"):
    """Simple chat interface simulation"""
    print(f"\n💬 Simple Chat Interface with {model}")
    print("=" * 60)

    client = get_client("perplexity", model=model)
    
    # Simulate a conversation focusing on current information
    conversation = [
        "Hello! What's the current weather situation globally?",
        "What are the most important tech news stories this week?",
        "Can you help me understand the latest developments in quantum computing?",
    ]

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with access to current, up-to-date information."}
    ]
    
    for user_input in conversation:
        print(f"👤 User: {user_input}")
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = await client.create_completion(messages, max_tokens=200)
        assistant_response = response.get("response", "No response")
        
        print(f"🤖 Perplexity: {assistant_response}")
        print()
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages

# =============================================================================
# Example 9: Temperature Sweep
# =============================================================================

async def parameters_example(model: str = "sonar-pro"):
    """Test different temperature settings"""
    print(f"\n🎛️  Temperature Sweep with {model}")
    print("=" * 60)

    client = get_client("perplexity", model=model)
    prompt = "Write a creative opening line for a science-fiction story about AI consciousness."
    
    for temp in [0.1, 0.7, 1.2]:
        print(f"\n🌡️  Temperature {temp}:")
        response = await client.create_completion(
            [{"role": "user", "content": prompt}], 
            temperature=temp, 
            max_tokens=100
        )
        print(f"   {response['response']}")
    
    return True

# =============================================================================
# Main Function
# =============================================================================

async def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="Perplexity Provider Example Script")
    parser.add_argument("--model", default="sonar-pro", help="Model to use (default: sonar-pro)")
    parser.add_argument("--skip-functions", action="store_true", help="Skip function calling")
    parser.add_argument("--test-search", action="store_true", help="Focus on search capabilities")
    parser.add_argument("--quick", action="store_true", help="Run only basic examples")
    
    args = parser.parse_args()
    
    print("🚀 Perplexity Provider Examples")
    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"API Key: {'✅ Set' if os.getenv('PERPLEXITY_API_KEY') else '❌ Missing'}")
    
    # Show model capabilities
    try:
        config = get_config()
        supports_streaming = config.supports_feature("perplexity", Feature.STREAMING, args.model)
        supports_json = config.supports_feature("perplexity", Feature.JSON_MODE, args.model)
        
        print(f"Model capabilities:")
        print(f"  Streaming: {'✅' if supports_streaming else '❌'}")
        print(f"  JSON Mode: {'✅' if supports_json else '❌'}")
        print(f"  Current Info: ✅ (Perplexity specialty)")
        
    except Exception as e:
        print(f"⚠️  Could not check capabilities: {e}")

    # Focus on search if requested
    if args.test_search:
        await current_info_example(args.model)
        return

    examples = [
        ("Feature Detection", lambda: feature_detection_example(args.model)),
        ("Basic Text", lambda: basic_text_example(args.model)),
        ("Streaming", lambda: streaming_example(args.model)),
        ("Current Information", lambda: current_info_example(args.model)),
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
        print(f"🔗 Perplexity provider is working perfectly with chuk-llm!")
        print(f"✨ Features tested: {args.model} capabilities")
    else:
        print(f"\n⚠️  Some examples failed. Check your API key and model access.")
        
        # Show model recommendations
        print(f"\n💡 Model Recommendations:")
        print(f"   • For research: sonar-deep-research (Tier 0+, 5-50 RPM)")
        print(f"   • For reasoning: sonar-reasoning-pro, sonar-reasoning")
        print(f"   • For search: sonar-pro, sonar")
        print(f"   • For offline chat: r1-1776 (no search)")
        print(f"   • Current models: sonar-pro, sonar-reasoning, sonar-reasoning-pro")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)