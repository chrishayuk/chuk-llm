#!/usr/bin/env python3
"""
Gemini OpenAI Compatibility Example
====================================

Demonstrates using Google Gemini through the OpenAI-compatible API endpoint.
This allows you to use Gemini with any OpenAI-compatible library or tool.

Based on: https://ai.google.dev/gemini-api/docs/openai

Key Features:
- Drop-in replacement for OpenAI API
- Use existing OpenAI SDK with Gemini models
- Compatible with LangChain, LlamaIndex, and other tools
- Supports chat completions, streaming, embeddings, and more

Requirements:
- pip install openai
- Set GEMINI_API_KEY environment variable

Usage:
    python gemini_openai_compatibility.py
    python gemini_openai_compatibility.py --test-all
    python gemini_openai_compatibility.py --compare
"""

import os
import sys
import time
import json
import asyncio
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check for API key
if not os.getenv("GEMINI_API_KEY"):
    print("❌ Please set GEMINI_API_KEY environment variable")
    print("   export GEMINI_API_KEY='your_api_key_here'")
    print("   Get your key at: https://aistudio.google.com/apikey")
    sys.exit(1)

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    print("❌ OpenAI SDK not installed")
    print("   pip install openai")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

# Gemini OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Available Gemini models through OpenAI API
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b", 
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
]

# Create clients
def create_gemini_client():
    """Create OpenAI client configured for Gemini"""
    return OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=GEMINI_BASE_URL
    )

def create_gemini_async_client():
    """Create async OpenAI client configured for Gemini"""
    return AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=GEMINI_BASE_URL
    )

# =============================================================================
# Example 1: Basic Chat Completion
# =============================================================================

def basic_chat_example():
    """Basic chat completion using Gemini through OpenAI API"""
    print("\n💬 Basic Chat Completion")
    print("=" * 60)
    
    client = create_gemini_client()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the concept of machine learning in simple terms."}
    ]
    
    print("📝 Sending chat completion request...")
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    
    duration = time.time() - start_time
    
    print(f"✅ Response received in {duration:.2f}s:")
    print(f"   {response.choices[0].message.content}")
    print(f"\n📊 Token Usage:")
    print(f"   Prompt tokens: {response.usage.prompt_tokens}")
    print(f"   Completion tokens: {response.usage.completion_tokens}")
    print(f"   Total tokens: {response.usage.total_tokens}")
    
    return response

# =============================================================================
# Example 2: Streaming Response
# =============================================================================

def streaming_example():
    """Streaming chat completion"""
    print("\n🌊 Streaming Chat Completion")
    print("=" * 60)
    
    client = create_gemini_client()
    
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence"}
    ]
    
    print("📝 Streaming response:")
    print("   ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print("\n\n✅ Streaming completed")
    return full_response

# =============================================================================
# Example 3: Function Calling
# =============================================================================

def function_calling_example():
    """Function calling with tools"""
    print("\n🔧 Function Calling")
    print("=" * 60)
    
    client = create_gemini_client()
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g., 'London, UK'"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
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
                            "description": "Search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather like in Tokyo and search for recent news about AI?"}
    ]
    
    print("📝 Requesting function calls...")
    
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    # Check for tool calls
    if response.choices[0].message.tool_calls:
        print(f"✅ Function calls requested: {len(response.choices[0].message.tool_calls)}")
        
        for tool_call in response.choices[0].message.tool_calls:
            print(f"\n   Function: {tool_call.function.name}")
            print(f"   Arguments: {tool_call.function.arguments}")
        
        # Simulate function execution
        messages.append(response.choices[0].message)
        
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == "get_weather":
                result = {"temperature": 22, "unit": "celsius", "condition": "sunny"}
            elif tool_call.function.name == "search_web":
                result = {"results": ["AI breakthrough in 2024", "New LLM releases"]}
            else:
                result = {"status": "completed"}
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # Get final response
        print("\n📝 Getting final response with tool results...")
        final_response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=messages
        )
        
        print(f"✅ Final response:")
        print(f"   {final_response.choices[0].message.content}")
        
        return final_response
    else:
        print("ℹ️  No function calls made")
        print(f"   Response: {response.choices[0].message.content}")
        return response

# =============================================================================
# Example 4: Multi-turn Conversation
# =============================================================================

def conversation_example():
    """Multi-turn conversation with context"""
    print("\n💭 Multi-turn Conversation")
    print("=" * 60)
    
    client = create_gemini_client()
    
    # Initialize conversation
    messages = [
        {"role": "system", "content": "You are a knowledgeable science teacher."}
    ]
    
    # Conversation turns
    turns = [
        "What is photosynthesis?",
        "What role does chlorophyll play in this process?",
        "How is this different from cellular respiration?"
    ]
    
    for i, user_input in enumerate(turns, 1):
        print(f"\n👤 Turn {i}: {user_input}")
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=messages,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        print(f"🤖 Response: {assistant_response[:200]}...")
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_response})
    
    print("\n✅ Conversation completed with context maintained")
    return messages

# =============================================================================
# Example 5: JSON Mode
# =============================================================================

def json_mode_example():
    """Structured JSON output"""
    print("\n📋 JSON Mode Output")
    print("=" * 60)
    
    client = create_gemini_client()
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs JSON."
        },
        {
            "role": "user",
            "content": """Create a JSON object for a book with the following fields:
            - title (string)
            - author (string)
            - year (integer)
            - genres (array of strings)
            - rating (number between 0 and 5)
            - bestseller (boolean)
            
            Make up a science fiction book."""
        }
    ]
    
    print("📝 Requesting JSON output...")
    
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    
    try:
        # Parse JSON
        book_data = json.loads(content)
        print("✅ Valid JSON received:")
        print(json.dumps(book_data, indent=2))
        
        # Validate structure
        required_fields = ["title", "author", "year", "genres", "rating", "bestseller"]
        missing_fields = [f for f in required_fields if f not in book_data]
        
        if missing_fields:
            print(f"⚠️  Missing fields: {missing_fields}")
        else:
            print("✅ All required fields present")
        
        return book_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"   Raw response: {content[:200]}...")
        return None

# =============================================================================
# Example 6: Different Models Comparison
# =============================================================================

def model_comparison_example():
    """Compare different Gemini models through OpenAI API"""
    print("\n📊 Model Comparison")
    print("=" * 60)
    
    client = create_gemini_client()
    
    prompt = "Explain quantum entanglement in one sentence."
    
    results = {}
    
    for model in GEMINI_MODELS:
        try:
            print(f"\n🔄 Testing {model}...")
            
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            duration = time.time() - start_time
            
            content = response.choices[0].message.content
            
            results[model] = {
                "response": content,
                "duration": duration,
                "tokens": response.usage.total_tokens,
                "success": True
            }
            
            print(f"   ✅ Response ({duration:.2f}s): {content[:100]}...")
            
        except Exception as e:
            results[model] = {
                "error": str(e),
                "success": False
            }
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n📈 Results Summary:")
    for model, result in results.items():
        if result["success"]:
            print(f"   {model}:")
            print(f"     Time: {result['duration']:.2f}s")
            print(f"     Tokens: {result['tokens']}")
        else:
            print(f"   {model}: Failed")
    
    return results

# =============================================================================
# Example 7: Async Operations
# =============================================================================

async def async_example():
    """Asynchronous operations with Gemini"""
    print("\n⚡ Async Operations")
    print("=" * 60)
    
    client = create_gemini_async_client()
    
    # Multiple concurrent requests
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What are neural networks?"
    ]
    
    print("📝 Sending concurrent async requests...")
    start_time = time.time()
    
    async def get_response(prompt):
        response = await client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return prompt, response.choices[0].message.content
    
    # Run concurrently
    tasks = [get_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    
    print(f"✅ All responses received in {duration:.2f}s")
    
    for prompt, response in results:
        print(f"\n   Q: {prompt}")
        print(f"   A: {response[:100]}...")
    
    return results

# =============================================================================
# Example 8: Error Handling
# =============================================================================

def error_handling_example():
    """Demonstrate error handling"""
    print("\n⚠️ Error Handling")
    print("=" * 60)
    
    client = create_gemini_client()
    
    # Test various error scenarios
    test_cases = [
        {
            "name": "Invalid model",
            "model": "invalid-model-name",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        {
            "name": "Empty messages",
            "model": "gemini-1.5-flash",
            "messages": []
        },
        {
            "name": "Too many tokens",
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1000000
        }
    ]
    
    for test in test_cases:
        print(f"\n🧪 Testing: {test['name']}")
        
        try:
            response = client.chat.completions.create(**test)
            print(f"   ✅ Unexpected success: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            print(f"   ⚠️  Expected error: {type(e).__name__}")
            print(f"   Message: {str(e)[:100]}...")
    
    print("\n✅ Error handling demonstrated")
    return True

# =============================================================================
# Example 9: Migration Guide
# =============================================================================

def migration_guide():
    """Show how to migrate from OpenAI to Gemini"""
    print("\n🔄 Migration from OpenAI to Gemini")
    print("=" * 60)
    
    print("\n📝 Original OpenAI code:")
    print("""
    from openai import OpenAI
    
    client = OpenAI(api_key="sk-...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
    """)
    
    print("\n✨ Migrated to Gemini (minimal changes):")
    print("""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    response = client.chat.completions.create(
        model="gemini-1.5-flash",  # Change model name
        messages=[{"role": "user", "content": "Hello"}]
    )
    """)
    
    print("\n🎯 Key differences:")
    print("   1. Change base_url to Gemini endpoint")
    print("   2. Use GEMINI_API_KEY instead of OpenAI key")
    print("   3. Change model names (gpt-3.5-turbo → gemini-1.5-flash)")
    print("   4. Everything else remains the same!")
    
    print("\n📊 Model mapping suggestions:")
    print("   • gpt-3.5-turbo → gemini-1.5-flash")
    print("   • gpt-4 → gemini-1.5-pro")
    print("   • gpt-4-turbo → gemini-1.5-pro")
    print("   • text-embedding-ada-002 → text-embedding-004")
    
    return True

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run examples"""
    parser = argparse.ArgumentParser(description="Gemini OpenAI Compatibility Examples")
    parser.add_argument("--test-all", action="store_true", help="Run all examples")
    parser.add_argument("--compare", action="store_true", help="Compare models")
    parser.add_argument("--async-mode", action="store_true", help="Run async example")
    
    args = parser.parse_args()
    
    print("🚀 Gemini OpenAI Compatibility Examples")
    print("=" * 60)
    print(f"API Key: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    print(f"Endpoint: {GEMINI_BASE_URL}")
    print(f"Available models: {', '.join(GEMINI_MODELS)}")
    
    try:
        if args.compare:
            model_comparison_example()
        elif args.async_mode:
            asyncio.run(async_example())
        elif args.test_all:
            # Run all examples
            examples = [
                ("Basic Chat", basic_chat_example),
                ("Streaming", streaming_example),
                ("Function Calling", function_calling_example),
                ("Conversation", conversation_example),
                ("JSON Mode", json_mode_example),
                ("Model Comparison", model_comparison_example),
                ("Error Handling", error_handling_example),
                ("Migration Guide", migration_guide),
            ]
            
            results = {}
            for name, func in examples:
                try:
                    print(f"\n{'='*60}")
                    result = func()
                    results[name] = {"success": True}
                    print(f"✅ {name} completed")
                except Exception as e:
                    results[name] = {"success": False, "error": str(e)}
                    print(f"❌ {name} failed: {e}")
            
            # Add async example
            try:
                print(f"\n{'='*60}")
                asyncio.run(async_example())
                results["Async Operations"] = {"success": True}
                print("✅ Async Operations completed")
            except Exception as e:
                results["Async Operations"] = {"success": False, "error": str(e)}
                print(f"❌ Async Operations failed: {e}")
            
            # Summary
            print(f"\n{'='*60}")
            print("📊 SUMMARY")
            print("="*60)
            successful = sum(1 for r in results.values() if r["success"])
            print(f"✅ Successful: {successful}/{len(results)}")
            
            for name, result in results.items():
                status = "✅" if result["success"] else "❌"
                print(f"   {status} {name}")
            
        else:
            # Run basic examples
            basic_chat_example()
            streaming_example()
            function_calling_example()
            
            print(f"\n{'='*60}")
            print("✅ Basic examples completed successfully!")
            print("\n💡 Tips:")
            print("   • Use --test-all to run all examples")
            print("   • Use --compare to compare different models")
            print("   • Use --async to test async operations")
            print("\n📚 Documentation:")
            print("   https://ai.google.dev/gemini-api/docs/openai")
            
    except KeyboardInterrupt:
        print("\n👋 Examples cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()