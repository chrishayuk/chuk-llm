#!/usr/bin/env python3
"""
Groq Client Demo Script
=======================

Demonstrates the capabilities of the Groq client with various models.

Prerequisites:
    pip install chuk-llm
    export GROQ_API_KEY="your-groq-api-key"

Usage:
    python groq_demo.py
"""

import asyncio
import os
import json
from typing import List, Dict, Any
from chuk_llm.llm.providers.groq_client import GroqAILLMClient


# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

# Check for API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  Please set GROQ_API_KEY environment variable")
    print("   Get your API key from: https://console.groq.com/keys")
    exit(1)

# Available models on Groq (as of January 2025)
PRODUCTION_MODELS = {
    "llama-3.3-70b-versatile": "Most capable Llama model",
    "llama-3.1-8b-instant": "Fast, efficient Llama model",
}

PREVIEW_MODELS = {
    "openai/gpt-oss-120b": "Open source reasoning model (120B)",
    "openai/gpt-oss-20b": "Smaller GPT-OSS model",
    "deepseek-r1-distill-llama-70b": "DeepSeek reasoning model",
    "qwen/qwen3-32b": "Qwen model with 40k output tokens",
    "moonshotai/kimi-k2-instruct": "Kimi model",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_subheader(title: str):
    """Print a formatted subheader"""
    print(f"\n📌 {title}")
    print("-" * 50)

def print_model_info(info: Dict[str, Any]):
    """Pretty print model information"""
    print(f"Provider: {info.get('provider')}")
    print(f"Model: {info.get('model')}")
    print(f"Status: {info.get('model_status', 'unknown')}")
    print(f"Family: {info.get('model_family', 'unknown')}")
    print(f"Context Length: {info.get('max_context_length', 'N/A'):,} tokens")
    print(f"Max Output: {info.get('max_output_tokens', 'N/A'):,} tokens")
    
    if 'features' in info:
        print(f"Features: {', '.join(info['features'])}")
    
    if 'groq_specific' in info:
        groq_info = info['groq_specific']
        print(f"Optimization: {groq_info.get('optimized_for', 'balanced')}")
        print(f"Context: {groq_info.get('huge_context', 'standard')}")

# =============================================================================
# DEMO 1: Basic Text Generation
# =============================================================================

async def demo_basic_generation():
    """Demonstrate basic text generation with Llama"""
    print_header("Demo 1: Basic Text Generation")
    
    client = GroqAILLMClient(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    
    # Get model info
    try:
        info = client.get_model_info()
        print("\n🤖 Using Model:")
        print_model_info(info)
    except Exception as e:
        # Silently handle config errors
        print(f"\n🤖 Using Model: {client.model}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Explain quantum computing in 2 sentences."}
    ]
    
    print("\n💬 User: Explain quantum computing in 2 sentences.")
    print("\n🤖 Assistant: ", end="")
    
    result = await client.create_completion(
        messages,
        temperature=0.7,
        max_tokens=100,
        stream=False
    )
    
    print(result["response"])

# =============================================================================
# DEMO 2: Streaming Response
# =============================================================================

async def demo_streaming():
    """Demonstrate streaming responses for real-time output"""
    print_header("Demo 2: Streaming Response")
    
    client = GroqAILLMClient(
        model="llama-3.1-8b-instant",  # Fast model for streaming
        api_key=GROQ_API_KEY
    )
    
    print(f"\n🚀 Using fast model: {client.model}")
    print("   Ultra-fast inference with Groq!")
    
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence."}
    ]
    
    print("\n💬 User: Write a haiku about artificial intelligence.")
    print("\n🤖 Assistant: ")
    
    # Stream the response
    stream = client.create_completion(
        messages,
        temperature=0.9,
        max_tokens=50,
        stream=True
    )
    
    full_response = ""
    async for chunk in stream:
        if chunk["response"]:
            print(chunk["response"], end="", flush=True)
            full_response += chunk["response"]
    
    print("\n")

# =============================================================================
# DEMO 3: Function Calling / Tools
# =============================================================================

async def demo_function_calling():
    """Demonstrate function calling with tools"""
    print_header("Demo 3: Function Calling with Tools")
    
    client = GroqAILLMClient(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
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
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather like in Tokyo and New York?"}
    ]
    
    print("\n🛠️  Available tools: get_weather, search_web")
    print("\n💬 User: What's the weather like in Tokyo and New York?")
    
    result = await client.create_completion(
        messages,
        tools=tools,
        temperature=0.3,
        stream=False
    )
    
    if result["tool_calls"]:
        print("\n🔧 Tool Calls:")
        for i, tool_call in enumerate(result["tool_calls"], 1):
            func = tool_call["function"]
            print(f"   {i}. {func['name']}({func['arguments']})")
    
    if result["response"]:
        print(f"\n🤖 Response: {result['response']}")

# =============================================================================
# DEMO 4: GPT-OSS Reasoning Model
# =============================================================================

async def demo_gpt_oss_reasoning():
    """Demonstrate GPT-OSS open source reasoning model"""
    print_header("Demo 4: GPT-OSS Reasoning Model")
    
    # Check if GPT-OSS is available
    model = "openai/gpt-oss-120b"
    
    print(f"\n🧠 Using reasoning model: {model}")
    print("   Open source model with reasoning capabilities!")
    
    client = GroqAILLMClient(
        model=model,
        api_key=GROQ_API_KEY
    )
    
    # Get model info
    try:
        info = client.get_model_info()
        print("\n📊 Model Info:")
        print_model_info(info)
    except Exception:
        # Silently handle config errors
        print(f"\n📊 Model: {model} (Preview)")
    
    messages = [
        {"role": "user", "content": """
        I have 3 apples. I eat 1 apple and give 1 apple to my friend.
        Then I buy 5 more apples. How many apples do I have now?
        Please show your reasoning step by step.
        """}
    ]
    
    print("\n💬 User: [Math reasoning problem about apples]")
    print("\n🤖 Assistant (with reasoning):\n")
    
    try:
        result = await client.create_completion(
            messages,
            temperature=0.1,  # Low temperature for reasoning
            max_tokens=200,
            stream=False
        )
        
        print(result["response"])
    except Exception as e:
        print(f"⚠️  Note: {model} may be in preview. Error: {e}")
        print("   Using fallback to standard model...")
        
        # Fallback to standard model
        client.model = "llama-3.3-70b-versatile"
        result = await client.create_completion(messages, max_tokens=200)
        print(result["response"])

# =============================================================================
# DEMO 5: Large Context Window
# =============================================================================

async def demo_large_context():
    """Demonstrate Groq's huge context windows (131k tokens)"""
    print_header("Demo 5: Large Context Window (131k tokens)")
    
    client = GroqAILLMClient(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    
    print(f"\n📚 Model: {client.model}")
    print(f"   Context window: 131,072 tokens!")
    print(f"   Max output: 32,768 tokens!")
    
    # Simulate a long document
    long_context = """
    Chapter 1: The Beginning
    [Imagine this is a very long document with thousands of words...]
    
    Chapter 2: The Middle
    [More content here...]
    
    Chapter 3: The End
    [Final content...]
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful summarization assistant."},
        {"role": "user", "content": f"Summarize this document:\n\n{long_context}"}
    ]
    
    print("\n💬 User: [Sending long document for summarization...]")
    print("\n🤖 Assistant: ")
    
    result = await client.create_completion(
        messages,
        temperature=0.3,
        max_tokens=150,
        stream=False
    )
    
    print(result["response"])

# =============================================================================
# DEMO 6: Model Comparison
# =============================================================================

async def demo_model_comparison():
    """Compare different models on the same task"""
    print_header("Demo 6: Model Comparison")
    
    models_to_test = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    
    # Same prompt for all models
    messages = [
        {"role": "user", "content": "Write a creative title for a sci-fi movie about AI."}
    ]
    
    print("\n🎬 Task: Write a creative title for a sci-fi movie about AI\n")
    
    for model_name in models_to_test:
        client = GroqAILLMClient(
            model=model_name,
            api_key=GROQ_API_KEY
        )
        
        print(f"🤖 {model_name}:")
        
        try:
            import time
            start_time = time.time()
            
            result = await client.create_completion(
                messages,
                temperature=0.9,
                max_tokens=20,
                stream=False
            )
            
            elapsed = time.time() - start_time
            print(f"   📝 \"{result['response']}\"")
            print(f"   ⏱️  Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
        
        print()

# =============================================================================
# DEMO 7: Error Handling
# =============================================================================

async def demo_error_handling():
    """Demonstrate error handling and fallbacks"""
    print_header("Demo 7: Error Handling")
    
    client = GroqAILLMClient(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    
    print("\n🛡️  Testing error handling with complex tools...")
    
    # Create a tool that might cause issues
    complex_tools = [
        {
            "type": "function",
            "function": {
                "name": "complex_tool_name",  # Simplified for demo
                "description": "A tool with complex functionality",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Input data"},
                        "mode": {
                            "type": "string", 
                            "enum": ["fast", "accurate"],
                            "description": "Processing mode"
                        }
                    },
                    "required": ["data"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Use the complex tool to analyze this data: 'test input'."}
    ]
    
    try:
        result = await client.create_completion(
            messages,
            tools=complex_tools,
            stream=False
        )
        
        if result.get("tool_calls"):
            print("✅ Tool handling successful!")
            print(f"   Tool calls made: {len(result['tool_calls'])}")
            for tc in result["tool_calls"]:
                func = tc["function"]
                print(f"   - {func['name']}({func['arguments']})")
        
        if result.get("response"):
            print(f"   Response: {result['response'][:200]}...")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("   Attempting fallback without tools...")
        
        # Try without tools
        try:
            result = await client.create_completion(messages, stream=False)
            print(f"✅ Fallback successful!")
            print(f"   Response: {result['response'][:200]}...")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all demos"""
    print("\n" + "🚀" * 20)
    print("  GROQ CLIENT DEMO - Ultra-Fast Inference")
    print("  Showcasing Groq's capabilities with various models")
    print("🚀" * 20)
    
    demos = [
        ("Basic Generation", demo_basic_generation),
        ("Streaming", demo_streaming),
        ("Function Calling", demo_function_calling),
        ("GPT-OSS Reasoning", demo_gpt_oss_reasoning),
        ("Large Context", demo_large_context),
        ("Model Comparison", demo_model_comparison),
        ("Error Handling", demo_error_handling),
    ]
    
    print("\n📋 Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"   {i}. {name}")
    
    print("\n" + "-" * 70)
    user_input = input("Enter demo number (1-7) or 'all' to run all demos: ").strip()
    
    if user_input.lower() == 'all':
        for name, demo_func in demos:
            try:
                await demo_func()
                await asyncio.sleep(1)  # Brief pause between demos
            except Exception as e:
                print(f"\n⚠️  Error in {name}: {e}")
                continue
    else:
        try:
            demo_idx = int(user_input) - 1
            if 0 <= demo_idx < len(demos):
                await demos[demo_idx][1]()
            else:
                print("❌ Invalid demo number")
        except ValueError:
            print("❌ Please enter a valid number or 'all'")
    
    print("\n" + "=" * 70)
    print("  ✅ Demo Complete!")
    print("  Learn more at: https://console.groq.com")
    print("=" * 70)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()