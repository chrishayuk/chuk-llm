#!/usr/bin/env python3
"""
DeepSeek Diagnostic Test
========================

Isolate and diagnose the empty response issue with DeepSeek.
"""

import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

from chuk_llm.llm.client import get_client

async def test_simple_prompts():
    """Test with very simple prompts to isolate the issue"""
    print("🔍 DeepSeek Diagnostic Test")
    print("=" * 50)
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ DEEPSEEK_API_KEY not set")
        return
    
    # Test both models
    models = ["deepseek-chat", "deepseek-reasoner"]
    
    # Simple prompts that should always work
    test_prompts = [
        "Hello",
        "What is 2+2?",
        "Name one color.",
        "Say 'yes' or 'no'.",
        "What is Python?",
        "Explain AI in one sentence.",
    ]
    
    for model in models:
        print(f"\n🤖 Testing {model}")
        print("-" * 30)
        
        client = get_client("deepseek", model=model)
        
        for i, prompt in enumerate(test_prompts, 1):
            try:
                print(f"{i}. Prompt: '{prompt}'")
                
                # Simple message format
                messages = [{"role": "user", "content": prompt}]
                
                start_time = time.time()
                response = await client.create_completion(messages, max_tokens=50)
                duration = time.time() - start_time
                
                response_text = response.get("response", "").strip()
                
                if response_text:
                    print(f"   ✅ ({duration:.2f}s): {response_text[:100]}...")
                else:
                    print(f"   ❌ ({duration:.2f}s): EMPTY RESPONSE")
                    print(f"   📊 Full response: {response}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"   💥 Error: {e}")
        
        print()

async def test_different_parameters():
    """Test with different parameters to see what affects responses"""
    print("🎛️  Parameter Testing")
    print("=" * 50)
    
    client = get_client("deepseek", model="deepseek-reasoner")
    prompt = "What is the capital of France?"
    
    # Test different parameter combinations
    test_configs = [
        {"max_tokens": 10},
        {"max_tokens": 50},
        {"max_tokens": 100},
        {"temperature": 0.0, "max_tokens": 50},
        {"temperature": 0.5, "max_tokens": 50},
        {"temperature": 1.0, "max_tokens": 50},
        {"top_p": 0.9, "max_tokens": 50},
        {"top_p": 0.5, "max_tokens": 50},
    ]
    
    for config in test_configs:
        try:
            print(f"Config: {config}")
            
            messages = [{"role": "user", "content": prompt}]
            
            start_time = time.time()
            response = await client.create_completion(messages, **config)
            duration = time.time() - start_time
            
            response_text = response.get("response", "").strip()
            
            if response_text:
                print(f"   ✅ ({duration:.2f}s): {response_text}")
            else:
                print(f"   ❌ ({duration:.2f}s): EMPTY")
            
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"   💥 Error: {e}")
    
    print()

async def test_context_length():
    """Test if context length affects responses"""
    print("📏 Context Length Testing")
    print("=" * 50)
    
    client = get_client("deepseek", model="deepseek-reasoner")
    
    # Test with increasing context lengths
    base_message = {"role": "user", "content": "What is AI?"}
    
    context_lengths = [1, 2, 5, 10]
    
    for length in context_lengths:
        try:
            print(f"Context length: {length} messages")
            
            # Build context
            messages = []
            for i in range(length - 1):
                messages.extend([
                    {"role": "user", "content": f"Hello {i+1}"},
                    {"role": "assistant", "content": f"Hi there {i+1}!"}
                ])
            messages.append(base_message)
            
            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=50)
            duration = time.time() - start_time
            
            response_text = response.get("response", "").strip()
            
            if response_text:
                print(f"   ✅ ({duration:.2f}s): {response_text[:50]}...")
            else:
                print(f"   ❌ ({duration:.2f}s): EMPTY")
            
            await asyncio.sleep(1)  # Longer delay for context tests
            
        except Exception as e:
            print(f"   💥 Error: {e}")
    
    print()

async def test_system_messages():
    """Test if system messages affect responses"""
    print("🎭 System Message Testing")
    print("=" * 50)
    
    client = get_client("deepseek", model="deepseek-reasoner")
    prompt = "What is machine learning?"
    
    # Test with different system messages
    system_messages = [
        None,
        "You are a helpful assistant.",
        "You are a technical expert.",
        "Answer concisely.",
        "Think step by step.",
        "",  # Empty system message
    ]
    
    for i, system_msg in enumerate(system_messages):
        try:
            print(f"Test {i+1}: System = {repr(system_msg)}")
            
            messages = []
            if system_msg is not None:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": prompt})
            
            start_time = time.time()
            response = await client.create_completion(messages, max_tokens=50)
            duration = time.time() - start_time
            
            response_text = response.get("response", "").strip()
            
            if response_text:
                print(f"   ✅ ({duration:.2f}s): {response_text[:50]}...")
            else:
                print(f"   ❌ ({duration:.2f}s): EMPTY")
            
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"   💥 Error: {e}")
    
    print()

async def main():
    """Run all diagnostic tests"""
    print("🚀 DeepSeek Diagnostic Suite")
    print("=" * 60)
    
    await test_simple_prompts()
    await test_different_parameters()
    await test_context_length()
    await test_system_messages()
    
    print("🏁 Diagnostic complete!")
    print("\n💡 Look for patterns in the results:")
    print("   • Which prompts consistently work/fail?")
    print("   • Which parameters help/hurt?")
    print("   • Does context length matter?")
    print("   • Do system messages affect responses?")

if __name__ == "__main__":
    asyncio.run(main())