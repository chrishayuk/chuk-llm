#!/usr/bin/env python3
"""
Moonshot AI Diagnostic Test
============================

Test and diagnose Moonshot AI (Kimi) integration.
"""

import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv()

from chuk_llm.llm.client import get_client


async def test_simple_prompts():
    """Test with very simple prompts to verify basic functionality"""
    print("🔍 Moonshot AI Diagnostic Test")
    print("=" * 50)

    if not os.getenv("MOONSHOT_API_KEY"):
        print("❌ MOONSHOT_API_KEY not set")
        return

    # Test multiple Kimi models
    models = [
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
        "kimi-k2-thinking-turbo",
    ]

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

        try:
            client = get_client("moonshot", model=model)

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

        except Exception as e:
            print(f"   💥 Model initialization error: {e}")

        print()


async def test_different_parameters():
    """Test with different parameters to see what affects responses"""
    print("🎛️  Parameter Testing")
    print("=" * 50)

    client = get_client("moonshot", model="kimi-k2-turbo-preview")
    prompt = "What is the capital of France?"

    # Test different parameter combinations
    test_configs = [
        {"max_tokens": 10},
        {"max_tokens": 50},
        {"max_tokens": 100},
        {"temperature": 0.0, "max_tokens": 50},
        {"temperature": 0.6, "max_tokens": 50},
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
    """Test if context length affects responses (Kimi supports 256K context)"""
    print("📏 Context Length Testing")
    print("=" * 50)

    client = get_client("moonshot", model="kimi-k2-turbo-preview")

    # Test with increasing context lengths
    base_message = {"role": "user", "content": "What is AI?"}

    context_lengths = [1, 2, 5, 10]

    for length in context_lengths:
        try:
            print(f"Context length: {length} messages")

            # Build context
            messages = []
            for i in range(length - 1):
                messages.extend(
                    [
                        {"role": "user", "content": f"Hello {i + 1}"},
                        {"role": "assistant", "content": f"Hi there {i + 1}!"},
                    ]
                )
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

    client = get_client("moonshot", model="kimi-k2-turbo-preview")
    prompt = "What is machine learning?"

    # Test with different system messages
    system_messages = [
        None,
        "You are Kimi, an AI assistant provided by Moonshot AI.",
        "You are a technical expert.",
        "Answer concisely.",
        "Think step by step.",
        "",  # Empty system message
    ]

    for i, system_msg in enumerate(system_messages):
        try:
            print(f"Test {i + 1}: System = {repr(system_msg)}")

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


async def test_streaming():
    """Test streaming responses"""
    print("🌊 Streaming Testing")
    print("=" * 50)

    client = get_client("moonshot", model="kimi-k2-turbo-preview")
    prompt = "Count from 1 to 5."

    try:
        print(f"Prompt: '{prompt}'")
        messages = [{"role": "user", "content": prompt}]

        print("Response: ", end="", flush=True)

        start_time = time.time()
        chunk_count = 0

        async for chunk in client.stream_completion(messages, max_tokens=100):
            content = chunk.get("content", "")
            if content:
                print(content, end="", flush=True)
                chunk_count += 1

        duration = time.time() - start_time
        print(f"\n   ✅ ({duration:.2f}s): Received {chunk_count} chunks")

    except Exception as e:
        print(f"\n   💥 Error: {e}")

    print()


async def test_json_mode():
    """Test JSON mode if supported"""
    print("📋 JSON Mode Testing")
    print("=" * 50)

    client = get_client("moonshot", model="kimi-k2-turbo-preview")
    prompt = "Return a JSON object with your name and version."

    try:
        print(f"Prompt: '{prompt}'")
        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        response = await client.create_completion(
            messages,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        duration = time.time() - start_time

        response_text = response.get("response", "").strip()

        if response_text:
            print(f"   ✅ ({duration:.2f}s): {response_text}")
        else:
            print(f"   ❌ ({duration:.2f}s): EMPTY")

    except Exception as e:
        print(f"   💥 Error: {e}")

    print()


async def main():
    """Run all diagnostic tests"""
    print("🚀 Moonshot AI Diagnostic Suite")
    print("=" * 60)

    await test_simple_prompts()
    await test_different_parameters()
    await test_context_length()
    await test_system_messages()
    await test_streaming()
    await test_json_mode()

    print("🏁 Diagnostic complete!")
    print("\n💡 Moonshot AI (Kimi) Features:")
    print("   • Kimi K2: Industry-leading coding abilities")
    print("   • 256K context window (K2 models)")
    print("   • Built-in tools (web search, etc.)")
    print("   • Long-term thinking (K2-thinking models)")
    print("   • High-speed turbo variants (60-100 tokens/s)")


if __name__ == "__main__":
    asyncio.run(main())
