# diagnostics/streaming/streaming_diagnostic.py
"""
Simple Streaming Diagnostic
Updated to work with the new chuk-llm architecture
"""

import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client


async def test_fixed_chuk_llm():
    """Test the fixed chuk_llm client directly."""
    print("=== Testing Fixed chuk_llm Client ===")

    # Get client directly (same as ChukAgent does)
    client = get_client(provider="openai", model="gpt-4o-mini")

    print(f"Client type: {type(client)}")
    print(f"Client class: {client.__class__.__name__}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 10, one number per line"},
    ]

    print("\n🔍 Testing streaming=True...")
    start_time = time.time()

    try:
        # Call with streaming - NO AWAIT (returns async generator directly)
        response = client.create_completion(messages, stream=True)

        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")

        # Check if it's an async generator (not awaited)
        if hasattr(response, "__aiter__"):
            print("✅ Got async generator directly (not awaited)")

            chunk_count = 0
            first_chunk_time = None
            last_time = start_time

            print("Response: ", end="", flush=True)

            async for chunk in response:
                current_time = time.time()
                relative_time = current_time - start_time

                if first_chunk_time is None:
                    first_chunk_time = relative_time
                    print(f"\n🎯 FIRST CHUNK at: {relative_time:.3f}s")
                    print("Response: ", end="", flush=True)

                chunk_count += 1

                if isinstance(chunk, dict) and "response" in chunk:
                    chunk_text = chunk["response"] or ""
                    print(chunk_text, end="", flush=True)

                # Show timing for first few chunks
                if chunk_count <= 3:
                    interval = relative_time - (
                        first_chunk_time if chunk_count == 1 else last_time
                    )
                    print(
                        f"\n   Chunk {chunk_count}: {relative_time:.3f}s (interval: {interval:.3f}s)"
                    )
                    print("   Continuing: ", end="", flush=True)

                last_time = relative_time

            end_time = time.time() - start_time
            print("\n\n📊 CHUK_LLM STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s")
            print(f"   Total time: {end_time:.3f}s")
            print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")

            if first_chunk_time and first_chunk_time < 3.0:
                print("   ✅ GOOD: Quick first chunk")
            else:
                print("   ⚠️  SLOW: First chunk took too long")

            if end_time - first_chunk_time > 0.5:
                print("   ✅ STREAMING: Real streaming detected")
            else:
                print("   ⚠️  BUFFERED: Chunks arrived too quickly")

        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n🔍 Testing streaming=False...")
    try:
        # Test non-streaming - AWAIT this one
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print("✅ Non-streaming works correctly")
        else:
            print(f"❌ Unexpected non-streaming response: {response}")

    except Exception as e:
        print(f"❌ Non-streaming error: {e}")


async def test_multiple_providers():
    """Test streaming across different providers"""
    print("\n\n=== Testing Multiple Providers ===")

    providers_to_test = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("groq", "llama-3.3-70b-versatile"),
    ]

    for provider, model in providers_to_test:
        print(f"\n🔍 Testing {provider} with {model}...")

        try:
            client = get_client(provider=provider, model=model)
            messages = [{"role": "user", "content": "Say hello and count to 3"}]

            start_time = time.time()
            # Get stream directly (no await)
            response = client.create_completion(messages, stream=True)

            chunk_count = 0
            first_chunk_time = None
            content_chunks = []

            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time

                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    content_chunks.append(chunk["response"])

                # Limit chunks for demo
                if chunk_count >= 8:
                    break

            total_time = time.time() - start_time
            full_content = "".join(content_chunks)

            print(f"   ✅ {provider}: {chunk_count} chunks")
            print(f"   First chunk: {first_chunk_time:.3f}s")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Content: {full_content[:80]}...")

            # Quality assessment
            if first_chunk_time < 1.0:
                print("   🚀 FAST: Excellent response time")
            elif first_chunk_time < 2.0:
                print("   ✅ GOOD: Good response time")
            else:
                print("   ⚠️  SLOW: Response time could be better")

        except Exception as e:
            print(f"   ❌ {provider}: Error - {e}")


async def test_streaming_behavior():
    """Test detailed streaming behavior"""
    print("\n\n=== Detailed Streaming Behavior Test ===")

    try:
        client = get_client(provider="openai", model="gpt-4o-mini")

        # Test different message types
        test_cases = [
            ("Short response", [{"role": "user", "content": "Say hello"}]),
            (
                "Medium response",
                [{"role": "user", "content": "Write a short poem about clouds"}],
            ),
            (
                "Long response",
                [
                    {
                        "role": "user",
                        "content": "Explain how streaming works in 3 paragraphs",
                    }
                ],
            ),
        ]

        for test_name, messages in test_cases:
            print(f"\n📝 {test_name}:")

            start_time = time.time()
            response = client.create_completion(messages, stream=True)

            chunk_count = 0
            total_chars = 0
            chunk_times = []

            async for chunk in response:
                current_time = time.time() - start_time
                chunk_times.append(current_time)
                chunk_count += 1

                if isinstance(chunk, dict) and chunk.get("response"):
                    total_chars += len(chunk["response"])

                # Limit for demo
                if chunk_count >= 15:
                    break

            if chunk_times:
                print(f"   Chunks: {chunk_count}, Characters: {total_chars}")
                print(f"   First chunk: {chunk_times[0]:.3f}s")
                print(f"   Last chunk: {chunk_times[-1]:.3f}s")
                print(f"   Duration: {chunk_times[-1] - chunk_times[0]:.3f}s")

                # Calculate chunk rate
                if len(chunk_times) > 1:
                    avg_interval = (chunk_times[-1] - chunk_times[0]) / (
                        len(chunk_times) - 1
                    )
                    print(f"   Avg interval: {avg_interval * 1000:.1f}ms")

    except Exception as e:
        print(f"❌ Behavior test error: {e}")


if __name__ == "__main__":
    print("🚀 chuk-llm Streaming Diagnostic\n")

    # Run all tests
    asyncio.run(test_fixed_chuk_llm())
    asyncio.run(test_multiple_providers())
    asyncio.run(test_streaming_behavior())
