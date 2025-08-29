# diagnostics/streaming/providers/anthropic_streaming.py
"""
Test Anthropic provider streaming behavior.
Updated to work with the new chuk-llm architecture.
"""

import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_llm.llm.client import get_client


async def test_anthropic_streaming():
    """Test Anthropic streaming behavior."""
    print("=== Testing Anthropic Provider Streaming ===")

    try:
        # Get Anthropic client with updated model name
        client = get_client(
            provider="anthropic",
            model="claude-sonnet-4-20250514",  # Updated model name
        )

        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")

        messages = [
            {
                "role": "user",
                "content": "Write a short story about a robot learning to paint. Make it at least 100 words and tell it slowly.",
            }
        ]

        print("\n🔍 Testing Anthropic streaming=True...")
        start_time = time.time()

        # Test streaming - DON'T await it since it returns async generator
        response = client.create_completion(messages, stream=True)

        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")

        if hasattr(response, "__aiter__"):
            print("✅ Got async generator")

            chunk_count = 0
            first_chunk_time = None
            last_chunk_time = start_time
            full_response = ""
            chunk_intervals = []

            print("Response: ", end="", flush=True)

            async for chunk in response:
                current_time = time.time()
                relative_time = current_time - start_time

                if first_chunk_time is None:
                    first_chunk_time = relative_time
                    print(f"\n🎯 FIRST CHUNK at: {relative_time:.3f}s")
                    print("Response: ", end="", flush=True)
                else:
                    # Track intervals between chunks
                    interval = current_time - last_chunk_time
                    chunk_intervals.append(interval)

                chunk_count += 1

                if isinstance(chunk, dict) and "response" in chunk:
                    chunk_text = chunk["response"] or ""
                    print(chunk_text, end="", flush=True)
                    full_response += chunk_text

                # Show timing for first few chunks and every 5th chunk
                if chunk_count <= 5 or chunk_count % 5 == 0:
                    interval = current_time - last_chunk_time
                    print(
                        f"\n   Chunk {chunk_count}: {relative_time:.3f}s (interval: {interval:.4f}s)"
                    )
                    print("   Continuing: ", end="", flush=True)

                last_chunk_time = current_time

            end_time = time.time() - start_time

            # Calculate interval statistics
            if chunk_intervals:
                avg_interval = sum(chunk_intervals) / len(chunk_intervals)
                min_interval = min(chunk_intervals)
                max_interval = max(chunk_intervals)
            else:
                avg_interval = min_interval = max_interval = 0

            print("\n\n📊 ANTHROPIC STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s")
            print(f"   Total time: {end_time:.3f}s")
            print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
            print(f"   Response length: {len(full_response)} characters")
            print(f"   Avg chunk interval: {avg_interval * 1000:.1f}ms")
            print(f"   Min interval: {min_interval * 1000:.1f}ms")
            print(f"   Max interval: {max_interval * 1000:.1f}ms")

            # Quality assessment
            if chunk_count == 1:
                print("   ⚠️  FAKE STREAMING: Only one chunk (entire response at once)")
            elif chunk_count < 5:
                print("   ⚠️  LIMITED STREAMING: Very few chunks")
            else:
                print("   ✅ REAL STREAMING: Multiple chunks detected")

            if first_chunk_time < 1.5:
                print("   ✅ FAST: Excellent first chunk time")
            elif first_chunk_time < 3.0:
                print("   ✅ GOOD: Acceptable first chunk time")
            else:
                print("   ⚠️  SLOW: First chunk could be faster")

            streaming_duration = end_time - first_chunk_time if first_chunk_time else 0
            if streaming_duration > 2.0:
                print("   ✅ TRUE STREAMING: Long streaming duration")
            elif streaming_duration > 0.5:
                print("   ✅ PARTIAL STREAMING: Some streaming detected")
            else:
                print("   ⚠️  BUFFERED: Very short streaming duration")

            if avg_interval > 0.05:  # More than 50ms
                print("   ✅ NATURAL PACING: Realistic chunk intervals")
            elif avg_interval > 0.01:  # More than 10ms
                print("   ✅ MODERATE PACING: Reasonable chunk intervals")
            else:
                print("   ⚠️  RAPID DELIVERY: Chunks arriving very quickly")

        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")

        print("\n🔍 Testing Anthropic streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")

    except Exception as e:
        print(f"❌ Error testing Anthropic: {e}")
        import traceback

        traceback.print_exc()


async def test_anthropic_vs_openai():
    """Compare Anthropic vs OpenAI streaming behavior."""
    print("\n\n=== Anthropic vs OpenAI Streaming Comparison ===")

    # Same prompt for both
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence"}
    ]

    providers = [("anthropic", "claude-sonnet-4-20250514"), ("openai", "gpt-4o-mini")]

    results = {}

    for provider, model in providers:
        print(f"\n🔍 Testing {provider} with {model}...")

        try:
            client = get_client(provider=provider, model=model)

            start_time = time.time()
            response = client.create_completion(messages, stream=True)

            chunk_count = 0
            first_chunk_time = None
            content_length = 0

            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time

                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    content_length += len(chunk["response"])

                # Limit chunks for comparison
                if chunk_count >= 15:
                    break

            total_time = time.time() - start_time

            results[provider] = {
                "chunks": chunk_count,
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "content_length": content_length,
            }

            print(
                f"   {provider}: {chunk_count} chunks, first at {first_chunk_time:.3f}s, {content_length} chars"
            )

        except Exception as e:
            print(f"   {provider}: Error - {e}")
            results[provider] = None

    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    if all(results.values()):
        anthropic_result = results["anthropic"]
        openai_result = results["openai"]

        # Compare first chunk times
        anthro_first = anthropic_result["first_chunk"]
        openai_first = openai_result["first_chunk"]

        if anthro_first < openai_first:
            diff = openai_first - anthro_first
            print(f"   🚀 Anthropic faster first chunk by {diff * 1000:.0f}ms")
        else:
            diff = anthro_first - openai_first
            print(f"   🚀 OpenAI faster first chunk by {diff * 1000:.0f}ms")

        # Compare chunk counts
        if anthropic_result["chunks"] > openai_result["chunks"]:
            print(
                f"   📊 Anthropic more granular streaming ({anthropic_result['chunks']} vs {openai_result['chunks']} chunks)"
            )
        else:
            print(
                f"   📊 OpenAI more granular streaming ({openai_result['chunks']} vs {anthropic_result['chunks']} chunks)"
            )

    else:
        print("   ❌ Could not compare - one or both tests failed")


async def main():
    """Run all Anthropic streaming tests."""
    await test_anthropic_streaming()
    await test_anthropic_vs_openai()


if __name__ == "__main__":
    asyncio.run(main())
