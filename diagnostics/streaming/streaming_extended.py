#!/usr/bin/env python3
# diagnostics/streaming/streaming_extended.py
"""
Extended Streaming Test - Tests with longer content to verify real streaming
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


async def test_extended_streaming():
    """Test with a longer response to verify true streaming."""
    print("=== Extended Streaming Test ===")
    print("Testing with longer content to verify real-time streaming\n")

    client = get_client(provider="openai", model="gpt-4o-mini")

    # Ask for a longer response that will take time to generate
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": """Write a short story about a robot learning to paint.
        Include at least 5 paragraphs. Make each paragraph at least 3 sentences long.
        Number each paragraph.""",
        },
    ]

    print("🔍 Testing streaming with longer content...")
    start_time = time.time()

    # Get the stream directly (no await)
    response = client.create_completion(messages, stream=True)

    if not hasattr(response, "__aiter__"):
        print(f"❌ ERROR: Expected async generator, got {type(response)}")
        return

    print("✅ Got async generator\n")

    chunk_count = 0
    first_chunk_time = None
    last_chunk_time = start_time
    chunk_delays = []
    total_content = ""

    print("Streaming response:\n" + "-" * 60)

    async for chunk in response:
        current_time = time.time()
        relative_time = current_time - start_time

        if first_chunk_time is None:
            first_chunk_time = relative_time
            print(f"\n[FIRST CHUNK at {relative_time:.3f}s]\n")

        # Calculate delay between chunks
        chunk_delay = current_time - last_chunk_time
        if chunk_count > 0:  # Skip first chunk
            chunk_delays.append(chunk_delay)

        last_chunk_time = current_time
        chunk_count += 1

        if isinstance(chunk, dict) and "response" in chunk:
            chunk_text = chunk["response"] or ""
            total_content += chunk_text
            print(chunk_text, end="", flush=True)

    print("\n" + "-" * 60)

    end_time = time.time() - start_time

    # Calculate statistics
    avg_delay = sum(chunk_delays) / len(chunk_delays) if chunk_delays else 0
    max_delay = max(chunk_delays) if chunk_delays else 0
    min_delay = min(chunk_delays) if chunk_delays else 0

    print("\n📊 EXTENDED STREAMING ANALYSIS:")
    print(f"   Total chunks: {chunk_count}")
    print(f"   Total characters: {len(total_content)}")
    print(f"   First chunk delay: {first_chunk_time:.3f}s")
    print(f"   Total time: {end_time:.3f}s")
    print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
    print("\n   Chunk arrival stats:")
    print(f"   - Average delay between chunks: {avg_delay * 1000:.1f}ms")
    print(f"   - Max delay: {max_delay * 1000:.1f}ms")
    print(f"   - Min delay: {min_delay * 1000:.1f}ms")

    # Evaluate streaming quality
    print("\n   📈 Streaming Quality Assessment:")

    if first_chunk_time < 1.5:
        print("   ✅ EXCELLENT: Very fast first chunk (<1.5s)")
    elif first_chunk_time < 3.0:
        print("   ✅ GOOD: Fast first chunk (<3s)")
    else:
        print("   ⚠️  SLOW: First chunk delay >3s")

    # Check if chunks arrived gradually (true streaming) or all at once (buffered)
    streaming_ratio = (end_time - first_chunk_time) / end_time if end_time > 0 else 0

    if streaming_ratio > 0.5 and chunk_count > 10:
        print("   ✅ TRUE STREAMING: Chunks arrived gradually")
    elif streaming_ratio > 0.2:
        print("   ⚠️  PARTIAL STREAMING: Some buffering detected")
    else:
        print("   ❌ BUFFERED: Most chunks arrived immediately")

    if avg_delay > 0.01:  # More than 10ms average
        print("   ✅ NATURAL PACING: Chunks have realistic delays")
    else:
        print("   ⚠️  RAPID DELIVERY: Chunks arriving too quickly")


async def test_parallel_streaming():
    """Test multiple streaming requests in parallel to check for blocking."""
    print("\n\n=== Parallel Streaming Test ===")
    print("Testing 3 simultaneous streaming requests...\n")

    client = get_client(provider="openai", model="gpt-4o-mini")

    async def stream_request(request_id: int):
        messages = [
            {
                "role": "user",
                "content": f"Count from {request_id}0 to {request_id}9 slowly",
            }
        ]

        start = time.time()
        # Get stream directly (no await)
        response = client.create_completion(messages, stream=True)

        first_chunk_time = None
        chunks = []

        async for chunk in response:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start

            if isinstance(chunk, dict) and chunk.get("response"):
                chunks.append(chunk["response"])

        total_time = time.time() - start
        content = "".join(chunks).strip()

        print(
            f"Request {request_id}: First chunk at {first_chunk_time:.3f}s, "
            f"Total time: {total_time:.3f}s, Content: {content[:50]}..."
        )

        return first_chunk_time, total_time

    # Run 3 requests in parallel
    start_parallel = time.time()
    results = await asyncio.gather(
        stream_request(1), stream_request(2), stream_request(3)
    )
    total_parallel_time = time.time() - start_parallel

    print("\n📊 PARALLEL STREAMING RESULTS:")
    print(f"   Total parallel execution time: {total_parallel_time:.3f}s")

    # Check if requests were truly parallel
    max_individual_time = max(r[1] for r in results)
    if total_parallel_time < max_individual_time * 1.5:
        print("   ✅ TRUE PARALLEL: Requests processed simultaneously")
    else:
        print("   ❌ SEQUENTIAL: Requests appear to be blocking each other")


async def test_error_handling():
    """Test streaming behavior with errors."""
    print("\n\n=== Error Handling Test ===")

    # Test with invalid model
    try:
        client = get_client(provider="openai", model="invalid-model-xxx")

        messages = [{"role": "user", "content": "Hello"}]
        # Get stream directly (no await)
        response = client.create_completion(messages, stream=True)

        chunk_found = False
        async for chunk in response:
            chunk_found = True
            if chunk.get("error"):
                print(f"✅ Error properly streamed: {chunk.get('response')}")
                break
            else:
                print(f"Chunk: {chunk}")

        if not chunk_found:
            print("⚠️  No chunks received from invalid model")

    except Exception as e:
        print(f"✅ Exception properly raised: {e}")


async def test_multiple_providers():
    """Test streaming across multiple providers"""
    print("\n\n=== Multi-Provider Streaming Test ===")

    providers_to_test = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("groq", "llama-3.3-70b-versatile"),
    ]

    for provider, model in providers_to_test:
        print(f"\n🔍 Testing {provider} with {model}...")

        try:
            client = get_client(provider=provider, model=model)
            messages = [
                {"role": "user", "content": "Write a haiku about streaming data."}
            ]

            start_time = time.time()
            response = client.create_completion(messages, stream=True)

            chunk_count = 0
            content_chunks = []
            first_chunk_time = None

            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time

                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    content_chunks.append(chunk["response"])

                # Stop after reasonable number of chunks
                if chunk_count >= 10:
                    break

            total_time = time.time() - start_time
            full_content = "".join(content_chunks)

            print(
                f"   ✅ {provider}: {chunk_count} chunks, "
                f"first at {first_chunk_time:.3f}s, "
                f"total {total_time:.3f}s"
            )
            print(f"   Content: {full_content[:100]}...")

        except Exception as e:
            print(f"   ❌ {provider}: Error - {e}")


async def test_streaming_with_tools():
    """Test streaming with function calling"""
    print("\n\n=== Streaming + Tools Test ===")

    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    }

    try:
        client = get_client(provider="openai", model="gpt-4o-mini")
        messages = [
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Use the get_weather function.",
            }
        ]

        start_time = time.time()
        response = client.create_completion(messages, tools=[weather_tool], stream=True)

        chunk_count = 0
        tool_calls_found = []
        content_chunks = []

        async for chunk in response:
            chunk_count += 1

            if isinstance(chunk, dict):
                if chunk.get("response"):
                    content_chunks.append(chunk["response"])
                if chunk.get("tool_calls"):
                    tool_calls_found.extend(chunk["tool_calls"])

        total_time = time.time() - start_time

        print(f"   ✅ Streaming + Tools: {chunk_count} chunks in {total_time:.3f}s")
        print(f"   Content chunks: {len(content_chunks)}")
        print(f"   Tool calls found: {len(tool_calls_found)}")

        if tool_calls_found:
            for tc in tool_calls_found:
                print(f"   Tool: {tc.get('function', {}).get('name', 'unknown')}")

    except Exception as e:
        print(f"   ❌ Streaming + Tools failed: {e}")


if __name__ == "__main__":
    print("🚀 Extended LLM Streaming Tests\n")

    # Run all tests
    asyncio.run(test_extended_streaming())
    asyncio.run(test_parallel_streaming())
    asyncio.run(test_error_handling())
    asyncio.run(test_multiple_providers())
    asyncio.run(test_streaming_with_tools())
