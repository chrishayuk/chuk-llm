#!/usr/bin/env python3
"""
Streaming Performance Diagnostic Tool

Tests whether providers are truly streaming (progressive chunks)
or buffering (all chunks at end).

Usage:
    python diagnostics/streaming_performance.py              # Test all providers
    python diagnostics/streaming_performance.py azure_openai # Test specific provider
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

from chuk_llm.llm.client import get_client
from chuk_llm.core.models import Message, MessageRole


async def test_provider_streaming(provider: str, model: str):
    """Test streaming performance for a specific provider"""

    print("\n" + "=" * 80)
    print(f"Testing: {provider}/{model}")
    print("=" * 80)

    try:
        client = get_client(provider, model=model)
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return None

    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a paragraph about the solar system (50-100 words)."
        )
    ]

    # Test streaming
    print("\n🌊 Streaming test...")
    start = time.time()
    first_chunk_time = None
    chunk_times = []
    accumulated = ""

    try:
        async for chunk in client.create_completion(messages, max_tokens=300, stream=True):
            current_time = time.time() - start

            if first_chunk_time is None:
                first_chunk_time = current_time
                print(f"  ⚡ First chunk at {first_chunk_time:.3f}s")

            text = chunk.get('response', '')
            if text:
                accumulated += text

            chunk_times.append(current_time)

        total_time = time.time() - start

    except Exception as e:
        print(f"  ❌ Streaming failed: {e}")
        return None

    # Analysis
    print(f"\n📊 Results:")
    print(f"  Total time:      {total_time:.3f}s")
    print(f"  First chunk:     {first_chunk_time:.3f}s ({(first_chunk_time/total_time*100):.0f}% of total)")
    print(f"  Chunks received: {len(chunk_times)}")
    print(f"  Text length:     {len(accumulated)} chars")

    # Determine streaming quality
    if first_chunk_time is None:
        verdict = "❌ NO CHUNKS"
        quality = "failed"
    elif first_chunk_time / total_time < 0.2:
        verdict = "✅ EXCELLENT - True progressive streaming"
        quality = "excellent"
    elif first_chunk_time / total_time < 0.4:
        verdict = "✅ GOOD - Mostly progressive streaming"
        quality = "good"
    elif first_chunk_time / total_time < 0.8:
        verdict = "⚠️  DELAYED - Significant buffering before streaming"
        quality = "delayed"
    else:
        verdict = "❌ BUFFERED - Generates everything before streaming"
        quality = "buffered"

    print(f"\n{verdict}")

    if len(chunk_times) > 1:
        # Show chunk distribution
        quartile = total_time / 4
        q1 = sum(1 for t in chunk_times if t < quartile)
        q2 = sum(1 for t in chunk_times if quartile <= t < 2*quartile)
        q3 = sum(1 for t in chunk_times if 2*quartile <= t < 3*quartile)
        q4 = sum(1 for t in chunk_times if t >= 3*quartile)

        print(f"\nChunk distribution by time quartile:")
        print(f"  Q1: {q1:3d} chunks {' ▓' * min(40, q1)}")
        print(f"  Q2: {q2:3d} chunks {' ▓' * min(40, q2)}")
        print(f"  Q3: {q3:3d} chunks {' ▓' * min(40, q3)}")
        print(f"  Q4: {q4:3d} chunks {' ▓' * min(40, q4)}")

    return {
        'provider': provider,
        'model': model,
        'first_chunk_time': first_chunk_time,
        'total_time': total_time,
        'chunk_count': len(chunk_times),
        'quality': quality,
        'verdict': verdict
    }


async def main():
    """Test all configured providers or specific one"""

    # Determine which providers to test
    if len(sys.argv) > 1:
        provider_arg = sys.argv[1]
        model_arg = sys.argv[2] if len(sys.argv) > 2 else None

        if provider_arg == "azure_openai":
            model = model_arg or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            tests = [("azure_openai", model)]
        elif provider_arg == "openai":
            model = model_arg or "gpt-4o-mini"
            tests = [("openai", model)]
        elif provider_arg == "anthropic":
            model = model_arg or "claude-3-5-sonnet-latest"
            tests = [("anthropic", model)]
        else:
            print(f"Unknown provider: {provider_arg}")
            print("Supported: azure_openai, openai, anthropic")
            return
    else:
        # Test all configured providers
        tests = []

        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            azure_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            tests.append(("azure_openai", azure_model))

        if os.getenv("OPENAI_API_KEY"):
            tests.append(("openai", "gpt-4o-mini"))

        if os.getenv("ANTHROPIC_API_KEY"):
            tests.append(("anthropic", "claude-3-5-sonnet-latest"))

        if not tests:
            print("❌ No API keys found. Set one of:")
            print("   - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT")
            print("   - OPENAI_API_KEY")
            print("   - ANTHROPIC_API_KEY")
            return

    # Run tests
    results = []
    for provider, model in tests:
        result = await test_provider_streaming(provider, model)
        if result:
            results.append(result)
        await asyncio.sleep(1)  # Brief pause between tests

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for r in results:
            ratio = r['first_chunk_time'] / r['total_time'] if r['total_time'] > 0 else 1
            print(f"\n{r['provider']}/{r['model']}:")
            print(f"  First chunk: {r['first_chunk_time']:.3f}s / {r['total_time']:.3f}s ({ratio*100:.0f}%)")
            print(f"  Quality: {r['quality']}")


if __name__ == "__main__":
    asyncio.run(main())
