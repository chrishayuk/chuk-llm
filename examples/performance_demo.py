#!/usr/bin/env python3
# examples/performance_demo.py
"""
Clean Performance Demo - Sync vs Async Comparison
================================================

A cleaner version that avoids event loop issues.
"""

import time
import asyncio
import warnings
from chuk_llm import ask, ask_sync, stream

# Suppress event loop warnings
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")


def sync_benchmark():
    """Run synchronous benchmark."""
    questions = [
        "What is AI in one sentence?",
        "Capital of Japan?",
        "What is 2+2?",
    ]
    
    print("🐌 Synchronous Performance")
    print("-" * 40)
    start = time.time()
    
    for i, q in enumerate(questions, 1):
        response = ask_sync(q)
        print(f"  {i}. {q} → {len(response)} chars")
    
    sync_time = time.time() - start
    print(f"  Total: {sync_time:.2f}s\n")
    return sync_time


async def async_benchmark():
    """Run async benchmarks."""
    questions = [
        "What is AI in one sentence?",
        "Capital of Japan?",
        "What is 2+2?",
    ]
    
    # Sequential
    print("🐢 Async Sequential")
    print("-" * 40)
    start = time.time()
    
    for i, q in enumerate(questions, 1):
        response = await ask(q)
        print(f"  {i}. {q} → {len(response)} chars")
    
    seq_time = time.time() - start
    print(f"  Total: {seq_time:.2f}s\n")
    
    # Concurrent
    print("🚀 Async Concurrent")
    print("-" * 40)
    start = time.time()
    
    responses = await asyncio.gather(*[ask(q) for q in questions])
    
    for i, (q, r) in enumerate(zip(questions, responses), 1):
        print(f"  {i}. {q} → {len(r)} chars")
    
    concurrent_time = time.time() - start
    print(f"  Total: {concurrent_time:.2f}s\n")
    
    return seq_time, concurrent_time


async def streaming_demo():
    """Demonstrate streaming benefits."""
    print("🌊 Streaming Demo")
    print("-" * 40)
    
    prompt = "Count from 1 to 5"
    
    # Non-streaming
    start = time.time()
    response = await ask(prompt)
    non_stream_time = time.time() - start
    print(f"  Non-streaming: {non_stream_time:.2f}s ({len(response)} chars)")
    
    # Streaming
    start = time.time()
    first_token_time = None
    tokens = []
    
    async for token in stream(prompt):
        if first_token_time is None:
            first_token_time = time.time() - start
        tokens.append(token)
    
    stream_time = time.time() - start
    print(f"  First token: {first_token_time:.2f}s")
    print(f"  Complete: {stream_time:.2f}s")
    print(f"  Speedup: {non_stream_time/first_token_time:.1f}x perceived\n")


async def provider_race():
    """Race providers against each other."""
    print("🏁 Provider Race")
    print("-" * 40)
    
    question = "What is 2+2?"
    providers = ["openai", "anthropic", "groq"]
    
    async def test_provider(provider):
        try:
            start = time.time()
            response = await ask(question, provider=provider)
            return provider, time.time() - start, True
        except:
            return provider, 0, False
    
    results = await asyncio.gather(*[test_provider(p) for p in providers])
    
    # Sort by speed
    results.sort(key=lambda x: x[1] if x[2] else float('inf'))
    
    for provider, duration, success in results:
        if success:
            print(f"  {provider}: {duration:.2f}s ✅")
        else:
            print(f"  {provider}: Failed ❌")
    
    if results[0][2]:
        print(f"\n  🏆 Winner: {results[0][0]}!")


async def main_async():
    """Run all async demos."""
    seq_time, concurrent_time = await async_benchmark()
    await streaming_demo()
    await provider_race()
    return seq_time, concurrent_time


def main():
    """Main entry point."""
    print("⚡ ChukLLM Performance Demo")
    print("=" * 50)
    print()
    
    # Run sync benchmark
    sync_time = sync_benchmark()
    
    # Run async benchmarks
    seq_time, concurrent_time = asyncio.run(main_async())
    
    # Summary
    print("\n📊 Summary")
    print("=" * 50)
    print(f"Synchronous:      {sync_time:.2f}s (baseline)")
    print(f"Async Sequential: {seq_time:.2f}s ({sync_time/seq_time:.1f}x)")
    print(f"Async Concurrent: {concurrent_time:.2f}s ({sync_time/concurrent_time:.1f}x)")
    print(f"\n⚡ {sync_time/concurrent_time:.1f}x speedup with async concurrent!")
    print(f"💰 {sync_time - concurrent_time:.1f}s saved per request batch")


if __name__ == "__main__":
    main()