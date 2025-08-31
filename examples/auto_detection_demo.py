#!/usr/bin/env python3
"""
Demonstration of auto-detection for sync/async context in ChukLLM.

The same function works in both sync and async contexts without
needing to use _sync suffix or await in sync code.
"""

import asyncio
from chuk_llm import ask_ollama_granite

# Example 1: Sync usage (no await needed)
def sync_example():
    print("=== Sync Context Example ===")
    # This automatically detects sync context and runs synchronously
    result = ask_ollama_granite("What's 2+2? Answer in one word.")
    print(f"Sync result: {result}")
    print(f"Type: {type(result)}")  # Should be str, not coroutine
    print()

# Example 2: Async usage (with await)
async def async_example():
    print("=== Async Context Example ===")
    # This automatically detects async context and returns a coroutine
    result = await ask_ollama_granite("What's 3+3? Answer in one word.")
    print(f"Async result: {result}")
    print(f"Type: {type(result)}")  # Should be str
    print()

# Example 3: Multiple async calls in parallel
async def parallel_example():
    print("=== Parallel Async Example ===")
    # Create multiple coroutines
    tasks = [
        ask_ollama_granite("What's 4+4? Answer in one word."),
        ask_ollama_granite("What's 5+5? Answer in one word."),
        ask_ollama_granite("What's 6+6? Answer in one word.")
    ]
    # Run them in parallel
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results, 1):
        print(f"Task {i}: {result}")
    print()

def main():
    print("ChukLLM Auto-Detection Demo")
    print("=" * 40)
    print()
    
    # Run sync example
    sync_example()
    
    # Run async examples
    asyncio.run(async_example())
    asyncio.run(parallel_example())
    
    print("=" * 40)
    print("✅ All examples completed successfully!")
    print("\nKey takeaways:")
    print("1. No need for _sync suffix in synchronous code")
    print("2. Same function works with await in async context")
    print("3. Enables parallel execution in async contexts")

if __name__ == "__main__":
    main()