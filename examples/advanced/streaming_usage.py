#!/usr/bin/env python3
"""
Proper usage of streaming functions in ChukLLM.

Streaming functions are async-only and return async generators.
For sync contexts, use the ask functions instead.
"""

import asyncio
from chuk_llm import stream_ollama_granite, ask_ollama_granite

# Example 1: Streaming in async context (CORRECT)
async def async_streaming_example():
    print("=== Async Streaming Example ===")
    print("Streaming response: ", end="", flush=True)
    
    async for chunk in stream_ollama_granite("Tell me a short joke"):
        print(chunk, end="", flush=True)
    print("\n")

# Example 2: For sync context, use ask instead of stream
def sync_alternative():
    print("=== Sync Alternative (using ask) ===")
    # In sync context, use ask functions which auto-detect and work synchronously
    result = ask_ollama_granite("Tell me a short joke")
    print(f"Complete response: {result}")
    print()

# Example 3: If you really need streaming-like behavior in sync, 
# you can wrap it, but this is not recommended
def sync_streaming_wrapper():
    print("=== Sync Wrapper for Streaming (not recommended) ===")
    
    async def get_streamed_response():
        chunks = []
        async for chunk in stream_ollama_granite("Tell me a short joke"):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        return ''.join(chunks)
    
    # Run the async function synchronously
    result = asyncio.run(get_streamed_response())
    print("\n")
    return result

def main():
    print("ChukLLM Streaming Usage Examples")
    print("=" * 50)
    print()
    
    # Run async streaming example
    print("Running async streaming (recommended for streaming):")
    asyncio.run(async_streaming_example())
    
    # Show sync alternative
    print("For sync contexts, use ask functions:")
    sync_alternative()
    
    # Show wrapper approach (not recommended)
    print("Wrapper approach (creates new event loop):")
    sync_streaming_wrapper()
    
    print("=" * 50)
    print("\n📚 Summary:")
    print("- Streaming functions (stream_*) are async-only")
    print("- Use 'async for' to iterate over streamed chunks")
    print("- In sync contexts, use ask_* functions instead")
    print("- ask_* functions auto-detect context and work both sync/async")

if __name__ == "__main__":
    main()