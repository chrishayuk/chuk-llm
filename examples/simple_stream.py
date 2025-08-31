#!/usr/bin/env python3
"""Simple streaming example with ChukLLM"""

import asyncio
from chuk_llm import stream_ollama_granite

async def stream_example():
    """Stream a response token by token"""
    print("Streaming response from Ollama Granite:")
    print("-" * 40)
    
    # Stream the response - each chunk appears as it's generated
    async for chunk in stream_ollama_granite("Write a haiku about coding"):
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 40)

# Run the async function
if __name__ == "__main__":
    asyncio.run(stream_example())