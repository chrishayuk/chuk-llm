#!/usr/bin/env python3
"""
Streaming example - watch responses appear in real-time
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import stream_openai_gpt4o_mini, stream_anthropic_sonnet

async def main():
    print("=== Streaming Demo ===\n")
    
    # Stream from OpenAI
    print("Streaming from OpenAI:")
    async for chunk in stream_openai_gpt4o_mini("Write a haiku about AI"):
        print(chunk, end="", flush=True)
    print("\n")
    
    # Stream from Anthropic
    print("Streaming from Anthropic:")
    async for chunk in stream_anthropic_sonnet("Tell me a short joke"):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())