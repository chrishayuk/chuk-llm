#!/usr/bin/env python
"""
streaming_test.py
================
Simple test to verify streaming functionality works correctly.
Run this to debug streaming issues before running full diagnostics.
"""

import asyncio
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client


async def test_streaming(provider: str = "openai", model: str = "gpt-4o-mini"):
    """Test basic streaming functionality"""
    print(f"Testing streaming with {provider}:{model}")

    try:
        # Get client
        client = get_client(provider, model=model)
        print("✅ Client created successfully")

        # Test non-streaming first
        print("\n1. Testing non-streaming...")
        messages = [{"role": "user", "content": "Say hello!"}]
        response = await client.create_completion(messages)
        print(f"   Response: {response.get('response', 'No response')}")

        # Test streaming
        print("\n2. Testing streaming...")
        stream = client.create_completion(messages, stream=True)  # Remove await here
        print(f"   Stream type: {type(stream)}")
        print(f"   Has __aiter__: {hasattr(stream, '__aiter__')}")

        chunks = []
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            print(f"   Chunk {chunk_count}: {chunk}")
            if chunk.get("response"):
                chunks.append(chunk["response"])
            if chunk_count >= 3:  # Just get a few chunks
                break

        full_response = "".join(chunks)
        print(f"   Full response: {full_response}")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Success: {chunk_count > 1 and bool(full_response.strip())}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Test streaming with different providers"""
    providers_to_test = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("groq", "llama-3.3-70b-versatile"),
    ]

    for provider, model in providers_to_test:
        print("=" * 50)
        await test_streaming(provider, model)
        print()


if __name__ == "__main__":
    asyncio.run(main())
