#!/usr/bin/env python3
"""Test streaming and provide workaround if needed."""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

async def test_streaming_approaches():
    """Test different approaches to streaming."""
    
    print("🧪 Testing Streaming Approaches")
    print("=" * 40)
    
    # Approach 1: Direct import and test
    print("\n1️⃣ Testing direct stream import:")
    try:
        from chuk_llm.api.core import stream
        
        print(f"   Stream function: {stream}")
        print(f"   Function type: {type(stream)}")
        
        # Test what it returns
        stream_result = stream("Hello")
        print(f"   Stream result type: {type(stream_result)}")
        print(f"   Has __aiter__: {hasattr(stream_result, '__aiter__')}")
        print(f"   Has __anext__: {hasattr(stream_result, '__anext__')}")
        
        # Try to iterate
        print("   Attempting iteration:")
        chunk_count = 0
        async for chunk in stream_result:
            print(f"   Chunk {chunk_count}: '{chunk[:20]}{'...' if len(chunk) > 20 else ''}'")
            chunk_count += 1
            if chunk_count >= 3:  # Just get first few chunks
                break
        
        print(f"   ✅ Direct streaming works! Got {chunk_count} chunks")
        
    except Exception as e:
        print(f"   ❌ Direct streaming failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Approach 2: Test via chuk_llm import
    print("\n2️⃣ Testing via main chuk_llm import:")
    try:
        from chuk_llm import stream as main_stream
        
        print(f"   Main stream function: {main_stream}")
        print(f"   Same as direct? {main_stream is stream}")
        
        # Test iteration
        print("   Testing main import streaming:")
        chunk_count = 0
        async for chunk in main_stream("Count: 1, 2, 3", max_tokens=10):
            print(f"   Chunk {chunk_count}: '{chunk}'")
            chunk_count += 1
            if chunk_count >= 5:
                break
        
        print(f"   ✅ Main import streaming works! Got {chunk_count} chunks")
        
    except Exception as e:
        print(f"   ❌ Main import streaming failed: {e}")
    
    # Approach 3: Test provider-specific streaming
    print("\n3️⃣ Testing provider-specific streaming:")
    try:
        from chuk_llm import stream_openai
        
        print("   Testing stream_openai:")
        chunk_count = 0
        async for chunk in stream_openai("Hello world", max_tokens=8):
            print(f"   Chunk {chunk_count}: '{chunk}'")
            chunk_count += 1
            if chunk_count >= 3:
                break
        
        print(f"   ✅ Provider streaming works! Got {chunk_count} chunks")
        
    except Exception as e:
        print(f"   ❌ Provider streaming failed: {e}")
    
    # Approach 4: Manual streaming implementation
    print("\n4️⃣ Testing manual streaming workaround:")
    try:
        from chuk_llm import ask
        
        print("   Using ask() with streaming=True:")
        # This is a workaround - we'll ask for a response and simulate streaming
        response = await ask("Write 'Hello streaming world'", max_tokens=10)
        
        # Simulate streaming by breaking response into chunks
        words = response.split()
        for i, word in enumerate(words):
            print(f"{word} ", end="", flush=True)
            await asyncio.sleep(0.1)  # Simulate streaming delay
            if i >= 5:  # Limit output
                break
        
        print("\n   ✅ Manual streaming simulation works!")
        
    except Exception as e:
        print(f"   ❌ Manual streaming failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming_approaches())