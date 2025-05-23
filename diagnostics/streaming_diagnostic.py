# diagnostics/streaming_diagnostic.py
"""
Simple Streaming Diagnostic
"""
import asyncio
import time
from chuk_llm.llm.llm_client import get_llm_client
from chuk_llm.llm.provider_config import ProviderConfig

async def test_fixed_chuk_llm():
    """Test the fixed chuk_llm client directly."""
    print("=== Testing Fixed chuk_llm Client ===")
    
    # Get client directly (same as ChukAgent does)
    client = get_llm_client(
        provider="openai",
        model="gpt-4o-mini",
        config=ProviderConfig()
    )
    
    print(f"Client type: {type(client)}")
    print(f"Client class: {client.__class__.__name__}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 10, one number per line"}
    ]
    
    print("\n🔍 Testing streaming=True...")
    start_time = time.time()
    
    try:
        # Call with streaming
        response = client.create_completion(messages, stream=True)
        
        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")
        
        # Check if it's an async generator (not awaited)
        if hasattr(response, '__aiter__'):
            print("✅ Got async generator directly (not awaited)")
            
            chunk_count = 0
            first_chunk_time = None
            
            print("Response: ", end="", flush=True)
            
            async for chunk in response:
                current_time = time.time()
                relative_time = current_time - start_time
                
                if first_chunk_time is None:
                    first_chunk_time = relative_time
                    print(f"\n🎯 FIRST CHUNK at: {relative_time:.3f}s")
                    print("Response: ", end="", flush=True)
                
                chunk_count += 1
                
                if isinstance(chunk, dict) and "response" in chunk:
                    chunk_text = chunk["response"]
                    print(chunk_text, end="", flush=True)
                
                # Show timing for first few chunks
                if chunk_count <= 3:
                    interval = relative_time - (first_chunk_time if chunk_count == 1 else last_time)
                    print(f"\n   Chunk {chunk_count}: {relative_time:.3f}s")
                    print("   Continuing: ", end="", flush=True)
                    
                last_time = relative_time
            
            end_time = time.time() - start_time
            print(f"\n\n📊 CHUK_LLM STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s")
            print(f"   Total time: {end_time:.3f}s")
            print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
            
            if first_chunk_time and first_chunk_time < 3.0:
                print("   ✅ GOOD: Quick first chunk")
            else:
                print("   ⚠️  SLOW: First chunk took too long")
                
            if end_time - first_chunk_time > 0.5:
                print("   ✅ STREAMING: Real streaming detected")
            else:
                print("   ⚠️  BUFFERED: Chunks arrived too quickly")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🔍 Testing streaming=False...")
    try:
        # Test non-streaming
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print("✅ Non-streaming works correctly")
        else:
            print(f"❌ Unexpected non-streaming response: {response}")
            
    except Exception as e:
        print(f"❌ Non-streaming error: {e}")

if __name__ == "__main__":
    asyncio.run(test_fixed_chuk_llm())