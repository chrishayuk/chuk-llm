# diagnostics/groq_streaming.py
"""
Test Groq provider streaming behavior.
"""
import asyncio
import time
from chuk_llm.llm.llm_client import get_llm_client
from chuk_llm.llm.provider_config import ProviderConfig

async def test_groq_streaming():
    """Test Groq streaming behavior."""
    print("=== Testing Groq Provider Streaming ===")
    
    try:
        # Get Groq client
        client = get_llm_client(
            provider="groq",
            model="llama-3.3-70b-versatile",
            config=ProviderConfig()
        )
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write an engaging short story about a detective solving a mystery in a futuristic city. Make it at least 200 words with rich descriptions."}
        ]
        
        print("\n🔍 Testing Groq streaming=True...")
        start_time = time.time()
        
        # Test streaming - don't await since it should return async generator
        response = client.create_completion(messages, stream=True)
        
        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")
        
        if hasattr(response, '__aiter__'):
            print("✅ Got async generator")
            
            chunk_count = 0
            first_chunk_time = None
            full_response = ""
            chunk_times = []
            
            print("Response: ", end="", flush=True)
            
            async for chunk in response:
                current_time = time.time()
                relative_time = current_time - start_time
                
                if first_chunk_time is None:
                    first_chunk_time = relative_time
                    print(f"\n🎯 FIRST CHUNK at: {relative_time:.3f}s")
                    print("Response: ", end="", flush=True)
                
                chunk_count += 1
                chunk_times.append(relative_time)
                
                if isinstance(chunk, dict) and "response" in chunk:
                    chunk_text = chunk["response"] or ""
                    print(chunk_text, end="", flush=True)
                    full_response += chunk_text
                
                # Show timing for first few chunks and every 30th chunk
                if chunk_count <= 5 or chunk_count % 30 == 0:
                    interval = relative_time - (chunk_times[-2] if len(chunk_times) > 1 else first_chunk_time)
                    print(f"\n   Chunk {chunk_count}: {relative_time:.3f}s (gap: {interval:.3f}s)")
                    print("   Continuing: ", end="", flush=True)
            
            end_time = time.time() - start_time
            
            print(f"\n\n📊 GROQ STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s")
            print(f"   Total time: {end_time:.3f}s")
            print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
            print(f"   Response length: {len(full_response)} characters")
            
            if chunk_count == 1:
                print("   ⚠️  FAKE STREAMING: Only one chunk (entire response at once)")
            elif chunk_count > 1 and (end_time - first_chunk_time) < 0.1:
                print("   ⚠️  BUFFERED: All chunks arrived too quickly")
            else:
                print("   ✅ REAL STREAMING: Multiple chunks over time")
                
            # Analyze chunk timing if we have enough chunks
            if len(chunk_times) > 5:
                intervals = [chunk_times[i] - chunk_times[i-1] for i in range(1, min(len(chunk_times), 15))]
                avg_interval = sum(intervals) / len(intervals)
                print(f"   Avg chunk interval: {avg_interval:.3f}s")
                
                if avg_interval < 0.001:
                    print("   🚨 SEVERELY BUFFERED: Chunks arriving instantly")
                elif avg_interval < 0.01:
                    print("   ⚠️  MICRO-BUFFERED: Very fast chunks")
                elif avg_interval < 0.1:
                    print("   ✅ GOOD STREAMING: Reasonable timing")
                else:
                    print("   ✅ NATURAL: LLM-paced generation")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Groq streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Groq: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_groq_streaming())