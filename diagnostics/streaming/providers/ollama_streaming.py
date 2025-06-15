# diagnostics/streaming/providers/ollama_streaming.py
"""
Test Ollama provider streaming behavior.
"""
import asyncio
import time
from chuk_llm.llm.client import get_client
from chuk_llm.llm.configuration.provider_config import ProviderConfig

async def test_ollama_streaming():
    """Test Ollama streaming behavior."""
    print("=== Testing Ollama Provider Streaming ===")
    
    try:
        # Get Ollama client
        client = get_client(
            provider="ollama",
            model="qwen3",  # or whatever model you have available
            config=ProviderConfig()
        )
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Model: {client.model}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short creative story about a robot learning to cook. Make it at least 100 words with good details."}
        ]
        
        print("\n🔍 Testing Ollama streaming=True...")
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
                
                # Show timing for first few chunks and every 20th chunk
                if chunk_count <= 5 or chunk_count % 20 == 0:
                    interval = relative_time - (chunk_times[-2] if len(chunk_times) > 1 else first_chunk_time)
                    print(f"\n   Chunk {chunk_count}: {relative_time:.3f}s (gap: {interval:.3f}s)")
                    print("   Continuing: ", end="", flush=True)
            
            end_time = time.time() - start_time
            
            print(f"\n\n📊 OLLAMA STREAMING ANALYSIS:")
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
                intervals = [chunk_times[i] - chunk_times[i-1] for i in range(1, min(len(chunk_times), 10))]
                avg_interval = sum(intervals) / len(intervals)
                print(f"   Avg chunk interval: {avg_interval:.3f}s")
                
                if avg_interval < 0.001:
                    print("   🚨 SEVERELY BUFFERED: Chunks arriving instantly")
                elif avg_interval < 0.01:
                    print("   ⚠️  MICRO-BUFFERED: Very fast chunks")
                elif avg_interval < 0.1:
                    print("   ✅ GOOD STREAMING: Reasonable timing")
                elif avg_interval < 0.5:
                    print("   ✅ NATURAL: LLM-paced generation")
                else:
                    print("   🐌 SLOW: Very large gaps (might be local processing)")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Ollama streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ollama_streaming())