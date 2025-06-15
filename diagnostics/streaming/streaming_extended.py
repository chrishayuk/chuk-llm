#!/usr/bin/env python3
# diagnostics/streaming/streaming_extended.py
"""
Extended Streaming Test - Tests with longer content to verify real streaming
"""
import asyncio
import time
from chuk_llm.llm.client import get_client
from chuk_llm.llm.configuration.provider_config import ProviderConfig

async def test_extended_streaming():
    """Test with a longer response to verify true streaming."""
    print("=== Extended Streaming Test ===")
    print("Testing with longer content to verify real-time streaming\n")
    
    client = get_client(
        provider="openai",
        model="gpt-4o-mini",
        config=ProviderConfig()
    )
    
    # Ask for a longer response that will take time to generate
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """Write a short story about a robot learning to paint. 
        Include at least 5 paragraphs. Make each paragraph at least 3 sentences long.
        Number each paragraph."""}
    ]
    
    print("🔍 Testing streaming with longer content...")
    start_time = time.time()
    
    response = client.create_completion(messages, stream=True)
    
    if not hasattr(response, '__aiter__'):
        print(f"❌ ERROR: Expected async generator, got {type(response)}")
        return
    
    print("✅ Got async generator\n")
    
    chunk_count = 0
    first_chunk_time = None
    last_chunk_time = start_time
    chunk_delays = []
    total_content = ""
    
    print("Streaming response:\n" + "-"*60)
    
    async for chunk in response:
        current_time = time.time()
        relative_time = current_time - start_time
        
        if first_chunk_time is None:
            first_chunk_time = relative_time
            print(f"\n[FIRST CHUNK at {relative_time:.3f}s]\n")
        
        # Calculate delay between chunks
        chunk_delay = current_time - last_chunk_time
        if chunk_count > 0:  # Skip first chunk
            chunk_delays.append(chunk_delay)
        
        last_chunk_time = current_time
        chunk_count += 1
        
        if isinstance(chunk, dict) and "response" in chunk:
            chunk_text = chunk["response"]
            total_content += chunk_text
            print(chunk_text, end="", flush=True)
    
    print("\n" + "-"*60)
    
    end_time = time.time() - start_time
    
    # Calculate statistics
    avg_delay = sum(chunk_delays) / len(chunk_delays) if chunk_delays else 0
    max_delay = max(chunk_delays) if chunk_delays else 0
    min_delay = min(chunk_delays) if chunk_delays else 0
    
    print(f"\n📊 EXTENDED STREAMING ANALYSIS:")
    print(f"   Total chunks: {chunk_count}")
    print(f"   Total characters: {len(total_content)}")
    print(f"   First chunk delay: {first_chunk_time:.3f}s")
    print(f"   Total time: {end_time:.3f}s")
    print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
    print(f"\n   Chunk arrival stats:")
    print(f"   - Average delay between chunks: {avg_delay*1000:.1f}ms")
    print(f"   - Max delay: {max_delay*1000:.1f}ms")
    print(f"   - Min delay: {min_delay*1000:.1f}ms")
    
    # Evaluate streaming quality
    print(f"\n   📈 Streaming Quality Assessment:")
    
    if first_chunk_time < 1.5:
        print("   ✅ EXCELLENT: Very fast first chunk (<1.5s)")
    elif first_chunk_time < 3.0:
        print("   ✅ GOOD: Fast first chunk (<3s)")
    else:
        print("   ⚠️  SLOW: First chunk delay >3s")
    
    # Check if chunks arrived gradually (true streaming) or all at once (buffered)
    streaming_ratio = (end_time - first_chunk_time) / end_time if end_time > 0 else 0
    
    if streaming_ratio > 0.5 and chunk_count > 10:
        print("   ✅ TRUE STREAMING: Chunks arrived gradually")
    elif streaming_ratio > 0.2:
        print("   ⚠️  PARTIAL STREAMING: Some buffering detected")
    else:
        print("   ❌ BUFFERED: Most chunks arrived immediately")
    
    if avg_delay > 0.01:  # More than 10ms average
        print(f"   ✅ NATURAL PACING: Chunks have realistic delays")
    else:
        print(f"   ⚠️  RAPID DELIVERY: Chunks arriving too quickly")

async def test_parallel_streaming():
    """Test multiple streaming requests in parallel to check for blocking."""
    print("\n\n=== Parallel Streaming Test ===")
    print("Testing 3 simultaneous streaming requests...\n")
    
    client = get_client(
        provider="openai", 
        model="gpt-4o-mini",
        config=ProviderConfig()
    )
    
    async def stream_request(request_id: int):
        messages = [
            {"role": "user", "content": f"Count from {request_id}0 to {request_id}9 slowly"}
        ]
        
        start = time.time()
        response = client.create_completion(messages, stream=True)
        
        first_chunk_time = None
        chunks = []
        
        async for chunk in response:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            
            if isinstance(chunk, dict) and chunk.get("response"):
                chunks.append(chunk["response"])
        
        total_time = time.time() - start
        content = "".join(chunks).strip()
        
        print(f"Request {request_id}: First chunk at {first_chunk_time:.3f}s, "
              f"Total time: {total_time:.3f}s, Content: {content[:50]}...")
        
        return first_chunk_time, total_time
    
    # Run 3 requests in parallel
    start_parallel = time.time()
    results = await asyncio.gather(
        stream_request(1),
        stream_request(2),
        stream_request(3)
    )
    total_parallel_time = time.time() - start_parallel
    
    print(f"\n📊 PARALLEL STREAMING RESULTS:")
    print(f"   Total parallel execution time: {total_parallel_time:.3f}s")
    
    # Check if requests were truly parallel
    max_individual_time = max(r[1] for r in results)
    if total_parallel_time < max_individual_time * 1.5:
        print("   ✅ TRUE PARALLEL: Requests processed simultaneously")
    else:
        print("   ❌ SEQUENTIAL: Requests appear to be blocking each other")

async def test_error_handling():
    """Test streaming behavior with errors."""
    print("\n\n=== Error Handling Test ===")
    
    # Test with invalid model
    try:
        client = get_client(
            provider="openai",
            model="invalid-model-xxx",
            config=ProviderConfig()
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        response = client.create_completion(messages, stream=True)
        
        async for chunk in response:
            if chunk.get("error"):
                print(f"✅ Error properly streamed: {chunk.get('response')}")
                break
            else:
                print(f"Chunk: {chunk}")
                
    except Exception as e:
        print(f"✅ Exception properly raised: {e}")

if __name__ == "__main__":
    print("🚀 Extended LLM Streaming Tests\n")
    
    # Run all tests
    asyncio.run(test_extended_streaming())
    asyncio.run(test_parallel_streaming())
    asyncio.run(test_error_handling())