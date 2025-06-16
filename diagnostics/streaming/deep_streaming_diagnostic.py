# diagnostics/streaming/deep_streaming_diagnostic.py
"""
Deep diagnostic to find where streaming is getting buffered.
Updated to work with the new chuk-llm architecture.
"""
import asyncio
import time
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def test_client_directly():
    """Test the LLM client directly to see if it's the source of buffering."""
    print("=== Testing LLM Client Directly ===")
    
    # Create client directly (same as ChukAgent does)
    client = get_client(
        provider="openai",
        model="gpt-4o-mini"
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short story about a cat. Take your time and make it detailed."}
    ]
    
    print("🔍 Testing direct LLM client streaming...")
    start_time = time.time()
    
    try:
        # Call the client directly with streaming - NO AWAIT
        response_stream = client.create_completion(
            messages=messages,
            stream=True
        )
        
        print(f"⏱️  Response stream type: {type(response_stream)}")
        print(f"⏱️  Has __aiter__: {hasattr(response_stream, '__aiter__')}")
        
        chunk_count = 0
        first_chunk_time = None
        last_chunk_time = start_time
        chunk_intervals = []
        
        print("🔄 Iterating through LLM client stream...")
        print("Response: ", end="", flush=True)
        
        async for chunk in response_stream:
            current_time = time.time()
            relative_time = current_time - start_time
            
            if first_chunk_time is None:
                first_chunk_time = relative_time
                print(f"\n🎯 FIRST CHUNK from LLM at: {relative_time:.3f}s")
                print("Response: ", end="", flush=True)
            else:
                # Track intervals between chunks
                interval = current_time - last_chunk_time
                chunk_intervals.append(interval)
            
            chunk_count += 1
            
            # Show chunk details
            if isinstance(chunk, dict) and "response" in chunk:
                chunk_text = chunk["response"] or ""
                print(chunk_text, end="", flush=True)
            
            # Show timing for first few chunks and every 10th chunk
            if chunk_count <= 5 or chunk_count % 10 == 0:
                interval = current_time - last_chunk_time
                print(f"\n   Chunk {chunk_count}: {relative_time:.3f}s (interval: {interval:.4f}s)")
                print("   Continuing: ", end="", flush=True)
            
            last_chunk_time = current_time
        
        end_time = time.time() - start_time
        
        # Calculate interval statistics
        if chunk_intervals:
            avg_interval = sum(chunk_intervals) / len(chunk_intervals)
            min_interval = min(chunk_intervals)
            max_interval = max(chunk_intervals)
        else:
            avg_interval = min_interval = max_interval = 0
        
        print(f"\n\n📊 LLM CLIENT ANALYSIS:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   First chunk delay: {first_chunk_time:.3f}s")
        print(f"   Total time: {end_time:.3f}s")
        print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
        print(f"   Avg interval: {avg_interval*1000:.1f}ms")
        print(f"   Min interval: {min_interval*1000:.1f}ms")
        print(f"   Max interval: {max_interval*1000:.1f}ms")
        
        # Streaming quality assessment
        if first_chunk_time < 1.0:
            print("   ✅ EXCELLENT: Very fast first chunk")
        elif first_chunk_time < 2.0:
            print("   ✅ GOOD: Fast first chunk")
        else:
            print("   ⚠️  SLOW: First chunk could be faster")
        
        streaming_duration = end_time - first_chunk_time if first_chunk_time else 0
        if streaming_duration > 2.0:
            print("   ✅ TRUE STREAMING: Long streaming duration")
        elif streaming_duration > 0.5:
            print("   ✅ PARTIAL STREAMING: Some streaming detected")
        else:
            print("   ⚠️  BUFFERED: Very short streaming duration")
        
        if avg_interval > 0.02:  # More than 20ms
            print("   ✅ NATURAL PACING: Realistic chunk intervals")
        else:
            print("   ⚠️  RAPID DELIVERY: Chunks arriving very quickly")
        
    except Exception as e:
        print(f"❌ Error with direct LLM client: {e}")
        import traceback
        traceback.print_exc()

async def test_raw_openai():
    """Test raw OpenAI client to establish baseline."""
    print("\n=== Testing Raw OpenAI Client ===")
    
    try:
        import openai
        
        # Use environment variables for API key
        client = openai.AsyncOpenAI()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short story about a cat. Take your time and make it detailed."}
        ]
        
        print("🔍 Testing raw OpenAI streaming...")
        start_time = time.time()
        
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        chunk_count = 0
        first_chunk_time = None
        last_chunk_time = start_time
        chunk_intervals = []
        
        print("Response: ", end="", flush=True)
        
        async for chunk in stream:
            current_time = time.time()
            relative_time = current_time - start_time
            
            if first_chunk_time is None:
                first_chunk_time = relative_time
                print(f"\n🎯 FIRST CHUNK from raw OpenAI at: {relative_time:.3f}s")
                print("Response: ", end="", flush=True)
            else:
                interval = current_time - last_chunk_time
                chunk_intervals.append(interval)
            
            chunk_count += 1
            
            # Extract content from OpenAI chunk
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
            
            # Show timing for first few chunks
            if chunk_count <= 5:
                interval = current_time - last_chunk_time
                print(f"\n   Raw chunk {chunk_count}: {relative_time:.3f}s (interval: {interval:.4f}s)")
                print("   Continuing: ", end="", flush=True)
            
            last_chunk_time = current_time
        
        end_time = time.time() - start_time
        
        # Calculate interval statistics
        if chunk_intervals:
            avg_interval = sum(chunk_intervals) / len(chunk_intervals)
            min_interval = min(chunk_intervals)
            max_interval = max(chunk_intervals)
        else:
            avg_interval = min_interval = max_interval = 0
        
        print(f"\n\n📊 RAW OPENAI ANALYSIS:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   First chunk delay: {first_chunk_time:.3f}s")
        print(f"   Total time: {end_time:.3f}s")
        print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
        print(f"   Avg interval: {avg_interval*1000:.1f}ms")
        print(f"   Min interval: {min_interval*1000:.1f}ms")
        print(f"   Max interval: {max_interval*1000:.1f}ms")
        
    except ImportError:
        print("❌ OpenAI library not available for baseline test")
    except Exception as e:
        print(f"❌ Error with raw OpenAI: {e}")

async def test_comparison():
    """Compare chuk-llm vs raw OpenAI side by side."""
    print("\n=== Side-by-Side Comparison ===")
    
    # Same message for both
    messages = [
        {"role": "user", "content": "Write a haiku about streaming data"}
    ]
    
    print("🔍 Testing both implementations with same prompt...")
    
    # Test chuk-llm
    print("\n--- chuk-llm Implementation ---")
    try:
        client = get_client(provider="openai", model="gpt-4o-mini")
        
        start_time = time.time()
        response_stream = client.create_completion(messages, stream=True)
        
        chuk_chunks = 0
        chuk_first_chunk = None
        
        async for chunk in response_stream:
            if chuk_first_chunk is None:
                chuk_first_chunk = time.time() - start_time
            chuk_chunks += 1
            if chuk_chunks >= 10:  # Limit for comparison
                break
        
        chuk_total = time.time() - start_time
        print(f"   chuk-llm: {chuk_chunks} chunks, first at {chuk_first_chunk:.3f}s, total {chuk_total:.3f}s")
        
    except Exception as e:
        print(f"   chuk-llm error: {e}")
        chuk_first_chunk = chuk_total = None
    
    # Test raw OpenAI
    print("\n--- Raw OpenAI Implementation ---")
    try:
        import openai
        raw_client = openai.AsyncOpenAI()
        
        start_time = time.time()
        stream = await raw_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        raw_chunks = 0
        raw_first_chunk = None
        
        async for chunk in stream:
            if raw_first_chunk is None:
                raw_first_chunk = time.time() - start_time
            raw_chunks += 1
            if raw_chunks >= 10:  # Limit for comparison
                break
        
        raw_total = time.time() - start_time
        print(f"   Raw OpenAI: {raw_chunks} chunks, first at {raw_first_chunk:.3f}s, total {raw_total:.3f}s")
        
    except Exception as e:
        print(f"   Raw OpenAI error: {e}")
        raw_first_chunk = raw_total = None
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    if chuk_first_chunk and raw_first_chunk:
        diff = chuk_first_chunk - raw_first_chunk
        if abs(diff) < 0.1:
            print(f"   ✅ SIMILAR PERFORMANCE: {diff*1000:.0f}ms difference")
        elif diff > 0:
            print(f"   ⚠️  chuk-llm slower by {diff*1000:.0f}ms")
        else:
            print(f"   🚀 chuk-llm faster by {abs(diff)*1000:.0f}ms")
    else:
        print("   ❌ Could not compare - one or both tests failed")

async def main():
    """Run all diagnostic tests."""
    await test_client_directly()
    await test_raw_openai()
    await test_comparison()

if __name__ == "__main__":
    asyncio.run(main())