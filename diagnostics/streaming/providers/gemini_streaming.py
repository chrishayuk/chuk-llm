# diagnostics/streaming/providers/gemini_streaming.py
"""
Test Gemini provider streaming behavior.
Updated to work with the new chuk-llm architecture.
"""
import asyncio
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_llm.llm.client import get_client

async def test_gemini_streaming():
    """Test Gemini streaming behavior."""
    print("=== Testing Gemini Provider Streaming ===")
    
    try:
        # Get Gemini client
        client = get_client(
            provider="gemini",
            model="gemini-2.0-flash"
        )
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        
        messages = [
            {"role": "user", "content": "Write a creative story about a time traveler who discovers something unexpected. Make it at least 150 words with vivid details."}
        ]
        
        print("\n🔍 Testing Gemini streaming=True...")
        start_time = time.time()
        
        # Test streaming - don't await since it should return async generator
        response = client.create_completion(messages, stream=True)
        
        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")
        
        if hasattr(response, '__aiter__'):
            print("✅ Got async generator")
            
            chunk_count = 0
            first_chunk_time = None
            last_chunk_time = start_time
            full_response = ""
            chunk_times = []
            chunk_intervals = []
            
            print("Response: ", end="", flush=True)
            
            async for chunk in response:
                current_time = time.time()
                relative_time = current_time - start_time
                
                if first_chunk_time is None:
                    first_chunk_time = relative_time
                    print(f"\n🎯 FIRST CHUNK at: {relative_time:.3f}s")
                    print("Response: ", end="", flush=True)
                else:
                    # Track intervals between chunks
                    interval = current_time - last_chunk_time
                    chunk_intervals.append(interval)
                
                chunk_count += 1
                chunk_times.append(relative_time)
                
                if isinstance(chunk, dict) and "response" in chunk:
                    chunk_text = chunk["response"] or ""
                    print(chunk_text, end="", flush=True)
                    full_response += chunk_text
                
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
            
            print(f"\n\n📊 GEMINI STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s")
            print(f"   Total time: {end_time:.3f}s")
            print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
            print(f"   Response length: {len(full_response)} characters")
            print(f"   Avg chunk interval: {avg_interval*1000:.1f}ms")
            print(f"   Min interval: {min_interval*1000:.1f}ms")
            print(f"   Max interval: {max_interval*1000:.1f}ms")
            
            # Quality assessment
            if chunk_count == 1:
                print("   ⚠️  FAKE STREAMING: Only one chunk (entire response at once)")
            elif chunk_count < 5:
                print("   ⚠️  LIMITED STREAMING: Very few chunks")
            else:
                print("   ✅ REAL STREAMING: Multiple chunks detected")
            
            if first_chunk_time < 1.0:
                print("   🚀 EXCELLENT: Very fast first chunk")
            elif first_chunk_time < 2.0:
                print("   ✅ GOOD: Fast first chunk")
            else:
                print("   ⚠️  SLOW: First chunk could be faster")
            
            streaming_duration = end_time - first_chunk_time if first_chunk_time else 0
            if streaming_duration > 3.0:
                print("   ✅ TRUE STREAMING: Long streaming duration")
            elif streaming_duration > 1.0:
                print("   ✅ PARTIAL STREAMING: Some streaming detected")
            else:
                print("   ⚠️  BUFFERED: Very short streaming duration")
            
            # Gemini-specific analysis
            if avg_interval < 0.001:
                print("   🚨 SEVERELY BUFFERED: Chunks arriving instantly")
            elif avg_interval < 0.01:
                print("   ⚠️  MICRO-BUFFERED: Very fast chunks")
            elif avg_interval < 0.1:
                print("   ✅ GOOD STREAMING: Reasonable timing")
            else:
                print("   🐌 SLOW: Large gaps between chunks")
            
            # Calculate streaming efficiency
            if chunk_count > 0 and len(full_response) > 0:
                chars_per_chunk = len(full_response) / chunk_count
                print(f"   📝 Avg characters per chunk: {chars_per_chunk:.1f}")
                
                if chars_per_chunk < 5:
                    print("   ✅ GRANULAR: Very fine-grained streaming")
                elif chars_per_chunk < 20:
                    print("   ✅ BALANCED: Good streaming granularity")
                else:
                    print("   ⚠️  COARSE: Large chunks (less granular)")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Gemini streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Gemini: {e}")
        import traceback
        traceback.print_exc()

async def test_gemini_streaming_patterns():
    """Test different types of responses to understand Gemini's streaming patterns."""
    print("\n\n=== Gemini Streaming Patterns Analysis ===")
    
    test_cases = [
        ("Short response", "Write a haiku about AI"),
        ("Medium response", "Explain quantum computing in 2 paragraphs"),
        ("Long response", "Write a detailed story about space exploration with multiple characters"),
        ("Technical response", "Explain the differences between neural networks and decision trees"),
        ("Creative response", "Create a dialogue between two AI systems discussing creativity")
    ]
    
    try:
        client = get_client(provider="gemini", model="gemini-2.0-flash")
        
        for test_name, prompt in test_cases:
            print(f"\n📝 {test_name}: '{prompt[:50]}...'")
            
            messages = [{"role": "user", "content": prompt}]
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            total_chars = 0
            
            async for chunk in response:
                current_time = time.time()
                
                if first_chunk_time is None:
                    first_chunk_time = current_time - start_time
                
                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    total_chars += len(chunk["response"])
                
                # Limit chunks for analysis
                if chunk_count >= 20:
                    break
            
            total_time = time.time() - start_time
            streaming_duration = total_time - first_chunk_time if first_chunk_time else 0
            
            print(f"   Chunks: {chunk_count}, First: {first_chunk_time:.3f}s, "
                  f"Duration: {streaming_duration:.3f}s, Chars: {total_chars}")
            
            # Pattern analysis
            if chunk_count == 1:
                print("   Pattern: SINGLE_CHUNK (no streaming)")
            elif chunk_count < 5:
                print("   Pattern: FEW_CHUNKS (limited streaming)")
            elif streaming_duration < 0.5:
                print("   Pattern: FAST_BURST (quick streaming)")
            else:
                print("   Pattern: TRUE_STREAMING (gradual delivery)")
    
    except Exception as e:
        print(f"❌ Error in pattern analysis: {e}")

async def test_gemini_vs_others():
    """Compare Gemini streaming against other providers."""
    print("\n\n=== Gemini vs Other Providers ===")
    
    # Test same prompt across providers
    messages = [{"role": "user", "content": "Write a short poem about technology"}]
    
    providers = [
        ("gemini", "gemini-2.0-flash"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514")
    ]
    
    results = {}
    
    for provider, model in providers:
        print(f"\n🔍 Testing {provider}...")
        
        try:
            client = get_client(provider=provider, model=model)
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            content_length = 0
            
            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                
                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    content_length += len(chunk["response"])
                
                # Limit for comparison
                if chunk_count >= 15:
                    break
            
            total_time = time.time() - start_time
            
            results[provider] = {
                "chunks": chunk_count,
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "content_length": content_length
            }
            
            print(f"   {provider}: {chunk_count} chunks, first at {first_chunk_time:.3f}s, {content_length} chars")
            
        except Exception as e:
            print(f"   {provider}: Error - {e}")
            results[provider] = None
    
    # Compare results
    print("\n📊 STREAMING COMPARISON:")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        # Find fastest first chunk
        fastest_provider = min(valid_results.keys(), 
                             key=lambda p: valid_results[p]["first_chunk"])
        fastest_time = valid_results[fastest_provider]["first_chunk"]
        
        print(f"   🚀 Fastest first chunk: {fastest_provider} ({fastest_time:.3f}s)")
        
        # Find most granular streaming
        most_chunks_provider = max(valid_results.keys(),
                                 key=lambda p: valid_results[p]["chunks"])
        most_chunks = valid_results[most_chunks_provider]["chunks"]
        
        print(f"   📊 Most granular streaming: {most_chunks_provider} ({most_chunks} chunks)")
        
        # Gemini-specific analysis
        if "gemini" in valid_results:
            gemini_result = valid_results["gemini"]
            other_providers = [p for p in valid_results.keys() if p != "gemini"]
            
            if other_providers:
                avg_other_first_chunk = sum(valid_results[p]["first_chunk"] for p in other_providers) / len(other_providers)
                
                if gemini_result["first_chunk"] < avg_other_first_chunk:
                    diff = avg_other_first_chunk - gemini_result["first_chunk"]
                    print(f"   ✅ Gemini faster than average by {diff*1000:.0f}ms")
                else:
                    diff = gemini_result["first_chunk"] - avg_other_first_chunk
                    print(f"   ⚠️  Gemini slower than average by {diff*1000:.0f}ms")

async def main():
    """Run all Gemini streaming tests."""
    await test_gemini_streaming()
    await test_gemini_streaming_patterns()
    await test_gemini_vs_others()

if __name__ == "__main__":
    asyncio.run(main())