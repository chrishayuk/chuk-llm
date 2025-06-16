# diagnostics/streaming/providers/groq_streaming.py
"""
Test Groq provider streaming behavior.
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

async def test_groq_streaming():
    """Test Groq streaming behavior."""
    print("=== Testing Groq Provider Streaming ===")
    
    try:
        # Get Groq client
        client = get_client(
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        
        messages = [
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
                
                # Show timing for first few chunks and every 15th chunk
                if chunk_count <= 5 or chunk_count % 15 == 0:
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
            
            print(f"\n\n📊 GROQ STREAMING ANALYSIS:")
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
            
            if first_chunk_time < 0.5:
                print("   🚀 BLAZING FAST: Extremely quick first chunk")
            elif first_chunk_time < 1.0:
                print("   🚀 VERY FAST: Excellent first chunk time")
            elif first_chunk_time < 2.0:
                print("   ✅ FAST: Good first chunk time")
            else:
                print("   ⚠️  SLOW: First chunk could be faster")
            
            streaming_duration = end_time - first_chunk_time if first_chunk_time else 0
            if streaming_duration > 2.0:
                print("   ✅ TRUE STREAMING: Long streaming duration")
            elif streaming_duration > 0.5:
                print("   ✅ PARTIAL STREAMING: Some streaming detected")
            else:
                print("   ⚠️  BUFFERED: Very short streaming duration")
            
            # Groq-specific analysis (known for speed)
            if avg_interval < 0.001:
                print("   🚨 SEVERELY BUFFERED: Chunks arriving instantly")
            elif avg_interval < 0.005:
                print("   ⚡ ULTRA-FAST: Groq's signature speed")
            elif avg_interval < 0.02:
                print("   🚀 VERY FAST: Excellent streaming speed")
            elif avg_interval < 0.05:
                print("   ✅ FAST: Good streaming speed")
            else:
                print("   ⚠️  SLOW: Unexpectedly slow for Groq")
            
            # Calculate tokens per second estimate
            if streaming_duration > 0 and len(full_response) > 0:
                # Rough estimate: 4 chars per token
                estimated_tokens = len(full_response) / 4
                tokens_per_second = estimated_tokens / streaming_duration
                print(f"   📈 Estimated speed: {tokens_per_second:.0f} tokens/second")
                
                if tokens_per_second > 200:
                    print("   🚀 SPEED DEMON: Exceptionally fast generation")
                elif tokens_per_second > 100:
                    print("   ⚡ VERY FAST: Excellent generation speed")
                elif tokens_per_second > 50:
                    print("   ✅ FAST: Good generation speed")
                else:
                    print("   ⚠️  SLOW: Below expected Groq performance")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Groq streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Groq: {e}")
        import traceback
        traceback.print_exc()

async def test_groq_speed_variants():
    """Test different Groq models to compare speed characteristics."""
    print("\n\n=== Groq Model Speed Comparison ===")
    
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    
    prompt = "Write a creative short story about AI in 2 paragraphs"
    messages = [{"role": "user", "content": prompt}]
    
    results = {}
    
    for model in groq_models:
        print(f"\n🔍 Testing {model}...")
        
        try:
            client = get_client(provider="groq", model=model)
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            total_chars = 0
            intervals = []
            last_time = start_time
            
            async for chunk in response:
                current_time = time.time()
                
                if first_chunk_time is None:
                    first_chunk_time = current_time - start_time
                else:
                    intervals.append(current_time - last_time)
                
                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    total_chars += len(chunk["response"])
                
                last_time = current_time
                
                # Limit for comparison
                if chunk_count >= 25:
                    break
            
            total_time = time.time() - start_time
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            
            results[model] = {
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "chunks": chunk_count,
                "chars": total_chars,
                "avg_interval": avg_interval
            }
            
            print(f"   {model}: {chunk_count} chunks, first at {first_chunk_time:.3f}s")
            print(f"   Avg interval: {avg_interval*1000:.1f}ms, {total_chars} chars")
            
        except Exception as e:
            print(f"   {model}: Error - {e}")
            results[model] = None
    
    # Compare results
    print("\n📊 GROQ MODEL COMPARISON:")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        # Find fastest model
        fastest_model = min(valid_results.keys(), 
                          key=lambda m: valid_results[m]["first_chunk"])
        fastest_time = valid_results[fastest_model]["first_chunk"]
        
        print(f"   🚀 Fastest first chunk: {fastest_model} ({fastest_time:.3f}s)")
        
        # Find most efficient streaming
        best_streaming = min(valid_results.keys(),
                           key=lambda m: valid_results[m]["avg_interval"])
        best_interval = valid_results[best_streaming]["avg_interval"]
        
        print(f"   ⚡ Best streaming: {best_streaming} ({best_interval*1000:.1f}ms avg)")
        
        # Speed rankings
        print("\n   🏆 Speed Rankings:")
        sorted_by_speed = sorted(valid_results.items(), 
                               key=lambda x: x[1]["first_chunk"])
        
        for i, (model, data) in enumerate(sorted_by_speed, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔹"
            short_name = model.split("-")[1] if "-" in model else model
            print(f"   {emoji} {short_name}: {data['first_chunk']:.3f}s first chunk")

async def test_groq_vs_competition():
    """Compare Groq against other fast providers."""
    print("\n\n=== Groq vs Competition (Speed Test) ===")
    
    # Fast providers comparison
    providers = [
        ("groq", "llama-3.1-8b-instant"),  # Groq's fastest
        ("openai", "gpt-4o-mini"),         # OpenAI's fast model
        ("anthropic", "claude-sonnet-4-20250514")  # Anthropic's balanced
    ]
    
    messages = [{"role": "user", "content": "Count from 1 to 10 quickly"}]
    
    results = {}
    
    for provider, model in providers:
        print(f"\n🔍 Testing {provider} ({model})...")
        
        try:
            client = get_client(provider=provider, model=model)
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            
            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                
                chunk_count += 1
                
                # Limit for fair comparison
                if chunk_count >= 10:
                    break
            
            total_time = time.time() - start_time
            
            results[provider] = {
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "chunks": chunk_count
            }
            
            print(f"   {provider}: {chunk_count} chunks, first at {first_chunk_time:.3f}s")
            
        except Exception as e:
            print(f"   {provider}: Error - {e}")
            results[provider] = None
    
    # Speed comparison
    print("\n📊 SPEED COMPARISON:")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if "groq" in valid_results and len(valid_results) >= 2:
        groq_time = valid_results["groq"]["first_chunk"]
        
        print(f"   Groq first chunk: {groq_time:.3f}s")
        
        for provider, data in valid_results.items():
            if provider != "groq":
                other_time = data["first_chunk"]
                if groq_time < other_time:
                    speedup = other_time / groq_time
                    print(f"   Groq is {speedup:.1f}x faster than {provider}")
                else:
                    slowdown = groq_time / other_time
                    print(f"   {provider} is {slowdown:.1f}x faster than Groq")
        
        # Overall assessment
        all_times = [data["first_chunk"] for data in valid_results.values()]
        if groq_time == min(all_times):
            print("   🏆 GROQ WINS: Fastest overall")
        else:
            print("   🤔 Groq not the fastest in this test")

async def main():
    """Run all Groq streaming tests."""
    await test_groq_streaming()
    await test_groq_speed_variants()
    await test_groq_vs_competition()

if __name__ == "__main__":
    asyncio.run(main())