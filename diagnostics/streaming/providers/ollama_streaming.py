# diagnostics/streaming/providers/ollama_streaming.py
"""
Test Ollama provider streaming behavior.
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

async def test_ollama_streaming():
    """Test Ollama streaming behavior."""
    print("=== Testing Ollama Provider Streaming ===")
    
    try:
        # Get Ollama client - try different models that might be available
        available_models = ["gemma3", "llama3", "granite3.3", "mistral"]
        
        client = None
        model_used = None
        
        for model in available_models:
            try:
                print(f"🔍 Trying model: {model}")
                client = get_client(provider="ollama", model=model)
                model_used = model
                print(f"✅ Successfully connected with model: {model}")
                break
            except Exception as e:
                print(f"❌ Failed to connect with {model}: {e}")
                continue
        
        if not client:
            print("❌ Could not connect to any Ollama model")
            print("💡 Make sure Ollama is running and you have models installed:")
            print("   ollama serve")
            print("   ollama pull gemma3  # or another model")
            return
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Model: {model_used}")
        
        messages = [
            {"role": "user", "content": "Write a short creative story about a robot learning to cook. Make it at least 100 words with good details."}
        ]
        
        print(f"\n🔍 Testing Ollama streaming=True with {model_used}...")
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
            
            print(f"\n\n📊 OLLAMA STREAMING ANALYSIS:")
            print(f"   Model: {model_used}")
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
            
            # Local model specific assessment
            if first_chunk_time < 2.0:
                print("   🚀 FAST: Quick local generation")
            elif first_chunk_time < 5.0:
                print("   ✅ REASONABLE: Acceptable local performance")
            else:
                print("   ⚠️  SLOW: Local model might be large or CPU-bound")
            
            streaming_duration = end_time - first_chunk_time if first_chunk_time else 0
            if streaming_duration > 3.0:
                print("   ✅ TRUE STREAMING: Long streaming duration")
            elif streaming_duration > 1.0:
                print("   ✅ PARTIAL STREAMING: Some streaming detected")
            else:
                print("   ⚠️  BUFFERED: Very short streaming duration")
            
            # Ollama-specific analysis (local processing)
            if avg_interval < 0.01:
                print("   ⚡ VERY FAST: Excellent local streaming")
            elif avg_interval < 0.05:
                print("   ✅ FAST: Good local streaming")
            elif avg_interval < 0.2:
                print("   ✅ MODERATE: Reasonable local generation")
            elif avg_interval < 0.5:
                print("   ⚠️  SLOW: Large model or limited resources")
            else:
                print("   🐌 VERY SLOW: Might need more powerful hardware")
            
            # Calculate local performance metrics
            if streaming_duration > 0 and len(full_response) > 0:
                chars_per_second = len(full_response) / streaming_duration
                estimated_tokens = len(full_response) / 4  # Rough estimate
                tokens_per_second = estimated_tokens / streaming_duration
                
                print(f"   📈 Local performance: {chars_per_second:.0f} chars/sec, ~{tokens_per_second:.0f} tokens/sec")
                
                if tokens_per_second > 50:
                    print("   🚀 EXCELLENT: High-performance local generation")
                elif tokens_per_second > 20:
                    print("   ✅ GOOD: Solid local performance")
                elif tokens_per_second > 10:
                    print("   ✅ ACCEPTABLE: Reasonable for local model")
                else:
                    print("   ⚠️  SLOW: Consider smaller model or better hardware")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print(f"\n🔍 Testing Ollama streaming=False with {model_used}...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        import traceback
        traceback.print_exc()

async def test_ollama_model_comparison():
    """Test different Ollama models if available."""
    print("\n\n=== Ollama Model Comparison ===")
    
    models_to_test = [
        "gemma3",
        "llama3", 
        "granite3.3",
        "mistral"
    ]
    
    prompt = "Write a haiku about artificial intelligence"
    messages = [{"role": "user", "content": prompt}]
    
    successful_tests = 0
    results = {}
    
    for model in models_to_test:
        print(f"\n🔍 Testing {model}...")
        
        try:
            client = get_client(provider="ollama", model=model)
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            total_chars = 0
            
            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                
                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    total_chars += len(chunk["response"])
                
                # Limit for comparison
                if chunk_count >= 15:
                    break
            
            total_time = time.time() - start_time
            
            results[model] = {
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "chunks": chunk_count,
                "chars": total_chars
            }
            
            successful_tests += 1
            print(f"   ✅ {model}: {chunk_count} chunks, first at {first_chunk_time:.3f}s, {total_chars} chars")
            
        except Exception as e:
            print(f"   ❌ {model}: Not available or error - {e}")
            results[model] = None
    
    # Compare results
    if successful_tests >= 2:
        print(f"\n📊 OLLAMA MODEL COMPARISON ({successful_tests} models tested):")
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        # Find fastest model
        fastest_model = min(valid_results.keys(), 
                          key=lambda m: valid_results[m]["first_chunk"])
        fastest_time = valid_results[fastest_model]["first_chunk"]
        
        print(f"   🚀 Fastest first chunk: {fastest_model} ({fastest_time:.3f}s)")
        
        # Find most responsive streaming
        best_streaming = min(valid_results.keys(),
                           key=lambda m: valid_results[m]["total_time"] / max(valid_results[m]["chunks"], 1))
        
        print(f"   ⚡ Best streaming: {best_streaming}")
        
        # Model recommendations
        print("\n   💡 Model Recommendations:")
        for model, data in valid_results.items():
            speed_rating = "🚀" if data["first_chunk"] < 2.0 else "✅" if data["first_chunk"] < 5.0 else "⚠️"
            chunk_rating = "📊" if data["chunks"] > 10 else "📉"
            print(f"   {speed_rating}{chunk_rating} {model}: {data['first_chunk']:.1f}s first chunk, {data['chunks']} chunks")
    
    elif successful_tests == 1:
        print(f"\n📊 Only one model available for testing")
    else:
        print(f"\n❌ No Ollama models available for testing")
        print("💡 Install models with: ollama pull <model-name>")

async def test_ollama_system_info():
    """Test Ollama system and provide diagnostic info."""
    print("\n\n=== Ollama System Diagnostics ===")
    
    try:
        # Try to get basic system info through a simple request
        client = get_client(provider="ollama", model="gemma3")  # Try default model
        
        print("✅ Ollama connection successful")
        
        # Test a very simple request to check responsiveness
        messages = [{"role": "user", "content": "Hi"}]
        
        start_time = time.time()
        response = await client.create_completion(messages, stream=False)
        response_time = time.time() - start_time
        
        if isinstance(response, dict) and response.get("response"):
            print(f"✅ Basic response test: {response_time:.3f}s")
            print(f"   Response: {response['response'][:50]}...")
        else:
            print(f"⚠️  Unexpected response format: {response}")
        
        # Test streaming responsiveness
        start_time = time.time()
        stream = client.create_completion(messages, stream=True)
        
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            if chunk_count >= 3:  # Just test first few chunks
                break
        
        stream_time = time.time() - start_time
        print(f"✅ Streaming test: {chunk_count} chunks in {stream_time:.3f}s")
        
        # Performance assessment
        if response_time < 1.0:
            print("🚀 Excellent local performance")
        elif response_time < 3.0:
            print("✅ Good local performance")
        else:
            print("⚠️  Slow performance - consider:")
            print("   - Smaller model")
            print("   - More RAM/CPU")
            print("   - GPU acceleration")
    
    except Exception as e:
        print(f"❌ Ollama system test failed: {e}")
        print("\n💡 Ollama troubleshooting:")
        print("   1. Is Ollama running? (ollama serve)")
        print("   2. Are models installed? (ollama list)")
        print("   3. Try: ollama pull gemma3")

async def main():
    """Run all Ollama streaming tests."""
    await test_ollama_streaming()
    await test_ollama_model_comparison()
    await test_ollama_system_info()

if __name__ == "__main__":
    asyncio.run(main())