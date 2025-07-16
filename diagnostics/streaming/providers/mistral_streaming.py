# diagnostics/streaming/providers/mistral_streaming.py
"""
Test Mistral provider streaming behavior.
Updated to work with the new chuk-llm architecture and correct model names.
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

async def test_mistral_streaming():
    """Test Mistral streaming behavior."""
    print("=== Testing Mistral Provider Streaming ===")
    
    try:
        # Get Mistral client with correct model name from config
        client = get_client(
            provider="mistral",
            model="mistral-medium-2505"  # Use actual model from config
        )
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Provider name: {getattr(client, 'provider_name', 'unknown')}")
        
        messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it at least 100 words and tell it slowly."}
        ]
        
        print("\n🔍 Testing Mistral streaming=True...")
        start_time = time.time()
        
        # Test streaming - DON'T await it since it returns async generator
        response = client.create_completion(messages, stream=True)
        
        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")
        
        if hasattr(response, '__aiter__'):
            print("✅ Got async generator")
            
            chunk_count = 0
            first_chunk_time = None
            last_chunk_time = start_time
            full_response = ""
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
                
                if isinstance(chunk, dict) and "response" in chunk:
                    chunk_text = chunk["response"] or ""
                    print(chunk_text, end="", flush=True)
                    full_response += chunk_text
                elif isinstance(chunk, dict) and chunk.get("error"):
                    print(f"\n❌ Error chunk: {chunk}")
                    break
                
                # Show timing for first few chunks and every 5th chunk
                if chunk_count <= 5 or chunk_count % 5 == 0:
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
            
            print(f"\n\n📊 MISTRAL STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s" if first_chunk_time else "   No chunks received")
            print(f"   Total time: {end_time:.3f}s")
            print(f"   Streaming duration: {end_time - (first_chunk_time or end_time):.3f}s")
            print(f"   Response length: {len(full_response)} characters")
            print(f"   Avg chunk interval: {avg_interval*1000:.1f}ms" if chunk_intervals else "   No intervals")
            print(f"   Min interval: {min_interval*1000:.1f}ms" if chunk_intervals else "   No intervals")
            print(f"   Max interval: {max_interval*1000:.1f}ms" if chunk_intervals else "   No intervals")
            
            # Quality assessment
            if chunk_count == 0:
                print("   ❌ NO STREAMING: No chunks received")
            elif chunk_count == 1:
                print("   ⚠️  FAKE STREAMING: Only one chunk (entire response at once)")
            elif chunk_count < 5:
                print("   ⚠️  LIMITED STREAMING: Very few chunks")
            else:
                print("   ✅ REAL STREAMING: Multiple chunks detected")
            
            if first_chunk_time:
                if first_chunk_time < 1.5:
                    print("   ✅ FAST: Excellent first chunk time")
                elif first_chunk_time < 3.0:
                    print("   ✅ GOOD: Acceptable first chunk time")
                else:
                    print("   ⚠️  SLOW: First chunk could be faster")
            
            streaming_duration = end_time - (first_chunk_time or end_time)
            if streaming_duration > 2.0:
                print("   ✅ TRUE STREAMING: Long streaming duration")
            elif streaming_duration > 0.5:
                print("   ✅ PARTIAL STREAMING: Some streaming detected")
            else:
                print("   ⚠️  BUFFERED: Very short streaming duration")
            
            if avg_interval > 0.05:  # More than 50ms
                print("   ✅ NATURAL PACING: Realistic chunk intervals")
            elif avg_interval > 0.01:  # More than 10ms
                print("   ✅ MODERATE PACING: Reasonable chunk intervals")
            elif avg_interval > 0:
                print("   ⚠️  RAPID DELIVERY: Chunks arriving very quickly")
            else:
                print("   ⚠️  NO INTERVALS: Cannot assess pacing")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Mistral streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Mistral: {e}")
        import traceback
        traceback.print_exc()

async def test_mistral_model_capabilities():
    """Test different Mistral models and their capabilities."""
    print("\n\n=== Testing Mistral Model Capabilities ===")
    
    # Use actual model names from the config
    models_to_test = [
        ("mistral-medium-2505", "General purpose medium model"),
        ("magistral-medium-2506", "Reasoning-capable medium model"),
        ("magistral-small-2506", "Reasoning-capable small model"),
        ("codestral-2501", "Code-specialized model"),
        ("pixtral-large-2411", "Vision-capable model")
    ]
    
    for model, description in models_to_test:
        print(f"\n🔍 Testing {model} ({description})...")
        
        try:
            client = get_client(provider="mistral", model=model)
            
            # Get model info
            try:
                model_info = client.get_model_info()
                print(f"   Model info: {model_info}")
            except:
                print(f"   Model info: Not available")
            
            # Test basic streaming
            messages = [{"role": "user", "content": "Say hello in exactly 10 words."}]
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            
            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                chunk_count += 1
                
                # Limit chunks for testing
                if chunk_count >= 10:
                    break
            
            print(f"   Streaming: {chunk_count} chunks, first at {first_chunk_time:.3f}s" if first_chunk_time else "   Streaming: Failed")
            
        except Exception as e:
            print(f"   ❌ Error with {model}: {e}")

async def test_mistral_vs_competitors():
    """Compare Mistral vs other providers streaming behavior."""
    print("\n\n=== Mistral vs Competitors Streaming Comparison ===")
    
    # Same prompt for all
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence"}
    ]
    
    providers = [
        ("mistral", "mistral-medium-2505"),  # Use correct model name
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o-mini"),
        ("groq", "llama-3.3-70b-versatile")
    ]
    
    results = {}
    
    for provider, model in providers:
        print(f"\n🔍 Testing {provider} with {model}...")
        
        try:
            client = get_client(provider=provider, model=model)
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            content_length = 0
            chunk_times = []
            
            async for chunk in response:
                current_time = time.time() - start_time
                
                if first_chunk_time is None:
                    first_chunk_time = current_time
                
                chunk_count += 1
                chunk_times.append(current_time)
                
                if isinstance(chunk, dict) and chunk.get("response"):
                    content_length += len(chunk["response"])
                
                # Limit chunks for comparison
                if chunk_count >= 20:
                    break
            
            total_time = time.time() - start_time
            
            results[provider] = {
                "chunks": chunk_count,
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "content_length": content_length,
                "chunk_times": chunk_times
            }
            
            print(f"   {provider}: {chunk_count} chunks, first at {first_chunk_time:.3f}s, {content_length} chars, total {total_time:.3f}s")
            
        except Exception as e:
            print(f"   {provider}: Error - {e}")
            results[provider] = None
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        # Compare first chunk times
        fastest_first = min(valid_results.keys(), key=lambda k: valid_results[k]["first_chunk"])
        print(f"   🚀 Fastest first chunk: {fastest_first} ({valid_results[fastest_first]['first_chunk']:.3f}s)")
        
        # Compare chunk granularity
        most_chunks = max(valid_results.keys(), key=lambda k: valid_results[k]["chunks"])
        print(f"   📊 Most granular streaming: {most_chunks} ({valid_results[most_chunks]['chunks']} chunks)")
        
        # Compare total speed
        fastest_total = min(valid_results.keys(), key=lambda k: valid_results[k]["total_time"])
        print(f"   ⚡ Fastest total time: {fastest_total} ({valid_results[fastest_total]['total_time']:.3f}s)")
        
        # Mistral-specific analysis
        if "mistral" in valid_results:
            mistral_result = valid_results["mistral"]
            print(f"\n🎯 MISTRAL ANALYSIS:")
            print(f"   Chunks: {mistral_result['chunks']}")
            print(f"   First chunk: {mistral_result['first_chunk']:.3f}s")
            print(f"   Content: {mistral_result['content_length']} chars")
            
            # Compare to others
            other_providers = [k for k in valid_results.keys() if k != "mistral"]
            if other_providers:
                avg_first_chunk = sum(valid_results[p]["first_chunk"] for p in other_providers) / len(other_providers)
                avg_chunks = sum(valid_results[p]["chunks"] for p in other_providers) / len(other_providers)
                
                if mistral_result["first_chunk"] < avg_first_chunk:
                    print(f"   ✅ Mistral faster than average first chunk by {(avg_first_chunk - mistral_result['first_chunk'])*1000:.0f}ms")
                else:
                    print(f"   ⚠️  Mistral slower than average first chunk by {(mistral_result['first_chunk'] - avg_first_chunk)*1000:.0f}ms")
                
                if mistral_result["chunks"] > avg_chunks:
                    print(f"   ✅ Mistral more granular than average ({mistral_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
                else:
                    print(f"   ⚠️  Mistral less granular than average ({mistral_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
    
    else:
        print("   ❌ Could not compare - insufficient successful tests")

async def test_mistral_features():
    """Test Mistral-specific features like function calling and vision."""
    print("\n\n=== Testing Mistral Advanced Features ===")
    
    # Test function calling with a model that supports tools
    print("\n🔍 Testing Mistral function calling...")
    try:
        client = get_client(provider="mistral", model="mistral-medium-2505")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        messages = [{"role": "user", "content": "What's the weather like in Paris?"}]
        
        response = await client.create_completion(messages, tools=tools, stream=False)
        
        if isinstance(response, dict) and response.get("tool_calls"):
            print(f"   ✅ Function calling works: {len(response['tool_calls'])} tool calls")
            for tool_call in response["tool_calls"]:
                print(f"      Tool: {tool_call.get('function', {}).get('name')}")
        elif isinstance(response, dict) and response.get("response"):
            print(f"   ⚠️  Got text response instead of tool call: {response['response'][:100]}...")
        else:
            print(f"   ❌ Unexpected response format: {type(response)}")
        
    except Exception as e:
        print(f"   ❌ Function calling test failed: {e}")
    
    # Test vision with correct model name
    print("\n🔍 Testing Mistral vision capabilities...")
    try:
        # Use vision-capable model from config
        client = get_client(provider="mistral", model="pixtral-large-2411")
        
        # Test with a simple base64 image (1x1 red pixel)
        red_pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{red_pixel_b64}"
                        }
                    }
                ]
            }
        ]
        
        response = await client.create_completion(messages, stream=False)
        
        if isinstance(response, dict) and response.get("response"):
            print(f"   ✅ Vision works: {response['response'][:100]}...")
        else:
            print(f"   ❌ Vision test failed: {response}")
        
    except Exception as e:
        print(f"   ⚠️  Vision test failed (may not be supported): {e}")

    # Test reasoning models
    print("\n🔍 Testing Mistral reasoning capabilities...")
    try:
        client = get_client(provider="mistral", model="magistral-medium-2506")
        
        messages = [
            {"role": "user", "content": "Think step by step: If a train leaves station A at 2 PM traveling at 60 mph, and another train leaves station B at 3 PM traveling at 80 mph toward station A, and the stations are 200 miles apart, when will they meet?"}
        ]
        
        response = await client.create_completion(messages, stream=False)
        
        if isinstance(response, dict) and response.get("response"):
            content = response["response"]
            print(f"   ✅ Reasoning response received ({len(content)} chars)")
            # Check if the response shows step-by-step thinking
            if any(phrase in content.lower() for phrase in ["step", "first", "then", "calculate", "therefore"]):
                print(f"   ✅ Shows reasoning structure")
            else:
                print(f"   ⚠️  May not show explicit reasoning")
        else:
            print(f"   ❌ Reasoning test failed: {response}")
        
    except Exception as e:
        print(f"   ❌ Reasoning test failed: {e}")

async def main():
    """Run all Mistral streaming tests."""
    await test_mistral_streaming()
    await test_mistral_model_capabilities()
    await test_mistral_vs_competitors()
    await test_mistral_features()

if __name__ == "__main__":
    asyncio.run(main())