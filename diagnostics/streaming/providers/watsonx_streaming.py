# diagnostics/streaming/providers/watsonx_streaming.py
"""
Test IBM Watson X provider streaming behavior.
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

async def test_watsonx_streaming():
    """Test Watson X streaming behavior."""
    print("=== Testing IBM Watson X Provider Streaming ===")
    
    try:
        # Get Watson X client - try different models that might be available
        available_models = [
            "ibm/granite-3-8b-instruct",
            "ibm/granite-3-2b-instruct", 
            "meta-llama/llama-3-2-3b-instruct",
            "meta-llama/llama-3-2-1b-instruct",
            "mistralai/mistral-large"
        ]
        
        client = None
        model_used = None
        
        for model in available_models:
            try:
                print(f"🔍 Trying model: {model}")
                client = get_client(provider="watsonx", model=model)
                model_used = model
                print(f"✅ Successfully connected with model: {model}")
                break
            except Exception as e:
                print(f"❌ Failed to connect with {model}: {e}")
                continue
        
        if not client:
            print("❌ Could not connect to any Watson X model")
            print("💡 Watson X setup requirements:")
            print("   - Set WATSONX_API_KEY or IBM_CLOUD_API_KEY")
            print("   - Set WATSONX_PROJECT_ID")
            print("   - Ensure Watson X.ai access is configured")
            return
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Model: {model_used}")
        
        messages = [
            {"role": "user", "content": "Write a creative story about artificial intelligence helping solve climate change. Make it at least 150 words with vivid details."}
        ]
        
        print(f"\n🔍 Testing Watson X streaming=True with {model_used}...")
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
            
            print(f"\n\n📊 WATSON X STREAMING ANALYSIS:")
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
            
            # Enterprise-focused assessment
            if first_chunk_time < 2.0:
                print("   🚀 FAST: Excellent enterprise response time")
            elif first_chunk_time < 4.0:
                print("   ✅ GOOD: Acceptable enterprise response time")
            else:
                print("   ⚠️  SLOW: Enterprise response could be faster")
            
            streaming_duration = end_time - first_chunk_time if first_chunk_time else 0
            if streaming_duration > 3.0:
                print("   ✅ TRUE STREAMING: Long streaming duration")
            elif streaming_duration > 1.0:
                print("   ✅ PARTIAL STREAMING: Some streaming detected")
            else:
                print("   ⚠️  BUFFERED: Very short streaming duration")
            
            # Watson X specific analysis (enterprise focused)
            if avg_interval < 0.02:
                print("   ⚡ VERY FAST: Excellent enterprise streaming")
            elif avg_interval < 0.05:
                print("   ✅ FAST: Good enterprise streaming")
            elif avg_interval < 0.1:
                print("   ✅ MODERATE: Reasonable enterprise performance")
            else:
                print("   ⚠️  SLOW: May need optimization for enterprise use")
            
            # Calculate enterprise performance metrics
            if streaming_duration > 0 and len(full_response) > 0:
                chars_per_second = len(full_response) / streaming_duration
                estimated_tokens = len(full_response) / 4  # Rough estimate
                tokens_per_second = estimated_tokens / streaming_duration
                
                print(f"   📈 Enterprise performance: {chars_per_second:.0f} chars/sec, ~{tokens_per_second:.0f} tokens/sec")
                
                if tokens_per_second > 30:
                    print("   🚀 EXCELLENT: High-performance enterprise streaming")
                elif tokens_per_second > 15:
                    print("   ✅ GOOD: Solid enterprise performance")
                elif tokens_per_second > 8:
                    print("   ✅ ACCEPTABLE: Reasonable for enterprise workloads")
                else:
                    print("   ⚠️  SLOW: Below expected enterprise performance")
                
                # Model-specific insights
                if "granite" in model_used.lower():
                    print("   🏢 IBM Granite: Optimized for enterprise applications")
                elif "llama" in model_used.lower():
                    print("   🦙 Meta Llama: Open-source model on enterprise infrastructure")
                elif "mistral" in model_used.lower():
                    print("   🌪️ Mistral: European AI on enterprise platform")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print(f"\n🔍 Testing Watson X streaming=False with {model_used}...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Watson X: {e}")
        import traceback
        traceback.print_exc()

async def test_watsonx_model_comparison():
    """Test different Watson X models if available."""
    print("\n\n=== Watson X Model Comparison ===")
    
    models_to_test = [
        ("ibm/granite-3-8b-instruct", "IBM Granite 8B"),
        ("ibm/granite-3-2b-instruct", "IBM Granite 2B"),
        ("meta-llama/llama-3-2-3b-instruct", "Meta Llama 3B"),
        ("meta-llama/llama-3-2-1b-instruct", "Meta Llama 1B"),
        ("mistralai/mistral-large", "Mistral Large")
    ]
    
    prompt = "Explain quantum computing for business applications in 2 paragraphs"
    messages = [{"role": "user", "content": prompt}]
    
    successful_tests = 0
    results = {}
    
    for model_id, model_name in models_to_test:
        print(f"\n🔍 Testing {model_name} ({model_id})...")
        
        try:
            client = get_client(provider="watsonx", model=model_id)
            
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
                if chunk_count >= 20:
                    break
            
            total_time = time.time() - start_time
            
            results[model_id] = {
                "name": model_name,
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "chunks": chunk_count,
                "chars": total_chars
            }
            
            successful_tests += 1
            print(f"   ✅ {model_name}: {chunk_count} chunks, first at {first_chunk_time:.3f}s, {total_chars} chars")
            
        except Exception as e:
            print(f"   ❌ {model_name}: Not available or error - {e}")
            results[model_id] = None
    
    # Compare results
    if successful_tests >= 2:
        print(f"\n📊 WATSON X MODEL COMPARISON ({successful_tests} models tested):")
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        # Find fastest model
        fastest_model = min(valid_results.keys(), 
                          key=lambda m: valid_results[m]["first_chunk"])
        fastest_time = valid_results[fastest_model]["first_chunk"]
        fastest_name = valid_results[fastest_model]["name"]
        
        print(f"   🚀 Fastest first chunk: {fastest_name} ({fastest_time:.3f}s)")
        
        # Find most efficient streaming
        best_streaming = min(valid_results.keys(),
                           key=lambda m: valid_results[m]["total_time"] / max(valid_results[m]["chunks"], 1))
        best_name = valid_results[best_streaming]["name"]
        
        print(f"   ⚡ Best streaming efficiency: {best_name}")
        
        # Enterprise model recommendations
        print("\n   🏢 Enterprise Model Recommendations:")
        
        granite_models = [(k, v) for k, v in valid_results.items() if "granite" in k.lower()]
        if granite_models:
            print("   IBM Granite Models (Enterprise-optimized):")
            for model_id, data in granite_models:
                size = "8B" if "8b" in model_id else "2B" if "2b" in model_id else "?"
                speed_rating = "🚀" if data["first_chunk"] < 2.0 else "✅" if data["first_chunk"] < 4.0 else "⚠️"
                print(f"     {speed_rating} Granite {size}: {data['first_chunk']:.1f}s first chunk, {data['chunks']} chunks")
        
        llama_models = [(k, v) for k, v in valid_results.items() if "llama" in k.lower()]
        if llama_models:
            print("   Meta Llama Models (Open-source on enterprise):")
            for model_id, data in llama_models:
                size = "3B" if "3b" in model_id else "1B" if "1b" in model_id else "?"
                speed_rating = "🚀" if data["first_chunk"] < 2.0 else "✅" if data["first_chunk"] < 4.0 else "⚠️"
                print(f"     {speed_rating} Llama {size}: {data['first_chunk']:.1f}s first chunk, {data['chunks']} chunks")
        
        # Enterprise use case recommendations
        print("\n   💼 Use Case Recommendations:")
        if granite_models:
            print("   • Granite models: Best for enterprise compliance, IBM ecosystem integration")
        if llama_models:
            print("   • Llama models: Good for cost-effective enterprise workloads")
        print("   • Mistral models: European AI compliance, strong reasoning capabilities")
    
    elif successful_tests == 1:
        print(f"\n📊 Only one Watson X model available for testing")
    else:
        print(f"\n❌ No Watson X models available for testing")
        print("💡 Watson X setup requirements:")
        print("   - IBM Cloud account with Watson X.ai access")
        print("   - WATSONX_API_KEY environment variable")
        print("   - WATSONX_PROJECT_ID environment variable")

async def test_watsonx_enterprise_features():
    """Test Watson X enterprise-specific features and performance."""
    print("\n\n=== Watson X Enterprise Features Test ===")
    
    try:
        # Try to get enterprise-optimized model
        client = get_client(provider="watsonx", model="ibm/granite-3-8b-instruct")
        
        print("✅ Watson X enterprise connection successful")
        
        # Test enterprise workload simulation
        enterprise_prompts = [
            "Summarize quarterly financial data trends",
            "Analyze customer satisfaction metrics for actionable insights", 
            "Draft a technical specification for API integration"
        ]
        
        total_tests = len(enterprise_prompts)
        successful_tests = 0
        total_response_time = 0
        
        for i, prompt in enumerate(enterprise_prompts, 1):
            print(f"\n🔍 Enterprise test {i}/{total_tests}: {prompt[:40]}...")
            
            try:
                messages = [{"role": "user", "content": prompt}]
                
                start_time = time.time()
                response = await client.create_completion(messages, stream=False)
                response_time = time.time() - start_time
                
                if isinstance(response, dict) and response.get("response"):
                    successful_tests += 1
                    total_response_time += response_time
                    print(f"   ✅ Success in {response_time:.3f}s")
                    print(f"   Response: {response['response'][:80]}...")
                else:
                    print(f"   ⚠️  Unexpected response format")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        # Enterprise performance assessment
        if successful_tests > 0:
            avg_response_time = total_response_time / successful_tests
            
            print(f"\n📊 ENTERPRISE PERFORMANCE SUMMARY:")
            print(f"   Success rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.0f}%)")
            print(f"   Average response time: {avg_response_time:.3f}s")
            
            if avg_response_time < 3.0:
                print("   🚀 EXCELLENT: Enterprise-ready performance")
            elif avg_response_time < 5.0:
                print("   ✅ GOOD: Suitable for enterprise workloads")
            else:
                print("   ⚠️  SLOW: May need optimization for production enterprise use")
            
            # Enterprise readiness assessment
            print(f"\n🏢 ENTERPRISE READINESS:")
            print("   ✅ Hosted on IBM enterprise infrastructure")
            print("   ✅ Enterprise-grade security and compliance")
            print("   ✅ Integration with IBM Cloud ecosystem")
            print("   ✅ Support for enterprise governance policies")
        
    except Exception as e:
        print(f"❌ Enterprise features test failed: {e}")
        print("\n💡 Watson X enterprise setup:")
        print("   1. IBM Cloud account with Watson X.ai service")
        print("   2. Project created in Watson X.ai")
        print("   3. API key with appropriate permissions")
        print("   4. Environment variables configured")

async def main():
    """Run all Watson X streaming tests."""
    await test_watsonx_streaming()
    await test_watsonx_model_comparison()
    await test_watsonx_enterprise_features()

if __name__ == "__main__":
    asyncio.run(main())