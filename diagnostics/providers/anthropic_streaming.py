# diagnostics/anthropic_streaming.py
"""
Test Anthropic provider streaming behavior.
"""
import asyncio
import time
from chuk_llm.llm.llm_client import get_llm_client
from chuk_llm.llm.configuration.provider_config import ProviderConfig

async def test_anthropic_streaming():
    """Test Anthropic streaming behavior."""
    print("=== Testing Anthropic Provider Streaming ===")
    
    try:
        # Get Anthropic client
        client = get_llm_client(
            provider="anthropic",
            model="claude-3-7-sonnet-20250219",
            config=ProviderConfig()
        )
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it at least 100 words."}
        ]
        
        print("\n🔍 Testing Anthropic streaming=True...")
        start_time = time.time()
        
        # Test streaming - DON'T await it since it returns async generator
        response = client.create_completion(messages, stream=True)
        
        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")
        
        # Also need to fix the base interface - remove async from create_completion
        # since it returns async generator when streaming
        if hasattr(response, '__aiter__'):
            print("✅ Got async generator")
            
            chunk_count = 0
            first_chunk_time = None
            full_response = ""
            
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
                    chunk_text = chunk["response"] or ""
                    print(chunk_text, end="", flush=True)
                    full_response += chunk_text
                
                print(f"\n   Chunk {chunk_count}: {relative_time:.3f}s")
                print("   Continuing: ", end="", flush=True)
            
            end_time = time.time() - start_time
            
            print(f"\n\n📊 ANTHROPIC STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s")
            print(f"   Total time: {end_time:.3f}s")
            
            if chunk_count == 1:
                print("   ⚠️  FAKE STREAMING: Only one chunk (entire response at once)")
            else:
                print("   ✅ REAL STREAMING: Multiple chunks detected")
                
            print(f"   Response length: {len(full_response)} characters")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Anthropic streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print("✅ Non-streaming works correctly")
        
    except Exception as e:
        print(f"❌ Error testing Anthropic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_anthropic_streaming())