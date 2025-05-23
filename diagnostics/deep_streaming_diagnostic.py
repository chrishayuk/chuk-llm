# diagnostics/deep_streaming_diagnostic.py
"""
Deep diagnostic to find where streaming is getting buffered.
"""
import asyncio
import time
import logging
from chuk_llm.llm.llm_client import get_llm_client
from chuk_llm.llm.provider_config import ProviderConfig

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def test_llm_client_directly():
    """Test the LLM client directly to see if it's the source of buffering."""
    print("=== Testing LLM Client Directly ===")
    
    # Create client directly (same as ChukAgent does)
    client = get_llm_client(
        provider="openai",
        model="gpt-4o-mini",
        config=ProviderConfig()
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short story about a cat"}
    ]
    
    print("🔍 Testing direct LLM client streaming...")
    start_time = time.time()
    
    try:
        # Call the client directly with streaming
        response_stream = await client.create_completion(
            messages=messages,
            stream=True
        )
        
        print(f"⏱️  Response stream type: {type(response_stream)}")
        print(f"⏱️  Has __aiter__: {hasattr(response_stream, '__aiter__')}")
        
        chunk_count = 0
        first_chunk_time = None
        last_chunk_time = start_time
        
        print("🔄 Iterating through LLM client stream...")
        print("Response: ", end="", flush=True)
        
        async for chunk in response_stream:
            current_time = time.time()
            relative_time = current_time - start_time
            
            if first_chunk_time is None:
                first_chunk_time = relative_time
                print(f"\n🎯 FIRST CHUNK from LLM at: {relative_time:.3f}s")
                print("Response: ", end="", flush=True)
            
            chunk_count += 1
            
            # Show chunk details
            print(f"[{chunk_count}]", end="", flush=True)
            if isinstance(chunk, dict) and "response" in chunk:
                chunk_text = chunk["response"]
                print(chunk_text, end="", flush=True)
            
            # Time between chunks
            interval = current_time - last_chunk_time
            if chunk_count <= 5 or chunk_count % 100 == 0:
                print(f"\n   Chunk {chunk_count}: {relative_time:.3f}s (interval: {interval:.4f}s)")
                print("   Continuing: ", end="", flush=True)
            
            last_chunk_time = current_time
        
        end_time = time.time() - start_time
        print(f"\n\n📊 LLM CLIENT ANALYSIS:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   First chunk delay: {first_chunk_time:.3f}s")
        print(f"   Total time: {end_time:.3f}s")
        print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
        
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
            {"role": "user", "content": "Tell me a very short story"}
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
        
        print("Response: ", end="", flush=True)
        
        async for chunk in stream:
            current_time = time.time()
            relative_time = current_time - start_time
            
            if first_chunk_time is None:
                first_chunk_time = relative_time
                print(f"\n🎯 FIRST CHUNK from raw OpenAI at: {relative_time:.3f}s")
                print("Response: ", end="", flush=True)
            
            chunk_count += 1
            
            # Extract content from OpenAI chunk
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
        
        end_time = time.time() - start_time
        print(f"\n\n📊 RAW OPENAI ANALYSIS:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   First chunk delay: {first_chunk_time:.3f}s")
        print(f"   Total time: {end_time:.3f}s")
        
    except ImportError:
        print("❌ OpenAI library not available for baseline test")
    except Exception as e:
        print(f"❌ Error with raw OpenAI: {e}")

async def main():
    """Run all diagnostic tests."""
    await test_llm_client_directly()
    await test_raw_openai()

if __name__ == "__main__":
    asyncio.run(main())