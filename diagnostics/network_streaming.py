# diagnostics/network_streaming.py
"""
Test streaming at the network level
"""
import asyncio
import time
import os
import httpx
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
load_dotenv()

async def test_raw_http_streaming():
    """Test raw HTTP streaming to OpenAI to see if buffering is at network level."""
    print("=== Raw HTTP Streaming Test ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short poem about streaming"}
        ],
        "stream": True,
        "max_tokens": 200
    }
    
    print("🌐 Making raw HTTP request to OpenAI...")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST", 
                url, 
                headers=headers, 
                json=payload
            ) as response:
                
                print(f"⏱️  HTTP response status: {response.status_code}")
                print(f"⏱️  Response headers: {dict(response.headers)}")
                
                if response.status_code != 200:
                    content = await response.aread()
                    print(f"❌ Error response: {content}")
                    return
                
                chunk_count = 0
                first_chunk_time = None
                buffer = ""
                
                print("🔄 Reading raw HTTP stream...")
                print("Response: ", end="", flush=True)
                
                async for raw_chunk in response.aiter_bytes():
                    current_time = time.time()
                    relative_time = current_time - start_time
                    
                    if first_chunk_time is None:
                        first_chunk_time = relative_time
                        print(f"\n🎯 FIRST HTTP CHUNK at: {relative_time:.3f}s")
                        print("Response: ", end="", flush=True)
                    
                    chunk_count += 1
                    
                    # Show timing for first few chunks
                    if chunk_count <= 5:
                        print(f"\n   HTTP Chunk {chunk_count}: {relative_time:.3f}s, size: {len(raw_chunk)} bytes")
                        print("   Continuing: ", end="", flush=True)
                    
                    # Try to parse SSE data
                    try:
                        chunk_str = raw_chunk.decode('utf-8')
                        buffer += chunk_str
                        
                        # Process complete SSE events
                        while '\n\n' in buffer:
                            event, buffer = buffer.split('\n\n', 1)
                            
                            if event.startswith('data: '):
                                data = event[6:]  # Remove 'data: ' prefix
                                
                                if data.strip() == '[DONE]':
                                    continue
                                
                                try:
                                    import json
                                    parsed = json.loads(data)
                                    
                                    if 'choices' in parsed and parsed['choices']:
                                        delta = parsed['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            print(content, end="", flush=True)
                                
                                except json.JSONDecodeError:
                                    pass
                    
                    except UnicodeDecodeError:
                        pass
                
                end_time = time.time() - start_time
                print(f"\n\n📊 RAW HTTP ANALYSIS:")
                print(f"   Total HTTP chunks: {chunk_count}")
                print(f"   First chunk delay: {first_chunk_time:.3f}s")
                print(f"   Total time: {end_time:.3f}s")
                print(f"   Streaming duration: {end_time - first_chunk_time:.3f}s")
                
    except Exception as e:
        print(f"❌ Error with raw HTTP: {e}")
        import traceback
        traceback.print_exc()

async def test_httpx_streaming():
    """Test with httpx client to see streaming behavior."""
    print("\n=== HTTPX Client Streaming Test ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Count from 1 to 10 slowly, one number per line"}
        ],
        "stream": True,
        "max_tokens": 50
    }
    
    print("🔍 Testing HTTPX streaming...")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"⏱️  Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Error: {response.text}")
                return
            
            chunk_count = 0
            first_chunk_time = None
            
            print("Response: ", end="", flush=True)
            
            for line in response.iter_lines():
                current_time = time.time()
                relative_time = current_time - start_time
                
                if first_chunk_time is None and line.strip():
                    first_chunk_time = relative_time
                    print(f"\n🎯 FIRST LINE at: {relative_time:.3f}s")
                    print("Response: ", end="", flush=True)
                
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunk_count += 1
                    
                    try:
                        import json
                        data = line[6:]  # Remove 'data: ' prefix
                        parsed = json.loads(data)
                        
                        if 'choices' in parsed and parsed['choices']:
                            delta = parsed['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                print(content, end="", flush=True)
                    
                    except json.JSONDecodeError:
                        pass
            
            end_time = time.time() - start_time
            print(f"\n\n📊 HTTPX ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(f"   First chunk delay: {first_chunk_time:.3f}s")
            print(f"   Total time: {end_time:.3f}s")
            
    except Exception as e:
        print(f"❌ Error with HTTPX: {e}")

if __name__ == "__main__":
    asyncio.run(test_raw_http_streaming())
    asyncio.run(test_httpx_streaming())