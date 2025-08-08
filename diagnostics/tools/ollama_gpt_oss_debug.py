#!/usr/bin/env python3
"""
Deep debug of ChukLLM streaming to find exactly where it's failing
"""

import asyncio
import sys
import time
import logging
from pathlib import Path

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def deep_debug_streaming():
    print("🔍 DEEP DEBUG CHUK-LLM GPT-OSS STREAMING")
    print("=" * 60)
    
    try:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Import and check the full streaming path
        from chuk_llm.llm.providers.ollama_client import OllamaLLMClient
        
        model_name = "gpt-oss:latest"
        
        tools = [{
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query to execute"}
                    },
                    "required": ["query"]
                }
            }
        }]
        
        prompt = "Execute this SQL: SELECT name, email FROM users LIMIT 5"
        messages = [{"role": "user", "content": prompt}]
        
        print(f"🎯 Testing direct OllamaLLMClient with {model_name}")
        print(f"📝 Messages: {messages}")
        print(f"🔧 Tools: {len(tools)} provided")
        
        # Create client and test
        client = OllamaLLMClient(model_name)
        
        # Check client setup
        print(f"\n📊 Client Info:")
        print(f"  Model: {client.model}")
        print(f"  API Base: {client.api_base}")
        print(f"  Is Reasoning: {client._is_reasoning_model()}")
        print(f"  Supports Tools: {client.supports_feature('tools')}")
        print(f"  Supports Streaming: {client.supports_feature('streaming')}")
        
        # Test create_completion method directly
        print(f"\n🔄 Testing create_completion with stream=True...")
        
        try:
            start_time = time.time()
            completion = client.create_completion(
                messages=messages,
                tools=tools,
                stream=True,
                max_tokens=200,
                temperature=0.1
            )
            
            print(f"✅ create_completion returned: {type(completion)}")
            
            # Check if it's an async generator
            if hasattr(completion, '__aiter__'):
                print("✅ Returned object is async iterable")
                
                chunk_count = 0
                print("\n📥 Iterating through chunks:")
                
                try:
                    async for chunk in completion:
                        chunk_count += 1
                        print(f"  Chunk {chunk_count}: {repr(chunk)}")
                        
                        # Don't spam too many chunks
                        if chunk_count >= 20:
                            print("  ... (truncating for readability)")
                            break
                            
                except Exception as iter_error:
                    print(f"❌ Error during iteration: {iter_error}")
                    import traceback
                    traceback.print_exc()
                
                end_time = time.time()
                
                print(f"\n📊 Direct Streaming Summary:")
                print(f"  Total chunks: {chunk_count}")
                print(f"  Response time: {end_time - start_time:.2f}s")
                
                if chunk_count == 0:
                    print("❌ STILL NO CHUNKS - Issue is in OllamaLLMClient.create_completion")
                else:
                    print("✅ Chunks detected! Issue was in higher-level wrapper")
            else:
                print(f"❌ Returned object is not async iterable: {type(completion)}")
                
        except Exception as completion_error:
            print(f"❌ Error in create_completion: {completion_error}")
            import traceback
            traceback.print_exc()
        
        # Test the internal streaming method directly
        print(f"\n🔧 Testing _stream_completion_async directly...")
        
        try:
            start_time = time.time()
            
            # Call the internal streaming method
            stream_gen = client._stream_completion_async(
                messages=messages,
                tools=tools,
                max_tokens=200,
                temperature=0.1
            )
            
            print(f"✅ _stream_completion_async returned: {type(stream_gen)}")
            
            chunk_count = 0
            
            async for chunk in stream_gen:
                chunk_count += 1
                print(f"  Internal Chunk {chunk_count}: {repr(chunk)}")
                
                if chunk_count >= 20:
                    print("  ... (truncating)")
                    break
            
            end_time = time.time()
            
            print(f"\n📊 Internal Streaming Summary:")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Response time: {end_time - start_time:.2f}s")
            
            if chunk_count == 0:
                print("❌ NO CHUNKS FROM INTERNAL METHOD - Issue is in _stream_completion_async")
            else:
                print("✅ Internal method works! Issue is in create_completion wrapper")
                
        except Exception as internal_error:
            print(f"❌ Error in _stream_completion_async: {internal_error}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error in deep debug: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await deep_debug_streaming()

if __name__ == "__main__":
    asyncio.run(main())