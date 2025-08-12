#!/usr/bin/env python3
"""
Deep debug of ChukLLM streaming to test both tool calls and text responses
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
import json

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_tool_response():
    """Test streaming with a prompt that triggers tool use"""
    print("\n" + "="*60)
    print("TEST 1: TOOL-TRIGGERING PROMPT")
    print("="*60)
    
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
    
    print(f"🎯 Model: {model_name}")
    print(f"📝 Prompt: {prompt}")
    print(f"🔧 Tools provided: {len(tools)}")
    
    client = OllamaLLMClient(model_name)
    
    print(f"\n🔄 Starting streaming...")
    start_time = time.time()
    
    try:
        completion = client.create_completion(
            messages=messages,
            tools=tools,
            stream=True,
            max_tokens=200,
            temperature=0.1
        )
        
        chunk_count = 0
        text_chunks = []
        tool_calls_received = []
        
        async for chunk in completion:
            chunk_count += 1
            
            # Extract content
            response_text = chunk.get("response", "")
            tool_calls = chunk.get("tool_calls", [])
            reasoning = chunk.get("reasoning", {})
            
            # Track what we received
            if response_text:
                text_chunks.append(response_text)
                print(f"  📄 Text chunk {chunk_count}: '{response_text[:50]}{'...' if len(response_text) > 50 else ''}'")
            
            if tool_calls:
                for tc in tool_calls:
                    tool_calls_received.append(tc)
                    func_name = tc.get("function", {}).get("name", "unknown")
                    func_args = tc.get("function", {}).get("arguments", "{}")
                    print(f"  🔧 Tool call: {func_name}")
                    try:
                        args_dict = json.loads(func_args)
                        print(f"     Arguments: {json.dumps(args_dict, indent=2)}")
                    except:
                        print(f"     Arguments (raw): {func_args}")
            
            if reasoning.get("thinking_content"):
                print(f"  🤔 Thinking: '{reasoning['thinking_content'][:50]}...'")
        
        end_time = time.time()
        
        print(f"\n📊 Results:")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Text content: {''.join(text_chunks) if text_chunks else 'None'}")
        print(f"  Tool calls: {len(tool_calls_received)}")
        print(f"  Response time: {end_time - start_time:.2f}s")
        
        if tool_calls_received:
            print(f"\n✅ SUCCESS: Model correctly used tools for SQL execution")
        else:
            print(f"\n⚠️  WARNING: Expected tool call but got text response")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_text_response():
    """Test streaming with a prompt that should generate text"""
    print("\n" + "="*60)
    print("TEST 2: TEXT-GENERATING PROMPT (NO TOOLS)")
    print("="*60)
    
    from chuk_llm.llm.providers.ollama_client import OllamaLLMClient
    
    model_name = "gpt-oss:latest"
    prompt = "Tell me a short joke about databases"
    messages = [{"role": "user", "content": prompt}]
    
    print(f"🎯 Model: {model_name}")
    print(f"📝 Prompt: {prompt}")
    print(f"🔧 Tools provided: None")
    
    client = OllamaLLMClient(model_name)
    
    print(f"\n🔄 Starting streaming...")
    start_time = time.time()
    
    try:
        # No tools this time - should generate text
        completion = client.create_completion(
            messages=messages,
            stream=True,
            max_tokens=200,
            temperature=0.7
        )
        
        chunk_count = 0
        text_chunks = []
        
        async for chunk in completion:
            chunk_count += 1
            
            response_text = chunk.get("response", "")
            reasoning = chunk.get("reasoning", {})
            
            if response_text:
                text_chunks.append(response_text)
                # Print each chunk as it arrives for visual streaming effect
                print(response_text, end="", flush=True)
            
            if reasoning.get("thinking_content"):
                # Don't print thinking inline, save for summary
                pass
        
        end_time = time.time()
        
        full_response = ''.join(text_chunks)
        
        print(f"\n\n📊 Results:")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Response length: {len(full_response)} chars")
        print(f"  Response time: {end_time - start_time:.2f}s")
        
        if full_response:
            print(f"\n✅ SUCCESS: Text streaming worked correctly")
        else:
            print(f"\n⚠️  WARNING: No text content received")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_thinking_response():
    """Test streaming with a prompt that might trigger thinking"""
    print("\n" + "="*60)
    print("TEST 3: REASONING/THINKING PROMPT")
    print("="*60)
    
    from chuk_llm.llm.providers.ollama_client import OllamaLLMClient
    
    model_name = "gpt-oss:latest"
    prompt = "Think step by step: If a database has 1000 users and doubles every month, how many users will it have after 6 months?"
    messages = [{"role": "user", "content": prompt}]
    
    print(f"🎯 Model: {model_name}")
    print(f"📝 Prompt: {prompt}")
    print(f"🔧 Testing reasoning capabilities...")
    
    client = OllamaLLMClient(model_name)
    
    print(f"\n🔄 Starting streaming...")
    start_time = time.time()
    
    try:
        # The model should provide reasoning even without explicit think parameter
        # as it's prompted with "Think step by step"
        completion = client.create_completion(
            messages=messages,
            stream=True,
            max_tokens=500,
            temperature=0.3
        )
        
        chunk_count = 0
        text_chunks = []
        thinking_chunks = []
        
        print("\n📝 Response:")
        async for chunk in completion:
            chunk_count += 1
            
            response_text = chunk.get("response", "")
            reasoning = chunk.get("reasoning", {})
            
            if response_text:
                text_chunks.append(response_text)
                print(response_text, end="", flush=True)
            
            if reasoning.get("thinking_content"):
                thinking_chunks.append(reasoning["thinking_content"])
        
        end_time = time.time()
        
        full_response = ''.join(text_chunks)
        full_thinking = ''.join(thinking_chunks)
        
        print(f"\n\n📊 Results:")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Response length: {len(full_response)} chars")
        print(f"  Thinking length: {len(full_thinking)} chars")
        print(f"  Response time: {end_time - start_time:.2f}s")
        
        if full_thinking:
            print(f"\n🤔 Thinking process detected ({len(full_thinking)} chars)")
            print(f"  First 200 chars: {full_thinking[:200]}...")
        
        if full_response:
            print(f"\n✅ SUCCESS: Reasoning model streaming worked")
        else:
            print(f"\n⚠️  WARNING: No response received")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_thinking_levels():
    """Test if thinking parameter is supported"""
    print("\n" + "="*60)
    print("TEST 4: THINKING PARAMETER CHECK")
    print("="*60)
    
    print("ℹ️  Note: The 'think' parameter may not be supported by AsyncClient")
    print("   Reasoning models like gpt-oss will still think when prompted appropriately")
    
    from chuk_llm.llm.providers.ollama_client import OllamaLLMClient
    import ollama
    
    # Check ollama version and capabilities
    print(f"\n📦 Ollama package version: {getattr(ollama, '__version__', 'unknown')}")
    
    # Test with sync client if available
    try:
        print("\n🔍 Testing sync client with think parameter...")
        sync_response = ollama.chat(
            model='gpt-oss:latest',
            messages=[{'role': 'user', 'content': 'What is 2+2?'}],
            think='low',
            stream=False
        )
        
        if hasattr(sync_response, 'message'):
            msg = sync_response.message
            if hasattr(msg, 'thinking') and msg.thinking:
                print(f"✅ Sync client supports thinking! Thinking content length: {len(msg.thinking)}")
            else:
                print("ℹ️  Sync client accepted think parameter but no thinking content returned")
                
    except TypeError as e:
        if "'think'" in str(e):
            print("❌ Sync client doesn't support 'think' parameter")
        else:
            print(f"❌ Error testing sync client: {e}")
    except Exception as e:
        print(f"❌ Error testing sync client: {e}")
    
    # Show how to use thinking without explicit parameter
    print("\n💡 Tip: Use prompts like 'Think step by step' to trigger reasoning")
    print("   The model will use its thinking capabilities automatically")

async def main():
    print("🔍 COMPREHENSIVE CHUK-LLM GPT-OSS STREAMING TESTS")
    print("=" * 60)
    
    try:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Suppress some debug logging for cleaner output
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("chuk_llm.llm.providers._config_mixin").setLevel(logging.WARNING)
        
        # Run all tests
        await test_tool_response()
        await test_text_response()
        await test_thinking_response()
        await test_thinking_levels()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())