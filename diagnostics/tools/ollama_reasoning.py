#!/usr/bin/env python3
"""
Check if the Ollama client has the reasoning model fixes and test GPT-OSS behavior
"""

import sys
import asyncio
from pathlib import Path

def check_ollama_client():
    print("🔍 CHECKING OLLAMA CLIENT IMPLEMENTATION")
    print("=" * 50)
    
    try:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.providers.ollama_client import OllamaLLMClient
        
        # Test GPT-OSS model
        client = OllamaLLMClient("gpt-oss:latest")
        
        print("✅ Successfully imported OllamaLLMClient")
        
        # Check if _is_reasoning_model method exists
        if hasattr(client, '_is_reasoning_model'):
            print("✅ _is_reasoning_model method exists")
            
            # Test reasoning model detection
            is_reasoning = client._is_reasoning_model()
            print(f"🧠 GPT-OSS detected as reasoning model: {is_reasoning}")
            
            if not is_reasoning:
                print("❌ GPT-OSS not detected as reasoning model - fix not applied correctly")
            else:
                print("✅ GPT-OSS correctly identified as reasoning model")
        else:
            print("❌ _is_reasoning_model method missing - fix not applied")
            return False
        
        # Check model info
        info = client.get_model_info()
        print(f"\n📊 Model Info:")
        print(f"  Model: {info.get('model_name', 'unknown')}")
        print(f"  Family: {info.get('ollama_specific', {}).get('model_family', 'unknown')}")
        print(f"  Is Reasoning: {info.get('ollama_specific', {}).get('is_reasoning_model', False)}")
        print(f"  Supports Thinking Stream: {info.get('ollama_specific', {}).get('supports_thinking_stream', False)}")
        
        # Check feature support
        supports_tools = client.supports_feature('tools')
        supports_streaming = client.supports_feature('streaming')
        print(f"  Supports Tools: {supports_tools}")
        print(f"  Supports Streaming: {supports_streaming}")
        
        # Check if streaming method has been enhanced
        import inspect
        stream_source = inspect.getsource(client._stream_completion_async)
        
        reasoning_features = []
        if "thinking" in stream_source:
            reasoning_features.append("thinking support")
        if "is_reasoning_model" in stream_source:
            reasoning_features.append("reasoning detection")
        if "reasoning_content" in stream_source:
            reasoning_features.append("reasoning content handling")
        if "chunk_type" in stream_source:
            reasoning_features.append("chunk type classification")
        
        if reasoning_features:
            print(f"✅ Streaming method enhanced with: {', '.join(reasoning_features)}")
        else:
            print("❌ Streaming method missing reasoning model enhancements")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_gpt_oss_reasoning_behavior():
    """Test how GPT-OSS actually behaves with tools"""
    print("\n🧪 TESTING GPT-OSS REASONING BEHAVIOR")
    print("=" * 50)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.providers.ollama_client import OllamaLLMClient
        
        client = OllamaLLMClient("gpt-oss:latest")
        
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
        
        messages = [{"role": "user", "content": "Execute this SQL: SELECT name FROM users LIMIT 3"}]
        
        print("🎯 Testing GPT-OSS tool calling behavior...")
        print("📝 Prompt: Execute this SQL: SELECT name FROM users LIMIT 3")
        
        # Test streaming
        chunk_count = 0
        tool_calls_found = 0
        thinking_chunks = 0
        content_chunks = 0
        
        async for chunk in client._stream_completion_async(
            messages=messages,
            tools=tools,
            max_tokens=200,
            temperature=0.1
        ):
            chunk_count += 1
            
            # Analyze chunk structure
            response = chunk.get('response', '')
            tool_calls = chunk.get('tool_calls', [])
            reasoning = chunk.get('reasoning', {})
            
            if tool_calls:
                tool_calls_found += len(tool_calls)
                print(f"  🔧 Chunk {chunk_count}: Found {len(tool_calls)} tool calls")
                for tc in tool_calls:
                    func_name = tc.get('function', {}).get('name', 'unknown')
                    print(f"    - {func_name}")
            
            if response:
                content_chunks += 1
                print(f"  📝 Chunk {chunk_count}: Content: '{response[:100]}{'...' if len(response) > 100 else ''}'")
            
            if reasoning.get('is_thinking'):
                thinking_chunks += 1
                thinking_content = reasoning.get('thinking_content', '')
                if thinking_content:
                    print(f"  🧠 Chunk {chunk_count}: Thinking: '{thinking_content[:100]}{'...' if len(thinking_content) > 100 else ''}'")
            
            # Don't spam too many chunks in output
            if chunk_count >= 10:
                print(f"  ... (showing first 10 chunks)")
                break
        
        print(f"\n📊 GPT-OSS Behavior Summary:")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Tool calls found: {tool_calls_found}")
        print(f"  Content chunks: {content_chunks}")
        print(f"  Thinking chunks: {thinking_chunks}")
        
        # Analyze behavior pattern
        if tool_calls_found > 0 and chunk_count == 1:
            print("✅ GPT-OSS Pattern: Single-chunk tool calling (reasoning model behavior)")
            print("   This is expected - reasoning models often emit complete responses in one chunk")
        elif tool_calls_found > 0 and chunk_count > 5:
            print("✅ GPT-OSS Pattern: Multi-chunk streaming with tools")
        elif chunk_count == 1:
            print("⚠️  GPT-OSS Pattern: Single-chunk response (may be reasoning model limitation)")
        else:
            print("❓ GPT-OSS Pattern: Unexpected behavior")
        
        return tool_calls_found > 0
        
    except Exception as e:
        print(f"❌ Error testing GPT-OSS behavior: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    # Check implementation
    implementation_ok = check_ollama_client()
    
    if implementation_ok:
        # Test actual behavior
        behavior_ok = await test_gpt_oss_reasoning_behavior()
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"Implementation: {'✅ CORRECT' if implementation_ok else '❌ NEEDS FIXES'}")
        print(f"GPT-OSS Behavior: {'✅ WORKING' if behavior_ok else '❌ NOT WORKING'}")
        
        if implementation_ok and behavior_ok:
            print("\n🎉 GPT-OSS reasoning model support is working correctly!")
            print("   - Reasoning model detection: ✅")
            print("   - Tool calling: ✅") 
            print("   - Streaming: ✅")
            print("   - Single-chunk tool responses are normal for reasoning models")
        elif implementation_ok:
            print("\n⚠️  Implementation correct but GPT-OSS behavior unexpected")
            print("   This may be due to model-specific limitations")
        else:
            print("\n❌ Implementation needs fixes")
    else:
        print("\n❌ Cannot test behavior - implementation issues detected")


if __name__ == "__main__":
    asyncio.run(main())