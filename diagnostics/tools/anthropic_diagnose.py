#!/usr/bin/env python3
"""
Diagnostic script for Anthropic MCP tool compatibility and streaming performance.
Tests tool name handling, streaming behavior, and feature capabilities.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

async def test_anthropic_tool_handling():
    """Test that Anthropic handles MCP-style tool names correctly"""
    print("🧪 Testing Anthropic MCP Tool Name Handling")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.providers.anthropic_client import AnthropicLLMClient
        
        # Create client instance to test tool handling
        client = AnthropicLLMClient(model="claude-sonnet-4-20250514")
        
        # Test MCP-style tool names that would break Mistral
        mcp_tools = [
            "stdio.read_query",
            "filesystem.read_file", 
            "mcp.server:get_data",
            "some.complex:tool.name",
            "already_valid_name",
        ]
        
        print("Testing Anthropic tool name handling:")
        for tool_name in mcp_tools:
            test_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Test tool: {tool_name}",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
            
            # Test the sanitization method (should be pass-through)
            sanitized = client._sanitize_tool_names(test_tools)
            original_name = test_tools[0]["function"]["name"]
            sanitized_name = sanitized[0]["function"]["name"] if sanitized else "ERROR"
            
            status = "✅ PRESERVED" if original_name == sanitized_name else "❌ CHANGED"
            print(f"  {original_name:<30} -> {sanitized_name:<30} {status}")
        
        # Test tool conversion to Anthropic format
        print("\nTesting tool conversion to Anthropic format:")
        complex_tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",
                    "description": "Read from stdin",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Prompt text"}
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]
        
        converted = client._convert_tools(complex_tools)
        if converted:
            original = complex_tools[0]["function"]["name"]
            converted_name = converted[0]["name"]
            print(f"  Original: {original}")
            print(f"  Anthropic format: {converted_name}")
            print(f"  Status: {'✅ PRESERVED' if original == converted_name else '❌ CHANGED'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool handling: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_anthropic_streaming():
    """Test Anthropic streaming performance and behavior"""
    print("\n🧪 Testing Anthropic Streaming Performance")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="anthropic", model="claude-sonnet-4-20250514")
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Provider name: {getattr(client, 'provider_name', 'unknown')}")
        
        messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it at least 100 words and tell it slowly."}
        ]
        
        print("\n🔍 Testing Anthropic streaming=True...")
        start_time = time.time()
        
        # Test streaming
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
                
                # Show timing for first few chunks
                if chunk_count <= 5 or chunk_count % 10 == 0:
                    interval = current_time - last_chunk_time
                    print(f"\n   Chunk {chunk_count}: {relative_time:.3f}s (interval: {interval:.4f}s)")
                    print("   Continuing: ", end="", flush=True)
                
                last_chunk_time = current_time
            
            end_time = time.time() - start_time
            
            # Calculate statistics
            if chunk_intervals:
                avg_interval = sum(chunk_intervals) / len(chunk_intervals)
                min_interval = min(chunk_intervals)
                max_interval = max(chunk_intervals)
            else:
                avg_interval = min_interval = max_interval = 0
            
            print(f"\n\n📊 ANTHROPIC STREAMING ANALYSIS:")
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
                print("   ⚠️  FAKE STREAMING: Only one chunk")
            elif chunk_count < 5:
                print("   ⚠️  LIMITED STREAMING: Very few chunks")
            else:
                print("   ✅ REAL STREAMING: Multiple chunks detected")
            
            if first_chunk_time:
                if first_chunk_time < 2.0:
                    print("   ✅ FAST: Excellent first chunk time")
                elif first_chunk_time < 4.0:
                    print("   ✅ GOOD: Acceptable first chunk time")
                else:
                    print("   ⚠️  SLOW: First chunk could be faster")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Anthropic streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Anthropic streaming: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_anthropic_mcp_tools():
    """Test Anthropic with actual MCP-style tool calls"""
    print("\n🧪 Testing Anthropic with MCP-Style Tools")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="anthropic", model="claude-sonnet-4-20250514")
        
        # MCP-style tools that would break Mistral but should work fine with Anthropic
        mcp_tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",
                    "description": "Read a query from standard input",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to display"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filesystem.list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        ]
        
        messages = [
            {
                "role": "user",
                "content": "Please list the files in the current directory and then ask the user for their name"
            }
        ]
        
        print("Testing Anthropic with MCP-style tool names...")
        print(f"Tools: {[t['function']['name'] for t in mcp_tools]}")
        
        # Test non-streaming first
        response = await client.create_completion(
            messages=messages,
            tools=mcp_tools,
            stream=False
        )
        
        print("✅ SUCCESS: No tool naming errors!")
        
        if isinstance(response, dict):
            if response.get("tool_calls"):
                print(f"🔧 Tool calls made: {len(response['tool_calls'])}")
                for i, tool_call in enumerate(response["tool_calls"]):
                    func_name = tool_call.get("function", {}).get("name", "unknown")
                    print(f"   {i+1}. {func_name}")
                    
                    # Verify MCP names are preserved
                    if func_name in ["stdio.read_query", "filesystem.list_files"]:
                        print(f"      ✅ MCP name preserved: {func_name}")
                    else:
                        print(f"      ⚠️  Unexpected name: {func_name}")
                        
            elif response.get("response"):
                print(f"💬 Text response: {response['response'][:150]}...")
            else:
                print(f"❓ Unexpected response format")
        
        # Test streaming with tools
        print("\n🔄 Testing streaming with MCP tools...")
        stream_response = client.create_completion(
            messages=messages,
            tools=mcp_tools,
            stream=True
        )
        
        chunk_count = 0
        tool_calls_found = []
        
        async for chunk in stream_response:
            chunk_count += 1
            if chunk.get("tool_calls"):
                for tc in chunk["tool_calls"]:
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    tool_calls_found.append(tool_name)
                    print(f"🔧 Streaming tool call: {tool_name}")
            
            if chunk_count >= 10:  # Limit for testing
                break
        
        print(f"✅ Streaming completed: {chunk_count} chunks, {len(tool_calls_found)} tool calls")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_anthropic_features():
    """Test Anthropic advanced features"""
    print("\n🧪 Testing Anthropic Advanced Features")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="anthropic", model="claude-sonnet-4-20250514")
        
        # Test model info
        print("📋 Model capabilities:")
        model_info = client.get_model_info()
        
        features = model_info.get("features", [])
        print(f"   Features: {', '.join(features)}")
        print(f"   Max context: {model_info.get('max_context_length', 'unknown')}")
        print(f"   Max output: {model_info.get('max_output_tokens', 'unknown')}")
        print(f"   Vision support: {'✅' if 'vision' in features else '❌'}")
        print(f"   Tool support: {'✅' if 'tools' in features else '❌'}")
        print(f"   Streaming: {'✅' if 'streaming' in features else '❌'}")
        print(f"   JSON mode: {'✅' if 'json_mode' in features else '❌'}")
        
        # Test vision if supported
        if 'vision' in features:
            print("\n🖼️  Testing vision capabilities...")
            # Simple red pixel test (1x1 red pixel PNG)
            red_pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this 1x1 pixel image? Be specific about the color."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{red_pixel_b64}"
                            }
                        }
                    ]
                }
            ]
            
            print(f"   Sending image data URL: data:image/png;base64,{red_pixel_b64[:20]}...")
            
            try:
                vision_response = await client.create_completion(vision_messages, stream=False)
                if isinstance(vision_response, dict) and vision_response.get("response"):
                    response_text = vision_response['response']
                    print(f"   📝 Vision response: {response_text[:200]}...")
                    
                    # Check if it actually analyzed the image
                    if any(word in response_text.lower() for word in ['red', 'color', 'pixel', 'image']):
                        print(f"   ✅ Vision works: Model analyzed the image content")
                    elif "don't see" in response_text.lower() or "no image" in response_text.lower():
                        print(f"   ❌ Vision failed: Model didn't receive the image")
                    else:
                        print(f"   ⚠️  Vision unclear: Model responded but unclear if image was processed")
                elif vision_response.get("error"):
                    print(f"   ❌ Vision error: {vision_response.get('error')}")
                else:
                    print(f"   ❌ Vision test failed: Unexpected response format")
            except Exception as e:
                print(f"   ❌ Vision test exception: {e}")
                
            # Also test with a more complex image description
            print("\n🖼️  Testing with image description request...")
            describe_messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Describe this image in detail. What do you see?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{red_pixel_b64}"
                            }
                        }
                    ]
                }
            ]
            
            try:
                describe_response = await client.create_completion(describe_messages, stream=False)
                if isinstance(describe_response, dict) and describe_response.get("response"):
                    desc_text = describe_response['response']
                    print(f"   📝 Description: {desc_text[:150]}...")
                    
                    if "don't see" in desc_text.lower() or "no image" in desc_text.lower():
                        print(f"   ❌ Vision processing failed - image not received by model")
                        print(f"   💡 This suggests an issue with image format conversion or API call")
                    else:
                        print(f"   ✅ Vision processing appears to work")
                else:
                    print(f"   ❌ Description test failed")
            except Exception as e:
                print(f"   ❌ Description test exception: {e}")
        
        # Test JSON mode if supported
        if 'json_mode' in features:
            print("\n📊 Testing JSON mode...")
            json_messages = [
                {"role": "user", "content": "Return a JSON object with your name and capabilities"}
            ]
            
            json_response = await client.create_completion(
                json_messages, 
                stream=False,
                response_format={"type": "json_object"}
            )
            
            if isinstance(json_response, dict) and json_response.get("response"):
                try:
                    import json
                    parsed = json.loads(json_response["response"])
                    print(f"   ✅ JSON mode works: {json_response['response'][:100]}...")
                except json.JSONDecodeError:
                    print(f"   ⚠️  Response not valid JSON: {json_response['response'][:100]}...")
            else:
                print(f"   ❌ JSON mode test failed: {json_response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing features: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_anthropic_vs_competitors():
    """Compare Anthropic performance vs other providers"""
    print("\n🧪 Anthropic vs Competitors Performance")
    print("=" * 50)
    
    # Same test prompt for all providers
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence"}
    ]
    
    providers = [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("mistral", "mistral-medium-2505"),
        ("openai", "gpt-4o-mini"),
        ("groq", "llama-3.3-70b-versatile")
    ]
    
    results = {}
    
    for provider, model in providers:
        print(f"\n🔍 Testing {provider} with {model}...")
        
        try:
            from chuk_llm.llm.client import get_client
            client = get_client(provider=provider, model=model)
            
            start_time = time.time()
            response = client.create_completion(messages, stream=True)
            
            chunk_count = 0
            first_chunk_time = None
            content_length = 0
            
            async for chunk in response:
                current_time = time.time() - start_time
                
                if first_chunk_time is None:
                    first_chunk_time = current_time
                
                chunk_count += 1
                
                if isinstance(chunk, dict) and chunk.get("response"):
                    content_length += len(chunk["response"])
                
                # Limit for comparison
                if chunk_count >= 20:
                    break
            
            total_time = time.time() - start_time
            
            results[provider] = {
                "chunks": chunk_count,
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "content_length": content_length
            }
            
            print(f"   {provider}: {chunk_count} chunks, first at {first_chunk_time:.3f}s, {content_length} chars, total {total_time:.3f}s")
            
        except Exception as e:
            print(f"   {provider}: Error - {e}")
            results[provider] = None
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        fastest_first = min(valid_results.keys(), key=lambda k: valid_results[k]["first_chunk"])
        most_chunks = max(valid_results.keys(), key=lambda k: valid_results[k]["chunks"])
        fastest_total = min(valid_results.keys(), key=lambda k: valid_results[k]["total_time"])
        
        print(f"   🚀 Fastest first chunk: {fastest_first} ({valid_results[fastest_first]['first_chunk']:.3f}s)")
        print(f"   📊 Most granular streaming: {most_chunks} ({valid_results[most_chunks]['chunks']} chunks)")
        print(f"   ⚡ Fastest total time: {fastest_total} ({valid_results[fastest_total]['total_time']:.3f}s)")
        
        # Anthropic-specific analysis
        if "anthropic" in valid_results:
            anthropic_result = valid_results["anthropic"]
            print(f"\n🎯 ANTHROPIC ANALYSIS:")
            print(f"   Chunks: {anthropic_result['chunks']}")
            print(f"   First chunk: {anthropic_result['first_chunk']:.3f}s")
            print(f"   Content: {anthropic_result['content_length']} chars")
            
            # Compare to others
            other_providers = [k for k in valid_results.keys() if k != "anthropic"]
            if other_providers:
                avg_first_chunk = sum(valid_results[p]["first_chunk"] for p in other_providers) / len(other_providers)
                avg_chunks = sum(valid_results[p]["chunks"] for p in other_providers) / len(other_providers)
                
                if anthropic_result["first_chunk"] < avg_first_chunk:
                    print(f"   ✅ Anthropic faster than average first chunk by {(avg_first_chunk - anthropic_result['first_chunk'])*1000:.0f}ms")
                else:
                    print(f"   ⚠️  Anthropic slower than average first chunk by {(anthropic_result['first_chunk'] - avg_first_chunk)*1000:.0f}ms")
                
                if anthropic_result["chunks"] > avg_chunks:
                    print(f"   ✅ Anthropic more granular than average ({anthropic_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
                else:
                    print(f"   ⚠️  Anthropic less granular than average ({anthropic_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
    
    return len(valid_results) > 0


async def main():
    """Run all Anthropic diagnostic tests"""
    print("🚀 Testing Anthropic MCP Tool Compatibility & Performance")
    print("=" * 60)
    
    # Test 1: Tool name handling
    test1_passed = await test_anthropic_tool_handling()
    
    # Test 2: Streaming performance
    test2_passed = await test_anthropic_streaming() if test1_passed else False
    
    # Test 3: MCP tools integration
    test3_passed = await test_anthropic_mcp_tools() if test2_passed else False
    
    # Test 4: Advanced features
    test4_passed = await test_anthropic_features() if test3_passed else False
    
    # Test 5: Performance comparison
    test5_passed = await test_anthropic_vs_competitors() if test4_passed else False
    
    print("\n" + "=" * 60)
    print("🎯 ANTHROPIC DIAGNOSTIC RESULTS:")
    print(f"   Tool Name Handling: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Streaming Performance: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   MCP Tools Integration: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"   Advanced Features: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"   Performance Comparison: {'✅ PASS' if test5_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed]):
        print("\n🎉 ALL ANTHROPIC TESTS PASSED!")
        print("💡 Anthropic is ready for MCP CLI:")
        print("   mcp-cli chat --provider anthropic --model claude-sonnet-4-20250514")
        print("\n🔑 Key Advantages of Anthropic:")
        print("   • Native MCP tool name support (no sanitization needed)")
        print("   • Excellent vision capabilities")
        print("   • Large context windows")
        print("   • High-quality responses")
    else:
        print("\n❌ Some Anthropic tests failed.")
        print("💡 Check the implementation and ensure:")
        print("   1. anthropic_client.py has _sanitize_tool_names method")
        print("   2. AsyncAnthropic is properly configured")
        print("   3. API key is set correctly")


if __name__ == "__main__":
    asyncio.run(main())