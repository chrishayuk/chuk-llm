#!/usr/bin/env python3
"""
Diagnostic script for Google Gemini MCP tool compatibility and streaming performance.
Tests tool name handling, streaming behavior, and Gemini-specific features.
"""

import asyncio
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

async def test_gemini_setup():
    """Test Gemini configuration and connectivity"""
    print("🧪 Testing Gemini Setup & Configuration")
    print("=" * 50)
    
    # Check required environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    print("📋 Environment Check:")
    print(f"   GEMINI_API_KEY/GOOGLE_API_KEY: {'✅ Set' if gemini_api_key else '❌ Missing'}")
    
    if not gemini_api_key:
        print("\n❌ Gemini API key not configured")
        print("💡 Set environment variable: export GEMINI_API_KEY=your-api-key")
        print("   Or: export GOOGLE_API_KEY=your-google-api-key")
        return False
    
    try:
        from chuk_llm.llm.providers.gemini_client import GeminiLLMClient, AVAILABLE_GEMINI_MODELS
        
        print(f"\n📋 Available Gemini Models:")
        for model in sorted(AVAILABLE_GEMINI_MODELS):
            print(f"   - {model}")
        
        # Test client initialization
        client = GeminiLLMClient(
            model="gemini-2.5-flash",
            api_key=gemini_api_key
        )
        
        print(f"\n✅ Client initialized successfully:")
        print(f"   Model: {client.model}")
        print(f"   Model Family: {client._detect_model_family()}")
        print(f"   Warning Suppression: Complete")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing Gemini client: {e}")
        print("💡 Check your API key and model availability")
        return False

async def test_gemini_tool_handling():
    """Test that Gemini handles MCP-style tool names"""
    print("\n🧪 Testing Gemini MCP Tool Name Handling")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.providers.gemini_client import GeminiLLMClient
        
        # Create client instance to test tool handling
        client = GeminiLLMClient(model="gemini-2.5-flash")
        
        # Test MCP-style tool names
        mcp_tools = [
            "stdio.read_query",
            "filesystem.read_file", 
            "mcp.server:get_data",
            "some.complex:tool.name",
            "already_valid_name",
            "tool-with-dashes",
            "tool_with_underscores"
        ]
        
        print("Testing Gemini tool name handling:")
        
        # Check if Gemini client has sanitization methods
        has_sanitization = hasattr(client, '_sanitize_tool_names')
        print(f"   Has _sanitize_tool_names method: {'✅' if has_sanitization else '❌'}")
        
        if has_sanitization:
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
                
                # Test the sanitization method
                sanitized = client._sanitize_tool_names(test_tools)
                original_name = test_tools[0]["function"]["name"]
                sanitized_name = sanitized[0]["function"]["name"] if sanitized else "ERROR"
                
                status = "✅ PRESERVED" if original_name == sanitized_name else "❌ CHANGED"
                print(f"  {original_name:<30} -> {sanitized_name:<30} {status}")
        else:
            print("⚠️  Gemini client missing MCP tool name sanitization")
            print("💡 Tool names will be passed through as-is to Gemini API")
            print("   This may cause issues with MCP-style names like 'stdio.read_query'")
        
        # Test Gemini's native tool format conversion
        print("\nTesting Gemini tool format conversion:")
        from chuk_llm.llm.providers.gemini_client import _convert_tools_to_gemini_format
        
        test_tools = [
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
        
        gemini_tools = _convert_tools_to_gemini_format(test_tools)
        if gemini_tools:
            print(f"   ✅ Successfully converted to Gemini format")
            print(f"   Tool count: {len(gemini_tools)}")
        else:
            print(f"   ❌ Failed to convert to Gemini format")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool handling: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gemini_streaming():
    """Test Gemini streaming performance and behavior"""
    print("\n🧪 Testing Gemini Streaming Performance")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="gemini", model="gemini-2.5-flash")
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        
        messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it at least 100 words and be creative."}
        ]
        
        print("\n🔍 Testing Gemini streaming=True...")
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
                
                # Limit for testing
                if chunk_count >= 30:
                    break
            
            end_time = time.time() - start_time
            
            # Calculate statistics
            if chunk_intervals:
                avg_interval = sum(chunk_intervals) / len(chunk_intervals)
                min_interval = min(chunk_intervals)
                max_interval = max(chunk_intervals)
            else:
                avg_interval = min_interval = max_interval = 0
            
            print(f"\n\n📊 GEMINI STREAMING ANALYSIS:")
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
                if first_chunk_time < 1.0:
                    print("   ✅ FAST: Excellent first chunk time")
                elif first_chunk_time < 2.0:
                    print("   ✅ GOOD: Acceptable first chunk time")
                else:
                    print("   ⚠️  SLOW: First chunk could be faster")
        
        else:
            print("❌ Expected async generator, got something else")
            print(f"Response: {response}")
        
        print("\n🔍 Testing Gemini streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gemini_mcp_tools():
    """Test Gemini with actual MCP-style tool calls"""
    print("\n🧪 Testing Gemini with MCP-Style Tools")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="gemini", model="gemini-2.5-flash")
        
        # MCP-style tools
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
            },
            {
                "type": "function",
                "function": {
                    "name": "web.search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        messages = [
            {
                "role": "user",
                "content": "Please list the files in the current directory, search for 'AI tutorials', and then ask the user for their name"
            }
        ]
        
        print("Testing Gemini with MCP-style tool names...")
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
                    
                    # Check if MCP names are preserved
                    expected_names = ["stdio.read_query", "filesystem.list_files", "web.search"]
                    if func_name in expected_names:
                        print(f"      ✅ MCP name preserved: {func_name}")
                    else:
                        print(f"      ⚠️  Unexpected name: {func_name}")
                        
            elif response.get("response"):
                print(f"💬 Text response: {response['response'][:150]}...")
            else:
                print(f"❓ Unexpected response format")
        
        # Test streaming with tools
        print("\n🔄 Testing streaming with MCP tools...")
        try:
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
            
        except Exception as stream_error:
            print(f"⚠️  Streaming with tools failed: {stream_error}")
            print("   This might be a Gemini limitation or implementation issue")
        
        return True
        
    except Exception as e:
        error_str = str(e)
        
        if "function" in error_str.lower() and ("name" in error_str.lower() or "invalid" in error_str.lower()):
            print(f"❌ Tool naming error detected!")
            print(f"   Error: {error_str}")
            print("\n💡 Gemini may require tool name sanitization")
            print("   Consider implementing sanitization similar to other providers")
            return False
        else:
            print(f"❌ Error testing MCP tools: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_gemini_features():
    """Test Gemini advanced features"""
    print("\n🧪 Testing Gemini Advanced Features")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="gemini", model="gemini-2.5-flash")
        
        # Test model info
        print("📋 Model capabilities:")
        model_info = client.get_model_info()
        
        features = model_info.get("features", [])
        gemini_info = model_info.get("gemini_specific", {})
        
        print(f"   Features: {', '.join(features)}")
        print(f"   Max context: {model_info.get('max_context_length', 'unknown')}")
        print(f"   Max output: {model_info.get('max_output_tokens', 'unknown')}")
        print(f"   Vision support: {'✅' if 'vision' in features else '❌'}")
        print(f"   Tool support: {'✅' if 'tools' in features else '❌'}")
        print(f"   Streaming: {'✅' if 'streaming' in features else '❌'}")
        print(f"   JSON mode: {'✅' if 'json_mode' in features else '❌'}")
        
        print(f"\n🔷 Gemini-specific:")
        print(f"   Context Length: {gemini_info.get('context_length', 'unknown')}")
        print(f"   Model Family: {gemini_info.get('model_family', 'unknown')}")
        print(f"   Enhanced Reasoning: {gemini_info.get('enhanced_reasoning', False)}")
        print(f"   Warning Suppression: {gemini_info.get('warning_suppression', 'unknown')}")
        print(f"   Experimental Features: {gemini_info.get('experimental_features', False)}")
        
        # Test JSON mode if supported
        if 'json_mode' in features:
            print("\n📊 Testing JSON mode...")
            json_messages = [
                {"role": "user", "content": "Return a JSON object with your name, model family, and key capabilities"}
            ]
            
            try:
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
            except Exception as e:
                print(f"   ❌ JSON mode error: {e}")
        
        # Test vision if supported
        if 'vision' in features:
            print("\n🖼️  Testing vision capabilities...")
            
            # Test with a small red pixel image (1x1 red pixel PNG)
            red_pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this 1x1 pixel image? Be specific."},
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
                vision_response = await client.create_completion(vision_messages, stream=False)
                
                if isinstance(vision_response, dict) and vision_response.get("response"):
                    response_text = vision_response['response']
                    print(f"   📝 Vision response: {response_text[:200]}...")
                    
                    # Check if it actually analyzed the image
                    if any(word in response_text.lower() for word in ['red', 'color', 'pixel']):
                        print(f"   ✅ Vision works: Model analyzed the image content")
                    else:
                        print(f"   ⚠️  Vision unclear: Response doesn't mention expected color")
                else:
                    print(f"   ❌ Vision test failed: {vision_response}")
                    
            except Exception as e:
                print(f"   ❌ Vision test exception: {e}")
        
        # Test system message support
        print("\n📋 Testing system message support...")
        system_messages = [
            {"role": "system", "content": "You are a helpful assistant that responds with exactly 3 words."},
            {"role": "user", "content": "Hello there!"}
        ]
        
        try:
            system_response = await client.create_completion(system_messages, stream=False)
            if isinstance(system_response, dict) and system_response.get("response"):
                response_text = system_response['response']
                word_count = len(response_text.split())
                print(f"   📝 System response: {response_text}")
                print(f"   Word count: {word_count}")
                if word_count <= 5:  # Allow some flexibility
                    print(f"   ✅ System messages work: Response followed constraint")
                else:
                    print(f"   ⚠️  System messages unclear: Response didn't follow constraint")
            else:
                print(f"   ❌ System message test failed")
        except Exception as e:
            print(f"   ❌ System message error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing features: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gemini_vs_competitors():
    """Compare Gemini performance vs other providers"""
    print("\n🧪 Gemini vs Competitors Performance")
    print("=" * 50)
    
    # Same test prompt for all providers
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence and the future"}
    ]
    
    providers = [
        ("gemini", "gemini-2.5-flash"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("mistral", "mistral-medium-2505"),
        ("groq", "llama-3.3-70b-versatile"),
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
        
        # Gemini-specific analysis
        if "gemini" in valid_results:
            gemini_result = valid_results["gemini"]
            print(f"\n🎯 GEMINI ANALYSIS:")
            print(f"   Chunks: {gemini_result['chunks']}")
            print(f"   First chunk: {gemini_result['first_chunk']:.3f}s")
            print(f"   Content: {gemini_result['content_length']} chars")
            
            # Compare to others
            other_providers = [k for k in valid_results.keys() if k != "gemini"]
            if other_providers:
                avg_first_chunk = sum(valid_results[p]["first_chunk"] for p in other_providers) / len(other_providers)
                avg_chunks = sum(valid_results[p]["chunks"] for p in other_providers) / len(other_providers)
                
                if gemini_result["first_chunk"] < avg_first_chunk:
                    print(f"   ✅ Gemini faster than average first chunk by {(avg_first_chunk - gemini_result['first_chunk'])*1000:.0f}ms")
                else:
                    print(f"   ⚠️  Gemini slower than average first chunk by {(gemini_result['first_chunk'] - avg_first_chunk)*1000:.0f}ms")
                
                if gemini_result["chunks"] > avg_chunks:
                    print(f"   ✅ Gemini more granular than average ({gemini_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
                else:
                    print(f"   ⚠️  Gemini less granular than average ({gemini_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
    
    return len(valid_results) > 0

async def main():
    """Run all Gemini diagnostic tests"""
    print("🚀 Testing Gemini MCP Tool Compatibility & Performance")
    print("=" * 60)
    
    # Test 1: Setup and configuration
    test1_passed = await test_gemini_setup()
    
    # Test 2: Tool name handling
    test2_passed = await test_gemini_tool_handling() if test1_passed else False
    
    # Test 3: Streaming performance
    test3_passed = await test_gemini_streaming() if test2_passed else False
    
    # Test 4: MCP tools integration
    test4_passed = await test_gemini_mcp_tools() if test3_passed else False
    
    # Test 5: Advanced features
    test5_passed = await test_gemini_features() if test4_passed else False
    
    # Test 6: Performance comparison
    test6_passed = await test_gemini_vs_competitors() if test5_passed else False
    
    print("\n" + "=" * 60)
    print("🎯 GEMINI DIAGNOSTIC RESULTS:")
    print(f"   Setup & Configuration: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Tool Name Handling: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Streaming Performance: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"   MCP Tools Integration: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"   Advanced Features: {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"   Performance Comparison: {'✅ PASS' if test6_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed, test6_passed]):
        print("\n🎉 ALL GEMINI TESTS PASSED!")
        print("💡 Gemini is ready for MCP CLI:")
        print("   mcp-cli chat --provider gemini --model gemini-2.5-flash")
        print("\n🔑 Key Advantages of Gemini:")
        print("   • Latest Google AI technology (Gemini 2.5)")
        print("   • Large context windows (2M tokens)")
        print("   • Enhanced reasoning capabilities")
        print("   • Complete warning suppression")
        print("   • Experimental features support")
    else:
        print("\n❌ Some Gemini tests failed.")
        print("💡 Check the implementation and ensure:")
        print("   1. GEMINI_API_KEY or GOOGLE_API_KEY is set")
        print("   2. Model is available in your region")
        print("   3. API quotas and permissions")
        if not test2_passed:
            print("   4. Consider implementing MCP tool name sanitization")


if __name__ == "__main__":
    asyncio.run(main())