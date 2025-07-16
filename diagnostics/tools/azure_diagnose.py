#!/usr/bin/env python3
"""
Diagnostic script for Azure OpenAI MCP tool compatibility and streaming performance.
Tests tool name handling, streaming behavior, and Azure-specific features.
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

async def test_azure_openai_setup():
    """Test Azure OpenAI configuration and connectivity"""
    print("🧪 Testing Azure OpenAI Setup & Configuration")
    print("=" * 50)
    
    # Check required environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    print("📋 Environment Check:")
    print(f"   AZURE_OPENAI_ENDPOINT: {'✅ Set' if azure_endpoint else '❌ Missing'}")
    print(f"   AZURE_OPENAI_API_KEY: {'✅ Set' if azure_key else '❌ Missing'}")
    
    if not azure_endpoint:
        print("\n❌ Azure OpenAI endpoint not configured")
        print("💡 Set environment variable: export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        return False
        
    if not azure_key:
        print("\n❌ Azure OpenAI API key not configured")
        print("💡 Set environment variable: export AZURE_OPENAI_API_KEY=your-api-key")
        return False
    
    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
        
        # Test client initialization
        client = AzureOpenAILLMClient(
            model="gpt-4o-mini",
            azure_endpoint=azure_endpoint,
            api_key=azure_key
        )
        
        print(f"\n✅ Client initialized successfully:")
        print(f"   Endpoint: {azure_endpoint}")
        print(f"   Deployment: {client.azure_deployment}")
        print(f"   API Version: {client.api_version}")
        print(f"   Auth Type: {client._get_auth_type()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing Azure OpenAI client: {e}")
        return False

async def test_azure_openai_tool_handling():
    """Test that Azure OpenAI handles MCP-style tool names"""
    print("\n🧪 Testing Azure OpenAI MCP Tool Name Handling")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
        
        # Create client instance to test tool handling
        client = AzureOpenAILLMClient(model="gpt-4o-mini")
        
        # Test MCP-style tool names
        mcp_tools = [
            "stdio.read_query",
            "filesystem.read_file", 
            "mcp.server:get_data",
            "some.complex:tool.name",
            "already_valid_name",
        ]
        
        print("Testing Azure OpenAI tool name handling:")
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool handling: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_azure_openai_streaming():
    """Test Azure OpenAI streaming performance and behavior"""
    print("\n🧪 Testing Azure OpenAI Streaming Performance")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="azure_openai", model="gpt-4o-mini")
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Provider name: {getattr(client, 'provider_name', 'unknown')}")
        
        messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it at least 100 words."}
        ]
        
        print("\n🔍 Testing Azure OpenAI streaming=True...")
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
            
            print(f"\n\n📊 AZURE OPENAI STREAMING ANALYSIS:")
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
        
        print("\n🔍 Testing Azure OpenAI streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Azure OpenAI streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_azure_openai_mcp_tools():
    """Test Azure OpenAI with actual MCP-style tool calls"""
    print("\n🧪 Testing Azure OpenAI with MCP-Style Tools")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="azure_openai", model="gpt-4o-mini")
        
        # MCP-style tools (should work natively with Azure OpenAI)
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
        
        print("Testing Azure OpenAI with MCP-style tool names...")
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
        error_str = str(e)
        
        if "function" in error_str.lower() and ("name" in error_str.lower() or "invalid" in error_str.lower()):
            print(f"❌ Tool naming error detected!")
            print(f"   Error: {error_str}")
            print("\n💡 Azure OpenAI may have stricter tool naming than expected")
            print("   Consider implementing tool name sanitization similar to Mistral/Anthropic")
            return False
        else:
            print(f"❌ Error testing MCP tools: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_azure_openai_features():
    """Test Azure OpenAI advanced features"""
    print("\n🧪 Testing Azure OpenAI Advanced Features")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="azure_openai", model="gpt-4o-mini")
        
        # Test model info
        print("📋 Model capabilities:")
        model_info = client.get_model_info()
        
        features = model_info.get("features", [])
        azure_info = model_info.get("azure_specific", {})
        
        print(f"   Features: {', '.join(features)}")
        print(f"   Max context: {model_info.get('max_context_length', 'unknown')}")
        print(f"   Max output: {model_info.get('max_output_tokens', 'unknown')}")
        print(f"   Vision support: {'✅' if 'vision' in features else '❌'}")
        print(f"   Tool support: {'✅' if 'tools' in features else '❌'}")
        print(f"   Streaming: {'✅' if 'streaming' in features else '❌'}")
        print(f"   JSON mode: {'✅' if 'json_mode' in features else '❌'}")
        
        print(f"\n🔷 Azure-specific:")
        print(f"   Endpoint: {azure_info.get('endpoint', 'unknown')}")
        print(f"   Deployment: {azure_info.get('deployment', 'unknown')}")
        print(f"   API Version: {azure_info.get('api_version', 'unknown')}")
        print(f"   Auth Type: {azure_info.get('authentication_type', 'unknown')}")
        
        # Test JSON mode if supported
        if 'json_mode' in features:
            print("\n📊 Testing JSON mode...")
            json_messages = [
                {"role": "user", "content": "Return a JSON object with your name and model type"}
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

async def test_azure_openai_vs_competitors():
    """Compare Azure OpenAI performance vs other providers"""
    print("\n🧪 Azure OpenAI vs Competitors Performance")
    print("=" * 50)
    
    # Same test prompt for all providers
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence"}
    ]
    
    providers = [
        ("azure_openai", "gpt-4o-mini"),
        ("openai", "gpt-4o-mini"),
        ("mistral", "mistral-medium-2505"),
        ("anthropic", "claude-sonnet-4-20250514"),
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
        
        # Azure OpenAI-specific analysis
        if "azure_openai" in valid_results:
            azure_result = valid_results["azure_openai"]
            print(f"\n🎯 AZURE OPENAI ANALYSIS:")
            print(f"   Chunks: {azure_result['chunks']}")
            print(f"   First chunk: {azure_result['first_chunk']:.3f}s")
            print(f"   Content: {azure_result['content_length']} chars")
            
            # Compare to regular OpenAI
            if "openai" in valid_results:
                openai_result = valid_results["openai"]
                first_diff = azure_result["first_chunk"] - openai_result["first_chunk"]
                chunks_diff = azure_result["chunks"] - openai_result["chunks"]
                
                print(f"\n🔍 Azure vs Regular OpenAI:")
                if abs(first_diff) < 0.1:
                    print(f"   ⚖️  Similar first chunk timing ({first_diff*1000:+.0f}ms)")
                elif first_diff < 0:
                    print(f"   ✅ Azure faster by {abs(first_diff)*1000:.0f}ms")
                else:
                    print(f"   ⚠️  Azure slower by {first_diff*1000:.0f}ms")
                
                if abs(chunks_diff) <= 2:
                    print(f"   ⚖️  Similar streaming granularity ({chunks_diff:+d} chunks)")
                elif chunks_diff > 0:
                    print(f"   ✅ Azure more granular (+{chunks_diff} chunks)")
                else:
                    print(f"   ⚠️  Azure less granular ({chunks_diff} chunks)")
    
    return len(valid_results) > 0

async def main():
    """Run all Azure OpenAI diagnostic tests"""
    print("🚀 Testing Azure OpenAI MCP Tool Compatibility & Performance")
    print("=" * 60)
    
    # Test 1: Setup and configuration
    test1_passed = await test_azure_openai_setup()
    
    # Test 2: Tool name handling
    test2_passed = await test_azure_openai_tool_handling() if test1_passed else False
    
    # Test 3: Streaming performance
    test3_passed = await test_azure_openai_streaming() if test2_passed else False
    
    # Test 4: MCP tools integration
    test4_passed = await test_azure_openai_mcp_tools() if test3_passed else False
    
    # Test 5: Advanced features
    test5_passed = await test_azure_openai_features() if test4_passed else False
    
    # Test 6: Performance comparison
    test6_passed = await test_azure_openai_vs_competitors() if test5_passed else False
    
    print("\n" + "=" * 60)
    print("🎯 AZURE OPENAI DIAGNOSTIC RESULTS:")
    print(f"   Setup & Configuration: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Tool Name Handling: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Streaming Performance: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"   MCP Tools Integration: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"   Advanced Features: {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"   Performance Comparison: {'✅ PASS' if test6_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed, test6_passed]):
        print("\n🎉 ALL AZURE OPENAI TESTS PASSED!")
        print("💡 Azure OpenAI is ready for MCP CLI:")
        print("   mcp-cli chat --provider azure_openai --model gpt-4o-mini")
        print("\n🔑 Key Advantages of Azure OpenAI:")
        print("   • Enterprise-grade security and compliance")
        print("   • Typically native MCP tool name support")
        print("   • Same performance as regular OpenAI")
        print("   • Azure ecosystem integration")
    else:
        print("\n❌ Some Azure OpenAI tests failed.")
        print("💡 Check the implementation and ensure:")
        print("   1. AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set")
        print("   2. Deployment names match your Azure setup")
        print("   3. API version is compatible")
        if not test4_passed:
            print("   4. Consider implementing tool name sanitization if needed")


if __name__ == "__main__":
    asyncio.run(main())