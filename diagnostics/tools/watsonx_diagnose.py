#!/usr/bin/env python3
"""
Diagnostic script for IBM WatsonX MCP tool compatibility and streaming performance.
Tests tool name handling, streaming behavior, and WatsonX-specific features.
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

async def test_watsonx_setup():
    """Test WatsonX configuration and connectivity"""
    print("🧪 Testing WatsonX Setup & Configuration")
    print("=" * 50)
    
    # Check required environment variables
    watsonx_api_key = os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    watsonx_url = os.getenv("WATSONX_AI_URL")
    
    print("📋 Environment Check:")
    print(f"   WATSONX_API_KEY/IBM_CLOUD_API_KEY: {'✅ Set' if watsonx_api_key else '❌ Missing'}")
    print(f"   WATSONX_PROJECT_ID: {'✅ Set' if project_id else '❌ Missing'}")
    print(f"   WATSONX_AI_URL: {'✅ Set' if watsonx_url else '🔧 Using default'}")
    
    if not watsonx_api_key:
        print("\n❌ WatsonX API key not configured")
        print("💡 Set environment variable: export WATSONX_API_KEY=your-api-key")
        print("   Or: export IBM_CLOUD_API_KEY=your-ibm-cloud-api-key")
        return False
        
    if not project_id:
        print("\n❌ WatsonX Project ID not configured")
        print("💡 Set environment variable: export WATSONX_PROJECT_ID=your-project-id")
        return False
    
    try:
        from chuk_llm.llm.providers.watsonx_client import WatsonXLLMClient
        
        # Test client initialization
        client = WatsonXLLMClient(
            model="ibm/granite-3-3-8b-instruct",
            api_key=watsonx_api_key,
            project_id=project_id,
            watsonx_ai_url=watsonx_url
        )
        
        print(f"\n✅ Client initialized successfully:")
        print(f"   Model: {client.model}")
        print(f"   Project ID: {client.project_id}")
        print(f"   WatsonX URL: {client.watsonx_ai_url}")
        print(f"   Model Family: {client._detect_model_family()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing WatsonX client: {e}")
        print("💡 Check your credentials and project setup")
        return False

async def test_watsonx_tool_handling():
    """Test that WatsonX handles MCP-style tool names"""
    print("\n🧪 Testing WatsonX MCP Tool Name Handling")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.providers.watsonx_client import WatsonXLLMClient
        
        # Create client instance to test tool handling
        client = WatsonXLLMClient(model="ibm/granite-3-3-8b-instruct")
        
        # Test MCP-style tool names
        mcp_tools = [
            "stdio.read_query",
            "filesystem.read_file", 
            "mcp.server:get_data",
            "some.complex:tool.name",
            "already_valid_name",
        ]
        
        print("Testing WatsonX tool name handling:")
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
        
        # Test tool conversion to WatsonX format
        print("\nTesting tool conversion to WatsonX format:")
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
        
        sanitized = client._sanitize_tool_names(complex_tools)
        converted = client._convert_tools(sanitized)
        
        if converted:
            original = complex_tools[0]["function"]["name"]
            sanitized_name = sanitized[0]["function"]["name"] if sanitized else "ERROR"
            converted_name = converted[0]["function"]["name"]
            
            print(f"  Original: {original}")
            print(f"  Sanitized: {sanitized_name}")
            print(f"  WatsonX format: {converted_name}")
            print(f"  Mapping stored: {client._current_name_mapping}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool handling: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_watsonx_streaming():
    """Test WatsonX streaming performance and behavior"""
    print("\n🧪 Testing WatsonX Streaming Performance")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Provider name: {getattr(client, 'provider_name', 'unknown')}")
        
        messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it at least 100 words."}
        ]
        
        print("\n🔍 Testing WatsonX streaming=True...")
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
            
            print(f"\n\n📊 WATSONX STREAMING ANALYSIS:")
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
        
        print("\n🔍 Testing WatsonX streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing WatsonX streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_watsonx_mcp_tools():
    """Test WatsonX with actual MCP-style tool calls"""
    print("\n🧪 Testing WatsonX with MCP-Style Tools")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        
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
            }
        ]
        
        messages = [
            {
                "role": "user",
                "content": "Please list the files in the current directory and then ask the user for their name"
            }
        ]
        
        print("Testing WatsonX with MCP-style tool names...")
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
            print("\n💡 WatsonX may have stricter tool naming than expected")
            print("   Tool name sanitization may need enhancement")
            return False
        else:
            print(f"❌ Error testing MCP tools: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_watsonx_features():
    """Test WatsonX advanced features"""
    print("\n🧪 Testing WatsonX Advanced Features")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="watsonx", model="ibm/granite-3-3-8b-instruct")
        
        # Test model info
        print("📋 Model capabilities:")
        model_info = client.get_model_info()
        
        features = model_info.get("features", [])
        watsonx_info = model_info.get("watsonx_specific", {})
        
        print(f"   Features: {', '.join(features)}")
        print(f"   Max context: {model_info.get('max_context_length', 'unknown')}")
        print(f"   Max output: {model_info.get('max_output_tokens', 'unknown')}")
        print(f"   Vision support: {'✅' if 'vision' in features else '❌'}")
        print(f"   Tool support: {'✅' if 'tools' in features else '❌'}")
        print(f"   Streaming: {'✅' if 'streaming' in features else '❌'}")
        print(f"   Reasoning: {'✅' if 'reasoning' in features else '❌'}")
        
        print(f"\n🔷 WatsonX-specific:")
        print(f"   Model Family: {watsonx_info.get('model_family', 'unknown')}")
        print(f"   Project ID: {watsonx_info.get('project_id', 'unknown')}")
        print(f"   Enterprise Features: {watsonx_info.get('enterprise_features', False)}")
        print(f"   Tool Requirements: {model_info.get('tool_name_requirements', 'unknown')}")
        print(f"   MCP Compatibility: {model_info.get('mcp_compatibility', 'unknown')}")
        
        # Test with different model families
        model_families = [
            "ibm/granite-3-3-8b-instruct",
            "meta-llama/llama-3-3-70b-instruct",
            "mistralai/mistral-large-2"
        ]
        
        print(f"\n🧬 Testing model family detection:")
        for model_id in model_families:
            try:
                test_client = client.__class__(model=model_id)
                family = test_client._detect_model_family()
                print(f"   {model_id:<40} -> {family}")
            except Exception as e:
                print(f"   {model_id:<40} -> Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing features: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_watsonx_vs_competitors():
    """Compare WatsonX performance vs other providers"""
    print("\n🧪 WatsonX vs Competitors Performance")
    print("=" * 50)
    
    # Same test prompt for all providers
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence"}
    ]
    
    providers = [
        ("watsonx", "ibm/granite-3-3-8b-instruct"),
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
        
        # WatsonX-specific analysis
        if "watsonx" in valid_results:
            watsonx_result = valid_results["watsonx"]
            print(f"\n🎯 WATSONX ANALYSIS:")
            print(f"   Chunks: {watsonx_result['chunks']}")
            print(f"   First chunk: {watsonx_result['first_chunk']:.3f}s")
            print(f"   Content: {watsonx_result['content_length']} chars")
            
            # Compare to others
            other_providers = [k for k in valid_results.keys() if k != "watsonx"]
            if other_providers:
                avg_first_chunk = sum(valid_results[p]["first_chunk"] for p in other_providers) / len(other_providers)
                avg_chunks = sum(valid_results[p]["chunks"] for p in other_providers) / len(other_providers)
                
                if watsonx_result["first_chunk"] < avg_first_chunk:
                    print(f"   ✅ WatsonX faster than average first chunk by {(avg_first_chunk - watsonx_result['first_chunk'])*1000:.0f}ms")
                else:
                    print(f"   ⚠️  WatsonX slower than average first chunk by {(watsonx_result['first_chunk'] - avg_first_chunk)*1000:.0f}ms")
                
                if watsonx_result["chunks"] > avg_chunks:
                    print(f"   ✅ WatsonX more granular than average ({watsonx_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
                else:
                    print(f"   ⚠️  WatsonX less granular than average ({watsonx_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
    
    return len(valid_results) > 0

async def main():
    """Run all WatsonX diagnostic tests"""
    print("🚀 Testing WatsonX MCP Tool Compatibility & Performance")
    print("=" * 60)
    
    # Test 1: Setup and configuration
    test1_passed = await test_watsonx_setup()
    
    # Test 2: Tool name handling
    test2_passed = await test_watsonx_tool_handling() if test1_passed else False
    
    # Test 3: Streaming performance
    test3_passed = await test_watsonx_streaming() if test2_passed else False
    
    # Test 4: MCP tools integration
    test4_passed = await test_watsonx_mcp_tools() if test3_passed else False
    
    # Test 5: Advanced features
    test5_passed = await test_watsonx_features() if test4_passed else False
    
    # Test 6: Performance comparison
    test6_passed = await test_watsonx_vs_competitors() if test5_passed else False
    
    print("\n" + "=" * 60)
    print("🎯 WATSONX DIAGNOSTIC RESULTS:")
    print(f"   Setup & Configuration: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Tool Name Handling: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Streaming Performance: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"   MCP Tools Integration: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"   Advanced Features: {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"   Performance Comparison: {'✅ PASS' if test6_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed, test6_passed]):
        print("\n🎉 ALL WATSONX TESTS PASSED!")
        print("💡 WatsonX is ready for MCP CLI:")
        print("   mcp-cli chat --provider watsonx --model ibm/granite-3-3-8b-instruct")
        print("\n🔑 Key Advantages of WatsonX:")
        print("   • Enterprise-grade security and governance")
        print("   • IBM Granite models with strong reasoning")
        print("   • On-premises deployment options")
        print("   • Tool name sanitization for MCP compatibility")
    else:
        print("\n❌ Some WatsonX tests failed.")
        print("💡 Check the implementation and ensure:")
        print("   1. WATSONX_API_KEY and WATSONX_PROJECT_ID are set")
        print("   2. Project has proper model access")
        print("   3. API credentials are valid")
        if not test4_passed:
            print("   4. Tool name sanitization may need enhancement")


if __name__ == "__main__":
    asyncio.run(main())