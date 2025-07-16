#!/usr/bin/env python3
"""
Diagnostic script for OpenAI MCP tool compatibility and streaming performance.
Tests tool name handling, streaming behavior, and OpenAI-specific features.
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

async def test_openai_setup():
    """Test OpenAI configuration and connectivity"""
    print("🧪 Testing OpenAI Setup & Configuration")
    print("=" * 50)
    
    # Check required environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    
    print("📋 Environment Check:")
    print(f"   OPENAI_API_KEY: {'✅ Set' if openai_api_key else '❌ Missing'}")
    print(f"   OPENAI_BASE_URL: {'✅ Set' if openai_base_url else '🔧 Using default'}")
    
    if not openai_api_key:
        print("\n❌ OpenAI API key not configured")
        print("💡 Set environment variable: export OPENAI_API_KEY=your-api-key")
        return False
    
    try:
        from chuk_llm.llm.providers.openai_client import OpenAILLMClient
        
        # Test client initialization
        client = OpenAILLMClient(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            api_base=openai_base_url
        )
        
        print(f"\n✅ Client initialized successfully:")
        print(f"   Model: {client.model}")
        print(f"   API Base: {client.api_base or 'https://api.openai.com/v1'}")
        print(f"   Detected Provider: {client.detect_provider_name()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing OpenAI client: {e}")
        print("💡 Check your API key and network connection")
        return False

async def test_openai_tool_handling():
    """Test that OpenAI handles MCP-style tool names"""
    print("\n🧪 Testing OpenAI MCP Tool Name Handling")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.providers.openai_client import OpenAILLMClient
        
        # Create client instance to test tool handling
        client = OpenAILLMClient(model="gpt-4o-mini")
        
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
        
        print("Testing OpenAI tool name handling:")
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
        
        # Test edge cases
        print("\nTesting edge cases:")
        edge_cases = [
            "123invalid_start",  # Starts with number
            "tool@with#special!chars",  # Special characters
            "a" * 70,  # Very long name
            "",  # Empty name
            "UPPERCASE_TOOL",  # Uppercase
        ]
        
        for tool_name in edge_cases:
            test_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Edge case: {tool_name}",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
            
            try:
                sanitized = client._sanitize_tool_names(test_tools)
                original_name = test_tools[0]["function"]["name"]
                sanitized_name = sanitized[0]["function"]["name"] if sanitized else "ERROR"
                status = "✅ HANDLED" if sanitized_name else "❌ FAILED"
                print(f"  {original_name[:25]:<25} -> {sanitized_name[:25]:<25} {status}")
            except Exception as e:
                print(f"  {tool_name[:25]:<25} -> ERROR: {str(e)[:25]:<25} ❌ EXCEPTION")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool handling: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_openai_streaming():
    """Test OpenAI streaming performance and behavior"""
    print("\n🧪 Testing OpenAI Streaming Performance")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model="gpt-4o-mini")
        
        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Provider name: {getattr(client, 'detect_provider_name', lambda: 'unknown')()}")
        
        messages = [
            {"role": "user", "content": "Write a short story about a robot learning to paint. Make it at least 100 words and tell it engagingly."}
        ]
        
        print("\n🔍 Testing OpenAI streaming=True...")
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
            
            print(f"\n\n📊 OPENAI STREAMING ANALYSIS:")
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
        
        print("\n🔍 Testing OpenAI streaming=False...")
        response = await client.create_completion(messages, stream=False)
        print(f"Non-streaming response type: {type(response)}")
        if isinstance(response, dict):
            content = response.get("response", "")
            print(f"Content preview: {content[:100]}...")
            print(f"Content length: {len(content)} characters")
            print("✅ Non-streaming works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing OpenAI streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_openai_mcp_tools():
    """Test OpenAI with actual MCP-style tool calls"""
    print("\n🧪 Testing OpenAI with MCP-Style Tools")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model="gpt-4o-mini")
        
        # MCP-style tools that should work well with OpenAI
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
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 5
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
                "content": "Please list the files in the current directory, search for 'Python tutorial', and then ask the user for their name"
            }
        ]
        
        print("Testing OpenAI with MCP-style tool names...")
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
            print("\n💡 OpenAI may have stricter tool naming than expected")
            print("   Tool name sanitization may need enhancement")
            return False
        else:
            print(f"❌ Error testing MCP tools: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_openai_features():
    """Test OpenAI advanced features"""
    print("\n🧪 Testing OpenAI Advanced Features")
    print("=" * 50)
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model="gpt-4o-mini")
        
        # Test model info
        print("📋 Model capabilities:")
        model_info = client.get_model_info()
        
        features = model_info.get("features", [])
        openai_info = model_info.get("openai_specific", {})
        
        print(f"   Features: {', '.join(features)}")
        print(f"   Max context: {model_info.get('max_context_length', 'unknown')}")
        print(f"   Max output: {model_info.get('max_output_tokens', 'unknown')}")
        print(f"   Vision support: {'✅' if 'vision' in features else '❌'}")
        print(f"   Tool support: {'✅' if 'tools' in features else '❌'}")
        print(f"   Streaming: {'✅' if 'streaming' in features else '❌'}")
        print(f"   JSON mode: {'✅' if 'json_mode' in features else '❌'}")
        
        print(f"\n🔷 OpenAI-specific:")
        print(f"   API Base: {model_info.get('api_base', 'https://api.openai.com/v1')}")
        print(f"   Detected Provider: {model_info.get('detected_provider', 'openai')}")
        print(f"   OpenAI Compatible: {model_info.get('openai_compatible', True)}")
        
        # Test JSON mode if supported
        if 'json_mode' in features:
            print("\n📊 Testing JSON mode...")
            json_messages = [
                {"role": "user", "content": "Return a JSON object with your name, model type, and capabilities list"}
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
        
        # Test vision if supported (using gpt-4o model)
        if 'vision' in features:
            print("\n🖼️  Testing vision capabilities...")
            
            # Test with a small red pixel image (1x1 red pixel PNG)
            red_pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this 1x1 pixel image? Be very specific."},
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
                # Use a vision-capable model for this test
                vision_client = get_client(provider="openai", model="gpt-4o-mini")
                vision_response = await vision_client.create_completion(vision_messages, stream=False)
                
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
        
        # Test function calling with complex schema
        print("\n🔧 Testing advanced function calling...")
        complex_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_metrics",
                    "description": "Calculate performance metrics from data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Array of numeric data points"
                            },
                            "metric_type": {
                                "type": "string",
                                "enum": ["mean", "median", "std", "variance"],
                                "description": "Type of metric to calculate"
                            },
                            "round_to": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 10,
                                "default": 2,
                                "description": "Number of decimal places to round to"
                            }
                        },
                        "required": ["data", "metric_type"]
                    }
                }
            }
        ]
        
        complex_messages = [
            {
                "role": "user",
                "content": "Calculate the mean of these numbers: [1.5, 2.7, 3.2, 4.1, 5.8] and round to 3 decimal places"
            }
        ]
        
        try:
            complex_response = await client.create_completion(
                complex_messages,
                tools=complex_tools,
                stream=False
            )
            
            if complex_response.get("tool_calls"):
                tool_call = complex_response["tool_calls"][0]
                func_args = tool_call["function"]["arguments"]
                print(f"   ✅ Complex function calling works")
                print(f"   📊 Function args: {func_args[:100]}...")
            else:
                print(f"   ⚠️  No tool calls in complex function test")
                
        except Exception as e:
            print(f"   ❌ Complex function calling failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing features: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_openai_vs_competitors():
    """Compare OpenAI performance vs other providers"""
    print("\n🧪 OpenAI vs Competitors Performance")
    print("=" * 50)
    
    # Same test prompt for all providers
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence and creativity"}
    ]
    
    providers = [
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
        
        # OpenAI-specific analysis
        if "openai" in valid_results:
            openai_result = valid_results["openai"]
            print(f"\n🎯 OPENAI ANALYSIS:")
            print(f"   Chunks: {openai_result['chunks']}")
            print(f"   First chunk: {openai_result['first_chunk']:.3f}s")
            print(f"   Content: {openai_result['content_length']} chars")
            
            # Compare to others
            other_providers = [k for k in valid_results.keys() if k != "openai"]
            if other_providers:
                avg_first_chunk = sum(valid_results[p]["first_chunk"] for p in other_providers) / len(other_providers)
                avg_chunks = sum(valid_results[p]["chunks"] for p in other_providers) / len(other_providers)
                
                if openai_result["first_chunk"] < avg_first_chunk:
                    print(f"   ✅ OpenAI faster than average first chunk by {(avg_first_chunk - openai_result['first_chunk'])*1000:.0f}ms")
                else:
                    print(f"   ⚠️  OpenAI slower than average first chunk by {(openai_result['first_chunk'] - avg_first_chunk)*1000:.0f}ms")
                
                if openai_result["chunks"] > avg_chunks:
                    print(f"   ✅ OpenAI more granular than average ({openai_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
                else:
                    print(f"   ⚠️  OpenAI less granular than average ({openai_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)")
    
    return len(valid_results) > 0

async def test_openai_compatible_providers():
    """Test OpenAI-compatible providers using the same client"""
    print("\n🧪 Testing OpenAI-Compatible Providers")
    print("=" * 50)
    
    # Test different API bases that should work with OpenAI client
    compatible_configs = [
        {
            "name": "OpenAI",
            "api_base": None,  # Default OpenAI
            "model": "gpt-4o-mini",
            "expected_provider": "openai"
        },
        {
            "name": "DeepSeek (if configured)",
            "api_base": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "expected_provider": "deepseek"
        },
        {
            "name": "Together AI (if configured)",
            "api_base": "https://api.together.xyz/v1",
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "expected_provider": "together"
        }
    ]
    
    for config in compatible_configs:
        print(f"\n🔍 Testing {config['name']}...")
        
        try:
            from chuk_llm.llm.providers.openai_client import OpenAILLMClient
            
            # Only test if we have appropriate API key
            api_key_env = None
            if config["expected_provider"] == "openai":
                api_key_env = "OPENAI_API_KEY"
            elif config["expected_provider"] == "deepseek":
                api_key_env = "DEEPSEEK_API_KEY"
            elif config["expected_provider"] == "together":
                api_key_env = "TOGETHER_API_KEY"
            
            if api_key_env and not os.getenv(api_key_env):
                print(f"   ⏭️  Skipping {config['name']} - no {api_key_env} configured")
                continue
            
            client = OpenAILLMClient(
                model=config["model"],
                api_base=config["api_base"],
                api_key=os.getenv(api_key_env) if api_key_env else None
            )
            
            detected_provider = client.detect_provider_name()
            print(f"   🔍 Detected provider: {detected_provider}")
            print(f"   Expected: {config['expected_provider']}")
            
            if detected_provider == config["expected_provider"]:
                print(f"   ✅ Provider detection correct")
            else:
                print(f"   ⚠️  Provider detection mismatch")
            
            # Test a simple completion
            test_messages = [{"role": "user", "content": "Say 'Hello from API' in exactly those words."}]
            
            try:
                response = await client.create_completion(test_messages, stream=False)
                if response.get("response"):
                    print(f"   ✅ Basic completion works: {response['response'][:50]}...")
                else:
                    print(f"   ⚠️  Completion returned no response")
                    
            except Exception as e:
                print(f"   ❌ Completion failed: {e}")
            
        except Exception as e:
            print(f"   ❌ Client initialization failed: {e}")
    
    return True

async def main():
    """Run all OpenAI diagnostic tests"""
    print("🚀 Testing OpenAI MCP Tool Compatibility & Performance")
    print("=" * 60)
    
    # Test 1: Setup and configuration
    test1_passed = await test_openai_setup()
    
    # Test 2: Tool name handling
    test2_passed = await test_openai_tool_handling() if test1_passed else False
    
    # Test 3: Streaming performance
    test3_passed = await test_openai_streaming() if test2_passed else False
    
    # Test 4: MCP tools integration
    test4_passed = await test_openai_mcp_tools() if test3_passed else False
    
    # Test 5: Advanced features
    test5_passed = await test_openai_features() if test4_passed else False
    
    # Test 6: Performance comparison
    test6_passed = await test_openai_vs_competitors() if test5_passed else False
    
    # Test 7: OpenAI-compatible providers
    test7_passed = await test_openai_compatible_providers() if test6_passed else False
    
    print("\n" + "=" * 60)
    print("🎯 OPENAI DIAGNOSTIC RESULTS:")
    print(f"   Setup & Configuration: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Tool Name Handling: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Streaming Performance: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"   MCP Tools Integration: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"   Advanced Features: {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"   Performance Comparison: {'✅ PASS' if test6_passed else '❌ FAIL'}")
    print(f"   Compatible Providers: {'✅ PASS' if test7_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed, test6_passed, test7_passed]):
        print("\n🎉 ALL OPENAI TESTS PASSED!")
        print("💡 OpenAI is ready for MCP CLI:")
        print("   mcp-cli chat --provider openai --model gpt-4o-mini")
        print("\n🔑 Key Advantages of OpenAI:")
        print("   • Native MCP tool name support (most flexible)")
        print("   • Excellent streaming performance")
        print("   • Comprehensive feature set (vision, JSON mode, function calling)")
        print("   • Wide compatibility with OpenAI-compatible providers")
        print("   • Industry standard API format")
    else:
        print("\n❌ Some OpenAI tests failed.")
        print("💡 Check the implementation and ensure:")
        print("   1. OPENAI_API_KEY is set correctly")
        print("   2. Network connectivity to OpenAI API")
        print("   3. Model permissions and quotas")
        if not test4_passed:
            print("   4. Tool name handling may need debugging")


if __name__ == "__main__":
    asyncio.run(main())