#!/usr/bin/env python3
"""
Diagnostic script for Azure OpenAI universal tool compatibility and streaming performance.
Tests tool name handling, streaming behavior, and feature capabilities with the ToolCompatibilityMixin.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


async def test_azure_openai_tool_handling():
    """Test that Azure OpenAI handles universal tool names correctly with ToolCompatibilityMixin"""
    print("🧪 Testing Azure OpenAI Universal Tool Name Handling")
    print("=" * 60)

    try:
        from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

        # Create client instance to test tool handling
        client = AzureOpenAILLMClient(model="gpt-4o-mini")

        # Verify it has the ToolCompatibilityMixin
        from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

        has_mixin = isinstance(client, ToolCompatibilityMixin)
        print(f"✅ Has ToolCompatibilityMixin: {has_mixin}")

        if has_mixin:
            # Get compatibility info
            compatibility_info = client.get_tool_compatibility_info()
            print(f"Provider: {compatibility_info['provider']}")
            print(f"Compatibility level: {compatibility_info['compatibility_level']}")
            print(
                f"Requires sanitization: {compatibility_info['requires_sanitization']}"
            )
            print(f"Max name length: {compatibility_info['max_tool_name_length']}")
            print(f"Tool name pattern: {compatibility_info['tool_name_requirements']}")

            # Show sample transformations
            print("\nSample transformations from ToolCompatibilityMixin:")
            sample_transformations = compatibility_info.get(
                "sample_transformations", {}
            )
            for original, transformed in sample_transformations.items():
                status = "✅ PRESERVED" if original == transformed else "🔧 SANITIZED"
                print(f"  {original:<30} -> {transformed:<30} {status}")

        # Test universal tool naming patterns
        test_cases = [
            "stdio.read_query",
            "filesystem.read_file",
            "mcp.server:get_data",
            "web.api:search",
            "database.sql.execute",
            "service:method",
            "namespace:function",
            "complex.tool:method.v1",
            "already_valid_name",
            "tool-with-dashes",
            "tool_with_underscores",
        ]

        print("\nTesting individual tool name processing:")
        for original in test_cases:
            test_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": original,
                        "description": f"Test tool: {original}",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

            # Use the universal sanitization
            sanitized_tools = client._sanitize_tool_names(test_tools)
            sanitized_name = (
                sanitized_tools[0]["function"]["name"] if sanitized_tools else "ERROR"
            )

            # Check if mapping was created
            mapping = getattr(client, "_current_name_mapping", {})
            has_mapping = sanitized_name in mapping

            if original == sanitized_name:
                status = "✅ PRESERVED"
            else:
                status = "🔧 SANITIZED" if has_mapping else "❌ CHANGED"

            print(f"  {original:<30} -> {sanitized_name:<30} {status}")

            # Test restoration if mapping exists
            if has_mapping and sanitized_name in mapping:
                restored_name = mapping[sanitized_name]
                restoration_status = (
                    "✅ RESTORED"
                    if restored_name == original
                    else "❌ RESTORATION FAILED"
                )
                print(
                    f"    Mapping: {sanitized_name} -> {restored_name} {restoration_status}"
                )

        # Test validation
        print("\nTesting tool name validation:")
        validation_tests = [
            ("valid_tool", True),
            ("another-tool", True),
            ("simple123", True),
            ("stdio.read_query", None),  # Depends on Azure OpenAI's requirements
            ("tool:with:colons", None),  # Depends on Azure OpenAI's requirements
            ("tool with spaces", False),
        ]

        for name, expected_valid in validation_tests:
            is_valid, issues = client.validate_tool_names(
                [
                    {
                        "type": "function",
                        "function": {"name": name, "description": "Test"},
                    }
                ]
            )

            if expected_valid is None:
                # Don't check expectation for Azure OpenAI flexible names
                status = "ℹ️"
                print(
                    f"  {name:<20}: {'Valid' if is_valid else 'Invalid'} {status} (provider-dependent)"
                )
            else:
                status = "✅" if is_valid == expected_valid else "❌"
                print(f"  {name:<20}: {'Valid' if is_valid else 'Invalid'} {status}")

            if issues:
                for issue in issues[:1]:  # Show first issue only
                    print(f"    Issue: {issue}")

        return True

    except Exception as e:
        print(f"❌ Error testing tool handling: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_azure_openai_streaming():
    """Test Azure OpenAI streaming performance and behavior"""
    print("\n🧪 Testing Azure OpenAI Streaming Performance")
    print("=" * 60)

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(provider="azure_openai", model="gpt-4o-mini")

        print(f"Client type: {type(client)}")
        print(f"Client class: {client.__class__.__name__}")
        print(f"Provider name: {getattr(client, 'provider_name', 'unknown')}")
        print(f"Azure deployment: {getattr(client, 'azure_deployment', 'unknown')}")

        # Check if client has the ToolCompatibilityMixin
        from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

        has_tool_mixin = isinstance(client, ToolCompatibilityMixin)
        print(f"Has ToolCompatibilityMixin: {'✅' if has_tool_mixin else '❌'}")

        messages = [
            {
                "role": "user",
                "content": "Write a short story about a robot learning to paint. Make it at least 100 words and tell it slowly.",
            }
        ]

        print("\n🔍 Testing Azure OpenAI streaming=True...")
        start_time = time.time()

        # Test streaming
        response = client.create_completion(messages, stream=True)

        print(f"⏱️  Response type: {type(response)}")
        print(f"⏱️  Has __aiter__: {hasattr(response, '__aiter__')}")

        if hasattr(response, "__aiter__"):
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
                    print(
                        f"\n   Chunk {chunk_count}: {relative_time:.3f}s (interval: {interval:.4f}s)"
                    )
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

            print("\n\n📊 AZURE OPENAI STREAMING ANALYSIS:")
            print(f"   Total chunks: {chunk_count}")
            print(
                f"   First chunk delay: {first_chunk_time:.3f}s"
                if first_chunk_time
                else "   No chunks received"
            )
            print(f"   Total time: {end_time:.3f}s")
            print(
                f"   Streaming duration: {end_time - (first_chunk_time or end_time):.3f}s"
            )
            print(f"   Response length: {len(full_response)} characters")
            print(
                f"   Avg chunk interval: {avg_interval * 1000:.1f}ms"
                if chunk_intervals
                else "   No intervals"
            )
            print(
                f"   Min interval: {min_interval * 1000:.1f}ms"
                if chunk_intervals
                else "   No intervals"
            )
            print(
                f"   Max interval: {max_interval * 1000:.1f}ms"
                if chunk_intervals
                else "   No intervals"
            )

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


async def test_azure_openai_universal_tools():
    """Test Azure OpenAI with universal tool naming and bidirectional mapping"""
    print("\n🧪 Testing Azure OpenAI Universal Tool Compatibility")
    print("=" * 60)

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(provider="azure_openai", model="gpt-4o-mini")

        # Universal tool names that may require sanitization depending on Azure OpenAI configuration
        universal_tools = [
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
                                "description": "The prompt to display",
                            }
                        },
                        "required": ["prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web.api:search",
                    "description": "Search the web using an API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "database.sql.execute",
                    "description": "Execute a SQL query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query to execute",
                            }
                        },
                        "required": ["sql"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "user",
                "content": "Please search the web for 'AI news', read user input for their name, and then execute a simple SQL query",
            }
        ]

        print("Testing Azure OpenAI with universal tool names...")
        original_names = [t["function"]["name"] for t in universal_tools]
        print(f"Original tool names: {original_names}")

        # Test non-streaming first
        response = await client.create_completion(
            messages=messages, tools=universal_tools, stream=False
        )

        print("✅ SUCCESS: No tool naming errors with universal naming!")

        if isinstance(response, dict):
            if response.get("tool_calls"):
                print(f"🔧 Tool calls made: {len(response['tool_calls'])}")
                for i, tool_call in enumerate(response["tool_calls"]):
                    func_name = tool_call.get("function", {}).get("name", "unknown")
                    print(f"   {i + 1}. {func_name}")

                    # Verify original names are restored in response
                    if func_name in original_names:
                        print(f"      ✅ Original name restored: {func_name}")
                    else:
                        print(f"      ⚠️  Unexpected name in response: {func_name}")
                        print(f"         (Should be one of: {original_names})")

            elif response.get("response"):
                print(f"💬 Text response: {response['response'][:150]}...")
            else:
                print("❓ Unexpected response format")

        # Test streaming with universal tools
        print("\n🔄 Testing streaming with universal tools...")
        stream_response = client.create_completion(
            messages=messages, tools=universal_tools, stream=True
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

                    # Verify restored names in streaming
                    if tool_name in original_names:
                        print(f"      ✅ Correctly restored in stream: {tool_name}")
                    else:
                        print(f"      ⚠️  Unexpected name in stream: {tool_name}")

            if chunk_count >= 15:  # Limit for testing
                break

        print(
            f"✅ Streaming completed: {chunk_count} chunks, {len(tool_calls_found)} tool calls"
        )

        # Verify bidirectional mapping worked
        unique_tool_calls = list(set(tool_calls_found))
        if unique_tool_calls:
            print("🔄 Bidirectional mapping test:")
            for tool_name in unique_tool_calls:
                if tool_name in original_names:
                    print(f"   ✅ {tool_name} - correctly restored from sanitized form")
                else:
                    print(
                        f"   ❌ {tool_name} - not in original names, mapping may have failed"
                    )

        return True

    except Exception as e:
        print(f"❌ Error testing universal tools: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_azure_openai_features():
    """Test Azure OpenAI advanced features with updated expectations"""
    print("\n🧪 Testing Azure OpenAI Advanced Features")
    print("=" * 60)

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(provider="azure_openai", model="gpt-4o-mini")

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

        # Test Azure-specific info
        azure_info = model_info.get("azure_specific", {})
        if azure_info:
            print(f"   Azure endpoint: {azure_info.get('endpoint', 'unknown')}")
            print(f"   Azure deployment: {azure_info.get('deployment', 'unknown')}")
            print(f"   API version: {azure_info.get('api_version', 'unknown')}")

        # Test tool compatibility info
        if hasattr(client, "get_tool_compatibility_info"):
            print("\n🔧 Tool compatibility:")
            tool_info = client.get_tool_compatibility_info()
            print(
                f"   Compatibility level: {tool_info.get('compatibility_level', 'unknown')}"
            )
            print(
                f"   Requires sanitization: {tool_info.get('requires_sanitization', 'unknown')}"
            )
            print(
                f"   Max name length: {tool_info.get('max_tool_name_length', 'unknown')}"
            )
            print(
                f"   Forbidden chars: {len(tool_info.get('forbidden_characters', []))} characters"
            )

        # Test JSON mode if supported
        if "json_mode" in features:
            print("\n📊 Testing JSON mode...")
            json_messages = [
                {
                    "role": "user",
                    "content": "Return a JSON object with your name and capabilities",
                }
            ]

            json_response = await client.create_completion(
                json_messages, stream=False, response_format={"type": "json_object"}
            )

            if isinstance(json_response, dict) and json_response.get("response"):
                try:
                    import json

                    json.loads(json_response["response"])
                    print(
                        f"   ✅ JSON mode works: {json_response['response'][:100]}..."
                    )
                except json.JSONDecodeError:
                    print(
                        f"   ⚠️  Response not valid JSON: {json_response['response'][:100]}..."
                    )
            else:
                print(f"   ❌ JSON mode test failed: {json_response}")

        return True

    except Exception as e:
        print(f"❌ Error testing features: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_azure_openai_vs_competitors():
    """Compare Azure OpenAI performance vs other providers with universal tools"""
    print("\n🧪 Azure OpenAI vs Competitors Performance (Universal Tools)")
    print("=" * 70)

    # Same test prompt for all providers
    messages = [
        {"role": "user", "content": "Write a haiku about artificial intelligence"}
    ]

    # Test with a universal tool that requires sanitization
    universal_tools = [
        {
            "type": "function",
            "function": {
                "name": "poetry.generator:create",
                "description": "Generate creative poetry",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "style": {"type": "string", "description": "Poetry style"},
                        "topic": {"type": "string", "description": "Poetry topic"},
                    },
                },
            },
        }
    ]

    providers = [
        ("azure_openai", "gpt-4o-mini"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("mistral", "mistral-medium-2505"),
        ("openai", "gpt-4o-mini"),
    ]

    results = {}

    for provider, model in providers:
        print(f"\n🔍 Testing {provider} with {model} and universal tools...")

        try:
            from chuk_llm.llm.client import get_client

            client = get_client(provider=provider, model=model)

            # Test tool compatibility
            has_tool_mixin = hasattr(client, "get_tool_compatibility_info")
            if has_tool_mixin:
                tool_info = client.get_tool_compatibility_info()
                print(
                    f"   Tool compatibility: {tool_info.get('compatibility_level', 'unknown')}"
                )
                print(
                    f"   Requires sanitization: {tool_info.get('requires_sanitization', 'unknown')}"
                )

            start_time = time.time()
            response = client.create_completion(
                messages, tools=universal_tools, stream=True
            )

            chunk_count = 0
            first_chunk_time = None
            content_length = 0
            tool_calls_found = 0

            async for chunk in response:
                current_time = time.time() - start_time

                if first_chunk_time is None:
                    first_chunk_time = current_time

                chunk_count += 1

                if isinstance(chunk, dict):
                    if chunk.get("response"):
                        content_length += len(chunk["response"])
                    if chunk.get("tool_calls"):
                        tool_calls_found += len(chunk["tool_calls"])

                # Limit for comparison
                if chunk_count >= 20:
                    break

            total_time = time.time() - start_time

            results[provider] = {
                "chunks": chunk_count,
                "first_chunk": first_chunk_time,
                "total_time": total_time,
                "content_length": content_length,
                "tool_calls": tool_calls_found,
                "has_universal_tools": has_tool_mixin,
            }

            print(
                f"   {provider}: {chunk_count} chunks, first at {first_chunk_time:.3f}s, "
                f"{content_length} chars, {tool_calls_found} tools, total {total_time:.3f}s"
            )

        except Exception as e:
            print(f"   {provider}: Error - {e}")
            results[provider] = None

    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    valid_results = {k: v for k, v in results.items() if v is not None}

    if len(valid_results) >= 2:
        fastest_first = min(
            valid_results.keys(), key=lambda k: valid_results[k]["first_chunk"]
        )
        most_chunks = max(
            valid_results.keys(), key=lambda k: valid_results[k]["chunks"]
        )
        fastest_total = min(
            valid_results.keys(), key=lambda k: valid_results[k]["total_time"]
        )

        print(
            f"   🚀 Fastest first chunk: {fastest_first} ({valid_results[fastest_first]['first_chunk']:.3f}s)"
        )
        print(
            f"   📊 Most granular streaming: {most_chunks} ({valid_results[most_chunks]['chunks']} chunks)"
        )
        print(
            f"   ⚡ Fastest total time: {fastest_total} ({valid_results[fastest_total]['total_time']:.3f}s)"
        )

        # Universal tool compatibility analysis
        universal_providers = [
            k for k, v in valid_results.items() if v.get("has_universal_tools")
        ]
        print(
            f"   🔧 Universal tool support: {len(universal_providers)}/{len(valid_results)} providers"
        )
        for provider in universal_providers:
            print(f"      ✅ {provider}")

        # Azure OpenAI-specific analysis
        if "azure_openai" in valid_results:
            azure_result = valid_results["azure_openai"]
            print("\n🎯 AZURE OPENAI ANALYSIS:")
            print(f"   Chunks: {azure_result['chunks']}")
            print(f"   First chunk: {azure_result['first_chunk']:.3f}s")
            print(f"   Content: {azure_result['content_length']} chars")
            print(f"   Tool calls: {azure_result['tool_calls']}")
            print(
                f"   Universal tools: {'✅' if azure_result['has_universal_tools'] else '❌'}"
            )

            # Compare to others
            other_providers = [k for k in valid_results if k != "azure_openai"]
            if other_providers:
                avg_first_chunk = sum(
                    valid_results[p]["first_chunk"] for p in other_providers
                ) / len(other_providers)
                avg_chunks = sum(
                    valid_results[p]["chunks"] for p in other_providers
                ) / len(other_providers)

                if azure_result["first_chunk"] < avg_first_chunk:
                    print(
                        f"   ✅ Azure OpenAI faster than average first chunk by {(avg_first_chunk - azure_result['first_chunk']) * 1000:.0f}ms"
                    )
                else:
                    print(
                        f"   ⚠️  Azure OpenAI slower than average first chunk by {(azure_result['first_chunk'] - avg_first_chunk) * 1000:.0f}ms"
                    )

                if azure_result["chunks"] > avg_chunks:
                    print(
                        f"   ✅ Azure OpenAI more granular than average ({azure_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)"
                    )
                else:
                    print(
                        f"   ⚠️  Azure OpenAI less granular than average ({azure_result['chunks']:.1f} vs {avg_chunks:.1f} chunks)"
                    )

    return len(valid_results) > 0


async def main():
    """Run all Azure OpenAI diagnostic tests"""
    print("🚀 Testing Azure OpenAI Universal Tool Compatibility & Performance")
    print("=" * 80)

    # Test 1: Universal tool name handling
    test1_passed = await test_azure_openai_tool_handling()

    # Test 2: Streaming performance
    test2_passed = await test_azure_openai_streaming() if test1_passed else False

    # Test 3: Universal tools integration
    test3_passed = await test_azure_openai_universal_tools() if test2_passed else False

    # Test 4: Advanced features
    test4_passed = await test_azure_openai_features() if test3_passed else False

    # Test 5: Performance comparison
    test5_passed = await test_azure_openai_vs_competitors() if test4_passed else False

    print("\n" + "=" * 80)
    print("🎯 AZURE OPENAI DIAGNOSTIC RESULTS:")
    print(f"   Universal Tool Handling: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Streaming Performance: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Universal Tools Integration: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"   Advanced Features: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"   Performance Comparison: {'✅ PASS' if test5_passed else '❌ FAIL'}")

    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed]):
        print("\n🎉 ALL AZURE OPENAI TESTS PASSED!")
        print("💡 Azure OpenAI is ready for MCP CLI with universal tool compatibility:")
        print("   mcp-cli chat --provider azure_openai --model gpt-4o-mini")
        print("\n🔑 Key Advantages of Updated Azure OpenAI:")
        print("   • Universal tool name compatibility with bidirectional mapping")
        print("   • Consistent sanitization behavior across all providers")
        print("   • Azure-specific features (deployment mapping, versioning)")
        print("   • Enterprise authentication support")
        print("   • High-quality responses with OpenAI models")
        print("   • Real async streaming")
        print("\n🔧 Tool Name Examples:")
        print("   stdio.read_query -> sanitized if needed, restored in response")
        print("   web.api:search -> sanitized if needed, restored in response")
        print("   database.sql.execute -> sanitized if needed, restored in response")
    else:
        print("\n❌ Some Azure OpenAI tests failed.")
        print("💡 Check the implementation and ensure:")
        print("   1. azure_openai_client.py inherits from ToolCompatibilityMixin")
        print("   2. ToolCompatibilityMixin is properly initialized")
        print("   3. Azure OpenAI credentials are configured correctly")
        print("   4. Azure endpoint and deployment are set correctly")
        print("   5. Tool name sanitization and restoration is working")


if __name__ == "__main__":
    asyncio.run(main())
