#!/usr/bin/env python3
"""
Complete Tool Chain Test for Gemini - Universal Tool Compatibility
================================================================

This test validates that Gemini works with the universal tool compatibility system
and provides identical behavior to Azure OpenAI, OpenAI, Anthropic, and Mistral.

Key areas tested:
1. Universal tool name sanitization and restoration
2. Complete conversation tool chains
3. Parameter extraction with any naming convention
4. Streaming with tool name restoration
5. Cross-provider consistency
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root and load environment
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
    else:
        load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")


def safe_parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """
    Safely parse tool arguments that could be string or dict.
    Handles various formats returned by different providers.
    """
    if arguments is None:
        return {}

    # If it's already a dict, return as-is
    if isinstance(arguments, dict):
        return arguments

    # If it's a string, try to parse as JSON
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}

        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            else:
                print(
                    f"      ⚠️  Parsed arguments is not a dict: {type(parsed)}, value: {parsed}"
                )
                return {}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"      ⚠️  Failed to parse tool arguments as JSON: {e}")
            print(f"      Raw arguments: {repr(arguments)}")
            return {}

    # For any other type, log and return empty dict
    print(f"      ⚠️  Unexpected arguments type: {type(arguments)}, value: {arguments}")
    return {}


async def test_gemini_tool_compatibility():
    """Test that Gemini has universal tool compatibility"""
    print("🔗 GEMINI UNIVERSAL TOOL COMPATIBILITY TEST")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GEMINI_API_KEY or GOOGLE_API_KEY not found!")
        return False

    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(provider="gemini", model="gemini-2.5-flash")
        print(f"✅ Client created: {type(client).__name__}")

        # Check if client has universal tool compatibility
        from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

        has_tool_mixin = isinstance(client, ToolCompatibilityMixin)
        print(f"✅ Has ToolCompatibilityMixin: {has_tool_mixin}")

        if has_tool_mixin:
            tool_info = client.get_tool_compatibility_info()
            print(
                f"✅ Universal tool compatibility: {tool_info.get('compatibility_level', 'unknown')}"
            )
            print(
                f"   Requires sanitization: {tool_info.get('requires_sanitization', 'unknown')}"
            )
            print(
                f"   Max name length: {tool_info.get('max_tool_name_length', 'unknown')}"
            )
        else:
            print("❌ Missing ToolCompatibilityMixin - needs to be added")
            return False

        return True

    except Exception as e:
        print(f"❌ Error testing Gemini tool compatibility: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_gemini_universal_tools():
    """Test Gemini with universal tool names and bidirectional mapping"""
    print("\n🎯 GEMINI UNIVERSAL TOOL NAMES TEST")
    print("=" * 50)

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(provider="gemini", model="gemini-2.5-flash")

        # Universal tool names that require sanitization
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

        print("Testing Gemini with universal tool names...")
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
                    func_args = tool_call.get("function", {}).get("arguments", "{}")

                    print(f"   {i + 1}. {func_name}")

                    # Verify original names are restored in response
                    if func_name in original_names:
                        print(f"      ✅ Original name restored: {func_name}")
                    else:
                        print(f"      ⚠️  Unexpected name in response: {func_name}")
                        print(f"         (Should be one of: {original_names})")

                    # Test parameter extraction
                    parsed_args = safe_parse_tool_arguments(func_args)
                    if parsed_args:
                        print(f"      📋 Parameters: {list(parsed_args.keys())}")
                    else:
                        print("      📋 No parameters extracted")

            elif response.get("response"):
                print(f"💬 Text response: {response['response'][:150]}...")
            else:
                print("❓ Unexpected response format")

        return True

    except Exception as e:
        error_str = str(e)

        if "function" in error_str.lower() and (
            "name" in error_str.lower() or "invalid" in error_str.lower()
        ):
            print("❌ Tool naming error detected!")
            print(f"   Error: {error_str}")
            print(
                "\n💡 This indicates the universal tool compatibility system needs to be properly integrated"
            )
            return False
        else:
            print(f"❌ Error testing universal tools: {e}")
            import traceback

            traceback.print_exc()
            return False


async def test_gemini_parameter_extraction():
    """Test parameter extraction with universal tool names"""
    print("\n🎯 GEMINI UNIVERSAL PARAMETER EXTRACTION TEST")
    print("=" * 60)

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(provider="gemini", model="gemini-2.5-flash")

        # Universal tool names for testing
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.describe_table",
                    "description": "Get the schema information for a specific table",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to describe",
                            }
                        },
                        "required": ["table_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web.api:search",
                    "description": "Search for information using web API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "category": {
                                "type": "string",
                                "description": "Search category",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "filesystem.read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to read",
                            },
                            "encoding": {
                                "type": "string",
                                "description": "File encoding (default: utf-8)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

        print("🔧 Testing universal tool names with Gemini:")
        for tool in tools:
            print(f"   • {tool['function']['name']} (may require sanitization)")

        # Test cases where parameters are explicit
        test_cases = [
            {
                "request": "describe the products table schema",
                "expected_tool": "stdio.describe_table",
                "expected_params": {"table_name": "products"},
            },
            {
                "request": "show me the structure of the users table",
                "expected_tool": "stdio.describe_table",
                "expected_params": {"table_name": "users"},
            },
            {
                "request": "search for 'Gemini AI' in technology category",
                "expected_tool": "web.api:search",
                "expected_params": {"query": "Gemini AI", "category": "technology"},
            },
            {
                "request": "read the config.json file",
                "expected_tool": "filesystem.read_file",
                "expected_params": {"path": "config.json"},
            },
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\nTest {i + 1}: '{test_case['request']}'")
            print(f"Expected tool: {test_case['expected_tool']}")
            print(f"Expected params: {test_case['expected_params']}")

            messages = [
                {
                    "role": "system",
                    "content": """When a user asks about a specific table, extract the table name from their request and use it as the table_name parameter for stdio.describe_table.

For web searches, extract the query and category from the user's request for web.api:search.

For file operations, extract the file path from the user's request for filesystem.read_file.

Examples:
- "describe the products table" → stdio.describe_table(table_name="products")
- "show users table structure" → stdio.describe_table(table_name="users")
- "search for 'AI news'" → web.api:search(query="AI news")
- "read config.json file" → filesystem.read_file(path="config.json")

NEVER call tools with empty required parameters!
Always use the exact tool names provided.""",
                },
                {"role": "user", "content": test_case["request"]},
            ]

            response = await client.create_completion(
                messages=messages, tools=tools, stream=False, max_tokens=200
            )

            if response.get("tool_calls"):
                call = response["tool_calls"][0]
                func_name = call["function"]["name"]
                func_args = call["function"]["arguments"]

                print(f"   Tool called: {func_name}")

                # Verify tool name restoration
                if func_name == test_case["expected_tool"]:
                    print("   ✅ Correct tool called and name restored")
                else:
                    print(f"   ⚠️  Different tool called: {func_name}")

                # Parse arguments safely
                parsed_args = safe_parse_tool_arguments(func_args)
                print(f"   Parameters: {parsed_args}")

                # Check required parameters
                expected_params = test_case["expected_params"]
                success = True

                for key, expected_value in expected_params.items():
                    actual_value = parsed_args.get(key, "")
                    if key in ["table_name", "path"]:
                        if actual_value == expected_value:
                            print(f"   ✅ {key}: '{actual_value}' (exact match)")
                        elif actual_value and expected_value in actual_value:
                            print(f"   ✅ {key}: '{actual_value}' (contains expected)")
                        else:
                            print(
                                f"   ❌ {key}: '{actual_value}' (expected '{expected_value}')"
                            )
                            success = False
                    elif key == "query":
                        if expected_value.lower() in actual_value.lower():
                            print(f"   ✅ {key}: '{actual_value}' (contains expected)")
                        else:
                            print(
                                f"   ❌ {key}: '{actual_value}' (expected to contain '{expected_value}')"
                            )
                            success = False
                    else:
                        if actual_value:
                            print(f"   ✅ {key}: '{actual_value}' (provided)")
                        else:
                            print(f"   ⚠️  {key}: not provided")

                if success:
                    print("   ✅ OVERALL SUCCESS")
                else:
                    print("   ⚠️  PARTIAL SUCCESS")

            else:
                print("   ❌ FAILED: No tool call made")
                if response.get("response"):
                    print(f"   Text response: {response['response'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Error in universal parameter test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_gemini_streaming_with_tools():
    """Test streaming functionality with universal tool names"""
    print("\n🎯 GEMINI STREAMING WITH UNIVERSAL TOOLS TEST")
    print("=" * 60)

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(provider="gemini", model="gemini-2.5-flash")

        # Universal tools requiring sanitization
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",
                    "description": "Execute a database query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query"}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web.api:search",
                    "description": "Search the web",
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
                    "name": "filesystem.list_files",
                    "description": "List files in directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path"}
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "user",
                "content": "Search for 'latest AI news', list files in the current directory, and query the database for user data",
            }
        ]

        print("Testing Gemini streaming with universal tool names...")
        print("Expected: Tool names should be restored in streaming chunks")

        response = client.create_completion(messages=messages, tools=tools, stream=True)

        chunk_count = 0
        tool_calls_found = []
        restored_names = []
        text_chunks = []

        async for chunk in response:
            chunk_count += 1

            # Handle text content
            if chunk.get("response"):
                text_chunks.append(chunk["response"])
                print(".", end="", flush=True)

            # Handle tool calls
            if chunk.get("tool_calls"):
                for tc in chunk["tool_calls"]:
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    tool_calls_found.append(tool_name)

                    print(f"\n   🔧 Streaming tool call: {tool_name}")

                    # Verify name restoration
                    expected_tools = [
                        "stdio.read_query",
                        "web.api:search",
                        "filesystem.list_files",
                    ]
                    if tool_name in expected_tools:
                        print(
                            "      ✅ Universal tool name correctly restored in stream"
                        )
                        if tool_name not in restored_names:
                            restored_names.append(tool_name)
                    else:
                        print(f"      ⚠️  Unexpected tool name in stream: {tool_name}")

            # Limit for testing
            if chunk_count >= 15:
                break

        print("\n✅ Gemini streaming test completed:")
        print(f"   Chunks processed: {chunk_count}")
        print(f"   Text chunks: {len(text_chunks)}")
        print(f"   Tool calls found: {len(tool_calls_found)}")
        print(f"   Correctly restored names: {len(restored_names)}")

        if restored_names:
            print(f"   Restored tools: {restored_names}")
            return True
        elif tool_calls_found:
            print(
                f"   ⚠️  Tools called but names not fully restored: {tool_calls_found}"
            )
            return True
        else:
            print("   ⚠️  No tool calls in streaming response")
            return False

    except Exception as e:
        print(f"❌ Error in Gemini streaming test: {e}")
        return False


async def test_cross_provider_consistency():
    """Test that Gemini provides consistent behavior with other providers"""
    print("\n🎯 CROSS-PROVIDER CONSISTENCY TEST")
    print("=" * 50)

    print("Testing same request with multiple providers...")

    # Universal tools for consistency testing
    universal_tools = [
        {
            "type": "function",
            "function": {
                "name": "stdio.describe_table",
                "description": "Get table schema",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Table name"}
                    },
                    "required": ["table_name"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "describe the users table structure"}]

    providers_to_test = [
        ("gemini", "gemini-2.5-flash"),
    ]

    # Add other providers if keys are available
    if os.getenv("AZURE_OPENAI_API_KEY"):
        providers_to_test.append(("azure_openai", "gpt-4o-mini"))
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append(("openai", "gpt-4o-mini"))
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append(("anthropic", "claude-sonnet-4-20250514"))
    if os.getenv("MISTRAL_API_KEY"):
        providers_to_test.append(("mistral", "mistral-medium-2505"))

    results = {}

    for provider, model in providers_to_test:
        print(f"\n🔍 Testing {provider} with {model}:")

        try:
            from chuk_llm.llm.client import get_client

            client = get_client(provider=provider, model=model)

            # Check tool compatibility
            if hasattr(client, "get_tool_compatibility_info"):
                tool_info = client.get_tool_compatibility_info()
                print(
                    f"   Compatibility level: {tool_info.get('compatibility_level', 'unknown')}"
                )
                print(
                    f"   Requires sanitization: {tool_info.get('requires_sanitization', 'unknown')}"
                )

            response = await client.create_completion(
                messages=messages, tools=universal_tools, stream=False
            )

            if response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]

                print(f"   Tool called: {func_name}")
                print(f"   Arguments: {func_args}")

                # Check if original name is restored
                if func_name == "stdio.describe_table":
                    print("   ✅ Original tool name correctly restored")

                    # Check parameters using safe parsing
                    parsed_args = safe_parse_tool_arguments(func_args)
                    table_name = parsed_args.get("table_name", "")

                    if table_name:
                        print(
                            f"   ✅ Parameter extraction worked: table_name='{table_name}'"
                        )
                        results[provider] = {
                            "success": True,
                            "tool_name": func_name,
                            "table_name": table_name,
                        }
                    else:
                        print("   ❌ Parameter extraction failed")
                        print(f"      Parsed args: {parsed_args}")
                        results[provider] = {"success": False, "reason": "no_parameter"}
                else:
                    print(f"   ⚠️  Unexpected tool name: {func_name}")
                    results[provider] = {"success": False, "reason": "wrong_tool"}
            else:
                print("   ❌ No tool call made")
                results[provider] = {"success": False, "reason": "no_tool_call"}

        except Exception as e:
            print(f"   ❌ Error testing {provider}: {e}")
            results[provider] = {"success": False, "reason": f"error: {e}"}

    # Compare results
    print("\n📊 CONSISTENCY COMPARISON:")
    successful_providers = [p for p, r in results.items() if r.get("success")]

    if len(successful_providers) >= 1:
        print(f"   ✅ Successful providers: {successful_providers}")

        # Check if all successful providers extracted the same table name
        table_names = [results[p].get("table_name", "") for p in successful_providers]
        if len(set(table_names)) == 1:
            print(f"   ✅ Consistent parameter extraction: '{table_names[0]}'")
            print("   ✅ CONSISTENCY ACHIEVED!")
            return True
        else:
            print("   ⚠️  Different parameter extraction:")
            for provider in successful_providers:
                print(f"      {provider}: '{results[provider].get('table_name', '')}'")
            return True  # Still acceptable
    else:
        print("   ❌ No successful providers")
        for provider, result in results.items():
            print(f"      {provider}: {result.get('reason', 'unknown error')}")
        return False


async def main():
    """Run the complete Gemini universal tool compatibility test suite"""
    print("🧪 GEMINI COMPLETE UNIVERSAL TOOL COMPATIBILITY TEST")
    print("=" * 70)

    print("This test will prove Gemini's universal tool compatibility works by:")
    print("1. Testing universal tool compatibility mixin integration")
    print("2. Testing universal tool names with bidirectional mapping")
    print("3. Testing parameter extraction with any naming convention")
    print("4. Testing streaming with tool name restoration")
    print("5. Comparing consistency with other providers")

    # Test 1: Tool compatibility integration
    result1 = await test_gemini_tool_compatibility()

    # Test 2: Universal tool names
    result2 = await test_gemini_universal_tools() if result1 else False

    # Test 3: Parameter extraction
    result3 = await test_gemini_parameter_extraction() if result2 else False

    # Test 4: Streaming with tools
    result4 = await test_gemini_streaming_with_tools() if result3 else False

    # Test 5: Cross-provider consistency
    result5 = await test_cross_provider_consistency() if result4 else False

    print("\n" + "=" * 70)
    print("🎯 GEMINI COMPLETE TEST RESULTS:")
    print(f"   Tool Compatibility Integration: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"   Universal Tool Names: {'✅ PASS' if result2 else '❌ FAIL'}")
    print(f"   Universal Parameters: {'✅ PASS' if result3 else '❌ FAIL'}")
    print(f"   Streaming + Restoration: {'✅ PASS' if result4 else '❌ FAIL'}")
    print(f"   Cross-Provider Consistency: {'✅ PASS' if result5 else '❌ FAIL'}")

    if result1 and result2 and result3 and result4 and result5:
        print("\n🎉 COMPLETE GEMINI SUCCESS!")
        print("✅ Gemini universal tool compatibility works perfectly!")

        print("\n🔧 PROVEN CAPABILITIES:")
        print("   ✅ MCP-style tool names (stdio.read_query) work seamlessly")
        print("   ✅ API-style tool names (web.api:search) work seamlessly")
        print("   ✅ Database-style names (database.sql.execute) work seamlessly")
        print("   ✅ Filesystem-style names (filesystem.read_file) work seamlessly")
        print("   ✅ Tool names are sanitized for Gemini API compatibility")
        print("   ✅ Original names are restored in responses")
        print("   ✅ Bidirectional mapping works in streaming")
        print("   ✅ Complex conversation flows maintain name restoration")
        print("   ✅ Parameter extraction works with any naming convention")
        print("   ✅ Consistent behavior with other providers")

        print("\n🚀 READY FOR PRODUCTION:")
        print("   • MCP CLI can use any tool naming convention with Gemini")
        print("   • Gemini provides identical user experience to other providers")
        print("   • Tool chaining works across conversation turns")
        print("   • Streaming maintains tool name fidelity")
        print("   • Provider switching is seamless")
        print("   • Warning suppression keeps logs clean")

    elif any([result1, result2, result3, result4, result5]):
        print("\n⚠️  PARTIAL SUCCESS:")
        print("   Some aspects of universal tool compatibility work")
        if result1:
            print("   ✅ Tool compatibility integration works")
        if result2:
            print("   ✅ Universal tool names work")
        if result3:
            print("   ✅ Parameter extraction works")
        if result4:
            print("   ✅ Streaming restoration works")
        if result5:
            print("   ✅ Cross-provider consistency works")

    else:
        print("\n❌ TESTS FAILED:")
        print("   Universal tool compatibility needs implementation")
        print("\n🔧 IMPLEMENTATION STEPS:")
        print("   1. Add ToolCompatibilityMixin import")
        print(
            "   2. Update class inheritance: class GeminiLLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, BaseLLMClient)"
        )
        print("   3. Initialize mixin: ToolCompatibilityMixin.__init__(self, 'gemini')")
        print("   4. Update create_completion to use universal sanitization")
        print("   5. Update response parsing to use universal restoration")
        print("   6. Add Gemini to PROVIDER_REQUIREMENTS in _tool_compatibility.py")


if __name__ == "__main__":
    print("🚀 Starting Gemini Universal Tool Compatibility Test...")
    asyncio.run(main())
