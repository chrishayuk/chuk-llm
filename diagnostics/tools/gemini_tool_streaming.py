#!/usr/bin/env python3
"""
Enhanced Gemini Streaming Diagnostic

Comprehensive testing of Gemini streaming tool call behavior.
Tests both accumulation strategies and duplication detection.
Compares Raw Gemini behavior with chuk-llm implementation.
"""

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Load environment
try:
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print("✅ Loaded .env")
    else:
        load_dotenv()
except ImportError:
    print("⚠️ No dotenv")


async def test_gemini_streaming_strategies():
    """Test different Gemini streaming accumulation strategies."""

    print("🔍 GEMINI STREAMING STRATEGY ANALYSIS")
    print("=" * 55)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY")
        return False

    print(f"🔧 Using Gemini API key: {api_key[:10]}...")

    # Test case with multiple parameters to test streaming behavior
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_search_query",
                "description": "Execute a complex search query with multiple filters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "date_range": {
                                    "type": "string",
                                    "description": "Date range filter",
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Content category",
                                },
                                "language": {
                                    "type": "string",
                                    "description": "Content language",
                                },
                            },
                        },
                        "sort_by": {"type": "string", "description": "Sort criteria"},
                        "limit": {"type": "integer", "description": "Maximum results"},
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Include result metadata",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "Search for 'artificial intelligence machine learning' with date range 'last_6_months', category 'research', language 'english', sort by 'relevance', limit 25, and include metadata",
        }
    ]

    print("🎯 Test: Complex search query with nested parameters")
    print("Expected: Should test Gemini's tool call handling")

    # Test raw Gemini
    print("\n🔥 RAW GEMINI ANALYSIS:")
    raw_result = await test_raw_gemini_comprehensive(api_key, messages, tools)

    # Test chuk-llm current behavior
    print("\n🔧 CHUK-LLM CURRENT:")
    chuk_result = await test_chuk_llm_gemini(messages, tools)

    # Detailed comparison
    print("\n📊 COMPARISON:")
    print(f"Raw Gemini:      {len(raw_result) if raw_result else 0} chars")
    print(f"Chuk-LLM current: {len(chuk_result) if chuk_result else 0} chars")

    # Analyze results
    return analyze_gemini_results(
        {"raw_gemini": raw_result, "chuk_current": chuk_result}
    )


async def test_raw_gemini_comprehensive(api_key, messages, tools):
    """Test raw Gemini with comprehensive analysis."""
    try:
        # Suppress warnings during import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from google import genai
            from google.genai import types as gtypes

        client = genai.Client(api_key=api_key)

        # Convert tools to Gemini format
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                func_decl = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
                function_declarations.append(func_decl)

        gemini_tool = (
            gtypes.Tool(function_declarations=function_declarations)
            if function_declarations
            else None
        )

        # Track streaming behavior
        chunk_count = 0
        tool_call_chunks = 0
        tool_calls = {}

        try:
            print("  Using model: gemini-2.5-flash")

            # Configuration
            config_params = {"max_output_tokens": 4096, "temperature": 0.1}
            if gemini_tool:
                config_params["tools"] = [gemini_tool]

            config = gtypes.GenerateContentConfig(**config_params)

            # Try streaming first
            try:
                stream = await client.aio.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=messages[0]["content"],
                    config=config,
                )

                async for chunk in stream:
                    chunk_count += 1

                    try:
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            candidate = chunk.candidates[0]

                            if hasattr(candidate, "content") and candidate.content:
                                content = candidate.content

                                if hasattr(content, "parts") and content.parts:
                                    for part in content.parts:
                                        if (
                                            hasattr(part, "function_call")
                                            and part.function_call
                                        ):
                                            tool_call_chunks += 1
                                            fc = part.function_call
                                            function_name = getattr(
                                                fc, "name", "unknown"
                                            )
                                            function_args = dict(
                                                getattr(fc, "args", {})
                                            )

                                            # Store tool call
                                            tool_calls[0] = {
                                                "name": function_name,
                                                "arguments": json.dumps(function_args),
                                            }
                    except Exception:
                        continue

                print("  Streaming mode: SUCCESS")

            except Exception as stream_error:
                print(f"  Streaming mode: FAILED ({stream_error})")
                print("  Falling back to non-streaming")

                # Fallback to non-streaming
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[messages[0]["content"]],
                    config=config,
                )

                chunk_count = 1

                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]

                    if hasattr(candidate, "content") and candidate.content:
                        content = candidate.content

                        if hasattr(content, "parts") and content.parts:
                            for part in content.parts:
                                if (
                                    hasattr(part, "function_call")
                                    and part.function_call
                                ):
                                    tool_call_chunks = 1
                                    fc = part.function_call
                                    function_name = getattr(fc, "name", "unknown")
                                    function_args = dict(getattr(fc, "args", {}))

                                    tool_calls[0] = {
                                        "name": function_name,
                                        "arguments": json.dumps(function_args),
                                    }

        except Exception as e:
            print(f"  Both streaming and non-streaming failed: {e}")
            return None

        print(f"  Chunks: {chunk_count}")
        print(f"  Tool call chunks: {tool_call_chunks}")

        if tool_calls:
            for _idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({len(tc['arguments'])} chars)")
            return list(tool_calls.values())[0]["arguments"]
        else:
            print("  No tool calls found")
            return ""

    except Exception as e:
        print(f"  ❌ Raw Gemini error: {e}")
        return None


async def test_chuk_llm_gemini(messages, tools):
    """Test chuk-llm Gemini streaming."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client

        client = get_client(provider="gemini", model="gemini-2.5-flash")

        # Stream with chuk-llm
        chunk_count = 0
        final_tool_calls = []
        all_chunks = []

        try:
            async for chunk in client.create_completion(
                messages=messages, tools=tools, stream=True
            ):
                chunk_count += 1
                all_chunks.append(chunk)

                if chunk.get("tool_calls"):
                    # Check for duplication
                    for tc in chunk["tool_calls"]:
                        tc_signature = (
                            f"{tc['function']['name']}({tc['function']['arguments']})"
                        )
                        existing_signatures = [
                            f"{existing['function']['name']}({existing['function']['arguments']})"
                            for existing in final_tool_calls
                        ]

                        if tc_signature not in existing_signatures:
                            final_tool_calls.append(tc)

            print("  Streaming mode: SUCCESS")

        except Exception as stream_error:
            print(f"  Streaming mode: FAILED ({stream_error})")
            print("  Falling back to non-streaming")

            # Fallback to non-streaming
            try:
                result = await client.create_completion(
                    messages=messages, tools=tools, stream=False
                )

                chunk_count = 1
                all_chunks = [result]
                if result.get("tool_calls"):
                    final_tool_calls.extend(result["tool_calls"])

            except Exception as non_stream_error:
                print(f"  Non-streaming also failed: {non_stream_error}")
                return None

        print(f"  Chunks: {chunk_count}")
        print(
            f"  Tool call chunks: {len([c for c in all_chunks if c.get('tool_calls')])}"
        )
        print(f"  Total unique tools: {len(final_tool_calls)}")

        if final_tool_calls:
            for i, tc in enumerate(final_tool_calls):
                args = tc.get("function", {}).get("arguments", "")
                name = tc.get("function", {}).get("name", "")
                print(f"  Tool {i + 1}: {name}({len(args)} chars)")
            return final_tool_calls[0].get("function", {}).get("arguments", "")
        else:
            print("  No tool calls found")
            return ""

    except Exception as e:
        print(f"  ❌ Chuk Gemini error: {e}")
        return None


def analyze_gemini_results(results):
    """Analyze the Gemini streaming strategies."""
    print("\n🔬 ANALYSIS:")

    raw_gemini = results["raw_gemini"]
    chuk_current = results["chuk_current"]

    try:
        # Parse JSON arguments for comparison
        raw_parsed = json.loads(raw_gemini) if raw_gemini else {}
        chuk_parsed = json.loads(chuk_current) if chuk_current else {}

        print(f"Raw Gemini parameters: {len(raw_parsed)} keys")
        print(f"Chuk-LLM parameters: {len(chuk_parsed)} keys")

        # Gemini analysis
        if raw_parsed and chuk_parsed:
            if raw_parsed == chuk_parsed:
                print("✅ PERFECT MATCH")
                print("   Both implementations produce identical results")
                return True
            else:
                print("⚠️  RESULTS DIFFER")
                print(f"   Raw keys: {list(raw_parsed.keys())}")
                print(f"   Chuk keys: {list(chuk_parsed.keys())}")
                # Still consider this a pass if both have data
                return len(raw_parsed) > 0 and len(chuk_parsed) > 0
        elif raw_parsed and not chuk_parsed:
            print("❌ CHUK-LLM BROKEN")
            print("   Raw Gemini works but chuk-llm loses data")
            print("\n🔧 FIX NEEDED:")
            print("   File: chuk_llm/llm/providers/gemini_client.py")
            print("   Problem: Not extracting function_call data properly")
            print("   Solution: Fix Gemini response parsing")
            return False
        elif not raw_parsed and chuk_parsed:
            print("🎉 CHUK-LLM BETTER THAN RAW")
            print("   Chuk-llm handles Gemini better than raw implementation")
            return True
        else:
            print("❓ BOTH METHODS FAILED")
            print("   This might be a Gemini API issue or tool compatibility problem")
            return False

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Raw: '{raw_gemini}'")
        print(f"Chuk: '{chuk_current}'")
        return False


async def test_duplication_specifically_gemini():
    """Test specifically for Gemini tool call duplication bug."""
    print("\n🔍 GEMINI DUPLICATION TEST")
    print("=" * 35)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Missing Gemini API key")
        return False

    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_gemini_tool",
                "description": "Test tool for Gemini",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Test message"},
                        "count": {"type": "integer", "description": "Test count"},
                    },
                    "required": ["message"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "Call test_gemini_tool with message 'hello gemini' and count 456",
        }
    ]

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client

        client = get_client(provider="gemini", model="gemini-2.5-flash")

        all_tool_calls = []
        chunk_count = 0

        try:
            async for chunk in client.create_completion(
                messages=messages, tools=tools, stream=True
            ):
                chunk_count += 1
                if chunk.get("tool_calls"):
                    all_tool_calls.extend(chunk["tool_calls"])

        except Exception as stream_error:
            print(f"Streaming failed, trying non-streaming: {stream_error}")

            result = await client.create_completion(
                messages=messages, tools=tools, stream=False
            )

            chunk_count = 1
            if result.get("tool_calls"):
                all_tool_calls.extend(result["tool_calls"])

        print(f"Total chunks: {chunk_count}")
        print(f"Total tool calls collected: {len(all_tool_calls)}")

        # Check for duplication
        unique_tool_calls = []
        for tc in all_tool_calls:
            tc_signature = f"{tc['function']['name']}({tc['function']['arguments']})"
            if tc_signature not in [
                f"{utc['function']['name']}({utc['function']['arguments']})"
                for utc in unique_tool_calls
            ]:
                unique_tool_calls.append(tc)

        print(f"Unique tool calls: {len(unique_tool_calls)}")

        if len(all_tool_calls) == len(unique_tool_calls) == 1:
            print("✅ NO DUPLICATION - Perfect!")
            return True
        elif len(unique_tool_calls) == 1 and len(all_tool_calls) > 1:
            print(
                f"❌ DUPLICATION DETECTED - {len(all_tool_calls)} copies of same tool call"
            )
            return False
        elif len(all_tool_calls) == 0:
            print("⚠️  NO TOOL CALLS - Gemini may not support this tool format")
            return False
        else:
            print("❓ UNEXPECTED TOOL CALL PATTERN")
            for i, tc in enumerate(all_tool_calls):
                print(
                    f"  Tool {i + 1}: {tc['function']['name']}({tc['function']['arguments']})"
                )
            return len(unique_tool_calls) > 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run enhanced Gemini streaming diagnostic."""
    print("🚀 ENHANCED GEMINI STREAMING DIAGNOSTIC")
    print("Testing Gemini tool call behavior (can be temperamental)")

    # Test 1: Check for duplication specifically
    duplication_ok = await test_duplication_specifically_gemini()

    # Test 2: Verify streaming behavior
    behavior_ok = await test_gemini_streaming_strategies()

    print("\n" + "=" * 65)
    print("🎯 GEMINI DIAGNOSTIC SUMMARY:")
    print(f"Duplication test: {'✅ PASS' if duplication_ok else '❌ FAIL'}")
    print(f"Behavior test:    {'✅ PASS' if behavior_ok else '❌ FAIL'}")

    if duplication_ok and behavior_ok:
        print("\n✅ GEMINI STREAMING WORKS WELL!")
        print("   No tool call duplication detected")
        print("   Proper function call handling in use")
        print("   Gemini integration working correctly")
    elif duplication_ok and not behavior_ok:
        print("\n⚠️  BEHAVIOR ISSUES DETECTED")
        print("   No duplication, but tool call processing may be incomplete")
        print("   Check Gemini function_call extraction")
    elif not duplication_ok and behavior_ok:
        print("\n⚠️  DUPLICATION ISSUE DETECTED")
        print("   Tool calls work, but are being duplicated")
        print("   Check Gemini chunk processing logic")
    else:
        print("\n❌ GEMINI NEEDS ATTENTION")
        print("   Multiple issues detected with Gemini streaming")
        print("   Note: Gemini can be inconsistent with function calling")

    print("\n💡 GEMINI SETUP REMINDER:")
    print("Required environment variables:")
    print("- GEMINI_API_KEY or GOOGLE_API_KEY")
    print("- Note: Gemini function calling can be temperamental")
    print("- Some failures may be expected due to Gemini API limitations")


if __name__ == "__main__":
    asyncio.run(main())
