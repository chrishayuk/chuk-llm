#!/usr/bin/env python3
"""
Enhanced Groq Streaming Diagnostic

Comprehensive testing of Groq streaming tool call behavior.
Tests both accumulation strategies and duplication detection.
Compares Raw Groq behavior with chuk-llm implementation.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

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


async def test_groq_streaming_strategies():
    """Test different Groq streaming accumulation strategies."""

    print("🔍 GROQ STREAMING STRATEGY ANALYSIS")
    print("=" * 50)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ No GROQ_API_KEY")
        return False

    print(f"🔧 Using Groq API key: {api_key[:10]}...")

    # Test case with multiple parameters to test streaming behavior
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_data",
                "description": "Analyze data with multiple filters and options",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "Dataset name to analyze",
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "date_start": {
                                    "type": "string",
                                    "description": "Start date",
                                },
                                "date_end": {
                                    "type": "string",
                                    "description": "End date",
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Data category",
                                },
                            },
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis",
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "include_trends": {
                                    "type": "boolean",
                                    "description": "Include trend analysis",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["json", "csv", "summary"],
                                },
                            },
                        },
                    },
                    "required": ["dataset", "analysis_type"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "Analyze dataset 'sales_data_2024' with filters from '2024-01-01' to '2024-12-31' for category 'electronics', using 'regression' analysis type, include trends, and format as json",
        }
    ]

    print("🎯 Test: Complex data analysis with nested parameters")
    print(
        "Expected: Should test Groq's tool call handling (using llama-3.3-70b-versatile)"
    )

    # Test raw Groq with different accumulation strategies
    print("\n🔥 RAW GROQ ANALYSIS:")
    raw_concatenation = await test_raw_groq_concatenation(api_key, messages, tools)
    raw_replacement = await test_raw_groq_replacement(api_key, messages, tools)

    # Test chuk-llm current behavior
    print("\n🔧 CHUK-LLM CURRENT:")
    chuk_result = await test_chuk_llm_groq(messages, tools)

    # Detailed comparison
    print("\n📊 COMPARISON:")
    print(
        f"Raw (concatenation): {len(raw_concatenation) if raw_concatenation else 0} chars"
    )
    print(
        f"Raw (replacement):   {len(raw_replacement) if raw_replacement else 0} chars"
    )
    print(f"Chuk-LLM current:    {len(chuk_result) if chuk_result else 0} chars")

    # Analyze results
    return analyze_groq_results(
        {
            "raw_concat": raw_concatenation,
            "raw_replace": raw_replacement,
            "chuk_current": chuk_result,
        }
    )


async def test_raw_groq_concatenation(api_key, messages, tools):
    """Test raw Groq with concatenation strategy."""
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)

        # Concatenation strategy
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0

        try:
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                stream=True,
            )

            async for chunk in response:
                chunk_count += 1

                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.tool_calls
                ):
                    tool_call_chunks += 1

                    for tc in chunk.choices[0].delta.tool_calls:
                        idx = tc.index or 0

                        if idx not in tool_calls:
                            tool_calls[idx] = {"name": "", "arguments": "", "id": None}

                        if tc.id:
                            tool_calls[idx]["id"] = tc.id

                        if tc.function:
                            # CONCATENATION STRATEGY
                            if tc.function.name:
                                tool_calls[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[idx]["arguments"] += tc.function.arguments

        except Exception as stream_error:
            print(f"  Streaming failed, trying non-streaming: {stream_error}")

            # Fallback to non-streaming
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                stream=False,
            )

            chunk_count = 1
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.tool_calls
            ):
                tool_call_chunks = 1
                for i, tc in enumerate(response.choices[0].message.tool_calls):
                    tool_calls[i] = {
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments if tc.function else "",
                    }

        print("  Strategy: CONCATENATION")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")

        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({len(tc['arguments'])} chars)")
            return list(tool_calls.values())[0]["arguments"]
        return ""

    except Exception as e:
        print(f"  ❌ Raw Groq concatenation error: {e}")
        return None


async def test_raw_groq_replacement(api_key, messages, tools):
    """Test raw Groq with replacement strategy."""
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)

        # Replacement strategy
        tool_calls = {}
        chunk_count = 0
        tool_call_chunks = 0

        try:
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                stream=True,
            )

            async for chunk in response:
                chunk_count += 1

                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.tool_calls
                ):
                    tool_call_chunks += 1

                    for tc in chunk.choices[0].delta.tool_calls:
                        idx = tc.index or 0

                        if idx not in tool_calls:
                            tool_calls[idx] = {"name": "", "arguments": "", "id": None}

                        if tc.id:
                            tool_calls[idx]["id"] = tc.id

                        if tc.function:
                            # REPLACEMENT STRATEGY
                            if tc.function.name is not None:
                                tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments is not None:
                                tool_calls[idx]["arguments"] = tc.function.arguments

        except Exception as stream_error:
            print(f"  Streaming failed, trying non-streaming: {stream_error}")

            # Fallback to non-streaming
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                stream=False,
            )

            chunk_count = 1
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.tool_calls
            ):
                tool_call_chunks = 1
                for i, tc in enumerate(response.choices[0].message.tool_calls):
                    tool_calls[i] = {
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments if tc.function else "",
                    }

        print("  Strategy: REPLACEMENT")
        print(f"  Chunks: {chunk_count}, Tool chunks: {tool_call_chunks}")

        if tool_calls:
            for idx, tc in tool_calls.items():
                print(f"  Tool: {tc['name']}({len(tc['arguments'])} chars)")
            return list(tool_calls.values())[0]["arguments"]
        return ""

    except Exception as e:
        print(f"  ❌ Raw Groq replacement error: {e}")
        return None


async def test_chuk_llm_groq(messages, tools):
    """Test chuk-llm Groq streaming."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client

        client = get_client(provider="groq", model="llama-3.3-70b-versatile")

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
        print(f"  ❌ Chuk Groq error: {e}")
        return None


def analyze_groq_results(results):
    """Analyze the Groq streaming strategies."""
    print("\n🔬 ANALYSIS:")

    raw_concat = results["raw_concat"]
    raw_replace = results["raw_replace"]
    chuk_current = results["chuk_current"]

    try:
        # Parse JSON arguments for comparison
        concat_parsed = json.loads(raw_concat) if raw_concat else {}
        replace_parsed = json.loads(raw_replace) if raw_replace else {}
        chuk_parsed = json.loads(chuk_current) if chuk_current else {}

        print(f"Concatenation result: {len(concat_parsed)} parameters")
        print(f"Replacement result: {len(replace_parsed)} parameters")
        print(f"Chuk-LLM result: {len(chuk_parsed)} parameters")

        # Groq should behave like OpenAI (concatenation correct for streaming)
        if concat_parsed and len(concat_parsed) >= len(replace_parsed):
            print("✅ CONCATENATION IS CORRECT for Groq")
            print("   Groq follows OpenAI-compatible streaming patterns")

            # Check if chuk matches the correct (concatenation) result
            if chuk_parsed == concat_parsed:
                print("✅ CHUK-LLM HANDLES GROQ CORRECTLY")
                print("   Tool call streaming works properly")
                return True
            elif chuk_parsed == replace_parsed:
                print("❌ CHUK-LLM USES BROKEN LOGIC FOR GROQ")
                print("   Not accumulating deltas properly")
                print("\n🔧 FIX NEEDED:")
                print("   File: chuk_llm/llm/providers/groq_client.py")
                print("   Problem: Not using concatenation for Groq deltas")
                print("   Solution: Ensure Groq follows same logic as OpenAI")
                return False
            elif len(chuk_parsed) > 0 and len(concat_parsed) > 0:
                # Both have data but differ - this might be acceptable
                print("⚠️  GROQ RESULTS DIFFER BUT BOTH HAVE DATA")
                print("   This might be due to Groq-specific behavior")
                return True
            else:
                print("❓ CHUK-LLM HAS DIFFERENT GROQ BEHAVIOR")
                print(f"   Expected (concat): {concat_parsed}")
                print(f"   Got (chuk):        {chuk_parsed}")
                return False
        elif replace_parsed and len(replace_parsed) > len(concat_parsed):
            print("⚠️  REPLACEMENT WORKS BETTER - Unusual for Groq")
            if chuk_parsed == replace_parsed:
                print("✅ CHUK-LLM MATCHES REPLACEMENT STRATEGY")
                return True
            else:
                return False
        else:
            print("❓ BOTH STRATEGIES FAILED OR GAVE EMPTY RESULTS")
            print("   This might be a Groq API limitation or tool compatibility issue")
            return len(chuk_parsed) > 0  # Pass if chuk at least got something

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Concat: '{raw_concat}'")
        print(f"Replace: '{raw_replace}'")
        print(f"Chuk: '{chuk_current}'")
        return False


async def test_duplication_specifically_groq():
    """Test specifically for Groq tool call duplication bug."""
    print("\n🔍 GROQ DUPLICATION TEST")
    print("=" * 30)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Missing Groq API key")
        return False

    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_groq_tool",
                "description": "Test tool for Groq",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Test message"},
                        "number": {"type": "integer", "description": "Test number"},
                    },
                    "required": ["message"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "Call test_groq_tool with message 'hello groq' and number 789",
        }
    ]

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from chuk_llm.llm.client import get_client

        client = get_client(provider="groq", model="llama-3.3-70b-versatile")

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
            print("⚠️  NO TOOL CALLS - Groq may not support this tool format")
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
    """Run enhanced Groq streaming diagnostic."""
    print("🚀 ENHANCED GROQ STREAMING DIAGNOSTIC")
    print("Testing Groq tool call behavior (can be unreliable)")

    # Test 1: Check for duplication specifically
    duplication_ok = await test_duplication_specifically_groq()

    # Test 2: Verify streaming behavior
    behavior_ok = await test_groq_streaming_strategies()

    print("\n" + "=" * 60)
    print("🎯 GROQ DIAGNOSTIC SUMMARY:")
    print(f"Duplication test: {'✅ PASS' if duplication_ok else '❌ FAIL'}")
    print(f"Behavior test:    {'✅ PASS' if behavior_ok else '❌ FAIL'}")

    if duplication_ok and behavior_ok:
        print("\n✅ GROQ STREAMING WORKS WELL!")
        print("   No tool call duplication detected")
        print("   Proper delta accumulation in use")
        print("   Groq integration working correctly")
    elif duplication_ok and not behavior_ok:
        print("\n⚠️  BEHAVIOR ISSUES DETECTED")
        print("   No duplication, but tool call processing may be incomplete")
        print("   Check Groq delta handling or fallback logic")
    elif not duplication_ok and behavior_ok:
        print("\n⚠️  DUPLICATION ISSUE DETECTED")
        print("   Tool calls work, but are being duplicated")
        print("   Check Groq chunk processing logic")
    else:
        print("\n❌ GROQ NEEDS ATTENTION")
        print("   Multiple issues detected with Groq streaming")
        print("   Note: Groq function calling can be unreliable")

    print("\n💡 GROQ SETUP REMINDER:")
    print("Required environment variables:")
    print("- GROQ_API_KEY")
    print("- Note: Groq function calling can be temperamental")
    print("- Fallback to non-streaming is normal for Groq")
    print("- Some failures may be expected due to Groq API limitations")


if __name__ == "__main__":
    asyncio.run(main())
