#!/usr/bin/env python3
"""
Debug tool streaming issue in ChukLLM
Specifically test what happens when tools are involved vs regular text
"""

import asyncio
import os
import sys
from pathlib import Path

# Add chuk-llm to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def debug_tool_streaming():
    """Debug exactly what happens with tool streaming"""

    print("🔍 TOOL STREAMING DEBUG")
    print("=" * 50)

    # Test tool definition
    test_tool = {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute a SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"},
                    "database": {
                        "type": "string",
                        "description": "Database name",
                        "default": "main",
                    },
                },
                "required": ["query"],
            },
        },
    }

    from chuk_llm import stream

    # Test 1: Regular streaming (should work)
    print("\n🧪 Test 1: Regular text streaming (no tools)")
    print("-" * 30)

    chunk_count = 0
    full_response = ""

    try:
        async for chunk in stream(
            "What is 2+2? Answer briefly.", provider="openai", model="gpt-4o-mini"
        ):
            chunk_count += 1
            if chunk:
                full_response += str(chunk)
                print(
                    f"  Chunk {chunk_count}: '{chunk[:50]}{'...' if len(str(chunk)) > 50 else ''}'"
                )

        print(f"✅ Regular streaming: {chunk_count} chunks, {len(full_response)} chars")
        print(f"   Response: {full_response[:100]}...")

    except Exception as e:
        print(f"❌ Regular streaming failed: {e}")

    # Test 2: Tool streaming (currently broken)
    print("\n🧪 Test 2: Tool call streaming (currently broken)")
    print("-" * 30)

    chunk_count = 0
    full_response = ""

    try:
        async for chunk in stream(
            "Execute SQL: SELECT * FROM users LIMIT 5",
            provider="openai",
            model="gpt-4o-mini",
            tools=[test_tool],
        ):
            chunk_count += 1
            if chunk:
                full_response += str(chunk)
                print(
                    f"  Chunk {chunk_count}: '{chunk[:50]}{'...' if len(str(chunk)) > 50 else ''}'"
                )

        print(f"🔧 Tool streaming: {chunk_count} chunks, {len(full_response)} chars")
        print(f"   Response: {full_response[:100]}...")

        if chunk_count == 0:
            print("❌ PROBLEM: No chunks returned for tool call!")
            print("   This indicates the streaming generator isn't yielding properly")

    except Exception as e:
        print(f"❌ Tool streaming failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Let's debug the client directly
    print("\n🧪 Test 3: Direct client debugging")
    print("-" * 30)

    try:
        from chuk_llm.llm.client import get_client

        client = get_client(
            provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
        )

        print(f"✅ Got client: {type(client)}")
        print(f"   Model: {client.model}")
        print(f"   Provider: {getattr(client, 'detected_provider', 'unknown')}")

        # Test direct streaming
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Execute SQL: SELECT * FROM users LIMIT 5"},
        ]

        print("🔄 Testing direct client streaming with tools...")

        chunk_count = 0
        response_stream = client.create_completion(
            messages=messages, tools=[test_tool], stream=True, max_tokens=200
        )

        print(f"✅ Got response stream: {type(response_stream)}")
        print(f"   Is async iterable: {hasattr(response_stream, '__aiter__')}")

        # Try to iterate
        try:
            async for chunk in response_stream:
                chunk_count += 1
                print(
                    f"  Direct chunk {chunk_count}: {type(chunk)} - {str(chunk)[:100]}..."
                )

                if chunk_count >= 5:  # Limit for debugging
                    break

        except Exception as stream_error:
            print(f"❌ Direct streaming failed: {stream_error}")
            import traceback

            traceback.print_exc()

        print(f"🔧 Direct client streaming: {chunk_count} chunks received")

    except Exception as e:
        print(f"❌ Client debugging failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run the debugging"""
    await debug_tool_streaming()

    print("\n" + "=" * 60)
    print("🎯 DEBUG SUMMARY:")
    print("If Test 1 works but Test 2 doesn't, the issue is in core.py stream()")
    print("If Test 3 works but Test 2 doesn't, the issue is in the content extraction")
    print("If nothing works, there's a deeper client issue")


if __name__ == "__main__":
    asyncio.run(main())
