#!/usr/bin/env python3
"""
Clean OpenAI Client SQL Issue Test
==================================
Minimal test to isolate LIMIT10 concatenation in your OpenAI client.
"""

import asyncio
import json
import os

# Adjust import path as needed
try:
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient
except ImportError:
    print("❌ Could not import OpenAILLMClient - check path")
    exit(1)


def sql_tool():
    """Simple SQL tool definition"""
    return {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Execute SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to run"}
                },
                "required": ["sql"],
            },
        },
    }


def check_sql(response, mode):
    """Check if response contains LIMIT10 or similar issues"""
    issues = []

    if not response.get("tool_calls"):
        return ["No tool calls found"]

    for call in response["tool_calls"]:
        if call.get("function", {}).get("name") == "run_query":
            try:
                args = json.loads(call["function"]["arguments"])
                sql = args.get("sql", "")

                print(f"  {mode} SQL: {sql}")

                if "LIMIT10" in sql:
                    issues.append("Found LIMIT10")
                if "OFFSET10" in sql:
                    issues.append("Found OFFSET10")

            except Exception as e:
                issues.append(f"Parse error: {e}")

    return issues


async def test_sql_issue():
    """Test for SQL spacing issues"""
    print("🧪 Testing SQL spacing issues in OpenAI client\n")

    client = OpenAILLMClient(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "user", "content": "Get top 10 products with SQL using run_query tool"}
    ]

    tool = sql_tool()

    # Test regular mode
    print("📤 Regular mode:")
    try:
        response = await client.create_completion(messages, [tool], stream=False)
        issues = check_sql(response, "Regular")
        if issues:
            print(f"  ❌ Issues: {issues}")
        else:
            print("  ✅ No issues")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print()

    # Test streaming mode
    print("📡 Streaming mode:")
    try:
        chunks = []
        async for chunk in client.create_completion(messages, [tool], stream=True):
            chunks.append(chunk)

        # Reconstruct final response from chunks
        final_response = {"tool_calls": []}
        for chunk in chunks:
            if chunk.get("tool_calls"):
                final_response["tool_calls"].extend(chunk["tool_calls"])

        issues = check_sql(final_response, "Streaming")
        if issues:
            print(f"  ❌ Issues: {issues}")
        else:
            print("  ✅ No issues")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    await client.close()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        exit(1)

    asyncio.run(test_sql_issue())
