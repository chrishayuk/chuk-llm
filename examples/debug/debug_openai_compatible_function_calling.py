#!/usr/bin/env python3
"""
OpenAI-Compatible API Function Calling Debug Script
====================================================

Tests multiple approaches to determine what function calling methods work
for any OpenAI-compatible API endpoint.

Usage:
    python debug_openai_compatible_function_calling.py \\
        --provider advantage \\
        --model global/gpt-5-chat

    python debug_openai_compatible_function_calling.py \\
        --provider deepseek \\
        --model deepseek-chat

    python debug_openai_compatible_function_calling.py \\
        --provider groq \\
        --model llama-3.3-70b-versatile
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

# Test tools/functions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"]
            }
        }
    }
]

FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"}
            },
            "required": ["location"]
        }
    }
]


async def raw_api_call(api_base: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Make a raw API call and return the response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                return {
                    "error": f"HTTP {response.status_code}",
                    "detail": response.text[:200]
                }

            return response.json()
        except Exception as e:
            return {"error": str(e)}


def analyze_response(response: dict[str, Any]) -> dict[str, Any]:
    """Analyze API response for function calls."""
    if "error" in response:
        return {
            "success": False,
            "error": response["error"],
            "has_tool_calls": False,
            "has_function_call": False,
            "content": None
        }

    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})

    return {
        "success": True,
        "finish_reason": choice.get("finish_reason"),
        "has_tool_calls": bool(message.get("tool_calls")),
        "tool_calls": message.get("tool_calls"),
        "has_function_call": bool(message.get("function_call")),
        "function_call": message.get("function_call"),
        "content": message.get("content", "")[:200],
        "raw_message": message
    }


async def test_1_native_tools(api_base: str, api_key: str, model: str):
    """Test 1: Native OpenAI tools parameter (current standard)"""
    print("\n" + "="*70)
    print("TEST 1: Native OpenAI tools parameter")
    print("="*70)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": TOOLS,
        "max_tokens": 1000
    }

    response = await raw_api_call(api_base, api_key, payload)
    result = analyze_response(response)

    print(f"✓ API call successful: {result['success']}")
    print(f"✓ Has tool_calls: {result['has_tool_calls']}")
    print(f"✓ Has function_call: {result['has_function_call']}")
    print(f"✓ Content: {result['content']}")

    if result['has_tool_calls']:
        print(f"✅ WORKS: Tool calls returned!")
        print(f"   {json.dumps(result['tool_calls'], indent=2)}")
        return "native_tools"

    return None


async def test_2_legacy_functions(api_base: str, api_key: str, model: str):
    """Test 2: Legacy OpenAI functions parameter (pre-2023)"""
    print("\n" + "="*70)
    print("TEST 2: Legacy functions parameter")
    print("="*70)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "functions": FUNCTIONS,
        "max_tokens": 1000
    }

    response = await raw_api_call(api_base, api_key, payload)
    result = analyze_response(response)

    print(f"✓ API call successful: {result['success']}")
    print(f"✓ Has function_call: {result['has_function_call']}")
    print(f"✓ Content: {result['content']}")

    if result['has_function_call']:
        print(f"✅ WORKS: Function call returned!")
        print(f"   {json.dumps(result['function_call'], indent=2)}")
        return "legacy_functions"

    return None


async def test_3_tools_with_system_prompt(api_base: str, api_key: str, model: str):
    """Test 3: Tools with guiding system prompt"""
    print("\n" + "="*70)
    print("TEST 3: Tools + aggressive system prompt")
    print("="*70)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. You MUST use the available functions to answer questions. When you need to call a function, use the tools provided."
            },
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": TOOLS,
        "max_tokens": 1000
    }

    response = await raw_api_call(api_base, api_key, payload)
    result = analyze_response(response)

    print(f"✓ API call successful: {result['success']}")
    print(f"✓ Has tool_calls: {result['has_tool_calls']}")
    print(f"✓ Content: {result['content']}")

    if result['has_tool_calls']:
        print(f"✅ WORKS: System prompt triggered tool calls!")
        return "tools_with_prompt"

    return None


async def test_4_json_mode_function_call(api_base: str, api_key: str, model: str):
    """Test 4: Instruct model to output function calls as JSON"""
    print("\n" + "="*70)
    print("TEST 4: JSON mode for function calls")
    print("="*70)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to functions. "
                    "When you need to call a function, respond with ONLY a JSON object in this format:\n"
                    '{"name": "function_name", "arguments": {"param": "value"}}\n'
                    "Available functions:\n"
                    "- get_weather(location: string): Get weather for a location"
                )
            },
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "max_tokens": 1000
    }

    response = await raw_api_call(api_base, api_key, payload)
    result = analyze_response(response)

    print(f"✓ API call successful: {result['success']}")
    print(f"✓ Content: {result['content']}")

    # Check if content looks like a function call
    content = result.get('content', '')
    is_json_function_call = False

    if content:
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                is_json_function_call = True
                print(f"✅ WORKS: Model returned JSON function call!")
                print(f"   {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError:
            pass

    if is_json_function_call:
        return "json_mode"

    return None


async def test_5_tool_result_formats(api_base: str, api_key: str, model: str):
    """Test 5: Which format for tool results works?"""
    print("\n" + "="*70)
    print("TEST 5: Tool result message formats")
    print("="*70)

    # First get a function call via JSON mode
    payload1 = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "When asked about weather, respond with ONLY this JSON: "
                    '{"name": "get_weather", "arguments": {"location": "CITY"}}'
                )
            },
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "max_tokens": 500
    }

    response1 = await raw_api_call(api_base, api_key, payload1)
    result1 = analyze_response(response1)

    if not result1['success']:
        print("✗ Could not get initial function call")
        return None

    content1 = result1['content']
    print(f"✓ Got function call: {content1}")

    # Test different formats for tool results
    formats = {
        "tool_role": {
            "role": "tool",
            "content": '{"temperature": 22, "conditions": "sunny"}',
            "tool_call_id": "call_12345",
            "name": "get_weather"
        },
        "user_role": {
            "role": "user",
            "content": 'Tool result from get_weather: {"temperature": 22, "conditions": "sunny"}'
        },
        "function_role": {
            "role": "function",
            "name": "get_weather",
            "content": '{"temperature": 22, "conditions": "sunny"}'
        }
    }

    working_formats = []

    for format_name, tool_message in formats.items():
        print(f"\n  Testing '{format_name}':")

        payload2 = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": content1},
                tool_message
            ],
            "max_tokens": 500
        }

        response2 = await raw_api_call(api_base, api_key, payload2)
        result2 = analyze_response(response2)

        if result2['success'] and result2['content']:
            # Check if response mentions the temperature (indicates it used the tool result)
            if "22" in result2['content'] or "sunny" in result2['content'].lower():
                print(f"    ✅ Works! Response: {result2['content'][:100]}")
                working_formats.append(format_name)
            else:
                print(f"    ⚠️  API accepted but response doesn't use tool result")
                print(f"       Response: {result2['content'][:100]}")
        else:
            print(f"    ✗ Failed: {result2.get('error', 'Unknown error')}")

    if working_formats:
        print(f"\n✅ Working formats: {', '.join(working_formats)}")
        return working_formats
    else:
        print(f"\n❌ No tool result formats work")
        return None


async def test_6_check_models(api_base: str, api_key: str):
    """Test 6: Check what models are available"""
    print("\n" + "="*70)
    print("TEST 6: Available models")
    print("="*70)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{api_base}/models",
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                models = [m.get('id') for m in data.get('data', [])]
                print(f"✓ Found {len(models)} models")
                for model in models[:10]:
                    print(f"  - {model}")
                if len(models) > 10:
                    print(f"  ... and {len(models) - 10} more")
                return models
            else:
                print(f"✗ Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")

    return None


async def main():
    """Run all tests and provide recommendations"""
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible API function calling capabilities"
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Provider name (e.g., advantage, deepseek, groq)"
    )
    parser.add_argument(
        "--api-base",
        help="API base URL (default: from environment)"
    )
    parser.add_argument(
        "--api-key",
        help="API key (default: from environment)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to test"
    )

    args = parser.parse_args()

    # Get configuration
    provider = args.provider.upper()
    api_base = args.api_base or os.getenv(f"{provider}_API_BASE")
    api_key = args.api_key or os.getenv(f"{provider}_API_KEY")

    print("="*70)
    print(f"OPENAI-COMPATIBLE API FUNCTION CALLING DEBUG")
    print("="*70)
    print(f"Provider: {args.provider}")
    print(f"API Base: {api_base}")
    print(f"Model: {args.model}")
    print(f"API Key: {api_key[:20] if api_key else 'NOT SET'}...")
    print("="*70)

    if not api_key:
        print(f"\n❌ ERROR: {provider}_API_KEY not set!")
        print(f"   Set it via environment or --api-key parameter")
        return

    if not api_base:
        print(f"\n❌ ERROR: {provider}_API_BASE not set!")
        print(f"   Set it via environment or --api-base parameter")
        return

    results = {}

    # Run all tests
    results['native_tools'] = await test_1_native_tools(api_base, api_key, args.model)
    results['legacy_functions'] = await test_2_legacy_functions(api_base, api_key, args.model)
    results['tools_with_prompt'] = await test_3_tools_with_system_prompt(api_base, api_key, args.model)
    results['json_mode'] = await test_4_json_mode_function_call(api_base, api_key, args.model)
    results['tool_result_formats'] = await test_5_tool_result_formats(api_base, api_key, args.model)
    results['models'] = await test_6_check_models(api_base, api_key)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)

    working_approaches = [k for k, v in results.items() if v and k not in ['models', 'tool_result_formats']]

    if working_approaches:
        print(f"✅ {len(working_approaches)} approach(es) work!\n")

        if 'native_tools' in working_approaches:
            print("📝 RECOMMENDATION: Use native OpenAI tools")
            print("   ├─ This API fully supports OpenAI-style function calling")
            print("   ├─ Client can extend OpenAILLMClient directly")
            print("   └─ No custom implementation needed (like Moonshot client)")

        elif 'legacy_functions' in working_approaches:
            print("📝 RECOMMENDATION: Use legacy functions parameter")
            print("   ├─ This API uses pre-2023 OpenAI function calling")
            print("   ├─ Convert tools to functions format in client")
            print("   └─ Parse function_call instead of tool_calls")

        elif 'json_mode' in working_approaches:
            print("📝 RECOMMENDATION: Use JSON mode (like current Advantage implementation)")
            print("   ├─ Inject system prompt to guide JSON function calling")
            print("   ├─ Parse JSON from response content field")
            print("   ├─ Convert to standard tool_calls format")

            # Check tool result formats
            tool_formats = results.get('tool_result_formats')
            if tool_formats:
                print(f"   ├─ Tool result formats that work: {', '.join(tool_formats)}")
                if 'tool_role' not in tool_formats:
                    print("   └─ ⚠️  Convert 'tool' role messages to 'user' role (API doesn't support tool role)")
            else:
                print("   └─ ⚠️  Tool result handling may need custom implementation")
        else:
            print(f"📝 RECOMMENDATION: Use {working_approaches[0]} approach")

    else:
        print("❌ No approaches work!")
        print("\nPossible reasons:")
        print("  • API doesn't support function calling at all")
        print("  • Model doesn't support function calling")
        print("  • API configuration/authentication issue")
        print("  • Rate limiting or temporary service issue")

    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
