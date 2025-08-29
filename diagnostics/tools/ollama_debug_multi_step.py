#!/usr/bin/env python3
"""
Test GPT-OSS with proper conversation continuity for multiple tool calls
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_with_continuation():
    """Test GPT-OSS with proper conversation continuation after each tool call"""

    import ollama

    print("=" * 60)
    print("CONVERSATION CONTINUATION TEST")
    print("=" * 60)
    print("Testing proper multi-step tool calling with result feedback\n")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Initial request
    messages = [
        {
            "role": "user",
            "content": "Get the weather for Tokyo, New York, and Paris. Call get_weather for each city.",
        }
    ]

    client = ollama.AsyncClient()
    all_locations_called = []
    max_rounds = 5  # Prevent infinite loops

    for round_num in range(1, max_rounds + 1):
        print(f"\n📍 Round {round_num}:")
        print(f"Sending: {messages[-1]['content'][:100]}...")

        # Make the call
        response = await client.chat(
            model="gpt-oss:latest",
            messages=messages,
            tools=tools,
            options={"temperature": 0.1, "num_predict": 300},
        )

        # Check for tool calls
        tool_calls_made = []
        if hasattr(response, "message") and response.message:
            msg = response.message

            # Get the response content
            content = getattr(msg, "content", "")
            if content:
                print(f"Model response: {content[:150]}...")
                messages.append({"role": "assistant", "content": content})

            # Check for tool calls
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        func = tc.function
                        name = getattr(func, "name", "unknown")
                        args = getattr(func, "arguments", {})

                        if name == "get_weather" and isinstance(args, dict):
                            location = args.get("location", "unknown")
                            tool_calls_made.append(location)
                            all_locations_called.append(location)
                            print(f"  🔧 Tool call: get_weather({location})")

        # If tool calls were made, provide results and continue
        if tool_calls_made:
            # Simulate tool execution results
            results = []
            for location in tool_calls_made:
                result = f"Weather in {location}: 22°C, Partly cloudy"
                results.append(result)

            # Add tool results to conversation
            result_message = "Tool results:\n" + "\n".join(results)

            # Check if we have all three cities
            unique_locations = list(set(all_locations_called))
            remaining = {"Tokyo", "New York", "Paris"} - set(unique_locations)

            if remaining:
                # Ask for the remaining cities
                result_message += f"\n\nNow get the weather for: {', '.join(remaining)}"
                messages.append({"role": "user", "content": result_message})
            else:
                # We have all cities
                print(
                    f"\n✅ Success! Got all {len(unique_locations)} cities in {round_num} rounds"
                )
                print(f"Locations: {unique_locations}")
                return True
        else:
            # No tool calls made
            print("  ⚠️ No tool calls made")

            # Try to prompt more explicitly
            unique_so_far = list(set(all_locations_called))
            if len(unique_so_far) < 3:
                remaining = {"Tokyo", "New York", "Paris"} - set(unique_so_far)
                messages.append(
                    {
                        "role": "user",
                        "content": f"Please call get_weather for {', '.join(remaining)}",
                    }
                )
            else:
                break

    # Check final results
    unique_locations = list(set(all_locations_called))
    print("\n📊 Final Results:")
    print(f"Total rounds: {round_num}")
    print(f"Locations called: {unique_locations}")
    print(f"Success: {len(unique_locations) >= 3}")

    return len(unique_locations) >= 3


async def test_single_shot_prompts():
    """Test various single-shot prompts to see what works best"""

    import ollama

    print("\n" + "=" * 60)
    print("SINGLE-SHOT PROMPT TESTS")
    print("=" * 60)
    print("Testing what prompts work best for the first call\n")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    prompts = [
        "Get weather for Tokyo",
        "Get weather for Tokyo, then New York, then Paris",
        "Call get_weather with location='Tokyo'",
        "Use the get_weather tool to check Tokyo's weather",
    ]

    client = ollama.AsyncClient()

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        response = await client.chat(
            model="gpt-oss:latest",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            options={"temperature": 0.1, "num_predict": 200},
        )

        # Check for tool calls
        if hasattr(response, "message") and response.message:
            tool_calls = getattr(response.message, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        func = tc.function
                        name = getattr(func, "name", "")
                        args = getattr(func, "arguments", {})
                        location = (
                            args.get("location", "?") if isinstance(args, dict) else "?"
                        )
                        print(f"  ✅ Tool call: {name}({location})")
            else:
                content = getattr(response.message, "content", "")
                print(f"  ❌ No tool call, response: {content[:100]}...")


async def test_batch_tool():
    """Test if a batch tool works better"""

    import ollama

    print("\n" + "=" * 60)
    print("BATCH TOOL TEST")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_multiple_weather",
                "description": "Get weather for multiple cities at once",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of city names",
                        }
                    },
                    "required": ["cities"],
                },
            },
        }
    ]

    prompt = "Get the weather for Tokyo, New York, and Paris"
    print(f"Prompt: {prompt}")
    print("Using batch tool that accepts array\n")

    client = ollama.AsyncClient()

    response = await client.chat(
        model="gpt-oss:latest",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        options={"temperature": 0.1, "num_predict": 300},
    )

    if hasattr(response, "message") and response.message:
        tool_calls = getattr(response.message, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    func = tc.function
                    name = getattr(func, "name", "")
                    args = getattr(func, "arguments", {})
                    if isinstance(args, dict) and "cities" in args:
                        cities = args["cities"]
                        print(f"✅ Batch call: {name}({cities})")
                        print(f"   {len(cities)} cities in one call!")
                        return True
        else:
            print("❌ No tool call made")

    return False


async def main():
    """Run all tests"""

    print("🔬 GPT-OSS MULTI-TOOL CALLING ANALYSIS")
    print("Testing different approaches for multiple tool calls\n")

    # Test 1: Single-shot prompts
    print("Test 1: Understanding single-shot behavior")
    await test_single_shot_prompts()

    # Test 2: Batch tool
    print("\nTest 2: Batch tool approach")
    batch_success = await test_batch_tool()

    # Test 3: Proper continuation
    print("\nTest 3: Conversation continuation approach")
    continuation_success = await test_with_continuation()

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    if continuation_success:
        print("✅ SOLUTION FOUND: Conversation continuation works!")
        print("\nBest Practice for GPT-OSS multi-step workflows:")
        print("1. Make initial request")
        print("2. Get tool call and execute it")
        print("3. Return results and request next step")
        print("4. Repeat until all steps complete")
        print("\nThis is how reasoning models are designed to work.")

    if batch_success:
        print("\n✅ ALTERNATIVE: Batch tools work!")
        print("Design tools to accept arrays/lists when multiple items needed")

    if not continuation_success and not batch_success:
        print("❌ No working solution found")
        print("The model may have limitations with the current setup")


if __name__ == "__main__":
    asyncio.run(main())
