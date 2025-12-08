#!/usr/bin/env python3
"""
DeepSeek Reasoner (Thinking Mode) Example
==========================================

Demonstrates the DeepSeek reasoner model's unique capabilities:
- Extended reasoning with visible thinking process
- Tool calling with reasoning_content preservation
- Multi-turn conversations with tool use

Requirements:
- pip install chuk-llm
- Set DEEPSEEK_API_KEY environment variable

Usage:
    python deepseek_reasoner_example.py
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("DEEPSEEK_API_KEY"):
    print("❌ Please set DEEPSEEK_API_KEY environment variable")
    print("   export DEEPSEEK_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.core import Message, MessageRole, Tool, ToolFunction
    from chuk_llm.llm import get_client
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def demo_basic_reasoning():
    """Demonstrate basic reasoning without tools"""
    print_section("1. Basic Reasoning")

    client = get_client(provider="deepseek", model="deepseek-reasoner")

    messages = [
        Message(
            role=MessageRole.USER,
            content="If I have 3 apples and buy 2 more apples, then give away 1 apple, how many apples do I have? Show your reasoning.",
        )
    ]

    print(
        "📤 User: If I have 3 apples and buy 2 more apples, then give away 1 apple..."
    )
    print("\n⏳ Thinking...")

    response = await client.create_completion(messages=messages, stream=False)

    # DeepSeek reasoner includes reasoning_content
    if "reasoning_content" in response:
        print(f"\n🧠 Reasoning Process ({len(response['reasoning_content'])} chars):")
        print(f"   {response['reasoning_content'][:200]}...")

    print(f"\n💬 Response: {response.get('response', '')}")

    await client.close()


async def demo_tool_calling_with_reasoning():
    """Demonstrate tool calling with reasoning_content preservation"""
    print_section("2. Tool Calling with Reasoning")

    client = get_client(provider="deepseek", model="deepseek-reasoner")

    # Create tools
    get_weather_tool = Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        ),
    )

    calculate_tool = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Perform a mathematical calculation",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2'",
                    }
                },
                "required": ["expression"],
            },
        ),
    )

    # Turn 1: User asks a question that requires tool use
    messages = [
        Message(
            role=MessageRole.USER,
            content="What's the weather in San Francisco? If it's above 70°F, calculate 70 * 1.8 + 32",
        )
    ]

    print("📤 User: What's the weather in San Francisco? If it's above 70°F...")
    print("\n⏳ Thinking and analyzing tools...")

    response = await client.create_completion(
        messages=messages, tools=[get_weather_tool, calculate_tool], stream=False
    )

    print(f"\n🧠 Reasoning Process: {len(response.get('reasoning_content', ''))} chars")
    print(f"   Preview: {response.get('reasoning_content', '')[:150]}...")

    if response.get("tool_calls"):
        print(f"\n🔧 Tool Calls Made: {len(response['tool_calls'])}")
        for i, tc in enumerate(response["tool_calls"], 1):
            print(f"   {i}. {tc['function']['name']}")
            args = json.loads(tc["function"]["arguments"])
            print(f"      Args: {args}")

        # IMPORTANT: The reasoning_content field is preserved in the response
        # and must be included when sending the assistant message back to the API
        print(f"\n✅ reasoning_content preserved: {'reasoning_content' in response}")

        # Simulate tool execution
        tool_call_id = response["tool_calls"][0]["id"]
        messages_dicts = [
            messages[0].to_dict(),
            {
                "role": "assistant",
                "content": response.get("response"),
                "tool_calls": response.get("tool_calls"),
                "reasoning_content": response.get("reasoning_content"),  # CRITICAL!
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {"temperature": 75, "condition": "sunny", "unit": "fahrenheit"}
                ),
            },
        ]

        print("\n⏳ Processing tool result...")

        response2 = await client.create_completion(
            messages=messages_dicts,
            tools=[get_weather_tool, calculate_tool],
            stream=False,
        )

        print(f"\n💬 Final Response: {response2.get('response', '')}")

        if response2.get("tool_calls"):
            print(f"\n🔧 Additional Tool Calls: {len(response2['tool_calls'])}")
            for tc in response2["tool_calls"]:
                print(f"   - {tc['function']['name']}: {tc['function']['arguments']}")

    await client.close()


async def demo_complex_reasoning():
    """Demonstrate complex multi-step reasoning"""
    print_section("3. Complex Multi-Step Reasoning")

    client = get_client(provider="deepseek", model="deepseek-reasoner")

    search_tool = Tool(
        type="function",
        function=ToolFunction(
            name="search_database",
            description="Search a database for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "filters": {
                        "type": "object",
                        "description": "Optional filters",
                        "properties": {
                            "category": {"type": "string"},
                            "date_from": {"type": "string"},
                            "date_to": {"type": "string"},
                        },
                    },
                },
                "required": ["query"],
            },
        ),
    )

    messages = [
        Message(
            role=MessageRole.USER,
            content="I need to find all research papers about machine learning published in 2024. "
            "Can you help me construct the right search query with appropriate filters?",
        )
    ]

    print("📤 User: I need to find all research papers about machine learning...")
    print("\n⏳ Deep thinking about search strategy...")

    response = await client.create_completion(
        messages=messages, tools=[search_tool], stream=False
    )

    if "reasoning_content" in response:
        reasoning = response["reasoning_content"]
        print(f"\n🧠 Reasoning Process ({len(reasoning)} chars):")
        # Show first few lines of reasoning
        lines = reasoning.split("\n")[:5]
        for line in lines:
            if line.strip():
                print(f"   {line[:100]}")

    if response.get("tool_calls"):
        print(f"\n🔧 Constructed Search Query:")
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            print(f"   Query: {args.get('query', 'N/A')}")
            if "filters" in args:
                print(f"   Filters: {json.dumps(args['filters'], indent=6)}")

    print(f"\n💬 Explanation: {response.get('response', 'Tool call made')}")

    await client.close()


async def demo_streaming_with_reasoning():
    """Demonstrate streaming mode with reasoning content"""
    print_section("4. Streaming with Reasoning")

    client = get_client(provider="deepseek", model="deepseek-reasoner")

    messages = [
        Message(
            role=MessageRole.USER,
            content="Explain why Python uses 0-based indexing for lists.",
        )
    ]

    print("📤 User: Explain why Python uses 0-based indexing...")
    print("\n⏳ Streaming response...\n")

    # Note: In streaming mode, reasoning_content is streamed incrementally
    # Don't use 'await' before async for - create_completion returns an async generator
    stream = client.create_completion(messages=messages, stream=True)

    async for chunk in stream:
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)

        # reasoning_content is accumulated in the background
        if chunk.get("reasoning_content"):
            # This contains the accumulated reasoning so far
            pass

    print("\n")
    await client.close()


async def main():
    """Run all DeepSeek reasoner demos"""
    print("\n" + "=" * 70)
    print("🧠 DeepSeek Reasoner (Thinking Mode) Examples")
    print("=" * 70)
    print(f"Model: deepseek-reasoner")
    print(f"API Key: {os.getenv('DEEPSEEK_API_KEY', '')[:20]}...")
    print("=" * 70)

    try:
        # Demo 1: Basic reasoning
        await demo_basic_reasoning()

        # Demo 2: Tool calling with reasoning preservation
        await demo_tool_calling_with_reasoning()

        # Demo 3: Complex multi-step reasoning
        await demo_complex_reasoning()

        # Demo 4: Streaming with reasoning
        await demo_streaming_with_reasoning()

        print_section("✅ All Demos Completed")
        print("\n📚 Key Takeaways:")
        print(
            "  1. DeepSeek reasoner provides extended reasoning in 'reasoning_content'"
        )
        print(
            "  2. When using tools, reasoning_content MUST be preserved in conversation"
        )
        print(
            "  3. Multi-turn conversations work seamlessly with proper field preservation"
        )
        print("  4. Streaming mode accumulates reasoning_content in the background")
        print("\n💡 Best Practices:")
        print(
            "  - Always include reasoning_content when adding assistant messages to history"
        )
        print("  - Use reasoning_content to understand the model's thought process")
        print("  - The reasoner excels at complex, multi-step problems")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
