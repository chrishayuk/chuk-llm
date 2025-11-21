#!/usr/bin/env python3
"""
Basic Tool Calling / Function Calling
======================================

Demonstrates the clean, simple tool/function calling API.
Tools are just another parameter - no complexity!
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import ask, ask_sync
from chuk_llm.api.tools import tools_from_functions

# Define your tools as regular Python functions
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get current weather for a location."""
    # In real app, call weather API
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "sunny"
    }

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"

async def basic_tool_usage():
    """Basic tool calling with ask()."""
    print("=== Basic Tool Usage ===\n")

    # Create toolkit from functions
    toolkit = tools_from_functions(get_weather, calculate)

    # Use tools with ask() - it's just another parameter!
    response = await ask(
        "What's the weather in Paris and what's 15 * 4?",
        tools=toolkit.to_openai_format(),
        provider="openai",
        model="gpt-4o-mini"
    )

    # With tools, response is a dict with tool_calls
    print(f"Response: {response}\n")

async def single_tool():
    """Using a single tool."""
    print("=== Single Tool ===\n")

    toolkit = tools_from_functions(calculate)

    response = await ask(
        "What is 123 * 456?",
        tools=toolkit.to_openai_format(),
        model="gpt-4o-mini"
    )

    print(f"Result: {response}\n")

def sync_tools():
    """Tool calling with sync API."""
    print("=== Sync Tool Calling ===\n")

    toolkit = tools_from_functions(get_weather)

    response = ask_sync(
        "What's the weather in Tokyo?",
        tools=toolkit.to_openai_format(),
        model="gpt-4o-mini"
    )

    print(f"Response: {response}\n")

async def multiple_providers():
    """Tool calling works with different providers."""
    print("=== Multiple Providers ===\n")

    toolkit = tools_from_functions(calculate)

    # OpenAI
    response = await ask(
        "Calculate 99 + 1",
        tools=toolkit.to_openai_format(),
        provider="openai",
        model="gpt-4o-mini"
    )
    print(f"OpenAI: {response}")

    # Anthropic
    try:
        response = await ask(
            "Calculate 50 * 2",
            tools=toolkit.to_openai_format(),
            provider="anthropic",
            model="claude-3-5-sonnet-20241022"
        )
        print(f"Anthropic: {response}")
    except Exception as e:
        print(f"Anthropic: Not available - {e}")

    print()

async def complex_tool():
    """More complex tool with multiple parameters."""
    print("=== Complex Tool ===\n")

    def search_database(
        query: str,
        limit: int = 10,
        include_archived: bool = False
    ) -> list:
        """Search the database with filters."""
        # Mock database search
        return [
            {"id": 1, "title": f"Result for: {query}", "archived": False},
            {"id": 2, "title": f"Another result for: {query}", "archived": False}
        ][:limit]

    toolkit = tools_from_functions(search_database)

    response = await ask(
        "Search for Python tutorials, show me 5 results",
        tools=toolkit.to_openai_format(),
        model="gpt-4o-mini"
    )

    print(f"Response: {response}\n")

if __name__ == "__main__":
    asyncio.run(basic_tool_usage())
    asyncio.run(single_tool())
    sync_tools()
    asyncio.run(multiple_providers())
    asyncio.run(complex_tool())

    print("="*50)
    print("✅ Tools are just parameters - simple and clean!")
    print("="*50)
