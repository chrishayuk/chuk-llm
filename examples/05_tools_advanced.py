#!/usr/bin/env python3
"""
Advanced Tool Calling with Auto-Execution
==========================================

Demonstrates the Tools class that automatically executes
tools and returns the final response.

This is the most convenient way to use tools!
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import Tools, tool

# Method 1: Class-based tools with auto-execution
class WeatherTools(Tools):
    """Tools for weather information."""

    @tool(description="Get current weather for a location")
    def get_weather(self, location: str, unit: str = "celsius") -> dict:
        """Get current weather information."""
        print(f"  🔧 Tool called: get_weather({location}, {unit})")
        return {
            "location": location,
            "temperature": 22,
            "unit": unit,
            "condition": "sunny"
        }

    @tool(description="Get weather forecast")
    def get_forecast(self, location: str, days: int = 3) -> dict:
        """Get weather forecast for upcoming days."""
        print(f"  🔧 Tool called: get_forecast({location}, {days} days)")
        return {
            "location": location,
            "forecast": [
                {"day": 1, "temp": 23, "condition": "sunny"},
                {"day": 2, "temp": 21, "condition": "cloudy"},
                {"day": 3, "temp": 24, "condition": "partly cloudy"}
            ][:days]
        }

async def auto_execution():
    """Tools with automatic execution."""
    print("=== Auto-Execution ===\n")

    tools = WeatherTools()

    # The Tools.ask() method automatically:
    # 1. Sends prompt with tools
    # 2. Executes any tool calls
    # 3. Sends results back to LLM
    # 4. Returns final response

    response = await tools.ask(
        "What's the weather in London and show me the 3-day forecast?",
        model="gpt-4o-mini"
    )

    print(f"\nFinal response: {response}\n")

# Method 2: Multiple tool categories
class MathTools(Tools):
    """Mathematical operations."""

    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        print(f"  🔧 Tool called: add({a}, {b})")
        return a + b

    @tool
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        print(f"  🔧 Tool called: multiply({a}, {b})")
        return a * b

    @tool
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        print(f"  🔧 Tool called: power({base}^{exponent})")
        return base ** exponent

async def multiple_tool_calls():
    """LLM can make multiple tool calls."""
    print("=== Multiple Tool Calls ===\n")

    tools = MathTools()

    response = await tools.ask(
        "Calculate (5 + 3) * 2, then raise the result to the power of 2",
        model="gpt-4o-mini"
    )

    print(f"\nFinal response: {response}\n")

# Method 3: Combining multiple tool classes
class DatabaseTools(Tools):
    """Database operations."""

    def __init__(self):
        super().__init__()
        self.db = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "products": [
                {"id": 1, "name": "Widget", "price": 9.99},
                {"id": 2, "name": "Gadget", "price": 19.99}
            ]
        }

    @tool
    def query_table(self, table: str, limit: int = 10) -> list:
        """Query a database table."""
        print(f"  🔧 Tool called: query_table({table}, limit={limit})")
        return self.db.get(table, [])[:limit]

    @tool
    def search_users(self, name: str) -> list:
        """Search users by name."""
        print(f"  🔧 Tool called: search_users({name})")
        results = [u for u in self.db["users"] if name.lower() in u["name"].lower()]
        return results

async def stateful_tools():
    """Tools can maintain state."""
    print("=== Stateful Tools ===\n")

    tools = DatabaseTools()

    response = await tools.ask(
        "Show me all users, then search for Alice",
        model="gpt-4o-mini"
    )

    print(f"\nFinal response: {response}\n")

# Method 4: Error handling in tools
class SafeTools(Tools):
    """Tools with error handling."""

    @tool
    def divide(self, a: float, b: float) -> str:
        """Divide two numbers."""
        print(f"  🔧 Tool called: divide({a}, {b})")
        try:
            result = a / b
            return f"Result: {result}"
        except ZeroDivisionError:
            return "Error: Cannot divide by zero!"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def safe_eval(self, expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        print(f"  🔧 Tool called: safe_eval({expression})")
        try:
            # In production, use a safe eval library!
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

async def error_handling():
    """Tools handle errors gracefully."""
    print("=== Error Handling ===\n")

    tools = SafeTools()

    response = await tools.ask(
        "Calculate 10 / 0 and 10 / 2",
        model="gpt-4o-mini"
    )

    print(f"\nFinal response: {response}\n")

# Method 5: Streaming with tools
async def streaming_with_tools():
    """Stream responses while executing tools."""
    print("=== Streaming with Tools ===\n")

    tools = MathTools()

    print("Response: ", end="", flush=True)

    # Tools.stream() also auto-executes
    async for chunk in tools.stream(
        "Calculate 7 * 8, then add 10 to the result",
        model="gpt-4o-mini"
    ):
        print(chunk, end="", flush=True)

    print("\n")

if __name__ == "__main__":
    asyncio.run(auto_execution())
    asyncio.run(multiple_tool_calls())
    asyncio.run(stateful_tools())
    asyncio.run(error_handling())
    asyncio.run(streaming_with_tools())

    print("="*50)
    print("✅ Auto-execution makes tool calling effortless!")
    print("="*50)
