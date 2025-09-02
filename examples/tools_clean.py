#!/usr/bin/env python3
"""
Clean Tool Calling Examples
===========================

Shows the simplified, developer-friendly API for function calling.
"""

import asyncio
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from chuk_llm.api.tools import (
    tool,
    Tools,
    ToolKit,
    create_tool,
    tools_from_functions,
    ask_with_tools_simple
)


# ============================================================================
# Method 1: Decorator-based class approach (Cleanest)
# ============================================================================

class WeatherTools(Tools):
    """Clean class-based tool definition"""
    
    @tool(description="Get current weather for a city")
    def get_weather(self, location: str, unit: str = "celsius") -> Dict:
        """Get weather information"""
        # In real app, this would call a weather API
        weather_data = {
            "Paris": {"temp": 22, "condition": "sunny"},
            "London": {"temp": 18, "condition": "cloudy"},
            "New York": {"temp": 25, "condition": "clear"},
        }
        data = weather_data.get(location, {"temp": 20, "condition": "unknown"})
        data["unit"] = unit
        data["location"] = location
        return data
    
    @tool
    def get_forecast(self, location: str, days: int = 3) -> List[Dict]:
        """Get weather forecast for multiple days"""
        return [
            {"day": i, "temp": 20 + i, "location": location}
            for i in range(1, days + 1)
        ]


# ============================================================================
# Method 2: Simple function approach
# ============================================================================

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression"""
    # In production, use a safe expression evaluator
    try:
        return eval(expression)
    except:
        return 0.0


def get_time(timezone: str = "UTC") -> str:
    """Get current time in a timezone"""
    from datetime import datetime
    return f"Current time in {timezone}: {datetime.now().isoformat()}"


# ============================================================================
# Method 3: Explicit tool creation
# ============================================================================

def create_search_tool():
    """Create a search tool with explicit parameters"""
    return create_tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        func=lambda query, num_results=5: {
            "results": [f"Result {i} for '{query}'" for i in range(1, num_results + 1)]
        }
    )


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_class_based():
    """Demo the class-based approach"""
    print("\n🎯 Class-Based Tools Demo")
    print("-" * 40)
    
    tools = WeatherTools()
    
    # The AI can now use the tools
    response = await tools.ask("What's the weather in Paris?")
    print(f"Response: {response}")
    
    response = await tools.ask("Give me a 5-day forecast for London")
    print(f"Forecast: {response}")


async def demo_simple_functions():
    """Demo the simple function approach"""
    print("\n🔧 Simple Functions Demo")
    print("-" * 40)
    
    # Super simple - just pass functions
    response = await ask_with_tools_simple(
        "What's 25 * 4 + 10?",
        tools=[calculate, get_time]
    )
    print(f"Calculation: {response}")
    
    response = await ask_with_tools_simple(
        "What time is it in EST?",
        tools=[calculate, get_time]
    )
    print(f"Time: {response}")


async def demo_toolkit():
    """Demo the ToolKit approach"""
    print("\n🧰 ToolKit Demo")
    print("-" * 40)
    
    # Create a toolkit from functions
    toolkit = tools_from_functions(calculate, get_time, get_weather)
    
    response = await toolkit.ask("Calculate 15% of 200")
    print(f"Result: {response}")


def demo_sync():
    """Demo synchronous tool usage"""
    print("\n🔄 Sync Tools Demo")
    print("-" * 40)
    
    tools = WeatherTools()
    response = tools.ask_sync("What's the weather in New York?")
    print(f"Sync response: {response}")


async def demo_mixed_tools():
    """Demo mixing different tool creation methods"""
    print("\n🎨 Mixed Tools Demo")
    print("-" * 40)
    
    # Create a toolkit and add tools from different sources
    toolkit = ToolKit("mixed")
    
    # Add functions
    toolkit.add_function(calculate)
    toolkit.add_function(get_time)
    
    # Add explicit tool
    toolkit.add(create_search_tool())
    
    # Use the combined toolkit
    response = await toolkit.ask("Search for 'Python tutorials'")
    print(f"Search: {response}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos"""
    print("=" * 50)
    print("CLEAN TOOL CALLING API EXAMPLES")
    print("=" * 50)
    
    try:
        # Run async demos
        await demo_class_based()
        await demo_simple_functions()
        await demo_toolkit()
        await demo_mixed_tools()
        
        # Run sync demo
        demo_sync()
        
    except Exception as e:
        print(f"\n⚠️  Note: These examples require an OpenAI-compatible provider")
        print(f"   Error: {e}")
        print(f"   Set OPENAI_API_KEY to run the examples")
    
    print("\n" + "=" * 50)
    print("✅ Tool calling is now clean and simple!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())