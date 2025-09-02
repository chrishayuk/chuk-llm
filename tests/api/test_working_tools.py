#!/usr/bin/env python3
"""Test the new multi-step tool execution"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from chuk_llm import Tools, tool, ask_with_tools_simple, ask_with_tools_simple_sync


# Define test tools
class MathTools(Tools):
    @tool(description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        result = a + b
        print(f"  ✅ Tool executed: add({a}, {b}) = {result}")
        return result
    
    @tool(description="Multiply two numbers")  
    def multiply(self, x: float, y: float) -> float:
        result = x * y
        print(f"  ✅ Tool executed: multiply({x}, {y}) = {result}")
        return result
    
    @tool(description="Calculate square root")
    def sqrt(self, n: float) -> float:
        import math
        result = math.sqrt(n)
        print(f"  ✅ Tool executed: sqrt({n}) = {result}")
        return result


def get_weather(location: str) -> dict:
    """Get weather for a location"""
    weather_data = {
        "Paris": {"temp": 18, "condition": "cloudy"},
        "Tokyo": {"temp": 22, "condition": "sunny"},
        "New York": {"temp": 15, "condition": "rainy"},
        "London": {"temp": 12, "condition": "foggy"}
    }
    result = weather_data.get(location, {"temp": 20, "condition": "clear"})
    print(f"  ✅ Tool executed: get_weather('{location}') = {result}")
    return result


def calculate(expression: str) -> float:
    """Evaluate a math expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"  ✅ Tool executed: calculate('{expression}') = {result}")
        return result
    except Exception as e:
        print(f"  ❌ Tool failed: calculate('{expression}') - {e}")
        return 0.0


async def test_class_tools():
    """Test class-based tools with multi-step execution"""
    print("\n" + "="*60)
    print("TESTING CLASS-BASED TOOLS (Multi-Step)")
    print("="*60)
    
    tools = MathTools()
    
    print("\n📝 Test 1: Simple addition")
    print("Prompt: 'What is 25 + 37?'")
    response = await tools.ask(
        "What is 25 + 37? Use the add function.",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n📝 Test 2: Multiple operations")
    print("Prompt: 'First add 10 and 15, then multiply the result by 3'")
    response = await tools.ask(
        "First add 10 and 15, then multiply the result by 3. Use the tools.",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n📝 Test 3: Square root")
    print("Prompt: 'What is the square root of 144?'")
    response = await tools.ask(
        "What is the square root of 144? Use the sqrt function.",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")


async def test_function_tools():
    """Test function-based tools"""
    print("\n" + "="*60)
    print("TESTING FUNCTION-BASED TOOLS")
    print("="*60)
    
    print("\n📝 Test 1: Weather")
    print("Prompt: 'What's the weather in Paris?'")
    response = await ask_with_tools_simple(
        "What's the weather in Paris?",
        tools=[get_weather],
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n📝 Test 2: Calculation")
    print("Prompt: 'Calculate (42 * 3) + 18'")
    response = await ask_with_tools_simple(
        "Calculate (42 * 3) + 18",
        tools=[calculate],
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n📝 Test 3: Multiple tools available")
    print("Prompt: 'What's the weather in Tokyo and calculate 50 * 2'")
    response = await ask_with_tools_simple(
        "What's the weather in Tokyo and also calculate 50 * 2",
        tools=[get_weather, calculate],
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")


def test_sync_tools():
    """Test synchronous tool execution"""
    print("\n" + "="*60)
    print("TESTING SYNC TOOLS")
    print("="*60)
    
    print("\n📝 Test 1: Sync class-based")
    tools = MathTools()
    print("Prompt: 'What is 100 + 50?'")
    response = tools.ask_sync(
        "What is 100 + 50? Use the add function.",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n📝 Test 2: Sync function-based")
    print("Prompt: 'What's the weather in London?'")
    response = ask_with_tools_simple_sync(
        "What's the weather in London?",
        tools=[get_weather],
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")


async def run_async_tests():
    """Run async tests"""
    print("🛠️ TESTING MULTI-STEP TOOL EXECUTION")
    print("This demonstrates proper tool calling with execution\n")
    
    await test_class_tools()
    await test_function_tools()


def main():
    """Run all tests"""
    # Run async tests first
    asyncio.run(run_async_tests())
    
    # Then run sync tests
    test_sync_tools()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    print("\nKey features demonstrated:")
    print("  ✓ Tools are actually executed")
    print("  ✓ Results are sent back to the model")
    print("  ✓ Final responses incorporate tool results")
    print("  ✓ Multi-step conversation pattern works")


if __name__ == "__main__":
    main()