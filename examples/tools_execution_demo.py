#!/usr/bin/env python3
"""Demo of the multi-step tool execution with ChukLLM"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from chuk_llm import Tools, tool
from chuk_llm.api.tools import tools_from_functions


# Define demo tools using class-based approach
class MathTools(Tools):
    @tool(description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        result = a + b
        print(f"  ✅ Tool executed: add({a}, {b}) = {result}")
        return result
    
    @tool(description="Multiply two numbers")  
    def multiply(self, a: int, b: int) -> int:
        result = a * b
        print(f"  ✅ Tool executed: multiply({a}, {b}) = {result}")
        return result
    
    @tool(description="Calculate square root")
    def sqrt(self, n: float) -> float:
        import math
        result = math.sqrt(n)
        print(f"  ✅ Tool executed: sqrt({n}) = {result}")
        return result


# Function-based tools for demonstration
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
        # Safe evaluation with no builtins
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"  ✅ Tool executed: calculate('{expression}') = {result}")
        return result
    except Exception as e:
        print(f"  ❌ Tool failed: calculate('{expression}') - {e}")
        return 0.0


async def demo_class_tools():
    """Demo class-based tools with multi-step execution"""
    print("\n" + "="*60)
    print("DEMO: CLASS-BASED TOOLS (Multi-Step)")
    print("="*60)
    
    tools = MathTools()
    
    print("\n📝 Demo 1: Simple addition")
    print("Prompt: 'What is 25 + 37?'")
    try:
        response = await tools.ask(
            "What is 25 + 37? Use the add function.",
            model="gpt-4o-mini"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📝 Demo 2: Multi-step calculation")
    print("Prompt: 'Calculate (10 + 15) * 3'")
    try:
        response = await tools.ask(
            "Calculate (10 + 15) * 3. First add 10 and 15, then multiply the result by 3.",
            model="gpt-4o-mini"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📝 Demo 3: Square root calculation")
    print("Prompt: 'What's the square root of 144?'")
    try:
        response = await tools.ask(
            "What's the square root of 144?",
            model="gpt-4o-mini"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_function_tools():
    """Demo function-based tools"""
    print("\n" + "="*60)
    print("DEMO: FUNCTION-BASED TOOLS")
    print("="*60)
    
    print("\n📝 Demo 1: Weather")
    print("Prompt: 'What's the weather in Paris?'")
    try:
        toolkit = tools_from_functions(get_weather)
        response = await toolkit.ask(
            "What's the weather in Paris?",
            model="gpt-4o-mini",
            auto_execute=True
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📝 Demo 2: Calculation")
    print("Prompt: 'Calculate (42 * 3) + 18'")
    try:
        toolkit = tools_from_functions(calculate)
        response = await toolkit.ask(
            "Calculate (42 * 3) + 18",
            model="gpt-4o-mini",
            auto_execute=True
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📝 Demo 3: Multiple tools available")
    print("Prompt: 'What's the weather in Tokyo and calculate 50 * 2'")
    try:
        toolkit = tools_from_functions(get_weather, calculate)
        response = await toolkit.ask(
            "What's the weather in Tokyo and also calculate 50 * 2",
            model="gpt-4o-mini",
            auto_execute=True
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_sync_tools():
    """Demo synchronous tool execution"""
    print("\n" + "="*60)
    print("DEMO: SYNC TOOLS")
    print("="*60)
    
    print("\n📝 Demo 1: Sync class-based")
    tools = MathTools()
    print("Prompt: 'What is 100 + 50?'")
    try:
        response = tools.ask_sync(
            "What is 100 + 50? Use the add function.",
            model="gpt-4o-mini"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📝 Demo 2: Sync function-based")
    print("Prompt: 'What's the weather in London?'")
    try:
        toolkit = tools_from_functions(get_weather)
        response = toolkit.ask_sync(
            "What's the weather in London?",
            model="gpt-4o-mini",
            auto_execute=True
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def run_async_demos():
    """Run async demos"""
    print("🛠️ DEMONSTRATING MULTI-STEP TOOL EXECUTION")
    print("This demonstrates proper tool calling with execution\n")
    
    await demo_class_tools()
    await demo_function_tools()
    
    # Small delay to allow cleanup
    await asyncio.sleep(0.1)


def main():
    """Main entry point"""
    import sys
    
    print("\n" + "="*60)
    print("🎯 ChukLLM Tool Execution Demo")
    print("="*60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("This demo requires an OpenAI API key to run")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Run async demos with proper cleanup
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_async_demos())
    finally:
        # Suppress stderr during cleanup
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            loop.run_until_complete(asyncio.sleep(0))
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except:
            pass
        finally:
            loop.close()
            sys.stderr.close()
            sys.stderr = old_stderr
    
    # Run sync demos
    demo_sync_tools()
    
    print("\n" + "="*60)
    print("✅ All tool demonstrations complete!")
    print("="*60)


if __name__ == "__main__":
    main()