#!/usr/bin/env python3
"""
Clean Tools API Example
=======================
Demonstrates the simplified tool/function calling API.

Note: Tool execution follows OpenAI's pattern where the model
returns which tools to call, but actual execution requires a
follow-up conversation to send results back.
"""

import asyncio
import sys
from dotenv import load_dotenv

from chuk_llm import Tools, tool, ask, ask_sync, tools_from_functions

# Load environment variables
load_dotenv()


# Define tools using the class-based approach
class MathTools(Tools):
    """Mathematical operations tools"""
    
    @tool(description="Add two numbers together")
    def add(self, a: int, b: int) -> int:
        """Add two numbers"""
        result = a + b
        print(f"  [Tool executed: add({a}, {b}) = {result}]")
        return result
    
    @tool(description="Multiply two numbers")  
    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers"""
        result = x * y
        print(f"  [Tool executed: multiply({x}, {y}) = {result}]")
        return result
    
    @tool
    def factorial(self, n: int) -> int:
        """Calculate factorial of a number"""
        result = 1
        for i in range(1, n + 1):
            result *= i
        print(f"  [Tool executed: factorial({n}) = {result}]")
        return result


# Define standalone tool functions
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather for a location"""
    print(f"  [Tool executed: get_weather('{location}', '{unit}')]")
    # Mock weather data
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny"
    }


def calculate_expression(expression: str) -> float:
    """Safely evaluate a mathematical expression"""
    print(f"  [Tool executed: calculate_expression('{expression}')]")
    try:
        # In production, use a proper expression parser
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except:
        return 0.0


async def demo_class_based_tools():
    """Demonstrate class-based tools"""
    print("\n" + "="*50)
    print("CLASS-BASED TOOLS DEMO")
    print("="*50)
    
    tools = MathTools()
    
    # The model will identify which tool to use
    print("\n1. Testing addition:")
    response = await tools.ask(
        "What is 15 + 27?",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n2. Testing multiplication:")
    response = await tools.ask(
        "Calculate 12 times 8",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n3. Testing factorial:")
    response = await tools.ask(
        "What is the factorial of 5? Use the factorial function.",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    # Disable auto-execution to see raw tool calls
    print("\n4. Without auto-execution (see raw response):")
    response = await tools.ask(
        "What is 10 + 5?",
        auto_execute=False,
        model="gpt-4o-mini"
    )
    print(f"Raw response: {response}")


async def demo_function_tools():
    """Demonstrate function-based tools"""
    print("\n" + "="*50)
    print("FUNCTION-BASED TOOLS DEMO")
    print("="*50)
    
    # Create toolkit from functions
    toolkit = tools_from_functions(get_weather, calculate_expression)
    
    print("\n1. With auto-execution (using toolkit):")
    response = await toolkit.ask(
        "What's the weather in Paris?",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n2. Without auto-execution (raw ask with tools):")
    response = await ask(
        "Calculate (15 + 27) * 2",
        tools=toolkit.to_openai_format(),
        model="gpt-4o-mini"
    )
    print(f"Response (dict with tool_calls): {response}")
    
    print("\n3. Multiple tools with execution:")
    response = await toolkit.ask(
        "What's the weather in Tokyo and calculate 50 divided by 2",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")


def demo_sync_tools():
    """Demonstrate synchronous tool usage"""
    
    print("\n1. Sync class-based:")
    tools = MathTools()
    response = tools.ask_sync(
        "What is 100 divided by 4? Please calculate using the multiply function with 25 and 4 to verify.",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")
    
    print("\n2. Sync function-based:")
    toolkit = tools_from_functions(get_weather)
    response = toolkit.ask_sync(
        "What's the weather in London?",
        model="gpt-4o-mini"
    )
    print(f"Response: {response}")


async def run_all_async():
    """Run all async demos"""
    print("🛠️ ChukLLM Clean Tools API Demo")
    print("Demonstrating simplified tool/function calling\n")
    
    await demo_class_based_tools()
    await demo_function_tools()
    
    print("\n📌 Note: Tool execution now works with multi-step conversation pattern!")
    print("Tools are executed and results are sent back to the model.")


def run_all_sync():
    """Run all sync demos in a separate process"""
    print("🛠️ ChukLLM Clean Tools API Demo - Sync Mode")
    print("Demonstrating simplified tool/function calling\n")
    
    demo_sync_tools()
    
    print("\n" + "="*50)
    print("✅ All sync demos completed!")
    print("="*50)


def main():
    """Main entry point - choose async or sync"""
    import sys
    import subprocess
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sync-only":
        # Run sync version only (called from subprocess)
        demo_sync_tools()
    else:
        # Run async version
        asyncio.run(run_all_async())
        
        # Then run sync in a subprocess to avoid event loop issues
        print("\n" + "="*50)
        print("SYNCHRONOUS TOOLS DEMO")
        print("="*50)
        
        result = subprocess.run(
            [sys.executable, __file__, "--sync-only"],
            capture_output=True,
            text=True
        )
        
        # Print the sync output
        print(result.stdout.strip())
        
        print("\n" + "="*50)
        print("✅ All demos completed!")
        print("="*50)


if __name__ == "__main__":
    main()