#!/usr/bin/env python3
"""Minimal test for tool execution"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm import Tools, tool

class MathTools(Tools):
    @tool(description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        print(f"  ✅ Tool executed: add({a}, {b}) = {a + b}")
        return a + b

async def test():
    tools = MathTools()
    print("Testing: What is 5 + 3?")
    response = await tools.ask("What is 5 + 3?", model="gpt-4o-mini")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test())