#!/usr/bin/env python3
"""
GPT-OSS Multi-Step Conversation Demo
=====================================
Demonstrates proper multi-step tool calling with conversation continuation.
Shows how reasoning models work step-by-step through complex tasks.
"""

import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ConversationAgent:
    """Agent that manages multi-step conversations with tool execution"""

    def __init__(self, model: str = "gpt-oss:latest"):
        self.model = model
        self.conversation_history = []
        self.tool_execution_log = []

    async def execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool and return results"""

        # Log execution
        self.tool_execution_log.append(
            {"tool": name, "args": args, "timestamp": datetime.now().isoformat()}
        )

        # Simulate different tools
        if name == "get_weather":
            location = args.get("location", "Unknown")
            temps = {
                "Tokyo": 18,
                "New York": 12,
                "Paris": 15,
                "London": 10,
                "Sydney": 25,
                "Singapore": 30,
                "Berlin": 8,
                "Moscow": -5,
            }
            temp = temps.get(location, random.randint(10, 25))
            conditions = ["Sunny", "Partly cloudy", "Cloudy", "Light rain"]
            condition = random.choice(conditions)
            return f"Weather in {location}: {temp}°C, {condition}"

        elif name == "calculate":
            expression = args.get("expression", "")
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except Exception as e:
                return f"Calculation error: {e}"

        elif name == "search_flights":
            origin = args.get("origin", "")
            destination = args.get("destination", "")
            date = args.get("date", "")
            flights = random.randint(3, 8)
            price = random.randint(200, 800)
            return f"Found {flights} flights from {origin} to {destination} on {date}, starting at ${price}"

        elif name == "book_hotel":
            city = args.get("city", "")
            checkin = args.get("checkin", "")
            checkout = args.get("checkout", "")
            hotels = random.randint(5, 15)
            price = random.randint(80, 300)
            return f"Found {hotels} hotels in {city} from {checkin} to {checkout}, starting at ${price}/night"

        elif name == "send_email":
            to = args.get("to", "")
            subject = args.get("subject", "")
            return f"Email sent to {to} with subject: {subject}"

        elif name == "create_calendar_event":
            title = args.get("title", "")
            date = args.get("date", "")
            return f"Calendar event '{title}' created for {date}"

        else:
            return f"Executed {name} with args {args}"

    async def run_conversation(
        self, initial_prompt: str, tools: list[dict], max_rounds: int = 10
    ):
        """Run a multi-step conversation with tool execution"""

        import ollama

        print(f"\n{'=' * 70}")
        print("🤖 Starting Conversation")
        print(f"{'=' * 70}")
        print(f"Goal: {initial_prompt}\n")

        client = ollama.AsyncClient()
        messages = [{"role": "user", "content": initial_prompt}]

        for round_num in range(1, max_rounds + 1):
            print(f"📍 Step {round_num}:")

            # Show what we're sending (abbreviated)
            last_msg = messages[-1]["content"]
            if len(last_msg) > 150:
                print(f"   User: {last_msg[:150]}...")
            else:
                print(f"   User: {last_msg}")

            # Get model response
            response = await client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                options={"temperature": 0.2, "num_predict": 500},
            )

            if not hasattr(response, "message") or not response.message:
                print("   ❌ No response from model")
                break

            msg = response.message

            # Get response content
            content = getattr(msg, "content", "")
            if content:
                if len(content) > 150:
                    print(f"   Assistant: {content[:150]}...")
                else:
                    print(f"   Assistant: {content}")
                messages.append({"role": "assistant", "content": content})

            # Check for tool calls
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
                # Execute each tool call
                results = []
                for tc in tool_calls:
                    if hasattr(tc, "function"):
                        func = tc.function
                        name = getattr(func, "name", "unknown")
                        args = getattr(func, "arguments", {})

                        print(
                            f"   🔧 Calling: {name}({json.dumps(args, separators=(',', ':'))})"
                        )

                        # Execute tool
                        result = await self.execute_tool(name, args)
                        results.append(result)
                        print(f"   📊 Result: {result}")

                # Add results to conversation
                if results:
                    result_msg = "Tool results:\n" + "\n".join(
                        f"- {r}" for r in results
                    )
                    result_msg += "\n\nPlease continue with the task."
                    messages.append({"role": "user", "content": result_msg})
            else:
                # No tool calls, check if task is complete
                if any(
                    word in content.lower()
                    for word in ["complete", "finished", "done", "summary"]
                ):
                    print(f"\n✅ Task completed in {round_num} steps!")
                    break
                else:
                    # Prompt to continue
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue with the next step.",
                        }
                    )

        # Show execution summary
        self._show_summary()

    def _show_summary(self):
        """Show summary of tool executions"""

        if self.tool_execution_log:
            print("\n📊 Execution Summary:")
            print(f"   Total tool calls: {len(self.tool_execution_log)}")

            # Group by tool
            tool_counts = {}
            for entry in self.tool_execution_log:
                tool = entry["tool"]
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

            print("   Tools used:")
            for tool, count in tool_counts.items():
                print(f"     • {tool}: {count} call(s)")


async def demo_travel_planning():
    """Demo: Complex travel planning with multiple tools"""

    agent = ConversationAgent()

    prompt = """Plan a 3-day trip from New York to Tokyo:
    1. Check the weather in Tokyo
    2. Search for flights from New York to Tokyo
    3. Find hotels in Tokyo
    4. Create a calendar event for the trip
    5. Send a confirmation email
    Please execute each step using the appropriate tools."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search for flights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["origin", "destination", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_hotel",
                "description": "Search for hotels",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "checkin": {"type": "string"},
                        "checkout": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["title", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject"],
                },
            },
        },
    ]

    await agent.run_conversation(prompt, tools)


async def demo_data_analysis():
    """Demo: Multi-step data analysis with calculations"""

    agent = ConversationAgent()

    prompt = """Analyze weather data for multiple cities:
    1. Get weather for Tokyo, New York, and Paris
    2. Calculate the average temperature
    3. Find the warmest and coldest cities
    4. Send a summary report via email"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email report",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        },
    ]

    await agent.run_conversation(prompt, tools)


async def demo_simple_sequential():
    """Demo: Simple sequential task"""

    agent = ConversationAgent()

    prompt = "Get the weather for Tokyo, then New York, then Paris. Call get_weather for each city one at a time."

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    await agent.run_conversation(prompt, tools)


async def main():
    """Run demonstrations"""

    print("🚀 GPT-OSS MULTI-STEP CONVERSATION DEMONSTRATIONS")
    print("=" * 70)
    print("Showing how reasoning models handle complex multi-step tasks")
    print("through conversation continuation and tool execution.\n")

    demos = [
        ("Simple Sequential", demo_simple_sequential),
        ("Data Analysis", demo_data_analysis),
        ("Travel Planning", demo_travel_planning),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'=' * 70}")
        print(f"DEMO {i}: {name}")
        print(f"{'=' * 70}")

        try:
            await demo_func()
        except Exception as e:
            print(f"❌ Demo error: {e}")

        if i < len(demos):
            print("\nPress Enter for next demo...")
            input()

    print(f"\n{'=' * 70}")
    print("DEMONSTRATIONS COMPLETE")
    print(f"{'=' * 70}")
    print("\nKey Takeaways:")
    print("1. GPT-OSS executes one tool at a time (by design)")
    print("2. Conversation continuation enables complex workflows")
    print("3. Each tool result informs the next step")
    print("4. This step-by-step approach shows real reasoning")
    print("\nThis is how production agents should work with reasoning models!")


if __name__ == "__main__":
    asyncio.run(main())
