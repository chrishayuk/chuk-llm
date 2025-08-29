#!/usr/bin/env python3
"""
GPT-OSS Tool Calling Demo
==========================
Automated demonstration of GPT-OSS tool calling capabilities.
Shows real-world scenarios with simulated tool execution.
"""

import asyncio
import json
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ToolSimulator:
    """Simulates realistic tool execution results"""

    @staticmethod
    def execute(name: str, args: dict) -> str:
        """Execute tool and return realistic results"""

        if name == "get_weather":
            location = args.get("location", "Unknown")
            temps = {
                "Tokyo": 18,
                "New York": 12,
                "Paris": 15,
                "London": 10,
                "Sydney": 25,
            }
            temp = temps.get(location, random.randint(10, 30))
            conditions = ["Sunny", "Partly cloudy", "Cloudy", "Light rain"]
            return f"{temp}°C, {random.choice(conditions)}"

        elif name == "calculate":
            expr = args.get("expression", "")
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return str(result)
            except:
                return "Error"

        elif name == "query_database":
            query = args.get("query", "")
            if "COUNT" in query.upper():
                return str(random.randint(100, 5000))
            elif "SELECT" in query.upper():
                return f"{random.randint(10, 100)} rows returned"
            return "Query executed"

        elif name == "web_search":
            query = args.get("query", "")
            return f"Found {random.randint(100, 10000)} results for '{query}'"

        elif name == "send_email":
            to = args.get("to", "")
            return f"Email sent to {to}"

        elif name == "read_file":
            filepath = args.get("filepath", "")
            if "config" in filepath:
                return (
                    '{"database": "production", "port": 5432, "host": "db.example.com"}'
                )
            return f"File content from {filepath}"

        elif name == "call_api":
            endpoint = args.get("endpoint", "")
            if "users" in endpoint:
                return '{"count": 1547, "active": 1203}'
            return '{"status": "success"}'

        elif name == "schedule_meeting":
            return f"Meeting scheduled for {args.get('date', 'tomorrow')}"

        return f"Executed {name}"


async def run_scenario(
    name: str, prompt: str, tools: list, scenario_num: int, total: int
):
    """Execute a single scenario"""

    from chuk_llm import stream

    print(f"\n{'=' * 70}")
    print(f"📍 Scenario {scenario_num}/{total}: {name}")
    print(f"{'=' * 70}")
    print(f"📝 Task: {prompt}\n")

    # Show available tools
    print("🛠️  Available tools:")
    for tool in tools:
        func_name = tool["function"]["name"]
        desc = tool["function"]["description"]
        print(f"   • {func_name}: {desc}")

    print("\n🤖 GPT-OSS Processing...")
    print("-" * 50)

    # Collect results
    chunks_received = 0
    tool_calls = []
    response_text = []
    thinking_text = []
    start_time = asyncio.get_event_loop().time()

    try:
        async for chunk in stream(
            prompt,
            provider="ollama",
            model="gpt-oss:latest",
            tools=tools,
            return_tool_calls=True,
            temperature=0.2,
            max_tokens=400,
        ):
            chunks_received += 1

            if isinstance(chunk, dict):
                # Collect response
                if chunk.get("response"):
                    text = chunk["response"]

                    # Check if thinking
                    if chunk.get("reasoning", {}).get("is_thinking"):
                        thinking_text.append(text)
                    else:
                        response_text.append(text)
                        # Show response in real-time (non-thinking only)
                        if len(response_text) <= 10:  # First few chunks
                            print(text, end="", flush=True)

                # Collect tool calls
                if chunk.get("tool_calls"):
                    for tc in chunk["tool_calls"]:
                        if tc not in tool_calls:
                            tool_calls.append(tc)
            else:
                # String response
                response_text.append(str(chunk))
                if len(response_text) <= 10:
                    print(str(chunk), end="", flush=True)

        # Complete the response line
        if response_text:
            full_response = "".join(response_text)
            if len(full_response) > 200:
                print("...[truncated]")
            else:
                print()  # New line after response

        elapsed = asyncio.get_event_loop().time() - start_time

        print("-" * 50)

        # Process tool calls
        if tool_calls:
            print("\n🔧 Tool Execution:")
            simulator = ToolSimulator()

            for i, tc in enumerate(tool_calls, 1):
                func = tc.get("function", {})
                func_name = func.get("name", "unknown")
                func_args_str = func.get("arguments", "{}")

                try:
                    func_args = (
                        json.loads(func_args_str)
                        if isinstance(func_args_str, str)
                        else func_args_str
                    )
                except:
                    func_args = {}

                print(
                    f"\n   Call {i}: {func_name}({json.dumps(func_args, separators=(',', ':'))})"
                )

                # Simulate execution
                result = simulator.execute(func_name, func_args)
                print(f"   → Result: {result}")

        # Show thinking summary if present
        if thinking_text:
            thinking_full = "".join(thinking_text)
            print(
                f"\n💭 Reasoning: {len(thinking_full)} chars of thinking process captured"
            )

        # Summary
        print("\n📊 Performance:")
        print(f"   • Time: {elapsed:.1f}s")
        print(f"   • Chunks: {chunks_received}")
        print(f"   • Tool calls: {len(tool_calls)}")

        if tool_calls:
            tools_used = list({tc["function"]["name"] for tc in tool_calls})
            print(f"   • Tools used: {', '.join(tools_used)}")
            print("   ✅ Success: Task completed with tools")
        else:
            print("   ⚠️  No tool calls made")

        return len(tool_calls) > 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def main():
    """Run the automated demo"""

    print("🚀 GPT-OSS TOOL CALLING DEMONSTRATION")
    print("=" * 70)
    print("Automated demonstration showing GPT-OSS reasoning model")
    print("performing real-world tasks with function calling.\n")

    # Define scenarios
    scenarios = [
        {
            "name": "Weather Comparison",
            "prompt": "What's the weather in Tokyo and New York? Calculate the temperature difference.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Math expression to evaluate",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                },
            ],
        },
        {
            "name": "Database Analytics",
            "prompt": "Query the users table to count active users, then send a summary email to admin@company.com",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "query_database",
                        "description": "Execute SQL query on database",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "SQL query to execute",
                                },
                                "table": {
                                    "type": "string",
                                    "description": "Target table name",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "description": "Send an email message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "to": {
                                    "type": "string",
                                    "description": "Recipient email address",
                                },
                                "subject": {
                                    "type": "string",
                                    "description": "Email subject line",
                                },
                                "body": {
                                    "type": "string",
                                    "description": "Email body content",
                                },
                            },
                            "required": ["to", "subject", "body"],
                        },
                    },
                },
            ],
        },
        {
            "name": "Research Task",
            "prompt": "Search for 'quantum computing breakthroughs 2024' and summarize findings",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query terms",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum results to return",
                                    "default": 10,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        },
        {
            "name": "Configuration Management",
            "prompt": "Read the database configuration from config.json and connect to verify it's working",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {
                                    "type": "string",
                                    "description": "Path to the file",
                                }
                            },
                            "required": ["filepath"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "query_database",
                        "description": "Execute SQL query on database",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "SQL query to execute",
                                },
                                "database": {
                                    "type": "string",
                                    "description": "Database name",
                                    "default": "main",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
            ],
        },
        {
            "name": "API Monitoring",
            "prompt": "Check the user count from /api/users endpoint. If over 1000, schedule a scaling meeting for tomorrow.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "call_api",
                        "description": "Make an API request",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "endpoint": {
                                    "type": "string",
                                    "description": "API endpoint URL",
                                },
                                "method": {
                                    "type": "string",
                                    "description": "HTTP method",
                                    "default": "GET",
                                },
                            },
                            "required": ["endpoint"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "schedule_meeting",
                        "description": "Schedule a meeting",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Meeting title",
                                },
                                "date": {
                                    "type": "string",
                                    "description": "Meeting date/time",
                                },
                                "attendees": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of attendee emails",
                                },
                            },
                            "required": ["title", "date", "attendees"],
                        },
                    },
                },
            ],
        },
    ]

    # Run all scenarios
    successful = 0
    total = len(scenarios)

    for i, scenario in enumerate(scenarios, 1):
        success = await run_scenario(
            scenario["name"], scenario["prompt"], scenario["tools"], i, total
        )
        if success:
            successful += 1

        # Brief pause between scenarios
        if i < total:
            await asyncio.sleep(1)

    # Final summary
    print(f"\n{'=' * 70}")
    print("📊 FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"✅ Successful scenarios: {successful}/{total}")

    if successful == total:
        print("🎉 Perfect score! All scenarios completed successfully.")
    elif successful >= total * 0.8:
        print("👍 Excellent performance! Most scenarios worked well.")
    elif successful >= total * 0.6:
        print("👌 Good performance. Some scenarios need attention.")
    else:
        print("⚠️  Needs improvement. Check model configuration.")

    print("\n🏁 Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
