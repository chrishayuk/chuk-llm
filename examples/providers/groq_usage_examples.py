#!/usr/bin/env python3
"""
Groq Provider Example Usage Script
==================================

Demonstrates all features of the Groq provider using standardized demos.

Requirements:
- pip install chuk-llm
- Set GROQ_API_KEY environment variable

Usage:
    python groq_usage_examples.py
    python groq_usage_examples.py --model llama-3.3-70b-versatile
    python groq_usage_examples.py --skip-tools
    python groq_usage_examples.py --skip-vision
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("GROQ_API_KEY"):
    print("❌ Please set GROQ_API_KEY environment variable")
    print("   export GROQ_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.configuration import get_config
    from chuk_llm.llm.client import get_client

    # Import common demos
    examples_dir = Path(__file__).parent.parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from common_demos import (
        demo_basic_completion,
        demo_streaming,
        demo_function_calling,
        demo_vision,
        demo_json_mode,
        demo_reasoning,
        demo_structured_outputs,
        demo_conversation,
        demo_model_discovery,
        demo_error_handling,
        run_all_demos,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


async def main():
    """Run all Groq demos"""
    parser = argparse.ArgumentParser(description="Groq Provider Examples")
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Model to use (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--skip-tools", action="store_true", help="Skip function calling demo"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision demo"
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=json, 6=reasoning, 7=structured, 8=conversation, 9=discovery, 10=errors)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"🚀 Groq Provider Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('GROQ_API_KEY')[:20]}...")
    print("=" * 70)

    client = get_client("groq", model=args.model)

    # Run specific demo or all demos
    if args.demo:
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "groq", args.model)),
            2: ("Streaming", demo_streaming(client, "groq", args.model)),
            3: ("Function Calling", demo_function_calling(client, "groq", args.model)),
            4: ("Vision", demo_vision(client, "groq", args.model)),
            5: ("JSON Mode", demo_json_mode(client, "groq", args.model)),
            6: ("Reasoning", demo_reasoning(client, "groq", args.model)),
            7: ("Structured Outputs", demo_structured_outputs(client, "groq", args.model)),
            8: ("Conversation", demo_conversation(client, "groq", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "groq", args.model)),
            10: ("Error Handling", demo_error_handling(client, "groq", args.model)),
        }

        name, demo_coro = demo_map[args.demo]
        try:
            await demo_coro
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run all demos
        await run_all_demos(
            client,
            "groq",
            args.model,
            skip_tools=args.skip_tools,
            skip_vision=args.skip_vision
        )

    print("\n" + "=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)
    print("\nℹ️  Tips:")
    print("  - Use --demo N to run specific demo")
    print("  - Use --skip-tools to skip function calling")
    print("  - Use --skip-vision to skip vision demo")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
