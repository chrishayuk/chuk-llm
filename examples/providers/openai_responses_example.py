#!/usr/bin/env python3
"""
OpenAI Responses API - Comprehensive Example
=============================================

Complete demonstration using the Chat Completions API with all features.
Shows stateful conversations, function calling, JSON mode, and more.

Requirements:
- Set OPENAI_API_KEY environment variable

Usage:
    python openai_responses_example.py
    python openai_responses_example.py --demo 1
    python openai_responses_example.py --model gpt-4o
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.llm.client import get_client
    from chuk_llm.core.models import Message, Tool, ToolFunction, TextContent, ImageUrlContent
    from chuk_llm.core.enums import MessageRole, ToolType, ContentType
    from chuk_llm.configuration import Feature, get_config

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
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


async def main():
    """
    Comprehensive OpenAI example using standard Chat Completions API.
    """
    parser = argparse.ArgumentParser(description="OpenAI Responses Example")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=json, 6=reasoning, 7=structured, 8=conversation, 9=discovery, 10=errors)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🤖 OpenAI Comprehensive Example")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('OPENAI_API_KEY')[:20]}...")
    print("=" * 70)

    # Create OpenAI client
    client = get_client("openai", model="gpt-4o-mini")

    client = get_client("openai", model=args.model)

    # Run specific demo or all demos
    if args.demo:
        # Create specialized clients for specific demos
        vision_client = get_client("openai", model="gpt-4o")  # Vision requires gpt-4o
        reasoning_client = get_client("openai", model="gpt-5-mini")  # Reasoning demo

        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "openai", args.model)),
            2: ("Streaming", demo_streaming(client, "openai", args.model)),
            3: ("Function Calling", demo_function_calling(client, "openai", args.model)),
            4: ("Vision", demo_vision(vision_client, "openai", "gpt-4o")),  # Use vision client
            5: ("JSON Mode", demo_json_mode(client, "openai", args.model)),
            6: ("Reasoning", demo_reasoning(reasoning_client, "openai", "gpt-5-mini")),  # Use reasoning client
            7: ("Structured Outputs", demo_structured_outputs(client, "openai", args.model)),
            8: ("Conversation", demo_conversation(client, "openai", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "openai", args.model)),
            10: ("Error Handling", demo_error_handling(client, "openai", args.model)),
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
        demos = [
            ("Basic Completion", demo_basic_completion(client, "openai", args.model)),
            ("Streaming", demo_streaming(client, "openai", args.model)),
            ("Function Calling", demo_function_calling(client, "openai", args.model)),
            ("Vision", demo_vision(client, "openai", "gpt-4o")),
            ("JSON Mode", demo_json_mode(client, "openai", args.model)),
            ("Reasoning", demo_reasoning(client, "openai", "gpt-5-mini")),
            ("Structured Outputs", demo_structured_outputs(client, "openai", args.model)),
            ("Conversation", demo_conversation(client, "openai", args.model)),
            ("Model Discovery", demo_model_discovery(client, "openai", args.model)),
            ("Error Handling", demo_error_handling(client, "openai", args.model)),
        ]

        for name, demo_coro in demos:
            try:
                await demo_coro
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)
    print("\nℹ️  Tips:")
    print("  - Use --demo N to run specific demo")
    print("  - Use --model gpt-4o for vision support")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
