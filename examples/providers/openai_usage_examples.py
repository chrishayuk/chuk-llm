#!/usr/bin/env python3
"""
OpenAI Provider Example Usage Script
=====================================

Demonstrates all features of the OpenAI provider using standardized demos.
Includes GPT-4, GPT-5, o1/o3 reasoning models, vision, audio, and function calling.

Requirements:
- pip install chuk-llm
- Set OPENAI_API_KEY environment variable

Usage:
    python openai_usage_examples.py
    python openai_usage_examples.py --model gpt-4o
    python openai_usage_examples.py --skip-tools
    python openai_usage_examples.py --skip-vision
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your_api_key_here'")
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
        demo_audio_input,
        demo_parameters,
        demo_model_comparison,
        demo_dynamic_model_call,
        demo_error_handling,
        run_all_demos,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


async def main():
    """Run all OpenAI demos"""
    parser = argparse.ArgumentParser(description="OpenAI Provider Examples")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
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
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=json, 6=reasoning, 7=structured, 8=conversation, 9=discovery, 10=audio, 11=parameters, 12=comparison, 13=dynamic, 14=errors)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"🚀 OpenAI Provider Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('OPENAI_API_KEY')[:20]}...")
    print("=" * 70)

    client = get_client("openai", model=args.model)

    # Run specific demo or all demos
    if args.demo:
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "openai", args.model)),
            2: ("Streaming", demo_streaming(client, "openai", args.model)),
            3: ("Function Calling", demo_function_calling(client, "openai", args.model)),
            4: ("Vision", demo_vision(client, "openai", args.model)),
            5: ("JSON Mode", demo_json_mode(client, "openai", args.model)),
            6: ("Reasoning", demo_reasoning(client, "openai", args.model)),
            7: ("Structured Outputs", demo_structured_outputs(client, "openai", args.model)),
            8: ("Conversation", demo_conversation(client, "openai", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "openai", args.model)),
            10: ("Audio Input", demo_audio_input(client, "openai", args.model)),
            11: ("Parameters", demo_parameters(client, "openai", args.model)),
            12: ("Model Comparison", demo_model_comparison("openai", ["gpt-4o-mini", "gpt-4o"])),
            13: ("Dynamic Model Call", demo_dynamic_model_call("openai")),
            14: ("Error Handling", demo_error_handling(client, "openai", args.model)),
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
            "openai",
            args.model,
            skip_tools=args.skip_tools,
            skip_vision=args.skip_vision,
            comparison_models=["gpt-4o-mini", "gpt-4o"]
        )

    print("\n" + "=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)
    print("\nℹ️  Tips:")
    print("  - Use --demo N to run specific demo")
    print("  - Use --skip-tools to skip function calling")
    print("  - Use --skip-vision to skip vision demo")
    print("  - Try reasoning models: gpt-5-mini, o1, o3")
    print("  - Try audio models: gpt-4o-audio-preview")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
