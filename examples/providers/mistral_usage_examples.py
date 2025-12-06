#!/usr/bin/env python3
"""
Mistral AI Provider Example Usage Script
========================================

Demonstrates all features of the Mistral AI provider using standardized demos.

Requirements:
- pip install chuk-llm
- Set MISTRAL_API_KEY environment variable

Usage:
    python mistral_usage_examples.py
    python mistral_usage_examples.py --model mistral-large-latest  # Mistral Large 3
    python mistral_usage_examples.py --model mistral-large-2512    # Mistral Large 3 (December 2025)
    python mistral_usage_examples.py --model ministral-3b-2512     # Ministral 3B
    python mistral_usage_examples.py --model ministral-8b-2512     # Ministral 8B
    python mistral_usage_examples.py --model ministral-14b-2512    # Ministral 14B
    python mistral_usage_examples.py --skip-tools
    python mistral_usage_examples.py --skip-vision
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("MISTRAL_API_KEY"):
    print("❌ Please set MISTRAL_API_KEY environment variable")
    print("   export MISTRAL_API_KEY='your_api_key_here'")
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
    """Run all Mistral AI demos"""
    parser = argparse.ArgumentParser(description="Mistral AI Provider Examples")
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="Model to use (default: mistral-large-latest)",
    )
    parser.add_argument(
        "--skip-tools", action="store_true", help="Skip function calling demo"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision demo"
    )
    parser.add_argument(
        "--skip-audio", action="store_true", default=True, help="Skip audio demo (default: True)"
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=json, 6=reasoning, 7=structured, 8=conversation, 9=discovery, 10=audio, 11=parameters, 12=comparison, 13=dynamic, 14=errors)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"🚀 Mistral AI Provider Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('MISTRAL_API_KEY')[:20]}...")
    print("=" * 70)

    client = get_client("mistral", model=args.model)

    # Audio model (skipped by default - voxtral requires special setup)
    audio_model = args.model

    if args.demo:
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "mistral", args.model)),
            2: ("Streaming", demo_streaming(client, "mistral", args.model)),
            3: ("Function Calling", demo_function_calling(client, "mistral", args.model)),
            4: ("Vision", demo_vision(client, "mistral", args.model)),
            5: ("JSON Mode", demo_json_mode(client, "mistral", args.model)),
            6: ("Reasoning", demo_reasoning(client, "mistral", args.model)),
            7: ("Structured Outputs", demo_structured_outputs(client, "mistral", args.model)),
            8: ("Conversation", demo_conversation(client, "mistral", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "mistral", args.model)),
            10: ("Audio Input", demo_audio_input(client, "mistral", audio_model)),
            11: ("Parameters", demo_parameters(client, "mistral", args.model)),
            12: ("Model Comparison", demo_model_comparison("mistral", ["mistral-large-2512", "ministral-8b-2512", "ministral-14b-2512"])),
            13: ("Dynamic Model Call", demo_dynamic_model_call("mistral")),
            14: ("Error Handling", demo_error_handling(client, "mistral", args.model)),
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
            "mistral",
            args.model,
            skip_tools=args.skip_tools,
            skip_vision=args.skip_vision,
            skip_audio=args.skip_audio,
            audio_model=audio_model
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
