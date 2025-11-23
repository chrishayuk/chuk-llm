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
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=json, 6=reasoning, 7=structured, 8=conversation, 9=discovery, 11=parameters, 12=comparison, 13=dynamic, 14=errors) - Note: demo 10 skipped",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"🚀 Groq Provider Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('GROQ_API_KEY')[:20]}...")
    print("=" * 70)

    client = get_client("groq", model=args.model)

    # Use specialized models for specific capabilities
    gpt_oss_client = get_client("groq", model="openai/gpt-oss-120b")  # Tools, JSON
    vision_client = get_client("groq", model="meta-llama/llama-4-scout-17b-16e-instruct")  # Vision

    # Run specific demo or all demos
    if args.demo:
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "groq", args.model)),
            2: ("Streaming", demo_streaming(client, "groq", args.model)),
            3: ("Function Calling", demo_function_calling(gpt_oss_client, "groq", "openai/gpt-oss-120b")),
            4: ("Vision", demo_vision(vision_client, "groq", "meta-llama/llama-4-scout-17b-16e-instruct")),
            5: ("JSON Mode", demo_json_mode(gpt_oss_client, "groq", "openai/gpt-oss-120b")),
            6: ("Reasoning", demo_reasoning(client, "groq", args.model)),
            7: ("Structured Outputs", demo_structured_outputs(gpt_oss_client, "groq", "openai/gpt-oss-120b")),
            8: ("Conversation", demo_conversation(client, "groq", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "groq", args.model)),
            11: ("Parameters", demo_parameters(client, "groq", args.model)),
            12: ("Model Comparison", demo_model_comparison("groq", ["llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct"])),
            13: ("Dynamic Model Call", demo_dynamic_model_call("groq")),
            14: ("Error Handling", demo_error_handling(client, "groq", args.model)),
        }

        name, demo_coro = demo_map[args.demo]
        try:
            await demo_coro
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run all demos with appropriate models
        demos = [
            ("Basic Completion", demo_basic_completion(client, "groq", args.model)),
            ("Streaming", demo_streaming(client, "groq", args.model)),
        ]

        if not args.skip_tools:
            demos.append(("Function Calling", demo_function_calling(gpt_oss_client, "groq", "openai/gpt-oss-120b")))

        if not args.skip_vision:
            demos.append(("Vision", demo_vision(vision_client, "groq", "meta-llama/llama-4-scout-17b-16e-instruct")))

        demos.extend([
            ("JSON Mode", demo_json_mode(gpt_oss_client, "groq", "openai/gpt-oss-120b")),
            ("Reasoning", demo_reasoning(client, "groq", args.model)),
            ("Structured Outputs", demo_structured_outputs(gpt_oss_client, "groq", "openai/gpt-oss-120b")),
            ("Conversation", demo_conversation(client, "groq", args.model)),
            ("Model Discovery", demo_model_discovery(client, "groq", args.model)),
            # Note: Whisper uses /audio/transcriptions endpoint, not chat completions
            # Skipping audio demo as it requires different API
            ("Parameters", demo_parameters(client, "groq", args.model)),
            ("Model Comparison", demo_model_comparison("groq", ["llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct"])),
            ("Dynamic Model Call", demo_dynamic_model_call("groq")),
            ("Error Handling", demo_error_handling(client, "groq", args.model)),
        ])

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
    print("  - Use --demo N to run specific demo (demo 10 skipped - see note below)")
    print("  - Use --skip-tools to skip function calling")
    print("  - Use --skip-vision to skip vision demo")
    print("  - Demo 3, 5, 7 use openai/gpt-oss-120b (tools/JSON support)")
    print("  - Demo 4 uses meta-llama/llama-4-scout-17b-16e-instruct (vision)")
    print("\nℹ️  Note: Demo 10 (Audio) skipped - Whisper uses /audio/transcriptions API")
    print("   which requires a different endpoint than chat completions")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
