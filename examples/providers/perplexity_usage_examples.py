#!/usr/bin/env python3
"""
Perplexity Provider Example Usage Script
========================================

Demonstrates all features of the Perplexity provider using standardized demos.

Requirements:
- pip install chuk-llm
- Set PERPLEXITY_API_KEY environment variable

Usage:
    python perplexity_usage_examples.py
    python perplexity_usage_examples.py --model llama-3.1-sonar-large-128k-online
    python perplexity_usage_examples.py --skip-tools
    python perplexity_usage_examples.py --skip-vision
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("PERPLEXITY_API_KEY"):
    print("❌ Please set PERPLEXITY_API_KEY environment variable")
    print("   export PERPLEXITY_API_KEY='your_api_key_here'")
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
    """Run all Perplexity demos"""
    parser = argparse.ArgumentParser(description="Perplexity Provider Examples")
    parser.add_argument(
        "--model",
        default="sonar-pro",
        help="Model to use (default: sonar-pro)",
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
        choices=[1, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14],
        help="Run specific demo (1=basic, 2=streaming, 4=vision, 5=json, 6=reasoning, 7=structured, 8=conversation, 9=discovery, 11=parameters, 13=dynamic, 14=errors)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"🚀 Perplexity Provider Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('PERPLEXITY_API_KEY')[:20]}...")
    print("=" * 70)

    client = get_client("perplexity", model=args.model)

    # Run specific demo or all demos
    if args.demo:
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "perplexity", args.model)),
            2: ("Streaming", demo_streaming(client, "perplexity", args.model)),
            3: ("Function Calling", demo_function_calling(client, "perplexity", args.model)),
            4: ("Vision", demo_vision(client, "perplexity", args.model)),
            5: ("JSON Mode", demo_json_mode(client, "perplexity", args.model)),
            6: ("Reasoning", demo_reasoning(client, "perplexity", args.model)),
            7: ("Structured Outputs", demo_structured_outputs(client, "perplexity", args.model)),
            8: ("Conversation", demo_conversation(client, "perplexity", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "perplexity", args.model)),
            10: ("Audio Input", demo_audio_input(client, "perplexity", args.model)),
            11: ("Parameters", demo_parameters(client, "perplexity", args.model)),
            12: ("Model Comparison", demo_model_comparison("perplexity", ['llama-3.1-sonar-small-128k-online', 'llama-3.1-sonar-large-128k-online'])),
            13: ("Dynamic Model Call", demo_dynamic_model_call("perplexity")),
            14: ("Error Handling", demo_error_handling(client, "perplexity", args.model)),
        }

        name, demo_coro = demo_map[args.demo]
        try:
            await demo_coro
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run all demos - Perplexity handles requests gracefully even for unsupported features
        demos = [
            ("Basic Completion", demo_basic_completion(client, "perplexity", args.model)),
            ("Streaming", demo_streaming(client, "perplexity", args.model)),
            # Skip function calling - Perplexity doesn't support tools
        ]

        if not args.skip_vision:
            demos.append(("Vision", demo_vision(client, "perplexity", args.model)))

        demos.extend([
            ("JSON Mode", demo_json_mode(client, "perplexity", args.model)),
            ("Reasoning", demo_reasoning(client, "perplexity", args.model)),
            ("Structured Outputs", demo_structured_outputs(client, "perplexity", args.model)),
            ("Conversation", demo_conversation(client, "perplexity", args.model)),
            ("Model Discovery", demo_model_discovery(client, "perplexity", args.model)),
            # Skip audio - causes API errors
            ("Parameters", demo_parameters(client, "perplexity", args.model)),
            # Skip model comparison for now
            ("Dynamic Model Call", demo_dynamic_model_call("perplexity")),
            ("Error Handling", demo_error_handling(client, "perplexity", args.model)),
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
    print("  - Use --demo N to run specific demo")
    print("  - Perplexity specializes in search-grounded responses with citations")
    print("  - Demos 3 (tools), 10 (audio), 12 (comparison) skipped - not supported")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
