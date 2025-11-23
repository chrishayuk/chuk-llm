#!/usr/bin/env python3
"""
Moonshot AI (Kimi) Provider Example Usage Script
================================================

Demonstrates all features of the Moonshot AI provider using standardized demos.

Requirements:
- pip install chuk-llm
- Set MOONSHOT_API_KEY environment variable

Usage:
    python moonshot_usage_examples.py
    python moonshot_usage_examples.py --model kimi-k2-turbo-preview
    python moonshot_usage_examples.py --skip-tools
    python moonshot_usage_examples.py --skip-vision
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Check environment
if not os.getenv("MOONSHOT_API_KEY"):
    print("❌ Please set MOONSHOT_API_KEY environment variable")
    print("   export MOONSHOT_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    from chuk_llm.configuration import get_config
    from chuk_llm.llm.client import get_client
    from chuk_llm.core.models import Message
    from chuk_llm.core.enums import MessageRole

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
        demo_model_comparison,
        demo_dynamic_model_call,
        demo_error_handling,
        run_all_demos,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


async def demo_parameters_moonshot(client, provider: str, model: str):
    """
    Temperature and sampling parameters demonstration for Moonshot.
    Moonshot has a max temperature of 1.0 (not 1.5 like OpenAI).
    """
    print(f"\n{'='*70}")
    print("Demo 11: Temperature and Sampling Parameters")
    print(f"{'='*70}")

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Prompt: Write the first sentence of a story about a robot.\n")

    # Moonshot max temperature is 1.0
    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        messages = [
            Message(
                role=MessageRole.USER,
                content="Write the first sentence of a story about a robot."
            ),
        ]

        try:
            response = await client.create_completion(
                messages,
                temperature=temp,
                max_tokens=100
            )

            print(f"Temperature {temp}:")
            print(f"  {response['response']}\n")

        except Exception as e:
            print(f"Temperature {temp}:")
            print(f"  Error: {e}\n")

    print("✅ Lower temperature = more deterministic")
    print("   Higher temperature = more creative/random")
    print("ℹ️  Note: Moonshot max temperature is 1.0 (not 1.5 like OpenAI)")

    return None


async def main():
    """Run all Moonshot AI demos"""
    parser = argparse.ArgumentParser(description="Moonshot AI (Kimi) Provider Examples")
    parser.add_argument(
        "--model",
        default="kimi-k2-turbo-preview",
        help="Model to use (default: kimi-k2-turbo-preview)",
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
    print(f"🚀 Moonshot AI (Kimi) Provider Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('MOONSHOT_API_KEY')[:20]}...")
    print("=" * 70)

    client = get_client("moonshot", model=args.model)

    # Use vision-capable model for vision demos
    vision_client = get_client("moonshot", model="kimi-latest")  # kimi-latest supports vision

    # Run specific demo or all demos
    if args.demo:
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "moonshot", args.model)),
            2: ("Streaming", demo_streaming(client, "moonshot", args.model)),
            3: ("Function Calling", demo_function_calling(client, "moonshot", args.model)),
            4: ("Vision", demo_vision(vision_client, "moonshot", "kimi-latest")),
            5: ("JSON Mode", demo_json_mode(client, "moonshot", args.model)),
            6: ("Reasoning", demo_reasoning(client, "moonshot", args.model)),
            7: ("Structured Outputs", demo_structured_outputs(client, "moonshot", args.model)),
            8: ("Conversation", demo_conversation(client, "moonshot", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "moonshot", args.model)),
            10: ("Audio Input", demo_audio_input(client, "moonshot", args.model)),
            11: ("Parameters", demo_parameters_moonshot(client, "moonshot", args.model)),
            12: ("Model Comparison", demo_model_comparison("moonshot", ['kimi-k2-turbo-preview', 'kimi-k2-thinking-turbo'])),
            13: ("Dynamic Model Call", demo_dynamic_model_call("moonshot")),
            14: ("Error Handling", demo_error_handling(client, "moonshot", args.model)),
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
        # Use kimi-latest for vision (K2 models don't support vision)
        demos = [
            ("Basic Completion", demo_basic_completion(client, "moonshot", args.model)),
            ("Streaming", demo_streaming(client, "moonshot", args.model)),
        ]

        if not args.skip_tools:
            demos.append(("Function Calling", demo_function_calling(client, "moonshot", args.model)))

        if not args.skip_vision:
            demos.append(("Vision", demo_vision(vision_client, "moonshot", "kimi-latest")))

        demos.extend([
            ("JSON Mode", demo_json_mode(client, "moonshot", args.model)),
            ("Reasoning", demo_reasoning(client, "moonshot", args.model)),
            ("Structured Outputs", demo_structured_outputs(client, "moonshot", args.model)),
            ("Conversation", demo_conversation(client, "moonshot", args.model)),
            ("Model Discovery", demo_model_discovery(client, "moonshot", args.model)),
            # Audio not supported on Moonshot
            ("Parameters", demo_parameters_moonshot(client, "moonshot", args.model)),
            ("Model Comparison", demo_model_comparison("moonshot", ['kimi-k2-turbo-preview', 'kimi-k2-thinking-turbo'])),
            ("Dynamic Model Call", demo_dynamic_model_call("moonshot")),
            ("Error Handling", demo_error_handling(client, "moonshot", args.model)),
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
    print("  - Use --skip-tools to skip function calling")
    print("  - Use --skip-vision to skip vision demo")
    print("  - Vision demos use 'kimi-latest' model (supports vision)")
    print("  - Kimi K2 models feature:")
    print("    • Industry-leading coding abilities")
    print("    • Built-in tools like web search")
    print("    • 256K context window (k2-0905-preview, k2-turbo-preview)")
    print("    • Long-term thinking capabilities (k2-thinking models)")
    print("    • Max temperature: 1.0 (not 1.5 like OpenAI)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
