#!/usr/bin/env python3
"""
Azure OpenAI Provider Example Usage Script
==========================================

Demonstrates all features of the Azure OpenAI provider using standardized demos.

Requirements:
- pip install chuk-llm
- Set AZURE_OPENAI_API_KEY environment variable

Usage:
    python azure_usage_examples.py
    python azure_usage_examples.py --model gpt-4o
    python azure_usage_examples.py --skip-tools
    python azure_usage_examples.py --skip-vision
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this script)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

# Check environment
if not os.getenv("AZURE_OPENAI_API_KEY"):
    print("❌ Please set AZURE_OPENAI_API_KEY environment variable")
    print("   export AZURE_OPENAI_API_KEY='your_api_key_here'")
    sys.exit(1)

if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    print("❌ Please set AZURE_OPENAI_ENDPOINT environment variable")
    print("   export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'")
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
    """Run all Azure OpenAI demos"""
    parser = argparse.ArgumentParser(description="Azure OpenAI Provider Examples")
    parser.add_argument(
        "--model",
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        help=f"Model/deployment to use (default: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')})",
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
    print(f"🚀 Azure OpenAI Provider Examples")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API Key: {os.getenv('AZURE_OPENAI_API_KEY')[:20]}...")
    print("=" * 70)

    client = get_client("azure_openai", model=args.model)

    # Run specific demo or all demos
    if args.demo:
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "azure_openai", args.model)),
            2: ("Streaming", demo_streaming(client, "azure_openai", args.model)),
            3: ("Function Calling", demo_function_calling(client, "azure_openai", args.model)),
            4: ("Vision", demo_vision(client, "azure_openai", args.model)),
            5: ("JSON Mode", demo_json_mode(client, "azure_openai", args.model)),
            6: ("Reasoning", demo_reasoning(client, "azure_openai", args.model)),
            7: ("Structured Outputs", demo_structured_outputs(client, "azure_openai", args.model)),
            8: ("Conversation", demo_conversation(client, "azure_openai", args.model)),
            9: ("Model Discovery", demo_model_discovery(client, "azure_openai", args.model)),
            10: ("Error Handling", demo_error_handling(client, "azure_openai", args.model)),
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
            "azure_openai",
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
