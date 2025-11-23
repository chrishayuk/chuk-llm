#!/usr/bin/env python3
"""
Ollama Provider Example Usage Script
====================================

Demonstrates all features of the Ollama provider using standardized demos.

Features Demonstrated:
- ✅ Basic completion
- ✅ Streaming
- ✅ Function calling
- ✅ Vision (multimodal)
- ✅ JSON mode
- ✅ Conversation handling
- ✅ Model discovery
- ✅ Parameters (temperature, etc.)
- ✅ Model comparison
- ✅ Error handling
- ✅ Reasoning models (GPT-OSS, DeepSeek-R1)
- ✅ Type-safe Pydantic models (via common_demos)
- ✅ Zero magic strings (all enums)

Requirements:
- pip install chuk-llm
- Ollama running locally (ollama serve)
- At least one model pulled (e.g., ollama pull llama3.2)

Usage:
    python ollama_usage_examples.py
    python ollama_usage_examples.py --model llama3.2
    python ollama_usage_examples.py --model gpt-oss:20b  # Reasoning
    python ollama_usage_examples.py --skip-tools
    python ollama_usage_examples.py --demo 1  # Run specific demo
"""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    from chuk_llm.configuration import get_config
    from chuk_llm.llm.client import get_client

    # Import common demos
    examples_dir = Path(__file__).parent.parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from common_demos import (
        demo_basic_completion,
        demo_conversation,
        demo_dynamic_model_call,
        demo_error_handling,
        demo_function_calling,
        demo_json_mode,
        demo_model_comparison,
        demo_model_discovery,
        demo_parameters,
        demo_reasoning,
        demo_streaming,
        demo_structured_outputs,
        demo_vision,
        demo_vision_url,
        run_all_demos,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


async def check_ollama_available():
    """Check if Ollama is running and has models."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])

            if not models:
                print("❌ No Ollama models found!")
                print("   Pull a model with: ollama pull llama3.2")
                return False

            print(f"✓ Found {len(models)} Ollama model(s):")
            for model in models[:5]:
                print(f"  - {model.get('name', 'unknown')}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
            return True

    except (httpx.ConnectError, httpx.HTTPError):
        print("❌ Ollama is not running!")
        print("   Start with: ollama serve")
        print("   Then pull a model: ollama pull llama3.2")
        return False


async def main():
    """Run all Ollama demos"""
    parser = argparse.ArgumentParser(description="Ollama Provider Examples")
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Model to use for main demos (default: gpt-oss:20b)",
    )
    parser.add_argument(
        "--vision-model",
        default="llama3.2-vision",
        help="Model to use for vision demos (default: llama3.2-vision)",
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
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=vision-url, 6=json, 7=conversation, 8=discovery, 9=parameters, 10=comparison, 11=dynamic, 12=errors)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🦙 Ollama Provider Examples")
    print("=" * 70)

    # Check if Ollama is available
    if not await check_ollama_available():
        sys.exit(1)

    print(f"Main Model: {args.model}")
    print(f"Vision Model: {args.vision_model}")

    # Auto-detect capabilities from model name
    model_lower = args.model.lower()
    has_reasoning = "gpt-oss" in model_lower or "deepseek-r1" in model_lower or "qwq" in model_lower

    if has_reasoning:
        print("✓ Reasoning model detected")
    print("=" * 70)

    # Create main client
    client = get_client("ollama", model=args.model)

    # Create vision client if vision model is different
    if args.vision_model != args.model:
        vision_client = get_client("ollama", model=args.vision_model)
    else:
        vision_client = client

    if args.demo:
        # Run specific demo
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "ollama", args.model)),
            2: ("Streaming", demo_streaming(client, "ollama", args.model)),
            3: ("Function Calling", demo_function_calling(client, "ollama", args.model)),
            4: ("Vision", demo_vision(vision_client, "ollama", args.vision_model)),
            5: ("Vision with URL", demo_vision_url(vision_client, "ollama", args.vision_model)),
            6: ("JSON Mode", demo_json_mode(client, "ollama", args.model)),
            7: ("Conversation", demo_conversation(client, "ollama", args.model)),
            8: ("Model Discovery", demo_model_discovery(client, "ollama", args.model)),
            9: ("Parameters", demo_parameters(client, "ollama", args.model)),
            10: ("Model Comparison", demo_model_comparison("ollama", [args.model, args.vision_model])),
            11: ("Dynamic Model Call", demo_dynamic_model_call("ollama")),
            12: ("Error Handling", demo_error_handling(client, "ollama", args.model)),
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
            provider="ollama",
            model=args.model,
            skip_vision=args.skip_vision,  # Use separate vision model
            skip_audio=True,  # Ollama doesn't support audio input
            skip_tools=args.skip_tools,
            vision_client=vision_client,
            vision_model=args.vision_model,
        )

    print("\n" + "=" * 70)
    print("✓ All demos completed!")
    print("=" * 70)
    print("\n💡 Tips:")
    print("  - Ollama runs locally - no API key needed!")
    print("  - Vision models: ollama pull llama3.2-vision")
    print("  - Reasoning models: ollama pull gpt-oss:20b")
    print("  - List models: ollama list")
    print("  - Auto-discovery: chuk-llm models ollama")
    print("\n💡 Try different models:")
    print(f"  python {__file__} --model gpt-oss:20b  # Reasoning")
    print(f"  python {__file__} --model llama3.2-vision  # Vision")


if __name__ == "__main__":
    asyncio.run(main())
