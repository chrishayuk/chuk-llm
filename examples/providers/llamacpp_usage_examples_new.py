#!/usr/bin/env python3
"""
llama.cpp Provider Example Usage Script (Simplified)
====================================================

Demonstrates all features of the llama.cpp provider with auto-managed servers.
The LlamaCppLLMClient handles server startup and shutdown automatically!

Requirements:
- pip install chuk-llm
- llama-server in PATH or standard location
- Models downloaded (using ollama or manual download)

Usage:
    # Auto-start servers with models from Ollama cache:
    python llamacpp_usage_examples.py

    # Use custom model paths:
    python llamacpp_usage_examples.py --model /path/to/model.gguf

    # Skip specific demos:
    python llamacpp_usage_examples.py --skip-tools --skip-vision

    # Run specific demo:
    python llamacpp_usage_examples.py --demo 2  # Streaming demo
"""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    from chuk_llm.llm.providers.llamacpp_client import LlamaCppLLMClient
    from chuk_llm.registry.resolvers.llamacpp_ollama import discover_ollama_models

    # Import common demos
    examples_dir = Path(__file__).parent.parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from common_demos import run_all_demos
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


def find_model_by_pattern(models, patterns: list[str]):
    """Find first model matching any of the patterns."""
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for model in models:
            model_name_lower = model.name.lower()
            if pattern_lower in model_name_lower:
                return model
    return None


def find_model_path(model_spec: str) -> Path:
    """
    Find model path from various sources.

    Args:
        model_spec: Model name or path. Can be:
            - Full path to .gguf file
            - Model name to search in Ollama cache

    Returns:
        Path to model file

    Raises:
        FileNotFoundError: If model not found
    """
    # Check if it's already a valid path
    model_path = Path(model_spec)
    if model_path.exists() and model_path.suffix == ".gguf":
        return model_path

    # Try to find in Ollama cache
    print(f"🔍 Searching for '{model_spec}' in Ollama cache...")
    ollama_models = discover_ollama_models()
    if ollama_models:
        found_model = find_model_by_pattern(ollama_models, [model_spec])
        if found_model:
            print(f"✓ Found: {found_model.name}")
            return found_model.gguf_path

    # Not found
    raise FileNotFoundError(
        f"Model not found: {model_spec}\n"
        f"  - Provide full path to .gguf file\n"
        f"  - Or pull via Ollama: ollama pull {model_spec}"
    )


async def main():
    """Run all llama.cpp demos with auto-managed servers"""
    parser = argparse.ArgumentParser(
        description="llama.cpp Examples (Auto-managed Servers)"
    )
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Model for main demos (name or path, default: gpt-oss:20b)",
    )
    parser.add_argument(
        "--vision-model",
        default="llama3.2-vision",
        help="Model for vision demos (name or path, default: llama3.2-vision)",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=8192,
        help="Context size (default: 8192)",
    )
    parser.add_argument(
        "--skip-tools", action="store_true", help="Skip function calling demo"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision demos"
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=vision-url, 6=json, 7=conversation, 8=parameters, 9=comparison, 10=dynamic, 11=errors, 12=reasoning)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🦙 llama.cpp Examples (Auto-managed Servers)")
    print("=" * 70)
    print("Servers start automatically - no manual setup needed!")
    print("=" * 70)

    try:
        # Find models
        print(f"\n🔍 Finding models...")
        main_model_path = find_model_path(args.model)
        print(f"✓ Main model: {main_model_path.name}")
        print(f"  Size: {main_model_path.stat().st_size / (1024**3):.2f} GB")

        vision_model_path = None
        if not args.skip_vision:
            try:
                vision_model_path = find_model_path(args.vision_model)
                print(f"✓ Vision model: {vision_model_path.name}")
                print(f"  Size: {vision_model_path.stat().st_size / (1024**3):.2f} GB")
            except FileNotFoundError as e:
                print(f"⚠️  Vision model not found, skipping vision demos")
                args.skip_vision = True

        print("=" * 70)

        # Create clients with auto-managed servers
        # The servers will start automatically on first request!
        print(f"\n📦 Creating llama.cpp clients...")
        print("   (Servers will start automatically when needed)")

        async with LlamaCppLLMClient(
            model=main_model_path,
            ctx_size=args.ctx_size,
            n_gpu_layers=-1,  # Use all GPU layers
        ) as main_client:
            print(f"✓ Main client ready (server will start on first request)")

            # Prepare vision client if needed
            vision_client = main_client
            if not args.skip_vision and vision_model_path:
                vision_client = LlamaCppLLMClient(
                    model=vision_model_path,
                    ctx_size=4096,  # Smaller context for vision
                    n_gpu_layers=-1,
                )
                print(f"✓ Vision client ready (server will start on first request)")

            print("=" * 70)

            # Run demos
            try:
                await run_all_demos(
                    main_client,
                    provider="llamacpp",
                    model=str(main_model_path.name),
                    skip_vision=args.skip_vision,
                    skip_audio=True,  # llama.cpp doesn't support audio
                    skip_tools=args.skip_tools,
                    skip_discovery=True,  # Local models don't need discovery
                    vision_client=vision_client if vision_client != main_client else None,
                    vision_model=str(vision_model_path.name) if vision_model_path else None,
                    specific_demo=args.demo,
                )
            finally:
                # Clean up vision client if separate
                if vision_client != main_client:
                    await vision_client.close()

        print("\n" + "=" * 70)
        print("✓ All demos completed!")
        print("=" * 70)
        print("\n💡 Benefits of auto-managed servers:")
        print("  - No manual server startup needed")
        print("  - Automatic cleanup on exit")
        print("  - Multiple models running simultaneously")
        print("  - OpenAI-compatible API")
        print(f"  - 100% local (no API keys needed)")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
