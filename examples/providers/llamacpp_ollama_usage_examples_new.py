#!/usr/bin/env python3
"""
llama.cpp with Ollama Bridge Example (Simplified)
=================================================

Demonstrates using llama.cpp with Ollama's downloaded models.
Automatically discovers and reuses Ollama's GGUF files - no re-download!
Servers start and stop automatically!

Requirements:
- pip install chuk-llm
- llama-server binary in PATH (brew install llama.cpp)
- Ollama with models downloaded (e.g., ollama pull gpt-oss:20b)

Usage:
    python llamacpp_ollama_usage_examples.py
    python llamacpp_ollama_usage_examples.py --model gpt-oss:20b
    python llamacpp_ollama_usage_examples.py --skip-tools
    python llamacpp_ollama_usage_examples.py --demo 2  # Run only streaming demo
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


def display_available_models(models):
    """Display discovered Ollama models."""
    print(f"\n✓ Found {len(models)} Ollama model(s) (total: {sum(m.size_bytes for m in models) / 1e9:.1f} GB):")
    for i, model in enumerate(models[:10], 1):
        size_gb = model.size_bytes / (1024**3)
        print(f"  {i}. {model.name}")
        print(f"     Size: {size_gb:.2f} GB")
        print(f"     Path: {model.gguf_path}")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more")


def find_model_by_pattern(models, patterns: list[str]):
    """Find first model matching any of the patterns."""
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for model in models:
            model_name_lower = model.name.lower()
            if pattern_lower in model_name_lower:
                return model
    return None


async def main():
    """Run all llama.cpp + Ollama bridge demos"""
    parser = argparse.ArgumentParser(
        description="llama.cpp with Ollama Bridge Examples"
    )
    parser.add_argument(
        "--model",
        default="gpt-oss",
        help="Main model pattern to search for (default: gpt-oss)",
    )
    parser.add_argument(
        "--vision-model",
        default="llama3.2-vision",
        help="Vision model pattern to search for (default: llama3.2-vision)",
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
        "--skip-vision", action="store_true", help="Skip vision demo"
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="Run specific demo (1=basic, 2=streaming, 3=tools, 4=vision, 5=vision-url, 6=json, 7=conversation, 8=discovery, 9=parameters, 10=comparison, 11=dynamic, 12=errors)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🦙 llama.cpp + Ollama Bridge Examples")
    print("=" * 70)
    print("Reusing Ollama's downloaded models with llama.cpp server")
    print("=" * 70)

    # Discover Ollama models
    print("\n🔍 Discovering Ollama models...")
    models = discover_ollama_models()

    if not models:
        print("\n❌ No Ollama models found!")
        print("   Pull models with:")
        print("     ollama pull gpt-oss:20b")
        print("     ollama pull llama3.2-vision")
        sys.exit(1)

    display_available_models(models)

    # Find main model (prefer gpt-oss for reasoning)
    main_model = find_model_by_pattern(models, [args.model, "gpt-oss"])
    if not main_model:
        print(f"\n❌ No model found matching '{args.model}' or 'gpt-oss'")
        print("   Pull with: ollama pull gpt-oss:20b")
        sys.exit(1)

    print(f"\n📦 Main Model: {main_model.name}")
    print(f"   Path: {main_model.gguf_path}")
    print(f"   Size: {main_model.size_bytes / (1024**3):.2f} GB")

    # Find vision model if needed
    vision_model = None
    if not args.skip_vision:
        vision_model = find_model_by_pattern(models, [args.vision_model, "llama3.2-vision", "vision"])
        if vision_model:
            print(f"\n📦 Vision Model: {vision_model.name}")
            print(f"   Path: {vision_model.gguf_path}")
            print(f"   Size: {vision_model.size_bytes / (1024**3):.2f} GB")
        else:
            print(f"\n⚠️  No vision model found, skipping vision demos")
            print("   Pull with: ollama pull llama3.2-vision")
            args.skip_vision = True

    # Auto-detect capabilities
    model_lower = main_model.name.lower()
    has_reasoning = "gpt-oss" in model_lower or "deepseek-r1" in model_lower or "qwq" in model_lower
    if has_reasoning:
        print("\n✓ Reasoning model detected")

    print("=" * 70)

    try:
        # Create main client (server starts automatically on first request)
        print(f"\n📦 Creating llama.cpp clients (servers start automatically)...")

        async with LlamaCppLLMClient(
            model=main_model.gguf_path,
            ctx_size=args.ctx_size,
            n_gpu_layers=-1,  # Use all GPU layers
        ) as main_client:
            print(f"✓ Main client ready at {main_client._server_manager.base_url}")

            # Create vision client if needed
            vision_client = main_client
            if not args.skip_vision and vision_model:
                vision_client = LlamaCppLLMClient(
                    model=vision_model.gguf_path,
                    ctx_size=4096,  # Smaller context for vision
                    n_gpu_layers=-1,
                )
                print(f"✓ Vision client ready at {vision_client._server_manager.base_url}")

            print("=" * 70)

            # Run demos
            try:
                await run_all_demos(
                    main_client,
                    provider="llamacpp",
                    model=main_model.name,
                    skip_vision=args.skip_vision,
                    skip_audio=True,  # llama.cpp doesn't support audio input
                    skip_tools=args.skip_tools,
                    skip_discovery=True,  # llama.cpp models are local, discovery not applicable
                    vision_client=vision_client if vision_client != main_client else None,
                    vision_model=vision_model.name if vision_model else None,
                    specific_demo=args.demo,
                )
            finally:
                # Clean up vision client if separate
                if vision_client != main_client:
                    await vision_client.close()

        print("\n🛑 Servers stopped")

        print("\n" + "=" * 70)
        print("✓ All demos completed!")
        print("=" * 70)
        print("\n💡 Benefits of llama.cpp + Ollama:")
        print(f"  - Reused Ollama's downloads ({main_model.size_bytes / 1e9:.1f} GB)")
        print("  - Auto-managed servers (no manual startup)")
        print("  - Advanced llama.cpp features (grammars, custom sampling)")
        print("  - OpenAI-compatible API (works with existing code)")
        print("  - No API keys needed (100% local)")
        print(f"\n💡 Total storage saved: {sum(m.size_bytes for m in models) / 1e9:.1f} GB")
        print("   (no duplicate downloads!)")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n   Install llama-server:")
        print("     macOS: brew install llama.cpp")
        print("     Linux: Build from https://github.com/ggerganov/llama.cpp")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
