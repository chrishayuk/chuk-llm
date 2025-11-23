#!/usr/bin/env python3
"""
llama.cpp with Ollama Bridge Example Usage Script
=================================================

Demonstrates using llama.cpp server with Ollama's downloaded models.
Automatically discovers and reuses Ollama's GGUF files - no re-download!

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
import socket
import sys
from pathlib import Path

try:
    from chuk_llm.registry.resolvers.llamacpp_ollama import discover_ollama_models
    from chuk_llm.llm.providers.llamacpp_server import (
        LlamaCppServerConfig,
        LlamaCppServerManager,
    )
    from chuk_llm.llm.providers.openai_client import OpenAILLMClient

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
        demo_tokens_per_second,
        demo_vision,
        demo_vision_url,
        run_all_demos,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install with: pip install chuk-llm")
    sys.exit(1)


def find_available_port(start_port: int = 8082, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts}"
    )


def display_available_models(models):
    """Display discovered Ollama models."""
    print(
        f"\n✓ Found {len(models)} Ollama model(s) (total: {sum(m.size_bytes for m in models) / 1e9:.1f} GB):"
    )
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
        default="qwen3",
        help="Main model pattern to search for (default: qwen3 - use smallest model)",
    )
    parser.add_argument(
        "--vision-model",
        default="granite3.2-vision",
        help="Vision model pattern to search for (default: granite3.2-vision - smaller and faster)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8082,
        help="Port for main model server (default: 8082)",
    )
    parser.add_argument(
        "--vision-port",
        type=int,
        default=8083,
        help="Port for vision model server (default: 8083)",
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
        "--skip-vision",
        action="store_true",
        default=True,
        help="Skip vision demo (default: True - llama.cpp requires mmproj files)",
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable vision demo (requires mmproj file - advanced usage)",
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help="Run specific demo (1=basic, 2=streaming, 3=tokens/sec, 4=tools, 5=vision, 6=vision-url, 7=json, 8=conversation, 9=discovery, 10=parameters, 11=comparison, 12=dynamic, 13=errors)",
    )

    args = parser.parse_args()

    # Handle --enable-vision flag
    if args.enable_vision:
        args.skip_vision = False

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

    # Find main model (prefer qwen3 for speed, fallback to first available)
    main_model = find_model_by_pattern(models, [args.model, "qwen3", "llama3.2"])
    if not main_model:
        # Use first (smallest) model as fallback
        main_model = models[0]
        print(
            f"\n⚠️  No model found matching '{args.model}', using smallest: {main_model.name}"
        )

    print(f"\n📦 Main Model: {main_model.name}")
    print(f"   Path: {main_model.gguf_path}")
    print(f"   Size: {main_model.size_bytes / (1024**3):.2f} GB")

    # Find vision model if needed
    vision_model = None
    if not args.skip_vision:
        # Try to find vision models, preferring smaller ones
        vision_candidates = []
        for pattern in [
            args.vision_model,
            "granite3.2-vision",
            "llama3.2-vision",
            "vision",
        ]:
            model = find_model_by_pattern(models, [pattern])
            if model:
                vision_candidates.append(model)

        # Remove duplicates and sort by size (prefer smaller models)
        seen_paths = set()
        unique_candidates = []
        for model in vision_candidates:
            if model.gguf_path not in seen_paths:
                seen_paths.add(model.gguf_path)
                unique_candidates.append(model)
        unique_candidates.sort(key=lambda m: m.size_bytes)

        # Find first model that's reasonably sized (< 5 GB)
        for candidate in unique_candidates:
            size_gb = candidate.size_bytes / (1024**3)
            if size_gb < 5.0:
                vision_model = candidate
                break

        # If no small model found, use the smallest available but warn
        if not vision_model and unique_candidates:
            vision_model = unique_candidates[0]
            size_gb = vision_model.size_bytes / (1024**3)
            if size_gb >= 5.0:
                print(
                    f"\n⚠️  Vision model '{vision_model.name}' is large ({size_gb:.2f} GB)"
                )
                print("   This may cause llama-server to crash. Skipping vision demos.")
                print("   Try: ollama pull granite3.2-vision  # Smaller vision model")
                args.skip_vision = True
                vision_model = None

        if vision_model:
            print(f"\n📦 Vision Model: {vision_model.name}")
            print(f"   Path: {vision_model.gguf_path}")
            print(f"   Size: {vision_model.size_bytes / (1024**3):.2f} GB")
        elif not args.skip_vision:
            print(f"\n⚠️  No vision model found, skipping vision demos")
            print("   Pull with: ollama pull granite3.2-vision  # Small vision model")
            print(
                "            ollama pull llama3.2-vision       # Larger but more capable"
            )
            args.skip_vision = True

    # Auto-detect capabilities
    model_lower = main_model.name.lower()
    has_reasoning = (
        "gpt-oss" in model_lower or "deepseek-r1" in model_lower or "qwq" in model_lower
    )
    if has_reasoning:
        print("\n✓ Reasoning model detected")

    # Find available ports
    main_port = find_available_port(args.port)
    if main_port != args.port:
        print(f"\n⚠️  Port {args.port} in use, using {main_port} instead")

    vision_port = None
    if not args.skip_vision and vision_model:
        vision_port = find_available_port(args.vision_port)
        if vision_port == main_port:
            vision_port = find_available_port(main_port + 1)
        if vision_port != args.vision_port:
            print(f"⚠️  Port {args.vision_port} in use, using {vision_port} instead")

    print("=" * 70)

    try:
        # Configure main server
        main_config = LlamaCppServerConfig(
            model_path=main_model.gguf_path,
            port=main_port,
            ctx_size=args.ctx_size,
            n_gpu_layers=-1,  # Use all GPU layers
        )

        print(f"\n🚀 Starting main server on port {main_port}...")
        print(f"   Context size: {args.ctx_size}")
        print(f"   GPU layers: all (-1)")

        async with LlamaCppServerManager(main_config) as main_server:
            print(f"✓ Main server ready at {main_server.base_url}")

            # Get actual model name from server
            import httpx

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(f"{main_server.base_url}/v1/models")
                models_data = response.json()
                actual_model_name = (
                    models_data["data"][0]["id"]
                    if models_data.get("data")
                    else main_model.name
                )

            # Create main client
            client = OpenAILLMClient(
                model=actual_model_name,
                api_base=main_server.base_url,
            )

            # Start vision server if needed
            vision_client = client  # Default to main client
            actual_vision_model_name = actual_model_name

            if not args.skip_vision and vision_model:
                vision_config = LlamaCppServerConfig(
                    model_path=vision_model.gguf_path,
                    port=vision_port,
                    ctx_size=4096,  # Smaller context for vision
                    n_gpu_layers=-1,
                )

                print(f"🚀 Starting vision server on port {vision_port}...")

                async with LlamaCppServerManager(vision_config) as vision_server:
                    print(f"✓ Vision server ready at {vision_server.base_url}")

                    # Get actual vision model name
                    async with httpx.AsyncClient() as http_client:
                        response = await http_client.get(
                            f"{vision_server.base_url}/v1/models"
                        )
                        models_data = response.json()
                        actual_vision_model_name = (
                            models_data["data"][0]["id"]
                            if models_data.get("data")
                            else vision_model.name
                        )

                    vision_client = OpenAILLMClient(
                        model=actual_vision_model_name,
                        api_base=vision_server.base_url,
                    )

                    print("=" * 70)

                    # Run demos with both servers
                    await run_demos(
                        args,
                        client,
                        vision_client,
                        actual_model_name,
                        actual_vision_model_name,
                        models,
                    )
            else:
                # Run demos without vision
                print("=" * 70)
                await run_demos(
                    args, client, client, actual_model_name, actual_model_name, models
                )

        print("\n🛑 Servers stopped")

        print("\n" + "=" * 70)
        print("✓ All demos completed!")
        print("=" * 70)
        print("\n💡 Benefits of llama.cpp + Ollama:")
        print(f"  - Reused Ollama's downloads ({main_model.size_bytes / 1e9:.1f} GB)")
        print("  - Advanced llama.cpp features (grammars, custom sampling)")
        print("  - OpenAI-compatible API (works with existing code)")
        print("  - No API keys needed (100% local)")
        print(
            f"\n💡 Total storage saved: {sum(m.size_bytes for m in models) / 1e9:.1f} GB"
        )
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


async def run_demos(args, client, vision_client, main_model, vision_model, all_models):
    """Run the demos with the given clients."""
    if args.demo:
        # Run specific demo
        demo_map = {
            1: (
                "Basic Completion",
                demo_basic_completion(client, "llamacpp", main_model),
            ),
            2: ("Streaming", demo_streaming(client, "llamacpp", main_model)),
            3: (
                "Tokens Per Second",
                demo_tokens_per_second(client, "llamacpp", main_model),
            ),
            4: (
                "Function Calling",
                demo_function_calling(client, "llamacpp", main_model),
            ),
            5: ("Vision", demo_vision(vision_client, "llamacpp", vision_model)),
            6: (
                "Vision with URL",
                demo_vision_url(vision_client, "llamacpp", vision_model),
            ),
            7: ("JSON Mode", demo_json_mode(client, "llamacpp", main_model)),
            8: ("Conversation", demo_conversation(client, "llamacpp", main_model)),
            9: (
                "Model Discovery",
                demo_model_discovery(client, "llamacpp", main_model),
            ),
            10: ("Parameters", demo_parameters(client, "llamacpp", main_model)),
            11: (
                "Model Comparison",
                demo_model_comparison("llamacpp", [main_model, vision_model]),
            ),
            12: ("Dynamic Model Call", demo_dynamic_model_call("llamacpp")),
            13: ("Error Handling", demo_error_handling(client, "llamacpp", main_model)),
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
            provider="llamacpp",
            model=main_model,
            skip_vision=args.skip_vision,
            skip_audio=True,  # llama.cpp doesn't support audio input
            skip_tools=args.skip_tools,
            skip_discovery=True,  # llama.cpp models are local, discovery not applicable
            vision_client=vision_client if vision_client != client else None,
            vision_model=vision_model if vision_client != client else None,
        )


if __name__ == "__main__":
    asyncio.run(main())
