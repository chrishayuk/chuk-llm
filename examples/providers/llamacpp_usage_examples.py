#!/usr/bin/env python3
"""
llama.cpp Provider Example Usage Script
=======================================

Demonstrates all features of the llama.cpp provider using standardized demos.
Automatically starts and stops llama-server instances with different models.

Requirements:
- pip install chuk-llm
- llama-server in PATH or standard location
- Models downloaded (using ollama or manual download)

Usage:
    # Auto-start servers with default models:
    python llamacpp_usage_examples.py

    # Use custom models:
    python llamacpp_usage_examples.py --model gpt-oss-20b-mxfp4.gguf
    python llamacpp_usage_examples.py --vision-model qwen3-vl-30b.gguf

    # Skip auto-start and use existing server:
    python llamacpp_usage_examples.py --api-base http://localhost:8033 --no-auto-start

    # Skip specific demos:
    python llamacpp_usage_examples.py --skip-tools --skip-vision
"""

import argparse
import asyncio
import socket
import sys
from pathlib import Path

try:
    from chuk_llm.configuration import get_config
    from chuk_llm.llm.client import get_client
    from chuk_llm.llm.providers.llamacpp_server import LlamaCppServerConfig, LlamaCppServerManager

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


def find_available_port(start_port: int = 8033, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")


def find_model_in_ollama_cache(model_name: str) -> Path | None:
    """Find a model in Ollama's cache directory."""
    ollama_cache = Path.home() / "Library" / "Caches" / "llama.cpp"
    if not ollama_cache.exists():
        return None

    # Normalize model name for matching (remove : and convert to -)
    search_name = model_name.lower().replace(":", "-")

    # Try to find matching model
    for model_file in ollama_cache.glob("*.gguf"):
        # Skip mmproj files
        if "mmproj" in model_file.name.lower():
            continue

        file_name_lower = model_file.name.lower()

        # Check if search terms appear in filename
        if search_name in file_name_lower:
            return model_file

        # Also try matching individual parts (e.g., "gpt-oss" and "20b")
        search_parts = search_name.replace("-", " ").split()
        if len(search_parts) >= 2 and all(part in file_name_lower for part in search_parts):
            return model_file

    return None


def find_model_path(model_spec: str) -> Path:
    """
    Find model path from various sources.

    Args:
        model_spec: Model name or path. Can be:
            - Full path to .gguf file
            - Model name to search in Ollama cache
            - Model name pattern to search

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
    if found := find_model_in_ollama_cache(model_spec):
        return found

    # Not found
    raise FileNotFoundError(
        f"Model not found: {model_spec}\n"
        f"  - Provide full path to .gguf file\n"
        f"  - Or pull via Ollama: ollama pull {model_spec}\n"
        f"  - Or download manually to ~/models/"
    )


def find_mmproj_for_model(model_path: Path) -> Path | None:
    """Find the mmproj file for a vision model."""
    # Look in the same directory for mmproj file
    # Ollama stores them as: ggml-org_Model-Name_mmproj-model-name.gguf

    # Extract the prefix before the model filename
    # e.g., ggml-org_Qwen3-VL-30B-A3B-Instruct-Q8_0-GGUF_qwen3-vl-30b-a3b-instruct-q8_0.gguf
    # -> ggml-org_Qwen3-VL-30B-A3B-Instruct-Q8_0-GGUF_mmproj-

    for mmproj_file in model_path.parent.glob("*mmproj*.gguf"):
        # Check if it's related to this model
        model_name_part = model_path.stem.split("_")[-1] if "_" in model_path.stem else model_path.stem
        if model_name_part.lower().replace("-", "") in mmproj_file.name.lower().replace("-", ""):
            return mmproj_file

    return None


async def check_llamacpp_available(api_base: str):
    """Check if llama-server is running and get model info."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Get models (skip health check as it may not exist on all versions)
            response = await client.get(f"{api_base}/v1/models")
            response.raise_for_status()
            data = response.json()

            models = data.get("data", [])
            if not models:
                print("❌ No models loaded in llama-server!")
                return False

            print(f"✓ llama-server is running at {api_base}")
            print(f"  Found {len(models)} model(s):")
            for model in models:
                model_id = model.get("id", "unknown")
                meta = model.get("meta", {})
                size_gb = meta.get("size", 0) / (1024**3) if meta.get("size") else 0
                print(f"  - {model_id}")
                if size_gb:
                    print(f"    Size: {size_gb:.2f} GB")
            return True

    except (httpx.ConnectError, httpx.HTTPError) as e:
        print(f"❌ llama-server is not running at {api_base}!")
        print("\n   Start llama-server:")
        print("     llama-server -m /path/to/model.gguf --port 8033")
        print("\n   Or use Ollama models:")
        print("     See llamacpp_ollama_usage_examples.py")
        return False


async def main():
    """Run all llama.cpp demos"""
    parser = argparse.ArgumentParser(description="llama.cpp Provider Examples")
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Model for main demos (default: gpt-oss:20b from Ollama cache)",
    )
    parser.add_argument(
        "--vision-model",
        default="qwen3-vl",
        help="Model for vision demos (default: qwen3-vl from Ollama cache)",
    )
    parser.add_argument(
        "--api-base",
        help="Use existing llama-server instead of auto-start",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start servers (requires --api-base)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8033,
        help="Port for main model server (default: 8033)",
    )
    parser.add_argument(
        "--vision-port",
        type=int,
        default=8034,
        help="Port for vision model server (default: 8034)",
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
    print("🦙 llama.cpp Provider Examples (Auto-managed Servers)")
    print("=" * 70)

    # Use LlamaCppLLMClient for auto-managed server
    from chuk_llm.llm.providers.llamacpp_client import LlamaCppLLMClient

    # Determine if we're auto-starting or using existing server
    if args.api_base or args.no_auto_start:
        # Use existing server
        from chuk_llm.llm.providers.openai_client import OpenAILLMClient

        api_base = args.api_base or f"http://localhost:{args.port}"
        if not await check_llamacpp_available(api_base):
            sys.exit(1)

        print(f"Main Model: {args.model}")
        print(f"API Base: {api_base}")
        print("=" * 70)

        # Use OpenAILLMClient when connecting to existing server
        client = OpenAILLMClient(model=args.model, api_base=api_base)
        vision_client = client

        # Run demos (no auto-cleanup needed)
        await run_demos(args, client, vision_client)

    else:
        # Auto-start servers
        print("🚀 Auto-starting llama-server instances...")

        try:
            # Find models
            main_model_path = find_model_path(args.model)
            print(f"✓ Main model: {main_model_path.name}")

            vision_model_path = None
            if not args.skip_vision:
                try:
                    vision_model_path = find_model_path(args.vision_model)
                    print(f"✓ Vision model: {vision_model_path.name}")
                except FileNotFoundError as e:
                    print(f"⚠️  Vision model not found, skipping vision demos: {e}")
                    args.skip_vision = True

        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

        # Find available ports
        main_port = find_available_port(args.port)
        if main_port != args.port:
            print(f"⚠️  Port {args.port} in use, using {main_port} instead")

        # Find vision port, making sure it's different from main_port
        vision_port = None
        if not args.skip_vision and vision_model_path:
            # Start searching from vision_port, but skip main_port
            vision_port = find_available_port(args.vision_port)
            if vision_port == main_port:
                # If vision_port is same as main_port, find next available
                vision_port = find_available_port(main_port + 1)
            if vision_port != args.vision_port:
                print(f"⚠️  Port {args.vision_port} in use, using {vision_port} instead")

        print("=" * 70)

        # Create auto-managed clients
        async with LlamaCppLLMClient(
            model=main_model_path,
            port=main_port,
            ctx_size=8192,
            n_gpu_layers=-1,
        ) as client:
            print(f"✓ Main server started on port {main_port}")
            print(f"  Model name: {client.model}")

            main_api_base = client._server_manager.base_url

            # Create vision client if needed
            if not args.skip_vision and vision_model_path:
                # Find mmproj file for vision model
                mmproj_path = find_mmproj_for_model(vision_model_path)
                extra_args = []
                if mmproj_path:
                    print(f"✓ Found mmproj: {mmproj_path.name}")
                    extra_args = ["--mmproj", str(mmproj_path)]
                else:
                    print("⚠️  No mmproj file found for vision model")

                vision_client = LlamaCppLLMClient(
                    model=vision_model_path,
                    port=vision_port,
                    ctx_size=4096,
                    n_gpu_layers=-1,
                    extra_args=extra_args,
                )

                try:
                    # Start vision server
                    await vision_client.start_server()
                    print(f"✓ Vision server started on port {vision_port}")
                    print(f"  Model name: {vision_client.model}")
                    print("=" * 70)

                    vision_api_base = vision_client._server_manager.base_url

                    # Run demos with both servers
                    await run_demos(args, client, vision_client, main_api_base, vision_api_base)
                finally:
                    # Clean up vision server
                    await vision_client.close()
            else:
                # Run demos without vision
                print("=" * 70)
                await run_demos(args, client, client, main_api_base)

        print("\n🛑 Servers stopped")

    print("\n" + "=" * 70)
    print("✓ All demos completed!")
    print("=" * 70)
    print("\n💡 Tips:")
    print("  - llama.cpp runs locally - no API key needed!")
    print("  - Auto-starts servers with different models for different tasks")
    print("  - Models pulled via Ollama are automatically reused")
    print("  - Performance tuning: adjust --ctx-size, --n-gpu-layers, --threads")
    print("\n💡 Try different models:")
    print(f"  python {__file__} --model gpt-oss:20b --vision-model qwen2-vl")
    print(f"  python {__file__} --model llama3.2 --skip-vision")


async def run_demos(args, client, vision_client, main_api_base=None, vision_api_base=None):
    """Run the demos with the given clients."""
    # Get actual model names from clients
    main_model = client.model
    vision_model = vision_client.model if vision_client != client else main_model

    if args.demo:
        # Run specific demo
        demo_map = {
            1: ("Basic Completion", demo_basic_completion(client, "llamacpp", main_model)),
            2: ("Streaming", demo_streaming(client, "llamacpp", main_model)),
            3: ("Function Calling", demo_function_calling(client, "llamacpp", main_model)),
            4: ("Vision", demo_vision(vision_client, "llamacpp", vision_model)),
            5: ("Vision with URL", demo_vision_url(vision_client, "llamacpp", vision_model)),
            6: ("JSON Mode", demo_json_mode(client, "llamacpp", main_model)),
            7: ("Conversation", demo_conversation(client, "llamacpp", main_model)),
            8: ("Model Discovery", demo_model_discovery(client, "llamacpp", main_model)),
            9: ("Parameters", demo_parameters(client, "llamacpp", main_model)),
            10: ("Model Comparison", demo_model_comparison("llamacpp", [main_model])),
            11: ("Dynamic Model Call", demo_dynamic_model_call("llamacpp")),
            12: ("Error Handling", demo_error_handling(client, "llamacpp", main_model)),
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
            skip_tools=args.skip_tools,  # gpt-oss supports function calling with --jinja flag
            skip_discovery=True,  # llama.cpp models are local, discovery not applicable
            vision_client=vision_client if vision_client != client else None,
            vision_model=vision_model if vision_client != client else None,
        )


if __name__ == "__main__":
    asyncio.run(main())
