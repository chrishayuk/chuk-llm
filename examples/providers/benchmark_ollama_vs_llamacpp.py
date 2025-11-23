#!/usr/bin/env python3
"""
Ollama vs llama.cpp Performance Benchmark
==========================================

Compare tokens per second between Ollama and llama.cpp using the same GGUF model.

This benchmark:
1. Discovers Ollama models
2. Runs the same prompt on both Ollama and llama.cpp
3. Measures tokens/second for fair comparison
4. Shows detailed performance metrics

Prerequisites:
- Ollama running with at least one model pulled
- llama-server binary in PATH (brew install llama.cpp)
"""

import asyncio
import time
from pathlib import Path

from chuk_llm.core import Message, MessageRole
from chuk_llm.llm.providers.ollama_client import OllamaLLMClient
from chuk_llm.llm.providers.llamacpp_server import (
    LlamaCppServerConfig,
    LlamaCppServerManager,
)
from chuk_llm.llm.providers.openai_client import OpenAILLMClient
from chuk_llm.registry.resolvers.llamacpp_ollama import discover_ollama_models


async def benchmark_provider(
    client, provider_name: str, model_name: str, prompt: str
) -> dict:
    """
    Benchmark a provider's token generation speed.

    Returns:
        dict with metrics: tokens, duration, tok/s, etc.
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {provider_name}")
    print(f"Model: {model_name}")
    print(f"{'=' * 70}")

    messages = [Message(role=MessageRole.USER, content=prompt)]

    print(f"\n⏳ Running benchmark...")
    print("-" * 70)

    full_response = ""
    token_count = 0
    start_time = time.time()
    first_token_time = None

    try:
        async for chunk in client.create_completion(
            messages, stream=True, max_tokens=300, temperature=0.7
        ):
            if chunk.get("response"):
                if first_token_time is None:
                    first_token_time = time.time()

                content = chunk["response"]
                print(content, end="", flush=True)
                full_response += content
                # Rough approximation: ~4 characters per token
                token_count += max(1, len(content) // 4)

        end_time = time.time()

        # Calculate metrics
        total_duration = end_time - start_time
        time_to_first_token = (first_token_time - start_time) if first_token_time else 0
        generation_duration = end_time - (first_token_time or start_time)
        tokens_per_second = (
            token_count / generation_duration if generation_duration > 0 else 0
        )

        print("\n" + "-" * 70)

        return {
            "provider": provider_name,
            "model": model_name,
            "response": full_response,
            "token_count": token_count,
            "time_to_first_token": time_to_first_token,
            "generation_duration": generation_duration,
            "total_duration": total_duration,
            "tokens_per_second": tokens_per_second,
            "success": True,
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            "provider": provider_name,
            "model": model_name,
            "success": False,
            "error": str(e),
        }


async def main():
    """Run Ollama vs llama.cpp benchmark."""
    print("=" * 70)
    print("Ollama vs llama.cpp Performance Benchmark")
    print("=" * 70)

    # Standard prompt for consistent comparison
    prompt = "Write a detailed 200-word explanation of how neural networks learn through backpropagation."

    # Discover Ollama models
    print("\n🔍 Discovering Ollama models...")
    all_models = discover_ollama_models()

    # Filter for models with proper names (not SHA-256 digests)
    ollama_models = [m for m in all_models if not m.name.startswith("sha256-")]

    if not ollama_models:
        print("\n❌ No Ollama models found with proper names!")
        print("\nTo download a model:")
        print("  ollama pull llama3.2")
        print("  ollama pull mistral")
        return

    print(f"\n✓ Found {len(ollama_models)} Ollama model(s) with proper names")

    # Use the first (smallest) model for fair comparison
    model = ollama_models[0]
    # Extract clean model name for Ollama
    # Ollama expects just "model:tag" without "library/" prefix
    # E.g., "library/llama3.2:latest" -> "llama3.2:latest"
    # E.g., "vanilj/Phi-4:latest" -> "vanilj/Phi-4:latest"
    if model.name.startswith("library/"):
        model_name = model.name.replace("library/", "")
    else:
        model_name = model.name

    print(f"\nUsing model: {model_name}")
    print(f"Full name: {model.name}")
    print(f"GGUF path: {model.gguf_path}")
    print(f"Size: {model.size_bytes / (1024**3):.2f} GB")

    # Results storage
    results = []

    # =================================================================
    # Benchmark 1: Ollama
    # =================================================================
    print(f"\n{'#' * 70}")
    print("# Benchmark 1: Ollama")
    print(f"{'#' * 70}")

    ollama_client = OllamaLLMClient(model=model_name)
    ollama_result = await benchmark_provider(
        ollama_client, "Ollama", model_name, prompt
    )
    results.append(ollama_result)

    # =================================================================
    # Benchmark 2: llama.cpp
    # =================================================================
    print(f"\n{'#' * 70}")
    print("# Benchmark 2: llama.cpp")
    print(f"{'#' * 70}")

    try:
        # Start llama-server with the same GGUF file
        server_config = LlamaCppServerConfig(
            model_path=model.gguf_path,
            port=8033,
            ctx_size=2048,  # Match Ollama's default
            n_gpu_layers=-1,  # Use all GPU layers
        )

        async with LlamaCppServerManager(server_config) as server:
            # Get actual model name from server
            import httpx

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(f"{server.base_url}/v1/models")
                models_data = response.json()
                actual_model_name = (
                    models_data["data"][0]["id"]
                    if models_data.get("data")
                    else model_name
                )

            # Create OpenAI client pointing to llama.cpp server
            llamacpp_client = OpenAILLMClient(
                model=actual_model_name,
                api_base=server.base_url,
            )

            llamacpp_result = await benchmark_provider(
                llamacpp_client, "llama.cpp", model_name, prompt
            )
            results.append(llamacpp_result)

    except FileNotFoundError as e:
        print(f"\n❌ llama.cpp not found: {e}")
        print("\nPlease install llama.cpp:")
        print("  - macOS: brew install llama.cpp")
        print("  - Linux: Build from https://github.com/ggerganov/llama.cpp")
        results.append(
            {
                "provider": "llama.cpp",
                "model": model_name,
                "success": False,
                "error": str(e),
            }
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        results.append(
            {
                "provider": "llama.cpp",
                "model": model_name,
                "success": False,
                "error": str(e),
            }
        )

    # =================================================================
    # Results Summary
    # =================================================================
    print(f"\n{'=' * 70}")
    print("Performance Comparison Summary")
    print(f"{'=' * 70}")

    successful_results = [r for r in results if r.get("success")]

    if not successful_results:
        print("\n❌ No successful benchmarks to compare")
        return

    print(f"\nModel: {model_name}")
    print(f"Prompt: {prompt[:60]}...")
    print(
        f"\n{'Provider':<15} {'Tokens':<10} {'TTFT':<10} {'Gen Time':<12} {'Tok/s':<10}"
    )
    print("-" * 70)

    for result in successful_results:
        print(
            f"{result['provider']:<15} "
            f"{result['token_count']:<10} "
            f"{result['time_to_first_token']:<10.3f} "
            f"{result['generation_duration']:<12.3f} "
            f"{result['tokens_per_second']:<10.1f}"
        )

    # Calculate speedup if both succeeded
    if len(successful_results) == 2:
        ollama_tps = successful_results[0]["tokens_per_second"]
        llamacpp_tps = successful_results[1]["tokens_per_second"]

        if ollama_tps > 0 and llamacpp_tps > 0:
            print(f"\n{'=' * 70}")
            if llamacpp_tps > ollama_tps:
                speedup = llamacpp_tps / ollama_tps
                print(
                    f"🚀 llama.cpp is {speedup:.2f}x faster than Ollama ({llamacpp_tps:.1f} vs {ollama_tps:.1f} tok/s)"
                )
            else:
                speedup = ollama_tps / llamacpp_tps
                print(
                    f"🚀 Ollama is {speedup:.2f}x faster than llama.cpp ({ollama_tps:.1f} vs {llamacpp_tps:.1f} tok/s)"
                )

    print(f"\n{'=' * 70}")
    print("Benchmark complete!")
    print(f"{'=' * 70}")

    print("\n💡 Tips for improving performance:")
    print("  • Ensure GPU acceleration is enabled (Metal/CUDA/ROCm)")
    print("  • Use smaller quantization models (Q4_K_M vs Q8_0)")
    print("  • Adjust batch size and context window")
    print("  • Try different n_gpu_layers settings")


if __name__ == "__main__":
    asyncio.run(main())
