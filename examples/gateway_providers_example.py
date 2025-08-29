#!/usr/bin/env python3
"""
Gateway Providers Example
=========================

This example demonstrates using popular LLM gateway and proxy services
that provide unified access to multiple LLM providers.
"""

import os

from chuk_llm import ask_sync, stream_sync_iter


def demo_litellm():
    """
    Demonstrate LiteLLM gateway usage.
    LiteLLM provides a unified interface to 100+ LLM providers.
    """
    print("\n🚀 LITELLM GATEWAY")
    print("-" * 40)

    # Check if LiteLLM is configured
    if not os.getenv("LITELLM_API_KEY"):
        print("ℹ️  LiteLLM Setup Instructions:")
        print("1. Install: pip install litellm")
        print("2. Start proxy: litellm --config config.yaml")
        print("3. Set environment: export LITELLM_API_KEY=your-key")
        print("4. (Optional) export LITELLM_API_BASE=http://your-server:4000")
        print("\nExample config.yaml:")
        print("""
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: $OPENAI_API_KEY
  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-sonnet
      api_key: $ANTHROPIC_API_KEY
""")
        return

    # Use LiteLLM gateway
    try:
        response = ask_sync(
            "What is LiteLLM?",
            provider="litellm",
            model="gpt-3.5-turbo",  # Or any model configured in your gateway
            max_tokens=50,
        )
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"⚠️  Error: {e}")


def demo_openrouter():
    """
    Demonstrate OpenRouter usage.
    OpenRouter provides access to 100+ models through a unified API.
    """
    print("\n🌐 OPENROUTER")
    print("-" * 40)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ℹ️  OpenRouter Setup:")
        print("1. Sign up at https://openrouter.ai")
        print("2. Get API key from https://openrouter.ai/keys")
        print("3. Set environment: export OPENROUTER_API_KEY=sk-or-...")
        print("\nAvailable models include:")
        print("  - openai/gpt-4")
        print("  - anthropic/claude-3-opus")
        print("  - google/gemini-pro")
        print("  - meta-llama/llama-3-70b-instruct")
        print("  - deepseek/deepseek-chat")
        return

    try:
        # OpenRouter requires model names with provider prefix
        response = ask_sync(
            "What makes OpenRouter unique?",
            provider="openrouter",
            model="openai/gpt-3.5-turbo",  # Provider prefix required
            max_tokens=50,
            # OpenRouter supports custom headers
            extra_headers={
                "HTTP-Referer": "https://github.com/chuk-ai/chuk-llm",
                "X-Title": "ChukLLM Example",
            },
        )
        print(f"✅ Response: {response}")

        # Check different model pricing
        print("\n💰 Cost-effective option:")
        response = ask_sync(
            "Hello!",
            provider="openrouter",
            model="meta-llama/llama-3-70b-instruct",
            max_tokens=10,
        )
        print(f"   Llama response: {response}")

    except Exception as e:
        print(f"⚠️  Error: {e}")


def demo_vllm():
    """
    Demonstrate vLLM server usage.
    vLLM provides high-performance local inference.
    """
    print("\n⚡ VLLM SERVER")
    print("-" * 40)

    # Check if vLLM is running
    vllm_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")

    print("ℹ️  vLLM Setup:")
    print("1. Install: pip install vllm")
    print("2. Start server:")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("     --model meta-llama/Llama-3-8b-hf \\")
    print("     --port 8000")
    print("3. (Optional) export VLLM_API_BASE=http://localhost:8000/v1")
    print(f"\nConfigured endpoint: {vllm_base}")

    # Try to use vLLM if available
    try:
        response = ask_sync(
            "What is vLLM?",
            provider="vllm",
            model="meta-llama/Llama-3-8b-hf",  # Or your served model
            max_tokens=50,
            temperature=0.7,
        )
        print(f"\n✅ Response: {response}")

        # vLLM supports streaming
        print("\n🌊 Streaming response: ", end="", flush=True)
        for chunk in stream_sync_iter(
            "Count from 1 to 5", provider="vllm", max_tokens=20
        ):
            content = chunk if isinstance(chunk, str) else chunk.get("response", "")
            print(content, end="", flush=True)
        print()

    except Exception as e:
        if "Connection refused" in str(e):
            print(f"\n⚠️  vLLM server not running at {vllm_base}")
        else:
            print(f"\n⚠️  Error: {e}")


def demo_togetherai():
    """
    Demonstrate Together AI usage.
    Together AI provides scalable inference for open models.
    """
    print("\n🤝 TOGETHER AI")
    print("-" * 40)

    if not os.getenv("TOGETHER_API_KEY"):
        print("ℹ️  Together AI Setup:")
        print("1. Sign up at https://api.together.xyz")
        print("2. Get API key from https://api.together.xyz/settings/api-keys")
        print("3. Set environment: export TOGETHER_API_KEY=...")
        print("\nPopular models:")
        print("  - deepseek-ai/deepseek-v3 (SOTA reasoning)")
        print("  - meta-llama/Llama-3.3-70B-Instruct-Turbo")
        print("  - mistralai/Mixtral-8x7B-Instruct-v0.1")
        print("  - Qwen/QwQ-32B-Preview (reasoning)")
        return

    try:
        # Test with DeepSeek V3
        print("\n🧠 Testing DeepSeek V3:")
        response = ask_sync(
            "Solve: If x + 2 = 5, what is x?",
            provider="togetherai",
            model="deepseek-ai/deepseek-v3",
            max_tokens=50,
            temperature=0,
        )
        print(f"Response: {response}")

        # Test with Llama 3.3
        print("\n🦙 Testing Llama 3.3:")
        response = ask_sync(
            "Write a haiku about AI",
            provider="togetherai",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            max_tokens=50,
            temperature=0.7,
        )
        print(f"Response: {response}")

        # Test vision model
        print("\n👁️ Vision models available:")
        print("  - meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
        print("  - meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo")

    except Exception as e:
        print(f"⚠️  Error: {e}")


def demo_openai_compatible():
    """
    Demonstrate generic OpenAI-compatible configuration.
    Works with any service implementing the OpenAI API.
    """
    print("\n🔧 GENERIC OPENAI-COMPATIBLE")
    print("-" * 40)

    print("The openai_compatible provider works with any OpenAI-compatible service.")
    print("\nSupported services include:")
    print("  - LocalAI (local models)")
    print("  - FastChat (research models)")
    print("  - LM Studio (local GUI)")
    print("  - Ollama with OpenAI endpoint")
    print("  - Any custom OpenAI-compatible server")

    print("\nConfiguration options:")
    print("1. Use environment variables:")
    print("   export OPENAI_COMPATIBLE_API_BASE=http://localhost:8080/v1")
    print("   export OPENAI_COMPATIBLE_API_KEY=your-key")

    print("\n2. Or use dynamic registration:")
    print("""
from chuk_llm import register_openai_compatible

register_openai_compatible(
    name="my_local_ai",
    api_base="http://localhost:8080/v1",
    models=["llama3", "mistral", "phi3"]
)
""")

    # Try to use if configured
    if os.getenv("OPENAI_COMPATIBLE_API_BASE"):
        try:
            response = ask_sync("Hello!", provider="openai_compatible", max_tokens=10)
            print(f"\n✅ Response: {response}")
        except Exception as e:
            print(f"\n⚠️  Error: {e}")


def compare_gateways():
    """
    Compare different gateway options.
    """
    print("\n" + "=" * 60)
    print("GATEWAY COMPARISON")
    print("=" * 60)

    comparison = """
┌─────────────┬────────────────┬───────────┬──────────────┐
│ Gateway     │ Best For       │ Pricing   │ Key Features │
├─────────────┼────────────────┼───────────┼──────────────┤
│ LiteLLM     │ Self-hosted    │ Free      │ Fallbacks,   │
│             │ unified API    │           │ caching,     │
│             │                │           │ load balance │
├─────────────┼────────────────┼───────────┼──────────────┤
│ OpenRouter  │ Model variety  │ Pay per   │ 100+ models, │
│             │ & comparison   │ use       │ unified API, │
│             │                │           │ A/B testing  │
├─────────────┼────────────────┼───────────┼──────────────┤
│ vLLM        │ High-perf      │ Free      │ Fast local,  │
│             │ local serving  │ (self)    │ batching,    │
│             │                │           │ GPU optimized│
├─────────────┼────────────────┼───────────┼──────────────┤
│ Together AI │ Open models    │ Cheap     │ DeepSeek,    │
│             │ at scale       │ pay/use   │ Llama, fast  │
│             │                │           │ inference    │
└─────────────┴────────────────┴───────────┴──────────────┘
"""
    print(comparison)


def main():
    print("=" * 60)
    print("GATEWAY PROVIDERS EXAMPLE")
    print("=" * 60)
    print("""
Gateway providers offer unified access to multiple LLM models,
making it easy to switch between providers and compare performance.
""")

    # Run demonstrations
    demo_litellm()
    demo_openrouter()
    demo_vllm()
    demo_togetherai()
    demo_openai_compatible()
    compare_gateways()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Gateway providers enable:
✅ Single API for multiple LLM providers
✅ Easy model comparison and A/B testing
✅ Cost optimization across providers
✅ Fallback and load balancing
✅ Local and cloud deployment options

Quick Start:
1. Choose a gateway based on your needs
2. Set the appropriate API key environment variable
3. Use the provider name in your code:
   ask_sync("Hello", provider="litellm")
   ask_sync("Hello", provider="openrouter")
   ask_sync("Hello", provider="togetherai")

All gateways use the OpenAI-compatible client, ensuring
consistent behavior and feature support.
""")


if __name__ == "__main__":
    main()
