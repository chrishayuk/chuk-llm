#!/usr/bin/env python3
"""
OpenAI-Compatible Providers - Comprehensive Example
====================================================

Demonstrates using OpenAI-compatible providers (Groq, DeepSeek, Together, Mistral, etc.)
with the OpenAICompatibleClient.

These providers implement the OpenAI Chat Completions API format, allowing you to use
the same client code across multiple providers.

Features Demonstrated:
- ✅ Multiple provider support (Groq, DeepSeek, Together, Mistral, Perplexity)
- ✅ Basic completion
- ✅ Streaming
- ✅ Function/tool calling
- ✅ Provider comparison
- ✅ Error handling
- ✅ Zero magic strings (all enums)
- ✅ Type-safe Pydantic V2

Requirements:
- Set provider-specific API keys:
  - GROQ_API_KEY
  - DEEPSEEK_API_KEY
  - TOGETHER_API_KEY
  - MISTRAL_API_KEY
  - PERPLEXITY_API_KEY

Usage:
    python openai_compatible_example.py
    python openai_compatible_example.py --provider groq
    python openai_compatible_example.py --demo 1  # Run specific demo
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add src to path for development
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from chuk_llm.clients.openai_compatible import OpenAICompatibleClient
    from chuk_llm.core.models import (
        CompletionRequest,
        CompletionResponse,
        Message,
        Tool,
        ToolFunction,
    )
    from chuk_llm.core.enums import MessageRole, ToolType
    from chuk_llm.core import LLMError
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# Provider configurations
PROVIDERS = {
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "supports_tools": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "supports_tools": True,
    },
    "together": {
        "name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "supports_tools": True,
    },
    "mistral": {
        "name": "Mistral",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "default_model": "mistral-small-latest",
        "supports_tools": True,
    },
    "perplexity": {
        "name": "Perplexity",
        "base_url": "https://api.perplexity.ai",
        "api_key_env": "PERPLEXITY_API_KEY",
        "default_model": "llama-3.1-sonar-small-128k-online",
        "supports_tools": False,
    },
}


async def demo_basic_completion(provider_key: str):
    """Demo 1: Basic completion with OpenAI-compatible provider."""
    config = PROVIDERS[provider_key]
    print(f"\n{'='*60}")
    print(f"Demo 1: Basic Completion - {config['name']}")
    print(f"{'='*60}")
    print(f"Provider: {config['name']}")
    print(f"Model: {config['default_model']}")
    print(f"Base URL: {config['base_url']}")

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"⚠️  {config['api_key_env']} not set, skipping")
        return

    client = OpenAICompatibleClient(
        model=config["default_model"],
        api_key=api_key,
        base_url=config["base_url"],
    )

    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful AI assistant.",
            ),
            Message(
                role=MessageRole.USER,
                content=f"What is {config['name']}? (One sentence)",
            ),
        ],
        model=config["default_model"],
        temperature=0.7,
        max_tokens=100,
    )

    response: CompletionResponse = await client.complete(request)

    print(f"\n✅ Response: {response.content}")
    print(f"   Finish: {response.finish_reason}")
    print(f"   Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")

    await client.close()


async def demo_streaming(provider_key: str):
    """Demo 2: Streaming with OpenAI-compatible provider."""
    config = PROVIDERS[provider_key]
    print(f"\n{'='*60}")
    print(f"Demo 2: Streaming - {config['name']}")
    print(f"{'='*60}")
    print(f"Provider: {config['name']}")
    print(f"Feature: Real-time response streaming")

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"⚠️  {config['api_key_env']} not set, skipping")
        return

    client = OpenAICompatibleClient(
        model=config["default_model"],
        api_key=api_key,
        base_url=config["base_url"],
    )

    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Write a haiku about cloud computing.",
            )
        ],
        model=config["default_model"],
    )

    print("\n🌊 Streaming:")
    print("   ", end="", flush=True)

    async for chunk in client.stream(request):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n✅ Complete")

    await client.close()


async def demo_function_calling(provider_key: str):
    """Demo 3: Function calling with OpenAI-compatible provider."""
    config = PROVIDERS[provider_key]
    print(f"\n{'='*60}")
    print(f"Demo 3: Function Calling - {config['name']}")
    print(f"{'='*60}")
    print(f"Provider: {config['name']}")
    print(f"Supports tools: {config['supports_tools']}")

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"⚠️  {config['api_key_env']} not set, skipping")
        return

    if not config["supports_tools"]:
        print(f"⚠️  {config['name']} does not support function calling, skipping")
        return

    client = OpenAICompatibleClient(
        model=config["default_model"],
        api_key=api_key,
        base_url=config["base_url"],
    )

    # Define tools
    tools = [
        Tool(
            type=ToolType.FUNCTION,
            function=ToolFunction(
                name="get_weather",
                description="Get current weather for a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            ),
        ),
    ]

    request = CompletionRequest(
        messages=[
            Message(
                role=MessageRole.USER,
                content="What's the weather in London?",
            )
        ],
        model=config["default_model"],
        tools=tools,
        temperature=0.0,
    )

    response = await client.complete(request)

    if response.tool_calls:
        print(f"✅ Tool calls: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"   📞 {tc.function.name}({tc.function.arguments})")
    else:
        print(f"ℹ️  No tools called: {response.content}")

    await client.close()


async def demo_provider_comparison():
    """Demo 4: Compare multiple OpenAI-compatible providers."""
    print(f"\n{'='*60}")
    print(f"Demo 4: Provider Comparison")
    print(f"{'='*60}")
    print(f"Comparing OpenAI-compatible providers")

    prompt = "What is the capital of France? (One word)"

    for provider_key, config in PROVIDERS.items():
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            print(f"\n⚠️  {config['name']}: API key not set")
            continue

        client = None
        try:
            import time

            client = OpenAICompatibleClient(
                model=config["default_model"],
                api_key=api_key,
                base_url=config["base_url"],
            )

            request = CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=config["default_model"],
                temperature=0.0,
                max_tokens=10,
            )

            start = time.time()
            response = await client.complete(request)
            duration = time.time() - start

            print(f"\n✅ {config['name']} ({duration:.2f}s):")
            print(f"   Model: {config['default_model']}")
            print(f"   Response: {response.content}")

        except (Exception, LLMError) as e:
            print(f"\n❌ {config['name']}: {str(e)[:100]}")
        finally:
            if client:
                await client.close()


async def demo_error_handling(provider_key: str):
    """Demo 5: Error handling with OpenAI-compatible provider."""
    config = PROVIDERS[provider_key]
    print(f"\n{'='*60}")
    print(f"Demo 5: Error Handling - {config['name']}")
    print(f"{'='*60}")
    print(f"Provider: {config['name']}")

    # Test with invalid API key
    bad_client = OpenAICompatibleClient(
        model=config["default_model"],
        api_key="invalid-key",
        base_url=config["base_url"],
    )

    try:
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model=config["default_model"],
        )

        await bad_client.complete(request)

    except LLMError as e:
        print(f"✅ LLM error caught:")
        print(f"   Type: {e.error_type}")
        print(f"   Message: {e.error_message}")
    except Exception as e:
        print(f"✅ Error caught:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")

    await bad_client.close()


async def demo_model_discovery(provider_key: str):
    """Demo 6: Model discovery for OpenAI-compatible providers."""
    config = PROVIDERS[provider_key]
    print(f"\n{'='*60}")
    print(f"Demo 6: Model Discovery - {config['name']}")
    print(f"{'='*60}")
    print(f"Provider: {config['name']}")
    print(f"Using OpenAICompatibleDiscoverer for dynamic discovery")

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"⚠️  {config['api_key_env']} not set, skipping")
        return

    try:
        from chuk_llm.llm.discovery.general_discoverers import (
            OpenAICompatibleDiscoverer,
        )

        print(f"\n🔍 Discovering models from {config['name']} API...")
        discoverer = OpenAICompatibleDiscoverer(
            provider_name=provider_key,
            api_key=api_key,
            api_base=config["base_url"],
        )

        models_data = await discoverer.discover_models()
        print(f"\n📊 Found {len(models_data)} models")

        # Display models with capabilities
        for i, model in enumerate(models_data[:10], 1):  # Show first 10
            model_name = model.get("name")
            provider_spec = model.get("provider_specific", {})

            # Build capability string
            caps = []
            if provider_spec.get("reasoning_capable"):
                caps.append("🧠 reasoning")
            if provider_spec.get("supports_tools"):
                caps.append("🔧 tools")
            if provider_spec.get("supports_vision"):
                caps.append("👁️ vision")
            if provider_spec.get("has_web_search"):
                caps.append("🔍 search")

            cap_str = f" [{', '.join(caps)}]" if caps else ""
            print(f"   {i}. {model_name}{cap_str}")

        if len(models_data) > 10:
            print(f"   ... and {len(models_data) - 10} more models")

        # Test a dynamically discovered model
        if models_data:
            test_model = models_data[0].get("name")
            print(f"\n🧪 Testing dynamically discovered model: {test_model}")
            try:
                client = OpenAICompatibleClient(
                    model=test_model,
                    api_key=api_key,
                    base_url=config["base_url"],
                )

                request = CompletionRequest(
                    messages=[
                        Message(
                            role=MessageRole.USER,
                            content="Say hello in one creative word",
                        )
                    ],
                    model=test_model,
                    max_tokens=10,
                )

                response = await client.complete(request)
                print(f"   ✅ Model works: {response.content}")
                await client.close()
            except Exception as e:
                print(f"   ⚠️ Test failed: {str(e)[:100]}")

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback

        traceback.print_exc()


async def demo_multi_provider():
    """Demo 7: Using multiple providers in one application."""
    print(f"\n{'='*60}")
    print(f"Demo 7: Multi-Provider Application")
    print(f"{'='*60}")
    print(f"Using different providers for different tasks")

    # Fast provider for simple tasks (Groq)
    if os.getenv("GROQ_API_KEY"):
        print(f"\n🚀 Fast provider (Groq) for simple task:")
        groq_config = PROVIDERS["groq"]
        groq_client = None
        try:
            groq_client = OpenAICompatibleClient(
                model=groq_config["default_model"],
                api_key=os.getenv("GROQ_API_KEY"),
                base_url=groq_config["base_url"],
            )

            request = CompletionRequest(
                messages=[Message(role=MessageRole.USER, content="Say hello")],
                model=groq_config["default_model"],
                max_tokens=20,
            )

            response = await groq_client.complete(request)
            print(f"   {response.content}")
        except (Exception, LLMError) as e:
            print(f"   ❌ Error: {str(e)[:100]}")
        finally:
            if groq_client:
                await groq_client.close()

    # Research provider (Perplexity) for web search
    if os.getenv("PERPLEXITY_API_KEY"):
        print(f"\n🔍 Research provider (Perplexity) for web info:")
        pplx_config = PROVIDERS["perplexity"]
        pplx_client = None
        try:
            pplx_client = OpenAICompatibleClient(
                model=pplx_config["default_model"],
                api_key=os.getenv("PERPLEXITY_API_KEY"),
                base_url=pplx_config["base_url"],
            )

            request = CompletionRequest(
                messages=[
                    Message(
                        role=MessageRole.USER,
                        content="What's the latest news about AI? (One sentence)",
                    )
                ],
                model=pplx_config["default_model"],
                max_tokens=100,
            )

            response = await pplx_client.complete(request)
            print(f"   {response.content}")
        except (Exception, LLMError) as e:
            print(f"   ❌ Error: {str(e)[:100]}")
        finally:
            if pplx_client:
                await pplx_client.close()

    print(f"\n✅ Multi-provider pattern demonstrated")


async def main():
    """Run all demos."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="groq",
        choices=list(PROVIDERS.keys()),
        help="Provider to use",
    )
    parser.add_argument("--demo", type=int, help="Run specific demo (1-6)")

    args = parser.parse_args()

    print("🚀 OpenAI-Compatible Providers - Comprehensive Examples")
    print("=" * 60)
    print("Supported Providers:")
    for key, config in PROVIDERS.items():
        status = "✅" if os.getenv(config["api_key_env"]) else "⚠️ "
        print(f"  {status} {config['name']}: {config['default_model']}")
    print("=" * 60)
    print("Features:")
    print("  ✅ Same API across all providers")
    print("  ✅ Type-safe Pydantic V2 models")
    print("  ✅ Zero magic strings (all enums)")
    print("  ✅ Easy provider switching")
    print("=" * 60)

    demos = [
        lambda: demo_basic_completion(args.provider),
        lambda: demo_streaming(args.provider),
        lambda: demo_function_calling(args.provider),
        demo_provider_comparison,
        lambda: demo_error_handling(args.provider),
        lambda: demo_model_discovery(args.provider),
        demo_multi_provider,
    ]

    if args.demo:
        if 1 <= args.demo <= len(demos):
            await demos[args.demo - 1]()
        else:
            print(f"❌ Invalid demo. Choose 1-{len(demos)}")
            sys.exit(1)
    else:
        for demo in demos:
            try:
                await demo()
            except (Exception, LLMError) as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()

    print(f"\n{'='*60}")
    print("🎉 All Demos Complete!")
    print("=" * 60)
    print("\nKey Takeaway:")
    print("  OpenAI-compatible providers use the same API format")
    print("  Switch providers by changing base_url and api_key")
    print("  Same type-safe Pydantic models across all providers")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
