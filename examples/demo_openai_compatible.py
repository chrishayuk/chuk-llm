#!/usr/bin/env python3
"""
Using Custom OpenAI-Compatible API with ChukLLM (No Warnings)
============================================================

Simple example using environment variables with warning suppression.

Set these environment variables:
  export OPENAI_API_BASE="https://vllm.1jwefpwyy4k2.us-south.codeengine.appdomain.cloud/v1"
  export OPENAI_API_KEY="mytoken"
  export OPENAI_MODEL="azure-gpt-4o"

Then just run:
  python custom_openai_api.py
"""

import asyncio
import logging
import os

# Suppress the configuration warnings for openai_compatible
logging.getLogger("chuk_llm.llm.providers._config_mixin").setLevel(logging.ERROR)

from chuk_llm.llm.providers.openai_client import OpenAILLMClient


async def main():
    """Main example using environment variables."""

    # Get configuration from environment variables
    api_base = os.getenv(
        "OPENAI_API_BASE",
        "https://vllm.1jwefpwyy4k2.us-south.codeengine.appdomain.cloud/v1",
    )
    api_key = os.getenv("OPENAI_API_KEY", "mytoken")
    model = os.getenv("OPENAI_MODEL", "azure-gpt-4o")

    print("🚀 Custom OpenAI-Compatible API")
    print(f"  Endpoint: {api_base}")
    print(f"  Model: {model}")
    print("-" * 50)

    # List available models (if endpoint supports it)
    print("\n📋 Available Models:")
    try:
        from chuk_llm.llm.discovery.openai_discoverer import OpenAIModelDiscoverer

        discoverer = OpenAIModelDiscoverer(
            provider_name="custom", api_key=api_key, api_base=api_base
        )
        models = await discoverer.discover_models()
        if models:
            for model_info in models[:10]:  # Show first 10 models
                model_name = model_info.get("name", model_info.get("id", "unknown"))
                print(f"  • {model_name}")
            if len(models) > 10:
                print(f"  ... and {len(models) - 10} more")
        else:
            print("  (Model listing not available)")
    except Exception as e:
        print(f"  (Could not list models: {e})")

    print("-" * 50)

    # Create client using environment variables
    client = OpenAILLMClient(model=model, api_key=api_key, api_base=api_base)

    # Optional: Force detection as "openai" to use OpenAI defaults
    # This eliminates all warnings about missing config
    client.detected_provider = "openai"
    client.provider_name = "openai"  # For the config mixin

    # 1. Simple chat
    print("\n💬 Chat:")
    response = await client.create_completion(
        messages=[{"role": "user", "content": "What is OpenShift?"}], max_tokens=100
    )
    print(response.get("response"))

    # 2. Streaming
    print("\n⚡ Streaming:")
    async for chunk in client.create_completion(
        messages=[{"role": "user", "content": "List 3 benefits of Kubernetes"}],
        stream=True,
        max_tokens=100,
    ):
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)
    print()

    # 3. Conversation with context
    print("\n🗨️ Conversation:")
    conversation = [{"role": "user", "content": "What's Docker?"}]

    response = await client.create_completion(conversation, max_tokens=100)
    print(f"Assistant: {response.get('response')}")

    # Add to conversation and continue
    conversation.append({"role": "assistant", "content": response.get("response")})
    conversation.append(
        {"role": "user", "content": "How does it relate to Kubernetes?"}
    )

    response = await client.create_completion(conversation, max_tokens=100)
    print("\nUser: How does it relate to Kubernetes?")
    print(f"Assistant: {response.get('response')}")

    # 4. Tool calling (if supported by your endpoint)
    print("\n🔧 Tool Calling:")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city/timezone",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    try:
        response = await client.create_completion(
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco and what time is it there?",
                }
            ],
            tools=tools,
            max_tokens=150,
        )

        if response.get("tool_calls"):
            print("Tool calls requested:")
            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                print(f"  • {func_name}: {func_args}")

        if response.get("response"):
            print(f"Response: {response.get('response')}")
    except Exception as e:
        print(f"Tools not supported or error: {e}")

    # Cleanup
    await client.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
