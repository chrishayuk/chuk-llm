#!/usr/bin/env python3
"""
Registry-Based Model Discovery
================================

Demonstrates the powerful registry system that discovers models
dynamically and selects them based on capabilities, not names.

This is THE key differentiator of chuk-llm!
"""

import asyncio
from dotenv import load_dotenv
load_dotenv()

from chuk_llm.registry import get_registry

async def discover_all_models():
    """Discover all available models from all providers."""
    print("=== Discovering All Available Models ===\n")

    registry = await get_registry()
    models = await registry.get_models()

    print(f"Found {len(models)} total models\n")

    # Group by provider
    from collections import defaultdict
    by_provider = defaultdict(list)

    for model in models:
        by_provider[model.spec.provider].append(model)

    for provider, provider_models in sorted(by_provider.items()):
        print(f"{provider}: {len(provider_models)} models")

async def intelligent_selection():
    """Select models based on capabilities, not names."""
    print("\n=== Intelligent Model Selection ===\n")

    registry = await get_registry()

    # Find best cheap model with tool support
    model = await registry.find_best(
        requires_tools=True,
        quality_tier="cheap"
    )

    if model:
        print(f"Best cheap model with tools:")
        print(f"  Provider: {model.spec.provider}")
        print(f"  Model: {model.spec.name}")
        print(f"  Context: {model.capabilities.max_context:,} tokens")
        print(f"  Tier: {model.capabilities.quality_tier.value}")

    # Find best model for vision
    model = await registry.find_best(
        requires_vision=True,
        quality_tier="balanced"
    )

    if model:
        print(f"\nBest balanced model with vision:")
        print(f"  Provider: {model.spec.provider}")
        print(f"  Model: {model.spec.name}")
        print(f"  Context: {model.capabilities.max_context:,} tokens")

    # Find fastest model (Groq)
    model = await registry.find_best(
        provider="groq",
        requires_tools=True
    )

    if model:
        print(f"\nFastest model with tools (Groq):")
        print(f"  Model: {model.spec.name}")
        if model.capabilities.tokens_per_second:
            print(f"  Speed: {model.capabilities.tokens_per_second:.0f} tokens/sec")

async def query_with_requirements():
    """Query models with multiple requirements."""
    print("\n=== Query with Multiple Requirements ===\n")

    registry = await get_registry()

    from chuk_llm.registry import ModelQuery

    # Find all models that meet specific requirements
    results = await registry.query(ModelQuery(
        requires_tools=True,
        requires_vision=True,
        min_context=100_000,
        max_cost_per_1m_input=2.0,
        quality_tier="any"
    ))

    print(f"Models with tools, vision, 100k+ context, and <$2/M:")
    for model in results[:5]:  # Show first 5
        print(f"  • {model.spec.provider}:{model.spec.name}")
        print(f"    Context: {model.capabilities.max_context:,} tokens")
        if model.capabilities.input_cost_per_1m:
            print(f"    Cost: ${model.capabilities.input_cost_per_1m:.2f}/M tokens")

async def find_specific_model():
    """Find a specific model by name."""
    print("\n=== Find Specific Model ===\n")

    registry = await get_registry()

    # Find by provider and name
    model = await registry.find_model("openai", "gpt-4o-mini")

    if model:
        print(f"Found model: {model.spec.name}")
        print(f"  Max context: {model.capabilities.max_context:,} tokens")
        print(f"  Max output: {model.capabilities.max_output_tokens:,} tokens")
        print(f"  Tools: {model.capabilities.supports_tools}")
        print(f"  Vision: {model.capabilities.supports_vision}")
        print(f"  JSON mode: {model.capabilities.supports_json_mode}")
        print(f"  Streaming: {model.capabilities.supports_streaming}")
        print(f"  Quality: {model.capabilities.quality_tier.value}")

async def use_with_ask():
    """Use registry results with the ask() function."""
    print("\n=== Using Registry with ask() ===\n")

    from chuk_llm import ask

    registry = await get_registry()

    # Find best cheap model
    model = await registry.find_best(
        requires_tools=True,
        quality_tier="cheap"
    )

    if model:
        print(f"Using: {model.spec.provider}:{model.spec.name}")

        # Use the selected model
        answer = await ask(
            "What is 2+2?",
            provider=model.spec.provider,
            model=model.spec.name
        )

        print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(discover_all_models())
    asyncio.run(intelligent_selection())
    asyncio.run(query_with_requirements())
    asyncio.run(find_specific_model())
    asyncio.run(use_with_ask())

    print("\n" + "="*50)
    print("✅ Registry system is the capability brain of chuk-llm!")
    print("="*50)
