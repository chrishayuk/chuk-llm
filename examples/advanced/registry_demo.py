#!/usr/bin/env python3
"""
Dynamic Model Registry Demo
============================

Demonstrates the new capability-based model registry system.

This shows how chuk-llm can now:
1. Dynamically discover available models
2. Resolve capabilities without hardcoding
3. Intelligently select the best model for a task
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from chuk_llm.registry import get_registry


async def main():
    print("🚀 CHUK-LLM DYNAMIC MODEL REGISTRY DEMO")
    print("=" * 60)

    # Get the registry
    print("\n📋 Initializing registry...")
    registry = await get_registry()

    # Discover all available models
    print("\n🔍 Discovering available models...")
    models = await registry.get_models()

    print(f"\nFound {len(models)} models:")
    for model in models:
        caps = model.capabilities
        features = []
        if caps.supports_tools:
            features.append("tools")
        if caps.supports_vision:
            features.append("vision")
        if caps.supports_streaming:
            features.append("streaming")

        ctx = f"{caps.max_context // 1000}k" if caps.max_context else "?"
        cost = f"${caps.input_cost_per_1m:.2f}/1M" if caps.input_cost_per_1m else "?"

        print(f"  • {model.spec.provider:15s} {model.spec.name:40s} | {ctx:6s} | {cost:12s} | {', '.join(features)}")

    # Find best model for different tasks
    print("\n" + "=" * 60)
    print("🎯 INTELLIGENT MODEL SELECTION")
    print("=" * 60)

    # Task 1: Fast, cheap model with tools
    print("\n1️⃣ Finding best cheap model with tool support...")
    best_cheap = await registry.find_best(
        requires_tools=True,
        quality_tier="cheap",
    )
    if best_cheap:
        print(f"   → {best_cheap}")
        print(f"      Cost: ${best_cheap.capabilities.input_cost_per_1m}/1M input")
    else:
        print("   → No matching model found")

    # Task 2: Best model with vision and large context
    print("\n2️⃣ Finding best model with vision + 128k context...")
    best_vision = await registry.find_best(
        requires_vision=True,
        min_context=128_000,
        quality_tier="any",
    )
    if best_vision:
        print(f"   → {best_vision}")
        print(f"      Context: {best_vision.capabilities.max_context:,} tokens")
    else:
        print("   → No matching model found")

    # Task 3: Fastest model with tools
    print("\n3️⃣ Finding fastest model with tools (Groq)...")
    best_fast = await registry.find_best(
        requires_tools=True,
        provider="groq",
    )
    if best_fast:
        print(f"   → {best_fast}")
        if best_fast.capabilities.speed_hint_tps:
            print(f"      Speed: ~{best_fast.capabilities.speed_hint_tps:.0f} tokens/sec")
    else:
        print("   → No matching model found")

    # Task 4: Local model (Ollama)
    print("\n4️⃣ Finding local Ollama models...")
    from chuk_llm.registry import ModelQuery

    query = ModelQuery(provider="ollama")
    ollama_models = await registry.query(query)
    if ollama_models:
        print(f"   Found {len(ollama_models)} local models:")
        for model in ollama_models[:5]:  # Show first 5
            print(f"      • {model.spec.name}")
    else:
        print("   → No Ollama models available (is Ollama running?)")

    # Query example
    print("\n" + "=" * 60)
    print("🔎 CUSTOM QUERY EXAMPLE")
    print("=" * 60)

    from chuk_llm.registry import ModelQuery

    # Find all balanced-tier models with tools and streaming
    query = ModelQuery(
        requires_tools=True,
        quality_tier="balanced",
    )

    results = await registry.query(query)
    print(f"\nFound {len(results)} balanced-tier models with tools:")
    for model in results[:10]:
        cost = model.capabilities.input_cost_per_1m
        cost_str = f"${cost:.2f}/1M" if cost else "free"
        print(f"  • {model.spec.provider:12s} {model.spec.name:40s} {cost_str}")

    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print("\nThe registry is now the dynamic capability brain of CHUK!")
    print("No more hardcoded model lists - everything is discovered and resolved.")


if __name__ == "__main__":
    asyncio.run(main())
