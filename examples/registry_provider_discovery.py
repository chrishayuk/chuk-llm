#!/usr/bin/env python3
"""
Registry Provider Discovery Demo
=================================

Demonstrates how the registry dynamically discovers models from provider APIs.

This shows the power of the registry:
1. No hardcoded model lists
2. Automatic discovery from provider APIs
3. Smart capability inference
4. Persistent caching for speed

Requirements:
    - Set API keys in environment or .env file:
      - OPENAI_API_KEY
      - ANTHROPIC_API_KEY
      - GEMINI_API_KEY or GOOGLE_API_KEY
      - Ollama running locally (optional)
"""

import asyncio

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from chuk_llm.registry import (
    AnthropicModelSource,
    GeminiModelSource,
    ModelRegistry,
    OllamaSource,
    OpenAIModelSource,
    HeuristicCapabilityResolver,
    OllamaCapabilityResolver,
)


async def demo_single_provider(provider_name: str, source, resolver=None):
    """
    Demonstrate discovery from a single provider.

    Args:
        provider_name: Provider name for display
        source: ModelSource instance
        resolver: Optional CapabilityResolver instance
    """
    print(f"\n{'=' * 70}")
    print(f"🔍 Discovering {provider_name} Models")
    print('=' * 70)

    # Create registry with single provider
    resolvers = [HeuristicCapabilityResolver()]
    if resolver:
        resolvers.append(resolver)

    registry = ModelRegistry(sources=[source], resolvers=resolvers)

    # Discover models
    models = await registry.get_models()

    print(f"\nFound {len(models)} {provider_name} models:\n")

    # Show first 10 models with details
    for i, model in enumerate(models[:10], 1):
        spec = model.spec
        caps = model.capabilities

        # Format capabilities
        features = []
        if caps.supports_tools:
            features.append("tools")
        if caps.supports_vision:
            features.append("vision")
        if caps.supports_streaming:
            features.append("streaming")
        if caps.supports_json_mode:
            features.append("json")

        feature_str = ", ".join(features) if features else "none"
        context_str = f"{caps.max_context//1000}k" if caps.max_context else "?"
        tier_str = caps.quality_tier.value if caps.quality_tier else "unknown"

        print(f"{i:2}. {spec.name:<45} | {tier_str:8} | {context_str:>6} | {feature_str}")

    if len(models) > 10:
        print(f"\n... and {len(models) - 10} more models")

    return models


async def demo_all_providers():
    """Demonstrate discovery from all available providers."""
    print("\n" + "=" * 70)
    print("🚀 CHUK-LLM PROVIDER DISCOVERY DEMO")
    print("=" * 70)

    all_models = []

    # OpenAI Discovery
    try:
        openai_models = await demo_single_provider(
            "OpenAI",
            OpenAIModelSource()
        )
        all_models.extend(openai_models)
    except Exception as e:
        print(f"\n⚠️  OpenAI discovery failed: {e}")
        print("   (Set OPENAI_API_KEY to enable)")

    # Anthropic Discovery
    try:
        anthropic_models = await demo_single_provider(
            "Anthropic",
            AnthropicModelSource()
        )
        all_models.extend(anthropic_models)
    except Exception as e:
        print(f"\n⚠️  Anthropic discovery failed: {e}")
        print("   (Set ANTHROPIC_API_KEY to enable)")

    # Gemini Discovery
    try:
        gemini_models = await demo_single_provider(
            "Gemini",
            GeminiModelSource()
        )
        all_models.extend(gemini_models)
    except Exception as e:
        print(f"\n⚠️  Gemini discovery failed: {e}")
        print("   (Set GEMINI_API_KEY or GOOGLE_API_KEY to enable)")

    # Ollama Discovery
    try:
        ollama_models = await demo_single_provider(
            "Ollama",
            OllamaSource(),
            OllamaCapabilityResolver()
        )
        all_models.extend(ollama_models)
    except Exception as e:
        print(f"\n⚠️  Ollama discovery failed: {e}")
        print("   (Ensure Ollama is running locally)")

    # Summary
    print("\n" + "=" * 70)
    print("📊 DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"\nTotal models discovered: {len(all_models)}")

    # Count by provider
    from collections import Counter
    provider_counts = Counter(m.spec.provider for m in all_models)

    print("\nModels by provider:")
    for provider, count in sorted(provider_counts.items()):
        print(f"  • {provider:<15} {count:>3} models")

    # Count by quality tier
    tier_counts = Counter(m.capabilities.quality_tier.value for m in all_models)
    print("\nModels by quality tier:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  • {tier:<15} {count:>3} models")

    # Count capabilities
    tools_count = sum(1 for m in all_models if m.capabilities.supports_tools)
    vision_count = sum(1 for m in all_models if m.capabilities.supports_vision)

    print(f"\nCapabilities:")
    print(f"  • Tool calling:   {tools_count} models")
    print(f"  • Vision:         {vision_count} models")


async def demo_intelligent_selection():
    """Demonstrate intelligent model selection based on requirements."""
    print("\n" + "=" * 70)
    print("🎯 INTELLIGENT MODEL SELECTION")
    print("=" * 70)

    # Create full registry with all providers
    from chuk_llm.registry import get_registry

    registry = await get_registry(use_provider_apis=True)

    # Test different selection criteria
    scenarios = [
        {
            "name": "Cheap model with tool support",
            "query": {"requires_tools": True, "quality_tier": "cheap"},
        },
        {
            "name": "Best model with vision",
            "query": {"requires_vision": True, "quality_tier": "best"},
        },
        {
            "name": "Balanced model with 128k+ context",
            "query": {"quality_tier": "balanced", "min_context": 128_000},
        },
        {
            "name": "Any model with tools and JSON mode",
            "query": {"requires_tools": True, "requires_json_mode": True},
        },
    ]

    for scenario in scenarios:
        print(f"\n🔍 {scenario['name']}:")

        model = await registry.find_best(**scenario["query"])

        if model:
            print(f"   ✓ Found: {model.spec.provider}:{model.spec.name}")
            print(f"     Tier: {model.capabilities.quality_tier.value}")
            if model.capabilities.max_context:
                print(f"     Context: {model.capabilities.max_context:,} tokens")
            if model.capabilities.tokens_per_second:
                print(f"     Speed: {model.capabilities.tokens_per_second:.1f} tokens/sec")
        else:
            print("   ✗ No matching model found")


async def demo_caching():
    """Demonstrate the caching system."""
    print("\n" + "=" * 70)
    print("⚡ CACHING DEMONSTRATION")
    print("=" * 70)

    from chuk_llm.registry import get_registry
    import time

    print("\n1️⃣ First call (cold - will discover from APIs)...")
    start = time.time()
    registry1 = await get_registry(force_refresh=True, use_provider_apis=True)
    models1 = await registry1.get_models()
    cold_time = time.time() - start
    print(f"   ⏱  Discovered {len(models1)} models in {cold_time:.2f}s")

    print("\n2️⃣ Second call (warm - from memory cache)...")
    start = time.time()
    models2 = await registry1.get_models()
    warm_time = time.time() - start
    print(f"   ⚡ Retrieved {len(models2)} models in {warm_time:.3f}s")

    print("\n3️⃣ New registry instance (from disk cache)...")
    start = time.time()
    registry2 = await get_registry(use_provider_apis=True)
    models3 = await registry2.get_models()
    disk_time = time.time() - start
    print(f"   💾 Retrieved {len(models3)} models in {disk_time:.3f}s")

    # Show speedup
    print(f"\n📈 Performance improvement:")
    print(f"   • Memory cache: {cold_time/warm_time:.0f}x faster")
    print(f"   • Disk cache:   {cold_time/disk_time:.1f}x faster")

    # Show cache stats
    if registry2._disk_cache:
        stats = registry2._disk_cache.get_stats()
        print(f"\n💾 Cache statistics:")
        print(f"   • Total entries:  {stats['total_entries']}")
        print(f"   • Valid entries:  {stats['valid_entries']}")
        print(f"   • Cache file:     {stats['cache_file']}")
        print(f"   • Cache size:     {stats['cache_size_bytes']} bytes")


async def demo_custom_configuration():
    """Demonstrate custom registry configuration."""
    print("\n" + "=" * 70)
    print("🔧 CUSTOM CONFIGURATION")
    print("=" * 70)

    print("\n1️⃣ Registry with only OpenAI models:")
    registry = ModelRegistry(
        sources=[OpenAIModelSource()],
        resolvers=[HeuristicCapabilityResolver()],
    )
    models = await registry.get_models()
    print(f"   Found {len(models)} OpenAI models")

    print("\n2️⃣ Registry with custom cache TTL (1 hour):")
    registry = ModelRegistry(
        sources=[OpenAIModelSource()],
        resolvers=[HeuristicCapabilityResolver()],
        cache_ttl_hours=1,
    )
    print("   ✓ Cache will expire after 1 hour")

    print("\n3️⃣ Registry without persistent cache:")
    registry = ModelRegistry(
        sources=[OpenAIModelSource()],
        resolvers=[HeuristicCapabilityResolver()],
        enable_persistent_cache=False,
    )
    print("   ✓ Only using in-memory cache")


async def main():
    """Run all demonstrations."""
    print("\n" + "🎨" * 35)
    print("\n   CHUK-LLM PROVIDER DISCOVERY DEMO")
    print("   Showcasing Dynamic Model Discovery\n")
    print("🎨" * 35)

    # Run demonstrations
    await demo_all_providers()
    await demo_intelligent_selection()
    await demo_caching()
    await demo_custom_configuration()

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Models are discovered dynamically from provider APIs")
    print("  2. No hardcoded model lists - always up to date")
    print("  3. Smart capability inference for unknown models")
    print("  4. Persistent caching for fast subsequent access")
    print("  5. Flexible configuration for different use cases")
    print("\n🚀 The registry is the capability brain of CHUK!\n")


if __name__ == "__main__":
    asyncio.run(main())
