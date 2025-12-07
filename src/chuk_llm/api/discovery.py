# chuk_llm/api/discovery.py
"""
Model discovery API - Registry-based implementation.

Provides backward-compatible API using the new registry system.
"""

import asyncio
import logging
from typing import Any

from chuk_llm.core.constants import CapabilityKey
from chuk_llm.core.enums import Provider
from chuk_llm.registry import QualityTier, get_registry

log = logging.getLogger(__name__)


async def discover_models(
    provider_name: str,
    force_refresh: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Discover models from a provider using the registry system.

    Args:
        provider_name: Name of the provider (openai, anthropic, gemini, ollama)
        force_refresh: Force refresh of model cache
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        List of discovered model dictionaries

    Example:
        >>> models = await discover_models("openai")
        >>> print(f"Found {len(models)} models")
    """
    # Get registry instance
    registry = await get_registry(use_provider_apis=True, force_refresh=force_refresh)

    # Get all models
    all_models = await registry.get_models()

    # Filter by provider
    provider_models = [m for m in all_models if m.spec.provider == provider_name]

    if not provider_models:
        log.warning(f"No models found for provider: {provider_name}")
        return []

    # Convert to API format (backward compatible)
    return [
        {
            "name": model.spec.name,
            "provider": model.spec.provider,
            "family": model.spec.family,
            "context_length": model.capabilities.max_context,
            "max_output_tokens": model.capabilities.max_output_tokens,
            "features": _capabilities_to_features(model.capabilities),
            CapabilityKey.SUPPORTS_TOOLS.value: model.capabilities.supports_tools
            or False,
            CapabilityKey.SUPPORTS_VISION.value: model.capabilities.supports_vision
            or False,
            CapabilityKey.SUPPORTS_JSON_MODE.value: model.capabilities.supports_json_mode
            or False,
            CapabilityKey.SUPPORTS_STREAMING.value: model.capabilities.supports_streaming
            or False,
            "quality_tier": (
                model.capabilities.quality_tier.value
                if model.capabilities.quality_tier
                else "unknown"
            ),
            "tokens_per_second": model.capabilities.tokens_per_second,
        }
        for model in provider_models
    ]


def _capabilities_to_features(capabilities) -> list[str]:
    """Convert capabilities to feature list (backward compatibility)"""
    features = ["text"]  # All models support text

    if capabilities.supports_tools:
        features.append("tools")
    if capabilities.supports_vision:
        features.append("vision")
    if capabilities.supports_json_mode:
        features.append("json")
    if capabilities.supports_streaming:
        features.append("streaming")

    return features


def discover_models_sync(provider_name: str, **kwargs) -> list[dict[str, Any]]:
    """Synchronous version of discover_models"""
    return asyncio.run(discover_models(provider_name, **kwargs))


async def get_model_info(
    provider_name: str,
    model_name: str,
    **kwargs,
) -> dict[str, Any] | None:
    """
    Get detailed information about a specific model.

    Args:
        provider_name: Name of the provider
        model_name: Name of the model
        **kwargs: Additional arguments (ignored)

    Returns:
        Model information dictionary or None if not found

    Example:
        >>> info = await get_model_info("openai", "gpt-4o")
        >>> print(info["context_length"])
    """
    registry = await get_registry(use_provider_apis=True)
    all_models = await registry.get_models()

    # Find the model
    for model in all_models:
        if model.spec.provider == provider_name and model.spec.name == model_name:
            return {
                "name": model.spec.name,
                "provider": model.spec.provider,
                "family": model.spec.family,
                "context_length": model.capabilities.max_context,
                "max_output_tokens": model.capabilities.max_output_tokens,
                "features": _capabilities_to_features(model.capabilities),
                CapabilityKey.SUPPORTS_TOOLS.value: model.capabilities.supports_tools
                or False,
                CapabilityKey.SUPPORTS_VISION.value: model.capabilities.supports_vision
                or False,
                CapabilityKey.SUPPORTS_JSON_MODE.value: model.capabilities.supports_json_mode
                or False,
                CapabilityKey.SUPPORTS_STREAMING.value: model.capabilities.supports_streaming
                or False,
                "quality_tier": (
                    model.capabilities.quality_tier.value
                    if model.capabilities.quality_tier
                    else "unknown"
                ),
                "tokens_per_second": model.capabilities.tokens_per_second,
                "known_params": (
                    list(model.capabilities.known_params)
                    if model.capabilities.known_params
                    else []
                ),
            }

    return None


def get_model_info_sync(
    provider_name: str, model_name: str, **kwargs
) -> dict[str, Any] | None:
    """Synchronous version of get_model_info"""
    return asyncio.run(get_model_info(provider_name, model_name, **kwargs))


async def find_best_model(
    provider: str | None = None,
    requires_tools: bool = False,
    requires_vision: bool = False,
    requires_json_mode: bool = False,
    min_context: int | None = None,
    quality_tier: str = "any",
    **kwargs,
) -> dict[str, Any] | None:
    """
    Find the best model matching requirements.

    Args:
        provider: Filter to specific provider (optional)
        requires_tools: Model must support function calling
        requires_vision: Model must support vision/images
        requires_json_mode: Model must support JSON mode
        min_context: Minimum context length required
        quality_tier: Quality tier (best, balanced, cheap, any)
        **kwargs: Additional arguments (ignored)

    Returns:
        Best matching model info or None

    Example:
        >>> model = await find_best_model(requires_tools=True, quality_tier="balanced")
        >>> print(f"Use: {model['provider']}:{model['name']}")
    """
    registry = await get_registry(use_provider_apis=True)

    # Convert quality tier string to enum
    tier = QualityTier(quality_tier) if quality_tier != "any" else "any"

    # Find best match
    best_model = await registry.find_best(
        provider=provider,
        requires_tools=requires_tools,
        requires_vision=requires_vision,
        requires_json_mode=requires_json_mode,
        min_context=min_context,
        quality_tier=tier,
    )

    if not best_model:
        return None

    return {
        "name": best_model.spec.name,
        "provider": best_model.spec.provider,
        "family": best_model.spec.family,
        "context_length": best_model.capabilities.max_context,
        "max_output_tokens": best_model.capabilities.max_output_tokens,
        "features": _capabilities_to_features(best_model.capabilities),
        "quality_tier": (
            best_model.capabilities.quality_tier.value
            if best_model.capabilities.quality_tier
            else "unknown"
        ),
    }


def find_best_model_sync(**kwargs) -> dict[str, Any] | None:
    """Synchronous version of find_best_model"""
    return asyncio.run(find_best_model(**kwargs))


async def list_providers() -> list[str]:
    """
    List all available providers with discovered models.

    Returns:
        List of provider names

    Example:
        >>> providers = await list_providers()
        >>> print(f"Available: {', '.join(providers)}")
    """
    registry = await get_registry(use_provider_apis=True)
    all_models = await registry.get_models()

    # Get unique providers
    providers = sorted({m.spec.provider for m in all_models})
    return providers


def list_providers_sync() -> list[str]:
    """Synchronous version of list_providers"""
    return asyncio.run(list_providers())


async def show_discovered_models(
    provider_name: str,
    force_refresh: bool = False,
    **kwargs,
) -> None:
    """
    Display discovered models in a nice format.

    Args:
        provider_name: Name of the provider
        force_refresh: Force refresh of model cache
        **kwargs: Additional arguments (ignored)

    Example:
        >>> await show_discovered_models("openai")
    """
    models = await discover_models(provider_name, force_refresh=force_refresh)

    if not models:
        print(f"\n❌ No models found for {provider_name}")
        return

    print(f"\n🔍 Discovered {len(models)} {provider_name.title()} Models")
    print("=" * 70)

    # Group by family
    families: dict[str, list] = {}
    for model in models:
        family = model["family"] or "unknown"
        if family not in families:
            families[family] = []
        families[family].append(model)

    for family, family_models in sorted(families.items()):
        print(f"\n📁 {family.title()} ({len(family_models)} models):")

        for model in sorted(family_models, key=lambda x: x["name"]):
            ctx = (
                f"{model['context_length']:,}" if model["context_length"] else "Unknown"
            )
            tier = model["quality_tier"]
            features = ", ".join(model["features"])

            print(f"  • {model['name']}")
            print(f"    Context: {ctx} | Tier: {tier}")
            print(f"    Features: {features}")

            if model["tokens_per_second"]:
                print(f"    Speed: {model['tokens_per_second']:.1f} tokens/sec")
            print()


def show_discovered_models_sync(provider_name: str, **kwargs) -> None:
    """Synchronous version of show_discovered_models"""
    asyncio.run(show_discovered_models(provider_name, **kwargs))


def list_supported_providers() -> list[str]:
    """
    List providers that support discovery.

    Returns:
        List of supported provider names
    """
    return [
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value,
        Provider.GEMINI.value,
        Provider.OLLAMA.value,
    ]


# Export public API
__all__ = [
    "discover_models",
    "discover_models_sync",
    "get_model_info",
    "get_model_info_sync",
    "find_best_model",
    "find_best_model_sync",
    "list_providers",
    "list_providers_sync",
    "show_discovered_models",
    "show_discovered_models_sync",
    "list_supported_providers",
]
