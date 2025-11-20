"""
Dynamic capability registry for chuk-llm.

The registry system provides intelligent model discovery and capability resolution
across all providers without requiring constant library updates.

Example:
    ```python
    from chuk_llm.registry import get_registry

    # Get the default registry
    registry = await get_registry()

    # Find all available models
    models = await registry.get_models()

    # Find the best model for a task
    best = await registry.find_best(
        requires_tools=True,
        min_context=128_000,
        quality_tier="balanced"
    )
    ```
"""

from chuk_llm.registry.core import ModelRegistry
from chuk_llm.registry.models import (
    ModelCapabilities,
    ModelQuery,
    ModelSpec,
    ModelWithCapabilities,
    QualityTier,
)
from chuk_llm.registry.resolvers import (
    CapabilityResolver,
    OllamaCapabilityResolver,
    StaticCapabilityResolver,
)
from chuk_llm.registry.sources import (
    EnvProviderSource,
    ModelSource,
    OllamaSource,
)

# Singleton registry instance
_registry_instance: ModelRegistry | None = None


async def get_registry(
    *,
    sources: list[ModelSource] | None = None,
    resolvers: list[CapabilityResolver] | None = None,
    force_refresh: bool = False,
) -> ModelRegistry:
    """
    Get the global model registry instance.

    By default, creates a registry with:
    - EnvProviderSource (discovers providers via API keys)
    - OllamaSource (discovers local Ollama models)
    - StaticCapabilityResolver (baseline capabilities for major models)
    - OllamaCapabilityResolver (dynamic Ollama capabilities)

    Args:
        sources: Custom model sources (overrides defaults)
        resolvers: Custom capability resolvers (overrides defaults)
        force_refresh: Force recreation of registry instance

    Returns:
        ModelRegistry instance
    """
    global _registry_instance

    if _registry_instance is not None and not force_refresh:
        return _registry_instance

    # Default sources
    if sources is None:
        sources = [
            EnvProviderSource(include_ollama=False),  # Ollama handled separately
            OllamaSource(),
        ]

    # Default resolvers (order matters - later ones override earlier ones)
    if resolvers is None:
        resolvers = [
            StaticCapabilityResolver(),
            OllamaCapabilityResolver(),
        ]

    _registry_instance = ModelRegistry(sources=sources, resolvers=resolvers)
    return _registry_instance


__all__ = [
    # Core registry
    "ModelRegistry",
    "get_registry",
    # Models
    "ModelSpec",
    "ModelCapabilities",
    "ModelWithCapabilities",
    "ModelQuery",
    "QualityTier",
    # Sources
    "ModelSource",
    "EnvProviderSource",
    "OllamaSource",
    # Resolvers
    "CapabilityResolver",
    "StaticCapabilityResolver",
    "OllamaCapabilityResolver",
]
