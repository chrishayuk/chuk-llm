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

# Load environment variables from .env file before importing sources
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars only

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
    GeminiCapabilityResolver,
    HeuristicCapabilityResolver,
    OllamaCapabilityResolver,
    YamlCapabilityResolver,
)
from chuk_llm.registry.sources import (
    AnthropicModelSource,
    AzureOpenAIModelSource,
    DeepSeekModelSource,
    EnvProviderSource,
    GeminiModelSource,
    GroqModelSource,
    MistralModelSource,
    ModelSource,
    MoonshotModelSource,
    OllamaSource,
    OpenAICompatibleSource,
    OpenAIModelSource,
    OpenRouterModelSource,
    PerplexityModelSource,
    WatsonxModelSource,
)

# Singleton registry instance
_registry_instance: ModelRegistry | None = None


async def get_registry(
    *,
    sources: list[ModelSource] | None = None,
    resolvers: list[CapabilityResolver] | None = None,
    use_provider_apis: bool = True,
    force_refresh: bool = False,
) -> ModelRegistry:
    """
    Get the global model registry instance.

    By default, creates a registry with:
    - Provider-specific sources (OpenAI, Anthropic, Gemini APIs) if use_provider_apis=True
    - OR EnvProviderSource (basic env-based discovery) if use_provider_apis=False
    - OllamaSource (discovers local Ollama models)
    - YamlCapabilityResolver (tested capabilities from YAML cache)
    - OllamaCapabilityResolver (dynamic Ollama GGUF capabilities)
    - HeuristicCapabilityResolver (fallback heuristics for quality tier & context)

    Args:
        sources: Custom model sources (overrides defaults)
        resolvers: Custom capability resolvers (overrides defaults)
        use_provider_apis: Use provider-specific API sources for richer discovery
        force_refresh: Force recreation of registry instance

    Returns:
        ModelRegistry instance
    """
    global _registry_instance

    if _registry_instance is not None and not force_refresh:
        return _registry_instance

    # Default sources
    if sources is None:
        if use_provider_apis:
            # Use provider-specific sources for dynamic discovery
            sources = [
                OpenAIModelSource(),
                AzureOpenAIModelSource(),
                AnthropicModelSource(),
                GeminiModelSource(),
                DeepSeekModelSource(),
                MistralModelSource(),
                MoonshotModelSource(),
                GroqModelSource(),
                PerplexityModelSource(),
                OpenRouterModelSource(),
                WatsonxModelSource(),
                OllamaSource(),
            ]
        else:
            # Use simple env-based discovery
            sources = [
                EnvProviderSource(include_ollama=False),
                OllamaSource(),
            ]

    # Default resolvers (order matters - later ones override earlier ones)
    # Priority: Heuristics (fallback) → API metadata (dynamic) → YAML cache (tested data)
    if resolvers is None:
        resolvers = [
            HeuristicCapabilityResolver(),  # Lowest priority: heuristics for quality/context
            GeminiCapabilityResolver(),  # Medium priority: query Gemini API for real data
            OllamaCapabilityResolver(),  # Medium priority: query Ollama API
            YamlCapabilityResolver(),  # Highest priority: tested capability cache
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
    "OpenAICompatibleSource",
    "OpenAIModelSource",
    "AzureOpenAIModelSource",
    "AnthropicModelSource",
    "GeminiModelSource",
    "DeepSeekModelSource",
    "MistralModelSource",
    "MoonshotModelSource",
    "GroqModelSource",
    "PerplexityModelSource",
    "OpenRouterModelSource",
    "WatsonxModelSource",
    "OllamaSource",
    # Resolvers
    "CapabilityResolver",
    "HeuristicCapabilityResolver",
    "OllamaCapabilityResolver",
    "YamlCapabilityResolver",
    "GeminiCapabilityResolver",
]
