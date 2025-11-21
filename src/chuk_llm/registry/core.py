"""
Core model registry implementation.

The ModelRegistry orchestrates model discovery (via sources) and
capability resolution (via resolvers) to provide a unified view of
all available models.
"""

from __future__ import annotations

import asyncio
from typing import cast

from chuk_llm.registry.cache import RegistryCache
from chuk_llm.registry.models import (
    ModelCapabilities,
    ModelQuery,
    ModelSpec,
    ModelWithCapabilities,
)
from chuk_llm.registry.resolvers import CapabilityResolver
from chuk_llm.registry.sources import ModelSource


class ModelRegistry:
    """
    Central registry for model discovery and capability resolution.

    The registry:
    1. Discovers models via ModelSource implementations
    2. Resolves capabilities via CapabilityResolver implementations (layered)
    3. Provides intelligent model selection via queries

    Example:
        ```python
        registry = ModelRegistry(
            sources=[EnvProviderSource(), OllamaSource()],
            resolvers=[YamlCapabilityResolver(), OllamaCapabilityResolver()]
        )

        models = await registry.get_models()
        best = await registry.find_best(requires_tools=True, min_context=128_000)
        ```
    """

    def __init__(
        self,
        sources: list[ModelSource],
        resolvers: list[CapabilityResolver],
        enable_persistent_cache: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize the model registry.

        Args:
            sources: Model discovery sources (e.g., EnvProviderSource, OllamaSource)
            resolvers: Capability resolvers (e.g., YamlCapabilityResolver, InferenceCapabilityResolver)
            enable_persistent_cache: Enable persistent disk caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.sources = sources
        self.resolvers = resolvers
        self._memory_cache: dict[tuple[str, str], ModelWithCapabilities] | None = None
        self._disk_cache = (
            RegistryCache(ttl_hours=cache_ttl_hours)
            if enable_persistent_cache
            else None
        )

    async def get_models(
        self, *, force_refresh: bool = False
    ) -> list[ModelWithCapabilities]:
        """
        Get all available models with capabilities.

        Caching strategy:
        1. Check memory cache (fastest)
        2. Check disk cache (~/.cache/chuk-llm/, 24h TTL)
        3. Discover + resolve (slowest)

        Args:
            force_refresh: If True, bypass all caches and re-discover

        Returns:
            List of models with capabilities
        """
        # Check memory cache
        if self._memory_cache is not None and not force_refresh:
            return list(self._memory_cache.values())

        # Discover models from all sources
        specs = await self._discover_all_sources()

        # Resolve capabilities for each model (with disk caching)
        models = []
        for spec in specs:
            # Try disk cache first
            if self._disk_cache and not force_refresh:
                cached_model = self._disk_cache.get_model(spec.provider, spec.name)
                if cached_model:
                    models.append(cached_model)
                    continue

            # Resolve from resolvers
            model = await self._resolve_capabilities(spec)
            models.append(model)

            # Save to disk cache
            if self._disk_cache:
                self._disk_cache.set_capabilities(spec, model.capabilities)

        # Update memory cache
        self._memory_cache = {(m.spec.provider, m.spec.name): m for m in models}

        return models

    async def find_model(
        self, provider: str, name: str
    ) -> ModelWithCapabilities | None:
        """
        Find a specific model by provider and name.

        Args:
            provider: Provider name
            name: Model name

        Returns:
            Model with capabilities or None if not found
        """
        models = await self.get_models()

        for model in models:
            if model.spec.provider == provider and model.spec.name == name:
                return model

        return None

    async def find_best(
        self,
        *,
        requires_tools: bool = False,
        requires_vision: bool = False,
        requires_audio: bool = False,
        requires_json_mode: bool = False,
        min_context: int | None = None,
        quality_tier: str = "any",
        provider: str | None = None,
    ) -> ModelWithCapabilities | None:
        """
        Find the best model matching criteria.

        Scoring prioritizes:
        1. Meeting all requirements
        2. Cost efficiency (lower input cost preferred)
        3. Context window (larger preferred within tier)

        Args:
            requires_tools: Must support tool calling
            requires_vision: Must support vision
            requires_audio: Must support audio
            requires_json_mode: Must support JSON mode
            min_context: Minimum context window
            quality_tier: Quality tier ("best", "balanced", "cheap", "any")
            provider: Specific provider to use

        Returns:
            Best matching model or None
        """
        query = ModelQuery(
            requires_tools=requires_tools,
            requires_vision=requires_vision,
            requires_audio=requires_audio,
            requires_json_mode=requires_json_mode,
            min_context=min_context,
            quality_tier=quality_tier,  # type: ignore
            provider=provider,
        )

        models = await self.query(query)

        if not models:
            return None

        # Score models
        scored = [(self._score_model(m), m) for m in models]
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[0][1]

    async def query(self, query: ModelQuery) -> list[ModelWithCapabilities]:
        """
        Query for models matching criteria.

        Args:
            query: Model query parameters

        Returns:
            List of matching models
        """
        models = await self.get_models()
        return [m for m in models if query.matches(m)]

    async def _discover_all_sources(self) -> list[ModelSpec]:
        """
        Discover models from all sources.

        Returns:
            Deduplicated list of model specifications
        """
        # Run all sources in parallel
        results = await asyncio.gather(
            *[source.discover() for source in self.sources],
            return_exceptions=True,
        )

        # Collect specs, handling exceptions
        all_specs: list[ModelSpec] = []
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue
                continue
            # Type narrowing: result is list[ModelSpec] here (not BaseException)
            all_specs.extend(cast(list[ModelSpec], result))

        # Deduplicate by (provider, name)
        seen = set()
        unique_specs = []
        for spec in all_specs:
            key = (spec.provider, spec.name)
            if key not in seen:
                seen.add(key)
                unique_specs.append(spec)

        return unique_specs

    async def _resolve_all_capabilities(
        self, specs: list[ModelSpec]
    ) -> list[ModelWithCapabilities]:
        """
        Resolve capabilities for all model specs.

        Runs resolvers in sequence (layering), then processes all models in parallel.

        Args:
            specs: Model specifications

        Returns:
            Models with capabilities
        """
        # Resolve capabilities for each spec in parallel
        models = await asyncio.gather(
            *[self._resolve_capabilities(spec) for spec in specs],
            return_exceptions=True,
        )

        # Filter out exceptions
        return [m for m in models if isinstance(m, ModelWithCapabilities)]

    async def _resolve_capabilities(self, spec: ModelSpec) -> ModelWithCapabilities:
        """
        Resolve capabilities for a single model spec.

        Runs all resolvers in sequence, layering capabilities.

        Args:
            spec: Model specification

        Returns:
            Model with capabilities
        """
        capabilities = ModelCapabilities()

        # Run resolvers in sequence (later ones override earlier ones)
        for resolver in self.resolvers:
            try:
                partial_caps = await resolver.get_capabilities(spec)
                capabilities = capabilities.merge(partial_caps)
            except Exception:
                # Resolver failed - continue with other resolvers
                continue

        return ModelWithCapabilities(spec=spec, capabilities=capabilities)

    def _score_model(self, model: ModelWithCapabilities) -> float:
        """
        Score a model for ranking.

        Higher scores are better.

        Args:
            model: Model to score

        Returns:
            Score (higher is better)
        """
        score = 0.0

        # Prefer larger context windows
        if model.capabilities.max_context is not None:
            score += model.capabilities.max_context / 1000.0

        # Prefer faster models (measured tokens per second)
        if model.capabilities.tokens_per_second is not None:
            score += model.capabilities.tokens_per_second / 10.0

        return score
