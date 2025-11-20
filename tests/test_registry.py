"""
Tests for the model registry system.
"""

import pytest

from chuk_llm.registry import (
    ModelQuery,
    ModelRegistry,
    ModelSpec,
    QualityTier,
    get_registry,
)
from chuk_llm.registry.resolvers import StaticCapabilityResolver
from chuk_llm.registry.sources import EnvProviderSource


class TestModelSpec:
    """Test ModelSpec model."""

    def test_create_spec(self):
        """Test creating a model spec."""
        spec = ModelSpec(
            provider="openai",
            name="gpt-4o-mini",
            family="gpt-4o",
        )

        assert spec.provider == "openai"
        assert spec.name == "gpt-4o-mini"
        assert spec.family == "gpt-4o"

    def test_spec_is_hashable(self):
        """Test that specs can be used in sets."""
        spec1 = ModelSpec(provider="openai", name="gpt-4o-mini")
        spec2 = ModelSpec(provider="openai", name="gpt-4o-mini")
        spec3 = ModelSpec(provider="openai", name="gpt-4o")

        # Same specs should have same hash
        assert hash(spec1) == hash(spec2)

        # Different specs should (usually) have different hashes
        assert hash(spec1) != hash(spec3)

        # Can be added to set
        specs = {spec1, spec2, spec3}
        assert len(specs) == 2  # spec1 and spec2 are the same


class TestStaticResolver:
    """Test static capability resolver."""

    @pytest.mark.asyncio
    async def test_resolve_known_model(self):
        """Test resolving a known model."""
        resolver = StaticCapabilityResolver()
        spec = ModelSpec(provider="openai", name="gpt-4o-mini")

        caps = await resolver.get_capabilities(spec)

        assert caps.max_context == 128_000
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.quality_tier == QualityTier.BALANCED

    @pytest.mark.asyncio
    async def test_resolve_unknown_model(self):
        """Test resolving an unknown model."""
        resolver = StaticCapabilityResolver()
        spec = ModelSpec(provider="unknown", name="unknown-model")

        caps = await resolver.get_capabilities(spec)

        assert caps.max_context is None
        assert caps.supports_tools is None

    @pytest.mark.asyncio
    async def test_prefix_matching(self):
        """Test that versioned models match base models."""
        resolver = StaticCapabilityResolver()
        spec = ModelSpec(provider="openai", name="gpt-4o-2024-08-06")

        caps = await resolver.get_capabilities(spec)

        # Should match "gpt-4o" base model
        assert caps.max_context == 128_000
        assert caps.supports_tools is True


class TestEnvSource:
    """Test environment-based model source."""

    @pytest.mark.asyncio
    async def test_discover_models(self):
        """Test discovering models from environment."""
        source = EnvProviderSource(include_ollama=False)
        specs = await source.discover()

        # Should return list of ModelSpec objects
        assert isinstance(specs, list)
        for spec in specs:
            assert isinstance(spec, ModelSpec)


class TestModelQuery:
    """Test model query matching."""

    def test_query_matches(self):
        """Test query matching logic."""
        from chuk_llm.registry.models import ModelCapabilities, ModelWithCapabilities

        query = ModelQuery(
            requires_tools=True,
            min_context=100_000,
        )

        # Model that matches
        good_model = ModelWithCapabilities(
            spec=ModelSpec(provider="openai", name="gpt-4o"),
            capabilities=ModelCapabilities(
                max_context=128_000,
                supports_tools=True,
            ),
        )

        # Model that doesn't match (no tools)
        bad_model1 = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="no-tools"),
            capabilities=ModelCapabilities(
                max_context=128_000,
                supports_tools=False,
            ),
        )

        # Model that doesn't match (small context)
        bad_model2 = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="small-context"),
            capabilities=ModelCapabilities(
                max_context=50_000,
                supports_tools=True,
            ),
        )

        assert query.matches(good_model) is True
        assert query.matches(bad_model1) is False
        assert query.matches(bad_model2) is False


class TestModelRegistry:
    """Test model registry."""

    @pytest.mark.asyncio
    async def test_get_registry(self):
        """Test getting the global registry."""
        registry = await get_registry()

        assert isinstance(registry, ModelRegistry)

    @pytest.mark.asyncio
    async def test_registry_discover_models(self):
        """Test registry model discovery."""
        registry = await get_registry(force_refresh=True)
        models = await registry.get_models()

        # Should find at least some models
        assert len(models) > 0

        # All should have specs and capabilities
        for model in models:
            assert model.spec is not None
            assert model.capabilities is not None

    @pytest.mark.asyncio
    async def test_find_model(self):
        """Test finding a specific model."""
        registry = await get_registry()

        # Try to find a common model
        model = await registry.find_model("openai", "gpt-4o-mini")

        if model:  # Only if OpenAI is available
            assert model.spec.provider == "openai"
            assert model.spec.name == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_find_best(self):
        """Test finding best model matching criteria."""
        registry = await get_registry()

        # Find best model with vision
        best = await registry.find_best(requires_vision=True)

        if best:  # If any vision models available
            assert best.capabilities.supports_vision is True

    @pytest.mark.asyncio
    async def test_query_models(self):
        """Test querying models."""
        registry = await get_registry()

        query = ModelQuery(requires_tools=True)
        models = await registry.query(query)

        # All results should support tools
        for model in models:
            assert model.capabilities.supports_tools is True
