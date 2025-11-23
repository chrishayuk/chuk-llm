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
# StaticCapabilityResolver removed - using YAML cache instead
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


# TestStaticResolver removed - StaticCapabilityResolver has been replaced by YAML cache


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
    async def test_registry_with_disk_cache(self, tmp_path):
        """Test registry with disk cache enabled."""
        from chuk_llm.registry.cache import RegistryCache
        from chuk_llm.registry.core import ModelRegistry
        from chuk_llm.registry.models import ModelCapabilities
        from chuk_llm.registry.resolvers.base import BaseCapabilityResolver
        from chuk_llm.registry.sources.base import BaseModelSource

        class SimpleSource(BaseModelSource):
            """Source that returns test models."""
            async def discover(self) -> list[ModelSpec]:
                return [
                    ModelSpec(provider="test", name="model-1"),
                    ModelSpec(provider="test", name="model-2"),
                ]

        class SimpleResolver(BaseCapabilityResolver):
            """Resolver that returns test capabilities."""
            async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
                return ModelCapabilities(max_context=100_000)

        # Create custom cache with temp directory
        # Note: ModelRegistry constructor only takes enable_persistent_cache, not a cache object
        # But we can access the internal _disk_cache to verify it was populated
        registry = ModelRegistry(
            sources=[SimpleSource()],
            resolvers=[SimpleResolver()],
            enable_persistent_cache=True,
            cache_ttl_hours=24
        )

        # Force fresh discovery (which will populate the cache)
        models = await registry.get_models(force_refresh=True)

        # Models should be discovered
        assert len(models) == 2

        # Verify disk cache was created and used
        assert registry._disk_cache is not None

        # Check that cache has entries
        cached_model1 = registry._disk_cache.get_model("test", "model-1")
        assert cached_model1 is not None
        assert cached_model1.capabilities.max_context == 100_000

    @pytest.mark.asyncio
    async def test_registry_handles_source_exceptions(self):
        """Test that registry handles exceptions from sources."""
        from chuk_llm.registry.core import ModelRegistry
        from chuk_llm.registry.sources.base import BaseModelSource

        class FailingSource(BaseModelSource):
            """Source that always raises an exception."""
            async def discover(self) -> list[ModelSpec]:
                raise ValueError("Simulated failure")

        class WorkingSource(BaseModelSource):
            """Source that returns a model."""
            async def discover(self) -> list[ModelSpec]:
                return [ModelSpec(provider="test", name="working-model")]

        registry = ModelRegistry(
            sources=[FailingSource(), WorkingSource()],
            resolvers=[]
        )

        models = await registry.get_models()

        # Should get model from working source, despite failing source
        assert len(models) == 1
        assert models[0].spec.name == "working-model"

    @pytest.mark.asyncio
    async def test_registry_handles_resolver_exceptions(self):
        """Test that registry handles exceptions from resolvers."""
        from chuk_llm.registry.core import ModelRegistry
        from chuk_llm.registry.models import ModelCapabilities
        from chuk_llm.registry.resolvers.base import BaseCapabilityResolver
        from chuk_llm.registry.sources.base import BaseModelSource

        class SimpleSource(BaseModelSource):
            """Source that returns a test model."""
            async def discover(self) -> list[ModelSpec]:
                return [ModelSpec(provider="test", name="test-model")]

        class FailingResolver(BaseCapabilityResolver):
            """Resolver that always raises an exception."""
            async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
                raise RuntimeError("Simulated resolver failure")

        class WorkingResolver(BaseCapabilityResolver):
            """Resolver that returns capabilities."""
            async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
                return ModelCapabilities(max_context=100_000)

        registry = ModelRegistry(
            sources=[SimpleSource()],
            resolvers=[FailingResolver(), WorkingResolver()]
        )

        models = await registry.get_models()

        # Should get model with capabilities from working resolver
        assert len(models) == 1
        assert models[0].capabilities.max_context == 100_000

    @pytest.mark.asyncio
    async def test_all_resolvers_fail(self):
        """Test when all resolvers fail for a model."""
        from chuk_llm.registry.core import ModelRegistry
        from chuk_llm.registry.models import ModelCapabilities
        from chuk_llm.registry.resolvers.base import BaseCapabilityResolver
        from chuk_llm.registry.sources.base import BaseModelSource

        class SimpleSource(BaseModelSource):
            """Source that returns test models."""
            async def discover(self) -> list[ModelSpec]:
                return [ModelSpec(provider="test", name="test-model")]

        class FailingResolver1(BaseCapabilityResolver):
            """First failing resolver."""
            async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
                raise RuntimeError("Resolver 1 failed")

        class FailingResolver2(BaseCapabilityResolver):
            """Second failing resolver."""
            async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
                raise ValueError("Resolver 2 failed")

        registry = ModelRegistry(
            sources=[SimpleSource()],
            resolvers=[FailingResolver1(), FailingResolver2()]
        )

        models = await registry.get_models()

        # Should still get model, but with empty capabilities
        # (all resolvers failed, so capabilities remain default)
        assert len(models) == 1
        assert models[0].spec.name == "test-model"

    @pytest.mark.asyncio
    async def test_registry_with_no_resolvers(self):
        """Test registry with no capability resolvers."""
        from chuk_llm.registry.core import ModelRegistry
        from chuk_llm.registry.models import ModelCapabilities
        from chuk_llm.registry.sources.base import BaseModelSource

        class SimpleSource(BaseModelSource):
            """Source that returns test models."""
            async def discover(self) -> list[ModelSpec]:
                return [
                    ModelSpec(provider="test", name="model-1"),
                    ModelSpec(provider="test", name="model-2"),
                ]

        registry = ModelRegistry(
            sources=[SimpleSource()],
            resolvers=[]  # No resolvers
        )

        models = await registry.get_models()

        # Should get models with default (empty) capabilities since no resolvers
        assert len(models) == 2
        # With no resolvers, capabilities should all be None/default
        for model in models:
            assert isinstance(model.capabilities, ModelCapabilities)

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

    @pytest.mark.asyncio
    async def test_get_registry_without_provider_apis(self):
        """Test getting registry with use_provider_apis=False."""
        registry = await get_registry(use_provider_apis=False, force_refresh=True)
        assert isinstance(registry, ModelRegistry)
        # Should use EnvProviderSource instead of individual provider sources
        assert len(registry.sources) >= 2  # EnvProviderSource + OllamaSource

    @pytest.mark.asyncio
    async def test_get_registry_singleton(self):
        """Test that get_registry returns same instance."""
        registry1 = await get_registry()
        registry2 = await get_registry()  # Should return same instance
        assert registry1 is registry2

    @pytest.mark.asyncio
    async def test_find_model_nonexistent(self):
        """Test finding a model that doesn't exist."""
        registry = await get_registry()

        model = await registry.find_model("nonexistent", "fake-model")

        assert model is None

    @pytest.mark.asyncio
    async def test_find_best_no_matches(self):
        """Test find_best when no models match criteria."""
        registry = await get_registry()

        # Query for impossible combination
        best = await registry.find_best(
            requires_vision=True,
            requires_tools=True,
            min_context=10_000_000,  # Impossibly large
        )

        # Should return None when no models match
        assert best is None


    @pytest.mark.asyncio
    async def test_query_with_provider_filter(self):
        """Test querying with provider filter."""
        registry = await get_registry()

        query = ModelQuery(provider="openai")
        models = await registry.query(query)

        # All results should be from OpenAI (if any OpenAI models available)
        for model in models:
            assert model.spec.provider == "openai"

    @pytest.mark.asyncio
    async def test_query_multiple_criteria(self):
        """Test querying with multiple criteria."""
        registry = await get_registry()

        query = ModelQuery(
            requires_tools=True,
            requires_vision=False,
            quality_tier=QualityTier.BEST,
        )
        models = await registry.query(query)

        # All results should match all criteria
        for model in models:
            assert model.capabilities.supports_tools is True
            if model.capabilities.quality_tier:
                assert model.capabilities.quality_tier == QualityTier.BEST


class TestModelCapabilities:
    """Test ModelCapabilities model."""

    def test_merge_capabilities(self):
        """Test merging two capability objects."""
        from chuk_llm.registry.models import ModelCapabilities

        base = ModelCapabilities(max_context=100_000, supports_tools=False)
        override = ModelCapabilities(supports_tools=True, supports_vision=True)

        merged = base.merge(override)

        # Override values should win
        assert merged.supports_tools is True
        assert merged.supports_vision is True
        # Base values should be preserved if not overridden
        assert merged.max_context == 100_000

    def test_merge_with_none_values(self):
        """Test merging handles None values correctly."""
        from chuk_llm.registry.models import ModelCapabilities

        base = ModelCapabilities(max_context=100_000, supports_tools=True)
        override = ModelCapabilities(max_context=None, supports_vision=True)

        merged = base.merge(override)

        # Base value should be kept when override is None
        assert merged.max_context == 100_000
        assert merged.supports_tools is True
        assert merged.supports_vision is True




class TestModelWithCapabilities:
    """Test ModelWithCapabilities model."""

    def test_create_model_with_capabilities(self):
        """Test creating a model with capabilities."""
        from chuk_llm.registry.models import ModelCapabilities, ModelWithCapabilities

        spec = ModelSpec(provider="openai", name="gpt-4o")
        caps = ModelCapabilities(max_context=128_000)

        model = ModelWithCapabilities(spec=spec, capabilities=caps)

        assert model.spec == spec
        assert model.capabilities == caps

    def test_model_with_capabilities_str(self):
        """Test string representation of ModelWithCapabilities."""
        from chuk_llm.registry.models import ModelCapabilities, ModelWithCapabilities

        # Model with no capabilities
        spec1 = ModelSpec(provider="openai", name="basic-model")
        caps1 = ModelCapabilities()
        model1 = ModelWithCapabilities(spec=spec1, capabilities=caps1)
        assert "basic" in str(model1)

        # Model with tools
        caps2 = ModelCapabilities(supports_tools=True)
        model2 = ModelWithCapabilities(spec=spec1, capabilities=caps2)
        assert "tools" in str(model2)

        # Model with vision
        caps3 = ModelCapabilities(supports_vision=True)
        model3 = ModelWithCapabilities(spec=spec1, capabilities=caps3)
        assert "vision" in str(model3)

        # Model with context
        caps4 = ModelCapabilities(max_context=128_000)
        model4 = ModelWithCapabilities(spec=spec1, capabilities=caps4)
        assert "128k ctx" in str(model4)

        # Model with multiple capabilities
        caps5 = ModelCapabilities(
            supports_tools=True,
            supports_vision=True,
            max_context=200_000
        )
        model5 = ModelWithCapabilities(spec=spec1, capabilities=caps5)
        result = str(model5)
        assert "tools" in result
        assert "vision" in result
        assert "200k ctx" in result

    def test_model_spec_str(self):
        """Test string representation of ModelSpec."""
        spec = ModelSpec(provider="openai", name="gpt-4o")
        assert str(spec) == "openai:gpt-4o"

    def test_query_matches_audio_and_json(self):
        """Test query matching for audio and json mode."""
        from chuk_llm.registry.models import ModelCapabilities, ModelWithCapabilities

        # Test audio requirement
        query_audio = ModelQuery(requires_audio=True)
        model_no_audio = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="no-audio"),
            capabilities=ModelCapabilities(supports_audio=False)
        )
        model_with_audio = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="with-audio"),
            capabilities=ModelCapabilities(supports_audio=True)
        )
        assert query_audio.matches(model_no_audio) is False
        assert query_audio.matches(model_with_audio) is True

        # Test JSON mode requirement
        query_json = ModelQuery(requires_json_mode=True)
        model_no_json = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="no-json"),
            capabilities=ModelCapabilities(supports_json_mode=False)
        )
        model_with_json = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="with-json"),
            capabilities=ModelCapabilities(supports_json_mode=True)
        )
        assert query_json.matches(model_no_json) is False
        assert query_json.matches(model_with_json) is True

    def test_query_matches_max_context(self):
        """Test query matching with max_context constraint."""
        from chuk_llm.registry.models import ModelCapabilities, ModelWithCapabilities

        query = ModelQuery(max_context=100_000)

        # Model with context within limit
        model_ok = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="ok"),
            capabilities=ModelCapabilities(max_context=50_000)
        )
        assert query.matches(model_ok) is True

        # Model exceeding max context
        model_too_large = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="too-large"),
            capabilities=ModelCapabilities(max_context=150_000)
        )
        assert query.matches(model_too_large) is False

    def test_query_matches_speed(self):
        """Test query matching with speed constraints."""
        from chuk_llm.registry.models import ModelCapabilities, ModelWithCapabilities

        query = ModelQuery(min_speed_tps=50.0)

        # Model too slow
        model_slow = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="slow"),
            capabilities=ModelCapabilities(tokens_per_second=30.0)
        )
        assert query.matches(model_slow) is False

        # Model fast enough
        model_fast = ModelWithCapabilities(
            spec=ModelSpec(provider="test", name="fast"),
            capabilities=ModelCapabilities(tokens_per_second=100.0)
        )
        assert query.matches(model_fast) is True

    def test_query_matches_family(self):
        """Test query matching with family filter."""
        from chuk_llm.registry.models import ModelCapabilities, ModelWithCapabilities

        query = ModelQuery(family="gpt-4o")

        # Wrong family
        model_wrong = ModelWithCapabilities(
            spec=ModelSpec(provider="openai", name="gpt-3.5-turbo", family="gpt-3.5"),
            capabilities=ModelCapabilities()
        )
        assert query.matches(model_wrong) is False

        # Correct family
        model_correct = ModelWithCapabilities(
            spec=ModelSpec(provider="openai", name="gpt-4o-mini", family="gpt-4o"),
            capabilities=ModelCapabilities()
        )
        assert query.matches(model_correct) is True

    def test_capabilities_merge_with_known_params(self):
        """Test merging capabilities with known_params sets."""
        from chuk_llm.registry.models import ModelCapabilities

        base = ModelCapabilities(
            max_context=100_000,
            known_params={"temperature", "max_tokens"}
        )
        override = ModelCapabilities(
            supports_tools=True,
            known_params={"top_p", "frequency_penalty"}
        )

        merged = base.merge(override)

        # Both sets should be merged
        assert "temperature" in merged.known_params
        assert "max_tokens" in merged.known_params
        assert "top_p" in merged.known_params
        assert "frequency_penalty" in merged.known_params
        assert len(merged.known_params) == 4
