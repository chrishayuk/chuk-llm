# tests/test_discovery/test_base.py
"""
Tests for chuk_llm.llm.discovery.base module
"""

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from chuk_llm.configuration import Feature
from chuk_llm.llm.discovery.base import (
    BaseModelDiscoverer,
    DiscoveredModel,
    DiscovererFactory,
)


class TestDiscoveredModel:
    """Test DiscoveredModel dataclass"""

    def test_discovered_model_creation(self):
        """Test basic DiscoveredModel creation"""
        model = DiscoveredModel(name="test-model", provider="test-provider")

        assert model.name == "test-model"
        assert model.provider == "test-provider"
        assert model.size_bytes is None
        assert model.family == "unknown"
        assert model.capabilities == set()
        assert model.metadata == {}

    def test_discovered_model_with_full_data(self):
        """Test DiscoveredModel with all fields populated"""
        capabilities = {Feature.TEXT, Feature.VISION}
        metadata = {"custom_field": "value"}

        model = DiscoveredModel(
            name="gpt-4o",
            provider="openai",
            size_bytes=1000000,
            created_at="2024-01-01",
            modified_at="2024-01-15",
            version="1.0",
            metadata=metadata,
            family="gpt4",
            capabilities=capabilities,
            context_length=128000,
            max_output_tokens=8192,
            parameters="175B",
        )

        assert model.name == "gpt-4o"
        assert model.provider == "openai"
        assert model.size_bytes == 1000000
        assert model.family == "gpt4"
        assert model.capabilities == capabilities
        assert model.context_length == 128000
        assert model.max_output_tokens == 8192
        assert model.parameters == "175B"
        assert model.metadata == metadata

    def test_to_dict_conversion(self):
        """Test converting DiscoveredModel to dictionary"""
        capabilities = {Feature.TEXT, Feature.STREAMING}
        model = DiscoveredModel(
            name="test-model",
            provider="test-provider",
            family="test-family",
            capabilities=capabilities,
            context_length=8192,
            metadata={"key": "value"},
        )

        result = model.to_dict()

        assert result["name"] == "test-model"
        assert result["provider"] == "test-provider"
        assert result["family"] == "test-family"
        assert result["context_length"] == 8192
        assert result["metadata"] == {"key": "value"}
        assert set(result["capabilities"]) == {"text", "streaming"}

    def test_from_dict_creation(self):
        """Test creating DiscoveredModel from dictionary"""
        data = {
            "name": "test-model",
            "provider": "test-provider",
            "family": "test-family",
            "capabilities": ["text", "vision"],
            "context_length": 16384,
            "metadata": {"test": True},
        }

        model = DiscoveredModel.from_dict(data)

        assert model.name == "test-model"
        assert model.provider == "test-provider"
        assert model.family == "test-family"
        assert model.context_length == 16384
        assert model.metadata == {"test": True}
        assert Feature.TEXT in model.capabilities
        assert Feature.VISION in model.capabilities

    def test_from_dict_with_missing_fields(self):
        """Test creating DiscoveredModel from incomplete dictionary"""
        data = {"name": "minimal-model"}

        model = DiscoveredModel.from_dict(data)

        assert model.name == "minimal-model"
        assert model.provider == "unknown"
        assert model.family == "unknown"
        assert model.capabilities == set()
        assert model.metadata == {}

    def test_from_dict_with_invalid_capabilities(self):
        """Test handling invalid capability strings"""
        data = {
            "name": "test-model",
            "capabilities": ["text", "invalid_capability", "vision"],
        }

        with patch.object(Feature, "from_string") as mock_from_string:
            mock_from_string.side_effect = (
                lambda x: Feature.TEXT
                if x == "text"
                else Feature.VISION
                if x == "vision"
                else None
            )

            model = DiscoveredModel.from_dict(data)

            # Should filter out None values
            valid_capabilities = {cap for cap in model.capabilities if cap is not None}
            assert len(valid_capabilities) >= 2


class MockDiscoverer(BaseModelDiscoverer):
    """Mock discoverer for testing"""

    def __init__(self, provider_name: str, **config):
        super().__init__(provider_name, **config)
        self.discover_calls = 0
        self.mock_models = config.get("mock_models", [])
        self.should_fail = config.get("should_fail", False)

    async def discover_models(self) -> list[dict[str, Any]]:
        """Mock discovery implementation"""
        self.discover_calls += 1

        if self.should_fail:
            raise Exception("Mock discovery failure")

        return self.mock_models.copy()


class TestBaseModelDiscoverer:
    """Test BaseModelDiscoverer abstract base class"""

    def test_discoverer_initialization(self):
        """Test basic discoverer initialization"""
        discoverer = MockDiscoverer("test-provider", cache_timeout=600)

        assert discoverer.provider_name == "test-provider"
        assert discoverer.config["cache_timeout"] == 600
        assert discoverer._cache_timeout == 600
        assert discoverer._discovery_cache == {}
        assert discoverer._last_discovery is None

    @pytest.mark.asyncio
    async def test_discover_models_called(self):
        """Test that discover_models is called"""
        mock_models = [{"name": "test-model", "size": 1000}]
        discoverer = MockDiscoverer("test-provider", mock_models=mock_models)

        result = await discoverer.discover_models()

        assert result == mock_models
        assert discoverer.discover_calls == 1

    @pytest.mark.asyncio
    async def test_discover_with_cache_fresh_discovery(self):
        """Test fresh discovery without cache"""
        mock_models = [{"name": "model1"}, {"name": "model2"}]
        discoverer = MockDiscoverer("test-provider", mock_models=mock_models)

        result = await discoverer.discover_with_cache()

        assert result == mock_models
        assert discoverer.discover_calls == 1
        assert discoverer._last_discovery is not None

    @pytest.mark.asyncio
    async def test_discover_with_cache_uses_cache(self):
        """Test that cache is used when valid"""
        mock_models = [{"name": "cached-model"}]
        discoverer = MockDiscoverer(
            "test-provider", mock_models=mock_models, cache_timeout=60
        )

        # First call - should hit API
        result1 = await discoverer.discover_with_cache()

        # Second call within cache timeout - should use cache
        result2 = await discoverer.discover_with_cache()

        assert result1 == result2 == mock_models
        assert discoverer.discover_calls == 1  # Only called once due to cache

    @pytest.mark.asyncio
    async def test_discover_with_cache_force_refresh(self):
        """Test force refresh bypasses cache"""
        mock_models = [{"name": "model"}]
        discoverer = MockDiscoverer("test-provider", mock_models=mock_models)

        # First call
        await discoverer.discover_with_cache()

        # Force refresh should bypass cache
        await discoverer.discover_with_cache(force_refresh=True)

        assert discoverer.discover_calls == 2

    @pytest.mark.asyncio
    async def test_discover_with_cache_expired(self):
        """Test cache expiration"""
        mock_models = [{"name": "model"}]
        discoverer = MockDiscoverer(
            "test-provider", mock_models=mock_models, cache_timeout=0.1
        )

        # First call
        await discoverer.discover_with_cache()

        # Wait for cache to expire
        await asyncio.sleep(0.2)

        # Second call should refresh
        await discoverer.discover_with_cache()

        assert discoverer.discover_calls == 2

    @pytest.mark.asyncio
    async def test_discover_with_cache_handles_failure(self):
        """Test handling of discovery failures with cache fallback"""
        discoverer = MockDiscoverer("test-provider", should_fail=True)

        # Prime cache with successful data
        discoverer._discovery_cache["test-provider_models"] = (
            [{"name": "cached"}],
            time.time(),
        )

        # Make discoverer fail but return stale cache
        result = await discoverer.discover_with_cache()

        assert result == [{"name": "cached"}]

    @pytest.mark.asyncio
    async def test_discover_with_cache_failure_no_cache(self):
        """Test handling of discovery failures without cache"""
        discoverer = MockDiscoverer("test-provider", should_fail=True)

        result = await discoverer.discover_with_cache()

        assert result == []

    def test_normalize_model_data_basic(self):
        """Test basic model data normalization"""
        discoverer = MockDiscoverer("test-provider")
        raw_model = {
            "name": "test-model",
            "size": 1000000,
            "created_at": "2024-01-01",
            "version": "1.0",
        }

        result = discoverer.normalize_model_data(raw_model)

        assert isinstance(result, DiscoveredModel)
        assert result.name == "test-model"
        assert result.provider == "test-provider"
        assert result.size_bytes == 1000000
        assert result.created_at == "2024-01-01"
        assert result.version == "1.0"
        assert result.metadata == raw_model

    def test_normalize_model_data_minimal(self):
        """Test normalization with minimal data"""
        discoverer = MockDiscoverer("test-provider")
        raw_model = {}

        result = discoverer.normalize_model_data(raw_model)

        assert result.name == "unknown"
        assert result.provider == "test-provider"
        assert result.size_bytes is None
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_get_model_metadata_default(self):
        """Test default get_model_metadata implementation"""
        discoverer = MockDiscoverer("test-provider")

        result = await discoverer.get_model_metadata("any-model")

        assert result is None

    def test_get_discovery_info(self):
        """Test discovery info generation"""
        discoverer = MockDiscoverer(
            "test-provider", cache_timeout=300, custom_param="value"
        )
        discoverer._last_discovery = 1234567890.0
        discoverer._discovery_cache["test-provider_models"] = (
            [{"name": "model1"}, {"name": "model2"}],
            time.time(),
        )

        info = discoverer.get_discovery_info()

        assert info["provider"] == "test-provider"
        assert info["cache_timeout"] == 300
        assert info["last_discovery"] == 1234567890.0
        assert info["cached_models"] == 2
        assert "custom_param" in info["config"]
        assert "_cache_timeout" not in info["config"]  # Private fields excluded


class TestDiscovererFactory:
    """Test DiscovererFactory class"""

    def setup_method(self):
        """Reset factory state before each test"""
        DiscovererFactory._discoverers = {}
        DiscovererFactory._imported = False

    def test_register_discoverer(self):
        """Test registering a discoverer"""
        DiscovererFactory.register_discoverer("test", MockDiscoverer)

        assert "test" in DiscovererFactory._discoverers
        assert DiscovererFactory._discoverers["test"] == MockDiscoverer

    def test_create_discoverer_success(self):
        """Test successful discoverer creation"""
        DiscovererFactory.register_discoverer("test", MockDiscoverer)

        discoverer = DiscovererFactory.create_discoverer("test", param1="value1")

        assert isinstance(discoverer, MockDiscoverer)
        assert discoverer.provider_name == "test"
        assert discoverer.config["param1"] == "value1"

    def test_create_discoverer_not_found(self):
        """Test error when discoverer not found"""
        with pytest.raises(
            ValueError, match="No discoverer available for provider: nonexistent"
        ):
            DiscovererFactory.create_discoverer("nonexistent")

    def test_list_supported_providers_empty(self):
        """Test listing providers when none registered"""
        providers = DiscovererFactory.list_supported_providers()

        assert providers == []

    def test_list_supported_providers_with_providers(self):
        """Test listing registered providers"""
        DiscovererFactory.register_discoverer("provider1", MockDiscoverer)
        DiscovererFactory.register_discoverer("provider2", MockDiscoverer)

        providers = DiscovererFactory.list_supported_providers()

        assert set(providers) == {"provider1", "provider2"}

    @patch("chuk_llm.llm.discovery.base.log")
    def test_auto_import_discoverers_success(self, mock_log):
        """Test successful auto-import of discoverers"""
        with patch(
            "chuk_llm.llm.discovery.base.DiscovererFactory._discoverers",
            {"ollama": MockDiscoverer},
        ):
            DiscovererFactory._imported = False

            with patch("builtins.__import__"):  # Mock the imports
                DiscovererFactory._auto_import_discoverers()

            assert DiscovererFactory._imported is True
            mock_log.debug.assert_called()

    @patch("chuk_llm.llm.discovery.base.log")
    def test_auto_import_discoverers_failure(self, mock_log):
        """Test handling of import failures"""
        DiscovererFactory._imported = False

        with patch("builtins.__import__", side_effect=ImportError("Test import error")):
            DiscovererFactory._auto_import_discoverers()

        assert (
            DiscovererFactory._imported is True
        )  # Should still set to True to avoid retrying
        mock_log.warning.assert_called()

    def test_auto_import_discoverers_already_imported(self):
        """Test that auto-import is skipped when already imported"""
        DiscovererFactory._imported = True

        with patch("builtins.__import__") as mock_import:
            DiscovererFactory._auto_import_discoverers()
            mock_import.assert_not_called()

    def test_create_discoverer_triggers_auto_import(self):
        """Test that create_discoverer triggers auto-import"""
        DiscovererFactory._imported = False

        with patch.object(
            DiscovererFactory, "_auto_import_discoverers"
        ) as mock_auto_import:
            with pytest.raises(
                ValueError
            ):  # Will fail because no discoverer registered
                DiscovererFactory.create_discoverer("test")

            mock_auto_import.assert_called_once()

    def test_list_supported_providers_triggers_auto_import(self):
        """Test that list_supported_providers triggers auto-import"""
        DiscovererFactory._imported = False

        with patch.object(
            DiscovererFactory, "_auto_import_discoverers"
        ) as mock_auto_import:
            DiscovererFactory.list_supported_providers()

            mock_auto_import.assert_called_once()


class TestIntegration:
    """Integration tests for base discovery components"""

    def setup_method(self):
        """Reset factory state"""
        DiscovererFactory._discoverers = {}
        DiscovererFactory._imported = False

    @pytest.mark.asyncio
    async def test_full_discovery_workflow(self):
        """Test complete discovery workflow"""
        # Register a mock discoverer
        DiscovererFactory.register_discoverer("test", MockDiscoverer)

        # Mock model data
        mock_models = [
            {"name": "model1", "size": 1000000},
            {"name": "model2", "size": 2000000},
        ]

        # Create discoverer
        discoverer = DiscovererFactory.create_discoverer(
            "test", mock_models=mock_models
        )

        # Discover models
        raw_models = await discoverer.discover_with_cache()
        assert len(raw_models) == 2

        # Normalize models
        normalized_models = [discoverer.normalize_model_data(raw) for raw in raw_models]

        assert all(isinstance(model, DiscoveredModel) for model in normalized_models)
        assert normalized_models[0].name == "model1"
        assert normalized_models[1].name == "model2"

        # Test serialization round-trip
        for model in normalized_models:
            model_dict = model.to_dict()
            restored_model = DiscoveredModel.from_dict(model_dict)
            assert restored_model.name == model.name
            assert restored_model.provider == model.provider

    def test_multiple_providers(self):
        """Test managing multiple provider discoverers"""
        DiscovererFactory.register_discoverer("provider1", MockDiscoverer)
        DiscovererFactory.register_discoverer("provider2", MockDiscoverer)

        discoverer1 = DiscovererFactory.create_discoverer("provider1")
        discoverer2 = DiscovererFactory.create_discoverer("provider2")

        assert discoverer1.provider_name == "provider1"
        assert discoverer2.provider_name == "provider2"
        assert isinstance(discoverer1, MockDiscoverer)
        assert isinstance(discoverer2, MockDiscoverer)

        providers = DiscovererFactory.list_supported_providers()
        assert set(providers) == {"provider1", "provider2"}
