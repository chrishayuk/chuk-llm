"""Tests for chuk_llm/api/discovery.py - Model discovery API."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_llm.api.discovery import (
    _capabilities_to_features,
    discover_models,
    discover_models_sync,
    find_best_model,
    find_best_model_sync,
    get_model_info,
    get_model_info_sync,
    list_providers,
    list_providers_sync,
    list_supported_providers,
    show_discovered_models,
    show_discovered_models_sync,
)
from chuk_llm.registry import QualityTier


@pytest.fixture
def mock_model():
    """Create a mock model object."""
    model = MagicMock()
    model.spec.name = "test-model"
    model.spec.provider = "openai"
    model.spec.family = "gpt-4"
    model.capabilities.max_context = 128000
    model.capabilities.max_output_tokens = 4096
    model.capabilities.supports_tools = True
    model.capabilities.supports_vision = True
    model.capabilities.supports_json_mode = True
    model.capabilities.supports_streaming = True
    model.capabilities.quality_tier = QualityTier.BEST
    model.capabilities.tokens_per_second = 50.0
    model.capabilities.known_params = ["temperature", "max_tokens"]
    return model


@pytest.fixture
def mock_registry(mock_model):
    """Create a mock registry."""
    registry = AsyncMock()
    registry.get_models = AsyncMock(return_value=[mock_model])
    registry.find_best = AsyncMock(return_value=mock_model)
    return registry


class TestDiscoverModels:
    """Test discover_models function."""

    @pytest.mark.asyncio
    async def test_discover_models_success(self, mock_registry):
        """Test discovering models successfully."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            models = await discover_models("openai")

            assert len(models) == 1
            assert models[0]["name"] == "test-model"
            assert models[0]["provider"] == "openai"
            assert models[0]["context_length"] == 128000
            assert "tools" in models[0]["features"]

    @pytest.mark.asyncio
    async def test_discover_models_no_models_found(self, mock_registry):
        """Test discovering models when none found for provider."""
        mock_registry.get_models = AsyncMock(return_value=[])

        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            models = await discover_models("anthropic")

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_models_force_refresh(self, mock_registry):
        """Test discovering models with force_refresh."""
        with patch(
            "chuk_llm.api.discovery.get_registry", return_value=mock_registry
        ) as mock_get_registry:
            await discover_models("openai", force_refresh=True)

            mock_get_registry.assert_called_once_with(
                use_provider_apis=True, force_refresh=True
            )

    @pytest.mark.asyncio
    async def test_discover_models_filters_by_provider(self, mock_model):
        """Test that models are filtered by provider."""
        openai_model = mock_model
        anthropic_model = MagicMock()
        anthropic_model.spec.provider = "anthropic"
        anthropic_model.spec.name = "claude-3"
        anthropic_model.spec.family = "claude"
        anthropic_model.capabilities.max_context = 200000
        anthropic_model.capabilities.max_output_tokens = 4096
        anthropic_model.capabilities.supports_tools = True
        anthropic_model.capabilities.supports_vision = False
        anthropic_model.capabilities.supports_json_mode = False
        anthropic_model.capabilities.supports_streaming = True
        anthropic_model.capabilities.quality_tier = QualityTier.BEST
        anthropic_model.capabilities.tokens_per_second = 45.0

        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[openai_model, anthropic_model])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            models = await discover_models("openai")

            assert len(models) == 1
            assert models[0]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_discover_models_with_kwargs(self, mock_registry):
        """Test discovering models with extra kwargs (ignored)."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            models = await discover_models("openai", extra_param="ignored")

            assert len(models) == 1


class TestDiscoverModelsSync:
    """Test discover_models_sync function."""

    def test_discover_models_sync(self, mock_registry):
        """Test synchronous discover_models."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            models = discover_models_sync("openai")

            assert len(models) == 1
            assert models[0]["name"] == "test-model"


class TestGetModelInfo:
    """Test get_model_info function."""

    @pytest.mark.asyncio
    async def test_get_model_info_found(self, mock_registry):
        """Test getting model info when model exists."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            info = await get_model_info("openai", "test-model")

            assert info is not None
            assert info["name"] == "test-model"
            assert info["provider"] == "openai"
            assert info["context_length"] == 128000
            assert "known_params" in info
            assert "temperature" in info["known_params"]

    @pytest.mark.asyncio
    async def test_get_model_info_not_found(self, mock_registry):
        """Test getting model info when model doesn't exist."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            info = await get_model_info("openai", "nonexistent-model")

            assert info is None

    @pytest.mark.asyncio
    async def test_get_model_info_wrong_provider(self, mock_registry):
        """Test getting model info with wrong provider."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            info = await get_model_info("anthropic", "test-model")

            assert info is None

    @pytest.mark.asyncio
    async def test_get_model_info_with_kwargs(self, mock_registry):
        """Test getting model info with extra kwargs (ignored)."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            info = await get_model_info("openai", "test-model", extra="ignored")

            assert info is not None


class TestGetModelInfoSync:
    """Test get_model_info_sync function."""

    def test_get_model_info_sync(self, mock_registry):
        """Test synchronous get_model_info."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            info = get_model_info_sync("openai", "test-model")

            assert info is not None
            assert info["name"] == "test-model"


class TestFindBestModel:
    """Test find_best_model function."""

    @pytest.mark.asyncio
    async def test_find_best_model_success(self, mock_registry):
        """Test finding best model."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            model = await find_best_model(requires_tools=True)

            assert model is not None
            assert model["name"] == "test-model"
            assert "tools" in model["features"]

    @pytest.mark.asyncio
    async def test_find_best_model_not_found(self, mock_registry):
        """Test finding best model when none match."""
        mock_registry.find_best = AsyncMock(return_value=None)

        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            model = await find_best_model(requires_tools=True)

            assert model is None

    @pytest.mark.asyncio
    async def test_find_best_model_with_all_requirements(self, mock_registry):
        """Test finding best model with all requirements."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            model = await find_best_model(
                provider="openai",
                requires_tools=True,
                requires_vision=True,
                requires_json_mode=True,
                min_context=100000,
                quality_tier="best",
            )

            assert model is not None
            mock_registry.find_best.assert_called_once()
            call_kwargs = mock_registry.find_best.call_args[1]
            assert call_kwargs["provider"] == "openai"
            assert call_kwargs["requires_tools"] is True
            assert call_kwargs["requires_vision"] is True
            assert call_kwargs["min_context"] == 100000
            assert call_kwargs["quality_tier"] == QualityTier.BEST

    @pytest.mark.asyncio
    async def test_find_best_model_quality_tier_any(self, mock_registry):
        """Test finding best model with quality_tier='any'."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            model = await find_best_model(quality_tier="any")

            assert model is not None
            call_kwargs = mock_registry.find_best.call_args[1]
            assert call_kwargs["quality_tier"] == "any"

    @pytest.mark.asyncio
    async def test_find_best_model_with_kwargs(self, mock_registry):
        """Test finding best model with extra kwargs (ignored)."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            model = await find_best_model(extra="ignored")

            assert model is not None


class TestFindBestModelSync:
    """Test find_best_model_sync function."""

    def test_find_best_model_sync(self, mock_registry):
        """Test synchronous find_best_model."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            model = find_best_model_sync(requires_tools=True)

            assert model is not None
            assert model["name"] == "test-model"


class TestListProviders:
    """Test list_providers function."""

    @pytest.mark.asyncio
    async def test_list_providers(self, mock_model):
        """Test listing all providers."""
        openai_model = mock_model
        anthropic_model = MagicMock()
        anthropic_model.spec.provider = "anthropic"

        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[openai_model, anthropic_model])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            providers = await list_providers()

            assert len(providers) == 2
            assert "openai" in providers
            assert "anthropic" in providers
            # Should be sorted
            assert providers == sorted(providers)

    @pytest.mark.asyncio
    async def test_list_providers_empty(self):
        """Test listing providers when none available."""
        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            providers = await list_providers()

            assert providers == []


class TestListProvidersSync:
    """Test list_providers_sync function."""

    def test_list_providers_sync(self, mock_model):
        """Test synchronous list_providers."""
        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[mock_model])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            providers = list_providers_sync()

            assert "openai" in providers


class TestShowDiscoveredModels:
    """Test show_discovered_models function."""

    @pytest.mark.asyncio
    async def test_show_discovered_models_success(self, mock_registry, capsys):
        """Test showing discovered models."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            await show_discovered_models("openai")

            captured = capsys.readouterr()
            assert "Discovered" in captured.out
            assert "test-model" in captured.out

    @pytest.mark.asyncio
    async def test_show_discovered_models_none_found(self, capsys):
        """Test showing discovered models when none found."""
        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            await show_discovered_models("openai")

            captured = capsys.readouterr()
            assert "No models found" in captured.out

    @pytest.mark.asyncio
    async def test_show_discovered_models_with_force_refresh(self, mock_registry):
        """Test showing discovered models with force refresh."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            await show_discovered_models("openai", force_refresh=True)

            # Should not raise an error

    @pytest.mark.asyncio
    async def test_show_discovered_models_with_families(self, capsys):
        """Test showing discovered models grouped by family."""
        model1 = MagicMock()
        model1.spec.name = "gpt-4"
        model1.spec.provider = "openai"
        model1.spec.family = "gpt-4"
        model1.capabilities.max_context = 128000
        model1.capabilities.max_output_tokens = 4096
        model1.capabilities.supports_tools = True
        model1.capabilities.supports_vision = False
        model1.capabilities.supports_json_mode = True
        model1.capabilities.supports_streaming = True
        model1.capabilities.quality_tier = QualityTier.BEST
        model1.capabilities.tokens_per_second = 50.0

        model2 = MagicMock()
        model2.spec.name = "gpt-3.5-turbo"
        model2.spec.provider = "openai"
        model2.spec.family = "gpt-3.5"
        model2.capabilities.max_context = 16000
        model2.capabilities.max_output_tokens = 4096
        model2.capabilities.supports_tools = True
        model2.capabilities.supports_vision = False
        model2.capabilities.supports_json_mode = True
        model2.capabilities.supports_streaming = True
        model2.capabilities.quality_tier = QualityTier.BALANCED
        model2.capabilities.tokens_per_second = None

        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[model1, model2])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            await show_discovered_models("openai")

            captured = capsys.readouterr()
            assert "gpt-4" in captured.out
            assert "gpt-3.5-turbo" in captured.out


class TestShowDiscoveredModelsSync:
    """Test show_discovered_models_sync function."""

    def test_show_discovered_models_sync(self, mock_registry, capsys):
        """Test synchronous show_discovered_models."""
        with patch("chuk_llm.api.discovery.get_registry", return_value=mock_registry):
            show_discovered_models_sync("openai")

            captured = capsys.readouterr()
            assert "Discovered" in captured.out


class TestListSupportedProviders:
    """Test list_supported_providers function."""

    def test_list_supported_providers(self):
        """Test listing supported providers."""
        providers = list_supported_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert "gemini" in providers
        assert "ollama" in providers
        assert isinstance(providers, list)


class TestCapabilitiesToFeatures:
    """Test _capabilities_to_features helper function."""

    def test_capabilities_to_features_all_true(self):
        """Test converting capabilities when all are true."""
        capabilities = MagicMock()
        capabilities.supports_tools = True
        capabilities.supports_vision = True
        capabilities.supports_json_mode = True
        capabilities.supports_streaming = True

        features = _capabilities_to_features(capabilities)

        assert "text" in features
        assert "tools" in features
        assert "vision" in features
        assert "json" in features
        assert "streaming" in features

    def test_capabilities_to_features_all_false(self):
        """Test converting capabilities when all are false."""
        capabilities = MagicMock()
        capabilities.supports_tools = False
        capabilities.supports_vision = False
        capabilities.supports_json_mode = False
        capabilities.supports_streaming = False

        features = _capabilities_to_features(capabilities)

        assert features == ["text"]

    def test_capabilities_to_features_mixed(self):
        """Test converting capabilities with mixed values."""
        capabilities = MagicMock()
        capabilities.supports_tools = True
        capabilities.supports_vision = False
        capabilities.supports_json_mode = True
        capabilities.supports_streaming = False

        features = _capabilities_to_features(capabilities)

        assert "text" in features
        assert "tools" in features
        assert "vision" not in features
        assert "json" in features
        assert "streaming" not in features


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_discover_models_with_null_quality_tier(self):
        """Test discovering models when quality_tier is None."""
        model = MagicMock()
        model.spec.name = "test-model"
        model.spec.provider = "openai"
        model.spec.family = "gpt-4"
        model.capabilities.max_context = 128000
        model.capabilities.max_output_tokens = 4096
        model.capabilities.supports_tools = False
        model.capabilities.supports_vision = False
        model.capabilities.supports_json_mode = False
        model.capabilities.supports_streaming = False
        model.capabilities.quality_tier = None
        model.capabilities.tokens_per_second = None

        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[model])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            models = await discover_models("openai")

            assert models[0]["quality_tier"] == "unknown"

    @pytest.mark.asyncio
    async def test_get_model_info_with_null_known_params(self, mock_model):
        """Test getting model info when known_params is None."""
        mock_model.capabilities.known_params = None

        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[mock_model])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            info = await get_model_info("openai", "test-model")

            assert info["known_params"] == []

    @pytest.mark.asyncio
    async def test_show_discovered_models_with_null_family(self, capsys):
        """Test showing models when family is None."""
        model = MagicMock()
        model.spec.name = "test-model"
        model.spec.provider = "openai"
        model.spec.family = None
        model.capabilities.max_context = None
        model.capabilities.max_output_tokens = 4096
        model.capabilities.supports_tools = False
        model.capabilities.supports_vision = False
        model.capabilities.supports_json_mode = False
        model.capabilities.supports_streaming = False
        model.capabilities.quality_tier = QualityTier.CHEAP
        model.capabilities.tokens_per_second = None

        registry = AsyncMock()
        registry.get_models = AsyncMock(return_value=[model])

        with patch("chuk_llm.api.discovery.get_registry", return_value=registry):
            await show_discovered_models("openai")

            captured = capsys.readouterr()
            assert "unknown" in captured.out.lower()
            assert "Unknown" in captured.out
