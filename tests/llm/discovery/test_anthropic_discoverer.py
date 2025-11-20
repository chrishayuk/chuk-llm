"""
Comprehensive tests for Anthropic model discoverer
Target coverage: 95%+
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from chuk_llm.llm.discovery.anthropic_discoverer import AnthropicModelDiscoverer
from chuk_llm.llm.discovery.base import DiscoveredModel


class TestAnthropicModelDiscoverer:
    """Test AnthropicModelDiscoverer initialization and basic functionality"""

    def setup_method(self):
        """Setup test discoverer"""
        self.api_key = "test-api-key"
        self.discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic", api_key=self.api_key
        )

    def test_discoverer_initialization(self):
        """Test discoverer initialization with defaults"""
        assert self.discoverer.provider_name == "anthropic"
        assert self.discoverer.api_key == "test-api-key"
        assert self.discoverer.api_base == "https://api.anthropic.com/v1"
        assert self.discoverer.anthropic_version == "2023-06-01"

    def test_initialization_with_custom_version(self):
        """Test initialization with custom Anthropic version"""
        discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic",
            api_key="test-key",
            anthropic_version="2024-01-01",
        )
        assert discoverer.anthropic_version == "2024-01-01"

    def test_initialization_with_config(self):
        """Test initialization with additional config"""
        discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic",
            api_key="test-key",
            cache_timeout=600,
            anthropic_version="2024-01-01",
        )
        assert discoverer._cache_timeout == 600
        assert discoverer.anthropic_version == "2024-01-01"


class TestAnthropicDiscoverModels:
    """Test model discovery via Anthropic API"""

    def setup_method(self):
        """Setup test discoverer"""
        self.api_key = "test-api-key"
        self.discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic", api_key=self.api_key
        )

    @pytest.mark.asyncio
    async def test_discover_models_success(self):
        """Test successful model discovery"""
        mock_response_data = {
            "data": [
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                    "created_at": "2024-10-22T00:00:00Z",
                    "type": "chat",
                },
                {
                    "id": "claude-3-opus-20240229",
                    "display_name": "Claude 3 Opus",
                    "created_at": "2024-02-29T00:00:00Z",
                    "type": "chat",
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            models = await self.discoverer.discover_models()

        assert len(models) == 2

        # Check API call
        mock_client.return_value.get.assert_called_once()
        call_args = mock_client.return_value.get.call_args
        assert call_args[0][0] == "https://api.anthropic.com/v1/models"
        assert call_args[1]["headers"]["x-api-key"] == "test-api-key"
        assert call_args[1]["headers"]["anthropic-version"] == "2023-06-01"

        # Check model data
        sonnet_model = next(m for m in models if "sonnet" in m["name"].lower())
        assert sonnet_model["name"] == "claude-3-5-sonnet-20241022"
        assert sonnet_model["display_name"] == "Claude 3.5 Sonnet"
        assert sonnet_model["source"] == "anthropic_api"
        assert "provider_specific" in sonnet_model

    @pytest.mark.asyncio
    async def test_discover_models_with_custom_version(self):
        """Test discovery with custom API version"""
        discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic",
            api_key="test-key",
            anthropic_version="2024-01-01",
        )

        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            await discoverer.discover_models()

        # Verify custom version was used
        call_args = mock_client.return_value.get.call_args
        assert call_args[1]["headers"]["anthropic-version"] == "2024-01-01"

    @pytest.mark.asyncio
    async def test_discover_models_empty_response(self):
        """Test handling of empty model list"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            models = await self.discoverer.discover_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_discover_models_missing_id(self):
        """Test handling of model with missing ID"""
        mock_response_data = {
            "data": [
                {"display_name": "Test Model"},  # Missing 'id'
                {
                    "id": "claude-3-haiku-20240307",
                    "display_name": "Claude 3 Haiku",
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            models = await self.discoverer.discover_models()

        # Should skip model without ID
        assert len(models) == 1
        assert models[0]["name"] == "claude-3-haiku-20240307"

    @pytest.mark.asyncio
    async def test_discover_models_http_error(self):
        """Test handling of HTTP errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=Mock()
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            models = await self.discoverer.discover_models()

        # Should return empty list on error
        assert models == []

    @pytest.mark.asyncio
    async def test_discover_models_network_error(self):
        """Test handling of network errors"""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            models = await self.discoverer.discover_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_discover_models_timeout(self):
        """Test handling of timeout errors"""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timeout")
            )

            models = await self.discoverer.discover_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_discover_models_malformed_json(self):
        """Test handling of malformed JSON response"""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            models = await self.discoverer.discover_models()

        assert models == []


class TestAnthropicModelSpecifics:
    """Test Anthropic-specific model characteristics detection"""

    def setup_method(self):
        """Setup test discoverer"""
        self.discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic", api_key="test-key"
        )

    def test_opus_4_characteristics(self):
        """Test Opus 4 model characteristics"""
        model_data = {"id": "claude-opus-4-20250514"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-opus-4-20250514", model_data
        )

        assert specifics["model_family"] == "opus_4"
        assert specifics["tier"] == "flagship"
        assert specifics["supports_vision"] is True
        assert specifics["extended_thinking"] is True
        assert specifics["estimated_context_length"] == 200000
        assert specifics["max_output_tokens"] == 32000
        assert specifics["supports_streaming"] is True
        assert specifics["supports_tools"] is True

    def test_opus_characteristics(self):
        """Test Opus (non-4) model characteristics"""
        model_data = {"id": "claude-3-opus-20240229"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-3-opus-20240229", model_data
        )

        assert specifics["model_family"] == "opus"
        assert specifics["tier"] == "flagship"
        assert specifics["supports_vision"] is True  # 3-opus has vision
        assert specifics["estimated_context_length"] == 200000
        assert specifics["max_output_tokens"] == 4096

    def test_opus_without_vision(self):
        """Test older Opus without vision"""
        model_data = {"id": "claude-2-opus"}
        specifics = self.discoverer._get_anthropic_specifics("claude-2-opus", model_data)

        assert specifics["model_family"] == "opus"
        assert specifics["supports_vision"] is False

    def test_sonnet_4_characteristics(self):
        """Test Sonnet 4 model characteristics"""
        model_data = {"id": "claude-sonnet-4-20250514"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-sonnet-4-20250514", model_data
        )

        assert specifics["model_family"] == "sonnet_4"
        assert specifics["tier"] == "balanced"
        assert specifics["supports_vision"] is True
        assert specifics["extended_thinking"] is True
        assert specifics["estimated_context_length"] == 200000
        assert specifics["max_output_tokens"] == 32000

    def test_sonnet_3_5_characteristics(self):
        """Test Sonnet 3.5 model characteristics"""
        # Use 3.5 (not 3-5) to match the logic which looks for "3.5" substring
        model_data = {"id": "claude-3.5-sonnet-20241022"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-3.5-sonnet-20241022", model_data
        )

        assert specifics["model_family"] == "sonnet"
        assert specifics["tier"] == "balanced"
        assert specifics["supports_vision"] is True
        assert specifics["max_output_tokens"] == 8192

    def test_sonnet_3_7_characteristics(self):
        """Test Sonnet 3.7 model characteristics"""
        model_data = {"id": "claude-3.7-sonnet-20250515"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-3.7-sonnet-20250515", model_data
        )

        assert specifics["model_family"] == "sonnet"
        assert specifics["supports_vision"] is True
        assert specifics["max_output_tokens"] == 8192

    def test_sonnet_without_vision(self):
        """Test older Sonnet without vision"""
        model_data = {"id": "claude-2-sonnet"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-2-sonnet", model_data
        )

        assert specifics["model_family"] == "sonnet"
        assert specifics["supports_vision"] is False
        assert specifics["max_output_tokens"] == 4096

    def test_haiku_3_5_characteristics(self):
        """Test Haiku 3.5 model characteristics"""
        model_data = {"id": "claude-3-5-haiku-20241022"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-3-5-haiku-20241022", model_data
        )

        assert specifics["model_family"] == "haiku"
        assert specifics["tier"] == "fast"
        assert specifics["supports_vision"] is True
        assert specifics["estimated_context_length"] == 200000
        assert specifics["max_output_tokens"] == 4096

    def test_haiku_without_vision(self):
        """Test older Haiku without vision"""
        model_data = {"id": "claude-3-haiku-20240307"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-3-haiku-20240307", model_data
        )

        assert specifics["model_family"] == "haiku"
        assert specifics["supports_vision"] is False

    def test_claude_2_characteristics(self):
        """Test Claude 2 model characteristics"""
        model_data = {"id": "claude-2.1"}
        specifics = self.discoverer._get_anthropic_specifics("claude-2.1", model_data)

        assert specifics["model_family"] == "claude_2"
        assert specifics["tier"] == "legacy"
        assert specifics["supports_vision"] is False
        assert specifics["estimated_context_length"] == 100000
        assert specifics["max_output_tokens"] == 4096

    def test_instant_characteristics(self):
        """Test Claude Instant model characteristics"""
        model_data = {"id": "claude-instant-1.2"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-instant-1.2", model_data
        )

        assert specifics["model_family"] == "instant"
        assert specifics["tier"] == "fast_legacy"
        assert specifics["supports_vision"] is False
        assert specifics["estimated_context_length"] == 100000
        assert specifics["max_output_tokens"] == 4096

    def test_unknown_model_characteristics(self):
        """Test unknown model defaults"""
        model_data = {"id": "claude-future-model"}
        specifics = self.discoverer._get_anthropic_specifics(
            "claude-future-model", model_data
        )

        assert specifics["model_family"] == "unknown"
        assert specifics["supports_streaming"] is True
        assert specifics["supports_tools"] is True
        assert specifics["supports_vision"] is False

    def test_case_insensitive_detection(self):
        """Test that model detection is case-insensitive"""
        model_data = {"id": "CLAUDE-3-5-SONNET-20241022"}
        specifics = self.discoverer._get_anthropic_specifics(
            "CLAUDE-3-5-SONNET-20241022", model_data
        )

        assert specifics["model_family"] == "sonnet"
        assert specifics["supports_vision"] is True


class TestAnthropicNormalizeModelData:
    """Test normalization of model data to DiscoveredModel"""

    def setup_method(self):
        """Setup test discoverer"""
        self.discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic", api_key="test-key"
        )

    def test_normalize_model_data_complete(self):
        """Test normalization with complete model data"""
        raw_model = {
            "name": "claude-3-5-sonnet-20241022",
            "display_name": "Claude 3.5 Sonnet",
            "created_at": "2024-10-22T00:00:00Z",
            "type": "chat",
            "source": "anthropic_api",
            "provider_specific": {
                "model_family": "sonnet",
                "tier": "balanced",
                "supports_vision": True,
                "estimated_context_length": 200000,
                "max_output_tokens": 8192,
            },
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert isinstance(discovered, DiscoveredModel)
        assert discovered.name == "claude-3-5-sonnet-20241022"
        assert discovered.provider == "anthropic"
        assert discovered.created_at == "2024-10-22T00:00:00Z"
        assert discovered.family == "sonnet"
        assert discovered.context_length == 200000
        assert discovered.max_output_tokens == 8192
        assert discovered.metadata["display_name"] == "Claude 3.5 Sonnet"
        assert discovered.metadata["tier"] == "balanced"

    def test_normalize_model_data_minimal(self):
        """Test normalization with minimal model data"""
        raw_model = {
            "name": "claude-test",
            "provider_specific": {},
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert discovered.name == "claude-test"
        assert discovered.provider == "anthropic"
        assert discovered.family == "unknown"
        assert discovered.context_length is None
        assert discovered.max_output_tokens is None

    def test_normalize_model_data_missing_name(self):
        """Test normalization with missing name"""
        raw_model = {
            "provider_specific": {"model_family": "sonnet"},
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert discovered.name == "unknown"
        assert discovered.family == "sonnet"

    def test_normalize_model_data_missing_provider_specific(self):
        """Test normalization with missing provider_specific"""
        raw_model = {
            "name": "claude-test",
            "display_name": "Test Model",
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert discovered.name == "claude-test"
        assert discovered.family == "unknown"
        assert discovered.metadata["display_name"] == "Test Model"


class TestAnthropicDiscovererFactory:
    """Test discoverer registration with factory"""

    def test_discoverer_registered(self):
        """Test that Anthropic discoverer is registered"""
        from chuk_llm.llm.discovery.base import DiscovererFactory
        # Import to trigger registration
        from chuk_llm.llm.discovery import anthropic_discoverer  # noqa: F401

        supported = DiscovererFactory.list_supported_providers()
        assert "anthropic" in supported

    def test_create_discoverer_from_factory(self):
        """Test creating Anthropic discoverer from factory"""
        from chuk_llm.llm.discovery.base import DiscovererFactory
        # Import to trigger registration
        from chuk_llm.llm.discovery import anthropic_discoverer  # noqa: F401

        discoverer = DiscovererFactory.create_discoverer(
            "anthropic", api_key="test-key", anthropic_version="2024-01-01"
        )

        assert isinstance(discoverer, AnthropicModelDiscoverer)
        assert discoverer.provider_name == "anthropic"
        assert discoverer.api_key == "test-key"
        assert discoverer.anthropic_version == "2024-01-01"


class TestAnthropicCaching:
    """Test caching functionality inherited from base"""

    def setup_method(self):
        """Setup test discoverer"""
        self.discoverer = AnthropicModelDiscoverer(
            provider_name="anthropic",
            api_key="test-key",
            cache_timeout=300,
        )

    @pytest.mark.asyncio
    async def test_discover_with_cache_success(self):
        """Test cached discovery"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            # First call - should hit API
            models1 = await self.discoverer.discover_with_cache()
            assert len(models1) == 1

            # Second call - should use cache
            models2 = await self.discoverer.discover_with_cache()
            assert len(models2) == 1

            # API should only be called once
            assert mock_client.return_value.get.call_count == 1

    @pytest.mark.asyncio
    async def test_discover_with_cache_force_refresh(self):
        """Test forcing cache refresh"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            # First call
            await self.discoverer.discover_with_cache()

            # Force refresh
            await self.discoverer.discover_with_cache(force_refresh=True)

            # API should be called twice
            assert mock_client.return_value.get.call_count == 2

    def test_get_discovery_info(self):
        """Test getting discoverer information"""
        info = self.discoverer.get_discovery_info()

        assert info["provider"] == "anthropic"
        assert info["cache_timeout"] == 300
        assert "last_discovery" in info
        assert "cached_models" in info
        assert "config" in info
