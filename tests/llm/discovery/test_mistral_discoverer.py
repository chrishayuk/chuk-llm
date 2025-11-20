"""
Comprehensive tests for Mistral model discoverer
Target coverage: 100%
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from chuk_llm.llm.discovery.mistral_discoverer import MistralModelDiscoverer
from chuk_llm.llm.discovery.base import DiscoveredModel
# Import module to trigger factory registration
from chuk_llm.llm.discovery import mistral_discoverer  # noqa: F401


class TestMistralModelDiscoverer:
    """Test MistralModelDiscoverer initialization and basic functionality"""

    def setup_method(self):
        """Setup test discoverer"""
        self.api_key = "test-api-key"
        self.discoverer = MistralModelDiscoverer(
            provider_name="mistral", api_key=self.api_key
        )

    def test_discoverer_initialization(self):
        """Test discoverer initialization with defaults"""
        assert self.discoverer.provider_name == "mistral"
        assert self.discoverer.api_key == "test-api-key"
        assert self.discoverer.api_base == "https://api.mistral.ai/v1"

    def test_initialization_with_custom_api_base(self):
        """Test initialization with custom API base"""
        discoverer = MistralModelDiscoverer(
            provider_name="mistral",
            api_key="test-key",
            api_base="https://custom.api.mistral.ai/v1",
        )
        assert discoverer.api_base == "https://custom.api.mistral.ai/v1"

    def test_initialization_with_config(self):
        """Test initialization with additional config"""
        discoverer = MistralModelDiscoverer(
            provider_name="mistral",
            api_key="test-key",
            cache_timeout=600,
            api_base="https://custom.api/v1",
        )
        assert discoverer._cache_timeout == 600
        assert discoverer.api_base == "https://custom.api/v1"


class TestMistralDiscoverModels:
    """Test model discovery via Mistral API"""

    def setup_method(self):
        """Setup test discoverer"""
        self.api_key = "test-api-key"
        self.discoverer = MistralModelDiscoverer(
            provider_name="mistral", api_key=self.api_key
        )

    @pytest.mark.asyncio
    async def test_discover_models_success(self):
        """Test successful model discovery"""
        mock_response_data = {
            "data": [
                {
                    "id": "mistral-large-latest",
                    "created": 1234567890,
                    "owned_by": "mistralai",
                    "object": "model",
                },
                {
                    "id": "mistral-small-latest",
                    "created": 1234567891,
                    "owned_by": "mistralai",
                    "object": "model",
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
        assert call_args[0][0] == "https://api.mistral.ai/v1/models"
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key"

        # Check model data
        large_model = next(m for m in models if "large" in m["name"].lower())
        assert large_model["name"] == "mistral-large-latest"
        assert large_model["owned_by"] == "mistralai"
        assert large_model["source"] == "mistral_api"
        assert "provider_specific" in large_model

    @pytest.mark.asyncio
    async def test_discover_models_with_custom_api_base(self):
        """Test discovery with custom API base"""
        discoverer = MistralModelDiscoverer(
            provider_name="mistral",
            api_key="test-key",
            api_base="https://custom.api/v1",
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

        # Verify custom API base was used
        call_args = mock_client.return_value.get.call_args
        assert call_args[0][0] == "https://custom.api/v1/models"

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
                {"owned_by": "mistralai"},  # Missing 'id'
                {
                    "id": "mistral-small-latest",
                    "owned_by": "mistralai",
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
        assert models[0]["name"] == "mistral-small-latest"

    @pytest.mark.asyncio
    async def test_discover_models_http_error(self):
        """Test handling of HTTP errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=Mock()
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


class TestMistralModelSpecifics:
    """Test Mistral-specific model characteristics detection"""

    def setup_method(self):
        """Setup test discoverer"""
        self.discoverer = MistralModelDiscoverer(
            provider_name="mistral", api_key="test-key"
        )

    def test_magistral_characteristics(self):
        """Test Magistral (reasoning) model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("magistral-large-latest")

        assert specifics["model_family"] == "magistral"
        assert specifics["tier"] == "reasoning"
        assert specifics["reasoning_capable"] is True
        assert specifics["estimated_context_length"] == 128000
        assert specifics["max_output_tokens"] == 65536
        assert specifics["supports_streaming"] is True
        assert specifics["supports_tools"] is True
        assert specifics["supports_vision"] is False

    def test_mistral_large_characteristics(self):
        """Test Mistral Large model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("mistral-large-latest")

        assert specifics["model_family"] == "mistral_large"
        assert specifics["tier"] == "flagship"
        assert specifics["estimated_context_length"] == 128000
        assert specifics["max_output_tokens"] == 4096
        assert specifics["supports_streaming"] is True
        assert specifics["supports_tools"] is True

    def test_mistral_medium_characteristics(self):
        """Test Mistral Medium model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("mistral-medium-latest")

        assert specifics["model_family"] == "mistral_medium"
        assert specifics["tier"] == "balanced"
        assert specifics["estimated_context_length"] == 32000
        assert specifics["max_output_tokens"] == 4096

    def test_medium_in_name_characteristics(self):
        """Test model with 'medium' in name"""
        specifics = self.discoverer._get_mistral_specifics("medium-model-v1")

        assert specifics["model_family"] == "mistral_medium"
        assert specifics["tier"] == "balanced"

    def test_mistral_small_characteristics(self):
        """Test Mistral Small model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("mistral-small-latest")

        assert specifics["model_family"] == "mistral_small"
        assert specifics["tier"] == "fast"
        assert specifics["estimated_context_length"] == 32000
        assert specifics["max_output_tokens"] == 4096

    def test_small_in_name_characteristics(self):
        """Test model with 'small' in name"""
        specifics = self.discoverer._get_mistral_specifics("small-model-v1")

        assert specifics["model_family"] == "mistral_small"
        assert specifics["tier"] == "fast"

    def test_mixtral_characteristics(self):
        """Test Mixtral (MoE) model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("mixtral-8x7b-instruct-v0.1")

        assert specifics["model_family"] == "mixtral"
        assert specifics["tier"] == "moe"
        assert specifics["estimated_context_length"] == 32000
        assert specifics["max_output_tokens"] == 4096

    def test_pixtral_characteristics(self):
        """Test Pixtral (vision) model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("pixtral-12b-latest")

        assert specifics["model_family"] == "pixtral"
        assert specifics["tier"] == "vision"
        assert specifics["supports_vision"] is True
        assert specifics["estimated_context_length"] == 128000
        assert specifics["max_output_tokens"] == 4096

    def test_codestral_characteristics(self):
        """Test Codestral (coding) model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("codestral-latest")

        assert specifics["model_family"] == "codestral"
        assert specifics["tier"] == "code"
        assert specifics["specialization"] == "code_generation"
        assert specifics["estimated_context_length"] == 32000
        assert specifics["max_output_tokens"] == 4096

    def test_devstral_characteristics(self):
        """Test Devstral (coding variant) model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("devstral-latest")

        assert specifics["model_family"] == "codestral"
        assert specifics["tier"] == "code"
        assert specifics["specialization"] == "code_generation"

    def test_nemo_characteristics(self):
        """Test Mistral Nemo model characteristics"""
        specifics = self.discoverer._get_mistral_specifics("mistral-nemo-latest")

        assert specifics["model_family"] == "mistral_nemo"
        assert specifics["tier"] == "efficient"
        assert specifics["estimated_context_length"] == 128000
        assert specifics["max_output_tokens"] == 4096

    def test_unknown_model_characteristics(self):
        """Test unknown model defaults"""
        specifics = self.discoverer._get_mistral_specifics("mistral-future-model")

        assert specifics["model_family"] == "unknown"
        assert specifics["supports_streaming"] is True
        assert specifics["supports_tools"] is True
        assert specifics["supports_vision"] is False

    def test_case_insensitive_detection(self):
        """Test that model detection is case-insensitive"""
        specifics = self.discoverer._get_mistral_specifics("MISTRAL-LARGE-LATEST")

        assert specifics["model_family"] == "mistral_large"
        assert specifics["tier"] == "flagship"


class TestMistralNormalizeModelData:
    """Test normalization of model data to DiscoveredModel"""

    def setup_method(self):
        """Setup test discoverer"""
        self.discoverer = MistralModelDiscoverer(
            provider_name="mistral", api_key="test-key"
        )

    def test_normalize_model_data_complete(self):
        """Test normalization with complete model data"""
        raw_model = {
            "name": "mistral-large-latest",
            "created_at": 1234567890,
            "owned_by": "mistralai",
            "object": "model",
            "source": "mistral_api",
            "provider_specific": {
                "model_family": "mistral_large",
                "tier": "flagship",
                "estimated_context_length": 128000,
                "max_output_tokens": 4096,
            },
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert isinstance(discovered, DiscoveredModel)
        assert discovered.name == "mistral-large-latest"
        assert discovered.provider == "mistral"
        assert discovered.created_at == 1234567890
        assert discovered.family == "mistral_large"
        assert discovered.context_length == 128000
        assert discovered.max_output_tokens == 4096
        assert discovered.metadata["owned_by"] == "mistralai"
        assert discovered.metadata["tier"] == "flagship"

    def test_normalize_model_data_minimal(self):
        """Test normalization with minimal model data"""
        raw_model = {
            "name": "mistral-test",
            "provider_specific": {},
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert discovered.name == "mistral-test"
        assert discovered.provider == "mistral"
        assert discovered.family == "unknown"
        assert discovered.context_length is None
        assert discovered.max_output_tokens is None

    def test_normalize_model_data_missing_name(self):
        """Test normalization with missing name"""
        raw_model = {
            "provider_specific": {"model_family": "mistral_large"},
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert discovered.name == "unknown"
        assert discovered.family == "mistral_large"

    def test_normalize_model_data_missing_provider_specific(self):
        """Test normalization with missing provider_specific"""
        raw_model = {
            "name": "mistral-test",
            "owned_by": "mistralai",
        }

        discovered = self.discoverer.normalize_model_data(raw_model)

        assert discovered.name == "mistral-test"
        assert discovered.family == "unknown"
        assert discovered.metadata["owned_by"] == "mistralai"


class TestMistralDiscovererFactory:
    """Test discoverer registration with factory"""

    def test_discoverer_registered(self):
        """Test that Mistral discoverer is registered"""
        from chuk_llm.llm.discovery.base import DiscovererFactory
        # Import to trigger registration
        from chuk_llm.llm.discovery import mistral_discoverer  # noqa: F401

        supported = DiscovererFactory.list_supported_providers()
        assert "mistral" in supported

    def test_create_discoverer_from_factory(self):
        """Test creating Mistral discoverer from factory"""
        from chuk_llm.llm.discovery.base import DiscovererFactory
        # Import to trigger registration
        from chuk_llm.llm.discovery import mistral_discoverer  # noqa: F401

        discoverer = DiscovererFactory.create_discoverer(
            "mistral", api_key="test-key", api_base="https://custom.api/v1"
        )

        assert isinstance(discoverer, MistralModelDiscoverer)
        assert discoverer.provider_name == "mistral"
        assert discoverer.api_key == "test-key"
        assert discoverer.api_base == "https://custom.api/v1"


class TestMistralCaching:
    """Test caching functionality inherited from base"""

    def setup_method(self):
        """Setup test discoverer"""
        self.discoverer = MistralModelDiscoverer(
            provider_name="mistral",
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
                    "id": "mistral-large-latest",
                    "owned_by": "mistralai",
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

        assert info["provider"] == "mistral"
        assert info["cache_timeout"] == 300
        assert "last_discovery" in info
        assert "cached_models" in info
        assert "config" in info
