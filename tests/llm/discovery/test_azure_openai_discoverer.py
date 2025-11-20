"""
Comprehensive tests for Azure OpenAI model discoverer
Target coverage: 95%+
"""

from unittest.mock import AsyncMock, Mock, patch
import httpx
import pytest

from chuk_llm.llm.discovery.azure_openai_discoverer import AzureOpenAIModelDiscoverer
from chuk_llm.llm.discovery.base import DiscoveredModel


class TestAzureOpenAIModelDiscoverer:
    """Test Azure OpenAI discoverer initialization"""

    def test_discoverer_initialization_defaults(self):
        """Test initialization with defaults"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        assert discoverer.provider_name == "azure_openai"
        assert discoverer.api_key == "test-key"
        assert discoverer.azure_endpoint == "https://test.openai.azure.com"
        assert discoverer.api_version == "2024-02-01"
        assert discoverer.azure_ad_token is None

    def test_discoverer_initialization_with_ad_token(self):
        """Test initialization with Azure AD token"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_ad_token="test-token",
            azure_endpoint="https://test.openai.azure.com"
        )
        assert discoverer.azure_ad_token == "test-token"
        assert discoverer.api_key is None

    def test_discoverer_initialization_custom_version(self):
        """Test initialization with custom API version"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2023-12-01"
        )
        assert discoverer.api_version == "2023-12-01"

    def test_discoverer_initialization_from_env(self):
        """Test endpoint from environment variable"""
        with patch.dict('os.environ', {'AZURE_OPENAI_ENDPOINT': 'https://env.openai.azure.com'}):
            discoverer = AzureOpenAIModelDiscoverer(api_key="key")
            assert discoverer.azure_endpoint == "https://env.openai.azure.com"

    def test_discoverer_initialization_no_endpoint(self):
        """Test initialization without endpoint logs warning"""
        with patch.dict('os.environ', {}, clear=True):
            discoverer = AzureOpenAIModelDiscoverer(api_key="key")
            assert discoverer.azure_endpoint is None

    def test_discoverer_initialization_no_credentials(self):
        """Test initialization without credentials logs warning"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_endpoint="https://test.openai.azure.com"
        )
        assert discoverer.api_key is None
        assert discoverer.azure_ad_token is None

    def test_deployment_patterns_structure(self):
        """Test deployment patterns are properly structured"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )
        patterns = discoverer.deployment_patterns
        assert "reasoning_deployments" in patterns
        assert "vision_deployments" in patterns
        assert "chat_deployments" in patterns
        assert "embedding_deployments" in patterns


class TestAzureOpenAIDiscoverDeployments:
    """Test deployment discovery"""

    @pytest.mark.asyncio
    async def test_discover_models_deployments_success(self):
        """Test successful deployment discovery"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response_data = {
            "data": [
                {
                    "id": "gpt-4o-deployment",
                    "model": "gpt-4o",
                    "owner": "organization-owner",
                    "created_at": 1234567890,
                },
                {
                    "id": "gpt-35-turbo-deployment",
                    "model": "gpt-35-turbo",
                    "owner": "organization-owner",
                    "created_at": 1234567891,
                }
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

            models = await discoverer.discover_models()

        assert len(models) >= 1  # At least some models discovered

    @pytest.mark.asyncio
    async def test_discover_models_no_endpoint(self):
        """Test discovery without endpoint returns empty"""
        discoverer = AzureOpenAIModelDiscoverer(api_key="test-key")
        models = await discoverer.discover_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_discover_models_no_credentials(self):
        """Test discovery without credentials returns empty"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_endpoint="https://test.openai.azure.com"
        )
        models = await discoverer.discover_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_discover_deployments_http_error(self):
        """Test handling of HTTP errors"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

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

            models = await discoverer.discover_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_discover_deployments_network_error(self):
        """Test handling of network errors"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            models = await discoverer.discover_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_discover_available_models_network_error(self):
        """Test handling of network errors in _discover_available_models"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            # _discover_available_models doesn't catch errors, it raises them
            # The error handling is in discover_models()
            with pytest.raises(httpx.ConnectError):
                await discoverer._discover_available_models()

    @pytest.mark.asyncio
    async def test_discover_available_models_no_endpoint(self):
        """Test _discover_available_models with no endpoint"""
        with patch.dict('os.environ', {}, clear=True):
            discoverer = AzureOpenAIModelDiscoverer(api_key="test-key")
            models = await discoverer._discover_available_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_discover_available_models_no_auth(self):
        """Test _discover_available_models with no auth"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_endpoint="https://test.openai.azure.com"
        )
        models = await discoverer._discover_available_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_discover_available_models_success(self):
        """Test successful available models discovery"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response_data = {
            "data": [
                {
                    "id": "gpt-4o",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "openai",
                },
                {
                    "id": "gpt-35-turbo",
                    "object": "model",
                    "created": 1234567891,
                    "owned_by": "openai",
                }
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

            models = await discoverer._discover_available_models()

        assert len(models) == 2
        assert models[0]["provider"] == "azure_openai"
        assert models[0]["deployment_status"] == "available_for_deployment"
        assert models[0]["source"] == "azure_models_api"
        assert "azure_model_id" in models[0]

    @pytest.mark.asyncio
    async def test_discover_deployments_deprecated(self):
        """Test _discover_deployments returns empty (deprecated)"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )
        deployments = await discoverer._discover_deployments()
        assert deployments == []


class TestAzureOpenAICategorizeDeployment:
    """Test deployment categorization"""

    def setup_method(self):
        self.discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )

    def test_categorize_reasoning_deployment(self):
        """Test reasoning deployment categorization"""
        category = self.discoverer._categorize_deployment("o1-preview-deployment")
        assert category == "reasoning"

    def test_categorize_vision_deployment(self):
        """Test vision deployment categorization"""
        category = self.discoverer._categorize_deployment("gpt-4o-deployment")
        assert category == "vision"

    def test_categorize_chat_deployment(self):
        """Test chat deployment categorization"""
        category = self.discoverer._categorize_deployment("gpt-35-turbo-deployment")
        assert category == "chat"

    def test_categorize_embedding_deployment(self):
        """Test embedding deployment categorization"""
        category = self.discoverer._categorize_deployment("text-embedding-ada-002")
        assert category == "embedding"

    def test_categorize_general_deployment(self):
        """Test unknown deployment falls back to general"""
        category = self.discoverer._categorize_deployment("random-unknown-deployment")
        assert category == "general"


class TestAzureOpenAIGetAuthHeaders:
    """Test authentication header generation"""

    def test_get_auth_headers_with_api_key(self):
        """Test headers with API key"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        headers = discoverer._get_auth_headers()
        assert "api-key" in headers
        assert headers["api-key"] == "test-key"

    def test_get_auth_headers_with_ad_token(self):
        """Test headers with Azure AD token"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_ad_token="test-token",
            azure_endpoint="https://test.openai.azure.com"
        )
        headers = discoverer._get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token"

    def test_get_auth_headers_prefers_ad_token(self):
        """Test AD token is preferred over API key"""
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_ad_token="test-token",
            azure_endpoint="https://test.openai.azure.com"
        )
        headers = discoverer._get_auth_headers()
        assert "Authorization" in headers
        assert "api-key" not in headers

    def test_get_auth_headers_no_credentials(self):
        """Test headers with no credentials returns None"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_endpoint="https://test.openai.azure.com"
        )
        headers = discoverer._get_auth_headers()
        assert headers is None


class TestAzureOpenAINormalizeModel:
    """Test model normalization"""

    def setup_method(self):
        self.discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )

    def test_normalize_model_data_complete(self):
        """Test normalization with complete data"""
        raw_model = {
            "name": "gpt-4o",
            "deployment_name": "gpt-4o-deployment",
            "capabilities": ["chat", "vision"],
            "provider_specific": {
                "model_family": "gpt-4o",
                "context_length": 128000,
            }
        }

        discovered = self.discoverer.normalize_model_data(raw_model)
        assert isinstance(discovered, DiscoveredModel)
        assert discovered.name == "gpt-4o"
        assert discovered.provider == "azure_openai"

    def test_normalize_model_data_minimal(self):
        """Test normalization with minimal data"""
        raw_model = {
            "name": "test-model",
        }

        discovered = self.discoverer.normalize_model_data(raw_model)
        assert discovered.name == "test-model"
        assert discovered.provider == "azure_openai"


class TestAzureOpenAIDiscovererFactory:
    """Test factory registration"""

    def test_discoverer_registered(self):
        from chuk_llm.llm.discovery.base import DiscovererFactory
        # Import to trigger registration
        from chuk_llm.llm.discovery import azure_openai_discoverer  # noqa: F401

        supported = DiscovererFactory.list_supported_providers()
        assert "azure_openai" in supported

    def test_create_discoverer_from_factory(self):
        from chuk_llm.llm.discovery.base import DiscovererFactory
        # Import to trigger registration
        from chuk_llm.llm.discovery import azure_openai_discoverer  # noqa: F401

        discoverer = DiscovererFactory.create_discoverer(
            "azure_openai",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-01"
        )

        assert isinstance(discoverer, AzureOpenAIModelDiscoverer)
        assert discoverer.provider_name == "azure_openai"
        assert discoverer.api_key == "test-key"


class TestAzureOpenAIEnhanceDeployment:
    """Test deployment enhancement"""

    def setup_method(self):
        self.discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )

    def test_enhance_deployment_data(self):
        """Test enhancing deployment with model info"""
        deployment_data = {
            "id": "gpt-4o-deployment",
            "model": "gpt-4o",
            "created_at": 1234567890,
            "status": "succeeded",
            "scale_settings": {
                "scale_type": "standard",
                "capacity": 10,
            },
        }

        enhanced = self.discoverer._enhance_deployment_data(deployment_data)

        assert enhanced["name"] == "gpt-4o-deployment"
        assert enhanced["underlying_model"] == "gpt-4o"
        assert enhanced["deployment_id"] == "gpt-4o-deployment"
        assert enhanced["deployment_status"] == "deployed"
        assert enhanced["provider"] == "azure_openai"
        assert enhanced["source"] == "azure_deployments_api"
        assert "azure_specific" in enhanced
        assert enhanced["azure_specific"]["deployment_name"] == "gpt-4o-deployment"

    def test_extract_capacity_with_info(self):
        """Test capacity extraction with data"""
        deployment_data = {
            "scale_settings": {
                "scale_type": "standard",
                "capacity": 10,
            }
        }

        capacity = self.discoverer._extract_capacity(deployment_data)

        assert capacity is not None
        assert capacity["capacity"] == 10
        assert capacity["scale_type"] == "standard"

    def test_extract_capacity_no_info(self):
        """Test capacity extraction with no data"""
        deployment_data = {}

        capacity = self.discoverer._extract_capacity(deployment_data)

        assert capacity is None


class TestAzureOpenAISortKey:
    """Test Azure model sorting"""

    def setup_method(self):
        self.discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )

    def test_azure_model_sort_key_deployed(self):
        """Test deployed models sort first"""
        deployed = {"deployment_status": "deployed", "name": "gpt-4"}
        available = {"deployment_status": "available_for_deployment", "name": "gpt-4"}

        deployed_key = self.discoverer._azure_model_sort_key(deployed)
        available_key = self.discoverer._azure_model_sort_key(available)

        assert deployed_key < available_key

    def test_azure_model_sort_key_fallback(self):
        """Test fallback models sort last"""
        deployed = {"deployment_status": "deployed", "name": "gpt-4"}
        fallback = {"deployment_status": "assumed_available", "name": "gpt-4"}

        deployed_key = self.discoverer._azure_model_sort_key(deployed)
        fallback_key = self.discoverer._azure_model_sort_key(fallback)

        assert deployed_key < fallback_key


class TestAzureOpenAIFallbackModels:
    """Test fallback model generation"""

    def setup_method(self):
        self.discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )

    def test_get_azure_fallback_models(self):
        """Test generation of fallback models"""
        fallback_models = self.discoverer._get_azure_fallback_models()

        assert len(fallback_models) > 0

        # Check that models are converted to Azure format
        for model in fallback_models:
            assert model["provider"] == "azure_openai"
            assert model["source"] == "azure_fallback"
            assert "deployment_id" in model
            assert "underlying_model" in model
            assert "azure_specific" in model
            assert model["deployment_status"] == "assumed_available"


class TestAzureOpenAIDeploymentTesting:
    """Test deployment availability testing"""

    def setup_method(self):
        self.discoverer = AzureOpenAIModelDiscoverer(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com"
        )

    @pytest.mark.asyncio
    async def test_test_deployment_availability_no_credentials(self):
        """Test deployment availability check with no credentials"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_endpoint="https://test.openai.azure.com"
        )
        result = await discoverer.test_deployment_availability("gpt-4")
        assert result is False

    @pytest.mark.asyncio
    async def test_test_deployment_availability_no_endpoint(self):
        """Test deployment availability check with no endpoint"""
        with patch.dict('os.environ', {}, clear=True):
            discoverer = AzureOpenAIModelDiscoverer(api_key="test-key")
            result = await discoverer.test_deployment_availability("gpt-4")
            assert result is False

    @pytest.mark.asyncio
    async def test_test_deployment_availability_import_error(self):
        """Test deployment availability when openai not available"""
        with patch("builtins.__import__", side_effect=ImportError("No openai")):
            result = await self.discoverer.test_deployment_availability("gpt-4")
            assert result is False

    @pytest.mark.asyncio
    async def test_test_deployment_availability_api_error(self):
        """Test deployment availability with API error"""
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        mock_client.close = AsyncMock()
        mock_azure_openai_class = Mock(return_value=mock_client)
        mock_openai.AsyncAzureOpenAI = mock_azure_openai_class

        import sys
        sys.modules['openai'] = mock_openai

        result = await self.discoverer.test_deployment_availability("gpt-4")
        assert result is False

        # Cleanup
        if 'openai' in sys.modules:
            del sys.modules['openai']

    @pytest.mark.asyncio
    async def test_test_deployment_availability_success(self):
        """Test successful deployment availability check"""
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=Mock())
        mock_client.close = AsyncMock()
        mock_azure_openai_class = Mock(return_value=mock_client)
        mock_openai.AsyncAzureOpenAI = mock_azure_openai_class

        import sys
        sys.modules['openai'] = mock_openai

        result = await self.discoverer.test_deployment_availability("gpt-4")
        assert result is True

        # Cleanup
        if 'openai' in sys.modules:
            del sys.modules['openai']

    @pytest.mark.asyncio
    async def test_test_deployment_availability_with_ad_token(self):
        """Test deployment availability check with AD token"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_ad_token="test-token",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_openai = Mock()
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=Mock())
        mock_client.close = AsyncMock()
        mock_azure_openai_class = Mock(return_value=mock_client)
        mock_openai.AsyncAzureOpenAI = mock_azure_openai_class

        import sys
        sys.modules['openai'] = mock_openai

        result = await discoverer.test_deployment_availability("gpt-4")
        assert result is True

        # Verify AD token was used
        call_kwargs = mock_azure_openai_class.call_args.kwargs
        assert "azure_ad_token" in call_kwargs
        assert call_kwargs["azure_ad_token"] == "test-token"

        # Cleanup
        if 'openai' in sys.modules:
            del sys.modules['openai']

    @pytest.mark.asyncio
    async def test_create_deployment(self):
        """Test create_deployment returns False (not implemented)"""
        result = await self.discoverer.create_deployment("gpt-4", "my-deployment")
        assert result is False

    @pytest.mark.asyncio
    async def test_discover_deployments_by_testing_no_credentials(self):
        """Test deployment testing discovery with no credentials"""
        discoverer = AzureOpenAIModelDiscoverer(
            azure_endpoint="https://test.openai.azure.com"
        )
        deployments = await discoverer.discover_deployments_by_testing()
        assert deployments == []

    @pytest.mark.asyncio
    async def test_discover_deployments_by_testing_custom_names(self):
        """Test deployment testing with custom names"""
        # Mock test_deployment_availability to return True for one deployment
        async def mock_test(name):
            return name == "custom-deployment"

        with patch.object(
            self.discoverer,
            'test_deployment_availability',
            side_effect=mock_test
        ):
            deployments = await self.discoverer.discover_deployments_by_testing(
                common_deployment_names=["custom-deployment", "other-deployment"]
            )

        assert len(deployments) == 1
        assert deployments[0]["name"] == "custom-deployment"
        assert deployments[0]["deployment_status"] == "active_discovered"
        # Note: source gets overwritten by _categorize_model enhancement
        assert deployments[0]["provider"] == "azure_openai"
        assert deployments[0]["discovery_method"] == "availability_test"

    @pytest.mark.asyncio
    async def test_discover_deployments_by_testing_default_names(self):
        """Test deployment testing with default names"""
        # Mock test_deployment_availability to return True for gpt-4o
        async def mock_test(name):
            return name == "gpt-4o"

        with patch.object(
            self.discoverer,
            'test_deployment_availability',
            side_effect=mock_test
        ):
            deployments = await self.discoverer.discover_deployments_by_testing()

        assert len(deployments) == 1
        assert deployments[0]["name"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_discover_deployments_by_testing_error_handling(self):
        """Test deployment testing handles individual errors"""
        # Mock test_deployment_availability to raise error for one deployment
        async def mock_test(name):
            if name == "error-deployment":
                raise Exception("Test error")
            return name == "good-deployment"

        with patch.object(
            self.discoverer,
            'test_deployment_availability',
            side_effect=mock_test
        ):
            deployments = await self.discoverer.discover_deployments_by_testing(
                common_deployment_names=["error-deployment", "good-deployment"]
            )

        assert len(deployments) == 1
        assert deployments[0]["name"] == "good-deployment"
