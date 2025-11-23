"""Comprehensive tests for registry/sources/azure_openai.py"""

import pytest
import os
from unittest.mock import Mock, AsyncMock, patch
import httpx
from chuk_llm.registry.sources.azure_openai import AzureOpenAIModelSource
from chuk_llm.core.enums import Provider


class TestAzureOpenAIModelSourceInit:
    """Test AzureOpenAIModelSource initialization"""

    def test_init_with_all_params(self):
        """Test initialization with all parameters provided"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-02-01",
            timeout=15.0
        )

        assert source.api_key == "test-key"
        assert source.azure_endpoint == "https://test.openai.azure.com"
        assert source.api_version == "2024-02-01"
        assert source.timeout == 15.0

    def test_init_with_defaults(self):
        """Test initialization with environment variables"""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com"
        }):
            source = AzureOpenAIModelSource()

            assert source.api_key == "env-key"
            assert source.azure_endpoint == "https://env.openai.azure.com"
            assert source.api_version == "2024-02-01"
            assert source.timeout == 10.0

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from endpoint"""
        source = AzureOpenAIModelSource(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com/"
        )

        assert source.azure_endpoint == "https://test.openai.azure.com"

    def test_init_multiple_trailing_slashes(self):
        """Test that multiple trailing slashes are stripped"""
        source = AzureOpenAIModelSource(
            api_key="key",
            azure_endpoint="https://test.openai.azure.com///"
        )

        assert source.azure_endpoint == "https://test.openai.azure.com"

    def test_init_no_api_key(self):
        """Test initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            source = AzureOpenAIModelSource(
                azure_endpoint="https://test.openai.azure.com"
            )

            assert source.api_key is None
            assert source.azure_endpoint == "https://test.openai.azure.com"

    def test_init_no_endpoint(self):
        """Test initialization without endpoint"""
        with patch.dict(os.environ, {}, clear=True):
            source = AzureOpenAIModelSource(api_key="key")

            assert source.api_key == "key"
            assert source.azure_endpoint == ""

    def test_init_params_override_env(self):
        """Test that explicit params override environment variables"""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com"
        }):
            source = AzureOpenAIModelSource(
                api_key="param-key",
                azure_endpoint="https://param.openai.azure.com"
            )

            assert source.api_key == "param-key"
            assert source.azure_endpoint == "https://param.openai.azure.com"


class TestAzureOpenAIDiscover:
    """Test discover method"""

    @pytest.mark.asyncio
    async def test_discover_success(self):
        """Test successful model discovery"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "gpt-4",
                    "capabilities": {"chat_completion": True}
                },
                {
                    "id": "gpt-35-turbo",
                    "capabilities": {"chat_completion": True}
                }
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert len(specs) == 2
            assert all(spec.provider == Provider.AZURE_OPENAI.value for spec in specs)
            assert specs[0].name == "gpt-4"
            assert specs[0].family == "gpt-4"
            assert specs[1].name == "gpt-35-turbo"
            # Azure uses gpt-35 which doesn't match gpt-3.5 pattern
            assert specs[1].family is None

    @pytest.mark.asyncio
    async def test_discover_no_api_key(self):
        """Test discovery without API key"""
        source = AzureOpenAIModelSource(
            azure_endpoint="https://test.openai.azure.com"
        )

        specs = await source.discover()

        assert specs == []

    @pytest.mark.asyncio
    async def test_discover_no_endpoint(self):
        """Test discovery without endpoint"""
        source = AzureOpenAIModelSource(api_key="test-key")

        specs = await source.discover()

        assert specs == []

    @pytest.mark.asyncio
    async def test_discover_http_error(self):
        """Test discovery with HTTP error"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(side_effect=httpx.HTTPError("API Error"))
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert specs == []

    @pytest.mark.asyncio
    async def test_discover_connect_error(self):
        """Test discovery with connection error"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert specs == []

    @pytest.mark.asyncio
    async def test_discover_filters_non_chat_models(self):
        """Test that non-chat models are filtered out"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "gpt-4",
                    "capabilities": {"chat_completion": True}
                },
                {
                    "id": "text-embedding-ada-002",
                    "capabilities": {"chat_completion": False}
                },
                {
                    "id": "whisper-1",
                    "capabilities": {"chat_completion": True}
                }
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            # Only gpt-4 should be included
            assert len(specs) == 1
            assert specs[0].name == "gpt-4"

    @pytest.mark.asyncio
    async def test_discover_skips_empty_id(self):
        """Test that models with empty IDs are skipped"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "",
                    "capabilities": {"chat_completion": True}
                },
                {
                    "id": "gpt-4",
                    "capabilities": {"chat_completion": True}
                }
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert len(specs) == 1
            assert specs[0].name == "gpt-4"

    @pytest.mark.asyncio
    async def test_discover_handles_missing_data(self):
        """Test discovery with missing data field"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response = Mock()
        mock_response.json.return_value = {}

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert specs == []

    @pytest.mark.asyncio
    async def test_discover_handles_missing_capabilities(self):
        """Test discovery with missing capabilities field"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "gpt-4"
                    # No capabilities field
                }
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            # Should be skipped due to missing chat_completion capability
            assert specs == []

    @pytest.mark.asyncio
    async def test_discover_deduplicates_models(self):
        """Test that duplicate models are deduplicated"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "gpt-4",
                    "capabilities": {"chat_completion": True}
                },
                {
                    "id": "gpt-4",  # Duplicate
                    "capabilities": {"chat_completion": True}
                }
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            # Should only have one gpt-4
            assert len(specs) == 1
            assert specs[0].name == "gpt-4"

    @pytest.mark.asyncio
    async def test_discover_uses_correct_api_params(self):
        """Test that correct API parameters are used"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            api_version="2024-06-01",
            timeout=20.0
        )

        mock_response = Mock()
        mock_response.json.return_value = {"data": []}

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            await source.discover()

            # Verify AsyncClient was created with correct timeout
            mock_client.assert_called_once_with(timeout=20.0)

            # Verify get was called with correct params
            mock_instance.get.assert_called_once()
            call_args = mock_instance.get.call_args
            assert call_args[0][0] == "https://test.openai.azure.com/openai/models"
            assert call_args[1]["headers"] == {"api-key": "test-key"}
            assert call_args[1]["params"] == {"api-version": "2024-06-01"}


class TestIsChatModel:
    """Test _is_chat_model method"""

    def test_is_chat_model_gpt4(self):
        """Test GPT-4 is identified as chat model"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("gpt-4") is True
        assert source._is_chat_model("gpt-4-turbo") is True
        assert source._is_chat_model("gpt-4o") is True

    def test_is_chat_model_gpt35(self):
        """Test GPT-3.5/GPT-35 is identified as chat model"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        # Both gpt-3 patterns should match (gpt-3.5 and gpt-35)
        assert source._is_chat_model("gpt-3.5-turbo") is True
        assert source._is_chat_model("gpt-35-turbo") is True

    def test_is_chat_model_gpt5(self):
        """Test GPT-5 is identified as chat model"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("gpt-5") is True
        assert source._is_chat_model("gpt-5-turbo") is True

    def test_is_chat_model_o1(self):
        """Test O1 models are identified as chat models"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("o1-preview") is True
        assert source._is_chat_model("o1-mini") is True

    def test_is_chat_model_o3(self):
        """Test O3 models are identified as chat models"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("o3-mini") is True
        assert source._is_chat_model("o3") is True

    def test_is_chat_model_embedding(self):
        """Test embedding models are filtered out"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("text-embedding-ada-002") is False
        assert source._is_chat_model("text-embedding-3-small") is False

    def test_is_chat_model_whisper(self):
        """Test Whisper models are filtered out"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("whisper-1") is False

    def test_is_chat_model_tts(self):
        """Test TTS models are filtered out"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("tts-1") is False
        assert source._is_chat_model("tts-1-hd") is False

    def test_is_chat_model_dall_e(self):
        """Test DALL-E models are filtered out"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("dall-e-3") is False
        assert source._is_chat_model("dall-e-2") is False

    def test_is_chat_model_moderation(self):
        """Test moderation models are filtered out"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("text-moderation-latest") is False

    def test_is_chat_model_legacy_completion(self):
        """Test legacy completion models are filtered out"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("davinci-002") is False
        assert source._is_chat_model("babbage-002") is False
        assert source._is_chat_model("text-ada-001") is False
        assert source._is_chat_model("text-curie-001") is False

    def test_is_chat_model_case_insensitive(self):
        """Test case insensitivity"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._is_chat_model("GPT-4") is True
        assert source._is_chat_model("GPT-35-TURBO") is True  # Matches gpt-3 pattern
        assert source._is_chat_model("GPT-3.5-TURBO") is True  # Also matches gpt-3 pattern
        assert source._is_chat_model("TEXT-EMBEDDING-ADA-002") is False


class TestExtractFamily:
    """Test _extract_family method"""

    def test_extract_family_gpt5(self):
        """Test GPT-5 family extraction"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._extract_family("gpt-5") == "gpt-5"
        assert source._extract_family("gpt-5-turbo") == "gpt-5"
        assert source._extract_family("gpt5") == "gpt-5"

    def test_extract_family_gpt4o(self):
        """Test GPT-4o family extraction"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._extract_family("gpt-4o") == "gpt-4o"
        assert source._extract_family("gpt-4o-mini") == "gpt-4o"
        assert source._extract_family("gpt4o") == "gpt-4o"

    def test_extract_family_gpt4(self):
        """Test GPT-4 family extraction"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._extract_family("gpt-4") == "gpt-4"
        assert source._extract_family("gpt-4-turbo") == "gpt-4"
        assert source._extract_family("gpt4") == "gpt-4"

    def test_extract_family_gpt35(self):
        """Test GPT-3.5 family extraction"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        # Only gpt-3.5 (with dot) is recognized
        assert source._extract_family("gpt-3.5-turbo") == "gpt-3.5"
        assert source._extract_family("gpt-3.5") == "gpt-3.5"
        # Azure's gpt-35 (without dot) is not recognized
        assert source._extract_family("gpt-35-turbo") is None

    def test_extract_family_o1(self):
        """Test O1 family extraction"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._extract_family("o1-preview") == "o1"
        assert source._extract_family("o1-mini") == "o1"
        assert source._extract_family("custom-o1-deployment") == "o1"
        assert source._extract_family("deployment-o1") == "o1"

    def test_extract_family_o3(self):
        """Test O3 family extraction"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._extract_family("o3-mini") == "o3"
        assert source._extract_family("o3") == "o3"
        assert source._extract_family("custom-o3-deployment") == "o3"
        assert source._extract_family("deployment-o3") == "o3"

    def test_extract_family_unknown(self):
        """Test unknown model returns None"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._extract_family("unknown-model") is None
        assert source._extract_family("custom-deployment") is None

    def test_extract_family_case_insensitive(self):
        """Test case insensitivity"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        assert source._extract_family("GPT-4") == "gpt-4"
        assert source._extract_family("GPT-4O") == "gpt-4o"
        assert source._extract_family("O1-PREVIEW") == "o1"

    def test_extract_family_priority(self):
        """Test that more specific families take priority"""
        source = AzureOpenAIModelSource(api_key="key", azure_endpoint="endpoint")

        # gpt-4o should be detected before gpt-4
        assert source._extract_family("gpt-4o-mini") == "gpt-4o"
        # gpt-5 should be detected first
        assert source._extract_family("gpt-5-turbo") == "gpt-5"


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_discovery_workflow(self):
        """Test full discovery workflow with realistic data"""
        source = AzureOpenAIModelSource(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "capabilities": {"chat_completion": True}},
                {"id": "gpt-4-turbo", "capabilities": {"chat_completion": True}},
                {"id": "gpt-4o", "capabilities": {"chat_completion": True}},
                {"id": "gpt-35-turbo", "capabilities": {"chat_completion": True}},
                {"id": "text-embedding-ada-002", "capabilities": {"embeddings": True}},
                {"id": "whisper-1", "capabilities": {"transcription": True}},
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            # Should only get chat models
            assert len(specs) == 4
            model_names = {spec.name for spec in specs}
            assert model_names == {"gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-35-turbo"}

            # Check families (gpt-35 doesn't match gpt-3.5 pattern so it's None)
            families = {spec.family for spec in specs}
            assert "gpt-4" in families
            assert "gpt-4o" in families
            # gpt-35-turbo has None family since Azure uses gpt-35 not gpt-3.5
            assert None in families


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
