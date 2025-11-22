"""
Tests for OpenAICompatibleSource.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_llm.registry.sources.openai_compatible import OpenAICompatibleSource


class TestOpenAICompatibleSource:
    """Test OpenAICompatibleSource."""

    @pytest.mark.asyncio
    async def test_discover_with_api_key(self):
        """Test discovery with explicit API key."""
        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key"
        )

        mock_response_data = {
            "data": [
                {"id": "model-1"},
                {"id": "model-2"},
            ]
        }

        async def mock_get(*args, **kwargs):
            """Mock async get method."""
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            assert len(models) == 2
            assert models[0].name == "model-1"
            assert models[0].provider == "test-provider"

    @pytest.mark.asyncio
    async def test_discover_with_api_key_env(self):
        """Test discovery with API key from environment."""
        os.environ["TEST_API_KEY"] = "env-key"

        try:
            source = OpenAICompatibleSource(
                provider="test-provider",
                api_base="https://api.test.com/v1",
                api_key_env="TEST_API_KEY"
            )

            mock_response_data = {"data": [{"id": "model-1"}]}

            async def mock_get(*args, **kwargs):
                mock_resp = MagicMock()
                mock_resp.json = MagicMock(return_value=mock_response_data)
                mock_resp.raise_for_status = MagicMock()
                return mock_resp

            with patch("httpx.AsyncClient") as MockClient:
                mock_client = MagicMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_client

                models = await source.discover()

                assert len(models) == 1
        finally:
            del os.environ["TEST_API_KEY"]

    @pytest.mark.asyncio
    async def test_discover_auto_detect_api_key_uppercase(self):
        """Test auto-detection of API key from provider name (uppercase pattern)."""
        os.environ["TESTPROVIDER_API_KEY"] = "auto-key"

        try:
            source = OpenAICompatibleSource(
                provider="testprovider",
                api_base="https://api.test.com/v1"
            )

            assert source.api_key == "auto-key"
        finally:
            del os.environ["TESTPROVIDER_API_KEY"]

    @pytest.mark.asyncio
    async def test_discover_auto_detect_api_key_with_dash(self):
        """Test auto-detection with provider name containing dash."""
        os.environ["TEST_PROVIDER_API_KEY"] = "dash-key"

        try:
            source = OpenAICompatibleSource(
                provider="test-provider",
                api_base="https://api.test.com/v1"
            )

            assert source.api_key == "dash-key"
        finally:
            del os.environ["TEST_PROVIDER_API_KEY"]

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test that discovery returns empty list without API key."""
        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1"
        )

        # Make sure no env vars are set
        source.api_key = None

        models = await source.discover()

        assert models == []

    @pytest.mark.asyncio
    async def test_discover_with_http_error(self):
        """Test handling of HTTP errors."""
        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key"
        )

        async def mock_get_error(*args, **kwargs):
            import httpx
            raise httpx.HTTPError("Connection failed")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_with_model_filter(self):
        """Test discovery with custom model filter."""
        def filter_func(model_id: str) -> bool:
            return "mini" not in model_id

        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key",
            model_filter=filter_func
        )

        mock_response_data = {
            "data": [
                {"id": "model-large"},
                {"id": "model-mini"},
                {"id": "model-ultra"},
            ]
        }

        async def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            # Should filter out model-mini
            assert len(models) == 2
            assert any(m.name == "model-large" for m in models)
            assert any(m.name == "model-ultra" for m in models)
            assert not any(m.name == "model-mini" for m in models)

    @pytest.mark.asyncio
    async def test_discover_with_family_extractor(self):
        """Test discovery with custom family extractor."""
        def extract_family(model_id: str) -> str | None:
            if "gpt" in model_id:
                return "gpt"
            return None

        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key",
            family_extractor=extract_family
        )

        mock_response_data = {
            "data": [
                {"id": "gpt-4"},
                {"id": "claude-3"},
            ]
        }

        async def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            # Check family extraction
            gpt_model = next((m for m in models if m.name == "gpt-4"), None)
            assert gpt_model is not None
            assert gpt_model.family == "gpt"

            claude_model = next((m for m in models if m.name == "claude-3"), None)
            assert claude_model is not None
            assert claude_model.family is None

    @pytest.mark.asyncio
    async def test_discover_with_invalid_response_format(self):
        """Test handling of invalid response format."""
        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key"
        )

        # Invalid response format
        mock_response_data = {"invalid": "format"}

        async def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_empty_model_ids(self):
        """Test that empty model IDs are skipped."""
        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key"
        )

        mock_response_data = {
            "data": [
                {"id": "valid-model"},
                {"id": ""},
                {"id": "another-valid"},
            ]
        }

        async def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            # Should skip empty ID
            assert len(models) == 2
            assert all(m.name for m in models)

    @pytest.mark.asyncio
    async def test_default_family_extraction(self):
        """Test default family extraction logic."""
        source = OpenAICompatibleSource(
            provider="test-provider",
            api_base="https://api.test.com/v1",
            api_key="test-key"
        )

        mock_response_data = {
            "data": [
                {"id": "gpt-4-turbo"},
                {"id": "gpt-3.5-turbo"},
                {"id": "llama-3.3-70b"},
                {"id": "llama-3.2-3b"},
                {"id": "llama-3.1-8b"},
                {"id": "llama-3-70b"},
                {"id": "llama-7b"},
                {"id": "mixtral-8x7b"},
                {"id": "mistral-7b"},
                {"id": "gemma-7b"},
                {"id": "qwen-7b"},
                {"id": "deepseek-v3"},
                {"id": "deepseek-v2"},
                {"id": "deepseek-chat"},
                {"id": "deepseek-reasoner"},
                {"id": "deepseek-coder"},
                {"id": "sonar-medium"},
                {"id": "claude-3"},
                {"id": "gemini-pro"},
            ]
        }

        async def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json = MagicMock(return_value=mock_response_data)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            # Verify family extraction for each model
            model_families = {m.name: m.family for m in models}

            assert model_families["gpt-4-turbo"] == "gpt-4"
            assert model_families["gpt-3.5-turbo"] == "gpt-3.5"
            assert model_families["llama-3.3-70b"] == "llama-3.3"
            assert model_families["llama-3.2-3b"] == "llama-3.2"
            assert model_families["llama-3.1-8b"] == "llama-3.1"
            assert model_families["llama-3-70b"] == "llama-3"
            assert model_families["llama-7b"] == "llama"
            assert model_families["mixtral-8x7b"] == "mixtral"
            assert model_families["mistral-7b"] == "mistral"
            assert model_families["gemma-7b"] == "gemma"
            assert model_families["qwen-7b"] == "qwen"
            assert model_families["deepseek-v3"] == "deepseek-v3"
            assert model_families["deepseek-v2"] == "deepseek-v2"
            assert model_families["deepseek-chat"] == "deepseek-chat"
            assert model_families["deepseek-reasoner"] == "deepseek-reasoner"
            assert model_families["deepseek-coder"] == "deepseek"
            assert model_families["sonar-medium"] == "sonar"
            assert model_families["claude-3"] == "claude"
            assert model_families["gemini-pro"] == "gemini"
