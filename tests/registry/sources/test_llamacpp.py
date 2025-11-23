"""
Comprehensive tests for LlamaCppModelSource.

Tests cover:
- Initialization with default and custom parameters
- Model discovery functionality
- Family extraction logic
- Error handling
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from chuk_llm.core.enums import Provider
from chuk_llm.registry.sources.llamacpp import LlamaCppModelSource


class TestLlamaCppModelSourceInit:
    """Test LlamaCppModelSource initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        source = LlamaCppModelSource()

        assert source.provider == Provider.LLAMA_CPP.value
        assert source.api_base == "http://localhost:8080"
        assert source.api_key is None
        assert source.timeout == 10.0

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base."""
        source = LlamaCppModelSource(api_base="http://localhost:9000")

        assert source.api_base == "http://localhost:9000"
        assert source.provider == Provider.LLAMA_CPP.value

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from api_base."""
        source = LlamaCppModelSource(api_base="http://localhost:8080/")

        assert source.api_base == "http://localhost:8080"

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        source = LlamaCppModelSource(api_key="test-key")

        assert source.api_key == "test-key"

    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"LLAMA_CPP_API_KEY": "env-key"}):
            source = LlamaCppModelSource()

            assert source.api_key == "env-key"

    def test_init_params_override_env(self):
        """Test that explicit params override environment variables."""
        with patch.dict(os.environ, {"LLAMA_CPP_API_KEY": "env-key"}):
            source = LlamaCppModelSource(api_key="param-key")

            assert source.api_key == "param-key"


class TestLlamaCppModelSourceDiscover:
    """Test model discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_success(self):
        """Test successful model discovery."""
        # LlamaCppModelSource doesn't require API key for local servers, but
        # OpenAICompatibleSource parent requires it. Use api_key="local" for local server
        source = LlamaCppModelSource(api_key="local")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "llama-3-8b-instruct"},
                {"id": "mistral-7b-v0.3"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert len(specs) == 2
            assert all(spec.provider == Provider.LLAMA_CPP.value for spec in specs)
            assert specs[0].name == "llama-3-8b-instruct"
            assert specs[0].family == "llama-3"
            assert specs[1].name == "mistral-7b-v0.3"
            assert specs[1].family == "mistral"

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty list without API key."""
        source = LlamaCppModelSource()

        # OpenAICompatibleSource requires API key, returns [] if None
        specs = await source.discover()

        assert specs == []

    @pytest.mark.asyncio
    async def test_discover_http_error(self):
        """Test discovery with HTTP error."""
        source = LlamaCppModelSource()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(side_effect=httpx.HTTPError("API Error"))
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert specs == []

    @pytest.mark.asyncio
    async def test_discover_connect_error(self):
        """Test discovery with connection error."""
        source = LlamaCppModelSource()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert specs == []

    @pytest.mark.asyncio
    async def test_discover_skips_empty_id(self):
        """Test that models with empty IDs are skipped."""
        source = LlamaCppModelSource(api_key="local")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": ""},
                {"id": "llama-3-8b"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert len(specs) == 1
            assert specs[0].name == "llama-3-8b"

    @pytest.mark.asyncio
    async def test_discover_handles_missing_data(self):
        """Test discovery with missing data field."""
        source = LlamaCppModelSource()

        mock_response = Mock()
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert specs == []

    @pytest.mark.asyncio
    async def test_discover_deduplicates_models(self):
        """Test that duplicate models are deduplicated."""
        source = LlamaCppModelSource(api_key="local")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "llama-3-8b"},
                {"id": "llama-3-8b"},  # Duplicate
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert len(specs) == 1
            assert specs[0].name == "llama-3-8b"

    @pytest.mark.asyncio
    async def test_discover_uses_correct_endpoint(self):
        """Test that correct endpoint is used."""
        source = LlamaCppModelSource(api_base="http://localhost:9000", api_key="local")

        mock_response = Mock()
        mock_response.json.return_value = {"data": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            await source.discover()

            # Verify get was called with correct endpoint
            mock_instance.get.assert_called_once()
            call_args = mock_instance.get.call_args
            assert call_args[0][0] == "http://localhost:9000/models"


class TestIsChatModel:
    """Test _is_chat_model method."""

    def test_is_chat_model_always_true(self):
        """Test that all models are considered chat models."""
        source = LlamaCppModelSource()

        # llama.cpp assumes all loaded models can be used for chat
        assert source._is_chat_model("llama-3-8b") is True
        assert source._is_chat_model("mistral-7b") is True
        assert source._is_chat_model("qwen-2.5") is True
        assert source._is_chat_model("any-model") is True


class TestExtractFamily:
    """Test _extract_family method."""

    def test_extract_family_llama3(self):
        """Test Llama 3 family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("llama-3-8b-instruct") == "llama-3"
        assert source._extract_family("llama3-8b") == "llama-3"
        assert source._extract_family("Llama-3.2-1B") == "llama-3"
        assert source._extract_family("meta-llama3-instruct") == "llama-3"

    def test_extract_family_llama2(self):
        """Test Llama 2 family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("llama-2-7b-chat") == "llama-2"
        assert source._extract_family("llama2-13b") == "llama-2"
        assert source._extract_family("Llama-2-70B") == "llama-2"

    def test_extract_family_llama_generic(self):
        """Test generic Llama family extraction."""
        source = LlamaCppModelSource()

        # Should match "llama" but not llama-2 or llama-3
        assert source._extract_family("llama-1-7b") == "llama"
        assert source._extract_family("llama-base") == "llama"

    def test_extract_family_mistral(self):
        """Test Mistral family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("mistral-7b-v0.3") == "mistral"
        assert source._extract_family("Mistral-7B-Instruct") == "mistral"
        assert source._extract_family("mistral-small") == "mistral"

    def test_extract_family_mixtral(self):
        """Test Mixtral family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("mixtral-8x7b") == "mixtral"
        assert source._extract_family("Mixtral-8x22B") == "mixtral"
        assert source._extract_family("mixtral-instruct") == "mixtral"

    def test_extract_family_qwen(self):
        """Test Qwen family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("qwen-2.5-7b") == "qwen"
        assert source._extract_family("Qwen2-72B") == "qwen"
        assert source._extract_family("qwen-instruct") == "qwen"

    def test_extract_family_deepseek(self):
        """Test DeepSeek family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("deepseek-v3") == "deepseek"
        assert source._extract_family("DeepSeek-Coder-7B") == "deepseek"
        assert source._extract_family("deepseek-chat") == "deepseek"

    def test_extract_family_gemma(self):
        """Test Gemma family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("gemma-2-9b") == "gemma"
        assert source._extract_family("Gemma-7B-IT") == "gemma"
        assert source._extract_family("gemma-instruct") == "gemma"

    def test_extract_family_phi(self):
        """Test Phi family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("phi-3-mini") == "phi"
        assert source._extract_family("Phi-2") == "phi"
        assert source._extract_family("phi-instruct") == "phi"

    def test_extract_family_command_r(self):
        """Test Command-R family extraction."""
        source = LlamaCppModelSource()

        assert source._extract_family("command-r-plus") == "command-r"
        assert source._extract_family("Command-R-35B") == "command-r"
        assert source._extract_family("command_r_v1") == "command-r"

    def test_extract_family_unknown(self):
        """Test unknown model returns None."""
        source = LlamaCppModelSource()

        assert source._extract_family("unknown-model") is None
        assert source._extract_family("custom-model-v1") is None
        assert source._extract_family("gpt-4") is None

    def test_extract_family_case_insensitive(self):
        """Test case insensitivity."""
        source = LlamaCppModelSource()

        assert source._extract_family("LLAMA-3-8B") == "llama-3"
        assert source._extract_family("MISTRAL-7B") == "mistral"
        assert source._extract_family("QWEN-2.5") == "qwen"

    def test_extract_family_priority(self):
        """Test that more specific families take priority."""
        source = LlamaCppModelSource()

        # llama-3 should be detected before generic llama
        assert source._extract_family("llama-3-8b") == "llama-3"
        # llama-2 should be detected before generic llama
        assert source._extract_family("llama-2-7b") == "llama-2"


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_discovery_workflow(self):
        """Test full discovery workflow with realistic data."""
        source = LlamaCppModelSource(api_key="local")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "llama-3-8b-instruct"},
                {"id": "llama-2-7b-chat"},
                {"id": "mistral-7b-v0.3"},
                {"id": "mixtral-8x7b"},
                {"id": "qwen-2.5-7b"},
                {"id": "gemma-2-9b"},
                {"id": "phi-3-mini"},
                {"id": "deepseek-coder-7b"},
                {"id": "command-r-plus"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            specs = await source.discover()

            assert len(specs) == 9
            model_names = {spec.name for spec in specs}
            assert "llama-3-8b-instruct" in model_names
            assert "mistral-7b-v0.3" in model_names

            # Check families
            families = {spec.family for spec in specs}
            assert "llama-3" in families
            assert "llama-2" in families
            assert "mistral" in families
            assert "mixtral" in families
            assert "qwen" in families
            assert "gemma" in families
            assert "phi" in families
            assert "deepseek" in families
            assert "command-r" in families


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
