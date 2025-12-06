"""
Comprehensive tests for registry model sources.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.anthropic import AnthropicModelSource
from chuk_llm.registry.sources.deepseek import DeepSeekModelSource
from chuk_llm.registry.sources.env import EnvProviderSource
from chuk_llm.registry.sources.gemini import GeminiModelSource
from chuk_llm.registry.sources.groq import GroqModelSource
from chuk_llm.registry.sources.mistral import MistralModelSource
from chuk_llm.registry.sources.moonshot import MoonshotModelSource
from chuk_llm.registry.sources.ollama import OllamaSource
from chuk_llm.registry.sources.openai import OpenAIModelSource
from chuk_llm.registry.sources.openrouter import OpenRouterModelSource
from chuk_llm.registry.sources.perplexity import PerplexityModelSource
from chuk_llm.registry.sources.watsonx import WatsonxModelSource


class TestAnthropicModelSource:
    """Test AnthropicModelSource."""

    @pytest.mark.asyncio
    async def test_discover_with_api_key(self):
        """Test discovering models with API key."""
        source = AnthropicModelSource(api_key="test-key")
        models = await source.discover()

        assert len(models) > 0
        assert all(m.provider == Provider.ANTHROPIC.value for m in models)
        # Check for Claude models (now includes Claude 4.x and sonnet models)
        assert any("claude" in m.name and "sonnet" in m.name for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovering models without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = AnthropicModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_uses_env_var(self):
        """Test that discover uses ANTHROPIC_API_KEY env var."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            source = AnthropicModelSource()
            models = await source.discover()

            assert len(models) > 0


class TestDeepSeekModelSource:
    """Test DeepSeekModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful model discovery."""
        source = DeepSeekModelSource(api_key="test-key")

        mock_response_data = {
            "data": [
                {"id": "deepseek-chat"},
                {"id": "deepseek-v3-chat"},
                {"id": "deepseek-coder"},  # Should be filtered
                {"id": ""},  # Empty ID should be skipped
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

            # Should have 2 valid models (chat models, not coder, not empty)
            assert len(models) >= 1
            assert all(m.provider == Provider.DEEPSEEK.value for m in models)
            assert any("deepseek-chat" in m.name for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = DeepSeekModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_api_error(self):
        """Test discovery handles API errors gracefully."""
        source = DeepSeekModelSource(api_key="test-key")

        async def mock_get_error(*args, **kwargs):
            """Mock async get that raises an error."""
            import httpx

            raise httpx.HTTPError("API Error")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_is_chat_model(self):
        """Test _is_chat_model filtering logic."""
        source = DeepSeekModelSource(api_key="test-key")

        assert source._is_chat_model("deepseek-chat")
        assert source._is_chat_model("deepseek-v3")
        assert not source._is_chat_model("deepseek-coder")

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = DeepSeekModelSource(api_key="test-key")

        assert source._extract_family("deepseek-v3-chat") == "deepseek-v3"
        assert source._extract_family("deepseek-v2.5-chat") == "deepseek-v2.5"
        assert source._extract_family("deepseek-v2") == "deepseek-v2"
        assert source._extract_family("deepseek-chat") == "deepseek-chat"
        assert source._extract_family("deepseek-reasoner") == "deepseek-reasoner"
        assert source._extract_family("unknown-model") is None


class TestMoonshotModelSource:
    """Test MoonshotModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful model discovery."""
        source = MoonshotModelSource(api_key="test-key")

        mock_response_data = {
            "data": [
                {"id": "kimi-k2-turbo-preview"},
                {"id": "kimi-k2-0905-preview"},
                {"id": "kimi-k2-thinking-turbo"},
                {"id": "kimi-latest"},
                {"id": "moonshot-v1-8k"},
                {"id": "moonshot-v1-32k-vision-preview"},
                {"id": ""},  # Empty ID should be skipped
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

            # Should have all valid models
            assert len(models) >= 1
            assert all(m.provider == Provider.MOONSHOT.value for m in models)
            assert any("kimi-k2-turbo-preview" in m.name for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = MoonshotModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_api_error(self):
        """Test discovery handles API errors gracefully."""
        source = MoonshotModelSource(api_key="test-key")

        async def mock_get_error(*args, **kwargs):
            """Mock async get that raises an error."""
            import httpx

            raise httpx.HTTPError("API Error")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_is_chat_model(self):
        """Test _is_chat_model filtering logic."""
        source = MoonshotModelSource(api_key="test-key")

        # Kimi K2 models
        assert source._is_chat_model("kimi-k2-turbo-preview")
        assert source._is_chat_model("kimi-k2-0905-preview")
        assert source._is_chat_model("kimi-k2-thinking-turbo")
        assert source._is_chat_model("kimi-latest")

        # Moonshot v1 models
        assert source._is_chat_model("moonshot-v1-8k")
        assert source._is_chat_model("moonshot-v1-32k")
        assert source._is_chat_model("moonshot-v1-128k-vision-preview")

        # Non-Moonshot models
        assert not source._is_chat_model("gpt-4")
        assert not source._is_chat_model("claude-3")

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = MoonshotModelSource(api_key="test-key")

        # Kimi K2 families
        assert source._extract_family("kimi-k2-thinking-turbo") == "kimi-k2-thinking"
        assert source._extract_family("kimi-k2-thinking") == "kimi-k2-thinking"
        assert source._extract_family("kimi-k2-turbo-preview") == "kimi-k2-turbo"
        assert source._extract_family("kimi-k2-0905-preview") == "kimi-k2"
        assert source._extract_family("kimi-latest") == "kimi-latest"

        # Moonshot v1 families
        assert source._extract_family("moonshot-v1-8k") == "moonshot-v1"
        assert source._extract_family("moonshot-v1-32k") == "moonshot-v1"
        assert source._extract_family("moonshot-v1-128k") == "moonshot-v1"
        assert source._extract_family("moonshot-v1-auto") == "moonshot-v1-auto"
        assert source._extract_family("moonshot-v1-8k-vision-preview") == "moonshot-v1-vision"
        assert source._extract_family("moonshot-v1-32k-vision-preview") == "moonshot-v1-vision"

        # Unknown models
        assert source._extract_family("unknown-model") is None


class TestGroqModelSource:
    """Test GroqModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful model discovery."""
        source = GroqModelSource(api_key="test-key")

        mock_response_data = {
            "data": [
                {"id": "llama-3.3-70b-versatile"},
                {"id": "llama3.2-90b-text-preview"},
                {"id": "gemma-7b-it"},
                {"id": ""},  # Empty ID should be skipped
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

            assert len(models) == 3
            assert all(m.provider == Provider.GROQ.value for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = GroqModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = GroqModelSource(api_key="test-key")

        assert source._extract_family("llama-3.3-70b-versatile") == "llama-3.3"
        assert source._extract_family("llama3.2-90b-text") == "llama-3.2"
        assert source._extract_family("llama-3.1-8b-instant") == "llama-3.1"
        assert source._extract_family("llama3-70b") == "llama-3"
        assert source._extract_family("gemma-7b-it") == "gemma"
        assert source._extract_family("mixtral-8x7b") == "mixtral"
        assert source._extract_family("qwen-2-7b") == "qwen"
        assert source._extract_family("deepseek-r1") == "deepseek"
        assert source._extract_family("unknown-model") is None


class TestMistralModelSource:
    """Test MistralModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful model discovery via Mistral SDK."""
        import sys

        source = MistralModelSource(api_key="test-key")

        # Mock the Mistral SDK response
        mock_model1 = MagicMock()
        mock_model1.id = "mistral-large-latest"

        mock_model2 = MagicMock()
        mock_model2.id = "mistral-small-latest"

        mock_model3 = MagicMock()
        mock_model3.id = "mistral-embed"  # Should be filtered

        mock_response = MagicMock()
        mock_response.data = [mock_model1, mock_model2, mock_model3]

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_response

        # Create a mock mistralai module
        mock_mistralai = MagicMock()
        mock_mistralai.Mistral.return_value = mock_client

        # Replace the module in sys.modules
        original_mistralai = sys.modules.get("mistralai")
        sys.modules["mistralai"] = mock_mistralai

        try:
            models = await source.discover()

            assert len(models) == 2  # Embed model should be filtered
            assert all(m.provider == Provider.MISTRAL.value for m in models)
            assert any("mistral-large" in m.name for m in models)
        finally:
            if original_mistralai:
                sys.modules["mistralai"] = original_mistralai
            else:
                del sys.modules["mistralai"]

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = MistralModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_sdk_error(self):
        """Test discovery handles SDK errors gracefully."""
        import sys

        source = MistralModelSource(api_key="test-key")

        # Create a mock mistralai module that raises an error
        mock_mistralai = MagicMock()
        mock_mistralai.Mistral.side_effect = Exception("SDK Error")

        original_mistralai = sys.modules.get("mistralai")
        sys.modules["mistralai"] = mock_mistralai

        try:
            models = await source.discover()
            assert models == []
        finally:
            if original_mistralai:
                sys.modules["mistralai"] = original_mistralai
            else:
                del sys.modules["mistralai"]

    @pytest.mark.asyncio
    async def test_is_chat_model(self):
        """Test _is_chat_model filtering logic."""
        source = MistralModelSource(api_key="test-key")

        assert source._is_chat_model("mistral-large-latest")
        assert source._is_chat_model("mistral-small-latest")
        assert not source._is_chat_model("mistral-embed")
        assert not source._is_chat_model("mistral-moderation")
        assert not source._is_chat_model("pixtral-ocr")

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = MistralModelSource(api_key="test-key")

        assert source._extract_family("magistral-8b") == "magistral"
        assert source._extract_family("codestral-latest") == "codestral"
        assert source._extract_family("devstral-8b") == "devstral"
        assert source._extract_family("voxtral-latest") == "voxtral"
        assert source._extract_family("pixtral-12b") == "pixtral"
        assert source._extract_family("ministral-8b") == "ministral"
        assert source._extract_family("mistral-large-latest") == "mistral-large"
        assert source._extract_family("mistral-medium-latest") == "mistral-medium"
        assert source._extract_family("mistral-small-latest") == "mistral-small"
        assert source._extract_family("mistral-tiny") == "mistral-tiny"
        assert source._extract_family("open-mistral-7b") == "open-mistral"
        assert source._extract_family("unknown-model") is None


class TestOpenAIModelSource:
    """Test OpenAIModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful model discovery."""
        source = OpenAIModelSource(api_key="test-key")

        mock_response_data = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-mini"},
                {"id": "gpt-3.5-turbo"},
                {"id": "text-embedding-ada-002"},  # Should be filtered
                {"id": "whisper-1"},  # Should be filtered
                {"id": ""},  # Empty ID should be skipped
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

            assert len(models) == 3  # Only chat models
            assert all(m.provider == Provider.OPENAI.value for m in models)
            assert any("gpt-4o" in m.name for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = OpenAIModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_is_chat_model(self):
        """Test _is_chat_model filtering logic."""
        source = OpenAIModelSource(api_key="test-key")

        # Should include
        assert source._is_chat_model("gpt-4o")
        assert source._is_chat_model("gpt-4-turbo")
        assert source._is_chat_model("gpt-3.5-turbo")
        assert source._is_chat_model("o1-preview")
        assert source._is_chat_model("o3-mini")

        # Should exclude
        assert not source._is_chat_model("text-embedding-ada-002")
        assert not source._is_chat_model("text-moderation-latest")
        assert not source._is_chat_model("whisper-1")
        assert not source._is_chat_model("tts-1")
        assert not source._is_chat_model("dall-e-3")
        assert not source._is_chat_model("babbage-002")
        assert not source._is_chat_model("davinci-002")

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = OpenAIModelSource(api_key="test-key")

        assert source._extract_family("gpt-5-turbo") == "gpt-5"
        assert source._extract_family("gpt-4o-mini") == "gpt-4o"
        assert source._extract_family("gpt-4-turbo") == "gpt-4"
        assert source._extract_family("gpt-3.5-turbo") == "gpt-3.5"
        assert source._extract_family("o1-preview") == "o1"
        assert source._extract_family("o3-mini") == "o3"
        assert source._extract_family("unknown-model") is None


class TestOpenRouterModelSource:
    """Test OpenRouterModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful model discovery."""
        source = OpenRouterModelSource(api_key="test-key")

        mock_response_data = {
            "data": [
                {"id": "openai/gpt-4o"},
                {"id": "anthropic/claude-3-opus"},
                {"id": "meta-llama/llama-3-70b-instruct"},
                {"id": ""},  # Empty ID should be skipped
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

            assert len(models) == 3
            assert all(m.provider == Provider.OPENROUTER.value for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = OpenRouterModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = OpenRouterModelSource(api_key="test-key")

        assert source._extract_family("openai/gpt-4-turbo") == "gpt-4"
        assert source._extract_family("openai/gpt-3.5-turbo") == "gpt-3.5"
        assert source._extract_family("anthropic/claude-3-opus") == "claude-3-opus"
        assert source._extract_family("anthropic/claude-3-sonnet") == "claude-3-sonnet"
        assert source._extract_family("anthropic/claude-3-haiku") == "claude-3-haiku"
        assert source._extract_family("anthropic/claude-2") == "claude"
        assert source._extract_family("meta-llama/llama-3-70b") == "llama-3"
        assert source._extract_family("meta-llama/llama-2-13b") == "llama"
        assert source._extract_family("google/gemini-pro") == "gemini"
        assert source._extract_family("mistralai/mistral-7b") == "mistral"
        assert source._extract_family("mistralai/mixtral-8x7b") == "mixtral"
        assert source._extract_family("qwen/qwen-7b") == "qwen"
        assert source._extract_family("deepseek/deepseek-coder") == "deepseek"
        assert source._extract_family("google/gemma-7b") == "gemma"
        assert source._extract_family("microsoft/phi-3") == "phi"
        assert source._extract_family("unknown/model") is None


class TestPerplexityModelSource:
    """Test PerplexityModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test model discovery returns known static models."""
        source = PerplexityModelSource(api_key="test-key")

        # Perplexity now uses a static list of known models
        # since they don't provide a /models endpoint
        models = await source.discover()

        # Should return all known Perplexity models
        assert len(models) == 8
        assert all(m.provider == Provider.PERPLEXITY.value for m in models)

        # Check some specific models exist
        model_names = [m.name for m in models]
        assert "sonar" in model_names
        assert "sonar-pro" in model_names
        assert "sonar-reasoning" in model_names

    @pytest.mark.asyncio
    async def test_discover_api_error_fallback(self):
        """Test discovery falls back to known models on API error."""
        source = PerplexityModelSource(api_key="test-key")

        async def mock_get_error(*args, **kwargs):
            """Mock async get that raises an error."""
            import httpx

            raise httpx.HTTPError("API Error")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            # Should return known models
            assert len(models) > 0
            assert all(m.provider == Provider.PERPLEXITY.value for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns known models even without API key."""
        # Perplexity uses static models, so API key is not required for discovery
        with patch.dict("os.environ", {}, clear=True):
            source = PerplexityModelSource(api_key=None)
            models = await source.discover()

            # Should still return the static list of known models
            assert len(models) == 8
            assert all(m.provider == Provider.PERPLEXITY.value for m in models)

    @pytest.mark.asyncio
    async def test_get_known_models(self):
        """Test _get_known_models returns expected models."""
        source = PerplexityModelSource(api_key="test-key")
        models = source._get_known_models()

        assert len(models) > 0
        assert all(m.provider == Provider.PERPLEXITY.value for m in models)
        assert all("sonar" in m.name for m in models)

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = PerplexityModelSource(api_key="test-key")

        assert source._extract_family("llama-3.1-sonar-small") == "sonar"
        assert source._extract_family("llama-3.1-70b-instruct") == "llama"
        assert source._extract_family("unknown-model") is None


class TestWatsonxModelSource:
    """Test WatsonxModelSource."""

    @pytest.mark.asyncio
    async def test_discover_returns_known_models(self):
        """Test that discover returns known WatsonX models."""
        source = WatsonxModelSource()
        models = await source.discover()

        assert len(models) > 0
        assert all(m.provider == Provider.WATSONX.value for m in models)
        assert any("granite" in m.name for m in models)
        assert any("llama" in m.name for m in models)

    @pytest.mark.asyncio
    async def test_discover_includes_granite_models(self):
        """Test that Granite models are included."""
        source = WatsonxModelSource()
        models = await source.discover()

        granite_models = [m for m in models if "granite" in m.name]
        assert len(granite_models) > 0
        # Check for the new family naming (granite-2, granite-3, granite-4, granite-code, etc.)
        granite_families = {m.family for m in granite_models}
        assert any("granite" in family for family in granite_families if family)
        assert any(m.family == "granite-code" for m in granite_models)

    @pytest.mark.asyncio
    async def test_discover_includes_llama_models(self):
        """Test that Llama models are included."""
        source = WatsonxModelSource()
        models = await source.discover()

        llama_models = [m for m in models if "llama" in m.name]
        assert len(llama_models) > 0


class TestEnvProviderSource:
    """Test EnvProviderSource."""

    @pytest.mark.asyncio
    async def test_discover_with_openai_key(self):
        """Test discovery with OPENAI_API_KEY."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            source = EnvProviderSource(include_ollama=False)
            models = await source.discover()

            openai_models = [m for m in models if m.provider == Provider.OPENAI.value]
            assert len(openai_models) > 0

    @pytest.mark.asyncio
    async def test_discover_with_anthropic_key(self):
        """Test discovery with ANTHROPIC_API_KEY."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            source = EnvProviderSource(include_ollama=False)
            models = await source.discover()

            anthropic_models = [
                m for m in models if m.provider == Provider.ANTHROPIC.value
            ]
            assert len(anthropic_models) > 0

    @pytest.mark.asyncio
    async def test_discover_with_multiple_keys(self):
        """Test discovery with multiple API keys."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "ANTHROPIC_API_KEY": "test-key",
                "GROQ_API_KEY": "test-key",
            },
            clear=True,
        ):
            source = EnvProviderSource(include_ollama=False)
            models = await source.discover()

            providers = {m.provider for m in models}
            assert Provider.OPENAI.value in providers
            assert Provider.ANTHROPIC.value in providers
            assert Provider.GROQ.value in providers

    @pytest.mark.asyncio
    async def test_discover_includes_ollama_by_default(self):
        """Test that Ollama is included by default."""
        with patch.dict("os.environ", {}, clear=True):
            source = EnvProviderSource(include_ollama=True)
            models = await source.discover()

            ollama_models = [m for m in models if m.provider == Provider.OLLAMA.value]
            assert len(ollama_models) > 0

    @pytest.mark.asyncio
    async def test_discover_excludes_ollama_when_disabled(self):
        """Test that Ollama can be excluded."""
        with patch.dict("os.environ", {}, clear=True):
            source = EnvProviderSource(include_ollama=False)
            models = await source.discover()

            ollama_models = [m for m in models if m.provider == Provider.OLLAMA.value]
            assert len(ollama_models) == 0

    @pytest.mark.asyncio
    async def test_discover_with_fallback_env_var(self):
        """Test discovery with fallback env var (GOOGLE_API_KEY for Gemini)."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True):
            source = EnvProviderSource(include_ollama=False)
            models = await source.discover()

            gemini_models = [m for m in models if m.provider == Provider.GEMINI.value]
            assert len(gemini_models) > 0

    @pytest.mark.asyncio
    async def test_discover_primary_key_overrides_fallback(self):
        """Test that primary API key is preferred over fallback."""
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "primary-key", "GOOGLE_API_KEY": "fallback-key"},
            clear=True,
        ):
            source = EnvProviderSource(include_ollama=False)
            models = await source.discover()

            gemini_models = [m for m in models if m.provider == Provider.GEMINI.value]
            assert len(gemini_models) > 0


class TestGeminiModelSource:
    """Test GeminiModelSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful model discovery via Gemini API."""
        source = GeminiModelSource(api_key="test-key")

        mock_response_data = {
            "models": [
                {
                    "name": "models/gemini-1.5-pro",
                    "displayName": "Gemini 1.5 Pro",
                },
                {
                    "name": "models/gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash",
                },
                {
                    "name": "models/text-embedding-004",
                    "displayName": "Text Embedding",  # Not a gemini model
                },
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

            # Should filter to only gemini models (not embedding)
            assert len(models) == 2
            assert all(m.provider == Provider.GEMINI.value for m in models)
            assert any("gemini-1.5-pro" in m.name for m in models)

    @pytest.mark.asyncio
    async def test_discover_without_api_key(self):
        """Test discovery returns empty without API key."""
        with patch.dict("os.environ", {}, clear=True):
            source = GeminiModelSource(api_key=None)
            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_discover_api_error(self):
        """Test discovery handles API errors gracefully."""
        source = GeminiModelSource(api_key="test-key")

        async def mock_get_error(*args, **kwargs):
            """Mock async get that raises an error."""
            import httpx

            raise httpx.HTTPError("API Error")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_is_generative_model(self):
        """Test _is_generative_model filtering."""
        source = GeminiModelSource(api_key="test-key")

        assert source._is_generative_model("gemini-1.5-pro")
        assert source._is_generative_model("gemini-2.0-flash")
        assert not source._is_generative_model("text-embedding-004")
        assert not source._is_generative_model("aqa-model")

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = GeminiModelSource(api_key="test-key")

        # Test all Gemini model families in descending order
        assert source._extract_family("gemini-3-pro") == "gemini-3"
        assert source._extract_family("gemini-2.5-pro") == "gemini-2.5"
        assert source._extract_family("gemini-2.5-flash") == "gemini-2.5"
        assert source._extract_family("gemini-2.0-flash") == "gemini-2"
        assert source._extract_family("gemini-1.5-pro") == "gemini-1.5"
        assert source._extract_family("gemini-1.5-flash") == "gemini-1.5"
        assert source._extract_family("gemini-1-pro") == "gemini-1"
        assert source._extract_family("gemini-pro") == "gemini"
        assert source._extract_family("unknown-model") is None


class TestOllamaSource:
    """Test OllamaSource."""

    @pytest.mark.asyncio
    async def test_discover_successful(self):
        """Test successful Ollama model discovery."""
        source = OllamaSource(base_url="http://localhost:11434")

        mock_response_data = {
            "models": [
                {"name": "llama3.2:latest", "model": "llama3.2:latest"},
                {"name": "codellama:13b", "model": "codellama:13b"},
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
            assert all(m.provider == Provider.OLLAMA.value for m in models)

    @pytest.mark.asyncio
    async def test_discover_connection_error(self):
        """Test discovery handles connection errors gracefully."""
        source = OllamaSource(base_url="http://localhost:11434")

        async def mock_get_error(*args, **kwargs):
            """Mock async get that raises a connection error."""
            import httpx

            raise httpx.ConnectError("Connection refused")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get = mock_get_error
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            models = await source.discover()

            assert models == []

    @pytest.mark.asyncio
    async def test_extract_family(self):
        """Test _extract_family logic."""
        source = OllamaSource()

        # Test with model data dict format
        assert source._extract_family({"name": "llama3.2:latest"}) == "llama-3"
        assert source._extract_family({"name": "llama3.1:70b"}) == "llama-3"
        assert source._extract_family({"name": "llama2:13b"}) == "llama-2"
        assert source._extract_family({"name": "llama:7b"}) == "llama"
        assert source._extract_family({"name": "mistral:7b"}) == "mistral"
        assert source._extract_family({"name": "gemma:7b"}) == "gemma"
        assert source._extract_family({"name": "phi3:mini"}) == "phi"
        assert source._extract_family({"name": "qwen:7b"}) == "qwen"
        assert source._extract_family({"name": "unknown:latest"}) is None
