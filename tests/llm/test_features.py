"""Comprehensive tests for features.py module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any, Dict, List, Optional
import asyncio

from chuk_llm.llm.features import (
    ProviderAdapter,
    UnifiedLLMInterface,
    quick_chat,
    multi_provider_chat,
    find_best_provider_for_task,
    validate_text_support,
)
from chuk_llm.configuration import Feature


class TestProviderAdapter:
    """Tests for ProviderAdapter class."""

    def test_supports_feature_with_valid_provider(self):
        """Test supports_feature with valid provider."""
        with patch('chuk_llm.llm.features.get_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.supports_feature.return_value = True
            mock_config.return_value = mock_cfg

            result = ProviderAdapter.supports_feature("openai", Feature.TEXT)
            assert result is True
            mock_cfg.supports_feature.assert_called_once_with("openai", Feature.TEXT, None)

    def test_supports_feature_with_model(self):
        """Test supports_feature with specific model."""
        with patch('chuk_llm.llm.features.get_config') as mock_config:
            mock_cfg = Mock()
            mock_cfg.supports_feature.return_value = True
            mock_config.return_value = mock_cfg

            result = ProviderAdapter.supports_feature("openai", Feature.VISION, "gpt-4")
            assert result is True
            mock_cfg.supports_feature.assert_called_once_with("openai", Feature.VISION, "gpt-4")

    def test_supports_feature_returns_false_on_error(self):
        """Test supports_feature returns False on exception."""
        with patch('chuk_llm.llm.features.get_config') as mock_config:
            mock_config.side_effect = Exception("Config error")

            result = ProviderAdapter.supports_feature("invalid", Feature.TEXT)
            assert result is False

    def test_validate_text_capability_success(self):
        """Test validate_text_capability when supported."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            result = ProviderAdapter.validate_text_capability("openai", "gpt-4")
            assert result is True

    def test_validate_text_capability_failure(self):
        """Test validate_text_capability when not supported."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=False):
            result = ProviderAdapter.validate_text_capability("invalid", "model")
            assert result is False

    def test_enable_json_mode_openai(self):
        """Test enabling JSON mode for OpenAI."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            kwargs = {}
            result = ProviderAdapter.enable_json_mode("openai", "gpt-4", kwargs)
            assert result["response_format"] == {"type": "json_object"}

    def test_enable_json_mode_gemini(self):
        """Test enabling JSON mode for Gemini."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            kwargs = {}
            result = ProviderAdapter.enable_json_mode("gemini", "gemini-pro", kwargs)
            assert result["generation_config"]["response_mime_type"] == "application/json"

    def test_enable_json_mode_anthropic_fallback(self):
        """Test enabling JSON mode for Anthropic (uses instruction fallback)."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=False):
            kwargs = {}
            result = ProviderAdapter.enable_json_mode("anthropic", "claude-3", kwargs)
            assert "_json_mode_instruction" in result
            assert "JSON" in result["_json_mode_instruction"]

    def test_enable_json_mode_groq(self):
        """Test enabling JSON mode for Groq."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            kwargs = {}
            result = ProviderAdapter.enable_json_mode("groq", "llama", kwargs)
            assert result["response_format"] == {"type": "json_object"}

    def test_enable_json_mode_mistral_large(self):
        """Test enabling JSON mode for Mistral large model."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            kwargs = {}
            result = ProviderAdapter.enable_json_mode("mistral", "mistral-large", kwargs)
            assert result["response_format"] == {"type": "json_object"}

    def test_enable_json_mode_mistral_small(self):
        """Test enabling JSON mode for Mistral small model (fallback)."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            kwargs = {}
            result = ProviderAdapter.enable_json_mode("mistral", "mistral-small", kwargs)
            assert "_json_mode_instruction" in result

    def test_enable_streaming(self):
        """Test enabling streaming."""
        kwargs = {}
        result = ProviderAdapter.enable_streaming("openai", "gpt-4", kwargs)
        assert result["stream"] is True

    def test_enable_streaming_preserves_kwargs(self):
        """Test that enable_streaming preserves other kwargs."""
        kwargs = {"temperature": 0.5}
        result = ProviderAdapter.enable_streaming("openai", "gpt-4", kwargs)
        assert result["stream"] is True
        assert result["temperature"] == 0.5

    def test_prepare_tools_empty(self):
        """Test preparing empty tools list."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            result = ProviderAdapter.prepare_tools("openai", "gpt-4", [])
            assert result is None

    def test_prepare_tools_none(self):
        """Test preparing None tools."""
        result = ProviderAdapter.prepare_tools("openai", "gpt-4", None)
        assert result is None

    def test_prepare_tools_openai(self):
        """Test preparing tools for OpenAI."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            tools = [{"type": "function", "function": {"name": "test"}}]
            result = ProviderAdapter.prepare_tools("openai", "gpt-4", tools)
            assert result == tools

    def test_prepare_tools_anthropic(self):
        """Test preparing tools for Anthropic."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            tools = [{"type": "function", "function": {"name": "test"}}]
            result = ProviderAdapter.prepare_tools("anthropic", "claude-3", tools)
            assert result == tools

    def test_prepare_tools_unsupported(self):
        """Test preparing tools for unsupported provider."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=False):
            tools = [{"type": "function", "function": {"name": "test"}}]
            result = ProviderAdapter.prepare_tools("unsupported", "model", tools)
            assert result is None

    def test_prepare_tools_gemini(self):
        """Test preparing tools for Gemini."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            tools = [{"type": "function", "function": {"name": "test"}}]
            result = ProviderAdapter.prepare_tools("gemini", "gemini-pro", tools)
            assert result == tools

    def test_prepare_tools_groq(self):
        """Test preparing tools for Groq."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            tools = [{"type": "function", "function": {"name": "test"}}]
            result = ProviderAdapter.prepare_tools("groq", "llama", tools)
            assert result == tools

    def test_set_temperature_openai(self):
        """Test setting temperature for OpenAI."""
        kwargs = {}
        result = ProviderAdapter.set_temperature("openai", 0.7, kwargs)
        assert result["temperature"] == 0.7

    def test_set_temperature_anthropic(self):
        """Test setting temperature for Anthropic."""
        kwargs = {}
        result = ProviderAdapter.set_temperature("anthropic", 0.5, kwargs)
        assert result["temperature"] == 0.5

    def test_set_temperature_gemini(self):
        """Test setting temperature for Gemini."""
        kwargs = {}
        result = ProviderAdapter.set_temperature("gemini", 0.8, kwargs)
        assert result["generation_config"]["temperature"] == 0.8

    def test_set_temperature_ollama(self):
        """Test setting temperature for Ollama."""
        kwargs = {}
        result = ProviderAdapter.set_temperature("ollama", 0.6, kwargs)
        assert result["options"]["temperature"] == 0.6

    def test_set_max_tokens_openai(self):
        """Test setting max tokens for OpenAI."""
        with patch('chuk_llm.llm.features.get_config') as mock_config:
            mock_config.side_effect = Exception("No config")
            kwargs = {}
            result = ProviderAdapter.set_max_tokens("openai", "gpt-4", 1000, kwargs)
            assert result["max_tokens"] == 1000

    def test_set_max_tokens_gemini(self):
        """Test setting max tokens for Gemini."""
        with patch('chuk_llm.llm.features.get_config') as mock_config:
            mock_config.side_effect = Exception("No config")
            kwargs = {}
            result = ProviderAdapter.set_max_tokens("gemini", "gemini-pro", 2000, kwargs)
            assert result["generation_config"]["max_output_tokens"] == 2000

    def test_set_max_tokens_ollama(self):
        """Test setting max tokens for Ollama."""
        with patch('chuk_llm.llm.features.get_config') as mock_config:
            mock_config.side_effect = Exception("No config")
            kwargs = {}
            result = ProviderAdapter.set_max_tokens("ollama", "llama", 500, kwargs)
            assert result["options"]["num_predict"] == 500

    def test_set_max_tokens_with_limit_check(self):
        """Test setting max tokens with model limit check."""
        with patch('chuk_llm.llm.features.get_config') as mock_config:
            mock_cfg = Mock()
            mock_provider = Mock()
            mock_model_caps = Mock()
            mock_model_caps.max_output_tokens = 2000

            mock_provider.get_model_capabilities.return_value = mock_model_caps
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            kwargs = {}
            result = ProviderAdapter.set_max_tokens("openai", "gpt-4", 5000, kwargs)
            # Should be capped at model limit
            assert result["max_tokens"] == 2000

    def test_add_system_message_openai(self):
        """Test adding system message for OpenAI."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            messages = [{"role": "user", "content": "Hello"}]
            result = ProviderAdapter.add_system_message(
                "openai", "gpt-4", messages, "You are a helpful assistant"
            )
            assert len(result) == 2
            assert result[0]["role"] == "system"
            assert result[0]["content"] == "You are a helpful assistant"

    def test_add_system_message_anthropic(self):
        """Test adding system message for Anthropic."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            messages = [{"role": "user", "content": "Hello"}]
            result = ProviderAdapter.add_system_message(
                "anthropic", "claude-3", messages, "You are helpful"
            )
            # Anthropic handles system messages in API call, not in messages
            assert result == messages

    def test_add_system_message_unsupported(self):
        """Test adding system message for unsupported provider."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=False):
            messages = [{"role": "user", "content": "Hello"}]
            result = ProviderAdapter.add_system_message(
                "unsupported", "model", messages, "System message"
            )
            # Should prepend as user message
            assert len(result) == 1
            assert "System" in result[0]["content"]

    def test_check_vision_support_no_vision_content(self):
        """Test check_vision_support with text-only messages."""
        messages = [{"role": "user", "content": "Hello"}]
        result = ProviderAdapter.check_vision_support("openai", "gpt-4", messages)
        assert result is True

    def test_check_vision_support_with_vision_supported(self):
        """Test check_vision_support with vision content and support."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=True):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}
                    ]
                }
            ]
            result = ProviderAdapter.check_vision_support("openai", "gpt-4-vision", messages)
            assert result is True

    def test_check_vision_support_with_vision_unsupported(self):
        """Test check_vision_support with vision content but no support."""
        with patch.object(ProviderAdapter, 'supports_feature', return_value=False):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "base64data"}
                    ]
                }
            ]
            result = ProviderAdapter.check_vision_support("openai", "gpt-3.5", messages)
            assert result is False


class TestUnifiedLLMInterface:
    """Tests for UnifiedLLMInterface class."""

    def test_init_with_provider_and_model(self):
        """Test initialization with provider and model."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_get_client.return_value = mock_client

            interface = UnifiedLLMInterface("openai", "gpt-4")
            assert interface.provider == "openai"
            assert interface.model == "gpt-4"

    def test_init_with_default_model(self):
        """Test initialization with default model."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-3.5-turbo"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_get_client.return_value = mock_client

            interface = UnifiedLLMInterface("openai")
            assert interface.model == "gpt-3.5-turbo"

    def test_init_validation_failure(self):
        """Test initialization fails on text capability validation."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=False):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "model"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            with pytest.raises(ValueError, match="doesn't support basic text completion"):
                UnifiedLLMInterface("invalid", "model")

    def test_get_capabilities(self):
        """Test getting capabilities."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True):

            # Setup for init
            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_model_caps = Mock()
            mock_model_caps.features = [Feature.TEXT, Feature.VISION, Feature.TOOLS]
            mock_model_caps.max_context_length = 8192
            mock_model_caps.max_output_tokens = 4096
            mock_provider.get_model_capabilities.return_value = mock_model_caps
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_get_client.return_value = mock_client

            interface = UnifiedLLMInterface("openai", "gpt-4")
            caps = interface.get_capabilities()

            assert caps["provider"] == "openai"
            assert caps["model"] == "gpt-4"
            assert caps["supports"]["text"] is True
            assert caps["supports"]["vision"] is True
            assert caps["supports"]["tools"] is True

    def test_get_capabilities_error(self):
        """Test get_capabilities with error."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True):

            # Setup for init - first call succeeds
            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_get_client.return_value = mock_client

            interface = UnifiedLLMInterface("openai", "gpt-4")

            # Now make get_config fail for get_capabilities
            mock_config.side_effect = Exception("Config error")

            caps = interface.get_capabilities()
            assert "error" in caps

    @pytest.mark.asyncio
    async def test_chat_basic(self):
        """Test basic chat."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch('chuk_llm.llm.features._ensure_pydantic_messages') as mock_ensure_msg, \
             patch('chuk_llm.llm.features._ensure_pydantic_tools') as mock_ensure_tools:

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            # create_completion is NOT async - it returns a dict or AsyncIterator
            mock_client.create_completion = Mock(return_value={"response": "Hello"})
            mock_get_client.return_value = mock_client

            mock_ensure_msg.return_value = [{"role": "user", "content": "Hi"}]
            mock_ensure_tools.return_value = None

            interface = UnifiedLLMInterface("openai", "gpt-4")
            # chat() is async and returns what create_completion returns
            result = await interface.chat([{"role": "user", "content": "Hi"}])

            assert result == {"response": "Hello"}

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self):
        """Test chat with temperature."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch.object(ProviderAdapter, 'set_temperature') as mock_set_temp, \
             patch('chuk_llm.llm.features._ensure_pydantic_messages'), \
             patch('chuk_llm.llm.features._ensure_pydantic_tools'):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_client.create_completion = AsyncMock(return_value={"response": "Hello"})
            mock_get_client.return_value = mock_client

            mock_set_temp.return_value = {"temperature": 0.7}

            interface = UnifiedLLMInterface("openai", "gpt-4")
            await interface.chat([{"role": "user", "content": "Hi"}], temperature=0.7)

            mock_set_temp.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_json_mode(self):
        """Test chat with JSON mode."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch.object(ProviderAdapter, 'enable_json_mode') as mock_json, \
             patch('chuk_llm.llm.features._ensure_pydantic_messages'), \
             patch('chuk_llm.llm.features._ensure_pydantic_tools'):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_client.create_completion = AsyncMock(return_value={"response": '{"key": "value"}'})
            mock_get_client.return_value = mock_client

            mock_json.return_value = {"response_format": {"type": "json_object"}}

            interface = UnifiedLLMInterface("openai", "gpt-4")
            await interface.chat([{"role": "user", "content": "Hi"}], json_mode=True)

            mock_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_simple_chat(self):
        """Test simple_chat."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch('chuk_llm.llm.features._ensure_pydantic_messages'), \
             patch('chuk_llm.llm.features._ensure_pydantic_tools'):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_client.create_completion = Mock(return_value={"response": "Hello there"})
            mock_get_client.return_value = mock_client

            interface = UnifiedLLMInterface("openai", "gpt-4")
            result = await interface.simple_chat("Hi")

            assert result == "Hello there"

    @pytest.mark.asyncio
    async def test_simple_chat_streaming(self):
        """Test simple_chat with streaming."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch('chuk_llm.llm.features._ensure_pydantic_messages'), \
             patch('chuk_llm.llm.features._ensure_pydantic_tools'):

            async def mock_stream():
                yield {"response": "Hello"}
                yield {"response": " there"}

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            # For streaming, create_completion returns an async iterator
            mock_client.create_completion = Mock(return_value=mock_stream())
            mock_get_client.return_value = mock_client

            interface = UnifiedLLMInterface("openai", "gpt-4")
            result = await interface.simple_chat("Hi", stream=True)

            assert result == "Hello there"


class TestQuickChat:
    """Tests for quick_chat helper function."""

    @pytest.mark.asyncio
    async def test_quick_chat(self):
        """Test quick_chat."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch('chuk_llm.llm.features._ensure_pydantic_messages'), \
             patch('chuk_llm.llm.features._ensure_pydantic_tools'):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_client.create_completion = Mock(return_value={"response": "Response"})
            mock_get_client.return_value = mock_client

            result = await quick_chat("openai", "gpt-4", "Hello")
            assert result == "Response"


class TestMultiProviderChat:
    """Tests for multi_provider_chat helper function."""

    @pytest.mark.asyncio
    async def test_multi_provider_chat_single(self):
        """Test multi_provider_chat with single provider."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch('chuk_llm.llm.features._ensure_pydantic_messages'), \
             patch('chuk_llm.llm.features._ensure_pydantic_tools'):

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_model_caps = Mock()
            mock_model_caps.features = [Feature.TEXT]
            mock_model_caps.max_context_length = 8192
            mock_model_caps.max_output_tokens = 4096
            mock_provider.get_model_capabilities.return_value = mock_model_caps
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_client.create_completion = Mock(return_value={"response": "Hi"})
            mock_get_client.return_value = mock_client

            result = await multi_provider_chat("Hello", ["openai"])
            assert "openai" in result
            assert result["openai"]["success"] is True

    @pytest.mark.asyncio
    async def test_multi_provider_chat_with_error(self):
        """Test multi_provider_chat with error."""
        with patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client:

            # Make get_config fail when trying to create UnifiedLLMInterface
            mock_config.side_effect = Exception("Config error")
            mock_get_client.side_effect = Exception("Client error")

            result = await multi_provider_chat("Hello", ["invalid"])
            assert "invalid" in result
            assert result["invalid"]["success"] is False


class TestFindBestProviderForTask:
    """Tests for find_best_provider_for_task helper function."""

    @pytest.mark.asyncio
    async def test_find_best_provider_basic(self):
        """Test find_best_provider_for_task basic."""
        with patch('chuk_llm.configuration.CapabilityChecker') as mock_checker, \
             patch('chuk_llm.llm.features.get_config') as mock_config, \
             patch('chuk_llm.llm.client.get_client') as mock_get_client, \
             patch.object(ProviderAdapter, 'validate_text_capability', return_value=True), \
             patch.object(ProviderAdapter, 'check_vision_support', return_value=True), \
             patch('chuk_llm.llm.features._ensure_pydantic_messages'), \
             patch('chuk_llm.llm.features._ensure_pydantic_tools'):

            mock_checker.get_best_provider_for_features.return_value = "openai"

            mock_cfg = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_model_caps = Mock()
            mock_model_caps.features = [Feature.TEXT]
            mock_model_caps.max_context_length = 8192
            mock_model_caps.max_output_tokens = 4096
            mock_provider.get_model_capabilities.return_value = mock_model_caps
            mock_cfg.get_provider.return_value = mock_provider
            mock_config.return_value = mock_cfg

            mock_client = Mock()
            mock_client.create_completion = Mock(return_value={"response": "Response"})
            mock_get_client.return_value = mock_client

            result = await find_best_provider_for_task("Hello", required_features=["text"])
            assert result is not None
            assert result["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_find_best_provider_no_provider(self):
        """Test find_best_provider_for_task with no suitable provider."""
        with patch('chuk_llm.configuration.CapabilityChecker') as mock_checker:
            mock_checker.get_best_provider_for_features.return_value = None

            result = await find_best_provider_for_task("Hello")
            assert result is None

    @pytest.mark.asyncio
    async def test_find_best_provider_with_error(self):
        """Test find_best_provider_for_task with error."""
        with patch('chuk_llm.configuration.CapabilityChecker') as mock_checker, \
             patch('chuk_llm.llm.features.get_config') as mock_config:

            mock_checker.get_best_provider_for_features.return_value = "openai"
            mock_config.side_effect = Exception("Error")

            result = await find_best_provider_for_task("Hello")
            assert result is None


class TestValidateTextSupport:
    """Tests for validate_text_support helper function."""

    def test_validate_text_support(self):
        """Test validate_text_support."""
        with patch.object(ProviderAdapter, 'validate_text_capability', return_value=True):
            result = validate_text_support("openai", "gpt-4")
            assert result is True

    def test_validate_text_support_failure(self):
        """Test validate_text_support failure."""
        with patch.object(ProviderAdapter, 'validate_text_capability', return_value=False):
            result = validate_text_support("invalid", "model")
            assert result is False
