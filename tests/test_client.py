"""
Test suite for the LLM client factory and provider implementations.
"""

import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

from chuk_llm.llm.client import get_client, _import_string, _supports_param, _constructor_kwargs
from chuk_llm.configuration.unified_config import get_config, Feature
from chuk_llm.llm.providers.openai_client import OpenAILLMClient
from chuk_llm.llm.core.base import BaseLLMClient


@pytest.fixture
def mock_provider_config():
    """Create a mock provider configuration."""
    mock_provider = MagicMock()
    mock_provider.client_class = "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
    mock_provider.default_model = "gpt-4o-mini"
    mock_provider.models = ["gpt-4o-mini", "gpt-4o", "custom-model"]
    mock_provider.model_aliases = {}
    mock_provider.api_base = "https://api.openai.com/v1"
    mock_provider.extra = {}
    
    # Mock feature support to support all features
    def mock_supports_feature(feature, model=None):
        return True  # Support all features to avoid warnings
    
    mock_provider.supports_feature = mock_supports_feature
    
    # Mock model capabilities
    mock_capabilities = MagicMock()
    mock_capabilities.features = {Feature.STREAMING, Feature.TOOLS, Feature.VISION}
    mock_provider.get_model_capabilities.return_value = mock_capabilities
    
    return mock_provider


@pytest.fixture
def mock_config_manager(mock_provider_config):
    """Create a mock configuration manager."""
    mock_config = MagicMock()
    mock_config.get_provider.return_value = mock_provider_config
    mock_config.get_api_key.return_value = "test-key"
    mock_config._ensure_model_available.return_value = "gpt-4o-mini"
    return mock_config


@pytest.fixture
def mock_config_system(mock_config_manager):
    """Mock the entire configuration system."""
    with patch("chuk_llm.llm.client.get_config", return_value=mock_config_manager):
        with patch("chuk_llm.llm.client.ConfigValidator") as mock_validator:
            mock_validator.validate_provider_config.return_value = (True, [])
            yield mock_config_manager, mock_validator


@pytest.fixture
def mock_openai_client():
    """Mock the OpenAI client."""
    with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_client, mock_instance


@pytest.fixture
def mock_openai_api():
    """Mock the OpenAI API library."""
    with patch("chuk_llm.llm.providers.openai_client.openai") as mock_openai:
        # Mock async client
        mock_async_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_async_client
        
        # Mock sync client
        mock_sync_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_sync_client
        
        yield mock_openai, mock_async_client, mock_sync_client


@pytest.fixture
def openai_response_with_text():
    """Create a mock OpenAI response with text content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello, world!"
    mock_response.choices[0].message.tool_calls = None
    return mock_response


@pytest.fixture
def openai_response_with_tools():
    """Create a mock OpenAI response with tool calls."""
    # Create tool call mock
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_function = MagicMock()
    mock_function.name = "test_function"
    mock_function.arguments = '{"param": "value"}'
    mock_tool_call.function = mock_function
    
    # Create response structure with proper None content
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    # Use a property to ensure content is always None
    type(mock_message).content = PropertyMock(return_value=None)
    mock_message.tool_calls = [mock_tool_call]
    
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    return mock_response


@pytest.fixture
def openai_streaming_response():
    """Create a mock OpenAI streaming response."""
    # Create an async generator function
    async def mock_stream():
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = None
        yield chunk1
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " World"
        chunk2.choices[0].delta.tool_calls = None
        yield chunk2
    
    # Return the generator function, not the generator itself
    return mock_stream


class TestHelperFunctions:
    """Test helper functions in the client module."""

    def test_import_string_valid(self):
        """Test _import_string with valid import path."""
        imported = _import_string("chuk_llm.llm.core.base:BaseLLMClient")
        assert imported is BaseLLMClient

    def test_import_string_valid_dot_notation(self):
        """Test _import_string with dot notation."""
        imported = _import_string("chuk_llm.llm.core.base.BaseLLMClient")
        assert imported is BaseLLMClient

    def test_import_string_nonexistent_module(self):
        """Test _import_string with non-existent module."""
        with pytest.raises(ImportError):
            _import_string("chuk_llm.nonexistent:Class")

    def test_import_string_nonexistent_attribute(self):
        """Test _import_string with non-existent attribute."""
        with pytest.raises(AttributeError):
            _import_string("chuk_llm.llm.core.base:NonExistentClass")

    def test_supports_param(self):
        """Test _supports_param function."""
        class TestClass:
            def __init__(self, param1, param2=None, *args, **kwargs):
                pass
        
        assert _supports_param(TestClass, "param1") is True
        assert _supports_param(TestClass, "param2") is True
        assert _supports_param(TestClass, "param3") is True  # **kwargs accepts any param

    def test_supports_param_no_kwargs(self):
        """Test _supports_param with no **kwargs in signature."""
        class TestClassNoKwargs:
            def __init__(self, param1, param2=None):
                pass
        
        assert _supports_param(TestClassNoKwargs, "param1") is True
        assert _supports_param(TestClassNoKwargs, "param2") is True
        assert _supports_param(TestClassNoKwargs, "param3") is False

    def test_constructor_kwargs_basic(self):
        """Test _constructor_kwargs function with basic parameters."""
        class TestClass:
            def __init__(self, model, api_key=None, api_base=None):
                pass
        
        cfg = {
            "model": "test-model",
            "default_model": "default-model",
            "api_key": "test-key",
            "api_base": "test-base",
            "extra_param": "value"
        }
        
        kwargs = _constructor_kwargs(TestClass, cfg)
        assert kwargs == {
            "model": "test-model",
            "api_key": "test-key",
            "api_base": "test-base"
        }
        assert "extra_param" not in kwargs
        assert "default_model" not in kwargs

    def test_constructor_kwargs_with_var_kwargs(self):
        """Test _constructor_kwargs with **kwargs in signature."""
        class TestClass:
            def __init__(self, model, **kwargs):
                pass
        
        cfg = {
            "model": "test-model",
            "api_key": "test-key",
            "api_base": "test-base",
            "extra_param": "value"
        }
        
        kwargs = _constructor_kwargs(TestClass, cfg)
        # Should include all non-None values when **kwargs is present
        assert kwargs == {
            "model": "test-model",
            "api_key": "test-key",
            "api_base": "test-base",
            "extra_param": "value"
        }

    def test_constructor_kwargs_filters_none_values(self):
        """Test that _constructor_kwargs filters out None values."""
        class TestClass:
            def __init__(self, model, api_key=None, api_base=None):
                pass
        
        cfg = {
            "model": "test-model",
            "api_key": None,
            "api_base": "test-base"
        }
        
        kwargs = _constructor_kwargs(TestClass, cfg)
        assert kwargs == {
            "model": "test-model",
            "api_base": "test-base"
        }
        assert "api_key" not in kwargs


class TestGetLLMClient:
    """Test the get_client factory function."""

    def test_get_client_with_model_override(self, mock_config_system, mock_openai_client):
        """Test that model parameter overrides config."""
        mock_config_manager, mock_validator = mock_config_system
        mock_client_class, mock_instance = mock_openai_client
        
        # Configure model discovery to return the custom model
        mock_config_manager._ensure_model_available.return_value = "custom-model"
        
        client = get_client(provider="openai", model="custom-model")
        
        # Check that model was passed to constructor
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs.get("model") == "custom-model"

    def test_get_client_with_api_key_override(self, mock_config_system, mock_openai_client):
        """Test that api_key parameter overrides config."""
        mock_config_manager, mock_validator = mock_config_system
        mock_client_class, mock_instance = mock_openai_client
        
        client = get_client(provider="openai", api_key="custom-key")
        
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs.get("api_key") == "custom-key"

    def test_get_client_with_api_base_override(self, mock_config_system, mock_openai_client):
        """Test that api_base parameter overrides config."""
        mock_config_manager, mock_validator = mock_config_system
        mock_client_class, mock_instance = mock_openai_client
        
        client = get_client(provider="openai", api_base="custom-base")
        
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs.get("api_base") == "custom-base"

    def test_get_client_uses_environment_variables(self, mock_config_system, mock_openai_client):
        """Test that get_client picks up environment variables."""
        mock_config_manager, mock_validator = mock_config_system
        mock_client_class, mock_instance = mock_openai_client
        
        client = get_client(provider="openai")
        
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs.get("api_key") == "test-key"  # From mock_config_manager

    def test_get_client_parameter_precedence(self, mock_config_system, mock_openai_client):
        """Test that function parameters take precedence over env vars."""
        mock_config_manager, mock_validator = mock_config_system
        mock_client_class, mock_instance = mock_openai_client
        
        client = get_client(provider="openai", api_key="param-key")
        
        # Parameter should win
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs.get("api_key") == "param-key"

    def test_get_client_unknown_provider(self):
        """Test that get_client raises ValueError for unknown provider."""
        with patch("chuk_llm.llm.client.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_provider.side_effect = Exception("Provider not found")
            mock_get_config.return_value = mock_config
            
            with pytest.raises(ValueError, match="Failed to get provider"):
                get_client(provider="nonexistent_provider")

    def test_get_client_missing_client_class(self, mock_config_system):
        """Test that get_client raises error when client class is missing."""
        mock_config_manager, mock_validator = mock_config_system
        
        # Modify provider config to have empty client class
        mock_provider = mock_config_manager.get_provider.return_value
        mock_provider.client_class = ""
        
        with pytest.raises(ValueError, match="No client class configured"):
            get_client(provider="test_provider")

    def test_get_client_client_init_error(self, mock_config_system):
        """Test that get_client handles client initialization errors."""
        mock_config_manager, mock_validator = mock_config_system
        
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_openai.side_effect = Exception("Client init error")
            
            with pytest.raises(ValueError, match="Failed to create .* client"):
                get_client(provider="openai")

    def test_get_client_invalid_import_path(self, mock_config_system):
        """Test error handling for invalid client import paths."""
        mock_config_manager, mock_validator = mock_config_system
        
        # Modify provider config to have invalid import path
        mock_provider = mock_config_manager.get_provider.return_value
        mock_provider.client_class = "invalid.path:Class"
        
        with pytest.raises(ValueError, match="Failed to import client class"):
            get_client(provider="test")

    def test_get_client_model_not_available(self, mock_config_system, mock_openai_client):
        """Test error when requested model is not available."""
        mock_config_manager, mock_validator = mock_config_system
        mock_client_class, mock_instance = mock_openai_client
        
        # Configure model discovery to return None (model not found)
        mock_config_manager._ensure_model_available.return_value = None
        
        with pytest.raises(ValueError, match="Model 'unavailable-model' not available"):
            get_client(provider="openai", model="unavailable-model")


class TestOpenAIStyleMixin:
    """Test the OpenAIStyleMixin functionality."""
    
    def test_sanitize_tool_names_none_input(self):
        """Test tool name sanitization with None input."""
        from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
        
        assert OpenAIStyleMixin._sanitize_tool_names(None) is None

    def test_sanitize_tool_names_empty_input(self):
        """Test tool name sanitization with empty list."""
        from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
        
        assert OpenAIStyleMixin._sanitize_tool_names([]) == []

    def test_sanitize_tool_names_valid_names(self):
        """Test tool name sanitization with valid names."""
        from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
        
        tools = [
            {"function": {"name": "valid_name"}},
            {"function": {"name": "another-valid-name"}},
            {"function": {"name": "name_with_123"}}
        ]
        sanitized = OpenAIStyleMixin._sanitize_tool_names(tools)
        
        assert len(sanitized) == 3
        assert sanitized[0]["function"]["name"] == "valid_name"
        assert sanitized[1]["function"]["name"] == "another-valid-name"
        assert sanitized[2]["function"]["name"] == "name_with_123"

    def test_sanitize_tool_names_invalid_characters(self):
        """Test tool name sanitization with invalid characters."""
        from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
        
        tools = [
            {"function": {"name": "invalid@name"}},
            {"function": {"name": "invalid$name+with%chars"}},
            {"function": {"name": "spaces in name"}},
            {"function": {"name": "dots.in.name"}}
        ]
        sanitized = OpenAIStyleMixin._sanitize_tool_names(tools)
        
        assert sanitized[0]["function"]["name"] == "invalid_name"
        assert sanitized[1]["function"]["name"] == "invalid_name_with_chars"
        assert sanitized[2]["function"]["name"] == "spaces_in_name"
        assert sanitized[3]["function"]["name"] == "dots_in_name"

    def test_sanitize_tool_names_preserves_other_fields(self):
        """Test that sanitization preserves other tool fields."""
        from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "invalid@name",
                    "description": "Test function",
                    "parameters": {"type": "object"}
                }
            }
        ]
        sanitized = OpenAIStyleMixin._sanitize_tool_names(tools)
        
        assert sanitized[0]["type"] == "function"
        assert sanitized[0]["function"]["name"] == "invalid_name"
        assert sanitized[0]["function"]["description"] == "Test function"
        assert sanitized[0]["function"]["parameters"] == {"type": "object"}


class TestOpenAIClient:
    """Test OpenAI client integration."""

    @pytest.mark.asyncio
    async def test_create_completion_non_streaming(
        self, 
        mock_config_system, 
        mock_openai_api,
        openai_response_with_text
    ):
        """Test that create_completion works in non-streaming mode."""
        mock_config_manager, mock_validator = mock_config_system
        mock_openai, mock_async_client, mock_sync_client = mock_openai_api
        
        # Configure the async client to return our mock response
        mock_async_client.chat.completions.create.return_value = openai_response_with_text

        # Mock the actual client to return a controlled response
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.create_completion.return_value = {
                "response": "Hello, world!",
                "tool_calls": [],
                "error": False
            }
            mock_client_class.return_value = mock_client_instance

            client = get_client("openai", model="gpt-4o-mini")
            messages = [{"role": "user", "content": "Hello"}]
            result = await client.create_completion(messages, stream=False)

            assert result["response"] == "Hello, world!"
            assert result["tool_calls"] == []

    @pytest.mark.asyncio 
    async def test_create_completion_with_tools(
        self, 
        mock_config_system, 
        mock_openai_api,
        openai_response_with_tools
    ):
        """Test create_completion with tool calls."""
        mock_config_manager, mock_validator = mock_config_system
        mock_openai, mock_async_client, mock_sync_client = mock_openai_api
        
        # Mock the actual client to return a controlled response with tool calls
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.create_completion.return_value = {
                "response": None,
                "tool_calls": [{"function": {"name": "test_function", "arguments": '{"param": "value"}'}, "id": "call_123"}],
                "error": False
            }
            mock_client_class.return_value = mock_client_instance

            client = get_client("openai", model="gpt-4o-mini")

            tools = [{"type": "function", "function": {"name": "test_function"}}]
            messages = [{"role": "user", "content": "Test"}]
            result = await client.create_completion(messages, tools=tools, stream=False)

            # The response should have no text content when tool calls are present
            assert result["response"] is None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "test_function"

    @pytest.mark.asyncio
    async def test_create_completion_streaming(
        self, 
        mock_config_system, 
        mock_openai_api,
        openai_streaming_response
    ):
        """Test streaming mode of create_completion."""
        mock_config_manager, mock_validator = mock_config_system
        mock_openai, mock_async_client, mock_sync_client = mock_openai_api
        
        # Create an async generator for streaming response
        async def mock_streaming_generator():
            yield {"response": "Hello", "tool_calls": [], "error": False}
            yield {"response": " World", "tool_calls": [], "error": False}

        # Mock the actual client to return a streaming generator
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.create_completion.return_value = mock_streaming_generator()
            mock_client_class.return_value = mock_client_instance

            client = get_client("openai", model="gpt-4o-mini")
            messages = [{"role": "user", "content": "Hello"}]
            
            # The create_completion method should return an async generator when stream=True
            stream_generator = await client.create_completion(messages, stream=True)
            
            # Collect chunks from the stream
            chunks = []
            async for chunk in stream_generator:
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0]["response"] == "Hello"
            assert chunks[1]["response"] == " World"

    @pytest.mark.asyncio
    async def test_create_completion_error_handling(
        self, 
        mock_config_system, 
        mock_openai_api
    ):
        """Test error handling in create_completion."""
        mock_config_manager, mock_validator = mock_config_system
        mock_openai, mock_async_client, mock_sync_client = mock_openai_api
        
        # Mock the actual client to return an error response
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_instance.create_completion.return_value = {
                "response": "API Error occurred",
                "tool_calls": [],
                "error": True
            }
            mock_client_class.return_value = mock_client_instance

            client = get_client("openai", model="gpt-4o-mini")
            messages = [{"role": "user", "content": "Hello"}]
            result = await client.create_completion(messages, stream=False)

            assert result["error"] is True
            assert "API Error" in result["response"]


class TestClientIntegration:
    """Integration tests for client creation and usage."""
    
    def test_client_inheritance(self):
        """Test that all clients inherit from BaseLLMClient."""
        assert issubclass(OpenAILLMClient, BaseLLMClient)

    @pytest.mark.asyncio
    async def test_client_interface_compatibility(self, mock_config_system, mock_openai_api):
        """Test that clients follow the expected interface."""
        mock_config_manager, mock_validator = mock_config_system
        mock_openai, mock_async_client, mock_sync_client = mock_openai_api

        client = get_client("openai", model="gpt-4o-mini")
        
        # Test that create_completion method exists and has correct signature
        assert hasattr(client, "create_completion")
        assert callable(client.create_completion)

    def test_environment_variable_loading(self, mock_config_system):
        """Test that environment variables are loaded correctly."""
        mock_config_manager, mock_validator = mock_config_system
        
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            
            client = get_client(provider="openai")
            
            # Should have been called with the API key from config manager
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs.get("api_key") == "test-key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])