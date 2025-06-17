"""
Test suite for the LLM client factory and provider implementations.
"""

import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from chuk_llm.llm.client import get_client, _import_string, _supports_param, _constructor_kwargs
from chuk_llm.configuration.unified_config import get_config
from chuk_llm.llm.providers.openai_client import OpenAILLMClient
from chuk_llm.llm.core.base import BaseLLMClient


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

    def test_get_client_with_model_override(self):
        """Test that model parameter overrides config."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            client = get_client(provider="openai", model="custom-model")
            
            # Check that model was passed to constructor
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs.get("model") == "custom-model"

    def test_get_client_with_api_key_override(self):
        """Test that api_key parameter overrides config."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            client = get_client(provider="openai", api_key="custom-key")
            
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs.get("api_key") == "custom-key"

    def test_get_client_with_api_base_override(self):
        """Test that api_base parameter overrides config."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            client = get_client(provider="openai", api_base="custom-base")
            
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs.get("api_base") == "custom-base"

    def test_get_client_uses_environment_variables(self):
        """Test that get_client picks up environment variables."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                client = get_client(provider="openai")
                
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs.get("api_key") == "env-key"

    def test_get_client_parameter_precedence(self):
        """Test that function parameters take precedence over env vars."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                client = get_client(
                    provider="openai", 
                    api_key="param-key"
                )
                
                # Parameter should win
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs.get("api_key") == "param-key"

    def test_get_client_unknown_provider(self):
        """Test that get_client raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_client(provider="nonexistent_provider")

    def test_get_client_missing_client_class(self):
        """Test that get_client raises error when client class is missing."""
        # Mock the config to return a provider with no client class
        mock_provider = MagicMock()
        mock_provider.client_class = ""  # Empty string instead of None
        mock_provider.default_model = "test-model"
        mock_provider.api_base = None
        mock_provider.api_key_env = "TEST_API_KEY"
        mock_provider.api_key_fallback_env = None
        mock_provider.name = "test_provider"
        mock_provider.extra = {}
        
        # Patch at the module level where get_config is imported
        with patch("chuk_llm.llm.client.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "fake-api-key"  # Provide API key to pass validation
            mock_get_config.return_value = mock_config
            
            # Mock ConfigValidator to pass validation
            with patch("chuk_llm.llm.client.ConfigValidator") as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])
                
                with pytest.raises(ValueError, match="No client class configured"):
                    get_client(provider="test_provider")

    def test_get_client_client_init_error(self):
        """Test that get_client handles client initialization errors."""
        # Mock the config to return a valid provider but mock the client to fail
        mock_provider = MagicMock()
        mock_provider.client_class = "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
        mock_provider.default_model = "test-model"
        mock_provider.api_base = None
        mock_provider.api_key_env = "OPENAI_API_KEY"
        mock_provider.api_key_fallback_env = None
        mock_provider.name = "openai"
        mock_provider.extra = {}
        
        with patch("chuk_llm.llm.client.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "fake-api-key"  # Provide API key to pass validation
            mock_get_config.return_value = mock_config
            
            # Mock ConfigValidator to pass validation
            with patch("chuk_llm.llm.client.ConfigValidator") as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])
                
                with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
                    mock_openai.side_effect = Exception("Client init error")
                    
                    with pytest.raises(ValueError, match="Failed to create .* client"):
                        get_client(provider="openai")

    def test_get_client_invalid_import_path(self):
        """Test error handling for invalid client import paths."""
        # Mock the config to return invalid import path
        mock_provider = MagicMock()
        mock_provider.client_class = "invalid.path:Class"
        mock_provider.default_model = "test-model"
        mock_provider.api_base = None
        mock_provider.api_key_env = "TEST_API_KEY"
        mock_provider.api_key_fallback_env = None
        mock_provider.name = "test"
        mock_provider.extra = {}
        
        # Patch at the module level where get_config is imported
        with patch("chuk_llm.llm.client.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "fake-api-key"  # Provide API key to pass validation
            mock_get_config.return_value = mock_config
            
            # Mock ConfigValidator to pass validation
            with patch("chuk_llm.llm.client.ConfigValidator") as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])
                
                with pytest.raises(ValueError, match="Failed to import client class"):
                    get_client(provider="test")
                    

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
    async def test_create_completion_non_streaming(self):
        """Test that create_completion works in non-streaming mode."""
        with patch("chuk_llm.llm.providers.openai_client.openai") as mock_openai:
            # Mock the response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Hello, world!"
            mock_response.choices[0].message.tool_calls = None
            
            # Mock the async client
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.AsyncOpenAI.return_value = mock_async_client
            
            # Mock the sync client  
            mock_sync_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_sync_client

            client = get_client("openai", model="gpt-4o-mini")

            messages = [{"role": "user", "content": "Hello"}]
            result = await client.create_completion(messages, stream=False)

            assert result["response"] == "Hello, world!"
            assert result["tool_calls"] == []

    @pytest.mark.asyncio 
    async def test_create_completion_with_tools(self):
        """Test create_completion with tool calls."""
        with patch("chuk_llm.llm.providers.openai_client.openai") as mock_openai:
            # Mock tool call response
            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_function"
            mock_tool_call.function.arguments = '{"param": "value"}'
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]
            
            # Mock the async client
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.AsyncOpenAI.return_value = mock_async_client
            
            # Mock the sync client
            mock_sync_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_sync_client

            client = get_client("openai", model="gpt-4o-mini")

            tools = [{"type": "function", "function": {"name": "test_function"}}]
            messages = [{"role": "user", "content": "Test"}]
            result = await client.create_completion(messages, tools=tools, stream=False)

            assert result["response"] is None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "test_function"

    @pytest.mark.asyncio
    async def test_create_completion_streaming(self):
        """Test streaming mode of create_completion."""
        with patch("chuk_llm.llm.providers.openai_client.openai") as mock_openai:
            # Mock streaming response
            async def mock_stream():
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello", tool_calls=None))])
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content=" World", tool_calls=None))])
            
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_stream())
            mock_openai.AsyncOpenAI.return_value = mock_async_client
            
            mock_sync_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_sync_client

            client = get_client("openai", model="gpt-4o-mini")

            messages = [{"role": "user", "content": "Hello"}]
            stream = client.create_completion(messages, stream=True)

            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0]["response"] == "Hello"
            assert chunks[1]["response"] == " World"

    @pytest.mark.asyncio
    async def test_create_completion_error_handling(self):
        """Test error handling in create_completion."""
        with patch("chuk_llm.llm.providers.openai_client.openai") as mock_openai:
            # Mock error
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_openai.AsyncOpenAI.return_value = mock_async_client
            
            mock_sync_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_sync_client

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
    async def test_client_interface_compatibility(self):
        """Test that clients follow the expected interface."""
        with patch("chuk_llm.llm.providers.openai_client.openai"):
            client = get_client("openai", model="gpt-4o-mini")
            
            # Test that create_completion method exists and has correct signature
            assert hasattr(client, "create_completion")
            assert callable(client.create_completion)

    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                client = get_client(provider="openai")
                
                # Should have been called with the environment variable
                call_kwargs = mock_client.call_args.kwargs
                assert call_kwargs.get("api_key") == "test-key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])