# tests/test_llm_client.py
"""
Test suite for the LLM client factory and provider implementations.
"""

import pytest
import importlib
import os
from unittest.mock import patch, MagicMock, PropertyMock

from chuk_llm.llm.llm_client import get_llm_client, _import_string, _supports_param, _constructor_kwargs
from chuk_llm.llm.provider_config import ProviderConfig
from chuk_llm.llm.providers.openai_client import OpenAILLMClient
from chuk_llm.llm.providers.base import BaseLLMClient


class TestHelperFunctions:
    """Test helper functions in the llm_client module."""

    def test_import_string_valid(self):
        """Test _import_string with valid import path."""
        imported = _import_string("chuk_llm.llm.providers.base:BaseLLMClient")
        assert imported is BaseLLMClient

    def test_import_string_invalid(self):
        """Test _import_string with invalid import path."""
        with pytest.raises(ImportError, match="Invalid import path"):
            _import_string("invalid_path")

    def test_import_string_nonexistent(self):
        """Test _import_string with non-existent module."""
        with pytest.raises(ImportError):
            _import_string("chuk_llm.nonexistent:Class")

    def test_supports_param(self):
        """Test _supports_param function."""
        class TestClass:
            def __init__(self, param1, param2=None):
                pass
        
        assert _supports_param(TestClass, "param1") is True
        assert _supports_param(TestClass, "param2") is True
        assert _supports_param(TestClass, "param3") is False

    def test_constructor_kwargs(self):
        """Test _constructor_kwargs function."""
        class TestClass:
            def __init__(self, model, api_key=None, api_base=None):
                pass
        
        cfg = {
            "model": "test-model",
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
        assert kwargs == {
            "model": "test-model",
            "api_key": "test-key",
            "api_base": "test-base"
        }
        # We don't pass extra_param because we're still filtering to known params


class TestGetLLMClient:
    """Test the get_llm_client factory function."""

    @pytest.mark.parametrize("provider_name, client_class_path", [
        ("openai", "chuk_llm.llm.providers.openai_client.OpenAILLMClient"),
        ("anthropic", "chuk_llm.llm.providers.anthropic_client.AnthropicLLMClient"),
        ("groq", "chuk_llm.llm.providers.groq_client.GroqAILLMClient"),
        ("gemini", "chuk_llm.llm.providers.gemini_client.GeminiLLMClient"),
        ("ollama", "chuk_llm.llm.providers.ollama_client.OllamaLLMClient"),
    ])
    def test_get_client_for_provider(self, provider_name, client_class_path):
        """Test factory returns correct client type for each provider."""
        with patch(client_class_path) as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            
            with patch("chuk_llm.llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": client_class_path,
                    "model": "default-model",
                    "api_key": None,
                    "api_base": None
                }
                
                client = get_llm_client(provider=provider_name)
                
                mock_client_class.assert_called_once()
                assert client == mock_instance

    def test_get_client_with_model_override(self):
        """Test that model parameter overrides config."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch("chuk_llm.llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.llm.providers.openai_client.OpenAILLMClient",
                    "model": "default-model",
                    "api_key": None,
                    "api_base": None
                }
                
                client = get_llm_client(provider="openai", model="custom-model")
                
                # Check only that the model was overridden
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs.get("model") == "custom-model"

    def test_get_client_with_api_key_override(self):
        """Test that api_key parameter overrides config."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch("chuk_llm.llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.llm.providers.openai_client.OpenAILLMClient",
                    "model": "default-model",
                    "api_key": None,
                    "api_base": None
                }
                
                client = get_llm_client(provider="openai", api_key="custom-key")
                
                # Only check that the custom API key was passed
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs.get("api_key") == "custom-key"

    def test_get_client_with_api_base_override(self):
        """Test that api_base parameter overrides config."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch("chuk_llm.llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.llm.providers.openai_client.OpenAILLMClient",
                    "model": "default-model",
                    "api_key": None,
                    "api_base": None
                }
                
                client = get_llm_client(provider="openai", api_base="custom-base")
                
                # Only check that the custom API base was passed
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs.get("api_base") == "custom-base"

    def test_get_client_with_custom_config(self):
        """Test that get_llm_client uses provided config."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            custom_config = MagicMock(spec=ProviderConfig)
            custom_config.get_provider_config.return_value = {
                "client": "chuk_llm.llm.providers.openai_client.OpenAILLMClient",
                "model": "custom-model",
                "api_key": "custom-key",
                "api_base": None
            }
            
            client = get_llm_client(provider="openai", config=custom_config)
            
            custom_config.get_provider_config.assert_called_once_with("openai")
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs.get("model") == "custom-model"
            assert call_kwargs.get("api_key") == "custom-key"

    def test_get_client_invalid_provider(self):
        """Test that get_llm_client raises ValueError for unknown provider."""
        with patch("chuk_llm.llm.provider_config.ProviderConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            
            # The real implementation returns an empty dict and then fails on missing 'client'
            mock_config.get_provider_config.return_value = {}
            
            with pytest.raises(ValueError, match="No 'client' class configured for provider"):
                get_llm_client(provider="nonexistent_provider")

    def test_get_client_missing_client_class(self):
        """Test that get_llm_client raises error when client class is missing."""
        # Create a real config object with a mock provider
        config = ProviderConfig()
        
        # Add a test provider with no client class
        config.providers = {
            "test_provider": {
                "model": "test-model"
                # No client key
            }
        }
        
        # This should raise ValueError
        with pytest.raises(ValueError):
            get_llm_client(provider="test_provider", config=config)

    def test_get_client_client_init_error(self):
        """Test that get_llm_client handles client initialization errors."""
        with patch("chuk_llm.llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_openai.side_effect = Exception("Client init error")
            
            with patch("chuk_llm.llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.llm.providers.openai_client.OpenAILLMClient",
                    "model": "default-model"
                }
                
                with pytest.raises(ValueError, match="Error initialising 'openai' client"):
                    get_llm_client(provider="openai")

    def test_set_host_if_api_base_provided(self):
        """Test that set_host is called if api_base is provided and method exists."""
        with patch("chuk_llm.llm.providers.ollama_client.OllamaLLMClient") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.set_host = MagicMock()
            mock_ollama.return_value = mock_instance
            
            with patch("chuk_llm.llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.llm.providers.ollama_client.OllamaLLMClient",
                    "model": "default-model",
                    "api_base": "http://localhost:11434"
                }
                
                # Check if _supports_param correctly identifies that api_base isn't in the constructor
                with patch("chuk_llm.llm.llm_client._supports_param", return_value=False):
                    client = get_llm_client(provider="ollama")
                    
                    mock_instance.set_host.assert_called_once_with("http://localhost:11434")


class TestOpenAIStyleMixin:
    """Test the OpenAIStyleMixin functionality."""
    
    def test_sanitize_tool_names(self):
        """Test tool name sanitization logic."""
        from chuk_llm.llm.openai_style_mixin import OpenAIStyleMixin
        
        # Test with no tools
        assert OpenAIStyleMixin._sanitize_tool_names(None) is None
        assert OpenAIStyleMixin._sanitize_tool_names([]) == []
        
        # Test with valid names (no change needed)
        tools = [
            {"function": {"name": "valid_name"}}
        ]
        sanitized = OpenAIStyleMixin._sanitize_tool_names(tools)
        assert sanitized[0]["function"]["name"] == "valid_name"
        
        # Test with invalid characters in name
        tools = [
            {"function": {"name": "invalid@name"}}
        ]
        sanitized = OpenAIStyleMixin._sanitize_tool_names(tools)
        assert sanitized[0]["function"]["name"] == "invalid_name"
        
        # Test with multiple tools, some invalid
        tools = [
            {"function": {"name": "valid_name"}},
            {"function": {"name": "invalid@name"}},
            {"function": {"name": "another-valid-name"}},
            {"function": {"name": "invalid$name+with%chars"}}
        ]
        sanitized = OpenAIStyleMixin._sanitize_tool_names(tools)
        assert sanitized[0]["function"]["name"] == "valid_name"
        assert sanitized[1]["function"]["name"] == "invalid_name"
        assert sanitized[2]["function"]["name"] == "another-valid-name"
        assert sanitized[3]["function"]["name"] == "invalid_name_with_chars"



@pytest.mark.asyncio
class TestOpenAIClient:
    """Test the OpenAI client implementation."""

    @patch("openai.AsyncOpenAI")  # Mock AsyncOpenAI instead of OpenAI
    async def test_create_completion(self, mock_async_openai):
        """Test that create_completion calls the OpenAI API correctly."""
        # Set up mock async client and response
        mock_async_client = MagicMock()
        mock_async_openai.return_value = mock_async_client
        
        # Create properly structured mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        
        # Mock the async create method
        from unittest.mock import AsyncMock
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create client and call method with new interface
        client = OpenAILLMClient(model="test-model", api_key="test-key")
        
        # Use new interface - get awaitable then await
        result_awaitable = client.create_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False
        )
        result = await result_awaitable
        
        # Verify the result
        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] == "Test response"
        assert "tool_calls" in result
        assert isinstance(result["tool_calls"], list)
        assert len(result["tool_calls"]) == 0

    @patch("openai.AsyncOpenAI")  # Mock AsyncOpenAI instead of OpenAI  
    async def test_create_completion_with_tools(self, mock_async_openai):
        """Test create_completion with tool calls."""
        # Set up mock async client
        mock_async_client = MagicMock()
        mock_async_openai.return_value = mock_async_client
        
        # Create properly structured mock response with tool calls
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        
        # Create mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"test": "value"}'
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        # Mock the async create method
        from unittest.mock import AsyncMock
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create client and call method with new interface
        client = OpenAILLMClient(model="test-model", api_key="test-key")
        result_awaitable = client.create_completion(
            messages=[{"role": "user", "content": "Use tool"}],
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
            stream=False
        )
        result = await result_awaitable
        
        # Verify the result structure for tool calls
        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] is None  # None when tools are used
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test_tool"
        assert result["tool_calls"][0]["id"] == "call_1"

    @patch("openai.AsyncOpenAI")  # Mock AsyncOpenAI instead of OpenAI
    async def test_create_completion_streaming(self, mock_async_openai):
        """Test streaming mode of create_completion."""
        mock_async_client = MagicMock()
        mock_async_openai.return_value = mock_async_client
        
        # Create mock streaming response
        async def mock_stream():
            # Create proper chunk objects with the right structure
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta = MagicMock()
            chunk1.choices[0].delta.content = "Hello"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.model = "gpt-4o-mini"
            chunk1.id = "chatcmpl-test"
            
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta = MagicMock()
            chunk2.choices[0].delta.content = " World"
            chunk2.choices[0].delta.tool_calls = None
            chunk2.model = "gpt-4o-mini"
            chunk2.id = "chatcmpl-test"
            
            yield chunk1
            yield chunk2
        
        # Mock the async create method to return stream
        from unittest.mock import AsyncMock
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        
        # Create client
        client = OpenAILLMClient(model="test-model", api_key="test-key")
        
        # Use new interface - get async generator directly (NO await)
        result = client.create_completion(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        # Should be async generator
        assert hasattr(result, "__aiter__")
        
        # Collect chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        # Verify chunks
        assert len(chunks) == 2
        assert chunks[0]["response"] == "Hello"
        assert chunks[1]["response"] == " World"
