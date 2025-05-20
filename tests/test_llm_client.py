# tests/test_llm_client.py
"""
Test suite for the LLM client factory and provider implementations.
"""

import pytest
import importlib
import os
from unittest.mock import patch, MagicMock, PropertyMock

from chuk_llm.llm_client import get_llm_client, _import_string, _supports_param, _constructor_kwargs
from chuk_llm.provider_config import ProviderConfig
from chuk_llm.providers.openai_client import OpenAILLMClient
from chuk_llm.providers.base import BaseLLMClient


class TestHelperFunctions:
    """Test helper functions in the llm_client module."""

    def test_import_string_valid(self):
        """Test _import_string with valid import path."""
        imported = _import_string("chuk_llm.providers.base:BaseLLMClient")
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
        ("openai", "chuk_llm.providers.openai_client.OpenAILLMClient"),
        ("anthropic", "chuk_llm.providers.anthropic_client.AnthropicLLMClient"),
        ("groq", "chuk_llm.providers.groq_client.GroqAILLMClient"),
        ("gemini", "chuk_llm.providers.gemini_client.GeminiLLMClient"),
        ("ollama", "chuk_llm.providers.ollama_client.OllamaLLMClient"),
    ])
    def test_get_client_for_provider(self, provider_name, client_class_path):
        """Test factory returns correct client type for each provider."""
        with patch(client_class_path) as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            
            with patch("chuk_llm.provider_config.ProviderConfig") as mock_config_class:
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
        with patch("chuk_llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch("chuk_llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.providers.openai_client.OpenAILLMClient",
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
        with patch("chuk_llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch("chuk_llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.providers.openai_client.OpenAILLMClient",
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
        with patch("chuk_llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            with patch("chuk_llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.providers.openai_client.OpenAILLMClient",
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
        with patch("chuk_llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            custom_config = MagicMock(spec=ProviderConfig)
            custom_config.get_provider_config.return_value = {
                "client": "chuk_llm.providers.openai_client.OpenAILLMClient",
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
        with patch("chuk_llm.provider_config.ProviderConfig") as mock_config_class:
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
        with patch("chuk_llm.providers.openai_client.OpenAILLMClient") as mock_openai:
            mock_openai.side_effect = Exception("Client init error")
            
            with patch("chuk_llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.providers.openai_client.OpenAILLMClient",
                    "model": "default-model"
                }
                
                with pytest.raises(ValueError, match="Error initialising 'openai' client"):
                    get_llm_client(provider="openai")

    def test_set_host_if_api_base_provided(self):
        """Test that set_host is called if api_base is provided and method exists."""
        with patch("chuk_llm.providers.ollama_client.OllamaLLMClient") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.set_host = MagicMock()
            mock_ollama.return_value = mock_instance
            
            with patch("chuk_llm.provider_config.ProviderConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config
                mock_config.get_provider_config.return_value = {
                    "client": "chuk_llm.providers.ollama_client.OllamaLLMClient",
                    "model": "default-model",
                    "api_base": "http://localhost:11434"
                }
                
                # Check if _supports_param correctly identifies that api_base isn't in the constructor
                with patch("chuk_llm.llm_client._supports_param", return_value=False):
                    client = get_llm_client(provider="ollama")
                    
                    mock_instance.set_host.assert_called_once_with("http://localhost:11434")


class TestOpenAIStyleMixin:
    """Test the OpenAIStyleMixin functionality."""
    
    def test_sanitize_tool_names(self):
        """Test tool name sanitization logic."""
        from chuk_llm.openai_style_mixin import OpenAIStyleMixin
        
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

    @patch("openai.OpenAI")
    async def test_create_completion(self, mock_openai):
        """Test that create_completion calls the OpenAI API correctly."""
        # Set up mock client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        
        # Mock run_in_executor to avoid real API calls
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            # Configure run_in_executor to return our mock response
            async def mock_run_in_executor(*args, **kwargs):
                return mock_response
                
            mock_loop.run_in_executor = MagicMock(side_effect=lambda *args: mock_run_in_executor())
            
            # Create client and call method
            client = OpenAILLMClient(model="test-model", api_key="test-key")
            result = await client.create_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Verify the result
            assert isinstance(result, dict)
            assert "response" in result
            assert result["response"] == "Test response"
            assert "tool_calls" in result
            assert isinstance(result["tool_calls"], list)
            assert len(result["tool_calls"]) == 0

    @patch("openai.OpenAI")
    async def test_create_completion_with_tools(self, mock_openai):
        """Test create_completion with tool calls."""
        # Set up mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Set up the mock response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"test": "value"}'
        
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        
        # Mock run_in_executor to avoid real API calls
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            # Configure run_in_executor to return our mock response
            async def mock_run_in_executor(*args, **kwargs):
                return mock_response
                
            mock_loop.run_in_executor = MagicMock(side_effect=lambda *args: mock_run_in_executor())
            
            # Create client and call method
            client = OpenAILLMClient(model="test-model", api_key="test-key")
            result = await client.create_completion(
                messages=[{"role": "user", "content": "Use tool"}],
                tools=[{"type": "function", "function": {"name": "test_tool"}}]
            )
            
            # Verify the result structure for tool calls
            assert isinstance(result, dict)
            assert "response" in result
            assert result["response"] is None  # None when tools are used
            assert "tool_calls" in result
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "test_tool"
            assert result["tool_calls"][0]["id"] == "call_1"

    @patch("openai.OpenAI")
    async def test_create_completion_streaming(self, mock_openai):
        """Test streaming mode of create_completion."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Set up streaming response generator
        class MockStreamingResponse:
            def __init__(self):
                self.chunks = [
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello", tool_calls=[]))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content=" World", tool_calls=[]))]),
                ]
            
            def __iter__(self):
                return iter(self.chunks)
        
        # Set up the chat completions create method to return streaming response
        mock_client.chat.completions.create = MagicMock(return_value=MockStreamingResponse())
        
        # Create client
        client = OpenAILLMClient(model="test-model", api_key="test-key")
        
        # Call streaming method (we'll need to mock _stream_from_blocking)
        with patch("chuk_llm.openai_style_mixin.OpenAIStyleMixin._stream_from_blocking") as mock_stream:
            # Set up the stream to return async iterator with chunks
            async def mock_aiter():
                yield {"response": "Hello", "tool_calls": []}
                yield {"response": " World", "tool_calls": []}
            
            mock_stream.return_value = mock_aiter()
            
            result = await client.create_completion(
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )
            
            # Test we get an async iterator
            assert hasattr(result, "__aiter__")
            
            # Collect the chunks
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0]["response"] == "Hello"
            assert chunks[1]["response"] == " World"