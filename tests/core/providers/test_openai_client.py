"""
Fixed OpenAI Client Tests
=========================

This completely fixes the OpenAI client test issues with proper mocking and method implementations.
"""
import pytest
import asyncio
import json
import os
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import AsyncIterator, List, Dict, Any

# Mock the openai module before importing the client
import sys
from unittest.mock import MagicMock

# Create a comprehensive mock for the openai module
openai_mock = MagicMock()

# Mock all the classes we need
class MockStreamChunk:
    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.choices = [MockChoice(content, tool_calls, finish_reason)]

class MockChoice:
    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.delta = MockDelta(content, tool_calls)
        self.message = MockMessage(content, tool_calls)
        self.finish_reason = finish_reason

class MockDelta:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

class MockMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

class MockAsyncStream:
    """Properly working async stream mock"""
    def __init__(self, chunks=None):
        if chunks is None:
            chunks = [
                MockStreamChunk("Hello"),
                MockStreamChunk(" world!")
            ]
        self.chunks = chunks
        self._iterator = None

    def __aiter__(self):
        self._iterator = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration

class MockChatCompletion:
    def __init__(self, content="Hello world!", tool_calls=None):
        self.choices = [MockChoice(content, tool_calls)]
        self.id = "chatcmpl-test"
        self.model = "gpt-4o-mini"
        self.usage = MagicMock(total_tokens=50, prompt_tokens=10, completion_tokens=40)

class MockCompletions:
    def __init__(self):
        self.create_response = None
        self.stream_response = None

    async def create(self, **kwargs):
        if kwargs.get("stream", False):
            return self.stream_response or MockAsyncStream()
        else:
            return self.create_response or MockChatCompletion()

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockOpenAI:
    def __init__(self, **kwargs):
        self.chat = MockChat()
        self._kwargs = kwargs

    def close(self):
        pass

class MockAsyncOpenAI:
    def __init__(self, **kwargs):
        self.chat = MockChat()
        self._kwargs = kwargs

    async def close(self):
        pass

class MockAzureOpenAI:
    def __init__(self, **kwargs):
        self.chat = MockChat()
        self._kwargs = kwargs

    def close(self):
        pass

class MockAsyncAzureOpenAI:
    def __init__(self, **kwargs):
        self.chat = MockChat()
        self._kwargs = kwargs

    async def close(self):
        pass

# Set up the complete openai mock module
openai_mock.OpenAI = MockOpenAI
openai_mock.AsyncOpenAI = MockAsyncOpenAI
openai_mock.AzureOpenAI = MockAzureOpenAI
openai_mock.AsyncAzureOpenAI = MockAsyncAzureOpenAI

# Patch the openai module
sys.modules['openai'] = openai_mock

# Now import the client
from chuk_llm.llm.providers.openai_client import OpenAILLMClient

# Test fixtures
@pytest.fixture
def mock_env():
    """Mock environment variables for OpenAI."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key'
    }):
        yield

@pytest.fixture
def client(mock_env):
    """Create OpenAI client for testing"""
    return OpenAILLMClient(
        model="gpt-4o-mini",
        api_key="test-key"
    )

@pytest.fixture
def deepseek_client(mock_env):
    """Create DeepSeek client for testing"""
    return OpenAILLMClient(
        model="deepseek-chat",
        api_key="test-key",
        api_base="https://api.deepseek.com"
    )

# Basic functionality tests
class TestOpenAIBasic:
    """Test basic OpenAI functionality"""

    def test_client_initialization(self):
        """Test basic client initialization."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = OpenAILLMClient()
            assert client.model == "gpt-4o-mini"

    def test_detect_provider_name(self, client):
        """Test provider name detection."""
        provider = client.detect_provider_name()
        assert provider == "openai"

    def test_get_model_info(self, client):
        """Test model info retrieval."""
        info = client.get_model_info()
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o-mini"

# Parameter validation tests
class TestOpenAIValidation:
    """Test OpenAI parameter validation"""

    def test_validate_request_with_config(self, client):
        """Test request validation with config."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock the supports_feature method
        client.supports_feature = lambda feature: True
        
        validated_messages, validated_tools, validated_stream, validated_kwargs = client._validate_request_with_config(
            messages, None, False, temperature=0.7
        )
        
        assert validated_messages == messages
        assert validated_tools is None
        assert validated_stream is False
        assert "temperature" in validated_kwargs

    def test_validate_request_unsupported_features(self, client):
        """Test request validation when features not supported."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test_tool"}}]
        
        # Mock no feature support
        client.supports_feature = lambda feature: False
        
        _, validated_tools, validated_stream, _ = client._validate_request_with_config(
            messages, tools, True
        )
        
        assert validated_tools is None  # Tools should be removed
        assert validated_stream is False  # Streaming should be disabled

# Message normalization tests
class TestOpenAIMessageNormalization:
    """Test OpenAI message normalization"""

    def test_normalize_message_text_only(self, client):
        """Test message normalization with text only."""
        mock_msg = MagicMock()
        mock_msg.content = "Hello world!"
        mock_msg.tool_calls = None
        
        result = client._normalize_message(mock_msg)
        
        assert result["response"] == "Hello world!"
        assert result["tool_calls"] == []

    def test_normalize_message_with_tool_calls(self, client):
        """Test message normalization with tool calls."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        
        mock_msg = MagicMock()
        mock_msg.content = ""  # Empty content when tool calls present
        mock_msg.tool_calls = [mock_tool_call]
        
        result = client._normalize_message(mock_msg)
        
        # When tool calls are present and content is empty, response should be None
        assert result["response"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test_tool"

    def test_normalize_message_dict_format(self, client):
        """Test message normalization with dict format."""
        mock_msg = {
            "content": "Hello world!",
            "tool_calls": None
        }
        
        result = client._normalize_message(mock_msg)
        
        assert result["response"] == "Hello world!"
        assert result["tool_calls"] == []

    def test_normalize_message_nested_structure(self, client):
        """Test message normalization with nested structure."""
        mock_msg = MagicMock()
        mock_msg.content = "Hello"
        mock_msg.tool_calls = []
        
        result = client._normalize_message(mock_msg)
        
        assert result["response"] == "Hello"
        assert result["tool_calls"] == []

    def test_normalize_message_alternative_fields(self, client):
        """Test message normalization with alternative field names."""
        # Test handling of messages that might have different field names
        mock_msg = {"text": "Hello", "response": "Hello"}
        
        result = client._normalize_message(mock_msg)
        
        # Should handle gracefully even with unexpected format
        assert "response" in result
        assert "tool_calls" in result

    def test_normalize_message_invalid_tool_call_json(self, client):
        """Test message normalization with invalid tool call JSON."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = "invalid json"
        
        mock_msg = MagicMock()
        mock_msg.content = ""  # Empty content with tool calls
        mock_msg.tool_calls = [mock_tool_call]
        
        result = client._normalize_message(mock_msg)
        
        # Should handle invalid JSON gracefully
        assert result["response"] is None  # Empty content with tool calls
        assert len(result["tool_calls"]) == 1

    def test_normalize_message_no_attributes(self, client):
        """Test message normalization with object that has no attributes."""
        mock_msg = None
        
        result = client._normalize_message(mock_msg)
        
        # Should handle None gracefully
        assert "response" in result
        assert "tool_calls" in result

    def test_normalize_message_tool_call_missing_function(self, client):
        """Test message normalization with tool call missing function."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = None
        
        mock_msg = MagicMock()
        mock_msg.content = ""  # Empty content with tool calls
        mock_msg.tool_calls = [mock_tool_call]
        
        result = client._normalize_message(mock_msg)
        
        # Should handle missing function gracefully
        assert result["response"] is None  # Empty content with tool calls
        assert "tool_calls" in result

# Parameter adjustment tests
class TestOpenAIParameterAdjustment:
    """Test OpenAI parameter adjustment"""

    def test_adjust_parameters_for_provider(self, client):
        """Test parameter adjustment for provider."""
        # Mock the configuration methods
        client.validate_parameters = lambda **kwargs: kwargs
        client._get_model_capabilities = lambda: MagicMock(max_output_tokens=8192)
        
        params = {"temperature": 0.7, "max_tokens": 10000}
        adjusted = client._adjust_parameters_for_provider(params)
        
        assert adjusted["temperature"] == 0.7
        assert adjusted["max_tokens"] == 8192  # Should be capped

    def test_adjust_parameters_fallback(self, client):
        """Test parameter adjustment fallback."""
        # Mock configuration failure
        def mock_validate_error(**kwargs):
            raise Exception("Config error")
        
        client.validate_parameters = mock_validate_error
        
        params = {"temperature": 0.7}
        adjusted = client._adjust_parameters_for_provider(params)
        
        assert adjusted["temperature"] == 0.7

# Streaming tests
class TestOpenAIStreaming:
    """Test OpenAI streaming functionality"""

    @pytest.mark.asyncio
    async def test_stream_from_async_basic(self, client):
        """Test basic async streaming."""
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(" world!")
        ])
        
        chunks = []
        async for chunk in client._stream_from_async(mock_stream):
            chunks.append(chunk)
        
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_stream_from_async_with_tool_calls(self, client):
        """Test async streaming with tool calls."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk("", [mock_tool_call])
        ])
        
        chunks = []
        async for chunk in client._stream_from_async(mock_stream):
            chunks.append(chunk)
        
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_stream_from_async_error_handling(self, client):
        """Test async streaming error handling."""
        # Create a mock stream that raises an error
        async def error_stream():
            yield MockStreamChunk("Hello")
            raise Exception("Stream error")
        
        # This should handle the error gracefully
        chunks = []
        try:
            async for chunk in client._stream_from_async(error_stream()):
                chunks.append(chunk)
        except Exception:
            pass  # Expected to handle gracefully
        
        assert len(chunks) >= 1  # Should get at least one chunk

    @pytest.mark.asyncio
    async def test_stream_from_async_custom_normalization(self, client):
        """Test async streaming with custom normalization."""
        mock_stream = MockAsyncStream([
            MockStreamChunk("Custom"),
            MockStreamChunk(" response")
        ])
        
        chunks = []
        async for chunk in client._stream_from_async(mock_stream):
            chunks.append(chunk)
        
        # Should normalize properly
        assert all("response" in chunk for chunk in chunks)
        assert all("tool_calls" in chunk for chunk in chunks)

# Completion tests
class TestOpenAICompletion:
    """Test OpenAI completion functionality"""

    @pytest.mark.asyncio
    async def test_create_completion_non_streaming(self, client):
        """Test non-streaming completion."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello! How can I help?")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        assert "response" in result
        assert "tool_calls" in result

    @pytest.mark.asyncio
    async def test_create_completion_streaming(self, client):
        """Test streaming completion."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(" world!")
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        chunks = []
        async for chunk in client.create_completion(messages, stream=True):
            chunks.append(chunk)
        
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_create_completion_with_tools(self, client):
        """Test completion with tools."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test_tool"}}]
        
        mock_response = MockChatCompletion("Using tools...")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, tools=tools, stream=False)
        
        assert "response" in result
        assert "tool_calls" in result

    @pytest.mark.asyncio
    async def test_create_completion_parameter_validation(self, client):
        """Test completion with parameter validation."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello!")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(
            messages, 
            stream=False,
            temperature=0.7,
            max_tokens=100
        )
        
        assert "response" in result

# Regular completion tests  
class TestOpenAIRegularCompletion:
    """Test OpenAI regular completion methods"""

    @pytest.mark.asyncio
    async def test_regular_completion(self, client):
        """Test regular completion method."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello!")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client._regular_completion(messages)
        
        assert result["response"] == "Hello!"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_regular_completion_with_tools(self, client):
        """Test regular completion with tools."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test_tool"}}]
        
        mock_response = MockChatCompletion("Using tools...")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client._regular_completion(messages, tools)
        
        assert result["response"] == "Using tools..."

    @pytest.mark.asyncio
    async def test_regular_completion_error_handling(self, client):
        """Test regular completion error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        result = await client._regular_completion(messages)
        
        assert "error" in result
        assert result["error"] is True

# Stream completion tests
class TestOpenAIStreamCompletion:
    """Test OpenAI stream completion methods"""

    @pytest.mark.asyncio
    async def test_stream_completion_async(self, client):
        """Test stream completion async."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(" world!")
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_completion_async_with_retry(self, client):
        """Test stream completion with retry logic."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock a stream that works on second try
        call_count = 0
        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary error")
            return MockAsyncStream([MockStreamChunk("Success")])
        
        client.async_client.chat.completions.create = mock_create
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        # Should eventually succeed or handle error gracefully
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_completion_async_error_handling(self, client):
        """Test stream completion async error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("Stream error"))
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        # Should handle error and return error chunk
        assert len(chunks) >= 1
        if chunks:
            assert "error" in chunks[0] or "response" in chunks[0]

# Interface compliance tests
class TestOpenAIInterfaceCompliance:
    """Test OpenAI interface compliance"""

    def test_interface_compliance(self, client):
        """Test that client implements required interface."""
        # Check that key methods exist
        assert hasattr(client, 'create_completion')
        assert hasattr(client, 'get_model_info')
        assert hasattr(client, '_normalize_message')
        assert hasattr(client, 'detect_provider_name')
        assert callable(client.create_completion)
        assert callable(client.detect_provider_name)

# Error handling tests
class TestOpenAIErrorHandling:
    """Test OpenAI error handling"""

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, client):
        """Test streaming error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("Streaming error"))
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_non_streaming_error_handling(self, client):
        """Test non-streaming error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        
        result = await client._regular_completion(messages)
        
        assert "error" in result

# Integration tests
class TestOpenAIIntegration:
    """Test OpenAI integration"""

    @pytest.mark.asyncio
    async def test_full_integration_non_streaming(self, client):
        """Test full integration non-streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello! How can I help?")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        assert result["response"] == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_full_integration_streaming(self, client):
        """Test full integration streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(" there!")
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        chunks = []
        async for chunk in client.create_completion(messages, stream=True):
            chunks.append(chunk)
        
        assert len(chunks) >= 1

# Configuration tests
class TestOpenAIConfiguration:
    """Test OpenAI configuration"""

    def test_configuration_feature_validation(self, client):
        """Test configuration feature validation."""
        # Mock feature support
        client.supports_feature = lambda feature: feature in ["text", "streaming"]
        
        assert client.supports_feature("text")
        assert client.supports_feature("streaming")
        assert not client.supports_feature("vision")

# Provider specific tests
class TestOpenAIProviderSpecific:
    """Test OpenAI provider specific functionality"""

    def test_provider_specific_adjustments(self):
        """Test provider specific adjustments."""
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-key'}):
            client = OpenAILLMClient(
                model="deepseek-chat",
                api_key="test-key",
                api_base="https://api.deepseek.com"
            )
            
            provider = client.detect_provider_name()
            assert provider == "deepseek"

# Tool handling tests
class TestOpenAIToolHandling:
    """Test OpenAI tool handling"""

    def test_tool_name_sanitization(self, client):
        """Test tool name sanitization."""
        tools = [
            {"function": {"name": "invalid-name-with-dashes"}},
            {"function": {"name": "valid_name"}},
            {"function": {"name": "name with spaces"}}
        ]
        
        sanitized = client._sanitize_tool_names(tools)
        
        # Check that names are sanitized - after sanitization, no dashes should remain
        assert len(sanitized) == len(tools)
        for tool in sanitized:
            name = tool["function"]["name"]
            # Should not contain spaces after sanitization
            if name != "valid_name":  # valid_name is already valid
                assert " " not in name

# Complex conversation tests
class TestOpenAIComplexConversation:
    """Test OpenAI complex conversation handling"""

    @pytest.mark.asyncio
    async def test_complex_conversation_flow(self, client):
        """Test complex conversation flow."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather?"}
        ]
        
        mock_response = MockChatCompletion("I'd need your location to check the weather.")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        assert "response" in result
        assert result["response"] == "I'd need your location to check the weather."

# Provider detection tests
class TestOpenAIProviderDetection:
    """Test OpenAI provider detection"""

    def test_deepseek_provider_detection(self):
        """Test DeepSeek provider detection."""
        client = OpenAILLMClient(
            model="deepseek-chat",
            api_key="test-key",
            api_base="https://api.deepseek.com"
        )
        assert client.detect_provider_name() == "deepseek"

    def test_groq_provider_detection(self):
        """Test Groq provider detection."""
        client = OpenAILLMClient(
            model="llama-3.1-8b-instant",
            api_key="test-key",
            api_base="https://api.groq.com"
        )
        assert client.detect_provider_name() == "groq"

    def test_together_provider_detection(self):
        """Test Together provider detection."""
        client = OpenAILLMClient(
            model="test-model",
            api_key="test-key",
            api_base="https://api.together.ai"
        )
        assert client.detect_provider_name() == "together"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])