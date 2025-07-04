"""
Fixed Azure OpenAI Client Tests
===============================

This completely fixes the Azure OpenAI test issues with proper mocking
and async stream handling.
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

# Create a proper mock for the openai module
openai_mock = MagicMock()

# Mock streaming classes that work properly with async
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

# Set up the openai mock module
openai_mock.AzureOpenAI = MockAzureOpenAI
openai_mock.AsyncAzureOpenAI = MockAsyncAzureOpenAI
openai_mock.OpenAI = MagicMock()
openai_mock.AsyncOpenAI = MagicMock()

# Patch the openai module
sys.modules['openai'] = openai_mock

# Now import the client
from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

# Test fixtures
@pytest.fixture
def mock_env():
    """Mock environment variables for Azure OpenAI."""
    with patch.dict(os.environ, {
        'AZURE_OPENAI_API_KEY': 'test-api-key',
        'AZURE_OPENAI_ENDPOINT': 'https://test-resource.openai.azure.com'
    }):
        yield

@pytest.fixture
def client(mock_env):
    """Create Azure OpenAI client for testing"""
    return AzureOpenAILLMClient(
        model="gpt-4o-mini",
        api_key="test-key",
        azure_endpoint="https://test-resource.openai.azure.com"
    )

@pytest.fixture
def client_with_deployment(mock_env):
    """Create Azure OpenAI client with custom deployment"""
    return AzureOpenAILLMClient(
        model="gpt-4o-mini",
        api_key="test-key",
        azure_endpoint="https://test-resource.openai.azure.com",
        azure_deployment="my-custom-deployment"
    )

# Basic functionality tests
class TestAzureOpenAIBasic:
    """Test basic Azure OpenAI functionality"""

    def test_client_initialization_basic(self, mock_env):
        """Test basic client initialization."""
        client = AzureOpenAILLMClient()
        assert client.model == "gpt-4o-mini"
        assert client.azure_deployment == "gpt-4o-mini"
        assert client.api_version == "2024-02-01"

    def test_client_initialization_custom_deployment(self, mock_env):
        """Test initialization with custom deployment."""
        client = AzureOpenAILLMClient(
            model="gpt-4o",
            azure_deployment="my-custom-gpt4-deployment"
        )
        assert client.model == "gpt-4o"
        assert client.azure_deployment == "my-custom-gpt4-deployment"

    def test_get_model_info_basic(self, client):
        """Test basic model info retrieval."""
        info = client.get_model_info()
        assert info["provider"] == "azure_openai"
        assert info["model"] == "gpt-4o-mini"
        assert "azure_specific" in info
        assert "openai_compatible" in info

    def test_auth_type_detection(self, client):
        """Test authentication type detection."""
        auth_type = client._get_auth_type()
        assert auth_type == "api_key"

# Streaming tests
class TestAzureOpenAIStreaming:
    """Test Azure OpenAI streaming functionality"""

    @pytest.mark.asyncio
    async def test_streaming_basic(self, client):
        """Test basic streaming functionality."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock the async client to return our mock stream
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(" world!")
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        # Mock the stream processing method
        async def mock_stream_from_async(stream):
            async for chunk in stream:
                yield {"response": chunk.choices[0].delta.content, "tool_calls": []}
        
        client._stream_from_async = mock_stream_from_async
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0]["response"] == "Hello"
        assert chunks[1]["response"] == " world!"

    @pytest.mark.asyncio
    async def test_streaming_with_tools(self, client):
        """Test streaming with tool calls."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [{"function": {"name": "get_weather"}}]
        
        # Mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "NYC"}'
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("Checking weather"),
            MockStreamChunk("", [mock_tool_call])
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        # Mock the stream processing method
        async def mock_stream_from_async(stream):
            async for chunk in stream:
                if chunk.choices[0].delta.tool_calls:
                    yield {
                        "response": "",
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}
                        }]
                    }
                else:
                    yield {"response": chunk.choices[0].delta.content, "tool_calls": []}
        
        client._stream_from_async = mock_stream_from_async
        
        chunks = []
        async for chunk in client._stream_completion_async(messages, tools):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0]["response"] == "Checking weather"
        assert len(chunks[1]["tool_calls"]) == 1

# Regular completion tests
class TestAzureOpenAICompletion:
    """Test Azure OpenAI regular completion functionality"""

    @pytest.mark.asyncio
    async def test_regular_completion_basic(self, client):
        """Test basic regular completion."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello! How can I help you?")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client._regular_completion(messages)
        
        assert result["response"] == "Hello! How can I help you?"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_regular_completion_with_tools(self, client):
        """Test regular completion with tools."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [{"function": {"name": "get_weather"}}]
        
        # Create a mock response that mimics having no actual tool calls
        # but with some content
        mock_response = MockChatCompletion("I can help you check the weather!")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client._regular_completion(messages, tools)
        
        assert result["response"] == "I can help you check the weather!"
        assert result["tool_calls"] == []  # No actual tool calls in this mock

# Error handling tests
class TestAzureOpenAIErrorHandling:
    """Test Azure OpenAI error handling"""

    @pytest.mark.asyncio
    async def test_regular_completion_error_handling(self, client):
        """Test error handling in regular completion."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock client to raise exception
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        result = await client._regular_completion(messages)
        
        assert "error" in result
        assert result["error"] is True
        assert "API Error" in result["response"]

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, client):
        """Test error handling in streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock client to raise exception
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("Streaming error"))
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert chunks[0]["error"] is True
        assert "Streaming error" in chunks[0]["response"]

# Main interface tests
class TestAzureOpenAIInterface:
    """Test Azure OpenAI main interface"""

    @pytest.mark.asyncio
    async def test_create_completion_non_streaming(self, client):
        """Test create_completion with non-streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        expected_result = {"response": "Hello!", "tool_calls": []}
        
        async def mock_regular_completion(messages, tools=None, **kwargs):
            return expected_result
        
        client._regular_completion = mock_regular_completion
        
        result = await client.create_completion(messages, stream=False)
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_create_completion_streaming(self, client):
        """Test create_completion with streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        async def mock_stream_completion(messages, tools=None, **kwargs):
            yield {"response": "chunk1", "tool_calls": []}
            yield {"response": "chunk2", "tool_calls": []}
        
        client._stream_completion_async = mock_stream_completion
        
        chunks = []
        async for chunk in client.create_completion(messages, stream=True):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0]["response"] == "chunk1"
        assert chunks[1]["response"] == "chunk2"

# Parameter handling tests
class TestAzureOpenAIParameters:
    """Test Azure OpenAI parameter handling"""

    def test_parameter_adjustment_basic(self, client):
        """Test basic parameter adjustment."""
        # Mock the configuration methods
        client.validate_parameters = lambda **kwargs: kwargs
        client._get_model_capabilities = lambda: MagicMock(max_output_tokens=8192)
        
        params = {"temperature": 0.7, "max_tokens": 10000}
        adjusted = client._adjust_parameters_for_provider(params)
        
        assert adjusted["temperature"] == 0.7
        assert adjusted["max_tokens"] == 8192  # Should be capped

    def test_parameter_adjustment_fallback(self, client):
        """Test parameter adjustment fallback."""
        # Mock configuration failure
        def mock_validate_error(**kwargs):
            raise Exception("Config error")
        
        client.validate_parameters = mock_validate_error
        
        params = {"temperature": 0.7}
        adjusted = client._adjust_parameters_for_provider(params)
        
        assert adjusted["temperature"] == 0.7
        assert adjusted["max_tokens"] == 4096  # Fallback default

# Environment and configuration tests
class TestAzureOpenAIConfig:
    """Test Azure OpenAI configuration"""

    def test_environment_variables(self):
        """Test environment variable usage."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_API_KEY': 'env-api-key',
            'AZURE_OPENAI_ENDPOINT': 'https://env-resource.openai.azure.com'
        }):
            client = AzureOpenAILLMClient()
            info = client.get_model_info()
            assert info["provider"] == "azure_openai"

    def test_missing_credentials(self):
        """Test behavior when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Authentication required"):
                AzureOpenAILLMClient(azure_endpoint="https://test.openai.azure.com")

    def test_missing_endpoint(self):
        """Test behavior when endpoint is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="azure_endpoint is required"):
                AzureOpenAILLMClient(api_key="test-key")

# Integration tests
class TestAzureOpenAIIntegration:
    """Test Azure OpenAI integration"""

    @pytest.mark.asyncio
    async def test_full_integration_non_streaming(self, client):
        """Test full integration for non-streaming completion."""
        messages = [{"role": "user", "content": "Hello Azure!"}]
        
        mock_response = MockChatCompletion("Hello! I'm running on Azure OpenAI.")
        
        captured_payload = {}
        async def mock_create(**kwargs):
            captured_payload.update(kwargs)
            return mock_response
        
        client.async_client.chat.completions.create = mock_create
        
        result = await client.create_completion(messages, stream=False)
        
        assert result["response"] == "Hello! I'm running on Azure OpenAI."
        assert result["tool_calls"] == []
        
        # Verify payload structure
        assert captured_payload["model"] == client.azure_deployment
        assert captured_payload["messages"] == messages
        assert captured_payload["stream"] is False

    @pytest.mark.asyncio
    async def test_full_integration_streaming(self, client):
        """Test full integration for streaming completion."""
        messages = [{"role": "user", "content": "Count to 3"}]
        
        captured_payload = {}
        async def mock_create(**kwargs):
            captured_payload.update(kwargs)
            return MockAsyncStream([
                MockStreamChunk("1"),
                MockStreamChunk("2"), 
                MockStreamChunk("3"),
                MockStreamChunk("!")
            ])
        
        client.async_client.chat.completions.create = mock_create
        
        # Mock the stream processing
        async def mock_stream_from_async(stream):
            async for chunk in stream:
                yield {"response": chunk.choices[0].delta.content, "tool_calls": []}
        
        client._stream_from_async = mock_stream_from_async
        
        count_parts = []
        async for chunk in client.create_completion(messages, stream=True):
            count_parts.append(chunk["response"])
        
        assert len(count_parts) == 4
        assert count_parts == ["1", "2", "3", "!"]
        assert captured_payload["stream"] is True

# Resource cleanup tests
class TestAzureOpenAICleanup:
    """Test Azure OpenAI resource cleanup"""

    @pytest.mark.asyncio
    async def test_client_close(self, client):
        """Test client resource cleanup."""
        client.async_client.close = AsyncMock()
        client.client.close = MagicMock()
        
        await client.close()
        
        client.async_client.close.assert_called_once()
        client.client.close.assert_called_once()

    def test_client_repr(self, client):
        """Test client string representation."""
        repr_str = repr(client)
        assert "AzureOpenAILLMClient" in repr_str
        assert client.azure_deployment in repr_str
        assert client.model in repr_str


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])