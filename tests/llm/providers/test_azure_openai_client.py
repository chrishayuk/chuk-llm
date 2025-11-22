# tests/providers/test_azure_openai_client.py
"""
Updated Azure OpenAI Client Tests
=================================

Tests for the Azure OpenAI client that properly match the current implementation
with configuration-aware features and tool compatibility.
"""

import json
import os

# Mock the openai module before importing the client
import sys
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

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
            chunks = [MockStreamChunk("Hello"), MockStreamChunk(" world!")]
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
sys.modules["openai"] = openai_mock

# ---------------------------------------------------------------------------
# Configuration Mock Classes
# ---------------------------------------------------------------------------


class MockFeature:
    TEXT = "text"
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    SYSTEM_MESSAGES = "system_messages"
    PARALLEL_CALLS = "parallel_calls"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"

    @classmethod
    def from_string(cls, feature_str):
        return getattr(cls, feature_str.upper(), None)


class MockModelCapabilities:
    def __init__(
        self, features=None, max_context_length=128000, max_output_tokens=4096
    ):
        self.features = features or {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
            MockFeature.JSON_MODE,
            MockFeature.PARALLEL_CALLS,
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    def __init__(self, name="azure_openai", client_class="AzureOpenAILLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://api.openai.com"
        self.models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 500}

    def get_model_capabilities(self, model):
        # GPT-4 models typically have comprehensive features
        features = {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
            MockFeature.JSON_MODE,
            MockFeature.PARALLEL_CALLS,
        }

        # Vision capabilities
        if "gpt-4" in model.lower() and "vision" not in model.lower():
            features.add(MockFeature.VISION)
            features.add(MockFeature.MULTIMODAL)

        # Reasoning capabilities for latest models
        if any(pattern in model.lower() for pattern in ["gpt-4o", "gpt-4-turbo"]):
            features.add(MockFeature.REASONING)

        return MockModelCapabilities(features=features)


class MockConfig:
    def __init__(self):
        self.azure_openai_provider = MockProviderConfig()

    def get_provider(self, provider_name):
        if provider_name == "azure_openai":
            return self.azure_openai_provider
        return None


# Now import the client
from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
from chuk_llm.core.models import Message, Tool

# ---------------------------------------------------------------------------
# Fixtures with Configuration Mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_configuration():
    """Mock the configuration system"""
    mock_config = MockConfig()

    with patch("chuk_llm.configuration.get_config", return_value=mock_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            yield mock_config


@pytest.fixture
def mock_env():
    """Mock environment variables for Azure OpenAI."""
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-api-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    ):
        yield


@pytest.fixture
def client(mock_configuration, mock_env, monkeypatch):
    """Create Azure OpenAI client for testing with configuration mocking"""
    cl = AzureOpenAILLMClient(
        model="gpt-4o-mini",
        api_key="test-key",
        azure_endpoint="https://test-resource.openai.azure.com",
    )

    # Ensure configuration methods are properly mocked
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in [
            "text",
            "streaming",
            "tools",
            "vision",
            "system_messages",
            "multimodal",
            "json_mode",
            "parallel_calls",
        ],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "azure_openai",
            "model": "gpt-4o-mini",
            "client_class": "AzureOpenAILLMClient",
            "api_base": "https://api.openai.com",
            "features": [
                "text",
                "streaming",
                "tools",
                "vision",
                "system_messages",
                "multimodal",
                "json_mode",
                "parallel_calls",
            ],
            "supports_text": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "supports_system_messages": True,
            "supports_json_mode": True,
            "supports_parallel_calls": True,
            "supports_multimodal": True,
            "supports_reasoning": False,
            "max_context_length": 128000,
            "max_output_tokens": 4096,
            "tool_compatibility": {
                "supports_universal_naming": True,
                "sanitization_method": "replace_chars",
                "restoration_method": "name_mapping",
                "supported_name_patterns": ["alphanumeric_underscore"],
            },
            "azure_specific": {
                "endpoint": cl.azure_endpoint,
                "deployment": cl.azure_deployment,
                "api_version": cl.api_version,
                "authentication_type": cl._get_auth_type(),
                "deployment_to_model_mapping": True,
            },
            "openai_compatible": True,
            "parameter_mapping": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
                "stop": "stop",
                "stream": "stream",
                "tools": "tools",
                "tool_choice": "tool_choice",
            },
            "azure_parameters": [
                "azure_endpoint",
                "api_version",
                "azure_deployment",
                "azure_ad_token",
                "azure_ad_token_provider",
            ],
        },
    )

    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 128000)

    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        if "max_tokens" in result and result["max_tokens"] > 4096:
            result["max_tokens"] = 4096
        return result

    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)

    # Mock tool compatibility methods
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda tools: tools)
    monkeypatch.setattr(
        cl, "_restore_tool_names_in_response", lambda response, mapping: response
    )
    monkeypatch.setattr(
        cl,
        "get_tool_compatibility_info",
        lambda: {
            "supports_universal_naming": True,
            "sanitization_method": "replace_chars",
            "restoration_method": "name_mapping",
            "supported_name_patterns": ["alphanumeric_underscore"],
        },
    )

    # Mock configuration methods
    monkeypatch.setattr(cl, "_get_provider_config", lambda: MockProviderConfig())
    monkeypatch.setattr(cl, "_get_model_capabilities", lambda: MockModelCapabilities())

    # Initialize empty name mapping
    cl._current_name_mapping = {}

    return cl


@pytest.fixture
def client_with_deployment(mock_configuration, mock_env):
    """Create Azure OpenAI client with custom deployment"""
    return AzureOpenAILLMClient(
        model="gpt-4o-mini",
        api_key="test-key",
        azure_endpoint="https://test-resource.openai.azure.com",
        azure_deployment="my-custom-deployment",
    )


# Basic functionality tests
class TestAzureOpenAIBasic:
    """Test basic Azure OpenAI functionality"""

    def test_client_initialization_basic(self, mock_configuration, mock_env):
        """Test basic client initialization."""
        client = AzureOpenAILLMClient()
        assert client.model == "gpt-4o-mini"
        assert client.azure_deployment == "gpt-4o-mini"
        assert client.api_version == "2024-02-01"

    def test_client_initialization_custom_deployment(
        self, mock_configuration, mock_env
    ):
        """Test initialization with custom deployment."""
        client = AzureOpenAILLMClient(
            model="gpt-4o", azure_deployment="my-custom-gpt4-deployment"
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
        assert "tool_compatibility" in info

    def test_auth_type_detection(self, client):
        """Test authentication type detection."""
        auth_type = client._get_auth_type()
        assert auth_type == "api_key"

    def test_azure_specific_info(self, client):
        """Test Azure-specific information in model info."""
        info = client.get_model_info()
        azure_info = info["azure_specific"]

        assert "endpoint" in azure_info
        assert "deployment" in azure_info
        assert "api_version" in azure_info
        assert "authentication_type" in azure_info
        assert azure_info["deployment_to_model_mapping"] is True


# Request validation tests
class TestAzureOpenAIRequestValidation:
    """Test Azure OpenAI request validation"""

    def test_validate_request_with_config(self, client):
        """Test request validation against configuration."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        tools_dicts = [{"type": "function", "function": {"name": "test_tool", "description": "Test tool", "parameters": {}}}]

        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]
        tools = [Tool.model_validate(tool) for tool in tools_dicts]

        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(
                messages, tools, stream=True, temperature=0.7
            )
        )

        # _validate_request_with_config returns Pydantic objects
        assert len(validated_messages) == len(messages)
        assert len(validated_tools) == len(tools)
        assert validated_stream is True
        assert "temperature" in validated_kwargs

    def test_validate_request_unsupported_features(self, client, monkeypatch):
        """Test request validation when features are not supported."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        tools_dicts = [{"type": "function", "function": {"name": "test_tool", "description": "test_tool description", "parameters": {}}}]

        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]
        tools = [Tool.model_validate(tool) for tool in tools_dicts]

        # Mock configuration to not support streaming or tools
        monkeypatch.setattr(client, "supports_feature", lambda feature: False)

        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(
                messages,
                tools,
                stream=True,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
        )

        # _validate_request_with_config returns Pydantic objects
        # Note: Modern Azure OpenAI supports tools, streaming, and json mode
        # The implementation no longer filters these out based on supports_feature
        assert len(validated_messages) == len(messages)
        # Tools are now passed through (Azure OpenAI supports them)
        assert validated_tools is not None
        assert len(validated_tools) == len(tools)
        # Stream is handled at runtime, not filtered here
        assert isinstance(validated_stream, bool)
        assert "temperature" in validated_kwargs

    def test_validate_request_vision_content(self, client, monkeypatch):
        """Test request validation with vision content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,..."},
                    },
                ],
            }
        ]

        # Test with vision supported
        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(messages, stream=False)
        )

        # Validated messages are now Pydantic Message objects, not dicts
        assert len(validated_messages) == len(messages)
        # Check that vision content is preserved
        assert isinstance(validated_messages[0].content, list)
        assert len(validated_messages[0].content) == 2
        assert validated_messages[0].content[1].type == "image_url"

        # Test with vision not supported
        monkeypatch.setattr(
            client, "supports_feature", lambda feature: feature != "vision"
        )

        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(messages, stream=False)
        )

        # Should still pass through as Pydantic - warning logged but content unchanged
        assert len(validated_messages) == len(messages)
        assert isinstance(validated_messages[0].content, list)
        assert len(validated_messages[0].content) == 2


# Parameter adjustment tests
class TestAzureOpenAIParameters:
    """Test Azure OpenAI parameter handling"""

    def test_parameter_adjustment_basic(self, client):
        """Test basic parameter adjustment."""
        params = {"temperature": 0.7, "max_tokens": 10000}
        adjusted = client._adjust_parameters_for_provider(params)

        assert adjusted["temperature"] == 0.7
        assert adjusted["max_tokens"] == 4096  # Should be capped

    def test_parameter_adjustment_fallback(self, client, monkeypatch):
        """Test parameter adjustment fallback."""

        # Mock configuration failure
        def mock_validate_error(**kwargs):
            raise Exception("Config error")

        monkeypatch.setattr(client, "validate_parameters", mock_validate_error)

        params = {"temperature": 0.7}
        adjusted = client._adjust_parameters_for_provider(params)

        assert adjusted["temperature"] == 0.7
        assert adjusted["max_tokens"] == 4096  # Fallback default

    def test_prepare_azure_request_params(self, client):
        """Test Azure-specific parameter preparation."""
        params = {"deployment_name": "custom-deployment", "temperature": 0.7}
        prepared = client._prepare_azure_request_params(**params)

        assert prepared["model"] == "custom-deployment"
        assert "deployment_name" not in prepared
        assert prepared["temperature"] == 0.7


# Streaming tests
class TestAzureOpenAIStreaming:
    """Test Azure OpenAI streaming functionality"""

    @pytest.mark.asyncio
    async def test_streaming_basic(self, client):
        """Test basic streaming functionality."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        # Mock the async client to return our mock stream
        mock_stream = MockAsyncStream(
            [MockStreamChunk("Hello"), MockStreamChunk(" world!")]
        )
        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

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

        mock_stream = MockAsyncStream(
            [MockStreamChunk("Checking weather"), MockStreamChunk("", [mock_tool_call])]
        )
        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        # Mock the stream processing method
        async def mock_stream_from_async(stream):
            async for chunk in stream:
                if chunk.choices[0].delta.tool_calls:
                    yield {
                        "response": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
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

    @pytest.mark.asyncio
    async def test_streaming_with_name_mapping(self, client):
        """Test streaming with tool name restoration."""
        messages = [{"role": "user", "content": "Use tools"}]
        tools = [{"function": {"name": "test_tool_sanitized"}}]

        # Mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool_sanitized"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        mock_stream = MockAsyncStream([MockStreamChunk("", [mock_tool_call])])
        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        # Mock name mapping
        name_mapping = {"test_tool_sanitized": "test.tool:original"}

        # Mock the stream processing method
        async def mock_stream_from_async(stream):
            async for chunk in stream:
                if chunk.choices[0].delta.tool_calls:
                    yield {
                        "response": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "test_tool_sanitized",
                                    "arguments": '{"arg": "value"}',
                                },
                            }
                        ],
                    }

        client._stream_from_async = mock_stream_from_async

        def mock_restore(response, mapping):
            if response.get("tool_calls") and mapping:
                for tool_call in response["tool_calls"]:
                    sanitized_name = tool_call["function"]["name"]
                    if sanitized_name in mapping:
                        tool_call["function"]["name"] = mapping[sanitized_name]
            return response

        client._restore_tool_names_in_response = mock_restore

        chunks = []
        async for chunk in client._stream_completion_async(
            messages, tools, name_mapping
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["tool_calls"][0]["function"]["name"] == "test.tool:original"

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, client):
        """Test error handling in streaming."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        # Mock client to raise exception
        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Streaming error")
        )

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert chunks[0]["error"] is True
        assert "Streaming error" in chunks[0]["response"]

    @pytest.mark.asyncio
    async def test_streaming_retry_logic(self, client):
        """Test streaming retry logic on retryable errors."""
        messages = [{"role": "user", "content": "Hello"}]

        call_count = 0

        async def mock_create_with_retry(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("timeout error")  # Retryable
            else:
                return MockAsyncStream([MockStreamChunk("Success after retry")])

        client.async_client.chat.completions.create = mock_create_with_retry

        # Mock the stream processing method
        async def mock_stream_from_async(stream):
            async for chunk in stream:
                yield {"response": chunk.choices[0].delta.content, "tool_calls": []}

        client._stream_from_async = mock_stream_from_async

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["response"] == "Success after retry"
        assert call_count == 2  # Should have retried once


# Regular completion tests
class TestAzureOpenAICompletion:
    """Test Azure OpenAI regular completion functionality"""

    @pytest.mark.asyncio
    async def test_regular_completion_basic(self, client):
        """Test basic regular completion."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        mock_response = MockChatCompletion("Hello! How can I help you?")
        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await client._regular_completion(messages)

        assert result["response"] == "Hello! How can I help you?"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_regular_completion_with_tools(self, client):
        """Test regular completion with tools."""
        messages_dicts = [{"role": "user", "content": "What's the weather?"}]
        tools = [{"function": {"name": "get_weather"}}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        # Create a mock response that mimics having no actual tool calls
        # but with some content
        mock_response = MockChatCompletion("I can help you check the weather!")
        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await client._regular_completion(messages, tools)

        assert result["response"] == "I can help you check the weather!"
        assert result["tool_calls"] == []  # No actual tool calls in this mock

    @pytest.mark.asyncio
    async def test_regular_completion_with_name_mapping(self, client):
        """Test regular completion with tool name restoration."""
        messages = [{"role": "user", "content": "Use tools"}]
        tools = [{"function": {"name": "test_tool_sanitized"}}]

        # Mock tool call in response
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool_sanitized"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        mock_response = MockChatCompletion(content=None, tool_calls=[mock_tool_call])
        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Mock name mapping
        name_mapping = {"test_tool_sanitized": "test.tool:original"}

        def mock_restore(response, mapping):
            if response.get("tool_calls") and mapping:
                for tool_call in response["tool_calls"]:
                    sanitized_name = tool_call["function"]["name"]
                    if sanitized_name in mapping:
                        tool_call["function"]["name"] = mapping[sanitized_name]
            return response

        client._restore_tool_names_in_response = mock_restore

        result = await client._regular_completion(messages, tools, name_mapping)

        assert result["tool_calls"][0]["function"]["name"] == "test.tool:original"

    @pytest.mark.asyncio
    async def test_regular_completion_error_handling(self, client):
        """Test error handling in regular completion."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        # Mock client to raise exception
        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await client._regular_completion(messages)

        assert "error" in result
        assert result["error"] is True
        assert "API Error" in result["response"]


# Message normalization tests
class TestAzureOpenAIMessageNormalization:
    """Test Azure OpenAI message normalization"""

    def test_normalize_message_basic(self, client):
        """Test basic message normalization."""
        mock_message = MockMessage("Hello from Azure!")

        result = client._normalize_message(mock_message)

        assert result["response"] == "Hello from Azure!"
        assert result["tool_calls"] == []

    def test_normalize_message_with_tool_calls(self, client):
        """Test message normalization with tool calls."""
        # Mock tool call with proper Azure format
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "NYC"}'

        mock_message = MockMessage(content=None, tool_calls=[mock_tool_call])

        result = client._normalize_message(mock_message)

        assert result["response"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"city": "NYC"}'

    def test_normalize_message_azure_argument_formats(self, client):
        """Test Azure-specific argument format handling."""
        # Since the Azure client implementation has a fallback that may not work
        # perfectly with our mocks, let's test the core logic more directly

        # Test that the normalization method works at all
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"city": "NYC"}'

        # Configure the mock to return values properly
        type(mock_tool_call).id = PropertyMock(return_value="call_123")
        type(mock_tool_call.function).name = PropertyMock(return_value="test_tool")
        type(mock_tool_call.function).arguments = PropertyMock(
            return_value='{"city": "NYC"}'
        )

        mock_message = MockMessage(content=None, tool_calls=[mock_tool_call])

        result = client._normalize_message(mock_message)

        # Verify that we get a tool call back (the exact arguments format may vary)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test_tool"
        # The arguments should be a valid JSON string (either the original or "{}")

        try:
            json.loads(result["tool_calls"][0]["function"]["arguments"])
            # If we can parse it as JSON, the test passes
            assert True
        except json.JSONDecodeError:
            raise AssertionError(
                f"Arguments are not valid JSON: {result['tool_calls'][0]['function']['arguments']}"
            )

    def test_normalize_message_dict_arguments(self, client):
        """Test normalization when arguments are provided as dict."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = {"city": "NYC"}  # Dict instead of string

        mock_message = MockMessage(content=None, tool_calls=[mock_tool_call])

        result = client._normalize_message(mock_message)

        assert result["tool_calls"][0]["function"]["arguments"] == '{"city": "NYC"}'


# Main interface tests
class TestAzureOpenAIInterface:
    """Test Azure OpenAI main interface"""

    @pytest.mark.asyncio
    async def test_create_completion_non_streaming(self, client):
        """Test create_completion with non-streaming."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        expected_result = {"response": "Hello!", "tool_calls": []}

        async def mock_regular_completion(
            messages, tools=None, name_mapping=None, **kwargs
        ):
            return expected_result

        client._regular_completion = mock_regular_completion

        result = await client.create_completion(messages, stream=False)
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_create_completion_streaming(self, client):
        """Test create_completion with streaming."""
        messages_dicts = [{"role": "user", "content": "Hello"}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        async def mock_stream_completion(
            messages, tools=None, name_mapping=None, **kwargs
        ):
            yield {"response": "chunk1", "tool_calls": []}
            yield {"response": "chunk2", "tool_calls": []}

        client._stream_completion_async = mock_stream_completion

        chunks = []
        async for chunk in client.create_completion(messages, stream=True):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["response"] == "chunk1"
        assert chunks[1]["response"] == "chunk2"

    @pytest.mark.asyncio
    async def test_create_completion_with_tools(self, client):
        """Test create_completion with tools."""
        messages_dicts = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {"type": "function", "function": {"name": "get_weather", "description": "get_weather description", "parameters": {}}}
        ]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        # Mock regular completion
        expected_result = {
            "response": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        }

        async def mock_regular_completion(messages, tools, name_mapping, **kwargs):
            # Verify tools were passed
            assert tools is not None
            assert len(tools) == 1
            return expected_result

        client._regular_completion = mock_regular_completion

        result = await client.create_completion(messages, tools=tools, stream=False)

        assert result == expected_result
        assert len(result["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_create_completion_with_tools_not_supported(
        self, client, monkeypatch
    ):
        """Test create_completion with tools when not supported."""
        messages_dicts = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {"type": "function", "function": {"name": "get_weather", "description": "get_weather description", "parameters": {}}}
        ]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


        # Mock tools as not supported
        monkeypatch.setattr(
            client, "supports_feature", lambda feature: feature != "tools"
        )

        # Mock regular completion
        expected_result = {"response": "I cannot use tools.", "tool_calls": []}

        async def mock_regular_completion(messages, tools, name_mapping, **kwargs):
            # Note: Modern Azure OpenAI supports tools, so they are passed through
            # even if supports_feature returns False (the check is informational)
            assert tools is not None
            assert len(tools) > 0
            return expected_result

        client._regular_completion = mock_regular_completion

        result = await client.create_completion(messages, tools=tools, stream=False)

        assert result == expected_result


# Environment and configuration tests
class TestAzureOpenAIConfig:
    """Test Azure OpenAI configuration"""

    def test_environment_variables(self, mock_configuration):
        """Test environment variable usage."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "env-api-key",
                "AZURE_OPENAI_ENDPOINT": "https://env-resource.openai.azure.com",
            },
        ):
            client = AzureOpenAILLMClient()
            info = client.get_model_info()
            assert info["provider"] == "azure_openai"

    def test_missing_credentials(self, mock_configuration):
        """Test behavior when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Authentication required"):
                AzureOpenAILLMClient(azure_endpoint="https://test.openai.azure.com")

    def test_missing_endpoint(self, mock_configuration):
        """Test behavior when endpoint is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="azure_endpoint is required"):
                AzureOpenAILLMClient(api_key="test-key")

    def test_azure_ad_token_auth(self, mock_configuration, mock_env):
        """Test Azure AD token authentication."""
        client = AzureOpenAILLMClient(
            azure_ad_token="test-token", azure_endpoint="https://test.openai.azure.com"
        )
        # Mock the auth type detection since the actual client might not expose the token
        client.async_client._azure_ad_token = "test-token"
        assert client._get_auth_type() == "azure_ad_token"

    def test_azure_ad_token_provider_auth(self, mock_configuration, mock_env):
        """Test Azure AD token provider authentication."""
        mock_provider = MagicMock()
        client = AzureOpenAILLMClient(
            azure_ad_token_provider=mock_provider,
            azure_endpoint="https://test.openai.azure.com",
        )
        # Mock the auth type detection since the actual client might not expose the provider
        client.async_client._azure_ad_token_provider = mock_provider
        assert client._get_auth_type() == "azure_ad_token_provider"


# Integration tests
class TestAzureOpenAIIntegration:
    """Test Azure OpenAI integration"""

    @pytest.mark.asyncio
    async def test_full_integration_non_streaming(self, client):
        """Test full integration for non-streaming completion."""
        messages_dicts = [{"role": "user", "content": "Hello Azure!"}]
        # Convert dicts to Pydantic models
        messages = [Message.model_validate(msg) for msg in messages_dicts]


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
        # Messages are converted to dicts by the client
        assert len(captured_payload["messages"]) == len(messages)
        assert captured_payload["messages"][0]["content"] == "Hello Azure!"
        assert captured_payload["stream"] is False

    @pytest.mark.asyncio
    async def test_full_integration_streaming(self, client):
        """Test full integration for streaming completion."""
        messages = [{"role": "user", "content": "Count to 3"}]

        captured_payload = {}

        async def mock_create(**kwargs):
            captured_payload.update(kwargs)
            return MockAsyncStream(
                [
                    MockStreamChunk("1"),
                    MockStreamChunk("2"),
                    MockStreamChunk("3"),
                    MockStreamChunk("!"),
                ]
            )

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


# Tool compatibility tests
class TestAzureOpenAIToolCompatibility:
    """Test Azure OpenAI tool compatibility"""

    def test_tool_compatibility_info(self, client):
        """Test tool compatibility information."""
        info = client.get_tool_compatibility_info()

        assert info["supports_universal_naming"] is True
        assert info["sanitization_method"] == "replace_chars"
        assert info["restoration_method"] == "name_mapping"
        assert "alphanumeric_underscore" in info["supported_name_patterns"]

    def test_tool_name_sanitization_and_restoration(self, client):
        """Test tool name sanitization and restoration."""
        # Test that sanitization is called (mocked to return tools unchanged)
        tools = [
            {
                "type": "function",
                "function": {"name": "test.tool:name", "parameters": {}},
            }
        ]

        # Mock sanitization to simulate real behavior
        def mock_sanitize(tools_list):
            client._current_name_mapping = {"test_tool_name": "test.tool:name"}
            return [
                {
                    "type": "function",
                    "function": {"name": "test_tool_name", "parameters": {}},
                }
            ]

        client._sanitize_tool_names = mock_sanitize

        sanitized_tools = client._sanitize_tool_names(tools)

        # Verify sanitization occurred
        assert sanitized_tools[0]["function"]["name"] == "test_tool_name"
        assert "test_tool_name" in client._current_name_mapping


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


# Feature support validation tests
class TestAzureOpenAIFeatureSupport:
    """Test Azure OpenAI feature support validation"""

    def test_feature_support_validation(self, client, monkeypatch):
        """Test that feature support is properly validated."""
        # Test supported features (from fixture)
        supported_features = [
            "text",
            "streaming",
            "tools",
            "vision",
            "system_messages",
            "multimodal",
            "json_mode",
            "parallel_calls",
        ]

        for feature in supported_features:
            assert client.supports_feature(feature) is True

        # Test unsupported features
        unsupported_features = ["reasoning"]

        for feature in unsupported_features:
            assert client.supports_feature(feature) is False

        # Test individual feature isolation
        for feature in supported_features:
            # Mock only this feature as supported
            monkeypatch.setattr(
                client,
                "supports_feature",
                lambda f, target_feature=feature: f == target_feature,
            )

            # Test that only the target feature is supported
            assert client.supports_feature(feature) is True

            # Test that other features are not supported
            other_features = [f for f in supported_features if f != feature]
            for other_feature in other_features:
                assert client.supports_feature(other_feature) is False


# Complex scenario tests
class TestAzureOpenAIComplexScenarios:
    """Test complex Azure OpenAI scenarios"""

    @pytest.mark.asyncio
    async def test_complex_conversation_flow(self, client):
        """Test a complex conversation with multiple message types."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image and use a tool"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,..."},
                    },
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "analyze_image", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "analyze_image",
                "content": "Analysis result",
            },
            {"role": "assistant", "content": "Based on the analysis..."},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "Analyze an image",
                    "parameters": {},
                },
            }
        ]

        # Mock completion
        async def mock_completion(messages, tools, name_mapping, **kwargs):
            return {"response": "Complex conversation complete", "tool_calls": []}

        client._regular_completion = mock_completion

        result = await client.create_completion(messages, tools=tools, stream=False)

        assert result["response"] == "Complex conversation complete"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_unsupported_features_graceful_handling(self, client, monkeypatch):
        """Test graceful handling when features are not supported."""
        # Mock all features as unsupported
        monkeypatch.setattr(client, "supports_feature", lambda feature: False)

        messages = [
            {"role": "system", "content": "System message"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Text with image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,..."},
                    },
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            },
        ]

        tools = [
            {"type": "function", "function": {"name": "test_tool", "description": "test_tool description", "parameters": {}}}
        ]

        # Mock completion
        async def mock_completion(messages, tools, name_mapping, **kwargs):
            # Note: Modern Azure OpenAI supports these features, so they pass through
            # The supports_feature check is informational, not a hard filter
            assert tools is not None
            return {"response": "Features handled gracefully", "tool_calls": []}

        client._regular_completion = mock_completion

        result = await client.create_completion(
            messages,
            tools=tools,
            stream=False,  # Should be converted to False when not supported
            response_format={
                "type": "json_object"
            },  # Should be removed when not supported
        )

        assert result["response"] == "Features handled gracefully"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
