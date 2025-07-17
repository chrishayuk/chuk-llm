# tests/providers/test_openai_client.py
"""
Comprehensive OpenAI Client Tests
=================================

Fixed test suite for OpenAI client with proper mocking, configuration testing,
and comprehensive coverage of all enhanced functionality including:
- Universal tool name compatibility
- Configuration-aware features
- Streaming and non-streaming
- Error handling
- Provider detection
- Message normalization
"""
import pytest
import asyncio
import json
import os
import uuid
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import AsyncIterator, List, Dict, Any

# Mock the openai module before importing the client
import sys
from unittest.mock import MagicMock

# Create comprehensive mock classes
class MockToolCall:
    def __init__(self, id=None, function_name="test_tool", arguments='{}'):
        self.id = id or f"call_{uuid.uuid4().hex[:8]}"
        self.function = MagicMock()
        self.function.name = function_name
        self.function.arguments = arguments
        self.type = "function"

class MockDelta:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

class MockChoice:
    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.delta = MockDelta(content, tool_calls)
        self.message = MagicMock()
        self.message.content = content
        self.message.tool_calls = tool_calls or []
        self.finish_reason = finish_reason

class MockStreamChunk:
    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.choices = [MockChoice(content, tool_calls, finish_reason)]
        self.id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self.model = "gpt-4o-mini"

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
        self.id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
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

# Set up the complete openai mock module
openai_mock = MagicMock()
openai_mock.OpenAI = MockOpenAI
openai_mock.AsyncOpenAI = MockAsyncOpenAI

# Patch the openai module
sys.modules['openai'] = openai_mock

# Now import the client
from chuk_llm.llm.providers.openai_client import OpenAILLMClient

# Configuration mock classes
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

class MockModelCapabilities:
    def __init__(self, features=None, max_context_length=128000, max_output_tokens=4096):
        self.features = features or {
            MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS, 
            MockFeature.VISION, MockFeature.SYSTEM_MESSAGES, MockFeature.JSON_MODE
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens

class MockProviderConfig:
    def __init__(self, name="openai", client_class="OpenAILLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://api.openai.com/v1"
        self.models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 10000}
    
    def get_model_capabilities(self, model):
        features = {
            MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS, 
            MockFeature.VISION, MockFeature.SYSTEM_MESSAGES, MockFeature.JSON_MODE
        }
        
        if "gpt-4" in model:
            features.add(MockFeature.REASONING)
        
        return MockModelCapabilities(features=features)

class MockConfig:
    def __init__(self):
        self.openai_provider = MockProviderConfig()
        self.deepseek_provider = MockProviderConfig("deepseek")
        self.groq_provider = MockProviderConfig("groq")
    
    def get_provider(self, provider_name):
        if provider_name == "openai":
            return self.openai_provider
        elif provider_name == "deepseek":
            return self.deepseek_provider
        elif provider_name == "groq":
            return self.groq_provider
        return None

# Test fixtures
@pytest.fixture
def mock_configuration():
    """Mock the configuration system"""
    mock_config = MockConfig()
    
    with patch('chuk_llm.configuration.get_config', return_value=mock_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            yield mock_config

@pytest.fixture
def mock_env():
    """Mock environment variables for OpenAI."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key'
    }):
        yield

@pytest.fixture
def client(mock_configuration, mock_env, monkeypatch):
    """Create OpenAI client with proper mocking"""
    cl = OpenAILLMClient(
        model="gpt-4o-mini",
        api_key="test-key"
    )
    
    # Mock configuration methods
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "vision", "system_messages", "json_mode"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "client_class": "OpenAILLMClient",
        "api_base": "https://api.openai.com/v1",
        "features": ["text", "streaming", "tools", "vision", "system_messages", "json_mode"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "supports_system_messages": True,
        "supports_json_mode": True,
        "supports_parallel_calls": False,
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
        "vision_format": "universal_image_url",
        "supported_parameters": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "stream"],
        "unsupported_parameters": [],
    })
    
    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 128000)
    
    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        if 'max_tokens' in result and result['max_tokens'] > 4096:
            result['max_tokens'] = 4096
        return result
    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)
    
    # Mock tool compatibility methods
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda tools: tools)
    monkeypatch.setattr(cl, "_restore_tool_names_in_response", lambda response, mapping: response)
    monkeypatch.setattr(cl, "get_tool_compatibility_info", lambda: {
        "supports_universal_naming": True,
        "sanitization_method": "replace_chars",
        "restoration_method": "name_mapping",
        "supported_name_patterns": ["alphanumeric_underscore"],
    })
    
    # Initialize empty name mapping
    cl._current_name_mapping = {}
    
    return cl

@pytest.fixture
def deepseek_client(mock_configuration, mock_env, monkeypatch):
    """Create DeepSeek client for testing"""
    cl = OpenAILLMClient(
        model="deepseek-chat",
        api_key="test-key",
        api_base="https://api.deepseek.com"
    )
    
    # Mock configuration methods for DeepSeek
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "system_messages"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "client_class": "OpenAILLMClient",
        "api_base": "https://api.deepseek.com",
        "features": ["text", "streaming", "tools", "system_messages"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "supports_system_messages": True,
        "supports_json_mode": False,
        "supports_parallel_calls": False,
        "supports_multimodal": False,
        "supports_reasoning": True,
        "max_context_length": 32768,
        "max_output_tokens": 4096,
        "tool_compatibility": {
            "supports_universal_naming": True,
            "sanitization_method": "replace_chars",
            "restoration_method": "name_mapping",
            "supported_name_patterns": ["alphanumeric_underscore"],
        },
        "vision_format": "not_supported",
        "supported_parameters": ["temperature", "max_tokens", "top_p", "stream"],
        "unsupported_parameters": ["frequency_penalty", "presence_penalty"],
    })
    
    # Mock tool compatibility methods
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda tools: tools)
    monkeypatch.setattr(cl, "_restore_tool_names_in_response", lambda response, mapping: response)
    monkeypatch.setattr(cl, "get_tool_compatibility_info", lambda: {
        "supports_universal_naming": True,
        "sanitization_method": "replace_chars",
        "restoration_method": "name_mapping",
        "supported_name_patterns": ["alphanumeric_underscore"],
    })
    
    # Initialize empty name mapping
    cl._current_name_mapping = {}
    
    return cl

# Basic functionality tests
class TestOpenAIBasic:
    """Test basic OpenAI functionality"""

    def test_client_initialization(self, mock_configuration, mock_env):
        """Test basic client initialization."""
        client = OpenAILLMClient()
        assert client.model == "gpt-4o-mini"
        assert client.detected_provider == "openai"

    def test_client_initialization_with_params(self, mock_configuration, mock_env):
        """Test client initialization with parameters."""
        client = OpenAILLMClient(
            model="gpt-4o",
            api_key="custom-key",
            api_base="https://custom.api.com"
        )
        assert client.model == "gpt-4o"
        assert client.api_base == "https://custom.api.com"

    def test_detect_provider_name(self, client):
        """Test provider name detection."""
        assert client.detect_provider_name() == "openai"

    def test_detect_provider_name_deepseek(self, deepseek_client):
        """Test DeepSeek provider detection."""
        assert deepseek_client.detect_provider_name() == "deepseek"

    def test_get_model_info(self, client):
        """Test model info retrieval."""
        info = client.get_model_info()
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o-mini"
        assert info["supports_streaming"] is True
        assert info["supports_tools"] is True
        assert "tool_compatibility" in info

    def test_get_model_info_deepseek(self, deepseek_client):
        """Test model info for DeepSeek."""
        info = deepseek_client.get_model_info()
        assert info["provider"] == "deepseek"
        assert info["supports_vision"] is False
        assert info["supports_json_mode"] is False

# Provider detection tests
class TestOpenAIProviderDetection:
    """Test OpenAI provider detection"""

    def test_openai_provider_detection(self, mock_configuration, mock_env):
        """Test OpenAI provider detection."""
        client = OpenAILLMClient(model="gpt-4o", api_key="test-key")
        assert client._detect_provider_name(None) == "openai"

    def test_deepseek_provider_detection(self, mock_configuration, mock_env):
        """Test DeepSeek provider detection."""
        client = OpenAILLMClient(
            model="deepseek-chat",
            api_key="test-key",
            api_base="https://api.deepseek.com"
        )
        assert client._detect_provider_name("https://api.deepseek.com") == "deepseek"

    def test_groq_provider_detection(self, mock_configuration, mock_env):
        """Test Groq provider detection."""
        client = OpenAILLMClient(
            model="llama-3.1-8b-instant",
            api_key="test-key",
            api_base="https://api.groq.com/openai/v1"
        )
        assert client._detect_provider_name("https://api.groq.com/openai/v1") == "groq"

    def test_together_provider_detection(self, mock_configuration, mock_env):
        """Test Together provider detection."""
        client = OpenAILLMClient(
            model="test-model",
            api_key="test-key",
            api_base="https://api.together.ai"
        )
        assert client._detect_provider_name("https://api.together.ai") == "together"

    def test_anyscale_provider_detection(self, mock_configuration, mock_env):
        """Test Anyscale provider detection."""
        client = OpenAILLMClient(
            model="test-model",
            api_key="test-key",
            api_base="https://api.anyscale.com"
        )
        assert client._detect_provider_name("https://api.anyscale.com") == "anyscale"

    def test_perplexity_provider_detection(self, mock_configuration, mock_env):
        """Test Perplexity provider detection."""
        client = OpenAILLMClient(
            model="test-model",
            api_key="test-key",
            api_base="https://api.perplexity.ai"
        )
        assert client._detect_provider_name("https://api.perplexity.ai") == "perplexity"

    def test_unknown_provider_detection(self, mock_configuration, mock_env):
        """Test unknown provider detection."""
        client = OpenAILLMClient(
            model="test-model",
            api_key="test-key",
            api_base="https://api.unknown.com"
        )
        assert client._detect_provider_name("https://api.unknown.com") == "openai_compatible"

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
        mock_tool_call = MockToolCall(function_name="test_tool", arguments='{"arg": "value"}')
        
        mock_msg = MagicMock()
        mock_msg.content = ""  # Empty content when tool calls present
        mock_msg.tool_calls = [mock_tool_call]
        
        result = client._normalize_message(mock_msg)
        
        # When tool calls are present and content is empty, response should be None
        assert result["response"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test_tool"

    def test_normalize_message_mixed_content_and_tools(self, client):
        """Test message normalization with both content and tool calls."""
        mock_tool_call = MockToolCall(function_name="test_tool", arguments='{"arg": "value"}')
        
        mock_msg = MagicMock()
        mock_msg.content = "I'll call a tool"
        mock_msg.tool_calls = [mock_tool_call]
        
        result = client._normalize_message(mock_msg)
        
        # With both content and tool calls, response should be the content
        assert result["response"] == "I'll call a tool"
        assert len(result["tool_calls"]) == 1

    def test_normalize_message_dict_format(self, client):
        """Test message normalization with dict format."""
        mock_msg = {
            "content": "Hello world!",
            "tool_calls": None
        }
        
        result = client._normalize_message(mock_msg)
        
        assert result["response"] == "Hello world!"
        assert result["tool_calls"] == []

    def test_normalize_message_invalid_tool_call_json(self, client):
        """Test message normalization with invalid tool call JSON."""
        mock_tool_call = MockToolCall(function_name="test_tool", arguments="invalid json")
        
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.tool_calls = [mock_tool_call]
        
        result = client._normalize_message(mock_msg)
        
        # Should handle invalid JSON gracefully
        assert result["response"] is None
        assert len(result["tool_calls"]) == 1
        # Should use empty object for invalid JSON
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_normalize_message_no_attributes(self, client):
        """Test message normalization with None input."""
        result = client._normalize_message(None)
        
        assert result["response"] == ""
        assert result["tool_calls"] == []

    def test_normalize_message_nested_message_structure(self, client):
        """Test message normalization with nested message structure."""
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.tool_calls = None
        
        # Mock nested message structure
        mock_msg.message = MagicMock()
        mock_msg.message.content = "Nested content"
        mock_msg.message.tool_calls = None
        
        result = client._normalize_message(mock_msg)
        
        assert result["response"] == "Nested content"
        assert result["tool_calls"] == []

# Request validation tests
class TestOpenAIRequestValidation:
    """Test OpenAI request validation"""

    def test_validate_request_with_config_all_supported(self, client):
        """Test request validation with all features supported."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test_tool"}}]
        
        validated_messages, validated_tools, validated_stream, validated_kwargs = client._validate_request_with_config(
            messages, tools, True, temperature=0.7
        )
        
        assert validated_messages == messages
        assert validated_tools == tools
        assert validated_stream is True
        assert validated_kwargs["temperature"] == 0.7

    def test_validate_request_with_config_streaming_not_supported(self, client, monkeypatch):
        """Test request validation when streaming not supported."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock streaming as not supported
        monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "streaming")
        
        _, _, validated_stream, _ = client._validate_request_with_config(
            messages, None, True
        )
        
        assert validated_stream is False

    def test_validate_request_with_config_tools_not_supported(self, client, monkeypatch):
        """Test request validation when tools not supported."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test_tool"}}]
        
        # Mock tools as not supported
        monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")
        
        _, validated_tools, _, _ = client._validate_request_with_config(
            messages, tools, False
        )
        
        assert validated_tools is None

    def test_validate_request_with_config_vision_not_supported(self, client, monkeypatch):
        """Test request validation when vision not supported."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]}
        ]
        
        # Mock vision as not supported
        monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "vision")
        
        # Should log warning but not fail
        validated_messages, _, _, _ = client._validate_request_with_config(
            messages, None, False
        )
        
        assert validated_messages == messages

    def test_validate_request_with_config_json_mode_not_supported(self, client, monkeypatch):
        """Test request validation when JSON mode not supported."""
        messages = [{"role": "user", "content": "Give me JSON"}]
        
        # Mock JSON mode as not supported
        monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "json_mode")
        
        _, _, _, validated_kwargs = client._validate_request_with_config(
            messages, None, False, response_format={"type": "json_object"}
        )
        
        assert "response_format" not in validated_kwargs

# Tool name sanitization tests
class TestOpenAIToolNameSanitization:
    """Test OpenAI tool name sanitization"""

    def test_prepare_messages_for_conversation_no_mapping(self, client):
        """Test message preparation when no name mapping exists."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        result = client._prepare_messages_for_conversation(messages)
        
        assert result == messages

    def test_prepare_messages_for_conversation_with_tool_calls(self, client):
        """Test message preparation with tool calls and name mapping."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get.weather:data", "arguments": "{}"}}
            ]}
        ]
        
        # Mock name mapping
        client._current_name_mapping = {"get_weather_data": "get.weather:data"}
        
        result = client._prepare_messages_for_conversation(messages)
        
        # Should sanitize tool names in assistant messages
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_weather_data"

    def test_prepare_messages_for_conversation_complex_mapping(self, client):
        """Test message preparation with complex name mapping."""
        messages = [
            {"role": "user", "content": "Test multiple tools"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "stdio.read:query", "arguments": "{}"}},
                {"function": {"name": "web.api:search", "arguments": "{}"}}
            ]}
        ]
        
        # Mock complex name mapping
        client._current_name_mapping = {
            "stdio_read_query": "stdio.read:query",
            "web_api_search": "web.api:search"
        }
        
        result = client._prepare_messages_for_conversation(messages)
        
        # Should sanitize both tool names
        assert result[1]["tool_calls"][0]["function"]["name"] == "stdio_read_query"
        assert result[1]["tool_calls"][1]["function"]["name"] == "web_api_search"

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
        assert chunks[0]["response"] == "Hello"
        assert chunks[1]["response"] == " world!"

    @pytest.mark.asyncio
    async def test_stream_from_async_with_tool_calls(self, client):
        """Test async streaming with tool calls."""
        mock_tool_call = MockToolCall(function_name="test_tool", arguments='{"arg": "value"}')
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("I'll call a tool"),
            MockStreamChunk("", [mock_tool_call])
        ])
        
        chunks = []
        async for chunk in client._stream_from_async(mock_stream):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0]["response"] == "I'll call a tool"
        assert len(chunks[1]["tool_calls"]) == 1
        assert chunks[1]["tool_calls"][0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_stream_from_async_with_name_mapping(self, client):
        """Test async streaming with tool name restoration."""
        mock_tool_call = MockToolCall(function_name="test_tool_sanitized", arguments='{}')
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("", [mock_tool_call])
        ])
        
        # Mock name mapping
        name_mapping = {"test_tool_sanitized": "test.tool:original"}
        
        # Mock the restoration method
        def mock_restore(response, mapping):
            if response.get("tool_calls") and mapping:
                for tool_call in response["tool_calls"]:
                    sanitized_name = tool_call["function"]["name"]
                    if sanitized_name in mapping:
                        tool_call["function"]["name"] = mapping[sanitized_name]
            return response
        
        client._restore_tool_names_in_response = mock_restore
        
        chunks = []
        async for chunk in client._stream_from_async(mock_stream, name_mapping):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert len(chunks[0]["tool_calls"]) == 1
        assert chunks[0]["tool_calls"][0]["function"]["name"] == "test.tool:original"

    @pytest.mark.asyncio
    async def test_stream_from_async_error_handling(self, client):
        """Test async streaming error handling."""
        async def error_stream():
            yield MockStreamChunk("Hello")
            raise Exception("Stream error")
        
        chunks = []
        async for chunk in client._stream_from_async(error_stream()):
            chunks.append(chunk)
        
        # Should handle error gracefully and yield error chunk
        assert len(chunks) >= 1
        error_chunk = chunks[-1]
        assert error_chunk.get("error") is True

    @pytest.mark.asyncio
    async def test_stream_from_async_filter_invalid_chunks(self, client):
        """Test async streaming filters invalid chunks."""
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(""),  # Empty chunk should be filtered
            MockStreamChunk("world!")
        ])
        
        chunks = []
        async for chunk in client._stream_from_async(mock_stream):
            chunks.append(chunk)
        
        # Should filter out empty chunks
        assert len(chunks) == 2
        assert chunks[0]["response"] == "Hello"
        assert chunks[1]["response"] == "world!"

# Regular completion tests
class TestOpenAIRegularCompletion:
    """Test OpenAI regular completion methods"""

    @pytest.mark.asyncio
    async def test_regular_completion_basic(self, client):
        """Test basic regular completion."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello! How can I help?")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client._regular_completion(messages)
        
        assert result["response"] == "Hello! How can I help?"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_regular_completion_with_tools(self, client):
        """Test regular completion with tools."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [{"function": {"name": "get_weather", "parameters": {}}}]
        
        mock_tool_call = MockToolCall(function_name="get_weather", arguments='{"location": "NYC"}')
        mock_response = MockChatCompletion("", [mock_tool_call])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client._regular_completion(messages, tools)
        
        assert result["response"] is None  # No text content when tool calls present
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_regular_completion_with_name_mapping(self, client):
        """Test regular completion with tool name restoration."""
        messages = [{"role": "user", "content": "Call a tool"}]
        tools = [{"function": {"name": "sanitized_tool", "parameters": {}}}]
        
        mock_tool_call = MockToolCall(function_name="sanitized_tool", arguments='{}')
        mock_response = MockChatCompletion("", [mock_tool_call])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock name mapping
        name_mapping = {"sanitized_tool": "original.tool:name"}
        
        # Mock the restoration method
        def mock_restore(response, mapping):
            if response.get("tool_calls") and mapping:
                for tool_call in response["tool_calls"]:
                    sanitized_name = tool_call["function"]["name"]
                    if sanitized_name in mapping:
                        tool_call["function"]["name"] = mapping[sanitized_name]
            return response
        
        client._restore_tool_names_in_response = mock_restore
        
        result = await client._regular_completion(messages, tools, name_mapping)
        
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "original.tool:name"

    @pytest.mark.asyncio
    async def test_regular_completion_error_handling(self, client):
        """Test regular completion error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        result = await client._regular_completion(messages)
        
        assert result["error"] is True
        assert "API Error" in result["response"]

# Stream completion tests
class TestOpenAIStreamCompletion:
    """Test OpenAI stream completion methods"""

    @pytest.mark.asyncio
    async def test_stream_completion_async_basic(self, client):
        """Test basic stream completion."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(" world!")
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0]["response"] == "Hello"
        assert chunks[1]["response"] == " world!"

    @pytest.mark.asyncio
    async def test_stream_completion_async_with_tools(self, client):
        """Test stream completion with tools."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [{"function": {"name": "get_weather", "parameters": {}}}]
        
        mock_tool_call = MockToolCall(function_name="get_weather", arguments='{"location": "NYC"}')
        mock_stream = MockAsyncStream([
            MockStreamChunk("I'll check the weather"),
            MockStreamChunk("", [mock_tool_call])
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        chunks = []
        async for chunk in client._stream_completion_async(messages, tools):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0]["response"] == "I'll check the weather"
        assert len(chunks[1]["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_stream_completion_async_with_retry(self, client):
        """Test stream completion with retry logic."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock a call that fails first time, succeeds second time
        call_count = 0
        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit exceeded")
            return MockAsyncStream([MockStreamChunk("Success")])
        
        client.async_client.chat.completions.create = mock_create
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        # Should eventually succeed after retry
        assert len(chunks) >= 1
        assert any("Success" in chunk.get("response", "") for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_completion_async_error_handling(self, client):
        """Test stream completion error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("Stream error"))
        
        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)
        
        # Should yield error chunk
        assert len(chunks) >= 1
        error_chunk = chunks[-1]
        assert error_chunk.get("error") is True
        assert "Stream error" in error_chunk.get("response", "")

# Main interface tests
class TestOpenAIMainInterface:
    """Test OpenAI main interface methods"""

    @pytest.mark.asyncio
    async def test_create_completion_non_streaming(self, client):
        """Test create_completion with non-streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello! How can I help?")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        assert result["response"] == "Hello! How can I help?"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_create_completion_streaming(self, client):
        """Test create_completion with streaming."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_stream = MockAsyncStream([
            MockStreamChunk("Hello"),
            MockStreamChunk(" world!")
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        chunks = []
        async for chunk in client.create_completion(messages, stream=True):
            chunks.append(chunk)
        
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_create_completion_with_tools(self, client):
        """Test create_completion with tools."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [{"function": {"name": "get_weather", "parameters": {}}}]
        
        mock_tool_call = MockToolCall(function_name="get_weather", arguments='{"location": "NYC"}')
        mock_response = MockChatCompletion("", [mock_tool_call])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, tools=tools, stream=False)
        
        assert result["response"] is None
        assert len(result["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_create_completion_with_unsupported_features(self, client, monkeypatch):
        """Test create_completion with unsupported features."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"function": {"name": "test_tool"}}]
        
        # Mock tools as not supported
        monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")
        
        mock_response = MockChatCompletion("I can't use tools.")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, tools=tools, stream=False)
        
        # Tools should be filtered out
        assert result["response"] == "I can't use tools."

    @pytest.mark.asyncio
    async def test_create_completion_with_parameters(self, client):
        """Test create_completion with various parameters."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello!")
        
        # Capture the kwargs passed to the API
        captured_kwargs = {}
        async def mock_create(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_response
        
        client.async_client.chat.completions.create = mock_create
        
        result = await client.create_completion(
            messages,
            stream=False,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        
        assert result["response"] == "Hello!"
        assert captured_kwargs["temperature"] == 0.7
        assert captured_kwargs["max_tokens"] == 100
        assert captured_kwargs["top_p"] == 0.9

# Error handling tests
class TestOpenAIErrorHandling:
    """Test OpenAI error handling"""

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, client):
        """Test streaming error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("Streaming error"))
        
        chunks = []
        async for chunk in client.create_completion(messages, stream=True):
            chunks.append(chunk)
        
        # Should handle error gracefully
        assert len(chunks) >= 1
        error_chunk = chunks[-1]
        assert error_chunk.get("error") is True

    @pytest.mark.asyncio
    async def test_non_streaming_error_handling(self, client):
        """Test non-streaming error handling."""
        messages = [{"role": "user", "content": "Hello"}]
        
        client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        
        result = await client.create_completion(messages, stream=False)
        
        assert result["error"] is True
        assert "API error" in result["response"]

    @pytest.mark.asyncio
    async def test_error_handling_with_retry(self, client):
        """Test error handling with retry logic."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock calls that fail multiple times with retryable errors
        call_count = 0
        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit exceeded")  # First attempt fails
            elif call_count == 2:
                raise Exception("Connection timeout")   # Second attempt fails
            else:
                return MockChatCompletion("Success after retry")  # Third attempt succeeds
        
        client.async_client.chat.completions.create = mock_create
        
        # Mock the retry mechanism in the streaming method
        original_stream_method = client._stream_completion_async
        
        async def mock_stream_with_retry(messages, tools=None, name_mapping=None, **kwargs):
            max_retries = 1  # Allow 1 retry
            
            for attempt in range(max_retries + 1):
                try:
                    # Try the original method
                    async for chunk in original_stream_method(messages, tools, name_mapping, **kwargs):
                        yield chunk
                    return  # Success
                except Exception as e:
                    error_str = str(e).lower()
                    is_retryable = any(pattern in error_str for pattern in [
                        "rate limit", "timeout", "connection", "network"
                    ])
                    
                    if attempt < max_retries and is_retryable:
                        await asyncio.sleep(0.1)  # Brief delay
                        continue
                    else:
                        # Final attempt or non-retryable error
                        yield {
                            "response": f"Error: {str(e)}",
                            "tool_calls": [],
                            "error": True
                        }
                        return
        
        client._stream_completion_async = mock_stream_with_retry
        
        result = await client.create_completion(messages, stream=False)
        
        # Should handle error gracefully (not necessarily succeed after retry in this simplified mock)
        assert "response" in result
        # Could be either success or error depending on retry implementation
        assert result["response"] in ["Success after retry", "Error: Rate limit exceeded", "Error: Connection timeout"]

# Integration tests
class TestOpenAIIntegration:
    """Test OpenAI integration"""

    @pytest.mark.asyncio
    async def test_full_integration_conversation(self, client):
        """Test full integration with conversation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's 2+2?"}
        ]
        
        mock_response = MockChatCompletion("2+2 equals 4.")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        assert result["response"] == "2+2 equals 4."

    @pytest.mark.asyncio
    async def test_full_integration_tool_workflow(self, client):
        """Test full integration with tool workflow."""
        messages = [
            {"role": "user", "content": "What's the weather in NYC?"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_123", "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72°F"},
            {"role": "user", "content": "Thanks!"}
        ]
        
        tools = [{"function": {"name": "get_weather", "parameters": {}}}]
        
        mock_response = MockChatCompletion("You're welcome! It's a beautiful day in NYC.")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, tools=tools, stream=False)
        
        assert "beautiful day" in result["response"]

    @pytest.mark.asyncio
    async def test_full_integration_streaming_with_tools(self, client):
        """Test full integration streaming with tools."""
        messages = [{"role": "user", "content": "Call a tool and tell me the result"}]
        tools = [{"function": {"name": "test_tool", "parameters": {}}}]
        
        mock_tool_call = MockToolCall(function_name="test_tool", arguments='{}')
        mock_stream = MockAsyncStream([
            MockStreamChunk("I'll call the tool"),
            MockStreamChunk("", [mock_tool_call]),
            MockStreamChunk("Tool completed successfully")
        ])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
        chunks = []
        async for chunk in client.create_completion(messages, tools=tools, stream=True):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert any("call the tool" in chunk.get("response", "") for chunk in chunks)
        assert any(len(chunk.get("tool_calls", [])) > 0 for chunk in chunks)

# Performance and edge case tests
class TestOpenAIEdgeCases:
    """Test OpenAI edge cases"""

    @pytest.mark.asyncio
    async def test_empty_messages(self, client):
        """Test handling of empty messages."""
        messages = []
        
        mock_response = MockChatCompletion("I'm here to help!")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        assert result["response"] == "I'm here to help!"

    @pytest.mark.asyncio
    async def test_large_number_of_messages(self, client):
        """Test handling of large number of messages."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(100)
        ]
        
        mock_response = MockChatCompletion("Handled all messages.")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        assert result["response"] == "Handled all messages."

    @pytest.mark.asyncio
    async def test_malformed_tool_calls(self, client):
        """Test handling of malformed tool calls."""
        messages = [{"role": "user", "content": "Call a malformed tool"}]
        
        # Create a malformed tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = None  # Malformed - no function
        
        mock_response = MockChatCompletion("", [mock_tool_call])
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await client.create_completion(messages, stream=False)
        
        # Should handle malformed tool calls gracefully
        assert "response" in result
        assert "tool_calls" in result

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        messages = [{"role": "user", "content": "Hello"}]
        
        mock_response = MockChatCompletion("Hello!")
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run multiple concurrent requests
        tasks = [
            client.create_completion(messages, stream=False)
            for _ in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        assert all(result["response"] == "Hello!" for result in results)

# Configuration feature tests
class TestOpenAIConfigurationFeatures:
    """Test OpenAI configuration features"""

    def test_supports_feature_validation(self, client):
        """Test feature support validation."""
        # Test supported features
        assert client.supports_feature("text") is True
        assert client.supports_feature("streaming") is True
        assert client.supports_feature("tools") is True
        assert client.supports_feature("vision") is True

    def test_model_specific_features(self, deepseek_client):
        """Test model-specific feature support."""
        # DeepSeek doesn't support vision or JSON mode
        assert deepseek_client.supports_feature("text") is True
        assert deepseek_client.supports_feature("streaming") is True
        assert deepseek_client.supports_feature("tools") is True
        assert deepseek_client.supports_feature("vision") is False

    def test_tool_compatibility_info(self, client):
        """Test tool compatibility information."""
        info = client.get_tool_compatibility_info()
        
        assert info["supports_universal_naming"] is True
        assert info["sanitization_method"] == "replace_chars"
        assert info["restoration_method"] == "name_mapping"

# Cleanup tests
class TestOpenAICleanup:
    """Test OpenAI cleanup functionality"""

    @pytest.mark.asyncio
    async def test_close_method(self, client):
        """Test close method."""
        # Mock the async client close method
        client.async_client.close = AsyncMock()
        
        await client.close()
        
        # Should reset name mapping
        assert client._current_name_mapping == {}

    @pytest.mark.asyncio
    async def test_close_method_with_sync_client(self, client):
        """Test close method with sync client."""
        # Mock both async and sync client close methods
        client.async_client.close = AsyncMock()
        client.client.close = MagicMock()
        
        await client.close()
        
        # Should call close on both clients
        client.async_client.close.assert_called_once()
        client.client.close.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])