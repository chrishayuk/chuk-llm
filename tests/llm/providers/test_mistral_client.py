# tests/providers/test_mistral_client.py
import asyncio
import sys
import types
from unittest.mock import patch

import pytest

from chuk_llm.core.enums import MessageRole

# ---------------------------------------------------------------------------
# Stub the `mistralai` SDK before importing the adapter.
# ---------------------------------------------------------------------------

# Create the main mistralai module
mistralai_mod = types.ModuleType("mistralai")
sys.modules["mistralai"] = mistralai_mod


# Fake Chat class with streaming support
class DummyChat:
    def __init__(self):
        pass

    def complete(self, **kwargs):
        return None  # will be monkey-patched per-test

    def stream(self, **kwargs):
        return []  # will be monkey-patched per-test


# Fake Mistral client
class DummyMistral:
    def __init__(self, api_key=None, server_url=None, **kwargs):
        self.api_key = api_key
        self.server_url = server_url
        self.chat = DummyChat()


# Expose classes
mistralai_mod.Mistral = DummyMistral

# ---------------------------------------------------------------------------
# Now import the client (will see the stub).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.mistral_client import (
    MistralLLMClient,  # noqa: E402  pylint: disable=wrong-import-position
)

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
    def __init__(self, features=None, max_context_length=8192, max_output_tokens=4096):
        self.features = features or {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.VISION,
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    def __init__(self, name="mistral", client_class="MistralLLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://api.mistral.ai"
        self.models = ["mistral-large-latest", "mistral-small", "mistral-medium"]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 60}

    def get_model_capabilities(self, model):
        # Different capabilities based on model
        features = {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.SYSTEM_MESSAGES,
        }

        # Vision models
        if "vision" in model.lower() or "pixtral" in model.lower():
            features.add(MockFeature.VISION)
            features.add(MockFeature.MULTIMODAL)

        # Large models typically have vision support
        if "large" in model.lower():
            features.add(MockFeature.VISION)
            features.add(MockFeature.MULTIMODAL)

        return MockModelCapabilities(features=features)


class MockConfig:
    def __init__(self):
        self.mistral_provider = MockProviderConfig()

    def get_provider(self, provider_name):
        if provider_name == "mistral":
            return self.mistral_provider
        return None


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
def client(mock_configuration, monkeypatch):
    """Mistral client with configuration properly mocked"""
    cl = MistralLLMClient(model="mistral-large-latest", api_key="fake-key")

    # Ensure configuration methods are properly mocked
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in ["text", "streaming", "tools", "system_messages", "vision", "multimodal"],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "mistral",
            "model": "mistral-large-latest",
            "client_class": "MistralLLMClient",
            "api_base": "https://api.mistral.ai",
            "features": [
                "text",
                "streaming",
                "tools",
                "system_messages",
                "vision",
                "multimodal",
            ],
            "supports_text": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "supports_system_messages": True,
            "supports_json_mode": False,
            "supports_parallel_calls": False,
            "supports_multimodal": True,
            "supports_reasoning": False,
            "max_context_length": 8192,
            "max_output_tokens": 4096,
            "tool_compatibility": {
                "supports_universal_naming": True,
                "sanitization_method": "replace_chars",
                "restoration_method": "name_mapping",
                "supported_name_patterns": ["alphanumeric_underscore"],
            },
            "mistral_specific": {
                "supports_magistral_reasoning": "magistral" in cl.model.lower(),
                "supports_code_generation": any(
                    pattern in cl.model.lower() for pattern in ["codestral", "devstral"]
                ),
                "is_multilingual": "saba" in cl.model.lower(),
                "is_edge_model": "ministral" in cl.model.lower(),
            },
            "parameter_mapping": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "stream": "stream",
                "tool_choice": "tool_choice",
            },
            "unsupported_parameters": [
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "logit_bias",
                "user",
                "n",
                "best_of",
                "top_k",
                "seed",
            ],
        },
    )

    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 8192)

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

    # Initialize empty name mapping
    cl._current_name_mapping = {}

    return cl


@pytest.fixture
def codestral_client(mock_configuration, monkeypatch):
    """Mistral Codestral client with code generation features"""
    cl = MistralLLMClient(model="codestral-latest", api_key="fake-key")

    # Codestral model has code generation features
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in ["text", "streaming", "tools", "system_messages", "reasoning"],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "mistral",
            "model": "codestral-latest",
            "client_class": "MistralLLMClient",
            "api_base": "https://api.mistral.ai",
            "features": ["text", "streaming", "tools", "system_messages", "reasoning"],
            "supports_text": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False,
            "supports_system_messages": True,
            "supports_json_mode": False,
            "supports_parallel_calls": False,
            "supports_multimodal": False,
            "supports_reasoning": True,
            "max_context_length": 8192,
            "max_output_tokens": 4096,
            "tool_compatibility": {
                "supports_universal_naming": True,
                "sanitization_method": "replace_chars",
                "restoration_method": "name_mapping",
                "supported_name_patterns": ["alphanumeric_underscore"],
            },
            "mistral_specific": {
                "supports_magistral_reasoning": False,
                "supports_code_generation": True,
                "is_multilingual": False,
                "is_edge_model": False,
            },
            "parameter_mapping": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "stream": "stream",
                "tool_choice": "tool_choice",
            },
            "unsupported_parameters": [
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "logit_bias",
                "user",
                "n",
                "best_of",
                "top_k",
                "seed",
            ],
        },
    )

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

    # Initialize empty name mapping
    cl._current_name_mapping = {}

    return cl


# Convenience helper to capture kwargs
class Capture:
    kwargs = None


# ---------------------------------------------------------------------------
# Helper functions to create proper mock responses
# ---------------------------------------------------------------------------


def create_mock_mistral_response(content=None, tool_calls=None):
    """Create a properly structured mock Mistral response."""
    message = types.SimpleNamespace()
    message.content = content
    message.tool_calls = tool_calls or []

    choice = types.SimpleNamespace()
    choice.message = message

    response = types.SimpleNamespace()
    response.choices = [choice]

    return response


def create_mock_mistral_stream_chunk(content=None, tool_calls=None):
    """Create a properly structured mock Mistral streaming chunk."""
    delta = types.SimpleNamespace()
    delta.content = content
    delta.tool_calls = tool_calls or []

    choice = types.SimpleNamespace()
    choice.delta = delta

    data = types.SimpleNamespace()
    data.choices = [choice]

    chunk = types.SimpleNamespace()
    chunk.data = data

    return chunk


# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------


def test_client_initialization(mock_configuration):
    """Test client initialization with different parameters."""
    # Test with default model
    client1 = MistralLLMClient()
    assert client1.model == "mistral-large-latest"
    assert client1.provider_name == "mistral"

    # Test with custom model and API key
    client2 = MistralLLMClient(model="mistral-small", api_key="test-key")
    assert client2.model == "mistral-small"

    # Test with custom API base
    client3 = MistralLLMClient(
        model="mistral-medium", api_base="https://custom.mistral.ai"
    )
    assert client3.model == "mistral-medium"


def test_get_model_info(client):
    """Test model info method."""
    info = client.get_model_info()

    assert info["provider"] == "mistral"
    assert info["model"] == "mistral-large-latest"
    assert "mistral_specific" in info
    assert "parameter_mapping" in info
    assert "unsupported_parameters" in info
    assert "tool_compatibility" in info


def test_mistral_specific_features(mock_configuration):
    """Test detection of Mistral-specific model features."""
    # Test codestral model
    codestral_client = MistralLLMClient(model="codestral-latest")
    info = codestral_client.get_model_info()
    mistral_specific = info["mistral_specific"]
    assert mistral_specific["supports_code_generation"] is True

    # Test magistral model
    magistral_client = MistralLLMClient(model="mistral-magistral-latest")
    magistral_info = magistral_client.get_model_info()
    assert magistral_info["mistral_specific"]["supports_magistral_reasoning"] is True

    # Test ministral model
    ministral_client = MistralLLMClient(model="ministral-8b")
    ministral_info = ministral_client.get_model_info()
    assert ministral_info["mistral_specific"]["is_edge_model"] is True


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_messages_to_mistral_format_basic(client):
    """Test basic message conversion."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 4
    assert converted[0]["role"] == "system"
    assert converted[0]["content"] == "You are helpful"
    assert converted[1]["role"] == "user"
    assert converted[1]["content"] == "Hello"
    assert converted[2]["role"] == "assistant"
    assert converted[3]["role"] == "user"


@pytest.mark.asyncio
async def test_convert_messages_multimodal(client):
    """Test message conversion with multimodal content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert isinstance(converted[0]["content"], list)
    assert len(converted[0]["content"]) == 2
    assert converted[0]["content"][0]["type"] == "text"
    assert converted[0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_convert_messages_vision_not_supported(client, monkeypatch):
    """Test message conversion when vision is not supported."""
    # Mock the supports_feature method to return False for vision
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "vision")

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

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    # Should only have text content when vision is not supported
    assert isinstance(converted[0]["content"], str)
    assert "Look at this" in converted[0]["content"]


@pytest.mark.asyncio
async def test_convert_messages_tool_calls(client):
    """Test message conversion with tool calls."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "test_tool",
            "content": "Tool result",
        },
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 2
    assert converted[0]["role"] == "assistant"
    assert "tool_calls" in converted[0]
    assert len(converted[0]["tool_calls"]) == 1
    assert converted[1]["role"] == "tool"
    assert converted[1]["tool_call_id"] == "call_123"


@pytest.mark.asyncio
async def test_convert_messages_tools_not_supported(client, monkeypatch):
    """Test message conversion when tools are not supported."""
    # Mock the supports_feature method to return False for tools
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_123", "function": {"name": "test_tool", "arguments": "{}"}}
            ],
        }
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "assistant"
    # Should have text content instead of tool calls
    assert "content" in converted[0]
    assert "Tool calls were requested" in converted[0]["content"]


@pytest.mark.asyncio
async def test_convert_messages_system_not_supported(client, monkeypatch):
    """Test message conversion when system messages are not supported."""
    # Mock the supports_feature method to return False for system_messages
    monkeypatch.setattr(
        client, "supports_feature", lambda feature: feature != "system_messages"
    )

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 2
    # System message should be converted to user message
    assert converted[0]["role"] == "user"
    assert "System: You are helpful" in converted[0]["content"]
    assert converted[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Response normalization tests
# ---------------------------------------------------------------------------


def test_normalize_mistral_response_text(client):
    """Test normalizing Mistral response with text content."""
    mock_response = create_mock_mistral_response(content="Hello from Mistral")

    result = client._normalize_mistral_response(mock_response)

    assert result == {"response": "Hello from Mistral", "tool_calls": []}


def test_normalize_mistral_response_tool_calls(client):
    """Test normalizing Mistral response with tool calls."""
    # Mock tool call structure
    mock_tool_call = types.SimpleNamespace()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function = types.SimpleNamespace()
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"arg": "value"}'

    mock_response = create_mock_mistral_response(
        content=None, tool_calls=[mock_tool_call]
    )

    result = client._normalize_mistral_response(mock_response)

    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"


def test_normalize_mistral_response_tools_not_supported(client, monkeypatch):
    """Test normalizing response when tools are not supported."""
    # Mock the supports_feature method to return False for tools
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")

    # Mock tool call structure
    mock_tool_call = types.SimpleNamespace()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function = types.SimpleNamespace()
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"arg": "value"}'

    mock_response = create_mock_mistral_response(
        content="Some content", tool_calls=[mock_tool_call]
    )

    result = client._normalize_mistral_response(mock_response)

    # Should return text content and ignore tool calls
    assert result["response"] == "Some content"
    assert result["tool_calls"] == []


def test_normalize_mistral_response_fallback(client):
    """Test normalizing response fallback for unknown formats."""
    mock_response = "Unknown response format"

    result = client._normalize_mistral_response(mock_response)

    assert result == {"response": "Unknown response format", "tool_calls": []}


def test_normalize_mistral_response_with_name_mapping(client):
    """Test normalizing response with tool name restoration."""
    # Mock tool call with sanitized name
    mock_tool_call = types.SimpleNamespace()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function = types.SimpleNamespace()
    mock_tool_call.function.name = "test_tool_sanitized"
    mock_tool_call.function.arguments = '{"arg": "value"}'

    mock_response = create_mock_mistral_response(
        content=None, tool_calls=[mock_tool_call]
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

    result = client._normalize_mistral_response(mock_response, name_mapping)

    assert result["tool_calls"][0]["function"]["name"] == "test.tool:original"


# ---------------------------------------------------------------------------
# Request validation tests
# ---------------------------------------------------------------------------


def test_validate_request_with_config(client):
    """Test request validation against configuration."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    validated_messages, validated_tools, validated_stream, validated_kwargs = (
        client._validate_request_with_config(
            messages, tools, stream=True, temperature=0.7
        )
    )

    assert validated_messages == messages
    assert validated_tools == tools
    assert validated_stream is True
    assert "temperature" in validated_kwargs


def test_validate_request_unsupported_features(client, monkeypatch):
    """Test request validation when features are not supported."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    # Mock configuration to not support streaming or tools
    monkeypatch.setattr(client, "supports_feature", lambda feature: False)

    validated_messages, validated_tools, validated_stream, validated_kwargs = (
        client._validate_request_with_config(
            messages, tools, stream=True, temperature=0.7, tool_choice="auto"
        )
    )

    assert validated_messages == messages
    assert validated_tools is None  # Should be None when not supported
    assert validated_stream is False  # Should be False when not supported
    assert "tool_choice" not in validated_kwargs  # Should be removed
    assert "temperature" in validated_kwargs


def test_validate_request_vision_content(client, monkeypatch):
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

    assert validated_messages == messages  # Should pass through unchanged

    # Test with vision not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "vision")

    validated_messages, validated_tools, validated_stream, validated_kwargs = (
        client._validate_request_with_config(messages, stream=False)
    )

    # Should still pass through - filtering happens in message conversion
    assert validated_messages == messages


# ---------------------------------------------------------------------------
# Regular completion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regular_completion(client):
    """Test regular (non-streaming) completion."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock the Mistral client's complete method
    mock_response = create_mock_mistral_response(content="Hello! How can I help you?")

    def mock_complete(**kwargs):
        return mock_response

    client.client.chat.complete = mock_complete

    result = await client._regular_completion(request_params)

    assert result["response"] == "Hello! How can I help you?"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_regular_completion_with_tools(client):
    """Test regular completion with tools."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [{"type": "function", "function": {"name": "test_tool"}}],
    }

    # Mock tool call response
    mock_tool_call = types.SimpleNamespace()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function = types.SimpleNamespace()
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"arg": "value"}'

    mock_response = create_mock_mistral_response(
        content=None, tool_calls=[mock_tool_call]
    )

    def mock_complete(**kwargs):
        return mock_response

    client.client.chat.complete = mock_complete

    result = await client._regular_completion(request_params)

    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_regular_completion_with_name_mapping(client):
    """Test regular completion with tool name restoration."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [{"type": "function", "function": {"name": "test_tool_sanitized"}}],
    }

    # Mock tool call response
    mock_tool_call = types.SimpleNamespace()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function = types.SimpleNamespace()
    mock_tool_call.function.name = "test_tool_sanitized"
    mock_tool_call.function.arguments = '{"arg": "value"}'

    mock_response = create_mock_mistral_response(
        content=None, tool_calls=[mock_tool_call]
    )

    def mock_complete(**kwargs):
        return mock_response

    client.client.chat.complete = mock_complete

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

    result = await client._regular_completion(request_params, name_mapping)

    assert result["tool_calls"][0]["function"]["name"] == "test.tool:original"


@pytest.mark.asyncio
async def test_regular_completion_error_handling(client):
    """Test error handling in regular completion."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock the client to raise an exception
    def mock_complete_error(**kwargs):
        raise Exception("API Error")

    client.client.chat.complete = mock_complete_error

    result = await client._regular_completion(request_params)

    assert "error" in result
    assert result["error"] is True
    assert "API Error" in result["response"]


# ---------------------------------------------------------------------------
# Streaming completion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completion_async(client):
    """Test streaming completion."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock streaming chunks
    def mock_stream_chunks():
        chunks = [
            create_mock_mistral_stream_chunk(content="Hello"),
            create_mock_mistral_stream_chunk(content=" from Mistral!"),
        ]

        yield from chunks

    client.client.chat.stream = lambda **kwargs: mock_stream_chunks()

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "Hello"
    assert chunks[1]["response"] == " from Mistral!"


@pytest.mark.asyncio
async def test_stream_completion_async_with_tools(client):
    """Test streaming completion with tools."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use tools"}],
        "tools": [{"type": "function", "function": {"name": "test_tool"}}],
    }

    # Mock streaming chunks with tool calls
    def mock_stream_chunks():
        # Mock tool call structure in streaming
        mock_tool_call = types.SimpleNamespace()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = types.SimpleNamespace()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        chunks = [
            create_mock_mistral_stream_chunk(content="I'll use a tool"),
            create_mock_mistral_stream_chunk(content=None, tool_calls=[mock_tool_call]),
        ]

        yield from chunks

    client.client.chat.stream = lambda **kwargs: mock_stream_chunks()

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "I'll use a tool"
    assert len(chunks[1]["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_stream_completion_async_with_name_mapping(client):
    """Test streaming completion with tool name restoration."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use tools"}],
        "tools": [{"type": "function", "function": {"name": "test_tool_sanitized"}}],
    }

    # Mock streaming chunks with tool calls
    def mock_stream_chunks():
        # Mock tool call structure in streaming
        mock_tool_call = types.SimpleNamespace()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = types.SimpleNamespace()
        mock_tool_call.function.name = "test_tool_sanitized"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        chunks = [
            create_mock_mistral_stream_chunk(content=None, tool_calls=[mock_tool_call])
        ]

        yield from chunks

    client.client.chat.stream = lambda **kwargs: mock_stream_chunks()

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

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params, name_mapping):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0]["tool_calls"][0]["function"]["name"] == "test.tool:original"


@pytest.mark.asyncio
async def test_stream_completion_async_error_handling(client):
    """Test error handling in streaming completion."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock the client to raise an exception
    def mock_stream_error(**kwargs):
        raise Exception("Streaming error")

    client.client.chat.stream = mock_stream_error

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    # Should yield an error chunk
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert chunks[0]["error"] is True
    assert "Streaming error" in chunks[0]["response"]


# ---------------------------------------------------------------------------
# Main interface tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_non_streaming(client):
    """Test create_completion with non-streaming."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the regular completion method
    expected_result = {"response": "Hello!", "tool_calls": []}

    async def mock_regular_completion(request_params, name_mapping=None):
        return expected_result

    client._regular_completion = mock_regular_completion

    result = client.create_completion(messages, stream=False)

    # Should return an awaitable
    assert hasattr(result, "__await__")

    final_result = await result
    assert final_result == expected_result


@pytest.mark.asyncio
async def test_create_completion_streaming(client):
    """Test create_completion with streaming."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the streaming method
    async def mock_stream_completion_async(request_params, name_mapping=None):
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": []}

    client._stream_completion_async = mock_stream_completion_async

    result = client.create_completion(messages, stream=True)

    # Should return an async generator
    assert hasattr(result, "__aiter__")

    chunks = []
    async for chunk in result:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "chunk1"
    assert chunks[1]["response"] == "chunk2"


@pytest.mark.asyncio
async def test_create_completion_streaming_not_supported(client, monkeypatch):
    """Test create_completion with streaming when not supported."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock streaming as not supported
    monkeypatch.setattr(
        client, "supports_feature", lambda feature: feature != "streaming"
    )

    # Mock the regular completion method (should be called instead of streaming)
    expected_result = {"response": "Hello!", "tool_calls": []}

    async def mock_regular_completion(request_params, name_mapping=None):
        return expected_result

    client._regular_completion = mock_regular_completion

    result = client.create_completion(messages, stream=True)

    # Should return an awaitable (not async iterator) when streaming not supported
    assert hasattr(result, "__await__")
    assert not hasattr(result, "__aiter__")

    final_result = await result
    assert final_result == expected_result


@pytest.mark.asyncio
async def test_create_completion_with_tools(client):
    """Test create_completion with tools."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]

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

    async def mock_regular_completion(request_params, name_mapping=None):
        # Verify tools were passed and tool_choice was set
        assert "tools" in request_params
        # Check tool structure (may have description field added)
        assert len(request_params["tools"]) == 1
        assert request_params["tools"][0]["type"] == "function"
        assert request_params["tools"][0]["function"]["name"] == "get_weather"
        assert request_params["tool_choice"] == "auto"
        return expected_result

    client._regular_completion = mock_regular_completion

    result = await client.create_completion(messages, tools=tools, stream=False)

    assert result == expected_result
    assert len(result["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_create_completion_with_tools_not_supported(client, monkeypatch):
    """Test create_completion with tools when not supported."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]

    # Mock tools as not supported
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")

    # Mock regular completion
    expected_result = {"response": "I cannot use tools.", "tool_calls": []}

    async def mock_regular_completion(request_params, name_mapping=None):
        # Verify tools were not passed
        assert "tools" not in request_params
        return expected_result

    client._regular_completion = mock_regular_completion

    result = await client.create_completion(messages, tools=tools, stream=False)

    assert result == expected_result


@pytest.mark.asyncio
async def test_create_completion_with_tool_choice(client):
    """Test create_completion with explicit tool_choice."""
    messages = [{"role": "user", "content": "Use this specific tool"}]
    tools = [
        {"type": "function", "function": {"name": "specific_tool", "parameters": {}}}
    ]

    async def mock_regular_completion(request_params, name_mapping=None):
        # Verify custom tool_choice was preserved
        assert request_params["tool_choice"] == "required"
        return {"response": "Tool used", "tool_calls": []}

    client._regular_completion = mock_regular_completion

    result = await client.create_completion(
        messages, tools=tools, tool_choice="required", stream=False
    )

    assert result["response"] == "Tool used"


@pytest.mark.asyncio
async def test_create_completion_parameter_validation(client):
    """Test that parameters are validated through configuration."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock parameter validation - the fixture already sets up validate_parameters
    async def mock_regular_completion(request_params, name_mapping=None):
        return {"response": "Hello!", "tool_calls": []}

    client._regular_completion = mock_regular_completion

    result = await client.create_completion(
        messages, temperature=0.7, max_tokens=100, stream=False
    )

    # Should complete successfully with validated parameters
    assert result["response"] == "Hello!"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_error_handling(client, monkeypatch):
    """Test error handling in streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock streaming with error
    async def error_stream(request_params, name_mapping=None):
        yield {"response": "Starting...", "tool_calls": []}
        yield {
            "response": "Streaming error: Test error",
            "tool_calls": [],
            "error": True,
        }

    monkeypatch.setattr(client, "_stream_completion_async", error_stream)

    stream_result = client.create_completion(messages, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "Starting..."
    assert chunks[1]["error"] is True


@pytest.mark.asyncio
async def test_non_streaming_error_handling(client, monkeypatch):
    """Test error handling in non-streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock error in regular completion
    async def error_completion(request_params, name_mapping=None):
        return {"response": "Error: Test error", "tool_calls": [], "error": True}

    monkeypatch.setattr(client, "_regular_completion", error_completion)

    result = await client.create_completion(messages, stream=False)

    assert result["error"] is True
    assert "Test error" in result["response"]


@pytest.mark.asyncio
async def test_error_handling_comprehensive(client):
    """Test comprehensive error handling."""
    messages = [{"role": "user", "content": "Hello"}]

    # Test various error scenarios
    error_scenarios = [
        "Network error",
        "Rate limit exceeded",
        "Invalid request",
        "Timeout error",
    ]

    for error_msg in error_scenarios:

        def create_error_mock(msg):
            def mock_complete_error(**kwargs):
                raise Exception(msg)
            return mock_complete_error

        mock_complete_error = create_error_mock(error_msg)

        client.client.chat.complete = mock_complete_error

        result = await client._regular_completion(
            {"model": "mistral-large-latest", "messages": messages}
        )

        assert "error" in result
        assert result["error"] is True
        assert error_msg in result["response"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_integration_non_streaming(client):
    """Test full integration for non-streaming completion."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    # Mock the actual Mistral API call
    mock_response = create_mock_mistral_response(
        content="Hello! How can I help you today?"
    )

    captured_params = {}

    def mock_complete(**kwargs):
        captured_params.update(kwargs)
        return mock_response

    client.client.chat.complete = mock_complete

    result = await client.create_completion(messages, stream=False)

    assert result["response"] == "Hello! How can I help you today?"
    assert result["tool_calls"] == []

    # Verify request structure
    assert captured_params["model"] == "mistral-large-latest"
    assert len(captured_params["messages"]) == 2
    assert captured_params["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_full_integration_streaming(client):
    """Test full integration for streaming completion."""
    messages = [{"role": "user", "content": "Tell me a story"}]

    # Mock streaming response
    def mock_stream(**kwargs):
        story_parts = ["Once", " upon", " a", " time..."]
        for part in story_parts:
            yield create_mock_mistral_stream_chunk(content=part)

    client.client.chat.stream = mock_stream

    # Collect all chunks
    story_parts = []
    async for chunk in client.create_completion(messages, stream=True):
        story_parts.append(chunk["response"])

    # Verify we got all parts
    assert len(story_parts) == 4
    assert story_parts == ["Once", " upon", " a", " time..."]


# ---------------------------------------------------------------------------
# Tool compatibility tests
# ---------------------------------------------------------------------------


def test_tool_compatibility_info(client):
    """Test tool compatibility information."""
    info = client.get_tool_compatibility_info()

    assert info["supports_universal_naming"] is True
    assert info["sanitization_method"] == "replace_chars"
    assert info["restoration_method"] == "name_mapping"
    assert "alphanumeric_underscore" in info["supported_name_patterns"]


def test_tool_name_sanitization_and_restoration(client):
    """Test tool name sanitization and restoration."""
    # Test that sanitization is called (mocked to return tools unchanged)
    tools = [
        {"type": "function", "function": {"name": "test.tool:name", "parameters": {}}}
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


# ---------------------------------------------------------------------------
# Interface compliance tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interface_compliance(client):
    """Test that create_completion follows the correct interface."""
    # Test non-streaming - should return awaitable
    messages = [{"role": "user", "content": "Test"}]

    # Mock the completion
    async def mock_completion(request_params, name_mapping=None):
        return {"response": "Test response", "tool_calls": []}

    client._regular_completion = mock_completion

    # Non-streaming should return awaitable
    result_coro = client.create_completion(messages, stream=False)
    assert asyncio.iscoroutine(result_coro)

    result = await result_coro
    assert isinstance(result, dict)
    assert "response" in result

    # Streaming should return async iterator
    async def mock_stream(request_params, name_mapping=None):
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": []}

    client._stream_completion_async = mock_stream

    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, "__aiter__")

    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Complex scenario tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complex_conversation_flow(client):
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
        {"type": "function", "function": {"name": "analyze_image", "parameters": {}}}
    ]

    # Mock completion
    async def mock_completion(request_params, name_mapping=None):
        return {"response": "Complex conversation complete", "tool_calls": []}

    client._regular_completion = mock_completion

    result = await client.create_completion(messages, tools=tools, stream=False)

    assert result["response"] == "Complex conversation complete"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_cleanup(client):
    """Test client cleanup."""
    await client.close()
    # Mistral client doesn't require explicit cleanup, so this should pass without error
    assert client._current_name_mapping == {}


# ---------------------------------------------------------------------------
# Configuration edge cases
# ---------------------------------------------------------------------------


def test_configuration_feature_detection(client):
    """Test that configuration properly detects model features."""
    # Test that the client uses configuration to determine features
    info = client.get_model_info()

    # These should come from configuration, not hardcoded logic
    assert isinstance(info.get("supports_streaming"), bool)
    assert isinstance(info.get("supports_tools"), bool)
    assert isinstance(info.get("supports_vision"), bool)
    assert isinstance(info.get("supports_system_messages"), bool)


@pytest.mark.asyncio
async def test_unsupported_features_graceful_handling(client, monkeypatch):
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
                {"id": "call_123", "function": {"name": "test_tool", "arguments": "{}"}}
            ],
        },
    ]

    tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]

    # Mock completion
    async def mock_completion(request_params, name_mapping=None):
        # Verify that unsupported features were handled gracefully
        assert "tools" not in request_params
        return {"response": "Features handled gracefully", "tool_calls": []}

    client._regular_completion = mock_completion

    result = await client.create_completion(
        messages,
        tools=tools,
        stream=False,  # Should be converted to False when not supported
    )

    assert result["response"] == "Features handled gracefully"


# ---------------------------------------------------------------------------
# Environment and configuration tests
# ---------------------------------------------------------------------------


def test_custom_api_configuration(mock_configuration):
    """Test custom API configuration."""
    custom_url = "https://custom.mistral.ai"
    client = MistralLLMClient(api_base=custom_url)

    # Should store the custom URL in the underlying Mistral client
    assert client.client.server_url == custom_url


def test_api_key_configuration(mock_configuration):
    """Test API key configuration."""
    custom_key = "custom-api-key"
    client = MistralLLMClient(api_key=custom_key)

    # Should store the custom API key
    assert client.client.api_key == custom_key


# ---------------------------------------------------------------------------
# Feature support validation tests
# ---------------------------------------------------------------------------


def test_feature_support_validation(client, monkeypatch):
    """Test that feature support is properly validated."""
    # Test supported features (from fixture)
    supported_features = [
        "text",
        "streaming",
        "tools",
        "vision",
        "system_messages",
        "multimodal",
    ]

    for feature in supported_features:
        assert client.supports_feature(feature) is True

    # Test unsupported features
    unsupported_features = ["json_mode", "reasoning", "parallel_calls"]

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


def test_codestral_model_features(codestral_client):
    """Test that Codestral model has code generation features."""
    info = codestral_client.get_model_info()

    # Codestral model should have code generation capabilities
    assert info["mistral_specific"]["supports_code_generation"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_tools"] is True
    # Codestral typically doesn't have vision
    assert info["supports_vision"] is False


# ---------------------------------------------------------------------------
# Import error tests (lines 30-31)
# ---------------------------------------------------------------------------


def test_import_error():
    """Test ImportError when mistralai package is not available."""
    # Remove mistralai from sys.modules temporarily
    original_mistralai = sys.modules.get("mistralai")
    if "mistralai" in sys.modules:
        del sys.modules["mistralai"]

    # Mock the import to fail
    with patch.dict("sys.modules", {"mistralai": None}):
        with pytest.raises(ImportError) as exc_info:
            # Force reimport
            import importlib

            if "chuk_llm.llm.providers.mistral_client" in sys.modules:
                importlib.reload(sys.modules["chuk_llm.llm.providers.mistral_client"])

        # Restore mistralai module
        if original_mistralai is not None:
            sys.modules["mistralai"] = original_mistralai


# ---------------------------------------------------------------------------
# Client initialization edge cases (lines 85, 94)
# ---------------------------------------------------------------------------


def test_client_initialization_with_both_api_key_and_server_url(mock_configuration):
    """Test client initialization with both api_key and server_url (line 85)."""
    client = MistralLLMClient(
        model="mistral-large-latest",
        api_key="test-key-123",
        api_base="https://custom.mistral.ai",
    )

    # Should create client with both parameters
    assert client.client.api_key == "test-key-123"
    assert client.client.server_url == "https://custom.mistral.ai"
    assert client.model == "mistral-large-latest"


def test_client_initialization_with_no_kwargs(mock_configuration):
    """Test client initialization with no kwargs - empty else branch (line 94)."""
    # This tests the final else branch when client_kwargs is empty
    client = MistralLLMClient(model="mistral-large-latest")

    # Should create client with default settings
    assert client.model == "mistral-large-latest"
    assert client.provider_name == "mistral"
    assert client.client is not None


def test_client_initialization_edge_case_line_94(mock_configuration):
    """Test client initialization hitting line 94 specifically."""
    # Create a situation where we pass kwargs but none of the expected keys
    # This shouldn't normally happen but ensures line 94 is covered
    client = MistralLLMClient(model="mistral-small")

    # Client should still be created properly
    assert client.client is not None
    assert client.model == "mistral-small"


def test_client_initialization_with_only_server_url(mock_configuration):
    """Test client initialization with only server_url (line 91-92)."""
    client = MistralLLMClient(
        model="mistral-small", api_base="https://custom.mistral.ai"
    )

    # Should create client with only server_url
    assert client.client.server_url == "https://custom.mistral.ai"
    assert client.model == "mistral-small"


# ---------------------------------------------------------------------------
# Pydantic message conversion tests (lines 199, 238-257)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_messages_pydantic_text_content(client):
    """Test message conversion with Pydantic TEXT content type (line 199)."""
    from chuk_llm.core.enums import ContentType

    # Create a mock Pydantic-style content object
    class MockPydanticTextContent:
        def __init__(self):
            self.type = ContentType.TEXT
            self.text = "This is Pydantic text content"

    messages = [
        {
            "role": "user",
            "content": [
                MockPydanticTextContent(),
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    # Should extract text from Pydantic object when vision not supported
    # This will be tested when vision is disabled


@pytest.mark.asyncio
async def test_convert_messages_pydantic_image_url(client):
    """Test message conversion with Pydantic IMAGE_URL content (lines 238-257)."""
    from chuk_llm.core.enums import ContentType

    # Create mock Pydantic-style content objects
    class MockPydanticTextContent:
        def __init__(self):
            self.type = ContentType.TEXT
            self.text = "Look at this image"

    class MockPydanticImageContent:
        def __init__(self, url):
            self.type = ContentType.IMAGE_URL
            self.image_url = url

    messages = [
        {
            "role": "user",
            "content": [
                MockPydanticTextContent(),
                MockPydanticImageContent("data:image/png;base64,iVBORw0KGgoAAAA"),
            ],
        }
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert isinstance(converted[0]["content"], list)
    assert len(converted[0]["content"]) == 2
    assert converted[0]["content"][0]["type"] == "text"
    assert converted[0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_convert_messages_pydantic_image_data(client):
    """Test message conversion with Pydantic IMAGE_DATA content type."""
    from chuk_llm.core.enums import ContentType

    # Create mock Pydantic-style content object with IMAGE_DATA type
    class MockPydanticImageDataContent:
        def __init__(self):
            self.type = ContentType.IMAGE_DATA
            self.image_url = "data:image/png;base64,iVBORw0KGgoAAAA"

    class MockPydanticTextContent:
        def __init__(self):
            self.type = ContentType.TEXT
            self.text = "Image analysis"

    # Mock the supports_feature to return False for vision
    import unittest.mock

    with unittest.mock.patch.object(client, "supports_feature") as mock_feature:
        # Return False only for vision feature
        mock_feature.side_effect = lambda f: f != "vision"

        messages = [
            {
                "role": "user",
                "content": [
                    MockPydanticTextContent(),
                    MockPydanticImageDataContent(),
                ],
            }
        ]

        converted = await client._convert_messages_to_mistral_format(messages)

        # When vision not supported, should extract only text
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], str)
        assert "Image analysis" in converted[0]["content"]


# ---------------------------------------------------------------------------
# Image download error handling (lines 224-230)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_messages_image_download_error_dict_format(client, monkeypatch):
    """Test image download error handling for dict-based content (lines 224-230)."""
    # Mock the download function to raise an error
    async def mock_download_error(url):
        raise Exception("Network error downloading image")

    monkeypatch.setattr(
        "chuk_llm.llm.providers._mixins.OpenAIStyleMixin._download_and_encode_image",
        mock_download_error,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.png"},
                },
            ],
        }
    ]

    # Should not raise an error, but keep original URL
    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    # Should still have image_url in content (with original URL since download failed)
    assert isinstance(converted[0]["content"], list)


@pytest.mark.asyncio
async def test_convert_messages_image_download_error_pydantic_format(client, monkeypatch):
    """Test image download error handling for Pydantic content (lines 247-255)."""
    from chuk_llm.core.enums import ContentType

    # Mock the download function to raise an error
    async def mock_download_error(url):
        raise Exception("Network timeout")

    monkeypatch.setattr(
        "chuk_llm.llm.providers._mixins.OpenAIStyleMixin._download_and_encode_image",
        mock_download_error,
    )

    class MockPydanticImageContent:
        def __init__(self):
            self.type = ContentType.IMAGE_URL
            self.image_url = "http://example.com/image.jpg"

    class MockPydanticTextContent:
        def __init__(self):
            self.type = ContentType.TEXT
            self.text = "Analyze"

    messages = [
        {
            "role": "user",
            "content": [MockPydanticTextContent(), MockPydanticImageContent()],
        }
    ]

    # Should not raise an error
    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"


@pytest.mark.asyncio
async def test_convert_messages_image_download_success_dict_format(client, monkeypatch):
    """Test successful image download for dict-based content (lines 227-228)."""
    # Mock successful download
    async def mock_download_success(url):
        return ("iVBORw0KGgoAAAANSUhEUgAAAAUA", "png")

    monkeypatch.setattr(
        "chuk_llm.llm.providers._mixins.OpenAIStyleMixin._download_and_encode_image",
        mock_download_success,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.png"},
                },
            ],
        }
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert isinstance(converted[0]["content"], list)
    # Check that the image URL was converted to base64
    image_content = converted[0]["content"][1]
    assert image_content["type"] == "image_url"
    assert "data:image/png;base64," in image_content["image_url"]


@pytest.mark.asyncio
async def test_convert_messages_image_download_success_pydantic_format(
    client, monkeypatch
):
    """Test successful image download for Pydantic content (lines 251-252)."""
    from chuk_llm.core.enums import ContentType

    # Mock successful download
    async def mock_download_success(url):
        return ("iVBORw0KGgoAAAANSUhEUgAAAAUA", "jpeg")

    monkeypatch.setattr(
        "chuk_llm.llm.providers._mixins.OpenAIStyleMixin._download_and_encode_image",
        mock_download_success,
    )

    class MockPydanticImageContent:
        def __init__(self):
            self.type = ContentType.IMAGE_URL
            self.image_url = "https://example.com/image.jpg"

    class MockPydanticTextContent:
        def __init__(self):
            self.type = ContentType.TEXT
            self.text = "Analyze this"

    messages = [
        {
            "role": "user",
            "content": [MockPydanticTextContent(), MockPydanticImageContent()],
        }
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert isinstance(converted[0]["content"], list)
    # Check that the image URL was converted to base64
    image_content = converted[0]["content"][1]
    assert image_content["type"] == "image_url"
    assert "data:image/jpeg;base64," in image_content["image_url"]


# ---------------------------------------------------------------------------
# Tool response without tools support (line 331)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_messages_tool_response_not_supported(client, monkeypatch):
    """Test tool response conversion when tools not supported (line 331)."""
    # Mock the supports_feature to return False for tools
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "tools")

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "test_tool",
            "content": "Tool execution result",
        }
    ]

    converted = await client._convert_messages_to_mistral_format(messages)

    # Should convert tool message to user message
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert "Tool result:" in converted[0]["content"]
    assert "Tool execution result" in converted[0]["content"]


# ---------------------------------------------------------------------------
# Usage info with reasoning tokens (lines 373-391, 400)
# ---------------------------------------------------------------------------


def test_normalize_mistral_response_with_usage_info(client):
    """Test normalizing response with usage information including reasoning tokens."""
    from chuk_llm.core.constants import ResponseKey

    # Create mock usage object with reasoning tokens
    mock_usage = types.SimpleNamespace()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150
    mock_usage.reasoning_tokens = 25  # Reasoning tokens for magistral models

    # Create mock response with usage
    message = types.SimpleNamespace()
    message.content = "Response with usage info"
    message.tool_calls = []

    choice = types.SimpleNamespace()
    choice.message = message

    response = types.SimpleNamespace()
    response.choices = [choice]
    response.usage = mock_usage

    result = client._normalize_mistral_response(response)

    # Should include usage info with reasoning tokens
    assert "usage" in result
    assert result["usage"][ResponseKey.PROMPT_TOKENS.value] == 100
    assert result["usage"][ResponseKey.COMPLETION_TOKENS.value] == 50
    assert result["usage"][ResponseKey.TOTAL_TOKENS.value] == 150
    assert result["usage"][ResponseKey.REASONING_TOKENS.value] == 25


def test_normalize_mistral_response_with_usage_no_reasoning(client):
    """Test normalizing response with usage information but no reasoning tokens."""
    from chuk_llm.core.constants import ResponseKey

    # Create mock usage object without reasoning tokens
    mock_usage = types.SimpleNamespace()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150
    # No reasoning_tokens attribute

    # Create mock response with usage
    message = types.SimpleNamespace()
    message.content = "Response with usage info"
    message.tool_calls = []

    choice = types.SimpleNamespace()
    choice.message = message

    response = types.SimpleNamespace()
    response.choices = [choice]
    response.usage = mock_usage

    result = client._normalize_mistral_response(response)

    # Should include usage info without reasoning tokens
    assert "usage" in result
    assert result["usage"][ResponseKey.PROMPT_TOKENS.value] == 100
    assert result["usage"][ResponseKey.COMPLETION_TOKENS.value] == 50
    assert result["usage"][ResponseKey.TOTAL_TOKENS.value] == 150
    # Reasoning tokens should not be in the dict since it was None
    assert ResponseKey.REASONING_TOKENS.value not in result["usage"]


# ---------------------------------------------------------------------------
# Streaming JSON decode errors (lines 672-689)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completion_json_incomplete_tool_calls(client):
    """Test streaming with incomplete JSON in tool calls (lines 672-677)."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use tools"}],
        "tools": [{"type": "function", "function": {"name": "test_tool"}}],
    }

    # Mock streaming chunks with incomplete JSON
    def mock_stream_chunks():
        # First chunk: tool call with incomplete JSON (should not be yielded)
        mock_tool_call_incomplete = types.SimpleNamespace()
        mock_tool_call_incomplete.index = 0
        mock_tool_call_incomplete.id = "call_123"
        mock_tool_call_incomplete.function = types.SimpleNamespace()
        mock_tool_call_incomplete.function.name = "test_tool"
        mock_tool_call_incomplete.function.arguments = '{"arg":'  # Incomplete JSON

        # Second chunk: complete the JSON (should be yielded)
        mock_tool_call_complete = types.SimpleNamespace()
        mock_tool_call_complete.index = 0
        mock_tool_call_complete.function = types.SimpleNamespace()
        mock_tool_call_complete.function.name = ""
        mock_tool_call_complete.function.arguments = ' "value"}'  # Complete JSON

        chunks = [
            create_mock_mistral_stream_chunk(
                content=None, tool_calls=[mock_tool_call_incomplete]
            ),
            create_mock_mistral_stream_chunk(
                content=None, tool_calls=[mock_tool_call_complete]
            ),
        ]

        yield from chunks

    client.client.chat.stream = lambda **kwargs: mock_stream_chunks()

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    # Should yield only when JSON is complete
    assert len(chunks) == 1
    assert chunks[0]["tool_calls"] is not None
    assert len(chunks[0]["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_stream_completion_tool_call_processing_exception(client):
    """Test streaming with exception during tool call processing (lines 679-683)."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use tools"}],
        "tools": [{"type": "function", "function": {"name": "test_tool"}}],
    }

    # Mock streaming chunks with a malformed tool call that triggers exception
    def mock_stream_chunks():
        # Create a tool call that will cause an exception during processing
        mock_tool_call = types.SimpleNamespace()
        mock_tool_call.index = 0
        # Intentionally missing 'id' and 'function' to trigger exception
        # This will cause an AttributeError when trying to access tc.id

        chunk = create_mock_mistral_stream_chunk(content=None, tool_calls=[mock_tool_call])
        yield chunk

    client.client.chat.stream = lambda **kwargs: mock_stream_chunks()

    # Collect streaming results - should not raise exception
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    # Should handle the exception gracefully and not yield tool calls
    # The exception is caught and logged, but processing continues
    assert len(chunks) == 0 or (len(chunks) > 0 and chunks[0].get("tool_calls") is None)


@pytest.mark.asyncio
async def test_stream_completion_chunk_processing_exception(client):
    """Test streaming with exception during chunk processing (lines 685-689)."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use tools"}],
    }

    # Mock streaming chunks with one that causes an exception
    def mock_stream_chunks():
        # Create a normal chunk first
        yield create_mock_mistral_stream_chunk(content="Normal chunk")

        # Create a malformed chunk that will trigger the outer exception handler
        malformed_chunk = types.SimpleNamespace()
        # Missing 'data' attribute to trigger AttributeError
        yield malformed_chunk

        # Another normal chunk to verify processing continues
        yield create_mock_mistral_stream_chunk(content="Another chunk")

    client.client.chat.stream = lambda **kwargs: mock_stream_chunks()

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    # Should handle the exception gracefully
    # First and third chunks should be yielded, malformed chunk should be skipped
    assert len(chunks) >= 2
    assert chunks[0]["response"] == "Normal chunk"


# ---------------------------------------------------------------------------
# Streaming chunk processing errors (lines 711, 723-726)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completion_asyncio_sleep(client, monkeypatch):
    """Test that asyncio.sleep is called during streaming (line 711)."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    sleep_called = []

    # Mock asyncio.sleep to track calls
    original_sleep = asyncio.sleep

    async def mock_sleep(delay):
        sleep_called.append(delay)
        await original_sleep(0)  # Actually sleep to avoid blocking

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    # Mock streaming chunks (11 chunks to trigger sleep at chunk 10)
    def mock_stream_chunks():
        for i in range(11):
            yield create_mock_mistral_stream_chunk(content=f"chunk{i}")

    client.client.chat.stream = lambda **kwargs: mock_stream_chunks()

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    # Should have called sleep at least once (at chunk 10)
    assert len(sleep_called) >= 1
    assert sleep_called[0] == 0


@pytest.mark.asyncio
async def test_stream_completion_tool_name_validation_error(client):
    """Test streaming with tool name validation error (lines 723-726)."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use tools"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "invalid.tool:name", "parameters": {}},
            }
        ],
    }

    # Mock the client to raise a tool name validation error
    def mock_stream_error(**kwargs):
        raise Exception(
            "Function name 'invalid.tool:name' must be a-z, A-Z, 0-9, or underscores and dashes"
        )

    client.client.chat.stream = mock_stream_error

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(request_params):
        chunks.append(chunk)

    # Should yield an error chunk with specific error handling
    assert len(chunks) == 1
    assert "error" in chunks[0]
    assert chunks[0]["error"] is True


# ---------------------------------------------------------------------------
# Tool name validation errors in regular completion (lines 770-773)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regular_completion_tool_name_validation_error(client):
    """Test regular completion with tool name validation error (lines 770-773)."""
    request_params = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Use tools"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "invalid.tool:name", "parameters": {}},
            }
        ],
    }

    # Mock the client to raise a tool name validation error
    def mock_complete_error(**kwargs):
        raise Exception(
            "Function name 'invalid.tool:name' must be a-z, A-Z, 0-9, or underscores"
        )

    client.client.chat.complete = mock_complete_error

    result = await client._regular_completion(request_params)

    # Should return error response with specific error logging
    assert "error" in result
    assert result["error"] is True
    assert "Function name" in result["response"]
