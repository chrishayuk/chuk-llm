# tests/providers/test_anthropic_client.py
import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub the `anthropic` SDK before importing the adapter.
# ---------------------------------------------------------------------------

anthropic_mod = types.ModuleType("anthropic")
sys.modules["anthropic"] = anthropic_mod

# Create submodule anthropic.types so that "from anthropic.types import X" works
anthropic_types_mod = types.ModuleType("anthropic.types")
sys.modules["anthropic.types"] = anthropic_types_mod
anthropic_mod.types = anthropic_types_mod


# Minimal ToolUseBlock type stub
class ToolUseBlock(dict):
    pass


# Expose ToolUseBlock under both anthropic and anthropic.types namespaces
anthropic_mod.ToolUseBlock = ToolUseBlock
anthropic_types_mod.ToolUseBlock = ToolUseBlock


# Mock stream context manager for async streaming
class MockStreamContext:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aiter__(self):
        for event in self.events:
            yield event


# Fake Messages client with stream support
class _DummyMessages:
    def create(self, *args, **kwargs):
        return None  # will be monkey-patched per-test

    def stream(self, *args, **kwargs):
        return MockStreamContext([])  # will be monkey-patched per-test


# Fake AsyncAnthropic client
class DummyAsyncAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = _DummyMessages()


# Fake sync Anthropic client (for backwards compatibility)
class DummyAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = _DummyMessages()


# Add both sync and async clients
anthropic_mod.Anthropic = DummyAnthropic
anthropic_mod.AsyncAnthropic = DummyAsyncAnthropic

# ---------------------------------------------------------------------------
# Now import the client (will see the stub).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.anthropic_client import (
    AnthropicLLMClient,
    _parse_claude_response,
)  # noqa: E402  pylint: disable=wrong-import-position

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
        self, features=None, max_context_length=200000, max_output_tokens=4096
    ):
        self.features = features or {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    def __init__(self, name="anthropic", client_class="AnthropicLLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://api.anthropic.com"
        self.models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 50}

    def get_model_capabilities(self, model):
        # Claude models typically have comprehensive features
        features = {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
        }

        # Haiku models have additional capabilities like JSON mode
        if "haiku" in model.lower() or "opus" in model.lower():
            features.add(MockFeature.JSON_MODE)
            features.add(MockFeature.REASONING)

        return MockModelCapabilities(features=features)


class MockConfig:
    def __init__(self):
        self.anthropic_provider = MockProviderConfig()

    def get_provider(self, provider_name):
        if provider_name == "anthropic":
            return self.anthropic_provider
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
    """Anthropic client with configuration properly mocked"""
    cl = AnthropicLLMClient(model="claude-3-5-sonnet-20241022", api_key="fake-key")

    # Ensure configuration methods are properly mocked
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in ["text", "streaming", "tools", "vision", "system_messages", "multimodal"],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "client_class": "AnthropicLLMClient",
            "api_base": "https://api.anthropic.com",
            "features": [
                "text",
                "streaming",
                "tools",
                "vision",
                "system_messages",
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
            "max_context_length": 200000,
            "max_output_tokens": 4096,
            "tool_compatibility": {
                "supports_universal_naming": True,
                "sanitization_method": "replace_chars",
                "restoration_method": "name_mapping",
                "supported_name_patterns": ["alphanumeric_underscore"],
            },
            "vision_format": "universal_image_url",
            "supported_parameters": ["temperature", "max_tokens", "top_p", "stream"],
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
                "response_format",
            ],
        },
    )

    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 200000)

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
def haiku_client(mock_configuration, monkeypatch):
    """Anthropic Haiku client with advanced features"""
    cl = AnthropicLLMClient(model="claude-3-5-haiku-20241022", api_key="fake-key")

    # Haiku model has additional features
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
            "reasoning",
        ],
    )

    monkeypatch.setattr(
        cl,
        "get_model_info",
        lambda: {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "client_class": "AnthropicLLMClient",
            "api_base": "https://api.anthropic.com",
            "features": [
                "text",
                "streaming",
                "tools",
                "vision",
                "system_messages",
                "multimodal",
                "json_mode",
                "reasoning",
            ],
            "supports_text": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "supports_system_messages": True,
            "supports_json_mode": True,
            "supports_parallel_calls": False,
            "supports_multimodal": True,
            "supports_reasoning": True,
            "max_context_length": 200000,
            "max_output_tokens": 4096,
            "tool_compatibility": {
                "supports_universal_naming": True,
                "sanitization_method": "replace_chars",
                "restoration_method": "name_mapping",
                "supported_name_patterns": ["alphanumeric_underscore"],
            },
            "vision_format": "universal_image_url",
            "supported_parameters": ["temperature", "max_tokens", "top_p", "stream"],
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
                "response_format",
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
# Response parsing tests
# ---------------------------------------------------------------------------


def test_parse_claude_response_text_only():
    """Test parsing Claude response with text only."""
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "Hello world"
    mock_response.content = [mock_text_block]

    result = _parse_claude_response(mock_response)

    assert result["response"] == "Hello world"
    assert result["tool_calls"] == []


def test_parse_claude_response_with_tool_calls():
    """Test parsing Claude response with tool calls."""
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "call_123"
    mock_tool_block.name = "get_weather"
    mock_tool_block.input = {"city": "NYC"}

    mock_response = MagicMock()
    mock_response.content = [mock_tool_block]

    result = _parse_claude_response(mock_response)

    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "NYC" in result["tool_calls"][0]["function"]["arguments"]


def test_parse_claude_response_mixed_content():
    """Test parsing Claude response with mixed text and tool calls."""
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "I'll check the weather"

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "call_123"
    mock_tool_block.name = "get_weather"
    mock_tool_block.input = {"city": "NYC"}

    mock_response = MagicMock()
    mock_response.content = [mock_text_block, mock_tool_block]

    result = _parse_claude_response(mock_response)

    # When tool calls are present, response should be None (following OpenAI pattern)
    assert result["response"] is None
    assert len(result["tool_calls"]) == 1


def test_parse_claude_response_empty():
    """Test parsing Claude response with empty content."""
    mock_response = MagicMock()
    mock_response.content = []

    result = _parse_claude_response(mock_response)

    assert result["response"] == ""
    assert result["tool_calls"] == []


def test_parse_claude_response_no_content():
    """Test parsing Claude response with no content attribute."""
    mock_response = MagicMock()
    # Remove content attribute
    if hasattr(mock_response, "content"):
        del mock_response.content

    result = _parse_claude_response(mock_response)

    assert result["response"] == ""
    assert result["tool_calls"] == []


# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------


def test_client_initialization(mock_configuration):
    """Test client initialization with different parameters."""
    # Test with default model
    client1 = AnthropicLLMClient()
    assert client1.model == "claude-3-5-sonnet-20241022"

    # Test with custom model and API key
    client2 = AnthropicLLMClient(model="claude-test", api_key="test-key")
    assert client2.model == "claude-test"

    # Test with API base
    client3 = AnthropicLLMClient(
        model="claude-test", api_base="https://custom.anthropic.com"
    )
    assert client3.model == "claude-test"


def test_get_model_info(client):
    """Test model info method."""
    info = client.get_model_info()

    assert info["provider"] == "anthropic"
    assert info["model"] == "claude-3-5-sonnet-20241022"
    assert "vision_format" in info
    assert "supported_parameters" in info
    assert "unsupported_parameters" in info
    assert "tool_compatibility" in info


# ---------------------------------------------------------------------------
# Tool conversion tests
# ---------------------------------------------------------------------------


def test_convert_tools(client):
    """Test tool conversion to Anthropic format."""
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]

    converted = client._convert_tools(openai_tools)

    assert len(converted) == 1
    assert converted[0]["name"] == "get_weather"
    assert converted[0]["description"] == "Get weather info"
    assert "input_schema" in converted[0]


def test_convert_tools_empty(client):
    """Test tool conversion with empty/None input."""
    assert client._convert_tools(None) == []
    assert client._convert_tools([]) == []


def test_convert_tools_error_handling(client):
    """Test tool conversion with malformed tools."""
    malformed_tools = [
        {"type": "function"},  # Missing function key
        {"function": {}},  # Missing name
        {"function": {"name": "valid_tool", "parameters": {}}},  # Valid
    ]

    converted = client._convert_tools(malformed_tools)
    assert len(converted) == 3  # Should handle all tools, using fallbacks


def test_convert_tools_nested_structure(client):
    """Test tool conversion with nested tool structure."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]

    converted = client._convert_tools(tools)

    assert len(converted) == 1
    assert converted[0]["name"] == "get_weather"
    assert converted[0]["input_schema"]["properties"]["city"]["type"] == "string"


# ---------------------------------------------------------------------------
# Parameter filtering tests
# ---------------------------------------------------------------------------


def test_filter_anthropic_params(client):
    """Test parameter filtering for Anthropic compatibility."""
    params = {
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "frequency_penalty": 0.5,  # Unsupported
        "stop": ["stop"],  # Unsupported
        "custom_param": "value",  # Unknown
    }

    filtered = client._filter_anthropic_params(params)

    assert "temperature" in filtered
    assert "max_tokens" in filtered
    assert "top_p" in filtered
    assert "frequency_penalty" not in filtered
    assert "stop" not in filtered
    assert "custom_param" not in filtered


def test_filter_anthropic_params_temperature_cap(client):
    """Test temperature capping at 1.0."""
    params = {"temperature": 2.0}
    filtered = client._filter_anthropic_params(params)
    assert filtered["temperature"] == 1.0


def test_filter_anthropic_params_adds_max_tokens(client):
    """Test that max_tokens is added if missing."""
    params = {"temperature": 0.7}
    filtered = client._filter_anthropic_params(params)
    assert "max_tokens" in filtered
    assert filtered["max_tokens"] <= 4096  # Should be reasonable default


def test_filter_anthropic_params_with_limits(client):
    """Test parameter filtering with configuration limits."""
    params = {"max_tokens": 10000}  # Above limit
    filtered = client._filter_anthropic_params(params)
    assert filtered["max_tokens"] == 4096  # Should be capped to limit


# ---------------------------------------------------------------------------
# JSON mode tests
# ---------------------------------------------------------------------------


def test_check_json_mode(haiku_client):
    """Test JSON mode detection with Haiku client that supports JSON mode."""
    # Test OpenAI-style response_format
    kwargs = {"response_format": {"type": "json_object"}}
    instruction = haiku_client._check_json_mode(kwargs)
    assert instruction is not None
    assert "JSON" in instruction

    # Test custom json mode instruction
    kwargs = {"_json_mode_instruction": "Custom JSON instruction"}
    instruction = haiku_client._check_json_mode(kwargs)
    assert instruction == "Custom JSON instruction"

    # Test no JSON mode
    kwargs = {}
    instruction = haiku_client._check_json_mode(kwargs)
    assert instruction is None


def test_check_json_mode_not_supported(client):
    """Test JSON mode when not supported by model."""
    # Sonnet model doesn't support JSON mode according to our mock
    kwargs = {"response_format": {"type": "json_object"}}
    instruction = client._check_json_mode(kwargs)
    assert instruction is None  # Should return None when not supported


# ---------------------------------------------------------------------------
# Message splitting tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_for_anthropic_async_basic(client):
    """Test basic message splitting."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]

    system_txt, anthropic_messages = await client._split_for_anthropic_async(messages)

    assert system_txt == "You are helpful"
    assert len(anthropic_messages) == 3  # Excluding system message
    assert anthropic_messages[0]["role"] == "user"
    assert anthropic_messages[1]["role"] == "assistant"
    assert anthropic_messages[2]["role"] == "user"


@pytest.mark.asyncio
async def test_split_for_anthropic_async_multimodal(client):
    """Test message splitting with multimodal content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]

    system_txt, anthropic_messages = await client._split_for_anthropic_async(messages)

    assert system_txt == ""
    assert len(anthropic_messages) == 1
    assert anthropic_messages[0]["role"] == "user"
    assert isinstance(anthropic_messages[0]["content"], list)

    # Check that image was converted to Anthropic format
    content = anthropic_messages[0]["content"]
    has_image = any(item.get("type") == "image" for item in content)
    assert has_image


@pytest.mark.asyncio
async def test_split_for_anthropic_async_multimodal_not_supported(client, monkeypatch):
    """Test message splitting with multimodal content when vision is not supported."""
    # Mock vision as not supported
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

    system_txt, anthropic_messages = await client._split_for_anthropic_async(messages)

    assert system_txt == ""
    assert len(anthropic_messages) == 1
    # Should only contain text content when vision not supported
    content = anthropic_messages[0]["content"]
    has_image = any(item.get("type") == "image" for item in content)
    assert not has_image


@pytest.mark.asyncio
async def test_split_for_anthropic_async_tool_calls(client):
    """Test message splitting with tool calls."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
    ]

    system_txt, anthropic_messages = await client._split_for_anthropic_async(messages)

    assert len(anthropic_messages) == 2
    assert anthropic_messages[0]["role"] == "assistant"
    assert anthropic_messages[0]["content"][0]["type"] == "tool_use"
    assert anthropic_messages[1]["role"] == "user"  # Tool response becomes user message
    assert anthropic_messages[1]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_split_for_anthropic_async_multiple_systems(client):
    """Test message splitting with multiple system messages."""
    messages = [
        {"role": "system", "content": "System prompt 1"},
        {"role": "system", "content": "System prompt 2"},
        {"role": "user", "content": "Hello"},
    ]

    system_txt, anthropic_messages = await client._split_for_anthropic_async(messages)

    # Should combine system messages
    assert "System prompt 1" in system_txt
    assert "System prompt 2" in system_txt
    assert len(anthropic_messages) == 1
    assert anthropic_messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Vision format conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_universal_vision_to_anthropic_async_data_url():
    """Test converting data URL to Anthropic format."""
    content_item = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        },
    }

    result = await AnthropicLLMClient._convert_universal_vision_to_anthropic_async(
        content_item
    )

    assert result["type"] == "image"
    assert result["source"]["type"] == "base64"
    assert result["source"]["media_type"] == "image/png"
    assert "data" in result["source"]


@pytest.mark.asyncio
async def test_convert_universal_vision_to_anthropic_async_invalid_data_url():
    """Test converting invalid data URL."""
    content_item = {"type": "image_url", "image_url": {"url": "data:invalid"}}

    result = await AnthropicLLMClient._convert_universal_vision_to_anthropic_async(
        content_item
    )

    assert result["type"] == "text"
    assert "Invalid image format" in result["text"]


@pytest.mark.asyncio
async def test_convert_universal_vision_to_anthropic_async_external_url():
    """Test converting external URL (should attempt download)."""
    content_item = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.png"},
    }

    # Mock the download function to avoid actual network calls
    with patch.object(AnthropicLLMClient, "_download_image_to_base64") as mock_download:
        mock_download.side_effect = Exception("Network error")

        result = await AnthropicLLMClient._convert_universal_vision_to_anthropic_async(
            content_item
        )

        assert result["type"] == "text"
        assert "Could not load image" in result["text"]


@pytest.mark.asyncio
async def test_convert_universal_vision_to_anthropic_async_string_url():
    """Test converting string URL format."""
    content_item = {
        "type": "image_url",
        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    }

    result = await AnthropicLLMClient._convert_universal_vision_to_anthropic_async(
        content_item
    )

    assert result["type"] == "image"
    assert result["source"]["type"] == "base64"
    assert result["source"]["media_type"] == "image/png"


# ---------------------------------------------------------------------------
# Regular completion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regular_completion_async(client):
    """Test regular (non-streaming) completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the async client's create method
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "Hello! How can I help you?"
    mock_response.content = [mock_text_block]

    async def mock_create(**kwargs):
        return mock_response

    client.async_client.messages.create = mock_create

    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        anth_tools=[],
        filtered_params={},
        name_mapping={},
    )

    assert result["response"] == "Hello! How can I help you?"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_regular_completion_async_with_system(client):
    """Test regular completion with system instruction."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are a helpful assistant."

    # Mock the async client's create method
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "Hello! I'm here to help."
    mock_response.content = [mock_text_block]

    captured_payload = {}

    async def mock_create(**kwargs):
        captured_payload.update(kwargs)
        return mock_response

    client.async_client.messages.create = mock_create

    result = await client._regular_completion_async(
        system=system,
        json_instruction=None,
        messages=messages,
        anth_tools=[],
        filtered_params={},
        name_mapping={},
    )

    assert result["response"] == "Hello! I'm here to help."
    assert captured_payload.get("system") == system


@pytest.mark.asyncio
async def test_regular_completion_async_with_json_instruction(client):
    """Test regular completion with JSON mode instruction."""
    messages = [{"role": "user", "content": "Give me JSON"}]
    json_instruction = "Respond with valid JSON only."

    # Mock the async client's create method
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = '{"result": "success"}'
    mock_response.content = [mock_text_block]

    captured_payload = {}

    async def mock_create(**kwargs):
        captured_payload.update(kwargs)
        return mock_response

    client.async_client.messages.create = mock_create

    result = await client._regular_completion_async(
        system=None,
        json_instruction=json_instruction,
        messages=messages,
        anth_tools=[],
        filtered_params={},
        name_mapping={},
    )

    assert result["response"] == '{"result": "success"}'
    assert json_instruction in captured_payload.get("system", "")


@pytest.mark.asyncio
async def test_regular_completion_async_error_handling(client):
    """Test error handling in regular completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the client to raise an exception
    async def mock_create_error(**kwargs):
        raise Exception("API Error")

    client.async_client.messages.create = mock_create_error

    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        anth_tools=[],
        filtered_params={},
        name_mapping={},
    )

    assert "error" in result
    assert result["error"] is True
    assert "API Error" in result["response"]


# ---------------------------------------------------------------------------
# Streaming completion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_completion_async(client):
    """Test streaming completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock streaming events
    class MockEvent:
        def __init__(self, event_type, **kwargs):
            self.type = event_type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockDelta:
        def __init__(self, text):
            self.text = text

    mock_events = [
        MockEvent("content_block_delta", delta=MockDelta("Hello")),
        MockEvent("content_block_delta", delta=MockDelta(" world!")),
    ]

    # Mock the stream context manager
    class MockStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def __aiter__(self):
            for event in mock_events:
                yield event

    def mock_stream_create(**kwargs):
        return MockStream()

    client.async_client.messages.stream = mock_stream_create

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        anth_tools=[],
        filtered_params={},
        name_mapping={},
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "Hello"
    assert chunks[1]["response"] == " world!"


@pytest.mark.asyncio
async def test_stream_completion_async_with_tool_calls(client):
    """Test streaming completion with tool calls."""
    messages = [{"role": "user", "content": "Call a tool"}]

    # Mock streaming events with tool use
    class MockEvent:
        def __init__(self, event_type, **kwargs):
            self.type = event_type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockContentBlock:
        def __init__(self):
            self.type = "tool_use"
            self.id = "call_123"
            self.name = "test_tool"
            self.input = {"arg": "value"}

    mock_events = [MockEvent("content_block_start", content_block=MockContentBlock())]

    # Mock the stream context manager
    class MockStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def __aiter__(self):
            for event in mock_events:
                yield event

    def mock_stream_create(**kwargs):
        return MockStream()

    client.async_client.messages.stream = mock_stream_create

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        anth_tools=[{"name": "test_tool"}],
        filtered_params={},
        name_mapping={},
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0]["response"] == ""
    assert len(chunks[0]["tool_calls"]) == 1
    assert chunks[0]["tool_calls"][0]["function"]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_stream_completion_async_error_handling(client):
    """Test error handling in streaming completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the streaming to raise an error
    def mock_stream_create_error(**kwargs):
        raise Exception("Streaming error")

    client.async_client.messages.stream = mock_stream_create_error

    # Collect streaming results
    chunks = []
    async for chunk in client._stream_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        anth_tools=[],
        filtered_params={},
        name_mapping={},
    ):
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

    async def mock_regular_completion_async(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        return expected_result

    client._regular_completion_async = mock_regular_completion_async

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
    async def mock_stream_completion_async(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
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

    async def mock_regular_completion_async(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        return expected_result

    client._regular_completion_async = mock_regular_completion_async

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

    async def mock_regular_completion_async(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        # Verify tools were converted
        assert len(anth_tools) == 1
        assert anth_tools[0]["name"] == "get_weather"
        return expected_result

    client._regular_completion_async = mock_regular_completion_async

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

    async def mock_regular_completion_async(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        # Verify tools were not passed
        assert len(anth_tools) == 0
        return expected_result

    client._regular_completion_async = mock_regular_completion_async

    result = await client.create_completion(messages, tools=tools, stream=False)

    assert result == expected_result


@pytest.mark.asyncio
async def test_create_completion_with_max_tokens(client):
    """Test create_completion with max_tokens parameter."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock regular completion to check parameters
    async def mock_regular_completion_async(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        # Verify max_tokens was included
        assert "max_tokens" in filtered_params
        assert filtered_params["max_tokens"] == 500
        return {"response": "Hello!", "tool_calls": []}

    client._regular_completion_async = mock_regular_completion_async

    result = await client.create_completion(messages, max_tokens=500, stream=False)

    assert result["response"] == "Hello!"


@pytest.mark.asyncio
async def test_create_completion_with_system_param(client):
    """Test create_completion with system parameter."""
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are a helpful assistant."

    # Mock regular completion to check system handling
    async def mock_regular_completion_async(
        system_arg,
        json_instruction,
        messages,
        anth_tools,
        filtered_params,
        name_mapping,
    ):
        assert system_arg == system
        return {"response": "Hello!", "tool_calls": []}

    client._regular_completion_async = mock_regular_completion_async

    result = await client.create_completion(messages, system=system, stream=False)

    assert result["response"] == "Hello!"


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

    # Mock the actual Anthropic API call
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "Hello! How can I help you today?"
    mock_response.content = [mock_text_block]

    captured_payload = {}

    async def mock_create(**kwargs):
        captured_payload.update(kwargs)
        return mock_response

    client.async_client.messages.create = mock_create

    result = await client.create_completion(messages, stream=False)

    assert result["response"] == "Hello! How can I help you today?"
    assert result["tool_calls"] == []

    # Verify payload structure
    assert captured_payload["model"] == "claude-3-5-sonnet-20241022"
    assert captured_payload["system"] == "You are helpful"
    assert len(captured_payload["messages"]) == 1
    assert captured_payload["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_full_integration_streaming(client):
    """Test full integration for streaming completion."""
    messages = [{"role": "user", "content": "Tell me a story"}]

    # Mock streaming response
    class MockEvent:
        def __init__(self, event_type, **kwargs):
            self.type = event_type
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockDelta:
        def __init__(self, text):
            self.text = text

    mock_events = [
        MockEvent("content_block_delta", delta=MockDelta("Once")),
        MockEvent("content_block_delta", delta=MockDelta(" upon")),
        MockEvent("content_block_delta", delta=MockDelta(" a")),
        MockEvent("content_block_delta", delta=MockDelta(" time...")),
    ]

    class MockStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def __aiter__(self):
            for event in mock_events:
                yield event

    def mock_stream_create(**kwargs):
        return MockStream()

    client.async_client.messages.stream = mock_stream_create

    # Collect all chunks
    story_parts = []
    async for chunk in client.create_completion(messages, stream=True):
        story_parts.append(chunk["response"])

    # Verify we got all parts
    assert len(story_parts) == 4
    assert story_parts == ["Once", " upon", " a", " time..."]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_error_handling(client):
    """Test error handling in streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock streaming with error
    async def error_stream(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        yield {"response": "Starting...", "tool_calls": []}
        yield {
            "response": "Streaming error: Test error",
            "tool_calls": [],
            "error": True,
        }

    client._stream_completion_async = error_stream

    stream_result = client.create_completion(messages, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "Starting..."
    assert chunks[1]["error"] is True


@pytest.mark.asyncio
async def test_non_streaming_error_handling(client):
    """Test error handling in non-streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock error in regular completion
    async def error_completion(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        return {"response": "Error: Test error", "tool_calls": [], "error": True}

    client._regular_completion_async = error_completion

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

        async def mock_create_error(**kwargs):
            raise Exception(error_msg)

        client.async_client.messages.create = mock_create_error

        result = await client._regular_completion_async(
            system=None,
            json_instruction=None,
            messages=messages,
            anth_tools=[],
            filtered_params={},
            name_mapping={},
        )

        assert "error" in result
        assert result["error"] is True
        assert error_msg in result["response"]


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


def test_response_with_tool_name_restoration(client):
    """Test response parsing with tool name restoration."""
    # Set up name mapping
    client._current_name_mapping = {"get_weather_data": "weather.api:get_data"}

    # Mock response with sanitized tool name
    mock_response = {
        "response": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather_data", "arguments": "{}"},
            }
        ],
    }

    # Mock restoration to simulate real behavior
    def mock_restore(response, mapping):
        if response.get("tool_calls") and mapping:
            for tool_call in response["tool_calls"]:
                sanitized_name = tool_call["function"]["name"]
                if sanitized_name in mapping:
                    tool_call["function"]["name"] = mapping[sanitized_name]
        return response

    client._restore_tool_names_in_response = mock_restore

    restored_response = client._restore_tool_names_in_response(
        mock_response, client._current_name_mapping
    )

    # Verify restoration occurred
    assert (
        restored_response["tool_calls"][0]["function"]["name"] == "weather.api:get_data"
    )


# ---------------------------------------------------------------------------
# Complex scenario tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complex_message_conversion(client):
    """Test message conversion with complex scenarios."""
    messages = [
        {"role": "system", "content": "System prompt 1"},
        {"role": "system", "content": "System prompt 2"},  # Multiple system messages
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_123", "function": {"name": "test_tool", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
        {"role": "assistant", "content": "Based on the tool result..."},
    ]

    system_txt, anthropic_messages = await client._split_for_anthropic_async(messages)

    # Should combine system messages
    assert "System prompt 1" in system_txt
    assert "System prompt 2" in system_txt

    # Should have proper message count (excluding system messages)
    assert len(anthropic_messages) == 4

    # Check message types
    assert anthropic_messages[0]["role"] == "user"
    assert anthropic_messages[1]["role"] == "assistant"
    assert anthropic_messages[2]["role"] == "user"  # Tool response becomes user message
    assert anthropic_messages[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_interface_compliance(client):
    """Test that create_completion follows the correct interface."""
    # Test non-streaming - should return awaitable
    messages = [{"role": "user", "content": "Test"}]

    # Mock the completion
    async def mock_completion(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
        return {"response": "Test response", "tool_calls": []}

    client._regular_completion_async = mock_completion

    # Non-streaming should return awaitable
    result_coro = client.create_completion(messages, stream=False)
    assert asyncio.iscoroutine(result_coro)

    result = await result_coro
    assert isinstance(result, dict)
    assert "response" in result

    # Streaming should return async iterator
    async def mock_stream(
        system, json_instruction, messages, anth_tools, filtered_params, name_mapping
    ):
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


def test_haiku_model_features(haiku_client):
    """Test that Haiku model has additional features."""
    info = haiku_client.get_model_info()

    # Haiku model should have additional capabilities
    assert info["supports_json_mode"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_tools"] is True
    assert info["supports_vision"] is True
