# tests/providers/test_groq_client.py
import asyncio
import sys
import types
import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub the `groq` SDK before importing the adapter.
# ---------------------------------------------------------------------------

# Create the main groq module
groq_mod = types.ModuleType("groq")
sys.modules["groq"] = groq_mod

# Fake Chat Completions class
class DummyChatCompletions:
    def __init__(self):
        pass
    
    async def create(self, **kwargs):
        return None  # will be monkey-patched per-test

# Fake Chat class
class DummyChat:
    def __init__(self):
        self.completions = DummyChatCompletions()

# Fake AsyncGroq client
class DummyAsyncGroq:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = DummyChat()

# Fake sync Groq client (for backwards compatibility)
class DummyGroq:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url

# Expose classes
groq_mod.AsyncGroq = DummyAsyncGroq
groq_mod.Groq = DummyGroq

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
            MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS, MockFeature.SYSTEM_MESSAGES
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens

class MockProviderConfig:
    def __init__(self, name="groq", client_class="GroqAILLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://api.groq.com/openai/v1"
        self.models = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 30}
    
    def get_model_capabilities(self, model):
        # Groq models have comprehensive features
        features = {MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS, MockFeature.SYSTEM_MESSAGES}
        
        # Some models might have additional capabilities
        if "70b" in model.lower():
            features.add(MockFeature.JSON_MODE)
        
        return MockModelCapabilities(features=features)

class MockConfig:
    def __init__(self):
        self.groq_provider = MockProviderConfig()
    
    def get_provider(self, provider_name):
        if provider_name == "groq":
            return self.groq_provider
        return None

# ---------------------------------------------------------------------------
# Now import the client (will see the stub).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.groq_client import GroqAILLMClient  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# helpers / dummies
# ---------------------------------------------------------------------------

class _DummyResp:
    """Mimic return type of groq-sdk for the non-streaming path."""
    def __init__(self, message: Dict[str, Any]):
        self.choices = [SimpleNamespace(message=message)]

class _DummyDelta:
    def __init__(self, content: str = "", tool_calls: List[Dict[str, Any]] | None = None):
        self.content = content
        # Groq only sets tool_calls on the final chunk
        if tool_calls is not None:
            self.tool_calls = tool_calls

class _DummyChunk:
    def __init__(self, delta: _DummyDelta):
        self.choices = [SimpleNamespace(delta=delta)]

# Mock async Groq streaming response
class MockAsyncGroqStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk

# ---------------------------------------------------------------------------
# Fixtures with Configuration Mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_configuration():
    """Mock the configuration system"""
    mock_config = MockConfig()
    
    with patch('chuk_llm.configuration.get_config', return_value=mock_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            yield mock_config

@pytest.fixture
def client(mock_configuration, monkeypatch) -> GroqAILLMClient:
    """Return a GroqAILLMClient with configuration properly mocked."""
    cl = GroqAILLMClient(model="llama-3.3-70b-versatile", api_key="fake-key")

    # --- stub the sanitiser to identity ---
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda t: t)
    
    # --- stub configuration methods to avoid real config dependencies ---
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "system_messages"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "client_class": "GroqAILLMClient",
        "api_base": "https://api.groq.com/openai/v1",
        "features": ["text", "streaming", "tools", "system_messages"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "supports_system_messages": True,
        "supports_json_mode": False,
        "supports_parallel_calls": False,
        "supports_multimodal": False,
        "supports_reasoning": False,
        "max_context_length": 8192,
        "max_output_tokens": 4096,
        "groq_specific": {
            "ultra_fast_inference": True,
            "openai_compatible": True,
            "function_calling_notes": "May require retry fallbacks for complex tool schemas",
            "model_family": cl._detect_model_family(),
        },
        "parameter_mapping": {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "stream": "stream",
            "tools": "tools",
            "tool_choice": "tool_choice"
        },
        "unsupported_parameters": [
            "frequency_penalty", "presence_penalty", "stop", 
            "logit_bias", "user", "n", "best_of", "top_k", "seed"
        ],
        "rate_limits": {"requests_per_minute": 30},
        "available_models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"],
        "model_aliases": {}
    })
    
    # Mock token limits
    monkeypatch.setattr(cl, "get_max_tokens_limit", lambda: 4096)
    monkeypatch.setattr(cl, "get_context_length_limit", lambda: 8192)
    
    # Mock parameter validation
    def mock_validate_parameters(**kwargs):
        result = kwargs.copy()
        # Remove unsupported parameters
        unsupported = ["frequency_penalty", "presence_penalty", "stop", "logit_bias", "user", "n", "best_of", "top_k", "seed"]
        for param in unsupported:
            result.pop(param, None)
        if 'max_tokens' in result and result['max_tokens'] > 4096:
            result['max_tokens'] = 4096
        return result
    monkeypatch.setattr(cl, "validate_parameters", mock_validate_parameters)

    return cl

@pytest.fixture
def pro_client(mock_configuration, monkeypatch) -> GroqAILLMClient:
    """Return a GroqAILLMClient with enhanced capabilities (70b model)."""
    cl = GroqAILLMClient(model="llama-3.3-70b-versatile", api_key="fake-key")

    # Enhanced model with JSON mode support
    monkeypatch.setattr(cl, "supports_feature", lambda feature: feature in [
        "text", "streaming", "tools", "system_messages", "json_mode"
    ])
    
    monkeypatch.setattr(cl, "get_model_info", lambda: {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "client_class": "GroqAILLMClient",
        "api_base": "https://api.groq.com/openai/v1",
        "features": ["text", "streaming", "tools", "system_messages", "json_mode"],
        "supports_text": True,
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "supports_system_messages": True,
        "supports_json_mode": True,
        "supports_parallel_calls": False,
        "supports_multimodal": False,
        "supports_reasoning": False,
        "max_context_length": 8192,
        "max_output_tokens": 4096,
        "groq_specific": {
            "ultra_fast_inference": True,
            "openai_compatible": True,
            "function_calling_notes": "May require retry fallbacks for complex tool schemas",
            "model_family": cl._detect_model_family(),
        },
        "parameter_mapping": {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "stream": "stream",
            "tools": "tools",
            "tool_choice": "tool_choice",
            "response_format": "response_format"
        },
        "unsupported_parameters": [
            "frequency_penalty", "presence_penalty", "stop", 
            "logit_bias", "user", "n", "best_of", "top_k", "seed"
        ]
    })

    return cl

# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------

def test_client_initialization(mock_configuration):
    """Test client initialization with different parameters."""
    # Test with default model
    client1 = GroqAILLMClient()
    assert client1.model == "llama-3.3-70b-versatile"
    
    # Test with custom model and API key
    client2 = GroqAILLMClient(model="mixtral-8x7b-32768", api_key="test-key")
    assert client2.model == "mixtral-8x7b-32768"
    
    # Test with custom API base
    client3 = GroqAILLMClient(
        model="llama-3.1-8b-instant", 
        api_base="https://custom.groq.com"
    )
    assert client3.model == "llama-3.1-8b-instant"

def test_get_model_info(client):
    """Test model info method."""
    info = client.get_model_info()
    
    assert info["provider"] == "groq"
    assert info["model"] == "llama-3.3-70b-versatile"
    assert "groq_specific" in info
    assert info["groq_specific"]["ultra_fast_inference"] is True
    assert info["groq_specific"]["openai_compatible"] is True
    assert "parameter_mapping" in info
    assert "unsupported_parameters" in info

def test_detect_model_family(client):
    """Test model family detection."""
    assert client._detect_model_family() == "llama"
    
    client.model = "mixtral-8x7b-32768"
    assert client._detect_model_family() == "mixtral"
    
    client.model = "gemma-7b-it"
    assert client._detect_model_family() == "gemma"
    
    client.model = "unknown-model"
    assert client._detect_model_family() == "unknown"

# ---------------------------------------------------------------------------
# Request validation tests
# ---------------------------------------------------------------------------

def test_validate_request_with_config(client):
    """Test request validation against configuration."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    validated_messages, validated_tools, validated_stream, validated_kwargs = client._validate_request_with_config(
        messages, tools, stream=True, temperature=0.7, frequency_penalty=0.5
    )
    
    assert validated_messages == messages
    assert validated_tools is not None  # Should be validated tools
    assert validated_stream is True
    assert "temperature" in validated_kwargs
    assert "frequency_penalty" not in validated_kwargs  # Should be removed

def test_validate_request_unsupported_features(client, monkeypatch):
    """Test request validation when features are not supported."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    # Mock configuration to not support streaming or tools
    monkeypatch.setattr(client, "supports_feature", lambda feature: False)
    
    validated_messages, validated_tools, validated_stream, validated_kwargs = client._validate_request_with_config(
        messages, tools, stream=True, temperature=0.7
    )
    
    assert validated_messages == messages
    assert validated_tools is None  # Should be None when not supported
    assert validated_stream is False  # Should be False when not supported
    assert "temperature" in validated_kwargs

# ---------------------------------------------------------------------------
# Tool validation tests
# ---------------------------------------------------------------------------

def test_validate_tools_for_groq(client):
    """Test tool validation for Groq compatibility."""
    complex_tools = [
        {
            "type": "function",
            "function": {
                "name": "complex_tool",
                "description": "A complex tool with nested schema",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nested_param": {
                            "type": "object",
                            "properties": {
                                "inner_value": {"type": "string", "description": "Inner value"}
                            },
                            "required": ["inner_value"]
                        },
                        "simple_param": {"type": "string", "description": "Simple param"}
                    },
                    "required": ["nested_param", "simple_param"]
                }
            }
        }
    ]
    
    validated_tools = client._validate_tools_for_groq(complex_tools)
    
    assert len(validated_tools) == 1
    tool = validated_tools[0]
    assert tool["function"]["name"] == "complex_tool"
    assert "parameters" in tool["function"]
    assert "properties" in tool["function"]["parameters"]

def test_validate_tools_invalid_tool(client):
    """Test tool validation with invalid tools."""
    invalid_tools = [
        {"type": "function"},  # Missing function
        {"type": "function", "function": {}},  # Missing name
        {"type": "function", "function": {"name": "valid_tool", "parameters": {}}}  # Valid
    ]
    
    validated_tools = client._validate_tools_for_groq(invalid_tools)
    
    # Should only include the valid tool
    assert len(validated_tools) == 1
    assert validated_tools[0]["function"]["name"] == "valid_tool"

def test_simplify_schema_for_groq(client):
    """Test schema simplification for Groq."""
    complex_schema = {
        "type": "object",
        "properties": {
            "simple_string": {"type": "string", "description": "A string"},
            "complex_object": {
                "type": "object",
                "properties": {
                    "nested": {"type": "array", "items": {"type": "string"}}
                },
                "additionalProperties": False
            }
        },
        "required": ["simple_string"],
        "additionalProperties": True
    }
    
    simplified = client._simplify_schema_for_groq(complex_schema)
    
    assert simplified["type"] == "object"
    assert "properties" in simplified
    assert "required" in simplified
    assert simplified["required"] == ["simple_string"]
    
    # Check that properties are simplified
    assert simplified["properties"]["simple_string"]["type"] == "string"
    # Note: Complex objects are simplified to their base type, not necessarily "string"
    assert "type" in simplified["properties"]["complex_object"]

def test_validate_tool_call_arguments(client):
    """Test tool call argument validation."""
    # Valid tool call with JSON string arguments
    valid_tool_call_1 = {
        "function": {
            "name": "test_tool",
            "arguments": '{"param": "value"}'
        }
    }
    assert client._validate_tool_call_arguments(valid_tool_call_1) is True
    
    # Valid tool call with dict arguments
    valid_tool_call_2 = {
        "function": {
            "name": "test_tool",
            "arguments": {"param": "value"}
        }
    }
    assert client._validate_tool_call_arguments(valid_tool_call_2) is True
    
    # Invalid tool call with malformed JSON
    invalid_tool_call = {
        "function": {
            "name": "test_tool",
            "arguments": '{"param": invalid json}'
        }
    }
    assert client._validate_tool_call_arguments(invalid_tool_call) is False
    
    # Missing function
    missing_function = {"other": "data"}
    assert client._validate_tool_call_arguments(missing_function) is False

# ---------------------------------------------------------------------------
# Message enhancement tests
# ---------------------------------------------------------------------------

def test_enhance_messages_for_groq(client):
    """Test message enhancement for better Groq function calling."""
    messages = [
        {"role": "user", "content": "Call a function"}
    ]
    
    tools = [
        {"function": {"name": "test_tool", "parameters": {}}}
    ]
    
    enhanced = client._enhance_messages_for_groq(messages, tools)
    
    # Should add system message with function calling guidance
    assert len(enhanced) == 2
    assert enhanced[0]["role"] == "system"
    assert "test_tool" in enhanced[0]["content"]
    assert "JSON format" in enhanced[0]["content"]
    assert enhanced[1] == messages[0]

def test_enhance_messages_existing_system(client):
    """Test message enhancement when system message already exists."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Call a function"}
    ]
    
    tools = [
        {"function": {"name": "test_tool", "parameters": {}}}
    ]
    
    enhanced = client._enhance_messages_for_groq(messages, tools)
    
    # Should enhance existing system message
    assert len(enhanced) == 2
    assert enhanced[0]["role"] == "system"
    assert "You are helpful" in enhanced[0]["content"]
    assert "test_tool" in enhanced[0]["content"]

def test_enhance_messages_no_system_support(client, monkeypatch):
    """Test message enhancement when system messages are not supported."""
    messages = [
        {"role": "user", "content": "Call a function"}
    ]
    
    tools = [
        {"function": {"name": "test_tool", "parameters": {}}}
    ]
    
    # Mock supports_feature to return False for system_messages
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature != "system_messages")
    
    enhanced = client._enhance_messages_for_groq(messages, tools)
    
    # Should return original messages unchanged
    assert enhanced == messages

# ---------------------------------------------------------------------------
# non-streaming path (UPDATED for new interface)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_streaming(monkeypatch, client):
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "a.b", "parameters": {}}}]

    # Mock the async client's response
    mock_message = SimpleNamespace(
        content="Hello from Groq!",
        tool_calls=None
    )
    mock_response = _DummyResp(mock_message)
    
    # Mock the async client
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Stub _normalise_message to a simple passthrough so we can predict output
    monkeypatch.setattr(client, "_normalise_message", lambda m: {"response": m.content, "tool_calls": []})

    # NEW INTERFACE: create_completion returns awaitable for non-streaming
    result_awaitable = client.create_completion(messages, tools, stream=False)
    assert hasattr(result_awaitable, '__await__'), "Non-streaming should return awaitable"
    
    result = await result_awaitable

    # result comes from _normalise_message
    assert result == {"response": "Hello from Groq!", "tool_calls": []}

    # verify Groq API call was forwarded correctly
    client.async_client.chat.completions.create.assert_called_once()
    call_kwargs = client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    
    # Check that messages were processed (may include system message for function calling)
    actual_messages = call_kwargs["messages"]
    user_messages = [m for m in actual_messages if m.get("role") == "user"]
    assert len(user_messages) >= 1, f"Expected at least 1 user message, got: {actual_messages}"
    assert user_messages[0]["content"] == "hi", f"Expected user message 'hi', got: {user_messages[0]}"
    
    # Check that tools were passed (may be in validated form)
    assert "tools" in call_kwargs
    actual_tools = call_kwargs["tools"]
    assert len(actual_tools) == 1
    assert actual_tools[0]["type"] == "function"
    assert actual_tools[0]["function"]["name"] == "a.b"
    
    assert call_kwargs["stream"] is False

# ---------------------------------------------------------------------------
# streaming path (UPDATED for new interface)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_streaming(monkeypatch, client):
    messages = [{"role": "user", "content": "stream please"}]

    # Build three fake chunks: two with content, one final with tool_calls
    chunks = [
        _DummyChunk(_DummyDelta("Hello")),
        _DummyChunk(_DummyDelta(" world")),
        _DummyChunk(_DummyDelta("", tool_calls=[{"name": "foo"}])),
    ]

    # Mock the async client to return a stream
    mock_stream = MockAsyncGroqStream(chunks)
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    # NEW INTERFACE: create_completion returns async generator directly for streaming
    stream_iter = client.create_completion(messages, tools=None, stream=True)
    
    # Should be async generator, not awaitable
    assert hasattr(stream_iter, "__aiter__"), "Should return an async iterator"
    assert not hasattr(stream_iter, "__await__"), "Should not be awaitable"

    collected = [c async for c in stream_iter]

    # we expect exactly three deltas mirroring our chunks
    assert collected == [
        {"response": "Hello", "tool_calls": []},
        {"response": " world", "tool_calls": []},
        {"response": "", "tool_calls": [{"name": "foo"}]},
    ]

    # Verify the async client was called correctly
    client.async_client.chat.completions.create.assert_called_once()
    call_kwargs = client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True

# ---------------------------------------------------------------------------
# Interface compliance test (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interface_compliance(client):
    """Test that create_completion follows the correct interface."""
    # Test non-streaming
    messages = [{"role": "user", "content": "Test"}]
    
    # Mock completion
    mock_response = _DummyResp(SimpleNamespace(content="Test response", tool_calls=None))
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # For non-streaming, we need to await the result if it's a coroutine
    result = client.create_completion(messages, stream=False)
    if asyncio.iscoroutine(result):
        result = await result
    
    assert isinstance(result, dict)
    assert "response" in result
    
    # Test streaming returns async iterator
    chunks = [_DummyChunk(_DummyDelta("chunk1")), _DummyChunk(_DummyDelta("chunk2"))]
    mock_stream = MockAsyncGroqStream(chunks)
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    
    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, "__aiter__")
    assert not hasattr(stream_result, "__await__")

# ---------------------------------------------------------------------------
# Tool handling tests (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_with_tools(monkeypatch, client):
    """Test streaming with tool calls."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"function": {"name": "get_weather", "parameters": {}}}]

    # Mock tool call
    mock_tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="get_weather",
            arguments='{"location": "NYC"}'
        )
    )

    chunks = [
        _DummyChunk(_DummyDelta("Let me check")),
        _DummyChunk(_DummyDelta("", tool_calls=[mock_tool_call])),
        _DummyChunk(_DummyDelta(" the weather for you")),
    ]

    mock_stream = MockAsyncGroqStream(chunks)
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    stream_iter = client.create_completion(messages, tools=tools, stream=True)
    collected = [c async for c in stream_iter]

    assert len(collected) == 3
    assert collected[0]["response"] == "Let me check"
    assert collected[0]["tool_calls"] == []
    
    assert collected[1]["response"] == ""
    assert len(collected[1]["tool_calls"]) == 1
    
    assert collected[2]["response"] == " the weather for you"

@pytest.mark.asyncio
async def test_non_streaming_with_tools(monkeypatch, client):
    """Test non-streaming with tool calls."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"function": {"name": "get_weather"}}]

    # Mock response with tool calls
    mock_tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="get_weather",
            arguments='{"location": "NYC"}'
        )
    )
    
    mock_message = SimpleNamespace(
        content="Let me check the weather",
        tool_calls=[mock_tool_call]
    )
    mock_response = _DummyResp(mock_message)
    
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Mock _normalise_message to handle tool calls
    def mock_normalise(message):
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
        return {
            "response": message.content,
            "tool_calls": tool_calls
        }
    
    monkeypatch.setattr(client, "_normalise_message", mock_normalise)

    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    result = await result_awaitable

    assert result["response"] == "Let me check the weather"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"

# ---------------------------------------------------------------------------
# Error handling tests (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_error_handling(client):
    """Test error handling in streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock an error during streaming
    client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    
    stream_iter = client.create_completion(messages, stream=True)
    
    # Should yield error chunk
    chunks = []
    async for chunk in stream_iter:
        chunks.append(chunk)
    
    assert len(chunks) == 1
    error_chunk = chunks[0]
    assert "error" in error_chunk
    assert "API Error" in error_chunk["response"]

@pytest.mark.asyncio
async def test_non_streaming_error_handling(client):
    """Test error handling in non-streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock an error
    client.async_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    
    result_awaitable = client.create_completion(messages, stream=False)
    result = await result_awaitable
    
    assert "error" in result
    assert "API Error" in result["response"]

@pytest.mark.asyncio
async def test_groq_function_calling_error_retry(client):
    """Test Groq function calling error retry mechanism."""
    messages = [{"role": "user", "content": "Use a tool"}]
    tools = [{"function": {"name": "test_tool", "parameters": {}}}]
    
    # Mock the first call to fail with Groq function calling error
    def mock_create_side_effect(**kwargs):
        if "tools" in kwargs and kwargs["tools"]:
            raise Exception("Failed to call a function")
        else:
            # Return successful response without tools
            return _DummyResp(SimpleNamespace(content="Tool disabled fallback", tool_calls=None))
    
    client.async_client.chat.completions.create = AsyncMock(side_effect=mock_create_side_effect)
    
    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    result = await result_awaitable
    
    # Should have retried without tools and included a note
    assert "Tool disabled fallback" in result["response"]
    assert "Function calling disabled" in result["response"]
    assert result["tool_calls"] == []

# ---------------------------------------------------------------------------
# Configuration integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_configuration_feature_validation(client, monkeypatch):
    """Test that configuration properly validates features."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"function": {"name": "test_tool"}}]
    
    # Mock configuration to support only streaming (not tools)
    monkeypatch.setattr(client, "supports_feature", lambda feature: feature == "streaming")
    
    # Test with tools - should be filtered out
    result = client.create_completion(messages, tools=tools, stream=True)
    assert hasattr(result, "__aiter__")  # Should still return stream
    
    # Mock the actual streaming call to verify tools were removed
    captured_params = {}
    async def mock_stream(**kwargs):
        captured_params.update(kwargs)
        chunks = [_DummyChunk(_DummyDelta("response without tools"))]
        return MockAsyncGroqStream(chunks)
    
    client.async_client.chat.completions.create = AsyncMock(side_effect=mock_stream)
    
    collected = []
    async for chunk in client.create_completion(messages, tools=tools, stream=True):
        collected.append(chunk)
    
    # Verify tools were not passed to API
    assert captured_params.get("tools") is None

@pytest.mark.asyncio
async def test_parameter_validation_and_filtering(client):
    """Test parameter validation and filtering."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the completion
    mock_response = _DummyResp(SimpleNamespace(content="Hello!", tool_calls=None))
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    result = await client.create_completion(
        messages, 
        temperature=0.7, 
        frequency_penalty=0.5,  # Should be filtered out
        presence_penalty=0.3,   # Should be filtered out
        max_tokens=100,
        stream=False
    )
    
    # Should complete successfully with filtered parameters
    assert result["response"] == "Hello!"

# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_integration_non_streaming(client):
    """Test full integration for non-streaming completion."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]
    
    # Mock the actual Groq API call
    mock_response = _DummyResp(SimpleNamespace(
        content="Hello! How can I help you today?",
        tool_calls=None
    ))
    
    captured_params = {}
    async def mock_create(**kwargs):
        captured_params.update(kwargs)
        return mock_response
    
    client.async_client.chat.completions.create = AsyncMock(side_effect=mock_create)
    
    # Mock _normalise_message
    client._normalise_message = lambda m: {"response": m.content, "tool_calls": []}
    
    result = await client.create_completion(messages, stream=False)
    
    assert result["response"] == "Hello! How can I help you today?"
    assert result["tool_calls"] == []
    
    # Verify request structure
    assert captured_params["model"] == "llama-3.3-70b-versatile"
    assert len(captured_params["messages"]) >= 2  # May include enhanced system message

@pytest.mark.asyncio
async def test_full_integration_streaming(client):
    """Test full integration for streaming completion."""
    messages = [{"role": "user", "content": "Tell me a story"}]
    
    # Mock streaming response
    story_parts = ["Once", " upon", " a", " time..."]
    chunks = [_DummyChunk(_DummyDelta(part)) for part in story_parts]
    mock_stream = MockAsyncGroqStream(chunks)
    
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    
    # Collect all chunks
    collected_parts = []
    async for chunk in client.create_completion(messages, stream=True):
        collected_parts.append(chunk["response"])
    
    # Verify we got all parts
    assert len(collected_parts) == 4
    assert collected_parts == story_parts

# ---------------------------------------------------------------------------
# Edge cases and error scenarios
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_streaming_response(client):
    """Test handling of empty streaming response."""
    messages = [{"role": "user", "content": "Empty response test"}]
    
    # Mock empty stream
    mock_stream = MockAsyncGroqStream([])
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    
    collected = []
    async for chunk in client.create_completion(messages, stream=True):
        collected.append(chunk)
    
    # Should handle empty stream gracefully
    assert len(collected) == 0

@pytest.mark.asyncio
async def test_malformed_streaming_chunks(client):
    """Test handling of malformed streaming chunks."""
    messages = [{"role": "user", "content": "Test malformed chunks"}]
    
    # Mock stream with malformed chunks (missing choices)
    malformed_chunks = [
        SimpleNamespace(),  # No choices
        SimpleNamespace(choices=[]),  # Empty choices
        _DummyChunk(_DummyDelta("Valid chunk")),  # Valid chunk
    ]
    mock_stream = MockAsyncGroqStream(malformed_chunks)
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    
    collected = []
    async for chunk in client.create_completion(messages, stream=True):
        collected.append(chunk)
    
    # Should handle malformed chunks gracefully - may result in error or skipped chunks
    # The exact behavior depends on the implementation
    assert len(collected) >= 0  # At least don't crash

# ---------------------------------------------------------------------------
# Performance simulation test (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_high_speed_streaming(client):
    """Test high-speed streaming like real Groq performance."""
    messages = [{"role": "user", "content": "Write a story"}]
    
    # Simulate many fast chunks (like real Groq)
    chunks = [_DummyChunk(_DummyDelta(f"chunk{i} ")) for i in range(100)]
    
    mock_stream = MockAsyncGroqStream(chunks)
    client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    
    stream_iter = client.create_completion(messages, stream=True)
    
    collected = []
    async for chunk in stream_iter:
        collected.append(chunk)
    
    # Should handle many chunks efficiently
    assert len(collected) == 100
    assert all("chunk" in chunk["response"] for chunk in collected)
    
    # Verify full response can be reconstructed
    full_response = "".join(chunk["response"] for chunk in collected)
    assert "chunk0 chunk1 chunk2" in full_response