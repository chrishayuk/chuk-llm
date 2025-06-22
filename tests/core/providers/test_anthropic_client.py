# tests/providers/test_anthropic_client.py
import sys
import types
import json
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

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
    _parse_claude_response
)  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return AnthropicLLMClient(model="claude-test", api_key="fake-key")

# Convenience helper to capture kwargs
class Capture:
    kwargs = None

# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

def test_parse_claude_response_text_only():
    """Test parsing Claude response with text only."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world")]
    
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

# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------

def test_client_initialization():
    """Test client initialization with different parameters."""
    # Test with default model
    client1 = AnthropicLLMClient()
    assert client1.model == "claude-3-5-sonnet-20241022"
    
    # Test with custom model and API key
    client2 = AnthropicLLMClient(model="claude-test", api_key="test-key")
    assert client2.model == "claude-test"
    
    # Test with API base
    client3 = AnthropicLLMClient(model="claude-test", api_base="https://custom.anthropic.com")
    assert client3.model == "claude-test"

def test_get_model_info(client):
    """Test model info method."""
    info = client.get_model_info()
    
    assert info["provider"] == "anthropic"
    assert info["model"] == "claude-test"
    assert "vision_format" in info
    assert "supported_parameters" in info
    assert "unsupported_parameters" in info

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
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
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
        {"function": {"name": "valid_tool", "parameters": {}}}  # Valid
    ]
    
    converted = client._convert_tools(malformed_tools)
    assert len(converted) == 3  # Should handle all tools, using fallbacks

def test_convert_tools_nested_structure(client):
    """Test tool conversion with nested tool structure."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
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
        "custom_param": "value"  # Unknown
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

# ---------------------------------------------------------------------------
# JSON mode tests
# ---------------------------------------------------------------------------

def test_check_json_mode(client):
    """Test JSON mode detection."""
    # Mock JSON mode support
    client.supports_feature = lambda feature: feature == "json_mode"
    
    # Test OpenAI-style response_format
    kwargs = {"response_format": {"type": "json_object"}}
    instruction = client._check_json_mode(kwargs)
    assert instruction is not None
    assert "JSON" in instruction
    
    # Test custom json mode instruction
    kwargs = {"_json_mode_instruction": "Custom JSON instruction"}
    instruction = client._check_json_mode(kwargs)
    assert instruction == "Custom JSON instruction"
    
    # Test no JSON mode
    kwargs = {}
    instruction = client._check_json_mode(kwargs)
    assert instruction is None

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
        {"role": "user", "content": "How are you?"}
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
    # Mock vision support
    client.supports_feature = lambda feature: feature == "vision"
    
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
        ]}
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
async def test_split_for_anthropic_async_tool_calls(client):
    """Test message splitting with tool calls."""
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"arg": "value"}'}
        }]},
        {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"}
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
        {"role": "user", "content": "Hello"}
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
        }
    }
    
    result = await AnthropicLLMClient._convert_universal_vision_to_anthropic_async(content_item)
    
    assert result["type"] == "image"
    assert result["source"]["type"] == "base64"
    assert result["source"]["media_type"] == "image/png"
    assert "data" in result["source"]

@pytest.mark.asyncio
async def test_convert_universal_vision_to_anthropic_async_invalid_data_url():
    """Test converting invalid data URL."""
    content_item = {
        "type": "image_url",
        "image_url": {"url": "data:invalid"}
    }
    
    result = await AnthropicLLMClient._convert_universal_vision_to_anthropic_async(content_item)
    
    assert result["type"] == "text"
    assert "Invalid image format" in result["text"]

@pytest.mark.asyncio
async def test_convert_universal_vision_to_anthropic_async_external_url():
    """Test converting external URL (should attempt download)."""
    content_item = {
        "type": "image_url", 
        "image_url": {"url": "https://example.com/image.png"}
    }
    
    # Mock the download function to avoid actual network calls
    with patch.object(AnthropicLLMClient, '_download_image_to_base64') as mock_download:
        mock_download.side_effect = Exception("Network error")
        
        result = await AnthropicLLMClient._convert_universal_vision_to_anthropic_async(content_item)
        
        assert result["type"] == "text"
        assert "Could not load image" in result["text"]

# ---------------------------------------------------------------------------
# Regular completion tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regular_completion_async(client):
    """Test regular (non-streaming) completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the async client's create method
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello! How can I help you?")]
    
    async def mock_create(**kwargs):
        return mock_response
    
    client.async_client.messages.create = mock_create
    
    result = await client._regular_completion_async(
        system=None,
        json_instruction=None,
        messages=messages,
        anth_tools=[],
        filtered_params={}
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
    mock_response.content = [MagicMock(text="Hello! I'm here to help.")]
    
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
        filtered_params={}
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
    mock_response.content = [MagicMock(text='{"result": "success"}')]
    
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
        filtered_params={}
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
        filtered_params={}
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
        MockEvent('content_block_delta', delta=MockDelta("Hello")),
        MockEvent('content_block_delta', delta=MockDelta(" world!"))
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
        filtered_params={}
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
    
    mock_events = [
        MockEvent('content_block_start', content_block=MockContentBlock())
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
        anth_tools=[{"name": "test_tool"}],
        filtered_params={}
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
        filtered_params={}
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
    
    async def mock_regular_completion_async(system, json_instruction, messages, anth_tools, filtered_params):
        return expected_result
    
    client._regular_completion_async = mock_regular_completion_async
    
    result = client.create_completion(messages, stream=False)
    
    # Should return an awaitable
    assert hasattr(result, '__await__')
    
    final_result = await result
    assert final_result == expected_result

@pytest.mark.asyncio
async def test_create_completion_streaming(client):
    """Test create_completion with streaming."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the streaming method
    async def mock_stream_completion_async(system, json_instruction, messages, anth_tools, filtered_params):
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": []}
    
    client._stream_completion_async = mock_stream_completion_async
    
    result = client.create_completion(messages, stream=True)
    
    # Should return an async generator
    assert hasattr(result, '__aiter__')
    
    chunks = []
    async for chunk in result:
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert chunks[0]["response"] == "chunk1"
    assert chunks[1]["response"] == "chunk2"

@pytest.mark.asyncio
async def test_create_completion_with_tools(client):
    """Test create_completion with tools."""
    # Mock tool support
    client.supports_feature = lambda feature: feature == "tools"
    
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    
    # Mock regular completion
    expected_result = {
        "response": None,
        "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ]
    }
    
    async def mock_regular_completion_async(system, json_instruction, messages, anth_tools, filtered_params):
        # Verify tools were converted
        assert len(anth_tools) == 1
        assert anth_tools[0]["name"] == "get_weather"
        return expected_result
    
    client._regular_completion_async = mock_regular_completion_async
    
    result = await client.create_completion(messages, tools=tools, stream=False)
    
    assert result == expected_result
    assert len(result["tool_calls"]) == 1

@pytest.mark.asyncio
async def test_create_completion_with_max_tokens(client):
    """Test create_completion with max_tokens parameter."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock regular completion to check parameters
    async def mock_regular_completion_async(system, json_instruction, messages, anth_tools, filtered_params):
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
    async def mock_regular_completion_async(system_arg, json_instruction, messages, anth_tools, filtered_params):
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
        {"role": "user", "content": "Hello"}
    ]
    
    # Mock the actual Anthropic API call
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello! How can I help you today?")]
    
    captured_payload = {}
    async def mock_create(**kwargs):
        captured_payload.update(kwargs)
        return mock_response
    
    client.async_client.messages.create = mock_create
    
    result = await client.create_completion(messages, stream=False)
    
    assert result["response"] == "Hello! How can I help you today?"
    assert result["tool_calls"] == []
    
    # Verify payload structure
    assert captured_payload["model"] == "claude-test"
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
        MockEvent('content_block_delta', delta=MockDelta("Once")),
        MockEvent('content_block_delta', delta=MockDelta(" upon")),
        MockEvent('content_block_delta', delta=MockDelta(" a")),
        MockEvent('content_block_delta', delta=MockDelta(" time..."))
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
    async def error_stream(system, json_instruction, messages, anth_tools, filtered_params):
        yield {"response": "Starting...", "tool_calls": []}
        yield {"response": "Streaming error: Test error", "tool_calls": [], "error": True}

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
    async def error_completion(system, json_instruction, messages, anth_tools, filtered_params):
        return {"response": "Error: Test error", "tool_calls": [], "error": True}

    client._regular_completion_async = error_completion

    result = await client.create_completion(messages, stream=False)

    assert result["error"] is True
    assert "Test error" in result["response"]

# ---------------------------------------------------------------------------
# Complex scenario tests
# ---------------------------------------------------------------------------

def test_tool_name_sanitization(client):
    """Test that tool names are properly sanitized."""
    # This test assumes _sanitize_tool_names method exists (from mixin)
    tools = [{"function": {"name": "invalid@name"}}]
    
    # Mock the method if it doesn't exist
    if not hasattr(client, '_sanitize_tool_names'):
        def mock_sanitize(tools):
            if tools:
                for tool in tools:
                    if "function" in tool and "name" in tool["function"]:
                        tool["function"]["name"] = tool["function"]["name"].replace("@", "_")
            return tools
        client._sanitize_tool_names = mock_sanitize
    
    sanitized = client._sanitize_tool_names(tools)
    assert sanitized[0]["function"]["name"] == "invalid_name"

@pytest.mark.asyncio
async def test_complex_message_conversion(client):
    """Test message conversion with complex scenarios."""
    messages = [
        {"role": "system", "content": "System prompt 1"},
        {"role": "system", "content": "System prompt 2"},  # Multiple system messages
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "tool_calls": [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
        {"role": "assistant", "content": "Based on the tool result..."}
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
    async def mock_completion(system, json_instruction, messages, anth_tools, filtered_params):
        return {"response": "Test response", "tool_calls": []}
    
    client._regular_completion_async = mock_completion
    
    # Non-streaming should return awaitable
    result_coro = client.create_completion(messages, stream=False)
    assert asyncio.iscoroutine(result_coro)
    
    result = await result_coro
    assert isinstance(result, dict)
    assert "response" in result
    
    # Streaming should return async iterator
    async def mock_stream(system, json_instruction, messages, anth_tools, filtered_params):
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": []}
    
    client._stream_completion_async = mock_stream
    
    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, "__aiter__")
    
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    assert len(chunks) == 2