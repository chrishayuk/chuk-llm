# tests/providers/test_anthropic_client.py
import sys
import types
import json
import pytest
import asyncio

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

# Fake Messages client with stream support
class _DummyMessages:
    def create(self, *args, **kwargs):
        return None  # will be monkey-patched per-test
    
    def stream(self, *args, **kwargs):
        return None  # will be monkey-patched per-test

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

from chuk_llm.llm.providers.anthropic_client import AnthropicLLMClient  # noqa: E402  pylint: disable=wrong-import-position


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
# Non‑streaming test (UPDATED for clean interface)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_stream(monkeypatch, client):
    # Simple chat sequence
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi Claude"},
    ]
    tools = [
        {"type": "function", "function": {"name": "foo", "parameters": {}}}
    ]

    # Sanitise no‑op so we can assert
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    # Patch _regular_completion to validate payload and return dummy response
    async def fake_regular_completion(payload):  # noqa: D401
        Capture.kwargs = payload
        # Simulate Claude text response
        return {"response": "Hello there!", "tool_calls": []}

    monkeypatch.setattr(client, "_regular_completion", fake_regular_completion)

    # Clean interface: create_completion returns awaitable when not streaming
    result = await client.create_completion(messages, tools=tools, stream=False)
    
    assert result == {"response": "Hello there!", "tool_calls": []}

    # Validate key bits of the payload sent to Claude
    assert Capture.kwargs["model"] == "claude-test"
    assert Capture.kwargs["system"] == "You are helpful."
    # tools converted gets placed into payload["tools"] – check basic structure
    conv_tools = Capture.kwargs["tools"]
    assert isinstance(conv_tools, list) and conv_tools[0]["name"] == "foo"

# ---------------------------------------------------------------------------
# Streaming test (UPDATED for clean streaming interface)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Stream please"}]

    # Mock the async streaming context manager
    class MockStreamManager:
        def __init__(self, events):
            self.events = events
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def __aiter__(self):
            for event in self.events:
                yield event

    # Mock stream events (simulating Anthropic's real streaming events)
    mock_events = [
        # Simulate content_block_delta events
        types.SimpleNamespace(
            type='content_block_delta',
            delta=types.SimpleNamespace(text='Hello')
        ),
        types.SimpleNamespace(
            type='content_block_delta', 
            delta=types.SimpleNamespace(text=' there!')
        ),
        # Simulate tool use event
        types.SimpleNamespace(
            type='content_block_start',
            content_block=types.SimpleNamespace(
                type='tool_use',
                id='tool_123',
                name='test_tool',
                input={'arg': 'value'}
            )
        )
    ]

    # Patch the async client's messages.stream method
    def fake_stream_method(**payload):
        Capture.kwargs = payload
        return MockStreamManager(mock_events)

    monkeypatch.setattr(client.async_client.messages, "stream", fake_stream_method)
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    # Clean interface: create_completion returns async generator directly
    iterator = client.create_completion(messages, tools=None, stream=True)
    
    # Should be an async generator
    assert hasattr(iterator, "__aiter__")
    
    # Collect all chunks
    received = []
    async for chunk in iterator:
        received.append(chunk)
    
    # Should have received text chunks and tool calls
    assert len(received) >= 2
    
    # First two should be text chunks
    assert received[0]["response"] == "Hello"
    assert received[0]["tool_calls"] == []
    
    assert received[1]["response"] == " there!"
    assert received[1]["tool_calls"] == []
    
    # Last should be tool call (if present)
    if len(received) > 2:
        assert received[2]["response"] == ""
        assert len(received[2]["tool_calls"]) == 1
        assert received[2]["tool_calls"][0]["function"]["name"] == "test_tool"

    # Validate payload was passed correctly to streaming
    assert Capture.kwargs["model"] == "claude-test"
    assert Capture.kwargs["messages"][0]["role"] == "user"
    assert Capture.kwargs["messages"][0]["content"] == [{"type": "text", "text": "Stream please"}]

# ---------------------------------------------------------------------------
# Test interface compliance (UPDATED for clean interface)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_interface_compliance(client):
    """Test that create_completion follows the correct interface."""
    # Test non-streaming - should return awaitable
    messages = [{"role": "user", "content": "Test"}]
    
    # Mock the completion
    async def mock_completion(payload):
        return {"response": "Test response", "tool_calls": []}
    
    import types
    client._regular_completion = mock_completion
    
    # Non-streaming should return awaitable
    result_coro = client.create_completion(messages, stream=False)
    assert asyncio.iscoroutine(result_coro)
    
    result = await result_coro
    assert isinstance(result, dict)
    assert "response" in result
    
    # Streaming should return async iterator
    async def mock_stream(payload):
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
# Test tool conversion
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

# ---------------------------------------------------------------------------
# Test message splitting
# ---------------------------------------------------------------------------

def test_split_for_anthropic(client):
    """Test message splitting into system + conversation."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ]
    
    system_txt, anthropic_messages = client._split_for_anthropic(messages)
    
    assert system_txt == "You are helpful"
    assert len(anthropic_messages) == 3  # Excluding system message
    assert anthropic_messages[0]["role"] == "user"
    assert anthropic_messages[1]["role"] == "assistant" 
    assert anthropic_messages[2]["role"] == "user"

def test_split_for_anthropic_multimodal(client):
    """Test message splitting with multimodal content."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
    
    system_txt, anthropic_messages = client._split_for_anthropic(messages)
    
    assert system_txt == ""
    assert len(anthropic_messages) == 1
    assert anthropic_messages[0]["role"] == "user"
    assert isinstance(anthropic_messages[0]["content"], list)

def test_split_for_anthropic_tool_calls(client):
    """Test message splitting with tool calls."""
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"}
    ]
    
    system_txt, anthropic_messages = client._split_for_anthropic(messages)
    
    assert len(anthropic_messages) == 2
    assert anthropic_messages[0]["role"] == "assistant"
    assert anthropic_messages[0]["content"][0]["type"] == "tool_use"
    assert anthropic_messages[1]["role"] == "user"  # Tool response becomes user message

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
    assert filtered["max_tokens"] == 1024

# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_error_handling(monkeypatch, client):
    """Test error handling in streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock streaming with error
    async def error_stream(payload):
        yield {"response": "Starting...", "tool_calls": []}
        yield {"response": "Streaming error: Test error", "tool_calls": [], "error": True}

    monkeypatch.setattr(client, "_stream_completion_async", error_stream)

    stream_result = client.create_completion(messages, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["response"] == "Starting..."
    assert chunks[1]["error"] is True

@pytest.mark.asyncio
async def test_non_streaming_error_handling(monkeypatch, client):
    """Test error handling in non-streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock error in regular completion
    async def error_completion(payload):
        return {"response": "Error: Test error", "tool_calls": [], "error": True}

    monkeypatch.setattr(client, "_regular_completion", error_completion)

    result = await client.create_completion(messages, stream=False)

    assert result["error"] is True
    assert "Test error" in result["response"]

# ---------------------------------------------------------------------------
# Integration and initialization tests
# ---------------------------------------------------------------------------

def test_client_initialization():
    """Test client initialization with different parameters."""
    # Test with default model
    client1 = AnthropicLLMClient()
    assert client1.model == "claude-3-7-sonnet-20250219"
    
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
    assert info["supports_streaming"] is True
    assert info["supports_tools"] is True
    assert info["supports_vision"] is True
    assert "supported_parameters" in info
    assert "unsupported_parameters" in info

@pytest.mark.asyncio
async def test_with_tool_calls(monkeypatch, client):
    """Test completion with tool calls."""
    messages = [{"role": "user", "content": "Call a tool"}]
    tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]

    # Mock tool call response
    async def fake_tool_completion(payload):
        return {
            "response": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"}
            }]
        }

    monkeypatch.setattr(client, "_regular_completion", fake_tool_completion)
    monkeypatch.setattr(client, "_sanitize_tool_names", lambda t: t)

    result = await client.create_completion(messages, tools=tools, stream=False)

    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"

def test_tool_name_sanitization(client):
    """Test that tool names are properly sanitized."""
    tools = [{"function": {"name": "invalid@name"}}]
    sanitized = client._sanitize_tool_names(tools)
    assert sanitized[0]["function"]["name"] == "invalid_name"

def test_complex_message_conversion(client):
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
    
    system_txt, anthropic_messages = client._split_for_anthropic(messages)
    
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