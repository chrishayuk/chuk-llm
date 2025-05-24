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
# Non‑streaming test (UPDATED for new interface)
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

    # NEW INTERFACE: create_completion is no longer async when not streaming
    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    
    # The result should be awaitable
    assert hasattr(result_awaitable, '__await__')
    result = await result_awaitable
    
    assert result == {"response": "Hello there!", "tool_calls": []}

    # Validate key bits of the payload sent to Claude
    assert Capture.kwargs["model"] == "claude-test"
    assert Capture.kwargs["system"] == "You are helpful."
    # tools converted gets placed into payload["tools"] – check basic structure
    conv_tools = Capture.kwargs["tools"]
    assert isinstance(conv_tools, list) and conv_tools[0]["name"] == "foo"

# ---------------------------------------------------------------------------
# Streaming test (UPDATED for new real streaming interface)
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

    # NEW INTERFACE: create_completion returns async generator directly
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
# Test interface compliance (NEW) - FIXED to be async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_interface_compliance(client):
    """Test that create_completion follows the correct interface."""
    # Test non-streaming
    messages = [{"role": "user", "content": "Test"}]
    
    # For non-streaming, we need to await the result
    result = client.create_completion(messages, stream=False)
    if asyncio.iscoroutine(result):
        result = await result
    
    assert isinstance(result, dict)
    assert "response" in result

# ---------------------------------------------------------------------------
# Test tool conversion (UPDATED)
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

# ---------------------------------------------------------------------------
# Test message splitting (UPDATED)
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

# ---------------------------------------------------------------------------
# Additional error handling and edge case tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_error_handling(monkeypatch, client):
    """Test error handling in streaming mode."""
    messages = [{"role": "user", "content": "test"}]

    # Mock streaming with error
    async def error_stream(payload):
        yield {"response": "Starting...", "tool_calls": []}
        # Yield error chunk instead of raising
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

    result_awaitable = client.create_completion(messages, stream=False)
    result = await result_awaitable

    assert result["error"] is True
    assert "Test error" in result["response"]

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

    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    result = await result_awaitable

    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"

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

def test_tool_name_sanitization(client):
    """Test that tool names are properly sanitized."""
    tools = [{"function": {"name": "invalid@name"}}]
    sanitized = client._sanitize_tool_names(tools)
    assert sanitized[0]["function"]["name"] == "invalid_name"

def test_empty_tools_conversion(client):
    """Test conversion with empty or None tools."""
    # Test with None
    result = client._convert_tools(None)
    assert result == []
    
    # Test with empty list
    result = client._convert_tools([])
    assert result == []

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