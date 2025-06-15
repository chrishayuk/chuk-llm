# tests/providers/test_groq_client.py
import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from chuk_llm.llm.providers.groq_client import GroqAILLMClient


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
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch) -> GroqAILLMClient:
    """Return a GroqAILLMClient with all helpers stubbed out."""
    cl = GroqAILLMClient(model="dummy", api_key="fake-key")

    # --- stub the sanitiser to identity ---
    monkeypatch.setattr(cl, "_sanitize_tool_names", lambda t: t)

    return cl


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
    assert call_kwargs["model"] == "dummy"
    
    # FIXED: Handle that Groq may inject system messages for tool guidance
    # Check that our original user message is present in the messages
    actual_messages = call_kwargs["messages"]
    user_messages = [m for m in actual_messages if m.get("role") == "user"]
    assert len(user_messages) >= 1, f"Expected at least 1 user message, got: {actual_messages}"
    assert user_messages[0]["content"] == "hi", f"Expected user message 'hi', got: {user_messages[0]}"
    
    assert call_kwargs["tools"] == tools
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
    
    # For non-streaming, we need to await the result if it's a coroutine
    result = client.create_completion(messages, stream=False)
    if asyncio.iscoroutine(result):
        result = await result
    
    assert isinstance(result, dict)
    assert "response" in result
    
    # Test streaming returns async iterator
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