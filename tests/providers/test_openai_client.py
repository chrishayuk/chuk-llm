"""
Tests for OpenAI client implementation.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from chuk_llm.llm.providers.openai_client import OpenAILLMClient


@pytest.fixture
def client():
    """Create a mock OpenAI client for testing."""
    with patch("chuk_llm.llm.providers.openai_client.openai"):
        return OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")


@pytest.mark.asyncio
async def test_create_completion_non_streaming(monkeypatch, client):
    """Test non-streaming completion."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock the _regular_completion method
    async def fake_regular_completion(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == [] or tls is None
        return {"response": "Hello there!", "tool_calls": []}
    
    monkeypatch.setattr(client, "_regular_completion", fake_regular_completion)
    
    # Test non-streaming
    result_awaitable = client.create_completion(messages, tools=None, stream=False)
    assert asyncio.iscoroutine(result_awaitable)
    
    result = await result_awaitable
    assert result["response"] == "Hello there!"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    """Test streaming completion."""
    messages = [{"role": "user", "content": "Hello again"}]

    # Mock the _stream_completion_async method
    async def fake_stream_completion_async(msgs, tls=None, **kwargs):
        assert msgs == messages
        assert tls == [] or tls is None  # Allow both [] and None
        yield {"response": "Hello", "tool_calls": []}
        yield {"response": " World", "tool_calls": []}

    monkeypatch.setattr(client, "_stream_completion_async", fake_stream_completion_async)

    # Use new interface - get async generator directly
    async_iter = client.create_completion(messages, tools=None, stream=True)
    assert hasattr(async_iter, "__aiter__")

    pieces = [chunk async for chunk in async_iter]
    assert len(pieces) == 2
    assert pieces[0]["response"] == "Hello"
    assert pieces[1]["response"] == " World"


@pytest.mark.asyncio
async def test_create_completion_with_tools(monkeypatch, client):
    """Test completion with tool calls."""
    messages = [{"role": "user", "content": "Use tool"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    # Mock the _regular_completion method
    async def fake_regular_completion(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == tools
        return {
            "response": None, 
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"}
            }]
        }
    
    monkeypatch.setattr(client, "_regular_completion", fake_regular_completion)
    
    result = await client.create_completion(messages, tools=tools, stream=False)
    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_create_completion_streaming_with_tools(monkeypatch, client):
    """Test streaming with tool calls."""
    messages = [{"role": "user", "content": "Use tool"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    # Mock streaming with tool calls
    async def fake_stream_with_tools(msgs, tls, **kwargs):
        yield {
            "response": "", 
            "tool_calls": [{
                "id": "call_1",
                "type": "function", 
                "function": {"name": "test_tool", "arguments": "{}"}
            }]
        }
        yield {"response": "Tool executed", "tool_calls": []}
    
    monkeypatch.setattr(client, "_stream_completion_async", fake_stream_with_tools)
    
    stream = client.create_completion(messages, tools=tools, stream=True)
    chunks = [chunk async for chunk in stream]
    
    assert len(chunks) == 2
    assert len(chunks[0]["tool_calls"]) == 1
    assert chunks[1]["response"] == "Tool executed"


@pytest.mark.asyncio
async def test_create_completion_error_handling(monkeypatch, client):
    """Test error handling in non-streaming mode."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock error in regular completion
    async def fake_error_completion(msgs, tls, **kwargs):
        return {"response": "Error: API Error", "tool_calls": [], "error": True}
    
    monkeypatch.setattr(client, "_regular_completion", fake_error_completion)
    
    result = await client.create_completion(messages, stream=False)
    assert result["error"] is True
    assert "API Error" in result["response"]


@pytest.mark.asyncio
async def test_create_completion_streaming_error_handling(monkeypatch, client):
    """Test error handling in streaming mode."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Mock streaming with error
    async def fake_error_stream(msgs, tls, **kwargs):
        yield {"response": "Start", "tool_calls": []}
        yield {"response": "Streaming error: API Error", "tool_calls": [], "error": True}
    
    monkeypatch.setattr(client, "_stream_completion_async", fake_error_stream)
    
    stream = client.create_completion(messages, stream=True)
    chunks = [chunk async for chunk in stream]
    
    assert len(chunks) == 2
    assert chunks[0]["response"] == "Start"
    assert chunks[1]["error"] is True


@pytest.mark.asyncio
async def test_interface_compliance(client):
    """Test that client follows the BaseLLMClient interface."""
    # Test non-streaming
    messages = [{"role": "user", "content": "Test"}]
    
    # For non-streaming, we need to await the result
    result = client.create_completion(messages, stream=False)
    if asyncio.iscoroutine(result):
        result = await result
    
    assert isinstance(result, dict)
    assert "response" in result


def test_client_instantiation():
    """Test that OpenAI client can be instantiated."""
    with patch("chuk_llm.llm.providers.openai_client.openai"):
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        assert client.model == "gpt-4o-mini"


def test_sanitize_tool_names():
    """Test tool name sanitization."""
    with patch("chuk_llm.llm.providers.openai_client.openai"):
        client = OpenAILLMClient()
        
        tools = [{"function": {"name": "invalid@name"}}]
        sanitized = client._sanitize_tool_names(tools)
        assert sanitized[0]["function"]["name"] == "invalid_name"