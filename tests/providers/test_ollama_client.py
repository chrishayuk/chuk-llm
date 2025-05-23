# tests/test_openai_client.py
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Stub the `openai` SDK before importing the adapter.
# ---------------------------------------------------------------------------

openai_mod = types.ModuleType("openai")
sys.modules["openai"] = openai_mod

# Mock OpenAI response classes
class MockChatCompletionChunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [types.SimpleNamespace(
            delta=types.SimpleNamespace(
                content=content,
                tool_calls=tool_calls
            )
        )]
        self.model = "gpt-4o-mini"
        self.id = "chatcmpl-test"

class MockChatCompletion:
    def __init__(self, content="Test response", tool_calls=None):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(
                content=content,
                tool_calls=tool_calls
            )
        )]
        self.model = "gpt-4o-mini"
        self.id = "chatcmpl-test"

# Mock AsyncOpenAI client
class MockAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace()
        self.chat.completions = types.SimpleNamespace()
        self.chat.completions.create = AsyncMock()

# Mock sync OpenAI client (for backwards compatibility)
class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace()
        self.chat.completions = types.SimpleNamespace()
        self.chat.completions.create = MagicMock()

openai_mod.AsyncOpenAI = MockAsyncOpenAI
openai_mod.OpenAI = MockOpenAI

# ---------------------------------------------------------------------------
# Now import the client (will see the stub).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.openai_client import OpenAILLMClient  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return OpenAILLMClient(model="gpt-4o-mini", api_key="fake-key")

# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_stream(client):
    """Test non-streaming completion."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    
    # Mock the async client response
    mock_response = MockChatCompletion(content="Hello there!")
    client.async_client.chat.completions.create.return_value = mock_response
    
    # Get the awaitable result
    result_awaitable = client.create_completion(messages, stream=False)
    assert hasattr(result_awaitable, '__await__')
    
    # Await the result
    result = await result_awaitable
    
    # Verify response format
    assert result["response"] == "Hello there!"
    assert result["tool_calls"] == []
    
    # Verify the API was called correctly
    client.async_client.chat.completions.create.assert_called_once()
    call_kwargs = client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["stream"] is False
    assert len(call_kwargs["messages"]) == 2

@pytest.mark.asyncio
async def test_create_completion_with_tools_non_stream(client):
    """Test non-streaming completion with tools."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object"}
            }
        }
    ]

    # Mock the response with tool calls
    expected_result = {
        "response": None,  # Should be None when tool calls are present
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}'
                }
            }
        ]
    }

    # Mock _regular_completion method with correct signature
    async def fake_regular_completion(self, msgs, tls, **kwargs):  # Add 'self' parameter
        return expected_result

    import types
    client._regular_completion = types.MethodType(fake_regular_completion, client)

    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    result = await result_awaitable

    # Verify tool calls in response - should be None when tools are used
    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"

# ---------------------------------------------------------------------------
# Streaming tests  
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_stream(client):
    """Test streaming completion."""
    messages = [{"role": "user", "content": "Tell me a story"}]
    
    # Mock streaming response
    async def mock_stream():
        yield MockChatCompletionChunk(content="Once")
        yield MockChatCompletionChunk(content=" upon") 
        yield MockChatCompletionChunk(content=" a time")
    
    client.async_client.chat.completions.create.return_value = mock_stream()
    
    # Get the async generator directly
    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, '__aiter__')
    assert not hasattr(stream_result, '__await__')
    
    # Collect chunks
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 3
    assert chunks[0]["response"] == "Once"
    assert chunks[1]["response"] == " upon"
    assert chunks[2]["response"] == " a time"
    
    # Verify API call
    client.async_client.chat.completions.create.assert_called_once()
    call_kwargs = client.async_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True

@pytest.mark.asyncio
async def test_create_completion_stream_with_tools(client):
    """Test streaming completion with tool calls."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"function": {"name": "get_weather"}}]
    
    # Mock streaming response with tool calls
    mock_tool_call = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="get_weather",
            arguments='{"location": "NYC"}'
        )
    )
    
    async def mock_stream():
        yield MockChatCompletionChunk(content="Let me check")
        yield MockChatCompletionChunk(content="", tool_calls=[mock_tool_call])
        yield MockChatCompletionChunk(content=" the weather")
    
    client.async_client.chat.completions.create.return_value = mock_stream()
    
    # Get streaming result
    stream_result = client.create_completion(messages, tools=tools, stream=True)
    
    # Collect chunks
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) == 3
    assert chunks[0]["response"] == "Let me check"
    assert chunks[0]["tool_calls"] == []
    
    assert chunks[1]["response"] == ""
    assert len(chunks[1]["tool_calls"]) == 1
    assert chunks[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    
    assert chunks[2]["response"] == " the weather"

# ---------------------------------------------------------------------------
# Interface compliance tests
# ---------------------------------------------------------------------------

def test_interface_compliance(client):
    """Test that the client follows the correct interface."""
    messages = [{"role": "user", "content": "test"}]
    
    # Streaming should return async generator directly
    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, '__aiter__')
    assert not hasattr(stream_result, '__await__')
    
    # Non-streaming should return awaitable
    non_stream_result = client.create_completion(messages, stream=False)
    assert hasattr(non_stream_result, '__await__')
    assert not hasattr(non_stream_result, '__aiter__')

# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_error_handling(client):
    """Test error handling in streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock an exception during streaming
    async def mock_error_stream():
        yield MockChatCompletionChunk(content="Starting...")
        raise Exception("Network error")
    
    client.async_client.chat.completions.create.return_value = mock_error_stream()
    
    # Get streaming result
    stream_result = client.create_completion(messages, stream=True)
    
    # Should yield error chunk
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    # Should have at least one chunk and an error chunk
    assert len(chunks) >= 1
    error_chunk = chunks[-1] 
    assert "error" in error_chunk
    assert "Network error" in error_chunk["response"]

@pytest.mark.asyncio
async def test_non_streaming_error_handling(client):
    """Test error handling in non-streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock an exception
    client.async_client.chat.completions.create.side_effect = Exception("API error")
    
    result_awaitable = client.create_completion(messages, stream=False)
    result = await result_awaitable
    
    assert "error" in result
    assert "API error" in result["response"]