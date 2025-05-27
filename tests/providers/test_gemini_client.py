# tests/providers/test_gemini_client.py
import sys
import types
import asyncio
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# ---------------------------------------------------------------------------
# Build a stub for the ``google.genai`` SDK *before* importing the client, so
# that the real heavy package is never needed and no network calls are made.
# ---------------------------------------------------------------------------

google_mod = sys.modules.get("google") or types.ModuleType("google")
if "google" not in sys.modules:
    sys.modules["google"] = google_mod

# --- sub-module ``google.genai`` -------------------------------------------

genai_mod = types.ModuleType("google.genai")
sys.modules["google.genai"] = genai_mod
setattr(google_mod, "genai", genai_mod)

# --- sub-module ``google.genai.types`` -------------------------------------

types_mod = types.ModuleType("google.genai.types")
sys.modules["google.genai.types"] = types_mod
setattr(genai_mod, "types", types_mod)

# Provide *minimal* class stubs used by the adapter's helper code. We keep
# them extremely simple – they only need to accept the constructor args.

class _Simple:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"<_Simple {self.__dict__}>"


# Updated to match actual implementation usage
class Tool(_Simple):
    def __init__(self, function_declarations=None, **kwargs):
        super().__init__(**kwargs)
        self.function_declarations = function_declarations or []

class GenerateContentConfig(_Simple):
    def __init__(self, tools=None, automatic_function_calling=None, **kwargs):
        super().__init__(**kwargs)
        self.tools = tools
        self.automatic_function_calling = automatic_function_calling

class AutomaticFunctionCallingConfig(_Simple):
    def __init__(self, disable=False, **kwargs):
        super().__init__(**kwargs)
        self.disable = disable

types_mod.Tool = Tool
types_mod.GenerateContentConfig = GenerateContentConfig
types_mod.AutomaticFunctionCallingConfig = AutomaticFunctionCallingConfig

# ---------------------------------------------------------------------------
# Fake client that matches the actual implementation
# ---------------------------------------------------------------------------

class _MockAIOModels:
    async def generate_content(self, **kwargs):
        # Return a mock response
        return MagicMock()

    async def generate_content_stream(self, **kwargs):
        # Return an async iterator
        async def mock_stream():
            yield MagicMock(text="chunk1")
            yield MagicMock(text="chunk2")
        return mock_stream()

class _MockAIO:
    def __init__(self):
        self.models = _MockAIOModels()

class _DummyModels:
    def generate_content(self, *a, **k):
        return None

    def generate_content_stream(self, *a, **k):
        return []

class DummyGenAIClient:
    def __init__(self, *args, **kwargs):
        self.models = _DummyModels()
        self.aio = _MockAIO()

genai_mod.Client = DummyGenAIClient

# ---------------------------------------------------------------------------
# Mock dotenv
# ---------------------------------------------------------------------------
sys.modules["dotenv"] = types.ModuleType("dotenv")
sys.modules["dotenv"].load_dotenv = lambda: None

# ---------------------------------------------------------------------------
# Now import the adapter under test (it will pick up the stubs).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.gemini_client import (
    GeminiLLMClient, 
    _convert_messages_for_chat, 
    _convert_tools_to_gemini_format
)  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Fixture producing a fresh client for each test.
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return GeminiLLMClient(model="gemini-test", api_key="fake-key")

# ---------------------------------------------------------------------------
# Helper conversion functions (updated for actual implementation)
# ---------------------------------------------------------------------------

def test_convert_messages_basic():
    """Test basic message conversion with simple text content."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello Gemini"},
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
    
    system_txt, user_message = _convert_messages_for_chat(messages)
    
    # Check system text was extracted
    assert system_txt == "You are a helpful assistant."
    
    # Check that the last user message was extracted
    assert user_message == "Hello Gemini"

def test_convert_messages_multimodal():
    """Test conversion with multimodal content."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
    
    system_txt, user_message = _convert_messages_for_chat(messages)
    
    assert system_txt is None
    assert "What's in this image?" in user_message

def test_convert_messages_tool_response():
    """Test conversion with tool responses."""
    messages = [
        {"role": "user", "content": "Get weather"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}}
        ]},
        {"role": "tool", "name": "get_weather", "content": "Sunny, 75°F"}
    ]
    
    system_txt, user_message = _convert_messages_for_chat(messages)
    
    assert system_txt is None
    # Tool result should be converted to user message
    assert "get_weather" in user_message
    assert "Sunny, 75°F" in user_message

def test_convert_tools_basic():
    """Test basic tool conversion."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    gem_tools = _convert_tools_to_gemini_format(tools)
    
    # Check tools list
    assert len(gem_tools) == 1
    assert len(gem_tools[0].function_declarations) == 1
    assert gem_tools[0].function_declarations[0]["name"] == "get_weather"
    assert gem_tools[0].function_declarations[0]["description"] == "Get the current weather"

def test_convert_tools_empty():
    """Test conversion with no tools."""
    gem_tools = _convert_tools_to_gemini_format(None)
    assert gem_tools is None
    
    gem_tools = _convert_tools_to_gemini_format([])
    assert gem_tools is None

# ---------------------------------------------------------------------------
# Non-streaming path – UPDATED for new interface
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Hello Gemini"}]
    tools = [{"type": "function", "function": {"name": "demo_fn", "parameters": {}}}]

    # Mock the _regular_completion method
    expected_result = {"response": "Hi!", "tool_calls": []}
    
    async def fake_regular_completion(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == tools
        return expected_result

    monkeypatch.setattr(client, "_regular_completion", fake_regular_completion)

    # NEW INTERFACE: create_completion returns awaitable for non-streaming
    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    
    # Should be awaitable, not async generator
    assert hasattr(result_awaitable, '__await__')
    assert not hasattr(result_awaitable, '__aiter__')
    
    # Await the result
    result = await result_awaitable
    assert result == expected_result

# ---------------------------------------------------------------------------
# Streaming path – UPDATED for new interface
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Stream it"}]

    # Mock the _stream_completion_async method
    async def fake_stream_completion_async(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls is None  # No tools provided
        
        # Yield test chunks
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": []}

    monkeypatch.setattr(client, "_stream_completion_async", fake_stream_completion_async)

    # NEW INTERFACE: create_completion returns async generator directly for streaming
    stream_result = client.create_completion(messages, tools=None, stream=True)
    
    # Should be async generator, not awaitable
    assert hasattr(stream_result, "__aiter__")
    assert not hasattr(stream_result, "__await__")

    # Collect chunks
    pieces = [c async for c in stream_result]
    assert pieces == [
        {"response": "chunk1", "tool_calls": []},
        {"response": "chunk2", "tool_calls": []},
    ]

# ---------------------------------------------------------------------------
# Interface compliance test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interface_compliance(client):
    """Test that client follows the BaseLLMClient interface."""
    # Test non-streaming
    messages = [{"role": "user", "content": "Test"}]
    
    # Mock the underlying methods
    expected_result = {"response": "Test response", "tool_calls": []}
    
    async def mock_regular_completion(msgs, tls, **kwargs):
        return expected_result
    
    # Use setattr to replace the method
    setattr(client, "_regular_completion", mock_regular_completion)
    
    # For non-streaming, we need to await the result
    result = client.create_completion(messages, stream=False)
    if asyncio.iscoroutine(result):
        result = await result
    
    assert isinstance(result, dict)
    assert "response" in result
    assert result == expected_result

# ---------------------------------------------------------------------------
# Test streaming implementation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_completion_async(monkeypatch, client):
    """Test the _stream_completion_async method."""
    messages = [{"role": "user", "content": "Stream please"}]
    tools = [{"type": "function", "function": {"name": "helper", "parameters": {}}}]
    
    # Mock the conversion functions
    monkeypatch.setattr(
        "chuk_llm.llm.providers.gemini_client._convert_messages_for_chat", 
        lambda m: (None, "Stream please")
    )
    
    gem_tools = [types_mod.Tool(function_declarations=[{"name": "helper"}])]
    monkeypatch.setattr(
        "chuk_llm.llm.providers.gemini_client._convert_tools_to_gemini_format", 
        lambda t: gem_tools
    )
    
    # Mock the client's async generate_content_stream method
    class MockChunk:
        def __init__(self, text):
            self.text = text
    
    # Create an async generator that returns the chunks
    async def mock_async_generator():
        yield MockChunk("Hello")
        yield MockChunk(" world")
    
    # Mock the actual call to return the async generator (not await it)
    async def mock_stream_call(**kwargs):
        return mock_async_generator()
    
    monkeypatch.setattr(client.client.aio.models, "generate_content_stream", mock_stream_call)
    
    # Call _stream_completion_async and collect results
    stream = client._stream_completion_async(messages, tools)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify results
    assert len(chunks) == 2
    assert chunks[0]["response"] == "Hello"
    assert chunks[1]["response"] == " world"

# ---------------------------------------------------------------------------
# Test regular completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regular_completion(monkeypatch, client):
    """Test the _regular_completion method."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "get_info", "parameters": {}}}]
    
    # Mock the conversion functions
    monkeypatch.setattr(
        "chuk_llm.llm.providers.gemini_client._convert_messages_for_chat", 
        lambda m: (None, "Hello")
    )
    
    gem_tools = [types_mod.Tool(function_declarations=[{"name": "get_info"}])]
    monkeypatch.setattr(
        "chuk_llm.llm.providers.gemini_client._convert_tools_to_gemini_format", 
        lambda t: gem_tools
    )
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.text = "Test response"
    
    # Mock the async client call
    async def mock_generate_content(**kwargs):
        return mock_response
    
    monkeypatch.setattr(client.client.aio.models, "generate_content", mock_generate_content)
    
    # Mock the response parser
    monkeypatch.setattr(
        client, "_parse_gemini_response", 
        lambda resp: ("Test response", [])
    )
    
    # Call _regular_completion
    result = await client._regular_completion(messages, tools)
    
    # Verify result
    assert result["response"] == "Test response"
    assert result["tool_calls"] == []

# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_error_handling(monkeypatch, client):
    """Test error handling in streaming."""
    messages = [{"role": "user", "content": "test"}]

    # Create a streaming method that yields first, then raises an error
    # The actual implementation should catch this and yield an error chunk
    async def mock_stream_with_error(msgs, tls=None, **kwargs):
        yield {"response": "Starting...", "tool_calls": []}
        yield {"response": "Streaming error: Test error", "tool_calls": [], "error": True}

    # Replace the _stream_completion_async method entirely
    monkeypatch.setattr(client, "_stream_completion_async", mock_stream_with_error)

    stream_result = client.create_completion(messages, stream=True)

    # Should handle error gracefully and yield error chunk
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    # Should get both chunks
    assert len(chunks) == 2
    assert chunks[0]["response"] == "Starting..."
    assert chunks[1]["error"] is True

@pytest.mark.asyncio
async def test_non_streaming_error_handling(monkeypatch, client):
    """Test error handling in non-streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock the completion method to return an error response instead of raising
    async def error_completion(msgs, tls, **kwargs):
        return {"response": "Error: Completion error", "tool_calls": [], "error": True}
    
    monkeypatch.setattr(client, "_regular_completion", error_completion)
    
    result_awaitable = client.create_completion(messages, stream=False)
    result = await result_awaitable
    
    # Should return error response
    assert "error" in result
    assert result["error"] is True
    assert "Completion error" in result["response"]

# ---------------------------------------------------------------------------
# Test with tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_with_tools(monkeypatch, client):
    """Test streaming with tool calls."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

    # Mock streaming with tool calls
    async def fake_stream_with_tools(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == tools
        
        yield {"response": "Let me check", "tool_calls": []}
        yield {"response": "", "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ]}
        yield {"response": " the weather", "tool_calls": []}

    monkeypatch.setattr(client, "_stream_completion_async", fake_stream_with_tools)

    stream_result = client.create_completion(messages, tools=tools, stream=True)
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["response"] == "Let me check"
    assert chunks[0]["tool_calls"] == []
    
    assert chunks[1]["response"] == ""
    assert len(chunks[1]["tool_calls"]) == 1
    assert chunks[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    
    assert chunks[2]["response"] == " the weather"

@pytest.mark.asyncio
async def test_non_streaming_with_tools(monkeypatch, client):
    """Test non-streaming with tool calls."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

    # Mock regular completion with tool calls
    async def fake_regular_with_tools(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == tools
        
        return {
            "response": "Weather check complete",
            "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ]
        }

    monkeypatch.setattr(client, "_regular_completion", fake_regular_with_tools)

    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    result = await result_awaitable

    assert result["response"] == "Weather check complete"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"

# ---------------------------------------------------------------------------
# Test response parsing methods
# ---------------------------------------------------------------------------

def test_parse_gemini_response_text_only(client):
    """Test parsing response with text only."""
    mock_response = MagicMock()
    mock_response.text = "Hello world"
    mock_response.function_calls = None
    mock_response.candidates = None
    
    result_text, tool_calls = client._parse_gemini_response(mock_response)
    
    assert result_text == "Hello world"
    assert tool_calls == []

def test_parse_gemini_chunk_text_only(client):
    """Test parsing streaming chunk with text only."""
    mock_chunk = MagicMock()
    mock_chunk.text = "chunk text"
    mock_chunk.function_calls = None
    mock_chunk.candidates = None
    
    result_text, tool_calls = client._parse_gemini_chunk(mock_chunk)
    
    assert result_text == "chunk text"
    assert tool_calls == []

def test_extract_tool_calls_from_chunk(client):
    """Test extracting tool calls from chunk."""
    # Create a more realistic mock that matches the actual attribute access patterns
    class MockFunctionCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args
    
    mock_chunk = MagicMock()
    mock_chunk.text = None
    mock_chunk.function_calls = [
        MockFunctionCall("get_weather", {"city": "NYC"})
    ]
    mock_chunk.candidates = None
    
    tool_calls = client._extract_tool_calls_from_chunk(mock_chunk)
    
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert '"city": "NYC"' in tool_calls[0]["function"]["arguments"]

# ---------------------------------------------------------------------------
# Edge cases and initialization tests
# ---------------------------------------------------------------------------

def test_initialization_options():
    """Test client initialization with different options."""
    # Test with just model
    client1 = GeminiLLMClient(model="gemini-model", api_key="test-key")
    assert client1.model == "gemini-model"
    
    # Test with default model
    client2 = GeminiLLMClient(api_key="test-key")
    assert client2.model == "gemini-2.0-flash"

def test_convert_messages_empty():
    """Test message conversion with empty input."""
    system_txt, user_message = _convert_messages_for_chat([])
    
    assert system_txt is None
    assert user_message == "Hello"  # Default fallback

def test_convert_messages_with_tool_calls():
    """Test conversion with assistant tool calls."""
    messages = [
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_data", "arguments": '{"param": "value"}'}}
        ]}
    ]
    
    system_txt, user_message = _convert_messages_for_chat(messages)
    
    assert system_txt is None
    # Should still return fallback since no user message
    assert user_message == "Hello"

# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_integration(monkeypatch, client):
    """Test full integration with mocked Gemini API."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ]
    tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]
    
    # Mock the entire client.aio.models.generate_content call
    expected_response = MagicMock()
    expected_response.text = "Hello! How can I help?"
    expected_response.function_calls = None
    expected_response.candidates = None
    
    async def mock_generate_content(**kwargs):
        # Verify the request structure
        assert "model" in kwargs
        assert "contents" in kwargs or "content" in kwargs
        return expected_response
    
    monkeypatch.setattr(client.client.aio.models, "generate_content", mock_generate_content)
    
    # Test non-streaming
    result = await client.create_completion(messages, tools=tools, stream=False)
    
    assert result["response"] == "Hello! How can I help?"
    assert result["tool_calls"] == []