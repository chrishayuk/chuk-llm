# tests/test_gemini_client.py
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


class Part(_Simple):
    @staticmethod
    def from_text(text: str):
        return Part(text=text)

    @staticmethod
    def from_function_response(name: str, response):
        return Part(func_resp={"name": name, "response": response})

    @staticmethod
    def from_function_call(name: str, args):
        return Part(func_call={"name": name, "args": args})


types_mod.Part = Part
types_mod.Content = _Simple
types_mod.Schema = _Simple
types_mod.FunctionDeclaration = _Simple
types_mod.Tool = _Simple

class _FCCMode:
    AUTO = "AUTO"

types_mod.FunctionCallingConfigMode = _FCCMode
types_mod.FunctionCallingConfig = _Simple
types_mod.ToolConfig = _Simple
types_mod.GenerateContentConfig = _Simple

# ---------------------------------------------------------------------------
# Fake client that the adapter will instantiate
# ---------------------------------------------------------------------------

class _DummyModels:
    def generate_content(self, *a, **k):
        # Never used in our patched tests – but must exist
        return None

    def generate_content_stream(self, *a, **k):
        # Never used because we patch _stream – but must exist
        return []

class DummyGenAIClient:
    def __init__(self, *args, **kwargs):
        self.models = _DummyModels()

genai_mod.Client = DummyGenAIClient

# ---------------------------------------------------------------------------
# Now import the adapter under test (it will pick up the stubs).
# ---------------------------------------------------------------------------

from chuk_llm.llm.providers.gemini_client import GeminiLLMClient, _convert_messages, _convert_tools, _parse_final_response, _parse_stream_chunk  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Fixture producing a fresh client for each test.
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return GeminiLLMClient(model="gemini-test", api_key="fake-key")

# ---------------------------------------------------------------------------
# Helper conversion functions (keeping existing tests)
# ---------------------------------------------------------------------------

def test_convert_messages_basic():
    """Test basic message conversion with simple text content."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello Gemini"},
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
    
    system_txt, gem_contents = _convert_messages(messages)
    
    # Check system text was extracted
    assert system_txt == "You are a helpful assistant."
    
    # Check converted messages
    assert len(gem_contents) == 2  # System message is extracted, not in contents
    
    # Check user message
    assert gem_contents[0].role == "user"
    assert len(gem_contents[0].parts) == 1
    assert getattr(gem_contents[0].parts[0], "text", None) == "Hello Gemini"
    
    # Check assistant message
    assert gem_contents[1].role == "model"
    assert len(gem_contents[1].parts) == 1
    assert getattr(gem_contents[1].parts[0], "text", None) == "Hello! How can I help you today?"

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
    
    gem_tools, tool_cfg = _convert_tools(tools)
    
    # Check tools list
    assert len(gem_tools) == 1
    assert len(gem_tools[0].function_declarations) == 1
    assert gem_tools[0].function_declarations[0].name == "get_weather"
    assert gem_tools[0].function_declarations[0].description == "Get the current weather"
    
    # Check tool config
    assert tool_cfg is not None
    assert hasattr(tool_cfg, "function_calling_config")
    assert tool_cfg.function_calling_config.mode == "AUTO"

def test_parse_stream_chunk_text():
    """Test parsing a stream chunk with text content."""
    # Mock a chunk with text
    chunk = MagicMock()
    chunk.text = "Hello, how are you?"
    
    delta_text, tool_calls = _parse_stream_chunk(chunk)
    
    assert delta_text == "Hello, how are you?"
    assert tool_calls == []

def test_parse_final_response_text():
    """Test parsing a final response with text content."""
    # Mock a response with text
    resp = MagicMock()
    candidate = MagicMock()
    content = MagicMock()
    part = MagicMock()
    part.text = "Final response text"
    content.parts = [part]
    candidate.content = content
    resp.candidates = [candidate]
    
    result = _parse_final_response(resp)
    
    assert result["response"] == "Final response text"
    assert result["tool_calls"] == []

# ---------------------------------------------------------------------------
# Non-streaming path – UPDATED for new interface
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_completion_non_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Hello Gemini"}]
    tools = [{"type": "function", "function": {"name": "demo.fn", "parameters": {}}}]

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
        yield {"response": "chunk2", "tool_calls": [{"name": "fn"}]}

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
        {"response": "chunk2", "tool_calls": [{"name": "fn"}]},
    ]

# ---------------------------------------------------------------------------
# Interface compliance test (NEW)
# ---------------------------------------------------------------------------

def test_interface_compliance(client):
    """Test that create_completion follows the correct interface."""
    messages = [{"role": "user", "content": "test"}]
    
    # Streaming should return async generator directly (not awaitable)
    stream_result = client.create_completion(messages, stream=True)
    assert hasattr(stream_result, '__aiter__')
    assert not hasattr(stream_result, '__await__')
    
    # Non-streaming should return awaitable
    non_stream_result = client.create_completion(messages, stream=False) 
    assert hasattr(non_stream_result, '__await__')
    assert not hasattr(non_stream_result, '__aiter__')

# ---------------------------------------------------------------------------
# Test new streaming implementation (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_completion_async(monkeypatch, client):
    """Test the new _stream_completion_async method."""
    messages = [{"role": "user", "content": "Stream please"}]
    tools = [{"type": "function", "function": {"name": "helper", "parameters": {}}}]
    
    # Mock _convert_messages
    system_txt = None
    gem_contents = [MagicMock()]
    monkeypatch.setattr(
        "chuk_llm.llm.providers.gemini_client._convert_messages", 
        lambda m: (system_txt, gem_contents)
    )
    
    # Mock _convert_tools
    gem_tools = [MagicMock()]
    tool_cfg = MagicMock()
    monkeypatch.setattr(
        "chuk_llm.llm.providers.gemini_client._convert_tools", 
        lambda t: (gem_tools, tool_cfg)
    )
    
    # Mock stream response objects
    class MockStreamItem:
        def __init__(self, text):
            self.text = text
    
    mock_stream_items = [
        MockStreamItem("Hello"),
        MockStreamItem(" world")
    ]
    
    # Mock asyncio.to_thread to return the stream directly
    async def fake_to_thread(func):
        return mock_stream_items
    
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    
    # Mock _parse_stream_chunk
    parse_results = [
        ("Hello", []),
        (" world", [])
    ]
    
    parse_mock = MagicMock(side_effect=parse_results)
    monkeypatch.setattr("chuk_llm.llm.providers.gemini_client._parse_stream_chunk", parse_mock)
    
    # Call _stream_completion_async and collect results
    stream = client._stream_completion_async(messages, tools)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify results
    assert len(chunks) == 2
    assert chunks[0] == {"response": "Hello", "tool_calls": []}
    assert chunks[1] == {"response": " world", "tool_calls": []}

# ---------------------------------------------------------------------------
# Test regular completion (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regular_completion(monkeypatch, client):
    """Test the _regular_completion method."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "get_info", "parameters": {}}}]
    
    # Mock asyncio.to_thread to call _create_sync
    expected_result = {"response": "Test response", "tool_calls": []}
    
    async def fake_to_thread(func, *args):
        # Should call _create_sync with messages and tools
        assert args == (messages, tools)
        return expected_result
    
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    
    # Call _regular_completion
    result = await client._regular_completion(messages, tools)
    
    # Verify result
    assert result == expected_result

# ---------------------------------------------------------------------------
# Error handling tests (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_error_handling(monkeypatch, client):
    """Test error handling in streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock the actual _stream_completion_async method using monkeypatch
    async def error_stream(msgs, tls, **kwargs):
        yield {"response": "Starting...", "tool_calls": []}
        # This should be caught and converted to error chunk
        raise Exception("Stream error")
    
    # Use monkeypatch for cleaner method replacement
    monkeypatch.setattr(client, "_stream_completion_async", error_stream)
    
    stream_result = client.create_completion(messages, stream=True)
    
    # Should handle error gracefully and yield error chunk
    chunks = []
    async for chunk in stream_result:
        chunks.append(chunk)
    
    # Should have starting chunk and potentially error chunk
    assert len(chunks) >= 1
    assert chunks[0]["response"] == "Starting..."

@pytest.mark.asyncio
async def test_non_streaming_error_handling(monkeypatch, client):
    """Test error handling in non-streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock the actual _regular_completion method using monkeypatch
    async def error_completion(msgs, tls, **kwargs):
        raise Exception("Completion error")
    
    # Use monkeypatch for cleaner method replacement
    monkeypatch.setattr(client, "_regular_completion", error_completion)
    
    result_awaitable = client.create_completion(messages, stream=False)
    
    # The error should be handled by the actual implementation or bubble up
    try:
        result = await result_awaitable
        # If no error handling in implementation, this might fail
        assert "error" in result or "Error" in str(result)
    except Exception as e:
        # If error bubbles up, that's also acceptable behavior for a test
        assert "Completion error" in str(e)

@pytest.mark.asyncio
async def test_non_streaming_error_handling(client):
    """Test error handling in non-streaming."""
    messages = [{"role": "user", "content": "test"}]
    
    # Mock the actual _regular_completion method with correct signature
    async def error_completion(self, msgs, tls, **kwargs):  # Add 'self' parameter
        raise Exception("Completion error")
    
    import types
    client._regular_completion = types.MethodType(error_completion, client)
    
    result_awaitable = client.create_completion(messages, stream=False)
    
    # The error should be handled by the actual implementation or bubble up
    try:
        result = await result_awaitable
        # If no error handling in implementation, this might fail
        assert "error" in result or "Error" in str(result)
    except Exception as e:
        # If error bubbles up, that's also acceptable behavior for a test
        assert "Completion error" in str(e)

# ---------------------------------------------------------------------------
# Test with tools (NEW)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_with_tools(monkeypatch, client):
    """Test streaming with tool calls."""
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [{"function": {"name": "get_weather", "parameters": {}}}]

    # Mock streaming with tool calls
    async def fake_stream_with_tools(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == tools
        
        yield {"response": "Let me check", "tool_calls": []}
        yield {"response": "", "tool_calls": [
            {"id": "call_123", "function": {"name": "get_weather", "arguments": "{}"}}
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
    tools = [{"function": {"name": "get_weather"}}]

    # Mock regular completion with tool calls
    async def fake_regular_with_tools(msgs, tls, **kwargs):
        assert msgs == messages
        assert tls == tools
        
        return {
            "response": "Weather check complete",
            "tool_calls": [
                {"id": "call_123", "function": {"name": "get_weather", "arguments": "{}"}}
            ]
        }

    monkeypatch.setattr(client, "_regular_completion", fake_regular_with_tools)

    result_awaitable = client.create_completion(messages, tools=tools, stream=False)
    result = await result_awaitable

    assert result["response"] == "Weather check complete"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"

# ---------------------------------------------------------------------------
# Edge cases and error handling (keeping existing tests)
# ---------------------------------------------------------------------------

def test_initialization_options():
    """Test client initialization with different options."""
    # Test with just model
    client1 = GeminiLLMClient(model="gemini-model")
    assert client1.model == "gemini-model"
    
    # Test with API key
    client2 = GeminiLLMClient(model="gemini-model", api_key="test-key")
    assert client2.model == "gemini-model"
    
    # Test with default model
    client3 = GeminiLLMClient()
    assert client3.model == "gemini-2.0-flash"

def test_convert_messages_empty():
    """Test message conversion with empty input."""
    system_txt, gem_contents = _convert_messages([])
    
    assert system_txt is None
    assert gem_contents == []

def test_convert_tools_empty():
    """Test conversion with no tools."""
    gem_tools, tool_cfg = _convert_tools(None)
    
    assert gem_tools == []
    assert tool_cfg is None