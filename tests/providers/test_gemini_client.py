# tests/test_gemini_client.py
import sys
import types
import asyncio
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

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
# Helper conversion functions
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


def test_convert_messages_tool_calls():
    """Test conversion of messages with tool calls."""
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {
                "id": "call_1", 
                "type": "function", 
                "function": {
                    "name": "weather", 
                    "arguments": '{"location": "London"}'
                }
            }
        ]},
        {"role": "tool", "content": '{"temp": 15, "conditions": "Cloudy"}', "name": "weather"}
    ]
    
    system_txt, gem_contents = _convert_messages(messages)
    
    # No system message
    assert system_txt is None
    
    # Check assistant/tool messages
    assert len(gem_contents) == 2
    
    # Check assistant message with tool call
    assert gem_contents[0].role == "model"
    assert hasattr(gem_contents[0].parts[0], "func_call")
    assert gem_contents[0].parts[0].func_call["name"] == "weather"
    assert gem_contents[0].parts[0].func_call["args"] == {"location": "London"}
    
    # Check tool response
    assert gem_contents[1].role == "tool"
    assert hasattr(gem_contents[1].parts[0], "func_resp")
    assert gem_contents[1].parts[0].func_resp["name"] == "weather"


def test_convert_messages_multimodal():
    """Test conversion of messages with multimodal content."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image_url": "data:image/jpeg;base64,/9j/..."}
        ]}
    ]
    
    system_txt, gem_contents = _convert_messages(messages)
    
    # No system message
    assert system_txt is None
    
    # Check multimodal content passed through
    assert len(gem_contents) == 1
    assert gem_contents[0].role == "user"
    assert gem_contents[0].parts == messages[0]["content"]


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


def test_convert_tools_empty():
    """Test conversion with no tools."""
    gem_tools, tool_cfg = _convert_tools(None)
    
    assert gem_tools == []
    assert tool_cfg is None


def test_convert_tools_invalid_schema():
    """Test conversion with invalid schema."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "parameters": "not a valid schema"  # Invalid schema
            }
        }
    ]
    
    gem_tools, tool_cfg = _convert_tools(tools)
    
    # Should register the tool with a permissive schema
    assert len(gem_tools) == 1
    assert len(gem_tools[0].function_declarations) == 1
    assert gem_tools[0].function_declarations[0].name == "test_function"
    assert hasattr(gem_tools[0].function_declarations[0].parameters, "additionalProperties")
    assert gem_tools[0].function_declarations[0].parameters.additionalProperties is True


# ---------------------------------------------------------------------------
# Response parsing functions
# ---------------------------------------------------------------------------

def test_parse_stream_chunk_text():
    """Test parsing a stream chunk with text content."""
    # Mock a chunk with text
    chunk = MagicMock()
    chunk.text = "Hello, how are you?"
    
    delta_text, tool_calls = _parse_stream_chunk(chunk)
    
    assert delta_text == "Hello, how are you?"
    assert tool_calls == []


def test_parse_stream_chunk_candidates():
    """Test parsing a stream chunk with candidates."""
    # Mock a chunk with candidates
    chunk = MagicMock()
    chunk.text = ""
    
    # Add candidate with text
    candidate = MagicMock()
    content = MagicMock()
    part = MagicMock()
    part.text = "Hello from candidate"
    content.parts = [part]
    candidate.content = content
    chunk.candidates = [candidate]
    
    delta_text, tool_calls = _parse_stream_chunk(chunk)
    
    assert delta_text == "Hello from candidate"
    assert tool_calls == []


def test_parse_stream_chunk_function_call():
    """Test parsing a stream chunk with a function call."""
    # Mock a chunk with function call
    chunk = MagicMock()
    chunk.text = ""
    
    # Add candidate with function call
    candidate = MagicMock()
    content = MagicMock()
    
    # Create a function call part
    part = MagicMock()
    part.text = None
    part.function_call = MagicMock()
    part.function_call.name = "get_weather"
    part.function_call.args = {"location": "London"}
    
    content.parts = [part]
    candidate.content = content
    chunk.candidates = [candidate]
    
    # Mock UUID for deterministic testing
    with patch('uuid.uuid4', return_value=MagicMock(hex='12345678abcdef')):
        delta_text, tool_calls = _parse_stream_chunk(chunk)
    
    assert delta_text == ""  # No text content
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_12345678"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "London"}


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


def test_parse_final_response_function_call():
    """Test parsing a final response with a function call."""
    # Mock a response with function call
    resp = MagicMock()
    candidate = MagicMock()
    content = MagicMock()
    
    # Create a function call part
    part = MagicMock()
    part.text = None
    part.function_call = MagicMock()
    part.function_call.name = "get_weather"
    part.function_call.args = {"location": "London"}
    
    content.parts = [part]
    candidate.content = content
    resp.candidates = [candidate]
    
    # Mock UUID for deterministic testing
    with patch('uuid.uuid4', return_value=MagicMock(hex='12345678abcdef')):
        result = _parse_final_response(resp)
    
    assert result["response"] is None  # No text when tool calls are present
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call_12345678"
    assert result["tool_calls"][0]["type"] == "function"
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"location": "London"}


# ---------------------------------------------------------------------------
# Non-streaming path – create_completion(stream=False)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_non_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Hello Gemini"}]
    tools = [{"type": "function", "function": {"name": "demo.fn", "parameters": {}}}]

    # Patch asyncio.to_thread so it simply calls the fn inline (no threads).
    async def fake_to_thread(func, *args, **kwargs):  # noqa: D401
        return func(*args, **kwargs)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    # Patch the _create_sync method to validate the forwarded args and return a
    # predictable result.
    def fake_create_sync(self, msgs, tls):  # noqa: D401 – simple func
        assert msgs == messages
        assert tls == tools
        return {"response": "Hi!", "tool_calls": []}

    monkeypatch.setattr(GeminiLLMClient, "_create_sync", fake_create_sync, raising=True)

    out = await client.create_completion(messages, tools=tools, stream=False)
    assert out == {"response": "Hi!", "tool_calls": []}


# ---------------------------------------------------------------------------
# Internal sync method test
# ---------------------------------------------------------------------------


def test_create_sync(monkeypatch, client):
    """Test the _create_sync method."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "get_info", "parameters": {}}}]
    
    # Mock _convert_messages
    system_txt = "Be helpful"
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
    
    # Mock generate_content
    mock_response = MagicMock()
    client.client.models.generate_content = MagicMock(return_value=mock_response)
    
    # Mock _parse_final_response
    expected_result = {"response": "Test response", "tool_calls": []}
    monkeypatch.setattr(
        "chuk_llm.llm.providers.gemini_client._parse_final_response", 
        lambda r: expected_result
    )
    
    # Call the method
    result = client._create_sync(messages, tools)
    
    # Verify result
    assert result == expected_result
    
    # Verify generate_content was called with correct parameters
    client.client.models.generate_content.assert_called_once()
    call_kwargs = client.client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-test"
    assert call_kwargs["contents"] == gem_contents
    assert call_kwargs["config"].system_instruction == system_txt
    assert call_kwargs["config"].tools == gem_tools
    assert call_kwargs["config"].tool_config == tool_cfg


# ---------------------------------------------------------------------------
# Streaming path – create_completion(stream=True)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_completion_stream(monkeypatch, client):
    messages = [{"role": "user", "content": "Stream it"}]

    # Fake _stream returns an async-generator so we can iterate
    async def _gen():
        yield {"response": "chunk1", "tool_calls": []}
        yield {"response": "chunk2", "tool_calls": [{"name": "fn"}]}

    async def fake_stream(self, msgs, tls):  # noqa: D401
        assert msgs == messages
        # No tools provided → adapter should forward None
        assert tls is None
        async for chunk in _gen():
            yield chunk

    monkeypatch.setattr(GeminiLLMClient, "_stream", fake_stream, raising=True)

    # Call streaming variant
    ai = await client.create_completion(messages, tools=None, stream=True)
    assert hasattr(ai, "__aiter__")

    pieces = [c async for c in ai]
    assert pieces == [
        {"response": "chunk1", "tool_calls": []},
        {"response": "chunk2", "tool_calls": [{"name": "fn"}]},
    ]


# Fix for the failing test_stream_internal
async def test_stream_internal(monkeypatch, client):
    """Test the internal _stream method."""
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
    
    # Mock streaming response
    class MockStreamItem:
        def __init__(self, text, tool_calls=None):
            self.text = text
            self.tool_calls = tool_calls
    
    mock_stream_items = [
        MockStreamItem("Hello"),
        MockStreamItem("World")
    ]
    
    client.client.models.generate_content_stream = MagicMock(return_value=mock_stream_items)
    
    # Mock _parse_stream_chunk
    parse_results = [
        ("Hello", []),
        ("World", [{"id": "call_1", "type": "function", "function": {"name": "helper", "arguments": "{}"}}])
    ]
    
    parse_mock = MagicMock(side_effect=parse_results)
    monkeypatch.setattr("chuk_llm.llm.providers.gemini_client._parse_stream_chunk", parse_mock)
    
    # Create async queue mock
    class MockQueue:
        def __init__(self):
            self.items = []
            
        async def get(self):
            if not self.items:
                return None
            return self.items.pop(0)
            
        def put_nowait(self, item):
            self.items.append(item)
    
    mock_queue = MockQueue()
    mock_queue.put_nowait(mock_stream_items[0])
    mock_queue.put_nowait(mock_stream_items[1])
    mock_queue.put_nowait(None)  # End sentinel
    
    monkeypatch.setattr(asyncio, "Queue", lambda: mock_queue)
    
    # Mock run_in_executor to just call the function
    # Fix: Add the missing executor parameter
    def fake_run_in_executor(executor_param, pool, func):
        func()
        
    mock_loop = MagicMock()
    mock_loop.run_in_executor = fake_run_in_executor
    
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: mock_loop)
    
    # Call _stream and collect results
    stream = client._stream(messages, tools)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    
    # Verify results
    assert len(chunks) == 2
    assert chunks[0] == {"response": "Hello", "tool_calls": []}
    assert chunks[1] == {"response": "World", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "helper", "arguments": "{}"}}]}
    
    # Verify generate_content_stream was called with correct parameters
    client.client.models.generate_content_stream.assert_called_once()
    call_kwargs = client.client.models.generate_content_stream.call_args.kwargs
    assert call_kwargs["model"] == "gemini-test"
    assert call_kwargs["contents"] == gem_contents
    assert call_kwargs["config"].tools == gem_tools
    assert call_kwargs["config"].tool_config == tool_cfg

# ---------------------------------------------------------------------------
# Edge cases and error handling
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


def test_convert_messages_multiple_system():
    """Test message conversion with multiple system messages."""
    messages = [
        {"role": "system", "content": "First system message"},
        {"role": "system", "content": "Second system message"},
        {"role": "user", "content": "Hello"}
    ]
    
    system_txt, gem_contents = _convert_messages(messages)
    
    # Only the first system message should be used
    assert system_txt == "First system message"
    assert len(gem_contents) == 1
    assert gem_contents[0].role == "user"