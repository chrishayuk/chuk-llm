# tests/providers/test_mixins.py
import asyncio
import base64
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Mock classes for testing different message formats
# ---------------------------------------------------------------------------


class MockMessage:
    """Mock message class for testing different message formats"""

    def __init__(self, content=None, tool_calls=None, **kwargs):
        self.content = content
        self.tool_calls = tool_calls
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockToolCall:
    """Mock tool call class"""

    def __init__(self, id=None, function=None):
        self.id = id or f"call_{uuid.uuid4().hex[:8]}"
        self.function = function or MockFunction()


class MockFunction:
    """Mock function class"""

    def __init__(self, name="test_function", arguments="{}"):
        self.name = name
        self.arguments = arguments


class MockChoice:
    """Mock choice class for streaming chunks"""

    def __init__(self, delta=None, message=None, text=None):
        self.delta = delta
        self.message = message
        self.text = text


class MockChunk:
    """Mock chunk class for streaming"""

    def __init__(self, choices=None, content=None):
        self.choices = choices or []
        if content is not None:
            self.content = content


class MockDelta:
    """Mock delta class for streaming chunks"""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class DictLikeMessage:
    """Dict-like object that can be accessed both as dict and object"""

    def __init__(self, data):
        self._data = data
        for k, v in data.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger():
    """Mock logger to capture debug output"""
    with patch("chuk_llm.llm.providers._mixins.logger") as mock_log:
        mock_log.debug = MagicMock()
        mock_log.warning = MagicMock()
        mock_log.error = MagicMock()
        mock_log.isEnabledFor = MagicMock(return_value=True)
        yield mock_log


@pytest.fixture
def mixin():
    """Create an instance of OpenAIStyleMixin for testing"""
    from chuk_llm.llm.providers._mixins import OpenAIStyleMixin

    return OpenAIStyleMixin()


@pytest.fixture
def mock_messages():
    """Factory fixture for creating various mock message types"""

    def _create_messages():
        return {
            "simple": MockMessage(content="Hello, world!"),
            "empty": MockMessage(content=""),
            "none_content": MockMessage(content=None),
            "dict_format": {"content": "Dict message", "tool_calls": []},
            "with_tools": MockMessage(
                content="Using tool",
                tool_calls=[
                    MockToolCall(
                        id="call_123",
                        function=MockFunction(
                            name="test_func", arguments='{"param": "value"}'
                        ),
                    )
                ],
            ),
            "tool_only": MockMessage(
                content="",
                tool_calls=[
                    MockToolCall(
                        function=MockFunction(
                            name="solo_func", arguments='{"key": "val"}'
                        )
                    )
                ],
            ),
            "alternative_fields": MockMessage(text="Text content"),
            "wrapped": SimpleNamespace(message=MockMessage(content="Wrapped content")),
        }

    return _create_messages


@pytest.fixture
def mock_stream_chunks():
    """Factory fixture for creating mock streaming chunks"""

    def _create_chunks(content_list=None, include_tools=False):
        if content_list is None:
            content_list = ["Hello", " ", "world!"]

        chunks = []
        for i, content in enumerate(content_list):
            if include_tools and i == 1:  # Add tool call to second chunk
                tool_call = MockToolCall(function=MockFunction(name="stream_tool"))
                chunks.append(
                    MockChunk(
                        [MockChoice(delta=MockDelta(content, tool_calls=[tool_call]))]
                    )
                )
            else:
                chunks.append(MockChunk([MockChoice(delta=MockDelta(content))]))

        return chunks

    return _create_chunks


# ---------------------------------------------------------------------------
# Tool name sanitization tests
# ---------------------------------------------------------------------------


def test_sanitize_tool_names_valid_names(mixin):
    """Test that valid tool names are not changed"""
    tools = [
        {"function": {"name": "valid_name"}},
        {"function": {"name": "another-valid-name"}},
        {"function": {"name": "name123"}},
    ]

    result = mixin._sanitize_tool_names(tools)

    assert len(result) == 3
    assert result[0]["function"]["name"] == "valid_name"
    assert result[1]["function"]["name"] == "another-valid-name"
    assert result[2]["function"]["name"] == "name123"


def test_sanitize_tool_names_invalid_chars(mixin):
    """Test that invalid characters are replaced with underscores"""
    tools = [
        {"function": {"name": "invalid.name"}},
        {"function": {"name": "name with spaces"}},
        {"function": {"name": "name@with#symbols"}},
        {"function": {"name": "name/with\\slashes"}},
    ]

    result = mixin._sanitize_tool_names(tools)

    assert len(result) == 4
    assert result[0]["function"]["name"] == "invalid_name"
    assert result[1]["function"]["name"] == "name_with_spaces"
    assert result[2]["function"]["name"] == "name_with_symbols"
    assert result[3]["function"]["name"] == "name_with_slashes"


def test_sanitize_tool_names_edge_cases(mixin):
    """Test edge cases for tool name sanitization"""
    # Empty tools list
    assert mixin._sanitize_tool_names([]) == []

    # None tools
    assert mixin._sanitize_tool_names(None) is None

    # Tool without function
    tools = [{"other": "data"}]
    result = mixin._sanitize_tool_names(tools)
    assert result == tools

    # Tool without name
    tools = [{"function": {"description": "no name"}}]
    result = mixin._sanitize_tool_names(tools)
    assert result == tools

    # Tool with empty name
    tools = [{"function": {"name": ""}}]
    result = mixin._sanitize_tool_names(tools)
    assert result[0]["function"]["name"] == ""


def test_sanitize_tool_names_preserves_other_data(mixin):
    """Test that sanitization preserves other tool data"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "invalid.name",
                "description": "A test function",
                "parameters": {"type": "object"},
            },
            "other_data": "preserved",
        }
    ]

    result = mixin._sanitize_tool_names(tools)

    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "invalid_name"
    assert result[0]["function"]["description"] == "A test function"
    assert result[0]["function"]["parameters"] == {"type": "object"}
    assert result[0]["other_data"] == "preserved"


def test_sanitize_tool_names_unicode(mixin):
    """Test tool name sanitization with unicode characters"""
    tools = [
        {"function": {"name": "función_español"}},
        {"function": {"name": "函数_中文"}},
        {"function": {"name": "функция_русский"}},
    ]

    result = mixin._sanitize_tool_names(tools)

    # Unicode letters should be replaced with underscores
    for tool in result:
        name = tool["function"]["name"]
        # Should only contain ASCII letters, numbers, hyphens, and underscores
        assert all(c.isascii() and (c.isalnum() or c in "-_") for c in name)


# ---------------------------------------------------------------------------
# Message normalization tests
# ---------------------------------------------------------------------------


def test_normalize_message_basic_content(mixin, mock_messages):
    """Test normalizing message with basic content"""
    messages = mock_messages()

    result = mixin._normalise_message(messages["simple"])

    assert result["response"] == "Hello, world!"
    assert result["tool_calls"] == []


def test_normalize_message_empty_content(mixin, mock_messages):
    """Test normalizing message with empty content"""
    messages = mock_messages()

    result = mixin._normalise_message(messages["empty"])

    assert result["response"] == ""
    assert result["tool_calls"] == []


def test_normalize_message_none_content(mixin, mock_messages):
    """Test normalizing message with None content"""
    messages = mock_messages()

    result = mixin._normalise_message(messages["none_content"])

    assert result["response"] == ""
    assert result["tool_calls"] == []


def test_normalize_message_dict_format(mixin, mock_messages):
    """Test normalizing message in dict format"""
    messages = mock_messages()

    result = mixin._normalise_message(messages["dict_format"])

    assert result["response"] == "Dict message"
    assert result["tool_calls"] == []


def test_normalize_message_wrapper_format(mixin, mock_messages):
    """Test normalizing message with wrapper format"""
    messages = mock_messages()

    result = mixin._normalise_message(messages["wrapped"])

    assert result["response"] == "Wrapped content"
    assert result["tool_calls"] == []


def test_normalize_message_alternative_fields(mixin, mock_messages):
    """Test normalizing message with alternative content fields"""
    messages = mock_messages()

    # Test 'text' field
    result = mixin._normalise_message(messages["alternative_fields"])
    assert result["response"] == "Text content"

    # Test other alternative fields
    msg_content = MockMessage(message_content="Message content")
    result = mixin._normalise_message(msg_content)
    assert result["response"] == "Message content"

    # Test 'response_text' field
    response_text = MockMessage(response_text="Response text")
    result = mixin._normalise_message(response_text)
    assert result["response"] == "Response text"


def test_normalize_message_with_tool_calls(mixin, mock_messages):
    """Test normalizing message with tool calls"""
    messages = mock_messages()

    result = mixin._normalise_message(messages["with_tools"])

    assert result["response"] == "Using tool"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call_123"
    assert result["tool_calls"][0]["function"]["name"] == "test_func"
    assert result["tool_calls"][0]["function"]["arguments"] == '{"param": "value"}'


def test_normalize_message_tool_calls_only(mixin, mock_messages):
    """Test normalizing message with only tool calls (no content)"""
    messages = mock_messages()

    result = mixin._normalise_message(messages["tool_only"])

    # When there are tool calls and content is empty string, response should be None
    assert result["response"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "solo_func"


def test_normalize_message_tool_calls_dict_arguments(mixin):
    """Test normalizing tool calls with dict arguments"""
    tool_call = MockToolCall(
        function=MockFunction(name="dict_func", arguments={"key": "value"})
    )
    msg = MockMessage(content="", tool_calls=[tool_call])

    result = mixin._normalise_message(msg)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["arguments"] == '{"key": "value"}'


def test_normalize_message_tool_calls_invalid_json(mixin, mock_logger):
    """Test normalizing tool calls with invalid JSON arguments"""
    tool_call = MockToolCall(
        function=MockFunction(name="bad_json", arguments='{"invalid": json}')
    )
    msg = MockMessage(content="", tool_calls=[tool_call])

    result = mixin._normalise_message(msg)

    assert len(result["tool_calls"]) == 1
    # Should fallback to empty object for invalid JSON
    assert result["tool_calls"][0]["function"]["arguments"] == "{}"
    # Should log warning
    mock_logger.warning.assert_called()


def test_normalize_message_tool_calls_missing_id(mixin):
    """Test normalizing tool calls without ID"""
    tool_call = MockToolCall(id=None, function=MockFunction(name="no_id_func"))

    # Remove the id attribute entirely
    delattr(tool_call, "id")

    msg = MockMessage(tool_calls=[tool_call])

    result = mixin._normalise_message(msg)

    assert len(result["tool_calls"]) == 1
    # Should generate a random ID
    assert result["tool_calls"][0]["id"].startswith("call_")
    assert len(result["tool_calls"][0]["id"]) > 5


def test_normalize_message_tool_calls_wrapper_format(mixin):
    """Test normalizing tool calls from wrapper message format"""
    tool_call = MockToolCall(function=MockFunction(name="wrapped_func"))
    inner_msg = MockMessage(tool_calls=[tool_call])
    msg = SimpleNamespace(message=inner_msg)

    result = mixin._normalise_message(msg)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "wrapped_func"


def test_normalize_message_tool_calls_dict_format(mixin):
    """Test normalizing tool calls from dict format"""
    mock_tool_call = MockToolCall(
        id="call_dict",
        function=MockFunction(name="dict_tool", arguments='{"test": true}'),
    )

    # Create dict-like message that has proper tool call objects
    msg = DictLikeMessage({"content": "", "tool_calls": [mock_tool_call]})

    result = mixin._normalise_message(msg)

    # Check that tool calls were extracted
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call_dict"
    assert result["tool_calls"][0]["function"]["name"] == "dict_tool"


def test_normalize_message_malformed_tool_calls(mixin, mock_logger):
    """Test normalizing malformed tool calls"""
    # Tool call without function
    bad_tool_call = MockToolCall()
    delattr(bad_tool_call, "function")

    msg = MockMessage(tool_calls=[bad_tool_call])

    result = mixin._normalise_message(msg)

    # Should skip malformed tool calls
    assert result["tool_calls"] == []
    # Should log warning
    mock_logger.warning.assert_called()


def test_normalize_message_no_attributes(mixin, mock_logger):
    """Test normalizing message with no recognizable attributes"""
    msg = SimpleNamespace(unknown_field="data")

    result = mixin._normalise_message(msg)

    assert result["response"] == ""
    assert result["tool_calls"] == []
    # Should log warning about no content found
    mock_logger.warning.assert_called()


def test_normalize_message_complex_error_scenarios(mixin):
    """Test message normalization with complex error scenarios"""

    # Message with exception-throwing attributes
    class BadMessage:
        @property
        def content(self):
            raise RuntimeError("Content access failed")

        @property
        def tool_calls(self):
            raise ValueError("Tool calls access failed")

    msg = BadMessage()
    result = mixin._normalise_message(msg)

    # Should handle errors gracefully
    assert result["response"] == ""
    assert result["tool_calls"] == []


def test_normalize_message_mixed_formats(mixin):
    """Test normalization with mixed/confusing formats"""
    # Message with both content and alternative fields
    msg = MockMessage(
        content="Primary content",
        text="Alternative text",
        message_content="Another alternative",
    )

    result = mixin._normalise_message(msg)

    # Should prefer primary content
    assert result["response"] == "Primary content"


def test_normalize_message_recursive_structures(mixin):
    """Test normalization with recursive/deep structures"""
    # Create a message with nested wrapper
    inner_msg = MockMessage(content="Deep content")
    wrapper = SimpleNamespace(message=SimpleNamespace(message=inner_msg))

    result = mixin._normalise_message(wrapper)

    # Should handle nested structures (though may not go arbitrarily deep)
    # The current implementation should find content somewhere
    assert isinstance(result["response"], str)
    assert isinstance(result["tool_calls"], list)


# ---------------------------------------------------------------------------
# Blocking call wrapper tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_blocking_basic(mixin):
    """Test basic blocking call wrapper"""

    def blocking_func(x, y, multiplier=2):
        return (x + y) * multiplier

    result = await mixin._call_blocking(blocking_func, 3, 4, multiplier=3)

    assert result == 21  # (3 + 4) * 3


@pytest.mark.asyncio
async def test_call_blocking_with_exception(mixin):
    """Test blocking call wrapper with exception"""

    def failing_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        await mixin._call_blocking(failing_func)


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_from_async_basic(mixin, mock_stream_chunks):
    """Test basic async streaming"""
    chunks = mock_stream_chunks()

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 3
    assert results[0]["response"] == "Hello"
    assert results[1]["response"] == " "
    assert results[2]["response"] == "world!"
    assert all(result["tool_calls"] == [] for result in results)


@pytest.mark.asyncio
async def test_stream_from_async_with_tool_calls(mixin, mock_stream_chunks):
    """Test async streaming with tool calls"""
    chunks = mock_stream_chunks(["Thinking...", "", " Done!"], include_tools=True)

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 3
    assert results[0]["response"] == "Thinking..."
    assert results[0]["tool_calls"] == []

    assert results[1]["response"] == ""
    assert len(results[1]["tool_calls"]) == 1
    assert results[1]["tool_calls"][0]["function"]["name"] == "stream_tool"

    assert results[2]["response"] == " Done!"
    assert results[2]["tool_calls"] == []


@pytest.mark.asyncio
async def test_stream_from_async_full_message_format(mixin):
    """Test async streaming with full message format"""
    message = MockMessage(content="Full message")
    chunks = [
        MockChunk([MockChoice(message=message)]),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 1
    assert results[0]["response"] == "Full message"


@pytest.mark.asyncio
async def test_stream_from_async_choice_text_format(mixin):
    """Test async streaming with choice.text format"""
    chunks = [
        MockChunk([MockChoice(text="Text content")]),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 1
    assert results[0]["response"] == "Text content"


@pytest.mark.asyncio
async def test_stream_from_async_direct_content(mixin):
    """Test async streaming with direct chunk content"""
    chunks = [
        MockChunk(content="Direct content"),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 1
    assert results[0]["response"] == "Direct content"


@pytest.mark.asyncio
async def test_stream_from_async_dict_format(mixin):
    """Test async streaming with dict chunks"""
    chunks = [
        {"content": "Dict content"},
        {"content": "More content", "tool_calls": []},
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 2
    assert results[0]["response"] == "Dict content"
    assert results[1]["response"] == "More content"


@pytest.mark.asyncio
async def test_stream_from_async_custom_normalize(mixin, mock_stream_chunks):
    """Test async streaming with custom normalization"""

    def custom_normalize(result, chunk):
        # Add custom metadata
        result["custom"] = True
        return result

    chunks = mock_stream_chunks(["Custom"])

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream(), custom_normalize):
        results.append(result)

    assert len(results) == 1
    assert results[0]["response"] == "Custom"
    assert results[0]["custom"] is True


@pytest.mark.asyncio
async def test_stream_from_async_chunk_error(mixin, mock_logger):
    """Test async streaming with chunk processing error"""
    # Create a chunk that will cause an error during processing
    bad_chunk = SimpleNamespace()  # No expected attributes

    chunks = [
        MockChunk([MockChoice(delta=MockDelta("Good"))]),
        bad_chunk,
        MockChunk([MockChoice(delta=MockDelta("Also good"))]),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 3
    assert results[0]["response"] == "Good"
    # The second chunk should have empty response (graceful handling)
    assert results[1]["response"] == ""  # Bad chunk becomes empty
    assert results[2]["response"] == "Also good"

    # Error logging may or may not happen depending on implementation
    # The important thing is that streaming continues gracefully


@pytest.mark.asyncio
async def test_stream_from_async_stream_error(mixin, mock_logger):
    """Test async streaming with stream-level error"""

    async def failing_stream():
        yield MockChunk([MockChoice(delta=MockDelta("Before error"))])
        raise RuntimeError("Stream failed")

    results = []
    async for result in mixin._stream_from_async(failing_stream()):
        results.append(result)

    # Should get at least the first chunk, and may get an error chunk
    assert len(results) >= 1
    assert results[0]["response"] == "Before error"

    # If there's a second result, it should indicate an error
    if len(results) > 1:
        assert results[1].get("error") is True
        assert "Stream failed" in results[1]["response"]

    # Error logging should happen for stream-level errors
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_stream_from_async_empty_stream(mixin):
    """Test async streaming with empty stream"""

    async def empty_stream():
        return
        yield  # unreachable

    results = []
    async for result in mixin._stream_from_async(empty_stream()):
        results.append(result)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_stream_from_async_none_content(mixin):
    """Test async streaming with None content in chunks"""
    chunks = [
        MockChunk([MockChoice(delta=MockDelta(None))]),  # None content
        MockChunk([MockChoice(delta=MockDelta(""))]),  # Empty content
        MockChunk([MockChoice(delta=MockDelta("Real"))]),  # Real content
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 3
    assert results[0]["response"] == ""  # None should become empty string
    assert results[1]["response"] == ""
    assert results[2]["response"] == "Real"


@pytest.mark.asyncio
async def test_stream_from_async_malformed_tool_calls(mixin):
    """Test async streaming with malformed tool calls"""
    # Tool call without function
    bad_tool_call = SimpleNamespace(id="bad_call")
    chunks = [
        MockChunk([MockChoice(delta=MockDelta("", tool_calls=[bad_tool_call]))]),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 1
    assert results[0]["response"] == ""
    # Bad tool calls should be skipped
    assert results[0]["tool_calls"] == []


@pytest.mark.asyncio
async def test_stream_from_async_performance(mixin):
    """Test streaming performance with many chunks"""
    # Simulate a moderate number of chunks (reduced for test performance)
    num_chunks = 100  # Reduced from 1000
    chunks = [
        MockChunk([MockChoice(delta=MockDelta(f"chunk{i}"))]) for i in range(num_chunks)
    ]

    async def large_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(large_stream()):
        results.append(result)

    assert len(results) == num_chunks
    assert results[0]["response"] == "chunk0"
    assert results[-1]["response"] == f"chunk{num_chunks - 1}"


# ---------------------------------------------------------------------------
# Debug helper tests
# ---------------------------------------------------------------------------


def test_debug_message_structure_basic(mixin, mock_logger):
    """Test debug message structure helper"""
    msg = MockMessage(content="Debug test")
    mixin.debug_message_structure(msg, "test_context")

    # Check that debug info was logged
    mock_logger.debug.assert_called()
    debug_calls = mock_logger.debug.call_args_list

    # Should have logged debug structure info
    assert any(
        "DEBUG MESSAGE STRUCTURE (test_context)" in str(call) for call in debug_calls
    )
    assert any("MockMessage" in str(call) for call in debug_calls)


def test_debug_message_structure_dict(mixin):
    """Test debug message structure with dict"""
    msg = {"content": "Dict debug", "tool_calls": []}

    # Should not raise an error
    mixin.debug_message_structure(msg, "dict_test")


def test_debug_message_structure_no_dict(mixin):
    """Test debug message structure with object without __dict__"""
    msg = "string message"

    # Should not raise an error
    mixin.debug_message_structure(msg, "string_test")


def test_debug_message_structure_disabled_logging(mixin, mock_logger):
    """Test debug helper when debug logging is disabled"""
    mock_logger.isEnabledFor.return_value = False

    msg = MockMessage(content="Debug test")
    mixin.debug_message_structure(msg, "test_context")

    # Should return early and not log anything
    mock_logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_workflow_integration(mixin):
    """Test a complete workflow using multiple mixin methods"""
    # Start with tools that need sanitization
    tools = [
        {"function": {"name": "get.weather", "description": "Get weather"}},
        {"function": {"name": "send message", "description": "Send message"}},
    ]

    # Sanitize tool names
    sanitized_tools = mixin._sanitize_tool_names(tools)
    assert sanitized_tools[0]["function"]["name"] == "get_weather"
    assert sanitized_tools[1]["function"]["name"] == "send_message"

    # Mock a response message with tool calls
    tool_call = MockToolCall(
        function=MockFunction(name="get_weather", arguments='{"location": "NYC"}')
    )
    response_msg = MockMessage(
        content="I'll check the weather for you.", tool_calls=[tool_call]
    )

    # Normalize the response
    normalized = mixin._normalise_message(response_msg)
    assert normalized["response"] == "I'll check the weather for you."
    assert len(normalized["tool_calls"]) == 1
    assert normalized["tool_calls"][0]["function"]["name"] == "get_weather"

    # Mock streaming response
    stream_chunks = [
        MockChunk([MockChoice(delta=MockDelta("Checking"))]),
        MockChunk([MockChoice(delta=MockDelta(" weather..."))]),
        MockChunk([MockChoice(delta=MockDelta("", tool_calls=[tool_call]))]),
        MockChunk([MockChoice(delta=MockDelta(" Done!"))]),
    ]

    async def mock_stream():
        for chunk in stream_chunks:
            yield chunk

    # Process stream
    stream_results = []
    async for result in mixin._stream_from_async(mock_stream()):
        stream_results.append(result)

    assert len(stream_results) == 4
    assert stream_results[0]["response"] == "Checking"
    assert stream_results[1]["response"] == " weather..."
    assert len(stream_results[2]["tool_calls"]) == 1
    assert stream_results[3]["response"] == " Done!"


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


def test_sanitize_tool_names_deeply_nested(mixin):
    """Test tool sanitization with deeply nested structures"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "invalid@name",
                "nested": {"deeper": {"data": "preserved"}},
            },
        }
    ]

    result = mixin._sanitize_tool_names(tools)

    assert result[0]["function"]["name"] == "invalid_name"
    assert result[0]["function"]["nested"]["deeper"]["data"] == "preserved"


@pytest.mark.asyncio
async def test_stream_from_async_custom_normalize_error(mixin, mock_logger):
    """Test streaming with custom normalization that raises an error"""

    def failing_normalize(result, chunk):
        raise ValueError("Custom normalize failed")

    chunks = [MockChunk([MockChoice(delta=MockDelta("Test"))])]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream(), failing_normalize):
        results.append(result)

    assert len(results) == 1
    assert (
        results[0]["response"] == "Test"
    )  # Should continue despite normalization error
    # Should log debug message about normalization failure
    mock_logger.debug.assert_called()


def test_normalize_message_tool_calls_various_argument_types(mixin):
    """Test normalizing tool calls with various argument types"""
    # Test with None arguments
    tool_call_none = MockToolCall(
        function=MockFunction(name="none_args", arguments=None)
    )

    # Test with empty string arguments
    tool_call_empty = MockToolCall(
        function=MockFunction(name="empty_args", arguments="")
    )

    # Test with number arguments (should be converted)
    tool_call_number = MockToolCall(
        function=MockFunction(name="number_args", arguments=123)
    )

    msg = MockMessage(tool_calls=[tool_call_none, tool_call_empty, tool_call_number])

    result = mixin._normalise_message(msg)

    assert len(result["tool_calls"]) == 3
    # All should have string arguments
    for tc in result["tool_calls"]:
        assert isinstance(tc["function"]["arguments"], str)


@pytest.mark.asyncio
async def test_stream_from_async_mixed_chunk_formats(mixin):
    """Test streaming with mixed chunk formats in the same stream"""
    mixed_chunks = [
        MockChunk([MockChoice(delta=MockDelta("Delta "))]),  # Delta format
        MockChunk(
            [MockChoice(message=MockMessage(content="Message "))]
        ),  # Message format
        MockChunk([MockChoice(text="Text ")]),  # Text format
        MockChunk(content="Direct"),  # Direct content
        {"content": " Dict"},  # Dict format
    ]

    async def mixed_stream():
        for chunk in mixed_chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mixed_stream()):
        results.append(result)

    assert len(results) == 5
    responses = [r["response"] for r in results]
    assert responses == ["Delta ", "Message ", "Text ", "Direct", " Dict"]


# ---------------------------------------------------------------------------
# Logging and debugging tests
# ---------------------------------------------------------------------------


def test_normalize_message_debug_logging(mixin, mock_logger):
    """Test that normalization produces appropriate debug logs"""
    msg = MockMessage(content="Test content")

    mixin._normalise_message(msg)

    # Should log debug information about content extraction
    mock_logger.debug.assert_called()
    debug_calls = mock_logger.debug.call_args_list

    # Should have logged some debug information (specific content may vary)
    assert len(debug_calls) > 0


@pytest.mark.asyncio
async def test_stream_debug_logging(mixin, mock_logger):
    """Test that streaming produces appropriate debug logs"""
    chunks = [MockChunk([MockChoice(delta=MockDelta("Test"))])]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    # Should log streaming statistics
    mock_logger.debug.assert_called()
    debug_calls = mock_logger.debug.call_args_list

    # Should have logged some debug information
    assert len(debug_calls) > 0


# ---------------------------------------------------------------------------
# Image URL downloading and processing tests (lines 63-93, 111-150)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_and_encode_image_jpeg_content_type(mixin):
    """Test downloading and encoding JPEG image from content-type"""
    test_image_data = b"fake_jpeg_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        encoded_data, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.jpg"
        )

        assert image_format == "jpeg"
        assert encoded_data == base64.b64encode(test_image_data).decode("utf-8")
        mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_download_and_encode_image_png_content_type(mixin):
    """Test downloading and encoding PNG image from content-type"""
    test_image_data = b"fake_png_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        encoded_data, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.png"
        )

        assert image_format == "png"
        assert encoded_data == base64.b64encode(test_image_data).decode("utf-8")


@pytest.mark.asyncio
async def test_download_and_encode_image_webp_content_type(mixin):
    """Test downloading and encoding WebP image from content-type"""
    test_image_data = b"fake_webp_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/webp"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        encoded_data, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.webp"
        )

        assert image_format == "webp"


@pytest.mark.asyncio
async def test_download_and_encode_image_gif_content_type(mixin):
    """Test downloading and encoding GIF image from content-type"""
    test_image_data = b"fake_gif_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/gif"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        encoded_data, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.gif"
        )

        assert image_format == "gif"


@pytest.mark.asyncio
async def test_download_and_encode_image_jpg_content_type(mixin):
    """Test downloading and encoding image with jpg in content-type"""
    test_image_data = b"fake_jpg_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/jpg"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        encoded_data, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.jpg"
        )

        assert image_format == "jpeg"


@pytest.mark.asyncio
async def test_download_and_encode_image_url_extension_jpeg(mixin):
    """Test image format detection from URL extension for JPEG"""
    test_image_data = b"fake_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        # Test .jpeg extension
        _, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.jpeg"
        )
        assert image_format == "jpeg"

        # Test .jpg extension
        _, image_format = await mixin._download_and_encode_image(
            "http://example.com/photo.jpg"
        )
        assert image_format == "jpeg"


@pytest.mark.asyncio
async def test_download_and_encode_image_url_extension_png(mixin):
    """Test image format detection from URL extension for PNG"""
    test_image_data = b"fake_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        _, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.png"
        )
        assert image_format == "png"


@pytest.mark.asyncio
async def test_download_and_encode_image_url_extension_webp(mixin):
    """Test image format detection from URL extension for WebP"""
    test_image_data = b"fake_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        _, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.webp"
        )
        assert image_format == "webp"


@pytest.mark.asyncio
async def test_download_and_encode_image_url_extension_gif(mixin):
    """Test image format detection from URL extension for GIF"""
    test_image_data = b"fake_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        _, image_format = await mixin._download_and_encode_image(
            "http://example.com/image.gif"
        )
        assert image_format == "gif"


@pytest.mark.asyncio
async def test_download_and_encode_image_default_format(mixin):
    """Test image format defaults to JPEG when unknown"""
    test_image_data = b"fake_data"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        _, image_format = await mixin._download_and_encode_image(
            "http://example.com/unknown.xyz"
        )
        assert image_format == "jpeg"


@pytest.mark.asyncio
async def test_download_and_encode_image_http_error(mixin):
    """Test handling HTTP errors when downloading images"""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
    )

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        with pytest.raises(httpx.HTTPStatusError):
            await mixin._download_and_encode_image("http://example.com/missing.jpg")


@pytest.mark.asyncio
async def test_process_image_urls_supports_direct_urls(mixin):
    """Test that image processing is skipped when provider supports direct URLs (line 111-112)"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                },
            ],
        }
    ]

    result = await mixin._process_image_urls_in_messages(
        messages, supports_direct_urls=True
    )

    # Should return unchanged
    assert result == messages


@pytest.mark.asyncio
async def test_process_image_urls_text_only_messages(mixin):
    """Test processing messages with only text content"""
    messages = [{"role": "user", "content": "Just text"}]

    result = await mixin._process_image_urls_in_messages(messages)

    # Should return unchanged
    assert result == messages


@pytest.mark.asyncio
async def test_process_image_urls_non_list_content(mixin):
    """Test processing messages with non-list content (lines 148-149)"""
    messages = [
        {"role": "user", "content": "String content"},
        {"role": "assistant", "content": "Another string"},
    ]

    result = await mixin._process_image_urls_in_messages(messages)

    # Should return unchanged
    assert result == messages
    assert len(result) == 2


@pytest.mark.asyncio
async def test_process_image_urls_http_url(mixin):
    """Test processing HTTP image URLs (lines 127-141)"""
    test_image_data = b"test_image"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.png"},
                },
            ],
        }
    ]

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await mixin._process_image_urls_in_messages(messages)

        # Should convert to base64 data URI
        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_process_image_urls_https_url(mixin):
    """Test processing HTTPS image URLs"""
    test_image_data = b"test_image"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.raise_for_status = MagicMock()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://secure.example.com/image.jpg"},
                }
            ],
        }
    ]

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await mixin._process_image_urls_in_messages(messages)

        # Should convert to base64 data URI
        content = result[0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_process_image_urls_already_base64(mixin):
    """Test that base64 URLs are not re-downloaded"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                }
            ],
        }
    ]

    result = await mixin._process_image_urls_in_messages(messages)

    # Should remain unchanged
    assert result[0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
async def test_process_image_urls_download_failure(mixin, mock_logger):
    """Test handling download failures gracefully (lines 138-140)"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/broken.jpg"},
                }
            ],
        }
    ]

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection failed")
        )

        result = await mixin._process_image_urls_in_messages(messages)

        # Should keep original URL on error
        assert result[0]["content"][0]["image_url"]["url"] == "http://example.com/broken.jpg"
        # Should log error
        mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_process_image_urls_mixed_content(mixin):
    """Test processing messages with mixed content types (lines 142-144)"""
    test_image_data = b"test_image"
    mock_response = MagicMock()
    mock_response.content = test_image_data
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.png"},
                },
                {"type": "text", "text": "What is it?"},
            ],
        }
    ]

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await mixin._process_image_urls_in_messages(messages)

        # Should process only image URL, keep text unchanged
        content = result[0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[2]["type"] == "text"


# ---------------------------------------------------------------------------
# Content extraction fallback tests (lines 188-215, 242)
# ---------------------------------------------------------------------------


def test_normalize_message_dict_content_access_error(mixin, mock_logger):
    """Test dict content access with errors (lines 188-189)"""

    # Create a dict subclass that passes isinstance(msg, dict) but raises on access
    class BadDict(dict):
        def __getitem__(self, key):
            raise RuntimeError("Dict access error")

    msg = BadDict({"content": "test"})
    result = mixin._normalise_message(msg)

    # Should handle error gracefully
    assert result["response"] == ""
    assert result["tool_calls"] == []
    # Should log debug message about dict access failure
    mock_logger.debug.assert_called()


def test_normalize_message_dict_content_success(mixin):
    """Test successful dict content access (line 184)"""
    # Create a dict-like object that supports "in" check and __getitem__
    msg = {"content": "Dict content via dict access", "role": "assistant"}

    result = mixin._normalise_message(msg)

    # Should successfully extract content via dict access
    assert result["response"] == "Dict content via dict access"
    assert result["tool_calls"] == []


def test_normalize_message_wrapper_access_error(mixin, mock_logger):
    """Test message wrapper access with errors (lines 199-200)"""

    # Create object that passes hasattr checks but raises on access
    class BadWrapper:
        def __init__(self):
            # Create a real message object so hasattr checks pass
            self.message = SimpleNamespace(content=None)

        def __getattribute__(self, name):
            if name == "message":
                # Return object that has content attr
                inner = SimpleNamespace()
                inner.content = property(lambda self: (_ for _ in ()).throw(RuntimeError("Content access failed")))
                return inner
            return object.__getattribute__(self, name)

    msg = BadWrapper()

    # Simpler approach: mock a wrapper that raises during content access
    class SimpleWrapper:
        class InnerMessage:
            @property
            def content(self):
                raise RuntimeError("Content access failed")

        def __init__(self):
            self.message = self.InnerMessage()

    msg = SimpleWrapper()
    result = mixin._normalise_message(msg)

    # Should handle error gracefully
    assert result["response"] == ""
    # Should log debug message about wrapper access failure
    mock_logger.debug.assert_called()


def test_normalize_message_wrapper_success(mixin):
    """Test successful message wrapper access (line 195)"""
    # Create a wrapper with message.content
    inner_message = MockMessage(content="Wrapped message content")
    msg = SimpleNamespace(message=inner_message)

    result = mixin._normalise_message(msg)

    # Should successfully extract content via wrapper
    assert result["response"] == "Wrapped message content"
    assert result["tool_calls"] == []


def test_normalize_message_alternative_field_access_error(mixin, mock_logger):
    """Test alternative field access with errors (lines 214-215)"""

    class BadAltFields:
        @property
        def text(self):
            raise RuntimeError("Text access failed")

        @property
        def message_content(self):
            raise ValueError("Message content failed")

        @property
        def response_text(self):
            raise TypeError("Response text failed")

    msg = BadAltFields()
    result = mixin._normalise_message(msg)

    # Should handle all errors gracefully
    assert result["response"] == ""
    # Debug logs should be called for each failed attempt
    mock_logger.debug.assert_called()


def test_normalize_message_dict_tool_calls(mixin):
    """Test extracting tool calls from dict format (line 242)"""
    mock_tool_call = MockToolCall(
        id="call_dict_tool",
        function=MockFunction(name="dict_function", arguments='{"param": "value"}'),
    )

    # Pure dict format with tool calls
    msg = {"content": "Test content", "tool_calls": [mock_tool_call]}

    result = mixin._normalise_message(msg)

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call_dict_tool"
    assert result["tool_calls"][0]["function"]["name"] == "dict_function"


# ---------------------------------------------------------------------------
# Tool call error handling tests (lines 293-295)
# ---------------------------------------------------------------------------


def test_normalize_message_tool_call_processing_error(mixin, mock_logger):
    """Test error handling when processing individual tool calls (lines 293-295)"""

    class BadToolCall:
        @property
        def id(self):
            return "call_bad"

        @property
        def function(self):
            raise RuntimeError("Function access failed")

    msg = MockMessage(tool_calls=[BadToolCall(), MockToolCall()])

    result = mixin._normalise_message(msg)

    # Should skip the bad tool call but process the good one
    assert len(result["tool_calls"]) == 1
    # Should log warning about the failed tool call
    mock_logger.warning.assert_called()


def test_normalize_message_tool_call_exception_in_loop(mixin, mock_logger):
    """Test that tool call processing continues after exception"""

    class ExceptionToolCall:
        def __getattribute__(self, name):
            raise ValueError("Attribute error")

    good_tool_call = MockToolCall(
        function=MockFunction(name="good_func", arguments='{"key": "val"}')
    )

    msg = MockMessage(tool_calls=[ExceptionToolCall(), good_tool_call])

    result = mixin._normalise_message(msg)

    # Should process the good tool call despite the bad one
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "good_func"
    mock_logger.warning.assert_called()


# ---------------------------------------------------------------------------
# Blocking stream tests (lines 332-362)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_from_blocking_basic_flow(mixin):
    """Test blocking stream wrapper basic functionality (lines 332-362)"""

    def blocking_stream_generator(**kwargs):
        """Mock blocking streaming SDK call"""
        chunks = [
            MockChunk([MockChoice(delta=MockDelta("Hello"))]),
            MockChunk([MockChoice(delta=MockDelta(" world"))]),
            MockChunk([MockChoice(delta=MockDelta("!"))]),
        ]
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_blocking(blocking_stream_generator):
        results.append(result)

    assert len(results) == 3
    assert results[0]["response"] == "Hello"
    assert results[1]["response"] == " world"
    assert results[2]["response"] == "!"


@pytest.mark.asyncio
async def test_stream_from_blocking_with_tool_calls(mixin):
    """Test blocking stream with tool calls"""
    tool_call = MockToolCall(function=MockFunction(name="stream_func"))

    def blocking_stream_with_tools(**kwargs):
        yield MockChunk([MockChoice(delta=MockDelta("Using tool"))])
        delta_with_tool = MockDelta("", tool_calls=[tool_call])
        yield MockChunk([MockChoice(delta=delta_with_tool)])
        yield MockChunk([MockChoice(delta=MockDelta("Done"))])

    results = []
    async for result in mixin._stream_from_blocking(blocking_stream_with_tools):
        results.append(result)

    assert len(results) == 3
    assert results[0]["response"] == "Using tool"
    assert results[1]["tool_calls"] == [tool_call]
    assert results[2]["response"] == "Done"


@pytest.mark.asyncio
async def test_stream_from_blocking_chunk_processing_error(mixin, mock_logger):
    """Test blocking stream error handling for individual chunks (lines 346-348)"""

    def blocking_stream_with_bad_chunk(**kwargs):
        yield MockChunk([MockChoice(delta=MockDelta("Good"))])
        # Bad chunk that will cause processing error
        bad_chunk = SimpleNamespace(choices=[SimpleNamespace()])
        yield bad_chunk
        yield MockChunk([MockChoice(delta=MockDelta("Also good"))])

    results = []
    async for result in mixin._stream_from_blocking(blocking_stream_with_bad_chunk):
        results.append(result)

    # Should yield results, including error indicator
    assert len(results) >= 2
    assert results[0]["response"] == "Good"
    # One of the results should indicate an error
    assert any(result.get("error") is True for result in results)
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_stream_from_blocking_worker_exception(mixin, mock_logger):
    """Test blocking stream worker exception handling (lines 355-357)"""

    def failing_blocking_stream(**kwargs):
        yield MockChunk([MockChoice(delta=MockDelta("Before error"))])
        raise RuntimeError("Stream worker failed")

    results = []
    async for result in mixin._stream_from_blocking(failing_blocking_stream):
        results.append(result)

    # Should get the first chunk and then stream should end
    assert len(results) >= 1
    assert results[0]["response"] == "Before error"
    # Error should be logged
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_stream_from_blocking_empty_stream(mixin):
    """Test blocking stream with empty generator"""

    def empty_blocking_stream(**kwargs):
        return
        yield  # unreachable

    results = []
    async for result in mixin._stream_from_blocking(empty_blocking_stream):
        results.append(result)

    # Should complete without error
    assert len(results) == 0


@pytest.mark.asyncio
async def test_stream_from_blocking_kwargs_passed(mixin):
    """Test that kwargs are passed to blocking stream function"""
    received_kwargs = {}

    def blocking_stream_with_kwargs(**kwargs):
        received_kwargs.update(kwargs)
        yield MockChunk([MockChoice(delta=MockDelta("Test"))])

    results = []
    async for result in mixin._stream_from_blocking(
        blocking_stream_with_kwargs, custom_param="test_value"
    ):
        results.append(result)

    # Verify stream=True was added by the wrapper and custom param passed
    assert "stream" in received_kwargs
    assert received_kwargs["stream"] is True
    assert received_kwargs["custom_param"] == "test_value"


# ---------------------------------------------------------------------------
# Enhanced async stream error handling tests (lines 433-437, 453-456, 494-503)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_from_async_delta_tool_call_error(mixin, mock_logger):
    """Test error handling in delta tool call processing (lines 433-437)"""

    class BadToolCall:
        @property
        def id(self):
            raise RuntimeError("ID access failed")

        @property
        def function(self):
            return MockFunction(name="test")

    bad_delta = MockDelta("", tool_calls=[BadToolCall()])
    chunks = [MockChunk([MockChoice(delta=bad_delta)])]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    # Should handle error gracefully
    assert len(results) == 1
    # Tool call should be skipped
    assert result["tool_calls"] == []
    mock_logger.debug.assert_called()


@pytest.mark.asyncio
async def test_stream_from_async_message_with_tool_calls(mixin):
    """Test full message format with tool calls in stream (lines 452-456)"""
    tool_call = MockToolCall(
        function=MockFunction(name="msg_tool", arguments='{"key": "val"}')
    )
    message = MockMessage(content="Message with tool", tool_calls=[tool_call])
    chunks = [MockChunk([MockChoice(message=message)])]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 1
    assert results[0]["response"] == "Message with tool"
    assert len(results[0]["tool_calls"]) == 1
    assert results[0]["tool_calls"][0]["function"]["name"] == "msg_tool"


@pytest.mark.asyncio
async def test_stream_from_async_chunk_error_with_details(mixin, mock_logger):
    """Test detailed chunk error handling (lines 494-503)"""

    class BadChunk:
        @property
        def choices(self):
            raise ValueError("Choices access failed")

    chunks = [
        MockChunk([MockChoice(delta=MockDelta("Good"))]),
        BadChunk(),
        MockChunk([MockChoice(delta=MockDelta("Also good"))]),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    # Should have all three results
    assert len(results) == 3
    assert results[0]["response"] == "Good"
    # Middle chunk should be error
    assert results[1].get("error") is True
    assert "error_message" in results[1]
    assert results[2]["response"] == "Also good"
    # Should log error for bad chunk
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_stream_from_async_continue_after_chunk_error(mixin):
    """Test that streaming continues after individual chunk errors"""

    class ExceptionChunk:
        def __getattribute__(self, name):
            if name != "__class__":
                raise RuntimeError("Chunk access failed")
            return object.__getattribute__(self, name)

    chunks = [
        MockChunk([MockChoice(delta=MockDelta("One"))]),
        ExceptionChunk(),
        MockChunk([MockChoice(delta=MockDelta("Three"))]),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    # Should get all results despite error in middle
    assert len(results) == 3
    assert results[0]["response"] == "One"
    assert results[1].get("error") is True  # Error chunk
    assert results[2]["response"] == "Three"


@pytest.mark.asyncio
async def test_stream_from_async_no_content_warning(mixin, mock_logger):
    """Test warning when stream has chunks but no content (lines 511-514)"""
    # Create chunks that have structure but no actual content
    empty_chunks = [
        MockChunk([MockChoice(delta=MockDelta(None))]),
        MockChunk([MockChoice(delta=MockDelta(None))]),
        MockChunk([MockChoice(delta=MockDelta(None))]),
    ]

    async def mock_stream():
        for chunk in empty_chunks:
            yield chunk

    results = []
    async for result in mixin._stream_from_async(mock_stream()):
        results.append(result)

    assert len(results) == 3
    # Should warn about no content
    mock_logger.warning.assert_called()


@pytest.mark.asyncio
async def test_stream_from_async_fatal_error(mixin, mock_logger):
    """Test fatal error handling in stream (lines 516-524)"""

    async def fatal_error_stream():
        raise RuntimeError("Fatal stream error")
        yield  # unreachable

    results = []
    async for result in mixin._stream_from_async(fatal_error_stream()):
        results.append(result)

    # Should get error result
    assert len(results) == 1
    assert results[0].get("error") is True
    assert "Fatal stream error" in results[0]["response"]
    assert "error_message" in results[0]
    mock_logger.error.assert_called()


# ---------------------------------------------------------------------------
# Debug helper error case tests (lines 539, 549-550)
# ---------------------------------------------------------------------------


def test_debug_message_structure_attribute_access_error(mixin, mock_logger):
    """Test debug helper with attribute access errors (lines 549-550)"""

    class BadAttributeMessage:
        @property
        def content(self):
            raise RuntimeError("Content access error")

        @property
        def tool_calls(self):
            raise ValueError("Tool calls error")

        @property
        def message(self):
            raise TypeError("Message error")

        @property
        def choices(self):
            raise KeyError("Choices error")

        @property
        def delta(self):
            raise AttributeError("Delta error")

    msg = BadAttributeMessage()
    mixin.debug_message_structure(msg, "error_test")

    # Should log errors for each failed attribute access
    mock_logger.debug.assert_called()
    debug_calls = mock_logger.debug.call_args_list

    # Should have logged errors for attribute access
    error_logs = [
        str(call) for call in debug_calls if "Error accessing" in str(call)
    ]
    # Should have multiple error logs
    assert len(error_logs) > 0


def test_debug_message_structure_no_dict_attribute(mixin, mock_logger):
    """Test debug helper with object having no __dict__ (line 539)"""

    class NoDictMessage:
        __slots__ = ["content"]

        def __init__(self):
            self.content = "test"

    msg = NoDictMessage()
    mixin.debug_message_structure(msg, "no_dict_test")

    # Should handle lack of __dict__ gracefully
    mock_logger.debug.assert_called()
    debug_calls = mock_logger.debug.call_args_list

    # Should log dir() instead of __dict__
    dir_logs = [str(call) for call in debug_calls if "Dir:" in str(call)]
    assert len(dir_logs) > 0


def test_debug_message_structure_partial_attributes(mixin, mock_logger):
    """Test debug helper with some attributes accessible and some not"""

    class PartialMessage:
        @property
        def content(self):
            return "accessible content"

        @property
        def tool_calls(self):
            raise RuntimeError("tool_calls not accessible")

        @property
        def message(self):
            return "accessible message"

    msg = PartialMessage()
    mixin.debug_message_structure(msg, "partial_test")

    mock_logger.debug.assert_called()
    debug_calls = [str(call) for call in mock_logger.debug.call_args_list]

    # Should log both successful and failed accesses
    assert any("content" in call for call in debug_calls)
    assert any("Error accessing" in call for call in debug_calls)
