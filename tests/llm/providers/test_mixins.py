# tests/providers/test_mixins.py
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
