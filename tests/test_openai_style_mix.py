# tests/test_openai_style_mixin.py
import asyncio
import json
import pytest
import re
import uuid
from unittest.mock import MagicMock, patch, PropertyMock

from chuk_llm.llm.openai_style_mixin import OpenAIStyleMixin

# Create a concrete class that uses the mixin for testing
class TestClient(OpenAIStyleMixin):
    """Test class that uses OpenAIStyleMixin."""
    pass

# ------------------------------------------------------------------ 
# Test _sanitize_tool_names
# ------------------------------------------------------------------ 
class TestSanitizeToolNames:
    def test_sanitize_tool_names_none(self):
        """Test that None tools input returns None."""
        result = TestClient._sanitize_tool_names(None)
        assert result is None

    def test_sanitize_tool_names_empty(self):
        """Test that empty tools list returns empty list."""
        result = TestClient._sanitize_tool_names([])
        assert result == []

    def test_sanitize_tool_names_valid(self):
        """Test that valid tool names are not changed."""
        tools = [
            {"function": {"name": "valid_name", "parameters": {}}},
            {"function": {"name": "also-valid-name", "parameters": {}}}
        ]
        result = TestClient._sanitize_tool_names(tools)
        assert result[0]["function"]["name"] == "valid_name"
        assert result[1]["function"]["name"] == "also-valid-name"

    def test_sanitize_tool_names_invalid(self):
        """Test that invalid tool names are sanitized."""
        tools = [
            {"function": {"name": "invalid@name", "parameters": {}}},
            {"function": {"name": "also!invalid$name", "parameters": {}}},
            {"function": {"name": "spaces are bad", "parameters": {}}}
        ]
        result = TestClient._sanitize_tool_names(tools)
        assert result[0]["function"]["name"] == "invalid_name"
        assert result[1]["function"]["name"] == "also_invalid_name"
        assert result[2]["function"]["name"] == "spaces_are_bad"

    def test_sanitize_tool_names_mixed(self):
        """Test processing a mix of valid and invalid names."""
        tools = [
            {"function": {"name": "valid_name", "parameters": {}}},
            {"function": {"name": "invalid@name", "parameters": {}}},
        ]
        result = TestClient._sanitize_tool_names(tools)
        assert result[0]["function"]["name"] == "valid_name"
        assert result[1]["function"]["name"] == "invalid_name"

    def test_sanitize_tool_names_no_function(self):
        """Test tool item with no function key."""
        tools = [
            {"type": "function"},  # No function key
            {"function": {"name": "valid_name", "parameters": {}}}
        ]
        result = TestClient._sanitize_tool_names(tools)
        # Should pass through unchanged
        assert result[0] == {"type": "function"}
        assert result[1]["function"]["name"] == "valid_name"

    def test_sanitize_tool_names_no_name(self):
        """Test function with no name key."""
        tools = [
            {"function": {"parameters": {}}},  # No name key
            {"function": {"name": "valid_name", "parameters": {}}}
        ]
        result = TestClient._sanitize_tool_names(tools)
        # Should pass through unchanged
        assert "name" not in result[0]["function"]
        assert result[1]["function"]["name"] == "valid_name"

# ------------------------------------------------------------------ 
# Test _call_blocking
# ------------------------------------------------------------------ 
@pytest.mark.asyncio
class TestCallBlocking:
    async def test_call_blocking_simple(self):
        """Test _call_blocking with a simple function."""
        def simple_func(x, y):
            return x + y
        
        result = await TestClient._call_blocking(simple_func, 5, 7)
        assert result == 12

    async def test_call_blocking_with_kwargs(self):
        """Test _call_blocking with kwargs."""
        def kw_func(x, y=10):
            return x + y
        
        result = await TestClient._call_blocking(kw_func, 5, y=15)
        assert result == 20

    async def test_call_blocking_with_exception(self):
        """Test _call_blocking with a function that raises an exception."""
        def error_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await TestClient._call_blocking(error_func)

    async def test_call_blocking_with_async_context(self):
        """Test that _call_blocking runs in a thread."""
        current_thread_id = None
        executor_thread_id = None
        
        def thread_func():
            nonlocal executor_thread_id
            import threading
            executor_thread_id = threading.get_ident()
            return executor_thread_id
        
        async def run_test():
            nonlocal current_thread_id
            import threading
            current_thread_id = threading.get_ident()
            return await TestClient._call_blocking(thread_func)
        
        result = await run_test()
        assert result == executor_thread_id
        assert current_thread_id != executor_thread_id

# ------------------------------------------------------------------ 
# Test _normalise_message
# ------------------------------------------------------------------ 
class TestNormaliseMessage:
    def test_normalise_message_text_only(self):
        """Test normalizing a message with only text content."""
        msg = MagicMock()
        msg.content = "Hello, world!"
        msg.tool_calls = None
        
        result = TestClient._normalise_message(msg)
        assert result == {"response": "Hello, world!", "tool_calls": []}

    def test_normalise_message_with_tool_calls(self):
        """Test normalizing a message with tool calls."""
        msg = MagicMock()
        msg.content = None
        
        tool_call1 = MagicMock()
        tool_call1.id = "call_1"
        tool_call1.function.name = "get_weather"
        tool_call1.function.arguments = '{"location": "London"}'
        
        tool_call2 = MagicMock()
        tool_call2.id = "call_2"
        tool_call2.function.name = "search"
        tool_call2.function.arguments = '{"query": "Python programming"}'
        
        msg.tool_calls = [tool_call1, tool_call2]
        
        result = TestClient._normalise_message(msg)
        assert result["response"] is None
        assert len(result["tool_calls"]) == 2
        
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"location": "London"}
        
        assert result["tool_calls"][1]["id"] == "call_2"
        assert result["tool_calls"][1]["function"]["name"] == "search"
        assert json.loads(result["tool_calls"][1]["function"]["arguments"]) == {"query": "Python programming"}

    def test_normalise_message_tool_call_no_id(self):
        """Test normalizing a tool call without an ID."""
        msg = MagicMock()
        msg.content = None
        
        tool_call = MagicMock()
        tool_call.id = None  # No ID provided
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"location": "London"}'
        
        msg.tool_calls = [tool_call]
        
        # Mock uuid to get a predictable result
        with patch('uuid.uuid4', return_value=MagicMock(hex='abcdef1234567890')):
            result = TestClient._normalise_message(msg)
        
        assert result["response"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_abcdef12"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_normalise_message_invalid_arguments(self):
        """Test normalizing a tool call with invalid JSON arguments."""
        msg = MagicMock()
        msg.content = None
        
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{invalid json}'  # Invalid JSON
        
        msg.tool_calls = [tool_call]
        
        result = TestClient._normalise_message(msg)
        assert result["response"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"  # Default empty JSON

    def test_normalise_message_dict_arguments(self):
        """Test normalizing a tool call with dictionary arguments (not a string)."""
        msg = MagicMock()
        msg.content = None
        
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = {"location": "London"}  # Dict instead of string
        
        msg.tool_calls = [tool_call]
        
        result = TestClient._normalise_message(msg)
        assert result["response"] is None
        assert len(result["tool_calls"]) == 1
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"location": "London"}

# ------------------------------------------------------------------ 
# Test _stream_from_blocking
# ------------------------------------------------------------------ 
@pytest.mark.asyncio
class TestStreamFromBlocking:
    async def test_stream_from_blocking_simple(self):
        """Test streaming functionality with simple chunks."""
        # Create mock chunks that the SDK would return
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = []
        
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.tool_calls = []
        
        # Mock SDK call that returns these chunks
        def mock_sdk_call(stream=False, **kwargs):
            assert stream is True  # Ensure stream=True is passed
            yield chunk1
            yield chunk2
        
        # Use the mixin to wrap this SDK call
        stream = TestClient._stream_from_blocking(mock_sdk_call, param1="value1")
        
        # Collect all chunks from the async iterator
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Verify the chunks were processed correctly
        assert len(chunks) == 2
        assert chunks[0] == {"response": "Hello", "tool_calls": []}
        assert chunks[1] == {"response": " world", "tool_calls": []}

    async def test_stream_from_blocking_tool_calls(self):
        """Test streaming with tool call chunks."""
        # Create a chunk with a tool call
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"location": "London"}'
        
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        chunk.choices[0].delta.tool_calls = [tool_call]
        
        # Mock SDK call
        def mock_sdk_call(stream=False, **kwargs):
            assert stream is True
            yield chunk
        
        # Use the mixin
        stream = TestClient._stream_from_blocking(mock_sdk_call)
        
        # Collect chunks
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Verify
        assert len(chunks) == 1
        assert chunks[0]["response"] == ""  # Empty string for content when there's a tool call
        assert len(chunks[0]["tool_calls"]) == 1
        assert chunks[0]["tool_calls"][0].id == "call_1"
        assert chunks[0]["tool_calls"][0].function.name == "get_weather"

    async def test_stream_from_blocking_empty(self):
        """Test streaming with no chunks."""
        # Mock SDK call that yields nothing
        def mock_sdk_call(stream=False, **kwargs):
            if False:  # Just to make it a generator
                yield None
            # No actual yield statements = empty generator
        
        # Use the mixin
        stream = TestClient._stream_from_blocking(mock_sdk_call)
        
        # Collect chunks
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Verify no chunks were received
        assert len(chunks) == 0

    async def test_stream_from_blocking_exception(self):
        """Test streaming with an exception in the generator."""
        # Mock SDK call that raises an exception
        def mock_sdk_call(stream=False, **kwargs):
            yield MagicMock()  # First chunk works
            raise ValueError("Test error")  # Then an error
        
        # Use the mixin
        stream = TestClient._stream_from_blocking(mock_sdk_call)
        
        # Get the first item (should work)
        first_item = await anext(stream.__aiter__())
        assert "response" in first_item
        
        # The second iteration should give None due to the worker ending after the exception
        # The exception is logged but not propagated to the main thread (as seen in your log)
        second_item = None
        try:
            second_item = await anext(stream.__aiter__())
        except StopAsyncIteration:
            pass  # Expected - the stream ends when the worker encounters an error
        
        assert second_item is None  # Stream should terminate after error