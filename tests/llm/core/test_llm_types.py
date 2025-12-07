"""
Tests for chuk_llm.llm.core.types
==================================

Test LLM response types, validation, and normalization.
"""

import json
import pytest
from pydantic import ValidationError

from chuk_llm.llm.core.types import (
    LLMResponse,
    ResponseValidator,
    StreamChunk,
    ToolCall,
    ToolCallFunction,
)


class TestToolCallFunction:
    """Test ToolCallFunction validation"""

    def test_basic_function(self):
        """Test basic function with string arguments"""
        func = ToolCallFunction(name="test_function", arguments='{"key": "value"}')
        assert func.name == "test_function"
        assert func.arguments == '{"key": "value"}'

    def test_dict_arguments_converted_to_json(self):
        """Test that dict arguments are converted to JSON string"""
        func = ToolCallFunction(name="test", arguments={"param": "value"})
        assert func.name == "test"
        assert isinstance(func.arguments, str)
        parsed = json.loads(func.arguments)
        assert parsed == {"param": "value"}

    def test_invalid_json_string_fallback(self):
        """Test that invalid JSON strings fall back to empty object"""
        func = ToolCallFunction(name="test", arguments="not valid json")
        assert func.arguments == "{}"

    def test_other_type_fallback(self):
        """Test that other types fall back to empty object"""
        func = ToolCallFunction(name="test", arguments=123)
        assert func.arguments == "{}"

    def test_valid_json_string_preserved(self):
        """Test that valid JSON strings are preserved"""
        json_str = '{"a": 1, "b": [1, 2, 3]}'
        func = ToolCallFunction(name="test", arguments=json_str)
        assert func.arguments == json_str

    def test_empty_dict_converted(self):
        """Test empty dict is converted to empty JSON object"""
        func = ToolCallFunction(name="test", arguments={})
        assert func.arguments == "{}"


class TestToolCall:
    """Test ToolCall model"""

    def test_basic_tool_call(self):
        """Test basic tool call creation"""
        tc = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="get_weather", arguments='{"city": "SF"}'),
        )
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"

    def test_default_type(self):
        """Test default type is 'function'"""
        tc = ToolCall(
            id="call_123",
            function=ToolCallFunction(name="test", arguments="{}"),
        )
        assert tc.type == "function"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden"""
        with pytest.raises(ValidationError):
            ToolCall(
                id="call_123",
                function=ToolCallFunction(name="test", arguments="{}"),
                extra_field="not allowed",
            )


class TestLLMResponse:
    """Test LLMResponse model"""

    def test_response_with_text(self):
        """Test response with text content"""
        resp = LLMResponse(response="Hello, world!")
        assert resp.response == "Hello, world!"
        assert resp.tool_calls == []
        assert resp.error is False
        assert resp.error_message is None

    def test_response_with_tool_calls(self):
        """Test response with tool calls"""
        tc = ToolCall(
            id="call_1",
            function=ToolCallFunction(name="test", arguments="{}"),
        )
        resp = LLMResponse(tool_calls=[tc])
        assert resp.response is None
        assert len(resp.tool_calls) == 1
        assert resp.error is False

    def test_response_with_both(self):
        """Test response with both text and tool calls"""
        tc = ToolCall(
            id="call_1",
            function=ToolCallFunction(name="test", arguments="{}"),
        )
        resp = LLMResponse(response="Processing...", tool_calls=[tc])
        assert resp.response == "Processing..."
        assert len(resp.tool_calls) == 1

    def test_error_response(self):
        """Test error response"""
        resp = LLMResponse(
            error=True,
            error_message="API Error",
            response=None,
            tool_calls=[],
        )
        assert resp.error is True
        assert resp.error_message == "API Error"

    def test_validation_requires_content_or_error(self):
        """Test that non-error response requires content or tool calls"""
        with pytest.raises(ValidationError) as exc_info:
            LLMResponse(response=None, tool_calls=[], error=False)
        assert "Response must have either text content or tool calls" in str(exc_info.value)

    def test_empty_response_allowed_if_error(self):
        """Test that empty response is allowed if error=True"""
        resp = LLMResponse(error=True, error_message="Test error")
        assert resp.error is True
        assert resp.response is None
        assert resp.tool_calls == []

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden"""
        with pytest.raises(ValidationError):
            LLMResponse(response="test", extra_field="not allowed")


class TestStreamChunk:
    """Test StreamChunk model"""

    def test_basic_stream_chunk(self):
        """Test basic stream chunk creation"""
        chunk = StreamChunk(response="partial ", chunk_index=0)
        assert chunk.response == "partial "
        assert chunk.chunk_index == 0
        assert chunk.is_final is False

    def test_final_chunk(self):
        """Test final chunk"""
        chunk = StreamChunk(response="done", is_final=True)
        assert chunk.is_final is True

    def test_chunk_with_timestamp(self):
        """Test chunk with timestamp"""
        import time
        ts = time.time()
        chunk = StreamChunk(response="test", timestamp=ts)
        assert chunk.timestamp == ts

    def test_empty_chunk_with_error_flag(self):
        """Test that empty chunks need error flag (inherits from LLMResponse)"""
        # StreamChunk still validates like LLMResponse - needs content or error flag
        chunk = StreamChunk(error=True, error_message="Empty chunk")
        assert chunk.response is None
        assert chunk.tool_calls == []

    def test_chunk_with_content_and_metadata(self):
        """Test chunk with content and metadata"""
        chunk = StreamChunk(response="data", chunk_index=5, is_final=False)
        assert chunk.response == "data"
        assert chunk.chunk_index == 5
        assert chunk.is_final is False


class TestResponseValidator:
    """Test ResponseValidator utility"""

    def test_validate_response_success(self):
        """Test successful response validation"""
        raw = {"response": "Hello"}
        result = ResponseValidator.validate_response(raw, is_streaming=False)
        assert isinstance(result, LLMResponse)
        assert result.response == "Hello"
        assert result.error is False

    def test_validate_stream_chunk_success(self):
        """Test successful stream chunk validation"""
        raw = {"response": "partial", "chunk_index": 0}
        result = ResponseValidator.validate_response(raw, is_streaming=True)
        assert isinstance(result, StreamChunk)
        assert result.response == "partial"

    def test_validate_response_with_tool_calls(self):
        """Test response validation with tool calls"""
        raw = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ]
        }
        result = ResponseValidator.validate_response(raw)
        assert isinstance(result, LLMResponse)
        assert len(result.tool_calls) == 1

    def test_validate_invalid_response_returns_error(self):
        """Test that invalid response returns error response"""
        raw = {"invalid_field": "data"}  # Missing required content
        result = ResponseValidator.validate_response(raw)
        assert isinstance(result, LLMResponse)
        assert result.error is True
        assert "Response validation failed" in result.error_message

    def test_validate_invalid_streaming_returns_error(self):
        """Test that invalid streaming response returns error StreamChunk"""
        raw = {"completely": "invalid"}
        result = ResponseValidator.validate_response(raw, is_streaming=True)
        assert isinstance(result, StreamChunk)
        assert result.error is True

    def test_validate_double_failure_uses_construct(self):
        """Test that if even error response fails, model_construct is used"""
        # Create a response that will fail validation twice
        # by mocking the error creation to also fail
        from unittest.mock import patch

        # Patch to make error response also fail
        with patch.object(
            LLMResponse, '__init__', side_effect=Exception("Test error")
        ):
            raw = {"invalid": "data"}
            result = ResponseValidator.validate_response(raw)
            # Should still return a valid error response via model_construct
            assert result.error is True

    def test_normalize_tool_calls_openai_format(self):
        """Test normalizing tool calls in OpenAI format"""
        raw_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            }
        ]
        normalized = ResponseValidator.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].id == "call_1"
        assert normalized[0].function.name == "get_weather"

    def test_normalize_tool_calls_alternative_format(self):
        """Test normalizing tool calls in alternative format"""
        raw_calls = [
            {
                "id": "call_123",
                "name": "get_weather",
                "arguments": {"city": "NYC"},
            }
        ]
        normalized = ResponseValidator.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].function.name == "get_weather"

    def test_normalize_tool_calls_auto_id(self):
        """Test that missing ID gets auto-generated"""
        raw_calls = [
            {"name": "test_func", "arguments": "{}"}
        ]
        normalized = ResponseValidator.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].id == "call_0"

    def test_normalize_tool_calls_skip_invalid(self):
        """Test that invalid tool calls are skipped with warning"""
        raw_calls = [
            {"id": "call_1", "function": {"name": "valid", "arguments": "{}"}},
            {"completely": "invalid"},  # This should be skipped
            {"id": "call_2", "function": {"name": "valid2", "arguments": "{}"}},
        ]
        normalized = ResponseValidator.normalize_tool_calls(raw_calls)
        assert len(normalized) == 2  # Only 2 valid ones
        assert normalized[0].id == "call_1"
        assert normalized[1].id == "call_2"

    def test_normalize_empty_list(self):
        """Test normalizing empty tool call list"""
        normalized = ResponseValidator.normalize_tool_calls([])
        assert normalized == []

    def test_normalize_tool_calls_with_complex_arguments(self):
        """Test normalizing tool calls with complex argument structures"""
        raw_calls = [
            {
                "name": "complex_func",
                "arguments": {
                    "nested": {"key": "value"},
                    "array": [1, 2, 3],
                },
            }
        ]
        normalized = ResponseValidator.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        # Arguments should be JSON string
        args = json.loads(normalized[0].function.arguments)
        assert args["nested"]["key"] == "value"
        assert args["array"] == [1, 2, 3]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_tool_call_function_with_none_arguments(self):
        """Test ToolCallFunction with None arguments"""
        func = ToolCallFunction(name="test", arguments=None)
        assert func.arguments == "{}"

    def test_tool_call_function_with_list_arguments(self):
        """Test ToolCallFunction with list arguments (should fallback)"""
        func = ToolCallFunction(name="test", arguments=[1, 2, 3])
        assert func.arguments == "{}"

    def test_llm_response_defaults(self):
        """Test LLMResponse default values"""
        resp = LLMResponse(response="test")
        assert resp.tool_calls == []
        assert resp.error is False
        assert resp.error_message is None

    def test_stream_chunk_defaults(self):
        """Test StreamChunk default values"""
        chunk = StreamChunk(response="test")
        assert chunk.chunk_index is None
        assert chunk.is_final is False
        assert chunk.timestamp is None

    def test_validator_with_partial_data(self):
        """Test validator handles partial/incomplete data"""
        raw = {"response": ""}  # Empty string
        result = ResponseValidator.validate_response(raw)
        # Empty string should be treated as no content
        assert result.error is True

    def test_nested_tool_call_validation(self):
        """Test deeply nested tool call structures"""
        tc = ToolCall(
            id="call_complex",
            function=ToolCallFunction(
                name="complex",
                arguments=json.dumps({
                    "level1": {
                        "level2": {
                            "level3": ["a", "b", "c"]
                        }
                    }
                })
            ),
        )
        assert tc.function.name == "complex"
        args = json.loads(tc.function.arguments)
        assert args["level1"]["level2"]["level3"] == ["a", "b", "c"]
