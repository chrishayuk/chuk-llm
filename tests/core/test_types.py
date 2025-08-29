# tests/core/test_types.py
"""
Minimal tests for core type definitions that work with the actual implementation
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
    """Test ToolCallFunction with string arguments only"""

    def test_valid_json_string_arguments(self):
        """Test with valid JSON string arguments"""
        func = ToolCallFunction(
            name="get_weather", 
            arguments='{"location": "San Francisco"}'
        )
        assert func.name == "get_weather"
        assert func.arguments == '{"location": "San Francisco"}'

    def test_invalid_json_becomes_empty_object(self):
        """Test invalid JSON falls back to empty object"""
        func = ToolCallFunction(
            name="test", 
            arguments="not valid json"
        )
        assert func.arguments == "{}"

    def test_empty_json_object(self):
        """Test empty JSON object"""
        func = ToolCallFunction(name="test", arguments="{}")
        assert func.arguments == "{}"


class TestToolCall:
    """Test ToolCall model"""

    def test_basic_tool_call(self):
        """Test creating a basic tool call"""
        tool_call = ToolCall(
            id="test_id",
            function=ToolCallFunction(name="test", arguments="{}")
        )
        assert tool_call.id == "test_id"
        assert tool_call.type == "function"  # default
        assert tool_call.function.name == "test"

    def test_tool_call_forbids_extra_fields(self):
        """Test that extra fields are forbidden"""
        with pytest.raises(ValidationError):
            ToolCall(
                id="test",
                function=ToolCallFunction(name="test", arguments="{}"),
                extra="not_allowed"
            )


class TestResponseValidator:
    """Test ResponseValidator utility"""

    def test_validate_response_handles_errors(self):
        """Test that validation errors are caught and return error response"""
        # Invalid data should return an error response
        result = ResponseValidator.validate_response(
            {"invalid": "data"}, 
            is_streaming=False
        )
        
        # Should get back an LLMResponse with error=True
        assert isinstance(result, LLMResponse)
        assert result.error is True
        assert "validation failed" in result.error_message.lower()

    def test_normalize_tool_calls_basic(self):
        """Test basic tool call normalization"""
        raw_tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": '{"param": "value"}'
                }
            }
        ]
        
        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)
        
        assert len(normalized) == 1
        assert normalized[0].id == "call_123"
        assert normalized[0].function.name == "test_function"

    def test_normalize_tool_calls_alternative_format(self):
        """Test normalizing alternative format (name at top level)"""
        raw_tool_calls = [
            {
                "id": "call_456",
                "name": "direct_function",
                "arguments": "{}"
            }
        ]
        
        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)
        
        assert len(normalized) == 1
        assert normalized[0].id == "call_456"
        assert normalized[0].function.name == "direct_function"

    def test_normalize_tool_calls_auto_id(self):
        """Test auto-generation of IDs"""
        raw_tool_calls = [
            {"name": "func1", "arguments": "{}"},
            {"name": "func2", "arguments": "{}"}
        ]
        
        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)
        
        assert len(normalized) == 2
        assert normalized[0].id == "call_0"
        assert normalized[1].id == "call_1"

    def test_normalize_skips_invalid_tool_calls(self):
        """Test that invalid tool calls are skipped"""
        raw_tool_calls = [
            {"id": "valid", "name": "test", "arguments": "{}"},
            {"invalid": "data"},  # Should be skipped
            {"id": "valid2", "function": {"name": "test2", "arguments": "{}"}}
        ]
        
        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)
        
        # Should only have 2 valid ones
        assert len(normalized) == 2
        assert normalized[0].id == "valid"
        assert normalized[1].id == "valid2"


# Test the models can at least be created with error=True
class TestModelsWithError:
    """Test that models can be created with error flag"""
    
    def test_llm_response_with_error(self):
        """LLMResponse should work with error=True"""
        response = LLMResponse(error=True, error_message="Test error")
        assert response.error is True
        assert response.error_message == "Test error"
    
    def test_stream_chunk_with_error(self):
        """StreamChunk should work with error=True"""
        chunk = StreamChunk(error=True, error_message="Stream error")
        assert chunk.error is True
        assert chunk.error_message == "Stream error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])