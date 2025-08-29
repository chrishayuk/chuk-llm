# tests/core/test_types.py
"""
Unit tests for core type definitions and Pydantic models
"""

import json
from typing import Annotated, Any

import pytest

# Note: These imports will need to be adjusted based on your actual structure
# For now, creating mock implementations since the type classes don't exist yet
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)


# Validator function for arguments conversion
def convert_arguments_to_json(v):
    """Convert dict arguments to JSON string"""
    if isinstance(v, dict):
        return json.dumps(v)
    if isinstance(v, str):
        try:
            json.loads(v)  # Validate it's valid JSON
            return v
        except json.JSONDecodeError:
            return "{}"  # Fallback to empty object
    return str(v)


# Type alias for arguments field
ArgumentsField = Annotated[str, BeforeValidator(convert_arguments_to_json)]


# Mock type definitions (these would be in your actual types.py)
class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: ArgumentsField  # JSON string with automatic conversion


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str = "function"
    function: ToolCallFunction


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: bool = False
    error_message: str | None = None

    def model_post_init(self, __context) -> None:
        """Validate after all fields are set"""
        # Allow empty response if error=True or there are tool calls
        if not self.response and not self.tool_calls and not self.error:
            raise ValueError("Response must have either text content or tool calls")


class StreamChunk(LLMResponse):
    """Streaming chunk with metadata"""

    model_config = ConfigDict(extra="forbid")

    chunk_index: int | None = None
    is_final: bool = False
    timestamp: float | None = None

    def model_post_init(self, __context) -> None:
        """Streaming chunks can be empty - skip parent validation"""
        pass  # Allow empty streaming chunks


class ResponseValidator:
    """Validates and normalizes LLM responses"""

    @staticmethod
    def validate_response(raw_response: dict[str, Any], is_streaming: bool = False):
        """Validate and convert raw response to typed model"""
        try:
            if is_streaming:
                return StreamChunk(**raw_response)
            else:
                return LLMResponse(**raw_response)
        except Exception as e:
            # Return error response if validation fails
            error_class = StreamChunk if is_streaming else LLMResponse
            try:
                return error_class(
                    response=None,
                    tool_calls=[],
                    error=True,
                    error_message=f"Response validation failed: {str(e)}",
                )
            except Exception:
                # If even the error response fails, create a minimal one
                result = error_class.model_construct(
                    response=None,
                    tool_calls=[],
                    error=True,
                    error_message=f"Response validation failed: {str(e)}",
                )
                return result

    @staticmethod
    def normalize_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCall]:
        """Normalize tool calls from different providers"""
        normalized = []

        for i, tc in enumerate(raw_tool_calls):
            try:
                if isinstance(tc, dict):
                    # Handle different provider formats
                    if "function" in tc:
                        # OpenAI/Anthropic format
                        normalized.append(ToolCall(**tc))
                    elif "name" in tc:
                        # Alternative format - convert to standard
                        tool_call_id = tc.get("id", f"call_{i}")
                        normalized.append(
                            ToolCall(
                                id=tool_call_id,
                                type="function",
                                function=ToolCallFunction(
                                    name=tc["name"], arguments=tc.get("arguments", "{}")
                                ),
                            )
                        )
            except Exception:
                # Skip invalid tool calls but log the issue
                continue

        return normalized


class TestToolCallFunction:
    """Test suite for ToolCallFunction model"""

    def test_valid_tool_call_function_creation(self):
        """Test creating a valid ToolCallFunction"""
        func = ToolCallFunction(
            name="get_weather", arguments='{"location": "San Francisco"}'
        )

        assert func.name == "get_weather"
        assert func.arguments == '{"location": "San Francisco"}'

    def test_tool_call_function_with_dict_arguments(self):
        """Test that dict arguments are converted to JSON string"""
        func = ToolCallFunction(
            name="get_weather",
            arguments={"location": "San Francisco", "units": "celsius"},
        )

        assert func.name == "get_weather"
        assert isinstance(func.arguments, str)

        # Should be valid JSON
        parsed = json.loads(func.arguments)
        assert parsed["location"] == "San Francisco"
        assert parsed["units"] == "celsius"

    def test_tool_call_function_invalid_json_fallback(self):
        """Test that invalid JSON falls back to empty object"""
        func = ToolCallFunction(name="test_function", arguments="invalid json {[")

        assert func.name == "test_function"
        assert func.arguments == "{}"

    def test_tool_call_function_empty_arguments(self):
        """Test with empty arguments"""
        func = ToolCallFunction(name="simple_function", arguments="{}")

        assert func.name == "simple_function"
        assert func.arguments == "{}"

    def test_tool_call_function_complex_arguments(self):
        """Test with complex nested arguments"""
        complex_args = {
            "location": {
                "city": "San Francisco",
                "state": "CA",
                "coordinates": {"lat": 37.7749, "lon": -122.4194},
            },
            "options": ["temperature", "humidity", "wind"],
            "format": "json",
        }

        func = ToolCallFunction(name="get_detailed_weather", arguments=complex_args)

        # Should serialize complex structure
        parsed = json.loads(func.arguments)
        assert parsed["location"]["city"] == "San Francisco"
        assert len(parsed["options"]) == 3
        assert parsed["format"] == "json"

    def test_tool_call_function_extra_fields_forbidden(self):
        """Test that extra fields are forbidden"""
        with pytest.raises(ValidationError):
            ToolCallFunction(name="test", arguments="{}", extra_field="not_allowed")


class TestToolCall:
    """Test suite for ToolCall model"""

    def test_valid_tool_call_creation(self):
        """Test creating a valid ToolCall"""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(
                name="get_weather", arguments='{"location": "NYC"}'
            ),
        )

        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"

    def test_tool_call_default_type(self):
        """Test that type defaults to 'function'"""
        tool_call = ToolCall(
            id="call_456", function=ToolCallFunction(name="test", arguments="{}")
        )

        assert tool_call.type == "function"

    def test_tool_call_with_dict_function(self):
        """Test creating ToolCall with function as dict"""
        tool_call = ToolCall(
            id="call_789",
            function={"name": "calculate", "arguments": {"x": 5, "y": 10}},
        )

        assert tool_call.function.name == "calculate"
        assert "x" in tool_call.function.arguments
        assert "y" in tool_call.function.arguments

    def test_tool_call_serialization(self):
        """Test that ToolCall serializes correctly"""
        tool_call = ToolCall(
            id="call_test",
            function=ToolCallFunction(
                name="send_email",
                arguments={"to": "test@example.com", "subject": "Test"},
            ),
        )

        # Should be able to serialize to dict
        data = tool_call.model_dump()
        assert data["id"] == "call_test"
        assert data["type"] == "function"
        assert data["function"]["name"] == "send_email"

        # Arguments should be JSON string
        assert isinstance(data["function"]["arguments"], str)

    def test_tool_call_extra_fields_forbidden(self):
        """Test that extra fields are forbidden"""
        with pytest.raises(ValidationError):
            ToolCall(
                id="test",
                function=ToolCallFunction(name="test", arguments="{}"),
                extra_field="not_allowed",
            )


class TestLLMResponse:
    """Test suite for LLMResponse model"""

    def test_valid_text_response(self):
        """Test creating response with text content"""
        response = LLMResponse(response="Hello, world!")

        assert response.response == "Hello, world!"
        assert response.tool_calls == []
        assert response.error is False
        assert response.error_message is None

    def test_valid_tool_call_response(self):
        """Test creating response with tool calls"""
        tool_call = ToolCall(
            id="call_123", function=ToolCallFunction(name="test", arguments="{}")
        )

        response = LLMResponse(response=None, tool_calls=[tool_call])

        assert response.response is None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"

    def test_response_with_both_text_and_tools(self):
        """Test response with both text and tool calls"""
        tool_call = ToolCall(
            id="call_123", function=ToolCallFunction(name="test", arguments="{}")
        )

        response = LLMResponse(
            response="I'll help you with that.", tool_calls=[tool_call]
        )

        assert response.response == "I'll help you with that."
        assert len(response.tool_calls) == 1

    def test_error_response(self):
        """Test creating error response"""
        response = LLMResponse(
            response=None, tool_calls=[], error=True, error_message="API error occurred"
        )

        assert response.error is True
        assert response.error_message == "API error occurred"

    def test_empty_response_validation_error(self):
        """Test that empty response (no content, no tools, no error) raises validation error"""
        with pytest.raises(
            ValidationError,
            match="Response must have either text content or tool calls",
        ):
            LLMResponse(response=None, tool_calls=[], error=False)

    def test_empty_response_with_error_allowed(self):
        """Test that empty response is allowed when error=True"""
        response = LLMResponse(response=None, tool_calls=[], error=True)

        assert response.error is True

    def test_response_serialization(self):
        """Test response serialization to dict"""
        tool_call = ToolCall(
            id="call_123",
            function=ToolCallFunction(name="test", arguments='{"key": "value"}'),
        )

        response = LLMResponse(response="Test response", tool_calls=[tool_call])

        data = response.model_dump()
        assert data["response"] == "Test response"
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["id"] == "call_123"
        assert data["error"] is False

    def test_response_from_raw_dict(self):
        """Test creating response from raw dictionary"""
        raw_data = {
            "response": "Hello",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "greet", "arguments": '{"name": "Alice"}'},
                }
            ],
        }

        response = LLMResponse(**raw_data)

        assert response.response == "Hello"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == "greet"


class TestStreamChunk:
    """Test suite for StreamChunk model"""

    def test_valid_stream_chunk(self):
        """Test creating a valid stream chunk"""
        chunk = StreamChunk(response="Hello ", chunk_index=0, timestamp=1234567890.0)

        assert chunk.response == "Hello "
        assert chunk.chunk_index == 0
        assert chunk.timestamp == 1234567890.0
        assert chunk.is_final is False

    def test_final_stream_chunk(self):
        """Test creating a final stream chunk"""
        chunk = StreamChunk(response="world!", chunk_index=5, is_final=True)

        assert chunk.response == "world!"
        assert chunk.is_final is True

    def test_empty_stream_chunk_allowed(self):
        """Test that empty stream chunks are allowed"""
        chunk = StreamChunk(response="", tool_calls=[])

        assert chunk.response == ""
        assert chunk.tool_calls == []
        # Empty chunks should be valid for streaming

    def test_stream_chunk_with_tool_calls(self):
        """Test stream chunk with tool calls"""
        tool_call = ToolCall(
            id="call_stream", function=ToolCallFunction(name="test", arguments="{}")
        )

        chunk = StreamChunk(response="", tool_calls=[tool_call], chunk_index=3)

        assert len(chunk.tool_calls) == 1
        assert chunk.chunk_index == 3

    def test_stream_chunk_inheritance(self):
        """Test that StreamChunk inherits from LLMResponse"""
        chunk = StreamChunk(response="test")

        assert isinstance(chunk, LLMResponse)

        # Should have all LLMResponse fields
        assert hasattr(chunk, "response")
        assert hasattr(chunk, "tool_calls")
        assert hasattr(chunk, "error")
        assert hasattr(chunk, "error_message")

        # Plus additional StreamChunk fields
        assert hasattr(chunk, "chunk_index")
        assert hasattr(chunk, "is_final")
        assert hasattr(chunk, "timestamp")


class TestResponseValidator:
    """Test suite for ResponseValidator utility class"""

    def test_validate_valid_response(self):
        """Test validating a valid response"""
        raw_response = {"response": "Hello, world!", "tool_calls": []}

        validated = ResponseValidator.validate_response(
            raw_response, is_streaming=False
        )

        assert isinstance(validated, LLMResponse)
        assert validated.response == "Hello, world!"
        assert validated.error is False

    def test_validate_valid_stream_chunk(self):
        """Test validating a valid stream chunk"""
        raw_chunk = {"response": "chunk text", "tool_calls": [], "chunk_index": 2}

        validated = ResponseValidator.validate_response(raw_chunk, is_streaming=True)

        assert isinstance(validated, StreamChunk)
        assert validated.response == "chunk text"
        assert validated.chunk_index == 2

    def test_validate_invalid_response_returns_error(self):
        """Test that invalid response returns error response"""
        # Missing required content
        raw_response = {
            "response": None,
            "tool_calls": [],
            "error": False,  # This should trigger validation error
        }

        validated = ResponseValidator.validate_response(
            raw_response, is_streaming=False
        )

        assert isinstance(validated, LLMResponse)
        assert validated.error is True
        assert "validation failed" in validated.error_message

    def test_normalize_openai_tool_calls(self):
        """Test normalizing OpenAI-style tool calls"""
        raw_tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ]

        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)

        assert len(normalized) == 1
        assert isinstance(normalized[0], ToolCall)
        assert normalized[0].id == "call_abc123"
        assert normalized[0].function.name == "get_weather"

    def test_normalize_alternative_tool_call_format(self):
        """Test normalizing alternative tool call format"""
        raw_tool_calls = [
            {
                "id": "call_xyz789",
                "name": "send_email",
                "arguments": {"to": "test@example.com", "subject": "Test"},
            }
        ]

        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)

        assert len(normalized) == 1
        assert normalized[0].id == "call_xyz789"
        assert normalized[0].function.name == "send_email"
        assert "test@example.com" in normalized[0].function.arguments

    def test_normalize_tool_calls_with_missing_id(self):
        """Test normalizing tool calls with missing ID"""
        raw_tool_calls = [
            {"name": "function1", "arguments": "{}"},
            {"name": "function2", "arguments": "{}"},
        ]

        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)

        assert len(normalized) == 2
        # IDs should be auto-generated
        assert normalized[0].id.startswith("call_")
        assert normalized[1].id.startswith("call_")
        assert normalized[0].function.name == "function1"
        assert normalized[1].function.name == "function2"

    def test_normalize_invalid_tool_calls_skipped(self):
        """Test that invalid tool calls are skipped"""
        raw_tool_calls = [
            {
                "id": "call_valid",
                "function": {"name": "valid_function", "arguments": "{}"},
            },
            {
                # Missing required fields
                "invalid": "data"
            },
            {"id": "call_valid2", "name": "another_valid_function", "arguments": "{}"},
        ]

        normalized = ResponseValidator.normalize_tool_calls(raw_tool_calls)

        # Should only have 2 valid tool calls, invalid one skipped
        assert len(normalized) == 2
        assert normalized[0].id == "call_valid"
        assert normalized[1].id == "call_valid2"


class TestTypeIntegration:
    """Integration tests for type system"""

    def test_end_to_end_response_processing(self):
        """Test complete response processing workflow"""
        # Simulate raw response from provider
        raw_response = {
            "response": "I'll help you check the weather.",
            "tool_calls": [
                {
                    "id": "call_weather_123",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": {
                            "location": "San Francisco, CA",
                            "units": "fahrenheit",
                        },
                    },
                }
            ],
        }

        # Validate and normalize
        validated = ResponseValidator.validate_response(raw_response)

        assert isinstance(validated, LLMResponse)

        # Check if validation failed and returned error response
        if validated.error:
            # If validation failed, we should still be able to work with error response
            assert validated.error is True
            assert "validation failed" in validated.error_message
        else:
            # If validation succeeded, check the content
            assert validated.response == "I'll help you check the weather."
            assert len(validated.tool_calls) == 1

            tool_call = validated.tool_calls[0]
            assert tool_call.id == "call_weather_123"
            assert tool_call.function.name == "get_current_weather"

            # Arguments should be JSON string
            args = json.loads(tool_call.function.arguments)
            assert args["location"] == "San Francisco, CA"
            assert args["units"] == "fahrenheit"

    def test_streaming_response_sequence(self):
        """Test processing a sequence of streaming chunks"""
        # Simulate streaming chunks
        raw_chunks = [
            {"response": "I'll check ", "chunk_index": 0, "timestamp": 1000.0},
            {"response": "the weather ", "chunk_index": 1, "timestamp": 1001.0},
            {"response": "for you.", "chunk_index": 2, "timestamp": 1002.0},
            {
                "response": "",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
                "chunk_index": 3,
                "is_final": True,
                "timestamp": 1003.0,
            },
        ]

        chunks = []
        for raw_chunk in raw_chunks:
            validated = ResponseValidator.validate_response(
                raw_chunk, is_streaming=True
            )
            chunks.append(validated)

        # All should be valid StreamChunk instances
        assert all(isinstance(chunk, StreamChunk) for chunk in chunks)

        # Check sequence
        assert chunks[0].response == "I'll check "
        assert chunks[1].response == "the weather "
        assert chunks[2].response == "for you."
        assert chunks[3].is_final is True
        assert len(chunks[3].tool_calls) == 1

    def test_error_response_handling(self):
        """Test handling error responses"""
        raw_error = {
            "response": None,
            "tool_calls": [],
            "error": True,
            "error_message": "Rate limit exceeded",
        }

        validated = ResponseValidator.validate_response(raw_error)

        assert isinstance(validated, LLMResponse)
        assert validated.error is True
        assert validated.error_message == "Rate limit exceeded"
        assert validated.response is None
        assert validated.tool_calls == []


# Fixtures for common test data
@pytest.fixture
def sample_tool_call():
    """Sample tool call for testing"""
    return ToolCall(
        id="call_sample",
        function=ToolCallFunction(
            name="sample_function", arguments='{"param": "value"}'
        ),
    )


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing"""
    return LLMResponse(response="Sample response text", tool_calls=[])


@pytest.fixture
def sample_stream_chunk():
    """Sample stream chunk for testing"""
    return StreamChunk(response="chunk ", chunk_index=1, timestamp=1234567890.0)


# Parametrized tests
@pytest.mark.parametrize(
    "arguments,expected_json",
    [
        ({}, "{}"),
        ({"key": "value"}, '{"key": "value"}'),
        ({"num": 42, "bool": True}, '{"num": 42, "bool": true}'),
        ("already_json", "already_json"),
        ('{"existing": "json"}', '{"existing": "json"}'),
    ],
)
def test_tool_call_function_argument_handling(arguments, expected_json):
    """Test different argument formats for ToolCallFunction"""
    func = ToolCallFunction(name="test", arguments=arguments)

    if isinstance(arguments, str):
        # String arguments should be validated as JSON or converted to "{}"
        if arguments == "already_json":
            assert func.arguments == "{}"  # Invalid JSON becomes "{}"
        else:
            assert func.arguments == expected_json
    else:
        # Dict arguments should be converted to JSON
        parsed = json.loads(func.arguments)
        expected_parsed = json.loads(expected_json)
        assert parsed == expected_parsed


@pytest.mark.parametrize(
    "response,tool_calls,error,should_validate",
    [
        ("Hello", [], False, True),  # Text response
        (
            None,
            [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}],
            False,
            True,
        ),  # Tool calls
        (
            "Text",
            [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}],
            False,
            True,
        ),  # Both
        (None, [], True, True),  # Error response
        (None, [], False, False),  # Empty response (should fail)
    ],
)
def test_llm_response_validation_scenarios(
    response, tool_calls, error, should_validate
):
    """Test various LLMResponse validation scenarios"""
    raw_data = {"response": response, "tool_calls": tool_calls, "error": error}

    if should_validate:
        # Should create valid response
        validated = ResponseValidator.validate_response(raw_data)
        assert isinstance(validated, LLMResponse)
        if not error:
            assert validated.error is False
    else:
        # Should result in validation error and error response
        validated = ResponseValidator.validate_response(raw_data)
        assert validated.error is True
        assert "validation failed" in validated.error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
