"""Comprehensive tests for registry/testing/capability_tests.py"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from chuk_llm.registry.testing import capability_tests
from chuk_llm.core.enums import MessageRole

# Import functions - prefix with underscore to prevent pytest from collecting them as tests
_test_tools_func = capability_tests.test_tools
_test_vision_func = capability_tests.test_vision
_test_json_mode_func = capability_tests.test_json_mode
_test_structured_outputs_func = capability_tests.test_structured_outputs
_test_streaming_func = capability_tests.test_streaming
_test_text_func = capability_tests.test_text
_test_chat_model_func = capability_tests.test_chat_model
RED_SQUARE_PNG = capability_tests.RED_SQUARE_PNG


class TestRedSquarePNG:
    """Test the RED_SQUARE_PNG constant"""

    def test_red_square_png_is_valid_base64(self):
        """Test that RED_SQUARE_PNG is a valid base64 string"""
        import base64

        try:
            decoded = base64.b64decode(RED_SQUARE_PNG)
            assert len(decoded) > 0
            # PNG files start with specific magic bytes
            assert decoded[:4] == b'\x89PNG'
        except Exception as e:
            pytest.fail(f"RED_SQUARE_PNG is not valid base64: {e}")


class TestToolsCapability:
    """Test the test_tools function"""

    @pytest.mark.asyncio
    async def test_tools_success_with_tool_calls(self):
        """Test successful tool call detection"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={
                "response": "Let me check the weather",
                "tool_calls": [
                    {
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}
                    }
                ]
            }
        )

        result = await _test_tools_func(mock_client)

        assert result is True
        mock_client.create_completion.assert_called_once()
        call_args = mock_client.create_completion.call_args
        assert call_args[1]["max_tokens"] == 50
        assert call_args[1]["timeout"] == 10.0

    @pytest.mark.asyncio
    async def test_tools_failure_no_tool_calls(self):
        """Test when model doesn't use tools"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "I cannot check the weather"}
        )

        result = await _test_tools_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_tools_error_response(self):
        """Test when API returns error"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "Tools not supported"}
        )

        result = await _test_tools_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_tools_exception_tool_related(self):
        """Test exception handling for tool-related errors"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Tool calling not supported")
        )

        result = await _test_tools_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_tools_exception_function_related(self):
        """Test exception handling for function-related errors"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Function calls are not supported")
        )

        result = await _test_tools_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_tools_exception_generic(self):
        """Test exception handling for generic errors"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Network error")
        )

        result = await _test_tools_func(mock_client)

        assert result is False


class TestVisionCapability:
    """Test the test_vision function"""

    @pytest.mark.asyncio
    async def test_vision_success_image_url(self):
        """Test successful vision with image_url format"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "This is a red square"}
        )

        result = await _test_vision_func(mock_client)

        assert result is True

    @pytest.mark.asyncio
    async def test_vision_success_with_choices(self):
        """Test successful vision with choices format"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Red image"}}]}
        )

        result = await _test_vision_func(mock_client)

        assert result is True

    @pytest.mark.asyncio
    async def test_vision_fallback_to_image_data(self):
        """Test fallback from image_url to image_data format"""
        mock_client = Mock()

        # First call fails, second succeeds
        mock_client.create_completion = AsyncMock(
            side_effect=[
                {"error": "image_url not supported"},
                {"response": "Red square"}
            ]
        )

        result = await _test_vision_func(mock_client)

        assert result is True
        assert mock_client.create_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_vision_both_formats_fail(self):
        """Test when both image formats fail"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "Vision not supported"}
        )

        result = await _test_vision_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_vision_exception_image_related(self):
        """Test exception with image-related error"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Image format not supported")
        )

        result = await _test_vision_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_vision_exception_vision_keyword(self):
        """Test exception with vision keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Vision capabilities not available")
        )

        result = await _test_vision_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_vision_exception_invalid_content(self):
        """Test exception with invalid content error"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Invalid content type")
        )

        result = await _test_vision_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_vision_exception_generic_first_call(self):
        """Test generic exception on first call, then success"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=[
                Exception("Timeout"),
                {"response": "Red"}
            ]
        )

        result = await _test_vision_func(mock_client)

        # Should return False on generic error
        assert result is False

    @pytest.mark.asyncio
    async def test_vision_empty_response(self):
        """Test when response is empty"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=[
                {},  # Empty response on first try
                {"response": "Red"}  # Success on second try
            ]
        )

        result = await _test_vision_func(mock_client)

        assert result is True


class TestJSONModeCapability:
    """Test the test_json_mode function"""

    @pytest.mark.asyncio
    async def test_json_mode_success(self):
        """Test successful JSON mode"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": '{"greeting": "Hello"}'}
        )

        result = await _test_json_mode_func(mock_client)

        assert result is True
        call_args = mock_client.create_completion.call_args
        assert call_args[1]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_json_mode_error_response(self):
        """Test when JSON mode returns error"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "JSON mode not supported"}
        )

        result = await _test_json_mode_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_json_mode_exception_json_keyword(self):
        """Test exception with json keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("JSON mode not available")
        )

        result = await _test_json_mode_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_json_mode_exception_response_format(self):
        """Test exception with response_format keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("response_format parameter not supported")
        )

        result = await _test_json_mode_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_json_mode_exception_invalid_parameter(self):
        """Test exception with invalid parameter"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Invalid parameter: response_format")
        )

        result = await _test_json_mode_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_json_mode_exception_generic(self):
        """Test generic exception"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Network timeout")
        )

        result = await _test_json_mode_func(mock_client)

        assert result is False


class TestStructuredOutputsCapability:
    """Test the test_structured_outputs function"""

    @pytest.mark.asyncio
    async def test_structured_outputs_success(self):
        """Test successful structured outputs"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": '{"name": "John", "age": 30}'}
        )

        result = await _test_structured_outputs_func(mock_client)

        assert result is True
        call_args = mock_client.create_completion.call_args
        assert "response_format" in call_args[1]
        assert call_args[1]["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_structured_outputs_error_response(self):
        """Test when structured outputs returns error"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "Structured outputs not supported"}
        )

        result = await _test_structured_outputs_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_structured_outputs_exception_json_schema(self):
        """Test exception with json_schema keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("json_schema format not supported")
        )

        result = await _test_structured_outputs_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_structured_outputs_exception_structured_keyword(self):
        """Test exception with structured keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Structured outputs not available")
        )

        result = await _test_structured_outputs_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_structured_outputs_exception_strict_keyword(self):
        """Test exception with strict keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Strict mode not supported")
        )

        result = await _test_structured_outputs_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_structured_outputs_exception_generic(self):
        """Test generic exception"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        result = await _test_structured_outputs_func(mock_client)

        assert result is False


class TestStreamingCapability:
    """Test the test_streaming function"""

    @pytest.mark.asyncio
    async def test_streaming_success(self):
        """Test successful streaming"""
        mock_client = Mock()

        async def mock_stream(*args, **kwargs):
            yield {"response": "Hello"}
            yield {"response": " world"}

        mock_client.create_completion = mock_stream

        result = await _test_streaming_func(mock_client)

        assert result is True

    @pytest.mark.asyncio
    async def test_streaming_success_single_chunk(self):
        """Test streaming with single chunk"""
        mock_client = Mock()

        async def mock_stream(*args, **kwargs):
            yield {"response": "Hello"}

        mock_client.create_completion = mock_stream

        result = await _test_streaming_func(mock_client)

        assert result is True

    @pytest.mark.asyncio
    async def test_streaming_no_chunks(self):
        """Test streaming with no chunks"""
        mock_client = Mock()

        async def mock_stream(*args, **kwargs):
            if False:
                yield {}

        mock_client.create_completion = mock_stream

        result = await _test_streaming_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_streaming_empty_response(self):
        """Test streaming with empty response fields"""
        mock_client = Mock()

        async def mock_stream(*args, **kwargs):
            yield {}
            yield {"response": ""}
            yield {"other": "data"}

        mock_client.create_completion = mock_stream

        result = await _test_streaming_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_streaming_exception(self):
        """Test streaming with exception"""
        mock_client = Mock()

        async def mock_stream(*args, **kwargs):
            raise Exception("Streaming not supported")

        mock_client.create_completion = mock_stream

        result = await _test_streaming_func(mock_client)

        assert result is False


class TestTextCapability:
    """Test the test_text function"""

    @pytest.mark.asyncio
    async def test_text_success_with_response(self):
        """Test successful text capability with response"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "Hello there!"}
        )

        result = await _test_text_func(mock_client)

        assert result is True

    @pytest.mark.asyncio
    async def test_text_success_with_choices(self):
        """Test successful text capability with choices"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hi"}}]}
        )

        result = await _test_text_func(mock_client)

        assert result is True

    @pytest.mark.asyncio
    async def test_text_error_response(self):
        """Test when text returns error"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "Model not available"}
        )

        result = await _test_text_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_text_empty_response(self):
        """Test with empty response"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={}
        )

        result = await _test_text_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_text_none_response(self):
        """Test with None response"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value=None
        )

        result = await _test_text_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_text_exception(self):
        """Test with exception"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("API error")
        )

        result = await _test_text_func(mock_client)

        assert result is False


class TestChatModelCapability:
    """Test the test_chat_model function"""

    @pytest.mark.asyncio
    async def test_chat_model_success(self):
        """Test successful chat model detection"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "test response"}
        )

        result = await _test_chat_model_func(mock_client)

        assert result is True

    @pytest.mark.asyncio
    async def test_chat_model_error_chat_keyword(self):
        """Test error response with chat keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "This is a chat model endpoint"}
        )

        result = await _test_chat_model_func(mock_client)

        # Error with "chat" keyword returns False
        assert result is False

    @pytest.mark.asyncio
    async def test_chat_model_error_completion_keyword(self):
        """Test error response with completion keyword"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "Use completion endpoint instead"}
        )

        result = await _test_chat_model_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_chat_model_error_other(self):
        """Test error response without chat/completion keywords"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "Model not found"}
        )

        result = await _test_chat_model_func(mock_client)

        # Generic error still returns True (assumes chat model)
        assert result is True

    @pytest.mark.asyncio
    async def test_chat_model_exception_not_supported(self):
        """Test exception with 'chat not supported'"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Chat is not supported for this model")
        )

        result = await _test_chat_model_func(mock_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_chat_model_exception_generic(self):
        """Test generic exception"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            side_effect=Exception("Network timeout")
        )

        result = await _test_chat_model_func(mock_client)

        # Generic exceptions assume chat model
        assert result is True


class TestIntegration:
    """Integration tests for capability testing"""

    @pytest.mark.asyncio
    async def test_all_capabilities_on_full_featured_client(self):
        """Test all capabilities on a fully-featured mock client"""
        mock_client = Mock()

        # Mock create_completion for different scenarios
        def mock_completion(*args, **kwargs):
            if kwargs.get("stream"):
                # Return an async generator directly
                async def stream_gen():
                    yield {"response": "chunk"}
                return stream_gen()

            # For non-stream calls, return AsyncMock
            if kwargs.get("tools"):
                async def tools_response():
                    return {"tool_calls": [{"function": {"name": "test"}}]}
                return tools_response()

            async def text_response():
                return {"response": "success"}
            return text_response()

        mock_client.create_completion = mock_completion

        # All basic tests should pass
        assert await _test_text_func(mock_client) is True
        assert await _test_chat_model_func(mock_client) is True
        assert await _test_tools_func(mock_client) is True
        assert await _test_streaming_func(mock_client) is True

    @pytest.mark.asyncio
    async def test_all_capabilities_on_limited_client(self):
        """Test all capabilities on a limited mock client"""
        mock_client = Mock()
        mock_client.create_completion = AsyncMock(
            return_value={"error": "Not supported"}
        )

        # All should fail gracefully
        assert await _test_text_func(mock_client) is False
        assert await _test_tools_func(mock_client) is False
        assert await _test_vision_func(mock_client) is False
        assert await _test_json_mode_func(mock_client) is False
        assert await _test_structured_outputs_func(mock_client) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
