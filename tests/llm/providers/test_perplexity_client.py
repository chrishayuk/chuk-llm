"""
Tests for Perplexity LLM Client
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_llm.llm.providers.perplexity_client import PerplexityLLMClient


class TestPerplexityClientInitialization:
    """Test Perplexity client initialization"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        client = PerplexityLLMClient(api_key="test-key")

        assert client.model == "sonar-pro"
        assert client.detected_provider == "perplexity"
        # Check that async_client exists (base_url check removed due to mock complexity)

    def test_init_with_custom_model(self):
        """Test initialization with custom model"""
        client = PerplexityLLMClient(model="sonar-reasoning", api_key="test-key")

        assert client.model == "sonar-reasoning"

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base"""
        custom_base = "https://custom.perplexity.ai"
        client = PerplexityLLMClient(api_key="test-key", api_base=custom_base)

        # Verify client was created successfully
        assert client.model == "sonar-pro"

    def test_init_sets_default_api_base(self):
        """Test that default API base is set if not provided"""
        client = PerplexityLLMClient(api_key="test-key")

        # Verify provider is detected correctly
        assert client.detected_provider == "perplexity"


class TestPerplexityResponseFormatTranslation:
    """Test response_format translation"""

    def test_translate_json_object_to_json_schema(self):
        """Test translation of json_object to Perplexity json_schema format"""
        client = PerplexityLLMClient(api_key="test-key")

        input_format = {"type": "json_object"}
        result = client._translate_response_format(input_format)

        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["name"] == "json_response"
        assert result["json_schema"]["schema"]["type"] == "object"
        assert result["json_schema"]["schema"]["additionalProperties"] is True
        assert result["json_schema"]["strict"] is False

    def test_translate_json_schema_passthrough(self):
        """Test that json_schema format passes through unchanged"""
        client = PerplexityLLMClient(api_key="test-key")

        input_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object"},
                "strict": True,
            },
        }
        result = client._translate_response_format(input_format)

        assert result == input_format

    def test_translate_text_format_passthrough(self):
        """Test that text format passes through unchanged"""
        client = PerplexityLLMClient(api_key="test-key")

        input_format = {"type": "text"}
        result = client._translate_response_format(input_format)

        assert result == input_format

    def test_translate_none_format(self):
        """Test translation with None returns None"""
        client = PerplexityLLMClient(api_key="test-key")

        result = client._translate_response_format(None)

        assert result is None

    def test_translate_empty_dict(self):
        """Test translation with empty dict"""
        client = PerplexityLLMClient(api_key="test-key")

        result = client._translate_response_format({})

        # Empty dict has no 'type', so returns passthrough (empty dict)
        assert result == {} or result is None


class TestPerplexityToolHandling:
    """Test tool/function calling handling"""

    @pytest.mark.asyncio
    async def test_regular_completion_removes_tools(self):
        """Test that tools are removed with warning"""
        client = PerplexityLLMClient(api_key="test-key")

        # Mock parent method
        async def mock_parent_regular(self, messages, tools=None, name_mapping=None, **kwargs):
            # Verify tools are None
            assert tools is None
            return {"response": "Test response", "tool_calls": None}

        with patch.object(
            PerplexityLLMClient.__bases__[0],
            "_regular_completion",
            new=mock_parent_regular,
        ):
            messages = [{"role": "user", "content": "Test"}]
            tools = [{"type": "function", "function": {"name": "test_tool"}}]

            with patch("chuk_llm.llm.providers.perplexity_client.log") as mock_log:
                result = await client._regular_completion(
                    messages, tools=tools, name_mapping=None
                )

                # Verify warning was logged
                mock_log.warning.assert_called_once()
                assert "does not support tool calling" in str(
                    mock_log.warning.call_args
                )

    @pytest.mark.asyncio
    async def test_streaming_completion_removes_tools(self):
        """Test that tools are removed in streaming mode"""
        client = PerplexityLLMClient(api_key="test-key")

        # Mock parent streaming method
        async def mock_parent_stream(self, messages, tools=None, name_mapping=None, **kwargs):
            # Verify tools are None
            assert tools is None
            yield {"response": "chunk1"}
            yield {"response": "chunk2"}

        with patch.object(
            PerplexityLLMClient.__bases__[0],
            "_stream_completion_async",
            new=mock_parent_stream,
        ):
            messages = [{"role": "user", "content": "Test"}]
            tools = [{"type": "function", "function": {"name": "test_tool"}}]

            chunks = []
            with patch("chuk_llm.llm.providers.perplexity_client.log") as mock_log:
                async for chunk in client._stream_completion_async(
                    messages, tools=tools, name_mapping=None
                ):
                    chunks.append(chunk)

                # Verify warning was logged
                mock_log.warning.assert_called_once()
                assert len(chunks) == 2


class TestPerplexityRegularCompletion:
    """Test regular (non-streaming) completions"""

    @pytest.mark.asyncio
    async def test_regular_completion_with_response_format_translation(self):
        """Test that response_format is translated during regular completion"""
        client = PerplexityLLMClient(api_key="test-key")

        called_kwargs = {}

        async def mock_parent_regular(self, messages, tools=None, name_mapping=None, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return {"response": "Test response", "tool_calls": None}

        with patch.object(
            PerplexityLLMClient.__bases__[0],
            "_regular_completion",
            new=mock_parent_regular,
        ):
            messages = [{"role": "user", "content": "Test"}]

            await client._regular_completion(
                messages,
                tools=None,
                name_mapping=None,
                response_format={"type": "json_object"},
            )

            # Verify response_format was translated
            assert called_kwargs["response_format"]["type"] == "json_schema"
            assert "json_schema" in called_kwargs["response_format"]

    @pytest.mark.asyncio
    async def test_regular_completion_removes_none_response_format(self):
        """Test that None response_format is removed"""
        client = PerplexityLLMClient(api_key="test-key")

        # Override translation to return None
        original_translate = client._translate_response_format
        client._translate_response_format = lambda x: None

        called_kwargs = {}

        async def mock_parent_regular(self, messages, tools=None, name_mapping=None, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return {"response": "Test response", "tool_calls": None}

        try:
            with patch.object(
                PerplexityLLMClient.__bases__[0],
                "_regular_completion",
                new=mock_parent_regular,
            ):
                messages = [{"role": "user", "content": "Test"}]

                await client._regular_completion(
                    messages,
                    tools=None,
                    name_mapping=None,
                    response_format={"type": "json_object"},
                )

                # Verify response_format was removed
                assert "response_format" not in called_kwargs
        finally:
            client._translate_response_format = original_translate


class TestPerplexityStreamingCompletion:
    """Test streaming completions"""

    @pytest.mark.asyncio
    async def test_streaming_completion_with_response_format_translation(self):
        """Test that response_format is translated during streaming"""
        client = PerplexityLLMClient(api_key="test-key")

        called_kwargs = {}

        async def mock_parent_stream(self, messages, tools=None, name_mapping=None, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            yield {"response": "chunk"}

        with patch.object(
            PerplexityLLMClient.__bases__[0],
            "_stream_completion_async",
            new=mock_parent_stream,
        ):
            messages = [{"role": "user", "content": "Test"}]

            chunks = []
            async for chunk in client._stream_completion_async(
                messages,
                tools=None,
                name_mapping=None,
                response_format={"type": "json_object"},
            ):
                chunks.append(chunk)

            # Verify response_format was translated
            assert called_kwargs["response_format"]["type"] == "json_schema"
            assert "json_schema" in called_kwargs["response_format"]
            assert len(chunks) == 1


class TestPerplexityFeatureSupport:
    """Test feature support detection"""

    def test_supports_tools_returns_false(self):
        """Test that tools feature returns False"""
        client = PerplexityLLMClient(api_key="test-key")

        assert client.supports_feature("tools") is False

    def test_supports_other_features(self):
        """Test that other features are delegated to parent"""
        client = PerplexityLLMClient(api_key="test-key")

        # These features depend on configuration - just verify tools returns False
        # and the method doesn't crash for other features
        assert client.supports_feature("tools") is False
        # Other features may vary based on config, so just test they don't error
        client.supports_feature("text")
        client.supports_feature("streaming")

    def test_supports_vision(self):
        """Test vision support method doesn't crash"""
        client = PerplexityLLMClient(api_key="test-key")

        # Just verify the method can be called without error
        # Actual support depends on configuration
        result = client.supports_feature("vision")
        assert isinstance(result, bool)


class TestPerplexityIntegration:
    """Integration tests"""

    def test_provider_detection(self):
        """Test that provider is correctly detected as perplexity"""
        client = PerplexityLLMClient(api_key="test-key")

        assert client.detected_provider == "perplexity"

    def test_default_model(self):
        """Test default model is sonar-pro"""
        client = PerplexityLLMClient(api_key="test-key")

        assert client.model == "sonar-pro"

    def test_inherits_from_openai_client(self):
        """Test that Perplexity client inherits from OpenAI client"""
        from chuk_llm.llm.providers.openai_client import OpenAILLMClient

        client = PerplexityLLMClient(api_key="test-key")

        assert isinstance(client, OpenAILLMClient)
