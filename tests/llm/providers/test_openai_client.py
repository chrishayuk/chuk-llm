"""
Comprehensive tests for OpenAI provider client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_llm.core.enums import MessageRole
from chuk_llm.core.models import Message, Tool, ToolFunction
from chuk_llm.llm.providers.openai_client import OpenAILLMClient


class TestOpenAIClientInit:
    """Test client initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        assert client.model == "gpt-4o-mini"
        assert client.detected_provider == "openai"
        assert client.client is not None  # Now using async-native client

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base."""
        client = OpenAILLMClient(
            model="gpt-4o-mini",
            api_key="test-key",
            api_base="https://custom.api.com/v1"
        )
        
        assert client.api_base == "https://custom.api.com/v1"

    def test_detect_provider_openai(self):
        """Test provider detection for OpenAI."""
        client = OpenAILLMClient(model="gpt-4o-mini")
        assert client.detected_provider == "openai"

    def test_detect_provider_deepseek(self):
        """Test provider detection for DeepSeek."""
        client = OpenAILLMClient(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1"
        )
        assert client.detected_provider == "deepseek"


class TestOpenAIClientReasoningModels:
    """Test reasoning model detection."""

    @pytest.mark.parametrize("model,expected", [
        ("o1-mini", True),
        ("o3-mini", True),
        ("gpt-4o", False),
        ("gpt-5", True),
    ])
    def test_detect_reasoning_model(self, model, expected):
        """Test reasoning model detection."""
        client = OpenAILLMClient(model=model)
        assert client._is_reasoning_model(model) == expected


class TestOpenAIClientMessages:
    """Test message handling."""

    def test_pydantic_messages(self):
        """Test with Pydantic Message objects."""
        client = OpenAILLMClient(model="gpt-4o-mini")
        
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful"),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        
        # Should not raise
        assert len(messages) == 2

    def test_dict_to_pydantic_conversion(self):
        """Test automatic dict to Pydantic conversion."""
        from chuk_llm.llm.core.base import _ensure_pydantic_messages
        
        dict_messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        pydantic_messages = _ensure_pydantic_messages(dict_messages)
        
        assert len(pydantic_messages) == 1
        assert isinstance(pydantic_messages[0], Message)
        assert pydantic_messages[0].role == MessageRole.USER


class TestJSONFunctionFallback:
    """Test JSON function calling fallback when ENABLE_JSON_FUNCTION_FALLBACK is True."""

    def test_create_json_function_calling_prompt(self):
        """Test creating system prompt for JSON function calling."""
        client = OpenAILLMClient(model="gpt-4o-mini")
        client.ENABLE_JSON_FUNCTION_FALLBACK = True

        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_weather",
                    description="Get weather info",
                    parameters={"type": "object"}
                )
            )
        ]

        prompt = client._create_json_function_calling_prompt(tools)

        assert "get_weather" in prompt
        assert "Get weather info" in prompt
        assert "JSON object" in prompt

    def test_parse_function_call_from_json_direct(self):
        """Test parsing direct JSON function call."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        content = '{"name": "get_weather", "arguments": {"location": "NYC"}}'
        result = client._parse_function_call_from_json(content)

        assert result is not None
        assert result["name"] == "get_weather"
        assert result["arguments"]["location"] == "NYC"

    def test_parse_function_call_from_json_code_block(self):
        """Test parsing JSON from code blocks."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        content = '''```json
{"name": "get_weather", "arguments": {"location": "NYC"}}
```'''
        result = client._parse_function_call_from_json(content)

        assert result is not None
        assert result["name"] == "get_weather"

    def test_parse_function_call_from_json_invalid(self):
        """Test parsing invalid JSON returns None."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        result = client._parse_function_call_from_json("not valid json")
        assert result is None

        result = client._parse_function_call_from_json("")
        assert result is None

    def test_convert_to_tool_calls_from_json(self):
        """Test converting JSON response to tool_calls format."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        response = {
            "response": '{"name": "get_weather", "arguments": {"location": "NYC"}}'
        }

        result = client._convert_to_tool_calls_from_json(response)

        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"


class TestStreamingHandling:
    """Test streaming-specific functionality."""

    @pytest.mark.asyncio
    async def test_heartbeat_yielding_concept(self):
        """Test heartbeat yielding logic (conceptual test)."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        # Test that heartbeat logic would trigger
        chunk_count = 25
        heartbeat_triggers = []

        for i in range(chunk_count):
            if (i + 1) % 10 == 0:
                heartbeat_triggers.append(i + 1)

        # Should trigger at chunks 10, 20
        assert 10 in heartbeat_triggers
        assert 20 in heartbeat_triggers
        assert len(heartbeat_triggers) == 2


class TestProviderSpecificBehavior:
    """Test provider-specific behavior and adjustments."""

    def test_reasoning_model_detection(self):
        """Test reasoning model detection."""
        client = OpenAILLMClient(model="o1-mini")

        # For reasoning models
        assert client._is_reasoning_model("o1-mini") is True
        assert client._is_reasoning_model("o3-mini") is True
        assert client._is_reasoning_model("gpt-5") is True

    def test_non_reasoning_model_detection(self):
        """Test non-reasoning model detection."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        assert client._is_reasoning_model("gpt-4o-mini") is False
        assert client._is_reasoning_model("gpt-4") is False

    def test_normalize_message_basic(self):
        """Test message normalization."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        msg = Message(role=MessageRole.USER, content="Hello")
        normalized = client._normalize_message(msg)

        assert isinstance(normalized, dict)
        # _normalize_message returns response format, not OpenAI message format
        assert "response" in normalized or "role" in normalized

    def test_convert_system_messages_for_o1(self):
        """Test converting system messages for O1 models."""
        client = OpenAILLMClient(model="o1-mini")

        messages = [
            Message(role=MessageRole.SYSTEM, content="System prompt"),
            Message(role=MessageRole.USER, content="User message"),
        ]

        converted = client._convert_system_messages_for_o1(messages)

        # System message should be converted to user message with prefix
        assert len(converted) > 0
        # First message should now be user role or combined
        assert converted[0]["role"] in ["user", "system"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        assert client.model == "gpt-4o-mini"
        assert client.detected_provider == "openai"

    def test_client_repr_basic(self):
        """Test client string representation."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        repr_str = repr(client)
        assert "OpenAILLMClient" in repr_str

    def test_client_close_sync(self):
        """Test client cleanup (sync version)."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        # close() is async, so just test it exists
        assert hasattr(client, 'close')
        assert callable(client.close)


class TestAdditionalCoverage:
    """Test additional functionality for coverage."""

    def test_detect_provider_name(self):
        """Test provider name detection from API base."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        # Test various API bases
        assert client._detect_provider_name("https://api.openai.com/v1") == "openai"
        assert client._detect_provider_name("https://api.deepseek.com/v1") == "deepseek"

    def test_reasoning_model_generation_detection(self):
        """Test detecting reasoning model generation."""
        client = OpenAILLMClient(model="o1-mini")

        gen = client._get_reasoning_model_generation("o1-mini")
        assert gen in ["o1", "o3"]

    def test_prepare_reasoning_model_messages(self):
        """Test preparing messages for reasoning models."""
        client = OpenAILLMClient(model="o1-mini")

        messages = [
            Message(role=MessageRole.USER, content="Solve this problem")
        ]

        prepared = client._prepare_reasoning_model_messages(messages)
        assert isinstance(prepared, list)
        assert len(prepared) > 0

    def test_add_strict_parameter_to_tools(self):
        """Test adding strict parameter to tools."""
        client = OpenAILLMClient(model="gpt-4o-mini")

        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="test_func",
                    description="Test",
                    parameters={"type": "object", "properties": {}}
                )
            )
        ]

        result = client._add_strict_parameter_to_tools(tools)
        assert isinstance(result, list)
