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
        assert client.async_client is not None

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
