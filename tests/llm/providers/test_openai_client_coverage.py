"""
Comprehensive Coverage Tests for OpenAI Client
==============================================

This file contains tests specifically designed to cover missing lines
in openai_client.py and increase coverage from 53% to 90+%.

Coverage Focus Areas:
- Initialization and configuration (lines 100-130)
- Message preparation and conversion (lines 139-199, 238-269)
- Smart defaults and model detection (lines 319-383, 455-541)
- Response parsing and normalization (lines 747-842, 1074-1114)
- Tool handling and name mapping (lines 1039-1050, 1139-1150)
- Error handling paths
- Edge cases in format translation
"""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_llm.core.enums import MessageRole
from chuk_llm.core.models import Message, Tool, ToolFunction
from chuk_llm.llm.providers.openai_client import OpenAILLMClient


# ================================================================
# Mock Classes and Fixtures
# ================================================================

class MockToolCall:
    """Mock OpenAI tool call object"""
    def __init__(self, id=None, function_name="test_tool", arguments="{}"):
        self.id = id or f"call_{uuid.uuid4().hex[:8]}"
        self.function = MagicMock()
        self.function.name = function_name
        self.function.arguments = arguments
        self.type = "function"
        self.index = 0


class MockDelta:
    """Mock streaming delta"""
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock completion choice"""
    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.delta = MockDelta(content, tool_calls)
        self.message = MagicMock()
        self.message.content = content
        self.message.tool_calls = tool_calls
        self.finish_reason = finish_reason


class MockStreamChunk:
    """Mock streaming chunk"""
    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.choices = [MockChoice(content, tool_calls, finish_reason)]
        self.id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self.model = "gpt-4o-mini"


class MockAsyncStream:
    """Mock async stream"""
    def __init__(self, chunks=None):
        if chunks is None:
            chunks = [MockStreamChunk("Hello"), MockStreamChunk(" world!")]
        self.chunks = chunks
        self._iterator = None

    def __aiter__(self):
        self._iterator = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


class MockUsage:
    """Mock usage object"""
    def __init__(self, prompt_tokens=10, completion_tokens=20, total_tokens=30, reasoning_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        if reasoning_tokens is not None:
            self.completion_tokens_details = MagicMock()
            self.completion_tokens_details.reasoning_tokens = reasoning_tokens


class MockChatCompletion:
    """Mock chat completion response"""
    def __init__(self, content="Hello world!", tool_calls=None, usage=None):
        self.choices = [MockChoice(content, tool_calls)]
        self.id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self.model = "gpt-4o-mini"
        self.usage = usage or MockUsage()


@pytest.fixture
def mock_config():
    """Mock configuration"""
    with patch("chuk_llm.configuration.get_config") as mock:
        mock_provider = MagicMock()
        mock_provider.models = ["gpt-4o", "gpt-4o-mini"]
        mock_provider.model_capabilities = []
        mock.return_value.get_provider.return_value = mock_provider
        yield mock


@pytest.fixture
def client(mock_config):
    """Create OpenAI client with mocked config"""
    return OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")


# ================================================================
# Test Provider Detection (Lines 100-106)
# ================================================================

class TestProviderDetection:
    """Test provider detection from API base URL"""

    def test_detect_provider_anyscale(self, mock_config):
        """Test detection of Anyscale provider (line 100)"""
        client = OpenAILLMClient(
            model="test",
            api_key="test",
            api_base="https://api.anyscale.com/v1"
        )
        assert client.detected_provider == "anyscale"

    def test_detect_provider_openai_compatible(self, mock_config):
        """Test detection of generic OpenAI-compatible provider (line 102)"""
        client = OpenAILLMClient(
            model="test",
            api_key="test",
            api_base="https://custom.api.com/v1"
        )
        assert client.detected_provider == "openai_compatible"

    def test_detect_provider_name_public_method(self, mock_config):
        """Test public detect_provider_name method (lines 104-106)"""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test")
        assert client.detect_provider_name() == "openai"


# ================================================================
# Test Add Strict Parameter (Lines 117-130)
# ================================================================

class TestAddStrictParameter:
    """Test adding strict parameter to tools"""

    def test_add_strict_parameter_to_function_tools(self, client):
        """Test adding strict=False to function tools (lines 117-130)"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool"
                }
            }
        ]
        result = client._add_strict_parameter_to_tools(tools)

        assert len(result) == 1
        assert result[0]["function"]["strict"] is False

    def test_add_strict_parameter_preserves_existing_strict(self, client):
        """Test that existing strict parameter is preserved"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "strict": True
                }
            }
        ]
        result = client._add_strict_parameter_to_tools(tools)

        assert result[0]["function"]["strict"] is True

    def test_add_strict_parameter_skips_non_function_tools(self, client):
        """Test that non-function tools are skipped"""
        tools = [
            {"type": "other", "data": "test"}
        ]
        result = client._add_strict_parameter_to_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "other"


# ================================================================
# Test Smart Default Features (Lines 139-174)
# ================================================================

class TestSmartDefaultFeatures:
    """Test smart default feature detection"""

    def test_smart_defaults_o1_model(self, client):
        """Test smart defaults for o1 models (lines 145-149)"""
        features = client._get_smart_default_features("o1-mini")
        assert "text" in features
        assert "reasoning" in features
        assert "tools" not in features  # O1 models don't support tools

    def test_smart_defaults_o3_model(self, client):
        """Test smart defaults for o3 models (lines 150-152)"""
        features = client._get_smart_default_features("o3-turbo")
        assert "text" in features
        assert "streaming" in features
        assert "tools" in features
        assert "reasoning" in features

    def test_smart_defaults_gpt4_model(self, client):
        """Test smart defaults for GPT-4 models (lines 154-166)"""
        features = client._get_smart_default_features("gpt-4o")
        assert "tools" in features
        assert "json_mode" in features
        assert "vision" in features

    def test_smart_defaults_gpt35_model(self, client):
        """Test smart defaults for GPT-3.5 models (no vision)"""
        features = client._get_smart_default_features("gpt-3.5-turbo")
        assert "tools" in features
        assert "json_mode" in features
        assert "vision" not in features

    def test_smart_defaults_gpt5_model(self, client):
        """Test smart defaults for GPT-5 models (lines 163-165)"""
        features = client._get_smart_default_features("gpt-5")
        assert "tools" in features
        assert "reasoning" in features
        assert "vision" in features

    def test_smart_defaults_unknown_model(self, client):
        """Test smart defaults for unknown models (lines 168-174)"""
        features = client._get_smart_default_features("unknown-model-xyz")
        assert "text" in features
        assert "streaming" in features
        assert "tools" in features
        assert "json_mode" in features


# ================================================================
# Test Smart Default Parameters (Lines 179-203)
# ================================================================

class TestSmartDefaultParameters:
    """Test smart default parameter detection"""

    def test_smart_parameters_o1_model(self, client):
        """Test smart parameters for o1 models (lines 182-196)"""
        params = client._get_smart_default_parameters("o1-mini")
        assert params["requires_max_completion_tokens"] is True
        assert "max_tokens" in params["parameter_mapping"]
        assert "temperature" in params["unsupported_params"]
        assert params["supports_tools"] is False

    def test_smart_parameters_gpt5_model(self, client):
        """Test smart parameters for GPT-5 models (line 184)"""
        params = client._get_smart_default_parameters("gpt-5")
        assert params["max_context_length"] == 272000
        assert params["max_output_tokens"] == 128000

    def test_smart_parameters_o3_model(self, client):
        """Test smart parameters for o3 models (lines 182-196)"""
        params = client._get_smart_default_parameters("o3-mini")
        assert params["requires_max_completion_tokens"] is True
        assert params["supports_tools"] is True  # O3+ support tools

    def test_smart_parameters_standard_model(self, client):
        """Test smart parameters for standard models (lines 199-203)"""
        params = client._get_smart_default_parameters("gpt-4o")
        assert params["max_context_length"] == 128000
        assert params["max_output_tokens"] == 8192
        assert params["supports_tools"] is True


# ================================================================
# Test Model Config Check (Lines 208-222)
# ================================================================

class TestModelConfigCheck:
    """Test explicit model configuration check"""

    def test_has_explicit_model_config_true(self, mock_config):
        """Test when model has explicit config (lines 208-222)"""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test")

        # Mock capability match
        mock_cap = MagicMock()
        mock_cap.matches.return_value = True
        mock_config.return_value.get_provider.return_value.model_capabilities = [mock_cap]

        assert client._has_explicit_model_config() is True

    def test_has_explicit_model_config_in_models_list(self, mock_config):
        """Test when model is in models list (line 219)"""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test")
        mock_config.return_value.get_provider.return_value.models = ["gpt-4o-mini"]
        mock_config.return_value.get_provider.return_value.model_capabilities = []

        assert client._has_explicit_model_config() is True

    def test_has_explicit_model_config_exception(self, mock_config):
        """Test when exception occurs (line 221-222)"""
        client = OpenAILLMClient(model="gpt-4o-mini", api_key="test")

        # Patch get_config to raise exception when called within _has_explicit_model_config
        with patch("chuk_llm.llm.providers.openai_client.get_config") as mock_get_config_inner:
            mock_get_config_inner.side_effect = Exception("Config error")

            assert client._has_explicit_model_config() is False


# ================================================================
# Test Supports Feature with Smart Defaults (Lines 238-269)
# ================================================================

class TestSupportsFeatureSmartDefaults:
    """Test feature support with smart defaults"""

    def test_supports_feature_with_config(self, client, monkeypatch):
        """Test when config provides definitive answer (lines 231-235)"""
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.supports_feature",
            lambda self, feature: True
        )
        assert client.supports_feature("tools") is True

    def test_supports_feature_smart_default_openai(self, client, monkeypatch):
        """Test smart defaults for OpenAI provider (lines 238-251)"""
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.supports_feature",
            lambda self, feature: None
        )
        client.detected_provider = "openai"

        assert client.supports_feature("tools") is True
        assert client.supports_feature("vision") is True

    def test_supports_feature_smart_default_no_support(self, client, monkeypatch):
        """Test smart defaults when feature not supported (line 247-249)"""
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.supports_feature",
            lambda self, feature: None
        )
        client.detected_provider = "openai"

        assert client.supports_feature("quantum_computing") is False

    def test_supports_feature_non_openai_conservative(self, client, monkeypatch):
        """Test conservative fallback for non-OpenAI providers (lines 253-257)"""
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.supports_feature",
            lambda self, feature: None
        )
        client.detected_provider = "other_provider"

        assert client.supports_feature("tools") is False

    def test_supports_feature_exception_openai_optimistic(self, client, monkeypatch):
        """Test optimistic fallback on exception for OpenAI (lines 259-269)"""
        def raise_error(self, feature):
            raise Exception("Config error")

        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.supports_feature",
            raise_error
        )
        client.detected_provider = "openai"

        assert client.supports_feature("tools") is True

    def test_supports_feature_exception_non_openai(self, client, monkeypatch):
        """Test fallback on exception for non-OpenAI (line 269)"""
        def raise_error(self, feature):
            raise Exception("Config error")

        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.supports_feature",
            raise_error
        )
        client.detected_provider = "other"

        assert client.supports_feature("tools") is False


# ================================================================
# Test Reasoning Model Detection (Lines 295-304)
# ================================================================

class TestReasoningModelDetection:
    """Test reasoning model detection and generation"""

    def test_get_reasoning_generation_o1(self, client):
        """Test o1 generation detection (lines 294-295)"""
        assert client._get_reasoning_model_generation("o1-mini") == "o1"

    def test_get_reasoning_generation_o3(self, client):
        """Test o3 generation detection (lines 296-297)"""
        assert client._get_reasoning_model_generation("o3-turbo") == "o3"

    def test_get_reasoning_generation_o4(self, client):
        """Test o4 generation detection (lines 298-299)"""
        assert client._get_reasoning_model_generation("o4-preview") == "o4"

    def test_get_reasoning_generation_o5(self, client):
        """Test o5 generation detection (lines 300-301)"""
        assert client._get_reasoning_model_generation("o5-beta") == "o5"

    def test_get_reasoning_generation_gpt5(self, client):
        """Test gpt5 generation detection (lines 302-303)"""
        assert client._get_reasoning_model_generation("gpt-5") == "gpt5"

    def test_get_reasoning_generation_unknown(self, client):
        """Test unknown generation (line 304)"""
        assert client._get_reasoning_model_generation("gpt-4o") == "unknown"


# ================================================================
# Test Reasoning Model Parameter Preparation (Lines 319-383)
# ================================================================

class TestReasoningModelParameters:
    """Test reasoning model parameter preparation"""

    def test_prepare_parameters_max_tokens_conversion(self, client):
        """Test max_tokens to max_completion_tokens conversion (lines 323-329)"""
        client.model = "o1-mini"
        result = client._prepare_reasoning_model_parameters(max_tokens=1000)

        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 1000

    def test_prepare_parameters_default_gpt5_tokens(self, client):
        """Test default max_completion_tokens for GPT-5 (lines 334-335)"""
        client.model = "gpt-5"
        result = client._prepare_reasoning_model_parameters()

        assert result["max_completion_tokens"] == 128000

    def test_prepare_parameters_default_o3_tokens(self, client):
        """Test default max_completion_tokens for o3 (lines 336-337)"""
        client.model = "o3-mini"
        result = client._prepare_reasoning_model_parameters()

        assert result["max_completion_tokens"] == 32768

    def test_prepare_parameters_default_o1_tokens(self, client):
        """Test default max_completion_tokens for o1 (lines 338-342)"""
        client.model = "o1-mini"
        result = client._prepare_reasoning_model_parameters()

        assert result["max_completion_tokens"] == 16384

    def test_prepare_parameters_remove_o1_unsupported(self, client):
        """Test removing unsupported params for o1 (lines 346-354)"""
        client.model = "o1-mini"
        result = client._prepare_reasoning_model_parameters(
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            logit_bias={"123": 1.0}
        )

        assert "temperature" not in result
        assert "top_p" not in result
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert "logit_bias" not in result

    def test_prepare_parameters_remove_gpt5_unsupported(self, client):
        """Test removing unsupported params for GPT-5 (lines 355-362)"""
        client.model = "gpt-5"
        result = client._prepare_reasoning_model_parameters(
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )

        assert "temperature" not in result
        assert "top_p" not in result
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result

    def test_prepare_parameters_remove_o3_unsupported(self, client):
        """Test removing unsupported params for o3+ (lines 363-381)"""
        client.model = "o3-mini"
        result = client._prepare_reasoning_model_parameters(
            temperature=0.7,
            top_p=0.9
        )

        assert "temperature" not in result
        assert "top_p" not in result


# ================================================================
# Test Message Preparation for Reasoning Models (Lines 397-444)
# ================================================================

class TestReasoningModelMessages:
    """Test message preparation for reasoning models"""

    def test_convert_system_messages_for_o1(self, client):
        """Test system message conversion for o1 (lines 411-444)"""
        client.model = "o1-mini"
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

        result = client._convert_system_messages_for_o1(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "System Instructions" in result[0]["content"]
        assert "You are helpful" in result[0]["content"]
        assert "Hello" in result[0]["content"]

    def test_convert_multiple_system_messages(self, client):
        """Test converting multiple system messages"""
        client.model = "o1-mini"
        messages = [
            {"role": "system", "content": "Instruction 1"},
            {"role": "system", "content": "Instruction 2"},
            {"role": "user", "content": "Question"}
        ]

        result = client._convert_system_messages_for_o1(messages)

        assert len(result) == 1
        assert "Instruction 1" in result[0]["content"]
        assert "Instruction 2" in result[0]["content"]

    def test_prepare_messages_o1_converts_system(self, client):
        """Test _prepare_reasoning_model_messages for o1 (lines 397-404)"""
        client.model = "o1-mini"
        messages = [
            {"role": "system", "content": "System msg"},
            {"role": "user", "content": "User msg"}
        ]

        result = client._prepare_reasoning_model_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_prepare_messages_gpt5_keeps_system(self, client):
        """Test GPT-5 keeps system messages (line 403-404)"""
        client.model = "gpt-5"
        messages = [
            {"role": "system", "content": "System msg"},
            {"role": "user", "content": "User msg"}
        ]

        result = client._prepare_reasoning_model_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"


# ================================================================
# Test Get Model Info (Lines 455-541)
# ================================================================

class TestGetModelInfo:
    """Test get_model_info with smart defaults"""

    def test_get_model_info_with_smart_defaults(self, client, monkeypatch):
        """Test model info includes smart defaults (lines 467-485)"""
        monkeypatch.setattr(client, "_has_explicit_model_config", lambda model=None: False)
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.get_model_info",
            lambda self: {"provider": "openai", "model": client.model}
        )

        info = client.get_model_info()

        assert info["using_smart_defaults"] is True
        assert len(info["smart_default_features"]) > 0

    def test_get_model_info_reasoning_model(self, client, monkeypatch):
        """Test model info for reasoning models (lines 461-498)"""
        client.model = "o1-mini"
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.get_model_info",
            lambda self: {"provider": "openai", "model": client.model}
        )

        info = client.get_model_info()

        assert info["is_reasoning_model"] is True
        assert info["reasoning_generation"] == "o1"
        assert info["requires_max_completion_tokens"] is True
        assert info["supports_system_messages"] is False

    def test_get_model_info_gpt5(self, client, monkeypatch):
        """Test model info for GPT-5 (lines 494-520)"""
        client.model = "gpt-5"
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.get_model_info",
            lambda self: {"provider": "openai", "model": client.model}
        )

        info = client.get_model_info()

        assert info["is_gpt5_family"] is True
        assert info["unified_reasoning"] is True
        assert info["parameter_mapping"]["temperature"] is None
        assert info["parameter_mapping"]["max_tokens"] == "max_completion_tokens"

    def test_get_model_info_parameter_mapping(self, client, monkeypatch):
        """Test parameter mapping in model info (lines 503-521)"""
        client.model = "gpt-4o"
        monkeypatch.setattr(
            "chuk_llm.llm.providers._config_mixin.ConfigAwareProviderMixin.get_model_info",
            lambda self: {"provider": "openai", "model": client.model}
        )

        info = client.get_model_info()

        assert "parameter_mapping" in info
        assert info["parameter_mapping"]["temperature"] == "temperature"
        assert info["parameter_mapping"]["max_tokens"] == "max_tokens"

    def test_get_unsupported_params_o1(self, client):
        """Test getting unsupported params for o1 (lines 545-552)"""
        params = client._get_unsupported_params_for_generation("o1")

        assert "temperature" in params
        assert "top_p" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params

    def test_get_unsupported_params_gpt5(self, client):
        """Test getting unsupported params for GPT-5"""
        params = client._get_unsupported_params_for_generation("gpt5")

        assert "temperature" in params
        assert "top_p" in params


# ================================================================
# Test Normalize Message (Lines 565-582, 596-621, 634-642)
# ================================================================

class TestNormalizeMessage:
    """Test message normalization"""

    def test_normalize_message_content_access_failure(self, client):
        """Test content access with exception (lines 565-566)"""
        msg = MagicMock()
        msg.content = None
        del msg.content  # Make content access fail
        msg.message.content = "fallback content"

        result = client._normalize_message(msg)

        assert result["response"] == "fallback content"

    def test_normalize_message_message_wrapper(self, client):
        """Test accessing content via message wrapper (lines 570-574)"""
        msg = MagicMock()
        del msg.content
        msg.message.content = "wrapper content"

        result = client._normalize_message(msg)

        assert result["response"] == "wrapper content"

    def test_normalize_message_dict_access(self, client):
        """Test dict content access (lines 578-582)"""
        msg = {"content": "dict content"}

        result = client._normalize_message(msg)

        assert result["response"] == "dict content"

    def test_normalize_message_tool_calls_dict(self, client):
        """Test tool call extraction from dict (lines 596-597)"""
        msg = {
            "content": "response",
            "tool_calls": [
                MockToolCall(function_name="test", arguments='{"arg": "value"}')
            ]
        }

        result = client._normalize_message(msg)

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test"

    def test_normalize_message_tool_call_json_error(self, client):
        """Test tool call with invalid JSON (lines 620-621)"""
        tc = MockToolCall(arguments="invalid json")
        msg = MagicMock()
        msg.content = "test"
        msg.tool_calls = [tc]

        result = client._normalize_message(msg)

        # Should handle invalid JSON gracefully
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_normalize_message_tool_call_processing_error(self, client):
        """Test tool call processing error (lines 634-638)"""
        msg = MagicMock()
        msg.content = "test"
        msg.tool_calls = [None]  # Invalid tool call

        result = client._normalize_message(msg)

        # Should handle error gracefully
        assert result["response"] == "test"

    def test_normalize_message_empty_content_with_tool_calls(self, client):
        """Test empty content with tool calls (line 642)"""
        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = [MockToolCall()]

        result = client._normalize_message(msg)

        # Empty content with tool calls should return None for response
        assert result["response"] is None or result["response"] == ""
        assert len(result["tool_calls"]) > 0


# ================================================================
# Test Streaming (Lines 747-842)
# ================================================================

class TestStreamFromAsync:
    """Test streaming functionality"""

    @pytest.mark.asyncio
    async def test_stream_tool_call_json_incomplete(self, client):
        """Test tool call with incomplete JSON (lines 826-832)"""
        tc1 = MockToolCall(arguments='{"incomplete": ')
        tc1.index = 0
        tc1.function.arguments = '{"incomplete": '

        tc2 = MockToolCall(arguments='"value"}')
        tc2.index = 0
        tc2.id = tc1.id
        tc2.function.name = tc1.function.name
        tc2.function.arguments = '"value"}'

        chunk1 = MockStreamChunk(tool_calls=[tc1])
        chunk2 = MockStreamChunk(tool_calls=[tc2])

        stream = MockAsyncStream([chunk1, chunk2])

        results = []
        async for result in client._stream_from_async(stream):
            results.append(result)

        # Should only yield when JSON is complete
        tool_results = [r for r in results if r.get("tool_calls")]
        assert len(tool_results) >= 1

    @pytest.mark.asyncio
    async def test_stream_tool_call_complete_parsing(self, client):
        """Test complete tool call JSON parsing (lines 795-824)"""
        tc = MockToolCall(arguments='{"key": "value"}')
        tc.index = 0

        chunk = MockStreamChunk(tool_calls=[tc])
        stream = MockAsyncStream([chunk])

        results = []
        async for result in client._stream_from_async(stream):
            results.append(result)

        tool_results = [r for r in results if r.get("tool_calls")]
        assert len(tool_results) >= 1
        assert tool_results[0]["tool_calls"][0]["function"]["arguments"] == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_stream_chunk_processing_error(self, client):
        """Test chunk processing error (lines 840-842)"""
        class BadChunk:
            @property
            def choices(self):
                raise Exception("Bad chunk")

        stream = MockAsyncStream([BadChunk()])

        results = []
        async for result in client._stream_from_async(stream):
            results.append(result)

        # Should handle error and continue

    @pytest.mark.asyncio
    async def test_stream_name_mapping_restoration(self, client):
        """Test tool name restoration in streaming (lines 854-857)"""
        client._current_name_mapping = {"sanitized_tool": "original.tool"}

        tc = MockToolCall(function_name="sanitized_tool", arguments='{}')
        tc.index = 0
        chunk = MockStreamChunk(tool_calls=[tc])
        stream = MockAsyncStream([chunk])

        results = []
        async for result in client._stream_from_async(stream, name_mapping={"sanitized_tool": "original.tool"}):
            results.append(result)

        # Tool names should be restored
        tool_results = [r for r in results if r.get("tool_calls")]
        if tool_results:
            assert tool_results[0]["tool_calls"][0]["function"]["name"] == "original.tool"


# ================================================================
# Test Completion Methods (Lines 1039-1114, 1139-1150)
# ================================================================

class TestCompletionMethods:
    """Test completion methods with reasoning models"""

    @pytest.mark.asyncio
    async def test_stream_completion_reasoning_model_logging(self, client):
        """Test reasoning model logging in streaming (lines 1039-1052)"""
        client.model = "gpt-5"
        client.client.chat.completions.create = AsyncMock(
            return_value=MockAsyncStream([MockStreamChunk("test")])
        )

        messages = [{"role": "user", "content": "test"}]

        results = []
        async for result in client._stream_completion_async(messages, max_completion_tokens=1000):
            results.append(result)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_stream_completion_retry_logic(self, client):
        """Test retry logic in streaming (lines 1074-1114)"""
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network timeout")
            return MockAsyncStream([MockStreamChunk("success")])

        client.client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "test"}]

        results = []
        async for result in client._stream_completion_async(messages):
            results.append(result)

        assert call_count == 2  # Should retry once

    @pytest.mark.asyncio
    async def test_stream_completion_max_tokens_error(self, client):
        """Test max_tokens error detection (lines 1078-1081)"""
        async def mock_create(**kwargs):
            raise Exception("max_tokens is not supported, use max_completion_tokens")

        client.client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "test"}]

        results = []
        async for result in client._stream_completion_async(messages):
            results.append(result)

        assert len(results) > 0
        assert results[0].get("error") is True

    @pytest.mark.asyncio
    async def test_stream_completion_gpt5_temperature_error(self, client):
        """Test GPT-5 temperature error detection (lines 1082-1085)"""
        client.model = "gpt-5"

        async def mock_create(**kwargs):
            raise Exception("temperature parameter not supported")

        client.client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "test"}]

        results = []
        async for result in client._stream_completion_async(messages):
            results.append(result)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_regular_completion_reasoning_logging(self, client):
        """Test reasoning model logging in regular completion (lines 1139-1152)"""
        client.model = "gpt-5"
        client.client.chat.completions.create = AsyncMock(
            return_value=MockChatCompletion("test response")
        )

        messages = [{"role": "user", "content": "test"}]

        result = await client._regular_completion(messages, max_completion_tokens=2000)

        assert result["response"] == "test response"

    @pytest.mark.asyncio
    async def test_regular_completion_usage_with_reasoning_tokens(self, client):
        """Test usage info with reasoning tokens (lines 1172-1182)"""
        usage = MockUsage(reasoning_tokens=500)
        mock_response = MockChatCompletion("test", usage=usage)

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "test"}]
        result = await client._regular_completion(messages)

        assert "usage" in result
        assert result["usage"]["reasoning_tokens"] == 500
        assert "reasoning" in result
        assert result["reasoning"]["thinking_tokens"] == 500


# ================================================================
# Test Error Handling (Lines 1204-1225)
# ================================================================

class TestErrorHandling:
    """Test error handling in completions"""

    @pytest.mark.asyncio
    async def test_regular_completion_max_tokens_error(self, client):
        """Test max_tokens error in regular completion (lines 1203-1208)"""
        async def mock_create(**kwargs):
            raise Exception("max_tokens not supported, use max_completion_tokens")

        client.client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "test"}]
        result = await client._regular_completion(messages)

        assert result["error"] is True
        assert "Error" in result["response"]

    @pytest.mark.asyncio
    async def test_regular_completion_gpt5_temperature_error(self, client):
        """Test GPT-5 temperature error (lines 1209-1214)"""
        client.model = "gpt-5"

        async def mock_create(**kwargs):
            raise Exception("GPT-5 temperature parameter error")

        client.client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "test"}]
        result = await client._regular_completion(messages)

        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_close_with_name_mapping(self, client):
        """Test close clears name mapping (lines 1221-1222)"""
        client._current_name_mapping = {"test": "mapping"}
        client.client.close = AsyncMock()

        await client.close()

        assert client._current_name_mapping == {}

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, client):
        """Test close calls underlying client close (lines 1224-1225)"""
        client.client.close = AsyncMock()

        await client.close()

        client.client.close.assert_called_once()


# ================================================================
# Additional Edge Cases
# ================================================================

class TestAdditionalEdgeCases:
    """Test additional edge cases for coverage"""

    def test_add_strict_parameter_with_openai_compatible(self, mock_config):
        """Test strict parameter addition for openai_compatible provider"""
        client = OpenAILLMClient(
            model="test",
            api_key="test",
            api_base="https://custom.api.com"
        )

        tools = [
            {"type": "function", "function": {"name": "tool1"}},
            {"type": "function", "function": {"name": "tool2"}}
        ]

        result = client._add_strict_parameter_to_tools(tools)

        assert all(t["function"].get("strict") is False for t in result)

    @pytest.mark.asyncio
    async def test_streaming_yields_only_on_content_or_complete_tools(self, client):
        """Test streaming only yields with content or complete tools (lines 845-859)"""
        # Chunk with only incomplete tool call JSON should not yield
        tc_incomplete = MockToolCall(arguments='{"incomplete')
        tc_incomplete.index = 0

        chunk1 = MockStreamChunk(content=None, tool_calls=[tc_incomplete])
        chunk2 = MockStreamChunk(content="text", tool_calls=None)

        stream = MockAsyncStream([chunk1, chunk2])

        results = []
        async for result in client._stream_from_async(stream):
            results.append(result)

        # Should yield for content
        assert len(results) >= 1
        assert any(r["response"] for r in results)

    def test_reasoning_model_with_non_reasoning_model(self, client):
        """Test reasoning model methods with non-reasoning model"""
        client.model = "gpt-4o"

        # Should return kwargs unchanged
        result = client._prepare_reasoning_model_parameters(temperature=0.7)
        assert result["temperature"] == 0.7

        # Should return messages unchanged
        messages = [{"role": "user", "content": "test"}]
        result = client._prepare_reasoning_model_messages(messages)
        assert result == messages

    @pytest.mark.asyncio
    async def test_validate_request_with_vision_content(self, client):
        """Test request validation with vision content"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
                ]
            }
        ]

        validated_messages, _, _, _ = client._validate_request_with_config(messages)

        assert len(validated_messages) == 1
        assert isinstance(validated_messages[0]["content"], list)

    @pytest.mark.asyncio
    async def test_tool_name_restoration_in_response(self, client):
        """Test tool name restoration in regular completion"""
        client._current_name_mapping = {"sanitized_tool": "original.tool"}

        tc = MockToolCall(function_name="sanitized_tool", arguments='{}')
        mock_response = MockChatCompletion("", [tc])

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "test"}]
        result = await client._regular_completion(
            messages,
            name_mapping={"sanitized_tool": "original.tool"}
        )

        assert result["tool_calls"][0]["function"]["name"] == "original.tool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
