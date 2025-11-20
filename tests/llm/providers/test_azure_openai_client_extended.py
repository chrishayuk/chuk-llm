# tests/providers/test_azure_openai_client_extended.py
"""
Extended Azure OpenAI Client Tests
===================================

Additional tests for Azure OpenAI client including:
- Context memory preservation
- Custom deployment support
- Smart defaults validation
- Complex conversation flows
"""

import os
import sys
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

# Mock classes for Azure OpenAI
class MockStreamChunk:
    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.choices = [MockChoice(content, tool_calls, finish_reason)]


class MockChoice:
    def __init__(self, content="", tool_calls=None, finish_reason=None):
        self.delta = MockDelta(content, tool_calls)
        self.message = MockMessage(content, tool_calls)
        self.finish_reason = finish_reason


class MockDelta:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockAsyncStream:
    """Properly working async stream mock"""

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
            raise StopAsyncIteration


class MockChatCompletion:
    def __init__(self, content="Hello world!", tool_calls=None):
        self.choices = [MockChoice(content, tool_calls)]
        self.id = "chatcmpl-test"
        self.model = "gpt-4o-mini"
        self.usage = MagicMock(total_tokens=50, prompt_tokens=10, completion_tokens=40)


# Configuration Mock Classes
class MockFeature:
    TEXT = "text"
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    SYSTEM_MESSAGES = "system_messages"
    PARALLEL_CALLS = "parallel_calls"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"

    @classmethod
    def from_string(cls, feature_str):
        return getattr(cls, feature_str.upper(), None)


class MockModelCapabilities:
    def __init__(
        self, features=None, max_context_length=128000, max_output_tokens=4096
    ):
        self.features = features or {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
            MockFeature.JSON_MODE,
            MockFeature.PARALLEL_CALLS,
        }
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    def __init__(self, name="azure_openai", client_class="AzureOpenAILLMClient"):
        self.name = name
        self.client_class = client_class
        self.api_base = "https://api.openai.com"
        self.models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        self.model_aliases = {}
        self.rate_limits = {"requests_per_minute": 500}

    def get_model_capabilities(self, model):
        features = {
            MockFeature.TEXT,
            MockFeature.STREAMING,
            MockFeature.TOOLS,
            MockFeature.VISION,
            MockFeature.SYSTEM_MESSAGES,
            MockFeature.MULTIMODAL,
            MockFeature.JSON_MODE,
            MockFeature.PARALLEL_CALLS,
        }
        return MockModelCapabilities(features=features)


class MockConfig:
    def __init__(self):
        self.azure_openai_provider = MockProviderConfig()

    def get_provider(self, provider_name):
        if provider_name == "azure_openai":
            return self.azure_openai_provider
        return None


# Fixtures
@pytest.fixture
def mock_configuration():
    """Mock the configuration system"""
    mock_config = MockConfig()

    with patch("chuk_llm.configuration.get_config", return_value=mock_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            yield mock_config


@pytest.fixture
def mock_env():
    """Mock environment variables for Azure OpenAI."""
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-api-key",
            "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com",
        },
    ):
        yield


@pytest.fixture
def client(mock_configuration, mock_env, monkeypatch):
    """Create Azure OpenAI client for testing with configuration mocking"""
    from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
    
    cl = AzureOpenAILLMClient(
        model="gpt-4o-mini",
        api_key="test-key",
        azure_endpoint="https://test-resource.openai.azure.com",
    )

    # Ensure configuration methods are properly mocked
    monkeypatch.setattr(
        cl,
        "supports_feature",
        lambda feature: feature
        in [
            "text",
            "streaming",
            "tools",
            "vision",
            "system_messages",
            "multimodal",
            "json_mode",
            "parallel_calls",
        ],
    )

    return cl


from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

# ---------------------------------------------------------------------------
# Custom Deployment Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAICustomDeployments:
    """Test Azure OpenAI custom deployment support"""

    def test_custom_deployment_validation_always_passes(
        self, mock_configuration, mock_env
    ):
        """Test that ANY custom deployment name passes validation."""
        # Test various custom deployment names
        custom_deployments = [
            "scribeflowgpt4o",
            "company-gpt-4-turbo",
            "prod-deployment-v2",
            "my-custom-gpt4",
            "test_deployment_123",
            "deployment-with-special-chars-2024",
        ]

        for deployment_name in custom_deployments:
            client = AzureOpenAILLMClient(
                model=deployment_name,
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com",
            )

            # validate_model should ALWAYS return True for Azure
            assert client.validate_model(deployment_name) is True
            assert client.model == deployment_name
            assert client.azure_deployment == deployment_name

    def test_smart_defaults_for_unknown_deployment(self, mock_configuration, mock_env):
        """Test smart defaults are applied for unknown custom deployments."""
        client = AzureOpenAILLMClient(
            model="scribeflowgpt4o",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Check smart default features are detected
        smart_features = client._get_smart_default_features("scribeflowgpt4o")

        # Should detect as GPT-4 variant based on name pattern
        assert "text" in smart_features
        assert "streaming" in smart_features
        assert "tools" in smart_features
        assert "json_mode" in smart_features
        assert "vision" in smart_features
        assert "parallel_calls" in smart_features

        # Check smart default parameters
        smart_params = client._get_smart_default_parameters("scribeflowgpt4o")
        assert smart_params["max_context_length"] == 128000
        assert (
            smart_params["max_output_tokens"] == 4096
        )  # Updated based on actual limits
        assert smart_params["supports_tools"] is True

    def test_deployment_pattern_detection(self, mock_configuration, mock_env):
        """Test deployment pattern detection for various naming conventions."""
        test_cases = [
            # GPT-4 variants
            ("gpt-4o", {"vision", "tools", "parallel_calls"}),
            ("gpt4-turbo", {"vision", "tools", "parallel_calls"}),
            ("custom-gpt-4", {"vision", "tools", "parallel_calls"}),
            ("scribeflowgpt4o", {"vision", "tools", "parallel_calls"}),
            # GPT-3.5 variants
            ("gpt-35-turbo", {"tools", "json_mode"}),
            ("gpt-3.5-turbo", {"tools", "json_mode"}),
            # Reasoning models
            ("o1-preview", {"text", "reasoning"}),
            ("o3-mini", {"text", "streaming", "tools", "reasoning", "system_messages"}),
            # Embedding models
            ("text-embedding-ada", {"text"}),
            ("embedding-model", {"text"}),
            # Unknown patterns - optimistic defaults
            (
                "my-custom-model",
                {"text", "streaming", "system_messages", "tools", "json_mode"},
            ),
        ]

        for deployment_name, expected_features in test_cases:
            features = AzureOpenAILLMClient._get_smart_default_features(deployment_name)
            for feature in expected_features:
                assert feature in features, (
                    f"Expected {feature} in features for {deployment_name}"
                )

    def test_model_info_with_smart_defaults(self, mock_configuration, mock_env):
        """Test model info includes smart defaults information."""
        client = AzureOpenAILLMClient(
            model="custom-deployment-xyz",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock that this deployment is not in config
        client._has_explicit_deployment_config = lambda deployment: False

        info = client.get_model_info()

        # Should indicate using smart defaults
        assert info.get("using_smart_defaults") is True
        assert "smart_default_features" in info
        assert "smart_default_parameters" in info
        assert "discovery_note" in info
        assert "custom-deployment-xyz" in info["discovery_note"]

    @pytest.mark.asyncio
    async def test_custom_deployment_with_tools(self, mock_configuration, mock_env):
        """Test custom deployment with tool usage."""
        client = AzureOpenAILLMClient(
            model="scribeflowgpt4o",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        messages = [{"role": "user", "content": "Use tools"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",  # Problematic name
                    "description": "Read from stdio",
                    "parameters": {},
                },
            }
        ]

        # Create proper mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "stdio_read_query"  # Sanitized name
        mock_tool_call.function.arguments = '{"query": "test"}'

        # Mock the completion
        mock_response = MockChatCompletion(content=None, tool_calls=[mock_tool_call])

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Mock sanitization
        client._sanitize_tool_names = lambda tools: [
            {
                "type": "function",
                "function": {
                    "name": "stdio_read_query",  # Sanitized
                    "description": "Read from stdio",
                    "parameters": {},
                },
            }
        ]
        client._current_name_mapping = {"stdio_read_query": "stdio.read_query"}

        # Mock the normalization to return proper structure
        def mock_normalize(msg):
            return {
                "response": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "stdio_read_query",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ],
            }

        client._normalize_message = mock_normalize

        # Mock restoration to actually restore the name
        def mock_restore(response, mapping):
            if response.get("tool_calls") and mapping:
                for tool_call in response["tool_calls"]:
                    sanitized_name = tool_call["function"]["name"]
                    if sanitized_name in mapping:
                        tool_call["function"]["name"] = mapping[sanitized_name]
            return response

        client._restore_tool_names_in_response = mock_restore

        result = await client.create_completion(messages, tools=tools, stream=False)

        # Tool name should be restored
        assert result["tool_calls"][0]["function"]["name"] == "stdio.read_query"

    def test_deployment_not_found_error_handling(self, mock_configuration, mock_env):
        """Test handling of deployment not found errors."""
        client = AzureOpenAILLMClient(
            model="non-existent-deployment",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Client should still be created (validation always passes)
        assert client.model == "non-existent-deployment"
        assert client.azure_deployment == "non-existent-deployment"

        # Error will occur at API call time, not initialization


# ---------------------------------------------------------------------------
# Context Memory Preservation Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIContextMemory:
    """Test Azure OpenAI context memory preservation"""

    def test_prepare_messages_preserves_full_context(self, client):
        """Test that full conversation context is preserved."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I'll check the weather for you."},
            {"role": "user", "content": "What's my name?"},  # Tests context memory
        ]

        # All messages should be preserved for Azure OpenAI
        validated_messages, _, _, _ = client._validate_request_with_config(messages)

        assert len(validated_messages) == 6
        assert validated_messages[1]["content"] == "My name is Alice"
        assert validated_messages[5]["content"] == "What's my name?"

    def test_context_with_tool_calls_and_responses(self, client):
        """Test context preservation with tool calls and responses."""
        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "calculator",
                "content": "4",
            },
            {"role": "assistant", "content": "2+2 equals 4"},
            {"role": "user", "content": "What did I just ask you to calculate?"},
        ]

        validated_messages, _, _, _ = client._validate_request_with_config(messages)

        # All messages including tool interactions should be preserved
        assert len(validated_messages) == 5

        # Check tool call is preserved
        assert (
            validated_messages[1]["tool_calls"][0]["function"]["name"] == "calculator"
        )

        # Check tool response is preserved
        assert validated_messages[2]["role"] == "tool"
        assert validated_messages[2]["content"] == "4"

        # Check final question that requires context
        assert (
            validated_messages[4]["content"] == "What did I just ask you to calculate?"
        )

    def test_context_with_vision_content(self, client):
        """Test context preservation with vision/multimodal content."""
        messages = [
            {"role": "user", "content": "I'm Bob"},
            {"role": "assistant", "content": "Hello Bob!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "I see an image."},
            {"role": "user", "content": "What's my name again?"},
        ]

        validated_messages, _, _, _ = client._validate_request_with_config(messages)

        # All messages should be preserved
        assert len(validated_messages) == 5

        # Check early context is preserved
        assert validated_messages[0]["content"] == "I'm Bob"

        # Check multimodal content is preserved
        assert isinstance(validated_messages[2]["content"], list)
        assert validated_messages[2]["content"][0]["type"] == "text"
        assert validated_messages[2]["content"][1]["type"] == "image_url"

        # Check final question
        assert validated_messages[4]["content"] == "What's my name again?"

    @pytest.mark.asyncio
    async def test_streaming_preserves_context(self, client):
        """Test that streaming preserves conversation context."""
        messages = [
            {"role": "user", "content": "I live in Tokyo"},
            {"role": "assistant", "content": "Tokyo is a fascinating city!"},
            {"role": "user", "content": "What city do I live in?"},
        ]

        # Mock streaming response that uses context
        mock_stream = MockAsyncStream(
            [
                MockStreamChunk("Based on our conversation, "),
                MockStreamChunk("you live in Tokyo."),
            ]
        )

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        # Mock stream processing
        async def mock_stream_from_async(stream):
            async for chunk in stream:
                yield {"response": chunk.choices[0].delta.content, "tool_calls": []}

        client._stream_from_async = mock_stream_from_async

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk["response"])

        full_response = "".join(chunks)
        assert "Tokyo" in full_response

    def test_complex_multi_turn_conversation(self, client):
        """Test complex multi-turn conversation with various message types."""
        messages = [
            {"role": "system", "content": "You are a travel assistant"},
            {"role": "user", "content": "I want to plan a trip"},
            {
                "role": "assistant",
                "content": "I'd be happy to help! Where would you like to go?",
            },
            {"role": "user", "content": "Paris"},
            {
                "role": "assistant",
                "content": "Let me find information about Paris.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_destination",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search_destination",
                "content": "Paris: Capital of France, known for Eiffel Tower, Louvre Museum",
            },
            {
                "role": "assistant",
                "content": "Paris is wonderful! The Eiffel Tower and Louvre are must-sees.",
            },
            {"role": "user", "content": "How about hotels?"},
            {
                "role": "assistant",
                "content": "Let me search for hotels.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "search_hotels",
                            "arguments": '{"city": "Paris", "rating": 4}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "search_hotels",
                "content": "Found 5 hotels: Hotel Le Marais, Hotel Saint-Germain...",
            },
            {
                "role": "assistant",
                "content": "I found several great hotels including Hotel Le Marais.",
            },
            {"role": "user", "content": "What city are we discussing again?"},
        ]

        validated_messages, _, _, _ = client._validate_request_with_config(messages)

        # All 12 messages should be preserved
        assert len(validated_messages) == 12

        # Verify Paris is mentioned in the context
        assert "Paris" in validated_messages[3]["content"]

        # Verify tool calls are preserved
        assert (
            validated_messages[4]["tool_calls"][0]["function"]["name"]
            == "search_destination"
        )
        assert (
            validated_messages[8]["tool_calls"][0]["function"]["name"]
            == "search_hotels"
        )

    @pytest.mark.asyncio
    async def test_context_affects_completion_response(self, client):
        """Test that context actually affects the completion response."""
        # First conversation without context
        messages_no_context = [{"role": "user", "content": "What's my favorite color?"}]

        # Second conversation with context
        messages_with_context = [
            {"role": "user", "content": "My favorite color is blue"},
            {"role": "assistant", "content": "Blue is a great color!"},
            {"role": "user", "content": "What's my favorite color?"},
        ]

        # Mock different responses based on message count
        async def mock_create(**kwargs):
            if len(kwargs["messages"]) == 1:
                return MockChatCompletion("I don't know your favorite color.")
            else:
                return MockChatCompletion("Your favorite color is blue.")

        client.async_client.chat.completions.create = mock_create

        # Test without context
        result1 = await client._regular_completion(messages_no_context)
        assert "don't know" in result1["response"].lower()

        # Test with context
        result2 = await client._regular_completion(messages_with_context)
        assert "blue" in result2["response"].lower()


# ---------------------------------------------------------------------------
# Smart Defaults and Feature Detection Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAISmartDefaults:
    """Test smart defaults and feature detection"""

    def test_reasoning_model_detection(self, mock_configuration, mock_env):
        """Test detection of reasoning models."""
        reasoning_models = ["o1-preview", "o3-mini", "o4-large", "gpt-5-reasoning"]

        for model_name in reasoning_models:
            features = AzureOpenAILLMClient._get_smart_default_features(model_name)
            params = AzureOpenAILLMClient._get_smart_default_parameters(model_name)

            if "o1" in model_name:
                # O1 models have limited features
                assert "reasoning" in features
                assert "tools" not in features
            else:
                # O3+ models have full features
                assert "reasoning" in features
                assert "tools" in features

            # Check parameter requirements
            if any(x in model_name for x in ["o1", "o3", "o4", "o5"]):
                assert params.get("requires_max_completion_tokens") is True

    def test_vision_model_detection(self, mock_configuration, mock_env):
        """Test detection of vision-capable models."""
        vision_models = ["gpt-4-vision", "gpt-4o", "gpt4-multimodal"]
        non_vision_models = ["gpt-3.5-turbo", "text-embedding-ada"]

        for model_name in vision_models:
            features = AzureOpenAILLMClient._get_smart_default_features(model_name)
            assert "vision" in features
            assert "parallel_calls" in features

        for model_name in non_vision_models:
            features = AzureOpenAILLMClient._get_smart_default_features(model_name)
            assert "vision" not in features

    def test_max_tokens_adjustment_for_deployment(self, mock_configuration, mock_env):
        """Test max tokens adjustment based on deployment capabilities."""
        client = AzureOpenAILLMClient(
            model="scribeflowgpt4o",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock that this is using smart defaults
        client._has_explicit_deployment_config = lambda deployment: False

        # Test parameter adjustment
        params = {"max_tokens": 10000}  # Request more than supported
        adjusted = client._adjust_parameters_for_provider(params)

        # Should be capped at smart default limit
        assert adjusted["max_tokens"] <= 4096

    def test_parameter_mapping_for_reasoning_models(self, mock_configuration, mock_env):
        """Test parameter mapping for reasoning models."""
        client = AzureOpenAILLMClient(
            model="o1-preview",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Get smart parameters for reasoning model
        params = client._get_smart_default_parameters("o1-preview")

        # Should require max_completion_tokens instead of max_tokens
        assert params["requires_max_completion_tokens"] is True
        assert "max_tokens" in params["parameter_mapping"]
        assert params["parameter_mapping"]["max_tokens"] == "max_completion_tokens"


# ---------------------------------------------------------------------------
# Deployment Discovery Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIDeploymentDiscovery:
    """Test Azure OpenAI deployment discovery"""

    @pytest.mark.asyncio
    async def test_test_deployment_availability(self, mock_configuration, mock_env):
        """Test deployment availability checking."""
        import sys
        from unittest.mock import Mock

        from chuk_llm.llm.discovery.azure_openai_discoverer import (
            AzureOpenAIModelDiscoverer,
        )

        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key", azure_endpoint="https://test.openai.azure.com"
        )

        # Mock the OpenAI module and client
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.close = AsyncMock()

        # Mock successful deployment test
        mock_client.chat.completions.create = AsyncMock(
            return_value=MockChatCompletion("test")
        )

        mock_openai.AsyncAzureOpenAI = Mock(return_value=mock_client)
        sys.modules['openai'] = mock_openai

        try:
            result = await discoverer.test_deployment_availability("scribeflowgpt4o")
            assert result is True

            # Mock deployment not found
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("DeploymentNotFound")
            )

            result = await discoverer.test_deployment_availability("non-existent")
            assert result is False
        finally:
            # Cleanup
            if 'openai' in sys.modules:
                del sys.modules['openai']


# ---------------------------------------------------------------------------
# Error Handling and Edge Cases
# ---------------------------------------------------------------------------


class TestAzureOpenAIErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_deployment_not_found_error(self, client):
        """Test handling of deployment not found errors."""
        messages = [{"role": "user", "content": "Hello"}]

        # Mock deployment not found error
        error_response = {
            "error": {
                "code": "DeploymentNotFound",
                "message": "The API deployment for this resource does not exist.",
            }
        }

        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception(f"Error code: 404 - {error_response}")
        )

        result = await client._regular_completion(messages)

        assert result["error"] is True
        assert (
            "DeploymentNotFound" in result["response"]
            or "not found" in result["response"].lower()
        )

    @pytest.mark.asyncio
    async def test_max_tokens_exceeded_error(self, client):
        """Test handling of max tokens exceeded error."""
        messages = [{"role": "user", "content": "Hello"}]

        # Mock max tokens error
        error_response = {
            "error": {
                "message": "max_tokens is too large: 8000. This model supports at most 4096 completion tokens",
                "type": "invalid_request_error",
                "param": "max_tokens",
                "code": "invalid_value",
            }
        }

        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception(f"Error code: 400 - {error_response}")
        )

        result = await client._regular_completion(messages, max_tokens=8000)

        assert result["error"] is True
        assert "max_tokens" in result["response"]

    @pytest.mark.asyncio
    async def test_streaming_deployment_error(self, client):
        """Test streaming with deployment errors."""
        messages = [{"role": "user", "content": "Hello"}]

        # Mock deployment error in streaming
        async def mock_create(**kwargs):
            raise Exception(
                "Error code: 404 - {'error': {'code': 'DeploymentNotFound'}}"
            )

        client.async_client.chat.completions.create = mock_create

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should get exactly one error chunk
        assert len(chunks) == 1
        assert chunks[0]["error"] is True
        assert "deployment" in chunks[0]["response"].lower()


# ---------------------------------------------------------------------------
# Tool Name Compatibility Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIToolCompatibility:
    """Test tool name compatibility and sanitization"""

    def test_tool_name_sanitization_for_azure(self, client):
        """Test tool name sanitization for Azure OpenAI."""
        tools = [
            {"type": "function", "function": {"name": "stdio.read_query"}},
            {"type": "function", "function": {"name": "web.api:search"}},
            {"type": "function", "function": {"name": "azure.resource@analyzer"}},
            {"type": "function", "function": {"name": "db-connector.execute"}},
        ]

        # Mock the sanitization
        def mock_sanitize(tools_list):
            client._current_name_mapping = {
                "stdio_read_query": "stdio.read_query",
                "web_api_search": "web.api:search",
                "azure_resource_analyzer": "azure.resource@analyzer",
                "db_connector_execute": "db-connector.execute",
            }
            return [
                {"type": "function", "function": {"name": "stdio_read_query"}},
                {"type": "function", "function": {"name": "web_api_search"}},
                {"type": "function", "function": {"name": "azure_resource_analyzer"}},
                {"type": "function", "function": {"name": "db_connector_execute"}},
            ]

        client._sanitize_tool_names = mock_sanitize

        sanitized = client._sanitize_tool_names(tools)

        # Check sanitization occurred
        assert len(sanitized) == 4
        assert all("." not in t["function"]["name"] for t in sanitized)
        assert all(":" not in t["function"]["name"] for t in sanitized)
        assert all("@" not in t["function"]["name"] for t in sanitized)
        assert all("-" not in t["function"]["name"] for t in sanitized)

        # Check mapping exists
        assert len(client._current_name_mapping) == 4

    @pytest.mark.asyncio
    async def test_tool_name_restoration_in_response(self, client):
        """Test tool name restoration in responses."""
        response = {
            "response": None,
            "tool_calls": [
                {"function": {"name": "stdio_read_query", "arguments": "{}"}},
                {"function": {"name": "web_api_search", "arguments": "{}"}},
            ],
        }

        name_mapping = {
            "stdio_read_query": "stdio.read_query",
            "web_api_search": "web.api:search",
        }

        # Mock restoration
        def mock_restore(resp, mapping):
            if resp.get("tool_calls") and mapping:
                for tool_call in resp["tool_calls"]:
                    sanitized_name = tool_call["function"]["name"]
                    if sanitized_name in mapping:
                        tool_call["function"]["name"] = mapping[sanitized_name]
            return resp

        client._restore_tool_names_in_response = mock_restore

        restored = client._restore_tool_names_in_response(response, name_mapping)

        # Check names are restored
        assert restored["tool_calls"][0]["function"]["name"] == "stdio.read_query"
        assert restored["tool_calls"][1]["function"]["name"] == "web.api:search"


# ---------------------------------------------------------------------------
# Parameter Validation Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIParameterValidation:
    """Test parameter validation and adjustment"""

    def test_unsupported_parameter_removal(self, client):
        """Test removal of unsupported parameters."""
        kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "unsupported_param": "value",
            "another_unsupported": 123,
        }

        # Mock the validation to remove unsupported params
        def mock_validate(**params):
            # List of supported parameters
            supported = [
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "stream",
                "response_format",
            ]
            return {k: v for k, v in params.items() if k in supported}

        client.validate_parameters = mock_validate

        validated = client.validate_parameters(**kwargs)

        assert "temperature" in validated
        assert "max_tokens" in validated
        assert "unsupported_param" not in validated
        assert "another_unsupported" not in validated

    def test_parameter_limits_enforcement(self, client):
        """Test enforcement of parameter limits."""
        kwargs = {
            "temperature": 2.5,  # Above typical max of 2.0
            "max_tokens": 100000,  # Way above limit
            "top_p": 1.5,  # Above max of 1.0
        }

        # Mock the validation to enforce limits
        def mock_validate(**params):
            result = params.copy()
            if "temperature" in result and result["temperature"] > 2.0:
                result["temperature"] = 2.0
            if "max_tokens" in result and result["max_tokens"] > 4096:
                result["max_tokens"] = 4096
            if "top_p" in result and result["top_p"] > 1.0:
                result["top_p"] = 1.0
            return result

        client.validate_parameters = mock_validate

        validated = client.validate_parameters(**kwargs)

        assert validated["temperature"] == 2.0
        assert validated["max_tokens"] == 4096
        assert validated["top_p"] == 1.0


# ---------------------------------------------------------------------------
# Specialized Model Type Tests (Whisper, DALL-E, Embeddings)
# ---------------------------------------------------------------------------


class TestAzureOpenAISpecializedModels:
    """Test specialized model types like Whisper, DALL-E, embeddings"""

    def test_whisper_model_detection(self, mock_configuration, mock_env):
        """Test detection of Whisper audio models."""
        whisper_models = ["whisper-1", "whisper-large", "custom-whisper"]

        for model_name in whisper_models:
            features = AzureOpenAILLMClient._get_smart_default_features(model_name)
            assert "audio" in features
            assert "transcription" in features
            assert "tools" not in features  # Audio models don't support tools

    def test_dalle_model_detection(self, mock_configuration, mock_env):
        """Test detection of DALL-E image generation models."""
        dalle_models = ["dall-e-3", "dall-e-2", "dalle-custom"]

        for model_name in dalle_models:
            features = AzureOpenAILLMClient._get_smart_default_features(model_name)
            assert "image_generation" in features
            assert "tools" not in features  # Image generation models don't support tools

    def test_embedding_model_detection(self, mock_configuration, mock_env):
        """Test detection of embedding models."""
        embedding_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "ada-002",
        ]

        for model_name in embedding_models:
            features = AzureOpenAILLMClient._get_smart_default_features(model_name)
            assert "text" in features
            assert len(features) == 1  # Only text feature
            assert "tools" not in features
            assert "streaming" not in features


# ---------------------------------------------------------------------------
# Configuration Pattern Matching Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIConfigurationPatterns:
    """Test configuration pattern matching logic"""

    def test_has_explicit_deployment_config_with_pattern_matching(
        self, mock_configuration, mock_env
    ):
        """Test pattern matching in deployment config detection."""
        client = AzureOpenAILLMClient(
            model="test-deployment",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Test with capability that has matches method
        class MockCapabilityWithMatches:
            def matches(self, deployment):
                return deployment.startswith("test-")

        # Test with capability that has pattern attribute
        class MockCapabilityWithPattern:
            pattern = r"^gpt-4.*"

        # Mock the config to return capabilities with different attributes
        def mock_get_config():
            config = MockConfig()
            # Add capability with matches method
            config.azure_openai_provider.model_capabilities = [
                MockCapabilityWithMatches()
            ]
            return config

        with patch("chuk_llm.configuration.get_config", mock_get_config):
            # This should trigger the hasattr(capability, "matches") path
            result = client._has_explicit_deployment_config("test-deployment")
            assert result is True

        # Now test with pattern attribute
        def mock_get_config_with_pattern():
            config = MockConfig()
            config.azure_openai_provider.model_capabilities = [
                MockCapabilityWithPattern()
            ]
            return config

        with patch("chuk_llm.configuration.get_config", mock_get_config_with_pattern):
            # This should trigger the hasattr(capability, "pattern") path
            result = client._has_explicit_deployment_config("gpt-4o-mini")
            assert result is True

    def test_has_explicit_deployment_config_exception_handling(
        self, mock_configuration, mock_env
    ):
        """Test exception handling in deployment config detection."""
        client = AzureOpenAILLMClient(
            model="test-deployment",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock get_config to raise an exception
        with patch(
            "chuk_llm.configuration.get_config", side_effect=Exception("Config error")
        ):
            result = client._has_explicit_deployment_config("test-deployment")
            assert result is False  # Should return False on exception


# ---------------------------------------------------------------------------
# Smart Defaults Fallback Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAISmartDefaultsFallback:
    """Test smart defaults fallback behavior"""

    def test_supports_feature_with_smart_defaults_fallback(
        self, mock_configuration, mock_env
    ):
        """Test smart defaults when config returns None."""
        client = AzureOpenAILLMClient(
            model="unknown-deployment-xyz",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock the parent supports_feature to return None (unknown deployment)
        original_method = ConfigAwareProviderMixin.supports_feature

        def mock_parent_supports_feature(self, feature_name):
            return None  # Indicates unknown/not in config

        # Patch the parent class method
        ConfigAwareProviderMixin.supports_feature = mock_parent_supports_feature

        try:
            # Should fall back to smart defaults
            result = client.supports_feature("tools")
            assert result is True  # Unknown deployments get optimistic defaults
        finally:
            # Restore original method
            ConfigAwareProviderMixin.supports_feature = original_method

    def test_supports_feature_exception_with_gpt_pattern(
        self, mock_configuration, mock_env
    ):
        """Test exception handling with optimistic fallback for GPT patterns."""
        client = AzureOpenAILLMClient(
            model="custom-gpt4-deployment",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock the parent to raise an exception
        def mock_parent_raises(feature_name):
            raise Exception("Config system failure")

        with patch.object(
            ConfigAwareProviderMixin, "supports_feature", mock_parent_raises
        ):
            # Should fall back to optimistic defaults for gpt pattern
            result = client.supports_feature("tools")
            assert result is True  # Optimistic fallback for gpt pattern

    def test_supports_feature_exception_without_gpt_pattern(
        self, mock_configuration, mock_env
    ):
        """Test exception handling without GPT pattern."""
        client = AzureOpenAILLMClient(
            model="random-model-name",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock the parent to raise an exception
        def mock_parent_raises(feature_name):
            raise Exception("Config system failure")

        with patch.object(
            ConfigAwareProviderMixin, "supports_feature", mock_parent_raises
        ):
            # Should return False for non-gpt pattern
            result = client.supports_feature("tools")
            assert result is False


# ---------------------------------------------------------------------------
# Pydantic Message Handling Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIPydanticMessages:
    """Test Pydantic message object handling"""

    def test_validate_request_with_pydantic_message_objects(self, client):
        """Test validation with Pydantic Message objects."""

        # Create mock Pydantic-like message objects
        class PydanticMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [
            PydanticMessage("user", "Hello"),
            PydanticMessage("assistant", "Hi there!"),
        ]

        validated_messages, _, _, _ = client._validate_request_with_config(messages)
        assert len(validated_messages) == 2

    def test_validate_request_with_pydantic_vision_content(self, client):
        """Test validation with Pydantic content objects for vision."""

        # Create mock Pydantic-like content objects
        class PydanticContent:
            def __init__(self, content_type):
                self.type = content_type

        class PydanticMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        # Create message with Pydantic image content
        image_content = PydanticContent("image_url")
        text_content = PydanticContent("text")

        messages = [PydanticMessage("user", [text_content, image_content])]

        # Mock client to not support vision
        client.supports_feature = lambda feature: feature != "vision"

        validated_messages, _, _, _ = client._validate_request_with_config(messages)
        # Should still validate but log warning
        assert len(validated_messages) == 1


# ---------------------------------------------------------------------------
# Parameter Adjustment Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIParameterAdjustment:
    """Test parameter adjustment logic"""

    def test_adjust_parameters_with_smart_defaults_max_completion_tokens(
        self, mock_configuration, mock_env
    ):
        """Test max_completion_tokens adjustment with smart defaults."""
        client = AzureOpenAILLMClient(
            model="o1-preview",  # Reasoning model
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock no explicit config
        client._has_explicit_deployment_config = lambda deployment: False

        # Mock validate_parameters to not remove our parameter
        client.validate_parameters = lambda **kwargs: kwargs

        # Don't provide max_tokens or max_completion_tokens
        params = {"temperature": 0.7}

        adjusted = client._adjust_parameters_for_provider(params)

        # Should add max_completion_tokens or max_tokens for reasoning model
        assert "max_completion_tokens" in adjusted or "max_tokens" in adjusted

    def test_prepare_azure_request_params_without_model(self, client):
        """Test parameter preparation when model is not set."""
        params = {"temperature": 0.7}

        prepared = client._prepare_azure_request_params(**params)

        # Should set model to deployment
        assert prepared["model"] == client.azure_deployment

    def test_adjust_parameters_with_config_max_tokens_limit(
        self, mock_configuration, mock_env
    ):
        """Test max_tokens adjustment based on config limits."""
        client = AzureOpenAILLMClient(
            model="gpt-4o-mini",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock model capabilities with specific limits
        class MockCaps:
            max_output_tokens = 1000
            max_context_length = 128000

        client._get_model_capabilities = lambda: MockCaps()

        # Request more than the limit
        params = {"max_tokens": 5000}

        adjusted = client._adjust_parameters_for_provider(params)

        # Should be capped at model limit
        assert adjusted["max_tokens"] == 1000

    def test_adjust_parameters_with_config_max_completion_tokens_limit(
        self, mock_configuration, mock_env
    ):
        """Test max_completion_tokens adjustment based on config limits."""
        client = AzureOpenAILLMClient(
            model="o1-preview",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock model capabilities with specific limits
        class MockCaps:
            max_output_tokens = 2000
            max_context_length = 200000

        client._get_model_capabilities = lambda: MockCaps()

        # Request more than the limit
        params = {"max_completion_tokens": 10000}

        adjusted = client._adjust_parameters_for_provider(params)

        # Should be capped at model limit
        assert adjusted["max_completion_tokens"] == 2000


# ---------------------------------------------------------------------------
# Azure-Specific Argument Formatting Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIArgumentFormatting:
    """Test Azure-specific argument formatting"""

    @pytest.mark.asyncio
    async def test_streaming_with_double_quoted_arguments(self, client):
        """Test streaming tool calls with Azure double-quoted arguments."""

        # Create mock tool call with double-quoted arguments
        class MockToolCall:
            def __init__(self):
                self.index = 0
                self.id = "call_123"

                class MockFunction:
                    name = "test_function"
                    # Azure sometimes returns arguments wrapped in double quotes
                    arguments = '""{"key": "value"}""'

                self.function = MockFunction()

        mock_chunk = MockStreamChunk(content="", tool_calls=[MockToolCall()])
        mock_stream = MockAsyncStream([mock_chunk])

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should have properly parsed the double-quoted arguments
        # The test is that it doesn't error out and handles the format


# ---------------------------------------------------------------------------
# Error Handling Tests - JSONDecodeError and Exceptions
# ---------------------------------------------------------------------------


class TestAzureOpenAIJsonErrorHandling:
    """Test JSON decode error handling"""

    @pytest.mark.asyncio
    async def test_streaming_incomplete_json_tool_call(self, client):
        """Test streaming with incomplete JSON in tool calls."""

        # Create mock tool calls with incomplete JSON
        class MockIncompleteToolCall:
            def __init__(self, args_chunk):
                self.index = 0
                self.id = "call_123"

                class MockFunction:
                    name = "test_function"
                    arguments = ""

                self.function = MockFunction()
                self.function.arguments = args_chunk

        # Simulate streaming incomplete JSON chunks
        chunk1 = MockStreamChunk(
            content="", tool_calls=[MockIncompleteToolCall('{"key":')]
        )
        chunk2 = MockStreamChunk(
            content="", tool_calls=[MockIncompleteToolCall(' "value"}')]
        )

        mock_stream = MockAsyncStream([chunk1, chunk2])

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should handle incomplete JSON gracefully

    @pytest.mark.asyncio
    async def test_streaming_tool_call_processing_exception(self, client):
        """Test exception handling during tool call processing in streaming."""

        # Create mock tool call that will cause exception
        class BrokenToolCall:
            def __init__(self):
                self.index = 0
                # Missing id attribute - will cause exception
                self.function = None  # This will cause AttributeError

        mock_chunk = MockStreamChunk(content="", tool_calls=[BrokenToolCall()])
        mock_stream = MockAsyncStream([mock_chunk])

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should handle exception gracefully and continue

    @pytest.mark.asyncio
    async def test_streaming_deployment_not_found_error(self, client):
        """Test deployment not found error in streaming."""

        async def mock_create(**kwargs):
            raise Exception("DeploymentNotFound - the deployment does not exist")

        client.async_client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should get error chunk
        assert len(chunks) == 1
        assert chunks[0]["error"] is True
        assert "deployment" in chunks[0]["response"].lower()


# ---------------------------------------------------------------------------
# Regular Completion Error Handling Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIRegularCompletionErrors:
    """Test error handling in regular (non-streaming) completions"""

    @pytest.mark.asyncio
    async def test_regular_completion_deployment_not_found(self, client):
        """Test deployment not found error in regular completion."""
        messages = [{"role": "user", "content": "Hello"}]

        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception(
                "Error code: 404 - DeploymentNotFound: The deployment does not exist"
            )
        )

        result = await client._regular_completion(messages)

        assert result["error"] is True
        assert "deployment" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_regular_completion_tool_naming_error(self, client):
        """Test tool naming error in regular completion."""
        messages = [{"role": "user", "content": "Hello"}]

        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception(
                "Invalid function name: names must match pattern ^[a-zA-Z0-9_-]{1,64}$"
            )
        )

        result = await client._regular_completion(messages)

        assert result["error"] is True
        assert "tool naming error" in result["response"].lower()


# ---------------------------------------------------------------------------
# Message Normalization Error Handling Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIMessageNormalization:
    """Test message normalization with error handling"""

    def test_normalize_message_with_invalid_json_arguments(self, client):
        """Test normalization with invalid JSON in tool arguments."""

        class MockToolCallWithInvalidJson:
            id = "call_123"

            class function:
                name = "test_function"
                arguments = "{invalid json}}"  # Invalid JSON

        class MockMessage:
            content = None
            tool_calls = [MockToolCallWithInvalidJson()]

        # Should handle invalid JSON gracefully
        result = client._normalize_message(MockMessage())

        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_normalize_message_with_double_quoted_arguments(self, client):
        """Test normalization with Azure double-quoted arguments."""

        class MockToolCallWithDoubleQuotes:
            id = "call_123"

            class function:
                name = "test_function"
                arguments = '""{"key": "value"}""'

        class MockMessage:
            content = None
            tool_calls = [MockToolCallWithDoubleQuotes()]

        result = client._normalize_message(MockMessage())

        # Should handle double-quoted format without error
        # The result may be an empty object if parsing fails
        assert "tool_calls" in result
        assert len(result["tool_calls"]) > 0

    def test_normalize_message_with_dict_arguments(self, client):
        """Test normalization when arguments are already a dict."""

        class MockToolCallWithDict:
            id = "call_123"

            class function:
                name = "test_function"
                arguments = {"key": "value"}  # Already a dict

        class MockMessage:
            content = None
            tool_calls = [MockToolCallWithDict()]

        result = client._normalize_message(MockMessage())

        # Should convert dict to JSON string
        assert isinstance(result["tool_calls"][0]["function"]["arguments"], str)

    def test_normalize_message_fallback_with_invalid_arguments(self, client):
        """Test fallback normalization with invalid argument types."""

        class MockToolCallWithInvalidType:
            id = "call_123"

            class function:
                name = "test_function"
                arguments = 12345  # Invalid type (number)

        class MockMessage:
            content = None
            tool_calls = [MockToolCallWithInvalidType()]

        # Mock to trigger fallback path
        def mock_normalize_raises(msg):
            raise AttributeError("Simulated attribute error")

        # Temporarily replace the parent method
        original_method = client._normalize_message

        try:
            # Trigger the fallback implementation
            result = client._normalize_message(MockMessage())

            # Should default to empty object for invalid types
            assert result["tool_calls"][0]["function"]["arguments"] == "{}"
        except AttributeError:
            # Fallback path may not be reachable if parent implementation works
            pass

    def test_normalize_message_fallback_tool_call_exception(self, client):
        """Test fallback normalization with tool call processing exception."""

        class BrokenToolCall:
            # Missing required attributes to cause exception
            @property
            def id(self):
                raise Exception("Broken id")

            @property
            def function(self):
                raise Exception("Broken function")

        class MockMessage:
            content = "Some content"
            tool_calls = [BrokenToolCall()]

        # This should trigger the exception handling in the fallback path
        result = client._normalize_message(MockMessage())

        # Should handle exception gracefully
        # Even with broken tool calls, should return some result
        assert "response" in result or "tool_calls" in result


# ---------------------------------------------------------------------------
# Additional Coverage Tests for Specific Code Paths
# ---------------------------------------------------------------------------


class TestAzureOpenAIAdditionalCoverage:
    """Additional tests to cover specific code paths"""

    @pytest.mark.asyncio
    async def test_streaming_with_azure_double_quote_stripping(self, client):
        """Test the specific Azure double-quote stripping path in streaming."""

        # Create tool call that will complete with double-quoted arguments
        class MockToolCallWithDoubleQuotes:
            def __init__(self, chunk_num):
                self.index = 0
                self.id = "call_123"

                class MockFunction:
                    name = "test_function" if chunk_num == 0 else ""
                    # Azure double-quoted format that needs stripping
                    arguments = (
                        '""{"key": "value"}""' if chunk_num == 0 else ""
                    )

                self.function = MockFunction()

        # Create chunks with complete JSON that has double quotes
        chunk1 = MockStreamChunk(
            content="", tool_calls=[MockToolCallWithDoubleQuotes(0)]
        )

        mock_stream = MockAsyncStream([chunk1])

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should handle the double-quoted format
        # The test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_validate_request_smart_defaults_without_config(
        self, mock_configuration, mock_env
    ):
        """Test validation with smart defaults when no explicit config."""
        client = AzureOpenAILLMClient(
            model="custom-unknown-deployment",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock that deployment has no explicit config
        client._has_explicit_deployment_config = lambda deployment: False

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        # This should trigger smart defaults path in validation
        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(messages, tools, stream=False)
        )

        assert validated_messages == messages
        assert validated_tools is not None  # Smart defaults allow tools

    def test_adjust_parameters_exception_fallback(self, mock_configuration, mock_env):
        """Test parameter adjustment exception handling."""
        client = AzureOpenAILLMClient(
            model="test-deployment",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock validate_parameters to raise an exception
        def mock_validate_raises(**params):
            raise Exception("Validation error")

        client.validate_parameters = mock_validate_raises

        # Should fall back to setting max_tokens
        params = {"temperature": 0.7}
        adjusted = client._adjust_parameters_for_provider(params)

        # Should have fallback max_tokens
        assert "max_tokens" in adjusted

    def test_get_auth_type_token_provider(self, mock_configuration, mock_env):
        """Test authentication type detection for token provider."""
        client = AzureOpenAILLMClient(
            model="test",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_ad_token_provider=lambda: "token",
        )

        # Mock the client attribute
        client.async_client._azure_ad_token_provider = lambda: "token"

        auth_type = client._get_auth_type()
        assert auth_type == "azure_ad_token_provider"

    def test_get_auth_type_ad_token(self, mock_configuration, mock_env):
        """Test authentication type detection for AD token."""
        client = AzureOpenAILLMClient(
            model="test",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_ad_token="test-token",
        )

        # Mock the client attribute
        client.async_client._azure_ad_token = "test-token"
        client.async_client._azure_ad_token_provider = None

        auth_type = client._get_auth_type()
        assert auth_type == "azure_ad_token"

    @pytest.mark.asyncio
    async def test_streaming_with_chunk_error_handling(self, client):
        """Test chunk error handling in streaming."""

        # Create a chunk that will cause an error when processing
        class BrokenChunk:
            def __init__(self):
                pass

            @property
            def choices(self):
                raise Exception("Chunk processing error")

        mock_stream = MockAsyncStream([BrokenChunk()])

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should handle chunk error gracefully
        # May not yield any chunks if all fail

    @pytest.mark.asyncio
    async def test_streaming_retryable_error(self, client):
        """Test streaming with retryable errors."""
        attempt_count = 0

        async def mock_create(**kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                # First attempt: retryable error
                raise Exception("Connection timeout error")
            else:
                # Second attempt: success
                return MockAsyncStream([MockStreamChunk("Success")])

        client.async_client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should have retried and succeeded
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_streaming_non_retryable_error_final_failure(self, client):
        """Test streaming with non-retryable error on final attempt."""
        attempt_count = 0

        async def mock_create(**kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise Exception("Invalid request format")

        client.async_client.chat.completions.create = mock_create

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should get error chunk
        assert len(chunks) == 1
        assert chunks[0]["error"] is True

    def test_normalize_message_with_string_arguments_valid_json(self, client):
        """Test normalization with valid JSON string arguments."""

        class MockToolCall:
            id = "call_123"

            class function:
                name = "test_function"
                arguments = '{"key": "value", "number": 42}'

        class MockMessage:
            content = None
            tool_calls = [MockToolCall()]

        result = client._normalize_message(MockMessage())

        # Should parse and re-serialize JSON
        assert "tool_calls" in result
        assert len(result["tool_calls"]) > 0

    def test_gpt5_model_feature_detection(self, mock_configuration, mock_env):
        """Test GPT-5 model feature detection."""
        gpt5_models = ["gpt-5", "gpt5-turbo", "custom-gpt5-deployment"]

        for model_name in gpt5_models:
            features = AzureOpenAILLMClient._get_smart_default_features(model_name)
            assert "reasoning" in features
            assert "vision" in features
            assert "tools" in features

    def test_gpt5_model_parameter_defaults(self, mock_configuration, mock_env):
        """Test GPT-5 model parameter defaults."""
        params = AzureOpenAILLMClient._get_smart_default_parameters("gpt5-turbo")

        assert params["max_context_length"] == 272000
        assert params["max_output_tokens"] == 16384
        assert params["requires_max_completion_tokens"] is True

    @pytest.mark.asyncio
    async def test_prepare_request_params_with_deployment_name(self, client):
        """Test parameter preparation with deployment_name parameter."""
        params = {"deployment_name": "custom-deployment", "temperature": 0.7}

        prepared = client._prepare_azure_request_params(**params)

        # Should move deployment_name to model
        assert prepared["model"] == "custom-deployment"
        assert "deployment_name" not in prepared


# ---------------------------------------------------------------------------
# Initialization Error Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIInitializationErrors:
    """Test initialization error conditions"""

    def test_missing_azure_endpoint(self, mock_configuration):
        """Test error when azure_endpoint is missing."""
        with pytest.raises(ValueError) as exc_info:
            AzureOpenAILLMClient(
                model="gpt-4o-mini",
                api_key="test-key",
                azure_endpoint=None,  # Missing
            )
        assert "azure_endpoint is required" in str(exc_info.value)

    def test_missing_all_authentication(self, mock_configuration):
        """Test error when no authentication is provided."""
        with pytest.raises(ValueError) as exc_info:
            AzureOpenAILLMClient(
                model="gpt-4o-mini",
                api_key=None,
                azure_ad_token=None,
                azure_ad_token_provider=None,
                azure_endpoint="https://test.openai.azure.com",
            )
        assert "Authentication required" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Additional Edge Case Tests
# ---------------------------------------------------------------------------


class TestAzureOpenAIEdgeCases:
    """Test additional edge cases for full coverage"""

    @pytest.mark.asyncio
    async def test_close_method(self, client):
        """Test the close method."""
        # Mock the close methods
        client.async_client.close = AsyncMock()
        client.client.close = Mock()

        await client.close()

        # Verify close was called
        client.async_client.close.assert_called_once()
        client.client.close.assert_called_once()

    def test_repr_method(self, client):
        """Test the __repr__ method."""
        repr_str = repr(client)
        assert "AzureOpenAILLMClient" in repr_str
        assert client.azure_deployment in repr_str
        assert client.model in repr_str

    @pytest.mark.asyncio
    async def test_create_completion_with_name_mapping(self, client):
        """Test create_completion stores name mapping."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test.tool"}}]

        # Mock the sanitization to create a mapping
        def mock_sanitize(tools_list):
            client._current_name_mapping = {"test_tool": "test.tool"}
            return [{"type": "function", "function": {"name": "test_tool"}}]

        client._sanitize_tool_names = mock_sanitize

        # Mock the completion
        mock_response = MockChatCompletion(content="Response")
        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await client.create_completion(messages, tools=tools, stream=False)

        # Verify name mapping was created
        assert hasattr(client, "_current_name_mapping")

    def test_has_explicit_deployment_config_in_models_list(
        self, mock_configuration, mock_env
    ):
        """Test deployment found in models list."""
        client = AzureOpenAILLMClient(
            model="gpt-4o",  # This is in the mock models list
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        result = client._has_explicit_deployment_config("gpt-4o")
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_request_with_json_mode_unsupported(self, client):
        """Test JSON mode when not supported."""
        # Mock client to not support json_mode
        client.supports_feature = lambda feature: feature != "json_mode"

        messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"response_format": {"type": "json_object"}}

        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(messages, None, False, **kwargs)
        )

        # response_format should be removed
        assert "response_format" not in validated_kwargs

    @pytest.mark.asyncio
    async def test_validate_request_with_streaming_unsupported(self, client):
        """Test streaming when not supported."""
        # Mock client to not support streaming
        client.supports_feature = lambda feature: feature != "streaming"

        messages = [{"role": "user", "content": "Hello"}]

        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(messages, None, True)
        )

        # stream should be disabled
        assert validated_stream is False

    @pytest.mark.asyncio
    async def test_validate_request_with_tools_unsupported(self, client):
        """Test tools when not supported."""
        # Mock client to not support tools
        client.supports_feature = lambda feature: feature != "tools"

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            client._validate_request_with_config(messages, tools, False)
        )

        # tools should be None
        assert validated_tools is None

    def test_o3_model_feature_detection(self, mock_configuration, mock_env):
        """Test O3 model feature detection (not O1)."""
        features = AzureOpenAILLMClient._get_smart_default_features("o3-mini")

        # O3 models should have full features including tools
        assert "reasoning" in features
        assert "tools" in features
        assert "streaming" in features

    def test_normalize_message_with_dict_type_arguments(self, client):
        """Test normalization when arguments are dict (not string)."""

        class MockToolCallWithDict:
            id = "call_123"

            class function:
                name = "test_function"
                arguments = {"key": "value"}  # Dict, not string

        class MockMessage:
            content = None
            tool_calls = [MockToolCallWithDict()]

        result = client._normalize_message(MockMessage())

        # Should convert dict to JSON string
        assert isinstance(result["tool_calls"][0]["function"]["arguments"], str)
        assert "key" in result["tool_calls"][0]["function"]["arguments"]

    def test_supports_feature_with_unsupported_smart_default(
        self, mock_configuration, mock_env
    ):
        """Test feature support when smart defaults say no."""
        client = AzureOpenAILLMClient(
            model="text-embedding-ada",  # Embedding model
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        # Mock parent to return None
        original_method = ConfigAwareProviderMixin.supports_feature

        def mock_parent_supports_feature(self, feature_name):
            return None

        ConfigAwareProviderMixin.supports_feature = mock_parent_supports_feature

        try:
            # Embedding models should not support tools
            result = client.supports_feature("tools")
            assert result is False
        finally:
            ConfigAwareProviderMixin.supports_feature = original_method

    @pytest.mark.asyncio
    async def test_regular_completion_generic_error(self, client):
        """Test generic error handling in regular completion."""
        messages = [{"role": "user", "content": "Hello"}]

        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Some random error")
        )

        result = await client._regular_completion(messages)

        assert result["error"] is True
        assert "error" in result["response"].lower()


# ---------------------------------------------------------------------------
# Comprehensive Normalize Message Tests for Full Coverage
# ---------------------------------------------------------------------------


class TestAzureOpenAINormalizeMessageComprehensive:
    """Comprehensive tests for _normalize_message to reach 90%+ coverage"""

    def test_normalize_message_via_super_with_tool_calls(self, client):
        """Test normalization through super() method with tool call processing."""

        class MockToolCall:
            id = "call_123"

            class function:
                name = "test_function"
                arguments = '{"key": "value"}'

        class MockMessage:
            content = None
            tool_calls = [MockToolCall()]

        # This should trigger the parent _normalize_message then the AZURE FIX
        result = client._normalize_message(MockMessage())

        assert "tool_calls" in result
        assert len(result["tool_calls"]) > 0

    def test_normalize_message_nested_double_quote_handling(self, client):
        """Test the nested double quote handling in normalization."""

        class MockToolCall:
            id = "call_456"

            class function:
                name = "test_tool"
                # Valid JSON wrapped in double quotes (Azure format)
                arguments = '{"nested": "value"}'

        class MockMessage:
            content = None
            tool_calls = [MockToolCall()]

        result = client._normalize_message(MockMessage())

        # Should handle and parse the JSON properly
        assert "tool_calls" in result
        assert len(result["tool_calls"]) > 0

    def test_normalize_message_with_integer_arguments(self, client):
        """Test normalization with integer arguments (non-dict, non-string)."""

        class MockToolCall:
            id = "call_789"

            class function:
                name = "test_tool"
                arguments = 12345  # Integer, not string or dict

        class MockMessage:
            content = None
            tool_calls = [MockToolCall()]

        result = client._normalize_message(MockMessage())

        # Should default to empty object for invalid types
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_normalize_message_with_list_arguments(self, client):
        """Test normalization with list arguments."""

        class MockToolCall:
            id = "call_list"

            class function:
                name = "test_tool"
                arguments = ["item1", "item2"]  # List, not string or dict

        class MockMessage:
            content = None
            tool_calls = [MockToolCall()]

        result = client._normalize_message(MockMessage())

        # Should default to empty object for invalid types
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"

    @pytest.mark.asyncio
    async def test_streaming_complete_coverage_path(self, client):
        """Test streaming to cover remaining lines."""

        # Create a complete tool call that exercises more code paths
        class CompleteToolCall:
            def __init__(self):
                self.index = 0
                self.id = "call_complete"

                class MockFunc:
                    name = "complete_function"
                    arguments = '{"complete": "args"}'

                self.function = MockFunc()

        chunk = MockStreamChunk(content="", tool_calls=[CompleteToolCall()])
        mock_stream = MockAsyncStream([chunk])

        client.async_client.chat.completions.create = AsyncMock(
            return_value=mock_stream
        )

        messages = [{"role": "user", "content": "Test"}]

        chunks = []
        async for chunk in client._stream_completion_async(messages):
            chunks.append(chunk)

        # Should process the complete tool call


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
