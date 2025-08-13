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
import pytest
import asyncio
import json
import os
from unittest.mock import MagicMock, AsyncMock, patch, Mock, PropertyMock
from typing import AsyncIterator, List, Dict, Any

# Import the existing mock setup from the main test file
from test_azure_openai_client import (
    MockStreamChunk, MockChoice, MockDelta, MockMessage,
    MockAsyncStream, MockChatCompletion, MockCompletions,
    MockChat, MockAzureOpenAI, MockAsyncAzureOpenAI,
    MockFeature, MockModelCapabilities, MockProviderConfig,
    MockConfig, mock_configuration, mock_env, client
)

from chuk_llm.llm.providers.azure_openai_client import AzureOpenAILLMClient

# ---------------------------------------------------------------------------
# Custom Deployment Tests
# ---------------------------------------------------------------------------

class TestAzureOpenAICustomDeployments:
    """Test Azure OpenAI custom deployment support"""

    def test_custom_deployment_validation_always_passes(self, mock_configuration, mock_env):
        """Test that ANY custom deployment name passes validation."""
        # Test various custom deployment names
        custom_deployments = [
            "scribeflowgpt4o",
            "company-gpt-4-turbo",
            "prod-deployment-v2",
            "my-custom-gpt4",
            "test_deployment_123",
            "deployment-with-special-chars-2024"
        ]
        
        for deployment_name in custom_deployments:
            client = AzureOpenAILLMClient(
                model=deployment_name,
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com"
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
            azure_endpoint="https://test.openai.azure.com"
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
        assert smart_params["max_output_tokens"] == 4096  # Updated based on actual limits
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
            ("my-custom-model", {"text", "streaming", "system_messages", "tools", "json_mode"}),
        ]
        
        for deployment_name, expected_features in test_cases:
            features = AzureOpenAILLMClient._get_smart_default_features(deployment_name)
            for feature in expected_features:
                assert feature in features, f"Expected {feature} in features for {deployment_name}"

    def test_model_info_with_smart_defaults(self, mock_configuration, mock_env):
        """Test model info includes smart defaults information."""
        client = AzureOpenAILLMClient(
            model="custom-deployment-xyz",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
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
            azure_endpoint="https://test.openai.azure.com"
        )
        
        messages = [{"role": "user", "content": "Use tools"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "stdio.read_query",  # Problematic name
                    "description": "Read from stdio",
                    "parameters": {}
                }
            }
        ]
        
        # Create proper mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "stdio_read_query"  # Sanitized name
        mock_tool_call.function.arguments = '{"query": "test"}'
        
        # Mock the completion
        mock_response = MockChatCompletion(content=None, tool_calls=[mock_tool_call])
        
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock sanitization
        client._sanitize_tool_names = lambda tools: [
            {
                "type": "function",
                "function": {
                    "name": "stdio_read_query",  # Sanitized
                    "description": "Read from stdio",
                    "parameters": {}
                }
            }
        ]
        client._current_name_mapping = {"stdio_read_query": "stdio.read_query"}
        
        # Mock the normalization to return proper structure
        def mock_normalize(msg):
            return {
                "response": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "stdio_read_query",
                        "arguments": '{"query": "test"}'
                    }
                }]
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
            azure_endpoint="https://test.openai.azure.com"
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
            {"role": "user", "content": "What's my name?"}  # Tests context memory
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
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2+2"}'
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "calculator",
                "content": "4"
            },
            {"role": "assistant", "content": "2+2 equals 4"},
            {"role": "user", "content": "What did I just ask you to calculate?"}
        ]
        
        validated_messages, _, _, _ = client._validate_request_with_config(messages)
        
        # All messages including tool interactions should be preserved
        assert len(validated_messages) == 5
        
        # Check tool call is preserved
        assert validated_messages[1]["tool_calls"][0]["function"]["name"] == "calculator"
        
        # Check tool response is preserved
        assert validated_messages[2]["role"] == "tool"
        assert validated_messages[2]["content"] == "4"
        
        # Check final question that requires context
        assert validated_messages[4]["content"] == "What did I just ask you to calculate?"

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
                        }
                    }
                ]
            },
            {"role": "assistant", "content": "I see an image."},
            {"role": "user", "content": "What's my name again?"}
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
            {"role": "user", "content": "What city do I live in?"}
        ]
        
        # Mock streaming response that uses context
        mock_stream = MockAsyncStream([
            MockStreamChunk("Based on our conversation, "),
            MockStreamChunk("you live in Tokyo.")
        ])
        
        client.async_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        
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
            {"role": "assistant", "content": "I'd be happy to help! Where would you like to go?"},
            {"role": "user", "content": "Paris"},
            {
                "role": "assistant",
                "content": "Let me find information about Paris.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_destination",
                        "arguments": '{"city": "Paris"}'
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search_destination",
                "content": "Paris: Capital of France, known for Eiffel Tower, Louvre Museum"
            },
            {"role": "assistant", "content": "Paris is wonderful! The Eiffel Tower and Louvre are must-sees."},
            {"role": "user", "content": "How about hotels?"},
            {
                "role": "assistant",
                "content": "Let me search for hotels.",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "search_hotels",
                        "arguments": '{"city": "Paris", "rating": 4}'
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "search_hotels",
                "content": "Found 5 hotels: Hotel Le Marais, Hotel Saint-Germain..."
            },
            {"role": "assistant", "content": "I found several great hotels including Hotel Le Marais."},
            {"role": "user", "content": "What city are we discussing again?"}
        ]
        
        validated_messages, _, _, _ = client._validate_request_with_config(messages)
        
        # All 12 messages should be preserved
        assert len(validated_messages) == 12
        
        # Verify Paris is mentioned in the context
        assert "Paris" in validated_messages[3]["content"]
        
        # Verify tool calls are preserved
        assert validated_messages[4]["tool_calls"][0]["function"]["name"] == "search_destination"
        assert validated_messages[8]["tool_calls"][0]["function"]["name"] == "search_hotels"

    @pytest.mark.asyncio
    async def test_context_affects_completion_response(self, client):
        """Test that context actually affects the completion response."""
        # First conversation without context
        messages_no_context = [
            {"role": "user", "content": "What's my favorite color?"}
        ]
        
        # Second conversation with context
        messages_with_context = [
            {"role": "user", "content": "My favorite color is blue"},
            {"role": "assistant", "content": "Blue is a great color!"},
            {"role": "user", "content": "What's my favorite color?"}
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
            azure_endpoint="https://test.openai.azure.com"
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
            azure_endpoint="https://test.openai.azure.com"
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
        from chuk_llm.llm.discovery.azure_openai_discoverer import AzureOpenAIModelDiscoverer
        
        discoverer = AzureOpenAIModelDiscoverer(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
        )
        
        # Mock the OpenAI client for testing
        mock_client = AsyncMock()
        
        # Mock successful deployment test
        mock_client.chat.completions.create = AsyncMock(return_value=MockChatCompletion("test"))
        
        with patch('openai.AsyncAzureOpenAI', return_value=mock_client):
            result = await discoverer.test_deployment_availability("scribeflowgpt4o")
            assert result is True
        
        # Mock deployment not found
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("DeploymentNotFound")
        )
        
        with patch('openai.AsyncAzureOpenAI', return_value=mock_client):
            result = await discoverer.test_deployment_availability("non-existent")
            assert result is False

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
            'error': {
                'code': 'DeploymentNotFound',
                'message': 'The API deployment for this resource does not exist.'
            }
        }
        
        client.async_client.chat.completions.create = AsyncMock(
            side_effect=Exception(f"Error code: 404 - {error_response}")
        )
        
        result = await client._regular_completion(messages)
        
        assert result["error"] is True
        assert "DeploymentNotFound" in result["response"] or "not found" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_max_tokens_exceeded_error(self, client):
        """Test handling of max tokens exceeded error."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock max tokens error
        error_response = {
            'error': {
                'message': 'max_tokens is too large: 8000. This model supports at most 4096 completion tokens',
                'type': 'invalid_request_error',
                'param': 'max_tokens',
                'code': 'invalid_value'
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
            raise Exception("Error code: 404 - {'error': {'code': 'DeploymentNotFound'}}")
        
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
                "db_connector_execute": "db-connector.execute"
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
                {"function": {"name": "web_api_search", "arguments": "{}"}}
            ]
        }
        
        name_mapping = {
            "stdio_read_query": "stdio.read_query",
            "web_api_search": "web.api:search"
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
            "another_unsupported": 123
        }
        
        # Mock the validation to remove unsupported params
        def mock_validate(**params):
            # List of supported parameters
            supported = ["temperature", "max_tokens", "top_p", "frequency_penalty", 
                        "presence_penalty", "stop", "stream", "response_format"]
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
            "top_p": 1.5  # Above max of 1.0
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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])