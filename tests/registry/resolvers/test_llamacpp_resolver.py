"""
Comprehensive tests for LlamaCppCapabilityResolver.

Tests cover:
- Initialization with default and custom parameters
- Capability resolution from server props
- Property parsing
- Quality tier inference
- Error handling
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec, QualityTier
from chuk_llm.registry.resolvers.llamacpp import LlamaCppCapabilityResolver


class TestLlamaCppCapabilityResolverInit:
    """Test LlamaCppCapabilityResolver initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        resolver = LlamaCppCapabilityResolver()

        assert resolver.api_base == "http://localhost:8080"
        assert resolver.timeout == 5.0

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base."""
        resolver = LlamaCppCapabilityResolver(api_base="http://localhost:9000")

        assert resolver.api_base == "http://localhost:9000"

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        resolver = LlamaCppCapabilityResolver(timeout=10.0)

        assert resolver.timeout == 10.0


class TestGetCapabilities:
    """Test get_capabilities method."""

    @pytest.mark.asyncio
    async def test_get_capabilities_success(self):
        """Test successful capability resolution."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b-instruct",
            family="llama-3",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "default_generation_settings": {
                "n_ctx": 8192,
            },
            "chat_template": "{% if messages[0]['role'] == 'system' %}...",
        }
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            capabilities = await resolver.get_capabilities(spec)

            assert capabilities.max_context == 8192
            assert capabilities.supports_streaming is True
            assert capabilities.supports_json_mode is True
            assert capabilities.supports_system_messages is True
            assert capabilities.source == "llamacpp_props"
            assert "temperature" in capabilities.known_params

    @pytest.mark.asyncio
    async def test_get_capabilities_wrong_provider(self):
        """Test that non-llamacpp providers return empty capabilities."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.OPENAI.value,
            name="gpt-4",
            family="gpt-4",
        )

        capabilities = await resolver.get_capabilities(spec)

        # Empty capabilities have None for all fields
        assert capabilities.max_context is None
        assert capabilities.supports_tools is None
        assert capabilities.supports_vision is None

    @pytest.mark.asyncio
    async def test_get_capabilities_http_error(self):
        """Test capability resolution with HTTP error."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(side_effect=httpx.HTTPError("API Error"))
            mock_client.return_value = mock_instance

            capabilities = await resolver.get_capabilities(spec)

            # Should return empty capabilities on error (None for all fields)
            assert capabilities.max_context is None
            assert capabilities.supports_tools is None

    @pytest.mark.asyncio
    async def test_get_capabilities_connect_error(self):
        """Test capability resolution with connection error."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client.return_value = mock_instance

            capabilities = await resolver.get_capabilities(spec)

            # Should return empty capabilities on error
            assert capabilities.max_context is None

    @pytest.mark.asyncio
    async def test_get_capabilities_uses_correct_endpoint(self):
        """Test that correct endpoint is used."""
        resolver = LlamaCppCapabilityResolver(api_base="http://localhost:9000")
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            await resolver.get_capabilities(spec)

            # Verify get was called with correct endpoint
            mock_instance.get.assert_called_once_with("http://localhost:9000/props")

    @pytest.mark.asyncio
    async def test_get_capabilities_uses_timeout(self):
        """Test that custom timeout is used."""
        resolver = LlamaCppCapabilityResolver(timeout=15.0)
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            await resolver.get_capabilities(spec)

            # Verify AsyncClient was created with correct timeout
            mock_client.assert_called_once_with(timeout=15.0)


class TestParseServerProps:
    """Test _parse_server_props method."""

    def test_parse_server_props_full_data(self):
        """Test parsing with full server props."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-70b-instruct",
            family="llama-3",
        )

        props = {
            "default_generation_settings": {
                "n_ctx": 8192,
            },
            "chat_template": "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
        }

        capabilities = resolver._parse_server_props(props, spec)

        assert capabilities.max_context == 8192
        assert capabilities.supports_streaming is True
        assert capabilities.supports_system_messages is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_vision is False
        assert capabilities.quality_tier == QualityTier.BEST
        assert capabilities.source == "llamacpp_props"

    def test_parse_server_props_with_tool_support(self):
        """Test parsing with tool/function support in chat template."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        props = {
            "chat_template": "Template with TOOL calling support",
        }

        capabilities = resolver._parse_server_props(props, spec)

        assert capabilities.supports_tools is True

    def test_parse_server_props_with_function_support(self):
        """Test parsing with function keyword in chat template."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        props = {
            "chat_template": "Template with FUNCTION calling support",
        }

        capabilities = resolver._parse_server_props(props, spec)

        assert capabilities.supports_tools is True

    def test_parse_server_props_without_tools(self):
        """Test parsing without tool support."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        props = {
            "chat_template": "Basic chat template",
        }

        capabilities = resolver._parse_server_props(props, spec)

        assert capabilities.supports_tools is False

    def test_parse_server_props_no_chat_template(self):
        """Test parsing without chat template."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        props = {}

        capabilities = resolver._parse_server_props(props, spec)

        assert capabilities.supports_system_messages is False
        assert capabilities.supports_tools is False

    def test_parse_server_props_empty_chat_template(self):
        """Test parsing with empty chat template."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        props = {
            "chat_template": "",
        }

        capabilities = resolver._parse_server_props(props, spec)

        # Empty template still counts as having chat_template in props (key exists)
        # but supports_tools should be False since the template is empty
        assert capabilities.supports_system_messages is True  # Key exists
        assert capabilities.supports_tools is False  # Empty template has no tool/function

    def test_parse_server_props_no_context(self):
        """Test parsing without context length."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        props = {}

        capabilities = resolver._parse_server_props(props, spec)

        assert capabilities.max_context is None

    def test_parse_server_props_known_params(self):
        """Test that known parameters are set."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-8b",
            family="llama-3",
        )

        props = {}

        capabilities = resolver._parse_server_props(props, spec)

        assert "temperature" in capabilities.known_params
        assert "top_p" in capabilities.known_params
        assert "top_k" in capabilities.known_params
        assert "max_tokens" in capabilities.known_params
        assert "frequency_penalty" in capabilities.known_params
        assert "presence_penalty" in capabilities.known_params


class TestInferQualityTier:
    """Test _infer_quality_tier method."""

    def test_infer_quality_tier_best(self):
        """Test inference of BEST quality tier."""
        resolver = LlamaCppCapabilityResolver()

        assert resolver._infer_quality_tier("llama-3-70b-instruct") == QualityTier.BEST
        assert resolver._infer_quality_tier("mistral-72b") == QualityTier.BEST
        assert resolver._infer_quality_tier("qwen-65b-chat") == QualityTier.BEST
        assert resolver._infer_quality_tier("model-80b") == QualityTier.BEST

    def test_infer_quality_tier_balanced(self):
        """Test inference of BALANCED quality tier."""
        resolver = LlamaCppCapabilityResolver()

        assert resolver._infer_quality_tier("llama-3-30b-instruct") == QualityTier.BALANCED
        assert resolver._infer_quality_tier("mixtral-32b") == QualityTier.BALANCED
        assert resolver._infer_quality_tier("qwen-34b-chat") == QualityTier.BALANCED
        assert resolver._infer_quality_tier("model-40b") == QualityTier.BALANCED

    def test_infer_quality_tier_cheap(self):
        """Test inference of CHEAP quality tier."""
        resolver = LlamaCppCapabilityResolver()

        assert resolver._infer_quality_tier("llama-3-7b-instruct") == QualityTier.CHEAP
        assert resolver._infer_quality_tier("mistral-8b") == QualityTier.CHEAP
        assert resolver._infer_quality_tier("qwen-13b-chat") == QualityTier.CHEAP
        assert resolver._infer_quality_tier("model-14b") == QualityTier.CHEAP

    def test_infer_quality_tier_unknown(self):
        """Test inference of UNKNOWN quality tier."""
        resolver = LlamaCppCapabilityResolver()

        assert resolver._infer_quality_tier("llama-3-instruct") == QualityTier.UNKNOWN
        assert resolver._infer_quality_tier("model-3b") == QualityTier.UNKNOWN
        assert resolver._infer_quality_tier("model-500m") == QualityTier.UNKNOWN
        assert resolver._infer_quality_tier("custom-model") == QualityTier.UNKNOWN

    def test_infer_quality_tier_case_insensitive(self):
        """Test case insensitivity."""
        resolver = LlamaCppCapabilityResolver()

        assert resolver._infer_quality_tier("LLAMA-3-70B") == QualityTier.BEST
        assert resolver._infer_quality_tier("MISTRAL-32B") == QualityTier.BALANCED
        assert resolver._infer_quality_tier("QWEN-7B") == QualityTier.CHEAP


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_capability_resolution_workflow(self):
        """Test full capability resolution workflow with realistic data."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-70b-instruct-q4",
            family="llama-3",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "default_generation_settings": {
                "n_ctx": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "chat_template": "{% if messages[0]['role'] == 'system' %}...with tool support...",
        }
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            capabilities = await resolver.get_capabilities(spec)

            # Should have full capabilities
            assert capabilities.max_context == 8192
            assert capabilities.supports_tools is True
            assert capabilities.supports_streaming is True
            assert capabilities.supports_system_messages is True
            assert capabilities.supports_json_mode is True
            assert capabilities.supports_vision is False
            assert capabilities.quality_tier == QualityTier.BEST
            assert capabilities.source == "llamacpp_props"
            assert len(capabilities.known_params) == 6

    @pytest.mark.asyncio
    async def test_minimal_server_props(self):
        """Test with minimal server props."""
        resolver = LlamaCppCapabilityResolver()
        spec = ModelSpec(
            provider=Provider.LLAMA_CPP.value,
            name="llama-3-7b",
            family="llama-3",
        )

        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            capabilities = await resolver.get_capabilities(spec)

            # Should have minimal capabilities
            assert capabilities.max_context is None
            assert capabilities.supports_tools is False
            assert capabilities.supports_streaming is True  # Always supported
            assert capabilities.supports_json_mode is True  # Always supported
            assert capabilities.quality_tier == QualityTier.CHEAP  # 7b model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
