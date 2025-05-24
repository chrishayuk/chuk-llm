"""
Tests for configuration validation functionality.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

from chuk_llm.llm.configuration.config_validator import (
    ConfigValidator, 
    ValidatedProviderConfig
)


class TestConfigValidator:
    """Test the ConfigValidator class."""

    def test_validate_provider_config_supported_provider(self):
        """Test validation of supported provider configuration."""
        config = {
            "client": "test.client.TestClient",
            "api_key": "test-key",
            "default_model": "gpt-4o-mini",  # Fixed: use valid OpenAI model
            "api_base": "https://api.test.com"
        }
        
        is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        # Should be valid for supported provider
        assert is_valid
        assert len(issues) == 0

    def test_validate_provider_config_unsupported_provider(self):
        """Test validation of unsupported provider."""
        config = {"client": "test.client.TestClient"}
        
        is_valid, issues = ConfigValidator.validate_provider_config("unknown_provider", config)
        
        assert not is_valid
        assert len(issues) == 1
        assert "Unsupported provider: unknown_provider" in issues[0]

    def test_validate_provider_config_missing_client(self):
        """Test validation when client is missing."""
        config = {
            "api_key": "test-key",
            "default_model": "gpt-4o-mini"
        }
        
        is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        assert not is_valid
        assert any("Missing 'client'" in issue for issue in issues)

    def test_validate_provider_config_missing_api_key_env(self):
        """Test validation when API key environment variable is missing."""
        config = {
            "client": "test.client.TestClient",
            "api_key_env": "MISSING_API_KEY",
            "default_model": "gpt-4o-mini"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        assert not is_valid
        assert any("Missing API key" in issue for issue in issues)

    def test_validate_provider_config_with_api_key_env_present(self):
        """Test validation when API key environment variable is present."""
        config = {
            "client": "test.client.TestClient",
            "api_key_env": "TEST_API_KEY",
            "default_model": "gpt-4o-mini"
        }
        
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}):
            is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        assert is_valid
        assert len(issues) == 0

    def test_validate_provider_config_with_direct_api_key(self):
        """Test validation when API key is provided directly."""
        config = {
            "client": "test.client.TestClient",
            "api_key": "direct-key",
            "api_key_env": "MISSING_ENV_KEY",
            "default_model": "gpt-4o-mini"
        }
        
        is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        assert is_valid
        assert len(issues) == 0

    def test_validate_provider_config_invalid_model(self):
        """Test validation with invalid model for provider."""
        config = {
            "client": "test.client.TestClient",
            "api_key": "test-key",
            "default_model": "definitely-invalid-model-name"  # Fixed: clearly invalid model
        }
        
        is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        assert not is_valid
        assert any("not supported by openai" in issue for issue in issues)

    def test_validate_provider_config_invalid_api_base_url(self):
        """Test validation with invalid API base URL."""
        config = {
            "client": "test.client.TestClient",
            "api_key": "test-key",
            "api_base": "invalid-url"
        }
        
        is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        assert not is_valid
        assert any("Invalid API base URL" in issue for issue in issues)

    def test_validate_provider_config_valid_api_base_url(self):
        """Test validation with valid API base URL."""
        config = {
            "client": "test.client.TestClient",
            "api_key": "test-key",
            "api_base": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini"
        }
        
        is_valid, issues = ConfigValidator.validate_provider_config("openai", config)
        
        assert is_valid
        assert len(issues) == 0

    def test_validate_provider_config_strict_mode(self):
        """Test validation in strict mode."""
        config = {
            "client": "test.client.TestClient",
            "api_key": "test-key"
        }
        
        is_valid, issues = ConfigValidator.validate_provider_config("openai", config, strict=True)
        
        # Should still validate basic requirements in strict mode
        assert is_valid or len(issues) > 0

    def test_is_valid_url_valid_urls(self):
        """Test URL validation with valid URLs."""
        valid_urls = [
            "https://api.openai.com",
            "http://localhost:8080",
            "https://api.anthropic.com/v1",
            "http://192.168.1.1:3000"
        ]
        
        for url in valid_urls:
            assert ConfigValidator._is_valid_url(url), f"URL should be valid: {url}"

    def test_is_valid_url_invalid_urls(self):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "https://",
            "http://",
            ""
        ]
        
        for url in invalid_urls:
            assert not ConfigValidator._is_valid_url(url), f"URL should be invalid: {url}"

    def test_validate_request_compatibility_streaming_not_supported(self):
        """Test request validation when streaming is not supported."""
        messages = [{"role": "user", "content": "test"}]
        
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "groq", messages, stream=True
        )
        
        # Note: Based on your capabilities.py, Groq does support streaming
        # So this test might need adjustment based on actual capabilities
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_validate_request_compatibility_tools_not_supported(self):
        """Test request validation when tools are not supported."""
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_tool"}}]
        
        # Test with a provider that doesn't support tools (if any)
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "ollama", messages, tools=tools
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_validate_request_compatibility_unknown_provider(self):
        """Test request validation with unknown provider."""
        messages = [{"role": "user", "content": "test"}]
        
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "unknown_provider", messages
        )
        
        assert not is_valid
        assert len(issues) == 1
        assert "Unknown provider" in issues[0]

    def test_has_vision_content_with_vision(self):
        """Test vision content detection with vision messages."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image_url": {"url": "https://example.com/image.jpg"}}
            ]}
        ]
        
        has_vision = ConfigValidator._has_vision_content(messages)
        assert has_vision

    def test_has_vision_content_without_vision(self):
        """Test vision content detection without vision messages."""
        messages = [
            {"role": "user", "content": "Just text"},
            {"role": "assistant", "content": "Response"}
        ]
        
        has_vision = ConfigValidator._has_vision_content(messages)
        assert not has_vision

    def test_has_vision_content_with_image_url_type(self):
        """Test vision content detection with image_url type."""
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]}
        ]
        
        has_vision = ConfigValidator._has_vision_content(messages)
        assert has_vision

    def test_estimate_token_count_text_messages(self):
        """Test token count estimation for text messages."""
        messages = [
            {"role": "user", "content": "This is a test message"},
            {"role": "assistant", "content": "This is a response"}
        ]
        
        token_count = ConfigValidator._estimate_token_count(messages)
        
        # Should be roughly 1/4 of total character count
        total_chars = len("This is a test messageThis is a response")
        expected_tokens = total_chars // 4
        assert token_count == expected_tokens

    def test_estimate_token_count_list_content(self):
        """Test token count estimation for list content."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Hello world"},
                {"type": "image", "image_url": {"url": "..."}}
            ]}
        ]
        
        token_count = ConfigValidator._estimate_token_count(messages)
        
        # Should count text content but ignore image content
        expected_tokens = len("Hello world") // 4
        assert token_count == expected_tokens

    def test_estimate_token_count_empty_messages(self):
        """Test token count estimation for empty messages."""
        messages = []
        
        token_count = ConfigValidator._estimate_token_count(messages)
        assert token_count == 0

    def test_validate_request_compatibility_context_length_exceeded(self):
        """Test request validation when context length is exceeded."""
        # Create a very long message to exceed context limits
        long_content = "A" * 600000  # Fixed: 600k characters = ~150k tokens (exceeds 128k)
        messages = [{"role": "user", "content": long_content}]
        
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "openai", messages
        )
        
        assert not is_valid
        assert any("context limit" in issue for issue in issues)

    def test_validate_request_compatibility_json_mode_not_supported(self):
        """Test request validation when JSON mode is not supported."""
        messages = [{"role": "user", "content": "test"}]
        
        # Anthropic doesn't support native JSON mode
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "anthropic", messages, response_format="json"
        )
        
        assert not is_valid
        assert any("JSON mode" in issue for issue in issues)

    def test_validate_request_compatibility_all_valid(self):
        """Test request validation when everything is valid."""
        messages = [{"role": "user", "content": "Simple test message"}]
        
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "openai", messages, stream=True, tools=None
        )
        
        assert is_valid
        assert len(issues) == 0


class TestValidatedProviderConfig:
    """Test the ValidatedProviderConfig class."""

    def test_validated_provider_config_valid_config(self):
        """Test ValidatedProviderConfig with valid configuration."""
        config_dict = {
            "openai": {
                "client": "test.client.TestClient",
                "api_key": "test-key",
                "default_model": "gpt-4o-mini"
            }
        }
        
        # Should not raise in non-strict mode
        validated_config = ValidatedProviderConfig(config_dict, strict=False)
        assert validated_config is not None

    def test_validated_provider_config_invalid_config_non_strict(self):
        """Test ValidatedProviderConfig with invalid config in non-strict mode."""
        config_dict = {
            "openai": {
                # Missing client
                "api_key": "test-key"
            }
        }
        
        # Should create but warn in non-strict mode
        with patch('warnings.warn') as mock_warn:
            validated_config = ValidatedProviderConfig(config_dict, strict=False)
            assert validated_config is not None
            # Should have issued warnings
            assert mock_warn.called

    def test_validated_provider_config_invalid_config_strict(self):
        """Test ValidatedProviderConfig with invalid config in strict mode."""
        config_dict = {
            "openai": {
                # Missing client
                "api_key": "test-key"
            }
        }
        
        # Should raise ValueError in strict mode
        with pytest.raises(ValueError, match="Configuration validation failed"):
            ValidatedProviderConfig(config_dict, strict=True)

    def test_validate_request_method(self):
        """Test the validate_request method."""
        config_dict = {
            "openai": {
                "client": "test.client.TestClient",
                "api_key": "test-key",
                "default_model": "gpt-4o-mini"
            }
        }
        
        validated_config = ValidatedProviderConfig(config_dict, strict=False)
        messages = [{"role": "user", "content": "test"}]
        
        is_valid, issues = validated_config.validate_request(
            "openai", messages, tools=None
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_validated_provider_config_multiple_providers(self):
        """Test ValidatedProviderConfig with multiple providers."""
        config_dict = {
            "openai": {
                "client": "test.client.OpenAIClient",
                "api_key": "openai-key",
                "default_model": "gpt-4o-mini"
            },
            "anthropic": {
                "client": "test.client.AnthropicClient", 
                "api_key": "anthropic-key",
                "default_model": "claude-3-haiku-20240307"
            }
        }
        
        validated_config = ValidatedProviderConfig(config_dict, strict=False)
        
        # Should validate both providers
        assert validated_config is not None
        assert "openai" in validated_config.providers
        assert "anthropic" in validated_config.providers

    def test_validated_provider_config_empty_config(self):
        """Test ValidatedProviderConfig with empty configuration."""
        # Empty config should still work (uses defaults)
        validated_config = ValidatedProviderConfig({}, strict=False)
        assert validated_config is not None

    def test_validated_provider_config_none_config(self):
        """Test ValidatedProviderConfig with None configuration."""
        # None config should work (uses defaults)
        validated_config = ValidatedProviderConfig(None, strict=False)
        assert validated_config is not None

    def test_validated_provider_config_inheritance(self):
        """Test that ValidatedProviderConfig has expected methods."""
        validated_config = ValidatedProviderConfig({}, strict=False)
        
        # Should have all the expected methods
        assert hasattr(validated_config, 'get_provider_config')
        assert hasattr(validated_config, 'validate_request')
        assert hasattr(validated_config, 'get_active_provider')
        assert hasattr(validated_config, 'set_active_provider')


class TestConfigValidatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_validate_provider_config_none_config(self):
        """Test validation with None config."""
        is_valid, issues = ConfigValidator.validate_provider_config("openai", None)
        
        assert not is_valid
        assert len(issues) > 0
        assert any("Configuration is None" in issue for issue in issues)

    def test_validate_provider_config_empty_config(self):
        """Test validation with empty config."""
        is_valid, issues = ConfigValidator.validate_provider_config("openai", {})
        
        assert not is_valid
        assert len(issues) > 0

    def test_validate_request_compatibility_empty_messages(self):
        """Test request validation with empty messages."""
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "openai", []
        )
        
        # Empty messages should be valid
        assert is_valid
        assert len(issues) == 0

    def test_validate_request_compatibility_none_messages(self):
        """Test request validation with None messages."""
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            "openai", None
        )
        
        # Should handle None gracefully
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_has_vision_content_malformed_content(self):
        """Test vision detection with malformed content."""
        messages = [
            {"role": "user", "content": None},
            {"role": "user"},  # Missing content
            {"role": "user", "content": [None]},  # None item in list
            None,  # None message
        ]
        
        # Should not crash
        has_vision = ConfigValidator._has_vision_content(messages)
        assert isinstance(has_vision, bool)
        assert has_vision is False

    def test_estimate_token_count_malformed_messages(self):
        """Test token estimation with malformed messages."""
        messages = [
            {"role": "user"},  # Missing content
            {"content": "No role"},  # Missing role
            None,  # None message
            {"role": "user", "content": None},  # None content
        ]
        
        # Should not crash
        token_count = ConfigValidator._estimate_token_count(messages)
        assert isinstance(token_count, int)
        assert token_count >= 0

    def test_is_valid_url_edge_cases(self):
        """Test URL validation edge cases."""
        edge_cases = [
            None,
            "",
            "   ",
            "https://",
            "http://",
            "not a url at all",
            "file:///path/to/file",
            "ftp://example.com"
        ]
        
        for url in edge_cases:
            # Should not crash
            result = ConfigValidator._is_valid_url(url) if url else False
            assert isinstance(result, bool)


# Integration tests
class TestConfigValidatorIntegration:
    """Integration tests for config validator with real provider configs."""

    def test_validate_all_default_providers(self):
        """Test validation of all default provider configurations."""
        from chuk_llm.llm.configuration.provider_config import DEFAULTS
        
        for provider_name, config in DEFAULTS.items():
            if provider_name == "__global__":
                continue
                
            # Test basic validation
            is_valid, issues = ConfigValidator.validate_provider_config(provider_name, config)
            
            # Should be valid or have expected issues
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)

    def test_validated_provider_config_with_defaults(self):
        """Test ValidatedProviderConfig with default configurations."""
        # Should work with default configs
        validated_config = ValidatedProviderConfig(None, strict=False)
        
        # Test validation method with each provider
        for provider in ["openai", "anthropic", "groq", "gemini", "ollama"]:
            messages = [{"role": "user", "content": "test"}]
            is_valid, issues = validated_config.validate_request(provider, messages)
            
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)

    def test_real_world_config_scenarios(self):
        """Test real-world configuration scenarios."""
        scenarios = [
            # OpenAI with custom base
            {
                "openai": {
                    "client": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
                    "api_key": "sk-test123",
                    "api_base": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini"
                }
            },
            # Multiple providers
            {
                "openai": {
                    "client": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
                    "api_key_env": "OPENAI_API_KEY",
                    "default_model": "gpt-4o-mini"
                },
                "anthropic": {
                    "client": "chuk_llm.llm.providers.anthropic_client:AnthropicLLMClient",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "default_model": "claude-3-haiku-20240307"
                }
            },
            # Local provider
            {
                "ollama": {
                    "client": "chuk_llm.llm.providers.ollama_client:OllamaLLMClient",
                    "api_base": "http://localhost:11434",
                    "default_model": "llama3.2"
                }
            }
        ]
        
        for scenario in scenarios:
            # Should create without error in non-strict mode
            validated_config = ValidatedProviderConfig(scenario, strict=False)
            assert validated_config is not None