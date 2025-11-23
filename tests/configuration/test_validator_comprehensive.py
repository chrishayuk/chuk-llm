"""
Comprehensive tests for ConfigValidator to achieve 90%+ coverage.

Missing lines to cover:
- Line 40: Invalid API base URL check
- Lines 81-82: Vision content check
- Lines 88-94: JSON mode and exception handling
- Line 119: Empty messages list in _has_vision_content
- Line 123: None message in messages list
"""

import os
from unittest.mock import Mock, patch

import pytest

from chuk_llm.configuration.models import Feature, ProviderConfig
from chuk_llm.configuration.validator import ConfigValidator


class TestValidateProviderConfigCoverage:
    """Test validate_provider_config to cover missing lines"""

    def test_invalid_api_base_url_detected(self):
        """Test that invalid API base URL is detected (line 40)"""
        # Create a provider with invalid URL that passes Pydantic validation
        # (Pydantic checks for http/https prefix, not full URL validity)
        provider = ProviderConfig(
            name="test",
            client_class="TestClient",
            api_base="http://", # Valid prefix but not a complete URL
            default_model="test-model",
        )

        # Now manually set an invalid URL to test the validator
        # (We need to bypass Pydantic validation for this test)
        provider.__dict__['api_base'] = "http://not a valid url with spaces"

        valid, issues = ConfigValidator.validate_provider_config(provider)

        # Should detect invalid URL
        assert not valid
        assert any("Invalid API base URL" in issue for issue in issues)

    def test_api_base_none_skips_validation(self):
        """Test that None api_base doesn't trigger validation"""
        provider = ProviderConfig(
            name="test",
            client_class="TestClient",
            api_base=None,
            default_model="test-model",
        )

        valid, issues = ConfigValidator.validate_provider_config(provider)

        # Should be valid - no URL to validate
        assert valid
        assert not any("Invalid API base URL" in issue for issue in issues)


class TestValidateRequestCompatibilityVision:
    """Test vision-related validation (lines 81-82)"""

    def test_vision_content_not_supported(self):
        """Test detection when vision content used without support"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "url": "https://example.com/image.jpg"},
                ],
            }
        ]

        with patch("chuk_llm.configuration.unified_config.get_config") as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature = Mock(
                side_effect=lambda f, m=None: f != Feature.VISION
            )

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test", model="test-model", messages=messages
            )

            assert not valid
            assert any("doesn't support vision" in issue for issue in issues)

    def test_vision_content_with_image_url_type(self):
        """Test vision detection with image_url type"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]

        with patch("chuk_llm.configuration.unified_config.get_config") as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature = Mock(
                side_effect=lambda f, m=None: f != Feature.VISION
            )

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test", model="test-model", messages=messages
            )

            assert not valid
            assert any("vision" in issue.lower() for issue in issues)


class TestValidateRequestCompatibilityJSON:
    """Test JSON mode validation (lines 88-94)"""

    def test_json_mode_not_supported(self):
        """Test detection when JSON mode used without support"""
        with patch("chuk_llm.configuration.unified_config.get_config") as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature = Mock(
                side_effect=lambda f, m=None: f != Feature.JSON_MODE
            )

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test",
                model="test-model",
                response_format="json",
            )

            assert not valid
            assert any("doesn't support JSON mode" in issue for issue in issues)

    def test_json_mode_supported(self):
        """Test that JSON mode passes when supported"""
        with patch("chuk_llm.configuration.unified_config.get_config") as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature = Mock(return_value=True)

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test",
                model="test-model",
                response_format="json",
            )

            assert valid
            assert len(issues) == 0

    def test_exception_handling_in_validation(self):
        """Test exception handling in validate_request_compatibility"""
        with patch("chuk_llm.configuration.unified_config.get_config") as mock_get_config:
            # Make get_config raise an exception
            mock_get_config.side_effect = Exception("Config error")

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test",
                model="test-model",
            )

            assert not valid
            assert any("Configuration error" in issue for issue in issues)

    def test_get_provider_raises_exception(self):
        """Test exception when getting provider"""
        with patch("chuk_llm.configuration.unified_config.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.get_provider.side_effect = ValueError("Unknown provider")
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="nonexistent",
                model="test-model",
            )

            assert not valid
            assert any("Configuration error" in issue for issue in issues)


class TestHasVisionContentEdgeCases:
    """Test _has_vision_content edge cases (lines 119, 123)"""

    def test_empty_messages_list(self):
        """Test _has_vision_content with empty messages list (line 119)"""
        result = ConfigValidator._has_vision_content([])
        assert result is False

    def test_none_messages_list(self):
        """Test _has_vision_content with None messages"""
        result = ConfigValidator._has_vision_content(None)  # type: ignore
        assert result is False

    def test_messages_with_none_items(self):
        """Test _has_vision_content with None items in list (line 123)"""
        messages = [
            None,  # None message - should be skipped
            {"role": "user", "content": "Hello"},
            None,  # Another None
        ]
        result = ConfigValidator._has_vision_content(messages)
        assert result is False

    def test_messages_with_empty_dict(self):
        """Test _has_vision_content with empty dict messages"""
        messages = [
            {},  # Empty dict
            {"role": "user"},  # No content
        ]
        result = ConfigValidator._has_vision_content(messages)
        assert result is False

    def test_mixed_none_and_vision_messages(self):
        """Test _has_vision_content with None and vision content mixed"""
        messages = [
            None,
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://example.com/img.jpg"},
                ],
            },
        ]
        result = ConfigValidator._has_vision_content(messages)
        assert result is True

    def test_string_content_no_vision(self):
        """Test _has_vision_content with string content (no vision)"""
        messages = [
            {"role": "user", "content": "Just text, no images"},
        ]
        result = ConfigValidator._has_vision_content(messages)
        assert result is False

    def test_list_content_no_image_type(self):
        """Test _has_vision_content with list content but no image types"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "audio", "url": "https://example.com/audio.mp3"},
                ],
            }
        ]
        result = ConfigValidator._has_vision_content(messages)
        assert result is False


class TestIsValidURL:
    """Test _is_valid_url method"""

    def test_valid_https_url(self):
        """Test valid HTTPS URL"""
        assert ConfigValidator._is_valid_url("https://api.example.com")
        assert ConfigValidator._is_valid_url("https://api.example.com/v1")
        assert ConfigValidator._is_valid_url("https://api.example.com:8080")

    def test_valid_http_url(self):
        """Test valid HTTP URL"""
        assert ConfigValidator._is_valid_url("http://localhost")
        assert ConfigValidator._is_valid_url("http://localhost:3000")
        assert ConfigValidator._is_valid_url("http://192.168.1.1")

    def test_invalid_urls(self):
        """Test various invalid URLs"""
        assert not ConfigValidator._is_valid_url("")
        assert not ConfigValidator._is_valid_url("not-a-url")
        assert not ConfigValidator._is_valid_url("ftp://example.com")
        assert not ConfigValidator._is_valid_url("http://")
        assert not ConfigValidator._is_valid_url("https://")

    def test_url_with_path(self):
        """Test URLs with paths"""
        assert ConfigValidator._is_valid_url("https://api.example.com/v1/chat")
        assert ConfigValidator._is_valid_url("http://localhost:8080/api")

    def test_ip_address_urls(self):
        """Test IP address URLs"""
        assert ConfigValidator._is_valid_url("http://127.0.0.1")
        assert ConfigValidator._is_valid_url("http://192.168.1.100:8080")
        assert ConfigValidator._is_valid_url("https://10.0.0.1/api")


class TestValidatorIntegration:
    """Integration tests for validator"""

    def test_complete_request_validation_all_features(self):
        """Test complete request with all features"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]

        tools = [{"type": "function", "function": {"name": "test"}}]

        with patch("chuk_llm.configuration.unified_config.get_config") as mock_get_config:
            mock_provider = Mock()
            # No features supported
            mock_provider.supports_feature = Mock(return_value=False)

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test",
                model="test-model",
                messages=messages,
                tools=tools,
                stream=True,
                response_format="json",
            )

            # Should have multiple issues
            assert not valid
            assert len(issues) >= 4  # streaming, tools, vision, json_mode
            assert any("streaming" in issue.lower() for issue in issues)
            assert any("function calling" in issue.lower() for issue in issues)
            assert any("vision" in issue.lower() for issue in issues)
            assert any("json mode" in issue.lower() for issue in issues)
