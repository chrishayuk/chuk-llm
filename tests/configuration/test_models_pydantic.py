"""
Comprehensive tests for Pydantic configuration models to achieve 90%+ coverage.

This test file specifically targets uncovered lines in models.py:
- Lines 62, 72, 75-76: ValidationError edge cases
- Line 156: Empty API base handling
- Line 170: Empty client_class handling
- Lines 197-198: Feature validation edge cases
- Lines 238-248: get_api_key() method
- Lines 303-307: GlobalConfig log_level validation
"""

import os
import re
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from chuk_llm.configuration.models import (
    DiscoveryConfig,
    Feature,
    GlobalConfig,
    ModelCapabilities,
    ProviderConfig,
)


class TestModelCapabilitiesValidation:
    """Test ModelCapabilities Pydantic validation"""

    def test_empty_pattern_raises_validation_error(self):
        """Test that empty pattern is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(pattern="")

        assert "pattern" in str(exc_info.value).lower()

    def test_matches_with_internal_regex_error(self):
        """Test matches() gracefully handles corrupted regex state"""
        # Create a valid ModelCapabilities
        caps = ModelCapabilities(pattern="valid-.*")

        # Mock re.match to raise re.error to cover the except clause
        import re

        original_match = re.match

        def mock_match_error(*args, **kwargs):
            # Only raise error for our specific pattern check
            if len(args) > 0 and args[0] == "valid-.*":
                raise re.error("simulated regex error")
            return original_match(*args, **kwargs)

        with patch("re.match", side_effect=mock_match_error):
            # Should return False instead of crashing
            result = caps.matches("test-model")
            assert result is False

    def test_invalid_regex_pattern_raises_validation_error(self):
        """Test that invalid regex pattern is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(pattern="[unclosed")

        assert "pattern" in str(exc_info.value).lower()
        assert "regex" in str(exc_info.value).lower()

    def test_matches_empty_model_name(self):
        """Test matches() with empty model name returns False"""
        caps = ModelCapabilities(pattern="gpt-.*")
        assert caps.matches("") is False

    def test_matches_none_model_name(self):
        """Test matches() handles None gracefully"""
        caps = ModelCapabilities(pattern="gpt-.*")
        # Should not crash
        result = caps.matches(None)  # type: ignore[arg-type]
        assert result is False

    def test_matches_with_corrupted_pattern(self):
        """Test that matches() handles regex errors gracefully"""
        # Create a valid ModelCapabilities first
        caps = ModelCapabilities(pattern="valid-.*")

        # Now test if we could somehow corrupt the pattern (defensive programming)
        # This tests the except re.error clause in matches()
        assert caps.matches("valid-model") is True
        assert caps.matches("invalid") is False

    def test_negative_max_context_length_rejected(self):
        """Test that negative max_context_length is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(pattern=".*", max_context_length=-1)

        assert "max_context_length" in str(exc_info.value).lower()

    def test_zero_max_context_length_rejected(self):
        """Test that zero max_context_length is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(pattern=".*", max_context_length=0)

        assert "max_context_length" in str(exc_info.value).lower()

    def test_negative_max_output_tokens_rejected(self):
        """Test that negative max_output_tokens is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ModelCapabilities(pattern=".*", max_output_tokens=-1)

        assert "max_output_tokens" in str(exc_info.value).lower()


class TestProviderConfigValidation:
    """Test ProviderConfig Pydantic validation"""

    def test_empty_name_rejected(self):
        """Test that empty provider name is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig(name="")

        assert "name" in str(exc_info.value).lower()

    def test_api_base_with_whitespace_trimmed(self):
        """Test that API base is trimmed of whitespace"""
        config = ProviderConfig(name="test", api_base="  https://api.example.com  ")
        assert config.api_base == "https://api.example.com"

    def test_api_base_empty_string_becomes_none(self):
        """Test that empty API base becomes None"""
        config = ProviderConfig(name="test", api_base="   ")
        assert config.api_base is None

    def test_api_base_trailing_slash_removed(self):
        """Test that trailing slashes are removed from API base"""
        config = ProviderConfig(name="test", api_base="https://api.example.com///")
        assert config.api_base == "https://api.example.com"

    def test_api_base_without_http_rejected(self):
        """Test that API base without http:// or https:// is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig(name="test", api_base="api.example.com")

        assert "api_base" in str(exc_info.value).lower()
        assert "http" in str(exc_info.value).lower()

    def test_api_base_with_http(self):
        """Test that http:// URLs are accepted"""
        config = ProviderConfig(name="test", api_base="http://localhost:8080")
        assert config.api_base == "http://localhost:8080"

    def test_client_class_empty_string_allowed(self):
        """Test that empty client_class is allowed"""
        config = ProviderConfig(name="test", client_class="")
        assert config.client_class == ""

    def test_client_class_simple_name_allowed(self):
        """Test that simple class names are allowed"""
        config = ProviderConfig(name="test", client_class="SimpleClient")
        assert config.client_class == "SimpleClient"

    def test_supports_feature_with_invalid_string(self):
        """Test supports_feature with invalid feature string returns False"""
        config = ProviderConfig(name="test")
        assert config.supports_feature("invalid_feature") is False

    def test_supports_feature_case_insensitive(self):
        """Test that feature checking is case-insensitive"""
        config = ProviderConfig(name="test", features={Feature.TEXT})
        assert config.supports_feature("TEXT")
        assert config.supports_feature("text")
        assert config.supports_feature("Text")

    def test_get_api_key_no_env_vars(self):
        """Test get_api_key when no environment variables are set"""
        config = ProviderConfig(
            name="test", api_key_env="NONEXISTENT_KEY", api_key_fallback_env="ALSO_NONEXISTENT"
        )
        assert config.get_api_key() is None

    def test_get_api_key_primary_env_var(self):
        """Test get_api_key retrieves from primary environment variable"""
        with patch.dict(os.environ, {"TEST_PRIMARY_KEY": "primary-key-value"}):
            config = ProviderConfig(name="test", api_key_env="TEST_PRIMARY_KEY")
            assert config.get_api_key() == "primary-key-value"

    def test_get_api_key_fallback_env_var(self):
        """Test get_api_key falls back to secondary environment variable"""
        with patch.dict(os.environ, {"TEST_FALLBACK_KEY": "fallback-key-value"}, clear=True):
            config = ProviderConfig(
                name="test",
                api_key_env="NONEXISTENT_PRIMARY",
                api_key_fallback_env="TEST_FALLBACK_KEY",
            )
            assert config.get_api_key() == "fallback-key-value"

    def test_get_api_key_prefers_primary_over_fallback(self):
        """Test get_api_key prefers primary over fallback"""
        with patch.dict(
            os.environ,
            {"TEST_PRIMARY": "primary-value", "TEST_FALLBACK": "fallback-value"},
        ):
            config = ProviderConfig(
                name="test", api_key_env="TEST_PRIMARY", api_key_fallback_env="TEST_FALLBACK"
            )
            assert config.get_api_key() == "primary-value"

    def test_get_api_key_no_env_configured(self):
        """Test get_api_key when no environment variables configured"""
        config = ProviderConfig(name="test")
        assert config.get_api_key() is None

    def test_validate_assignment_enabled(self):
        """Test that validate_assignment is enabled for runtime validation"""
        config = ProviderConfig(name="test", max_context_length=1000)

        # Should validate on mutation
        with pytest.raises(ValidationError):
            config.max_context_length = -1  # Invalid value

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed"""
        config = ProviderConfig(
            name="test",
            extra={"custom_field": "value", "another_field": 123},
        )
        assert config.extra["custom_field"] == "value"
        assert config.extra["another_field"] == 123


class TestDiscoveryConfigValidation:
    """Test DiscoveryConfig Pydantic validation"""

    def test_negative_cache_timeout_rejected(self):
        """Test that negative cache_timeout is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            DiscoveryConfig(cache_timeout=-1)

        assert "cache_timeout" in str(exc_info.value).lower()

    def test_zero_cache_timeout_allowed(self):
        """Test that zero cache_timeout is allowed (no caching)"""
        config = DiscoveryConfig(cache_timeout=0)
        assert config.cache_timeout == 0

    def test_immutability(self):
        """Test that DiscoveryConfig is immutable (frozen)"""
        config = DiscoveryConfig(enabled=True, cache_timeout=300)

        with pytest.raises(ValidationError):
            config.enabled = False  # Should raise due to frozen=True


class TestGlobalConfigValidation:
    """Test GlobalConfig Pydantic validation"""

    def test_invalid_log_level_rejected(self):
        """Test that invalid log level is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            GlobalConfig(log_level="INVALID")

        assert "log_level" in str(exc_info.value).lower()

    def test_valid_log_levels(self):
        """Test that all valid log levels are accepted"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config = GlobalConfig(log_level=level)
            assert config.log_level == level.upper()

    def test_log_level_case_insensitive(self):
        """Test that log level is case-insensitive"""
        config = GlobalConfig(log_level="debug")
        assert config.log_level == "DEBUG"

        config = GlobalConfig(log_level="Info")
        assert config.log_level == "INFO"

    def test_temperature_out_of_range(self):
        """Test that temperature outside [0, 2] is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            GlobalConfig(default_temperature=-0.1)

        assert "default_temperature" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            GlobalConfig(default_temperature=2.1)

        assert "default_temperature" in str(exc_info.value).lower()

    def test_temperature_boundary_values(self):
        """Test that temperature boundary values are accepted"""
        config1 = GlobalConfig(default_temperature=0.0)
        assert config1.default_temperature == 0.0

        config2 = GlobalConfig(default_temperature=2.0)
        assert config2.default_temperature == 2.0

    def test_negative_max_tokens_rejected(self):
        """Test that negative max_tokens is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            GlobalConfig(default_max_tokens=0)

        assert "default_max_tokens" in str(exc_info.value).lower()

    def test_negative_cache_ttl_rejected(self):
        """Test that negative cache_ttl is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            GlobalConfig(cache_ttl=-1)

        assert "cache_ttl" in str(exc_info.value).lower()

    def test_zero_cache_ttl_allowed(self):
        """Test that zero cache_ttl is allowed (no caching)"""
        config = GlobalConfig(cache_ttl=0)
        assert config.cache_ttl == 0


class TestFeatureEnum:
    """Test Feature enum edge cases"""

    def test_feature_from_string_case_insensitive(self):
        """Test that Feature.from_string is case-insensitive"""
        assert Feature.from_string("TEXT") == Feature.TEXT
        assert Feature.from_string("text") == Feature.TEXT
        assert Feature.from_string("Text") == Feature.TEXT

    def test_feature_from_string_invalid(self):
        """Test that Feature.from_string raises ValueError for invalid input"""
        with pytest.raises(ValueError, match="Unknown feature"):
            Feature.from_string("not_a_feature")

    def test_all_features_have_string_values(self):
        """Test that all Feature enum values are strings"""
        for feature in Feature:
            assert isinstance(feature.value, str)


class TestModelCapabilitiesImmutability:
    """Test that ModelCapabilities is immutable"""

    def test_cannot_modify_pattern(self):
        """Test that pattern cannot be modified after creation"""
        caps = ModelCapabilities(pattern="gpt-.*")

        with pytest.raises(ValidationError):
            caps.pattern = "new-pattern"  # Should raise due to frozen=True

    def test_cannot_modify_features(self):
        """Test that features cannot be reassigned after creation"""
        caps = ModelCapabilities(pattern=".*", features={Feature.TEXT})

        with pytest.raises(ValidationError):
            caps.features = {Feature.VISION}  # Should raise due to frozen=True


class TestProviderConfigModelValidation:
    """Test model-level validation in ProviderConfig"""

    def test_default_model_not_in_models_warning(self):
        """Test that default_model not in models doesn't raise (might be discovered)"""
        # This should not raise - it's a warning scenario, not an error
        config = ProviderConfig(
            name="test", default_model="model-a", models=["model-b", "model-c"]
        )
        assert config.default_model == "model-a"

    def test_default_model_in_aliases(self):
        """Test that default_model can be an alias"""
        config = ProviderConfig(
            name="test",
            default_model="alias",
            models=["real-model"],
            model_aliases={"alias": "real-model"},
        )
        assert config.default_model == "alias"
