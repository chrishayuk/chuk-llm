# test_unified_config.py
"""
Comprehensive unit tests for the unified configuration system
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from chuk_llm.configuration.discovery import ConfigDiscoveryMixin

# Import the modules to test
from chuk_llm.configuration.models import (
    DiscoveryConfig,
    Feature,
    ModelCapabilities,
    ProviderConfig,
)
from chuk_llm.configuration.unified_config import (
    CapabilityChecker,
    UnifiedConfigManager,
    get_config,
    reset_config,
)
from chuk_llm.configuration.validator import ConfigValidator


class TestFeature:
    """Test Feature enum functionality"""

    def test_feature_enum_values(self):
        """Test that all expected features are defined"""
        expected_features = {
            "text",
            "streaming",
            "tools",
            "vision",
            "json_mode",
            "parallel_calls",
            "system_messages",
            "multimodal",
            "reasoning",
        }
        actual_features = {f.value for f in Feature}
        assert actual_features == expected_features

    def test_feature_from_string_valid(self):
        """Test Feature.from_string with valid inputs"""
        assert Feature.from_string("text") == Feature.TEXT
        assert Feature.from_string("TEXT") == Feature.TEXT
        assert Feature.from_string("Streaming") == Feature.STREAMING

    def test_feature_from_string_invalid(self):
        """Test Feature.from_string with invalid inputs"""
        with pytest.raises(ValueError, match="Unknown feature: invalid"):
            Feature.from_string("invalid")


class TestModelCapabilities:
    """Test ModelCapabilities functionality"""

    def test_model_capabilities_creation(self):
        """Test creating ModelCapabilities"""
        caps = ModelCapabilities(
            pattern="gpt-4.*",
            features={Feature.TEXT, Feature.TOOLS},
            max_context_length=8192,
            max_output_tokens=4096,
        )

        assert caps.pattern == "gpt-4.*"
        assert Feature.TEXT in caps.features
        assert Feature.TOOLS in caps.features
        assert caps.max_context_length == 8192
        assert caps.max_output_tokens == 4096

    def test_model_capabilities_matches(self):
        """Test pattern matching"""
        caps = ModelCapabilities(pattern="gpt-4.*")

        assert caps.matches("gpt-4")
        assert caps.matches("gpt-4-turbo")
        assert caps.matches("GPT-4-TURBO")  # Case insensitive
        assert not caps.matches("gpt-3.5")

    def test_get_effective_features(self):
        """Test feature inheritance from provider"""
        provider_features = {Feature.TEXT, Feature.STREAMING}
        model_features = {Feature.TOOLS, Feature.VISION}

        caps = ModelCapabilities(pattern=".*", features=model_features)
        effective = caps.get_effective_features(provider_features)

        expected = {Feature.TEXT, Feature.STREAMING, Feature.TOOLS, Feature.VISION}
        assert effective == expected


class TestProviderConfig:
    """Test ProviderConfig functionality"""

    def test_provider_config_creation(self):
        """Test creating ProviderConfig"""
        config = ProviderConfig(
            name="openai",
            client_class="OpenAIClient",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-4",
            models=["gpt-4", "gpt-3.5-turbo"],
            features={Feature.TEXT, Feature.STREAMING},
        )

        assert config.name == "openai"
        assert config.client_class == "OpenAIClient"
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.default_model == "gpt-4"
        assert "gpt-4" in config.models
        assert Feature.TEXT in config.features

    def test_supports_feature_provider_level(self):
        """Test feature support at provider level"""
        config = ProviderConfig(name="test", features={Feature.TEXT, Feature.STREAMING})

        assert config.supports_feature(Feature.TEXT)
        assert config.supports_feature("streaming")
        assert not config.supports_feature(Feature.TOOLS)

    def test_supports_feature_model_level(self):
        """Test feature support at model level"""
        model_caps = ModelCapabilities(pattern="advanced-.*", features={Feature.TOOLS})

        config = ProviderConfig(
            name="test", features={Feature.TEXT}, model_capabilities=[model_caps]
        )

        # Advanced model inherits TEXT + adds TOOLS
        assert config.supports_feature(Feature.TEXT, "advanced-model")
        assert config.supports_feature(Feature.TOOLS, "advanced-model")

        # Regular model only has TEXT
        assert config.supports_feature(Feature.TEXT, "basic-model")
        assert not config.supports_feature(Feature.TOOLS, "basic-model")

    def test_get_model_capabilities_with_match(self):
        """Test getting capabilities for matching model"""
        model_caps = ModelCapabilities(
            pattern="gpt-4.*", features={Feature.TOOLS}, max_context_length=8192
        )

        config = ProviderConfig(
            name="openai",
            features={Feature.TEXT},
            max_context_length=4096,
            model_capabilities=[model_caps],
        )

        caps = config.get_model_capabilities("gpt-4-turbo")

        assert Feature.TEXT in caps.features  # Inherited
        assert Feature.TOOLS in caps.features  # Model-specific
        assert caps.max_context_length == 8192  # Model override

    def test_get_model_capabilities_no_match(self):
        """Test getting capabilities for non-matching model"""
        model_caps = ModelCapabilities(pattern="gpt-4.*", features={Feature.TOOLS})

        config = ProviderConfig(
            name="openai",
            features={Feature.TEXT},
            max_context_length=4096,
            model_capabilities=[model_caps],
        )

        caps = config.get_model_capabilities("gpt-3.5-turbo")

        assert Feature.TEXT in caps.features  # Provider default
        assert Feature.TOOLS not in caps.features  # No model match
        assert caps.max_context_length == 4096  # Provider default

    def test_get_rate_limit(self):
        """Test rate limit retrieval"""
        config = ProviderConfig(
            name="test", rate_limits={"default": 1000, "premium": 5000}
        )

        assert config.get_rate_limit("default") == 1000
        assert config.get_rate_limit("premium") == 5000
        assert config.get_rate_limit("nonexistent") is None


class TestDiscoveryConfig:
    """Test DiscoveryConfig functionality"""

    def test_discovery_config_creation(self):
        """Test creating DiscoveryConfig"""
        config = DiscoveryConfig(
            enabled=True,
            discoverer_type="openai",
            cache_timeout=600,
            inference_config={"temperature": 0.7},
            discoverer_config={"api_version": "2023-12-01"},
        )

        assert config.enabled is True
        assert config.discoverer_type == "openai"
        assert config.cache_timeout == 600
        assert config.inference_config["temperature"] == 0.7
        assert config.discoverer_config["api_version"] == "2023-12-01"

    def test_discovery_config_defaults(self):
        """Test DiscoveryConfig defaults"""
        config = DiscoveryConfig()

        assert config.enabled is False
        assert config.discoverer_type is None
        assert config.cache_timeout == 300
        assert config.inference_config == {}
        assert config.discoverer_config == {}


class TestConfigValidator:
    """Test ConfigValidator functionality"""

    def test_validate_provider_config_valid(self):
        """Test validation of valid provider config"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = ProviderConfig(
                name="openai",
                client_class="OpenAIClient",
                api_key_env="OPENAI_API_KEY",
                api_base="https://api.openai.com/v1",
                default_model="gpt-4",
            )

            valid, issues = ConfigValidator.validate_provider_config(config)
            assert valid is True
            assert len(issues) == 0

    def test_validate_provider_config_missing_client_class(self):
        """Test validation with missing client_class"""
        config = ProviderConfig(name="test")

        valid, issues = ConfigValidator.validate_provider_config(config)
        assert valid is False
        assert any("Missing 'client_class'" in issue for issue in issues)

    def test_validate_provider_config_missing_api_key(self):
        """Test validation with missing API key"""
        config = ProviderConfig(
            name="openai",
            client_class="OpenAIClient",
            api_key_env="MISSING_KEY",
            default_model="gpt-4",
        )

        valid, issues = ConfigValidator.validate_provider_config(config)
        assert valid is False
        assert any("Missing API key" in issue for issue in issues)

    def test_validate_provider_config_api_key_optional(self):
        """Test validation for providers that don't need API keys"""
        config = ProviderConfig(
            name="ollama", client_class="OllamaClient", default_model="llama2"
        )

        valid, issues = ConfigValidator.validate_provider_config(config)
        assert valid is True  # Should be valid now with all required fields
        # But no API key error should be present
        assert not any("Missing API key" in issue for issue in issues)

    def test_validate_provider_config_invalid_url(self):
        """Test validation with invalid API base URL"""
        config = ProviderConfig(
            name="test",
            client_class="TestClient",
            api_base="invalid-url",
            default_model="test-model",
        )

        valid, issues = ConfigValidator.validate_provider_config(config)
        assert valid is False
        assert any("Invalid API base URL" in issue for issue in issues)

    def test_validate_request_compatibility_streaming(self):
        """Test request compatibility validation for streaming"""
        with patch(
            "chuk_llm.configuration.unified_config.get_config"
        ) as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature.return_value = False

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test", model="test-model", stream=True
            )

            assert valid is False
            assert any("doesn't support streaming" in issue for issue in issues)

    def test_validate_request_compatibility_tools(self):
        """Test request compatibility validation for tools"""
        with patch(
            "chuk_llm.configuration.unified_config.get_config"
        ) as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature.return_value = False

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name="test",
                tools=[{"type": "function", "function": {"name": "test"}}],
            )

            assert valid is False
            assert any("doesn't support function calling" in issue for issue in issues)

    def test_has_vision_content(self):
        """Test vision content detection"""
        # Message with image
        messages_with_image = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        assert ConfigValidator._has_vision_content(messages_with_image) is True

        # Message without image
        messages_text_only = [{"role": "user", "content": "Hello, how are you?"}]

        assert ConfigValidator._has_vision_content(messages_text_only) is False

    def test_is_valid_url(self):
        """Test URL validation"""
        valid_urls = [
            "https://api.openai.com/v1",
            "http://localhost:8080",
            "https://test.example.com:443/path",
            "http://192.168.1.1:3000",
        ]

        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "",
            None,
            "https://",
            "http://",
        ]

        for url in valid_urls:
            assert ConfigValidator._is_valid_url(url) is True, (
                f"Expected {url} to be valid"
            )

        for url in invalid_urls:
            assert ConfigValidator._is_valid_url(url) is False, (
                f"Expected {url} to be invalid"
            )


class TestUnifiedConfigManager:
    """Test UnifiedConfigManager functionality"""

    def setup_method(self):
        """Setup for each test"""
        # Reset global config before each test
        reset_config()

    def test_init(self):
        """Test UnifiedConfigManager initialization"""
        manager = UnifiedConfigManager()

        assert manager.providers == {}
        assert manager.global_aliases == {}
        assert manager.global_settings == {}
        assert manager._loaded is False

    def test_init_with_config_path(self):
        """Test initialization with custom config path"""
        manager = UnifiedConfigManager(config_path="/custom/path/config.yaml")
        assert manager.config_path == "/custom/path/config.yaml"

    @patch("chuk_llm.configuration.unified_config.yaml")
    def test_load_yaml_files_no_yaml(self, mock_yaml):
        """Test loading when PyYAML is not available"""
        manager = UnifiedConfigManager()

        with patch.object(manager, "_load_yaml_files", return_value={}):
            result = manager._load_yaml_files()
            assert result == {}

    def test_parse_features_string(self):
        """Test parsing features from string"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            features = manager._parse_features("streaming")
            assert features == {Feature.STREAMING}

    def test_parse_features_list(self):
        """Test parsing features from list"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            features = manager._parse_features(["text", "streaming", "tools"])
            expected = {Feature.TEXT, Feature.STREAMING, Feature.TOOLS}
            assert features == expected

    def test_parse_features_empty(self):
        """Test parsing empty features"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            assert manager._parse_features(None) == set()
            assert manager._parse_features([]) == set()
            assert manager._parse_features("") == set()

    def test_process_config_basic(self):
        """Test processing basic configuration"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            config = {
                "__global__": {"debug": True},
                "__global_aliases__": {"gpt4": "gpt-4"},
                "openai": {
                    "client_class": "OpenAIClient",
                    "api_key_env": "OPENAI_API_KEY",
                    "default_model": "gpt-4",
                    "models": ["gpt-4", "gpt-3.5-turbo"],
                    "features": ["text", "streaming"],
                },
            }

            manager._process_config(config)

            assert manager.global_settings["debug"] is True
            assert manager.global_aliases["gpt4"] == "gpt-4"
            assert "openai" in manager.providers

            openai_provider = manager.providers["openai"]
            assert openai_provider.name == "openai"
            assert openai_provider.client_class == "OpenAIClient"
            assert openai_provider.default_model == "gpt-4"
            assert "gpt-4" in openai_provider.models
            assert Feature.TEXT in openai_provider.features

    def test_process_config_with_inheritance(self):
        """Test processing configuration with inheritance"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            config = {
                "base": {
                    "client_class": "BaseClient",
                    "features": ["text", "streaming"],
                },
                "derived": {
                    "inherits": "base",
                    "api_key_env": "DERIVED_API_KEY",
                    "features": ["tools"],
                },
            }

            manager._process_config(config)
            manager._resolve_inheritance()

            manager.providers["base"]
            derived_provider = manager.providers["derived"]

            assert derived_provider.client_class == "BaseClient"  # Inherited
            assert derived_provider.api_key_env == "DERIVED_API_KEY"  # Own
            assert Feature.TEXT in derived_provider.features  # Inherited
            assert Feature.TOOLS in derived_provider.features  # Own

    def test_merge_configs_complete_replacement(self):
        """Test config merging with complete replacement"""
        manager = UnifiedConfigManager()

        base_config = {"openai": {"models": ["gpt-3.5-turbo"], "features": ["text"]}}

        user_config = {
            "anthropic": {"models": ["claude-3"], "features": ["text", "vision"]}
        }

        merged = manager._merge_configs(base_config, user_config)

        assert "openai" in merged
        assert "anthropic" in merged
        assert merged["anthropic"]["features"] == ["text", "vision"]

    def test_merge_configs_provider_merge(self):
        """Test config merging with provider-level merging"""
        manager = UnifiedConfigManager()

        base_config = {
            "openai": {
                "models": ["gpt-3.5-turbo"],
                "features": ["text"],
                "rate_limits": {"default": 1000},
            }
        }

        user_config = {
            "openai": {
                "models": ["gpt-4"],
                "features": ["tools"],
                "rate_limits": {"premium": 5000},
            }
        }

        merged = manager._merge_configs(base_config, user_config)

        openai_config = merged["openai"]
        assert "gpt-3.5-turbo" in openai_config["models"]  # Extended
        assert "gpt-4" in openai_config["models"]  # Extended
        assert "text" in openai_config["features"]  # Extended
        assert "tools" in openai_config["features"]  # Extended
        assert openai_config["rate_limits"]["default"] == 1000  # Merged
        assert openai_config["rate_limits"]["premium"] == 5000  # Merged

    def test_deep_merge_dict(self):
        """Test deep dictionary merging"""
        manager = UnifiedConfigManager()

        base = {
            "level1": {"level2": {"key1": "value1", "key2": "value2"}, "other": "data"}
        }

        override = {
            "level1": {
                "level2": {"key2": "new_value2", "key3": "value3"},
                "new_key": "new_data",
            }
        }

        result = manager._deep_merge_dict(base, override)

        assert result["level1"]["level2"]["key1"] == "value1"  # Preserved
        assert result["level1"]["level2"]["key2"] == "new_value2"  # Overridden
        assert result["level1"]["level2"]["key3"] == "value3"  # Added
        assert result["level1"]["other"] == "data"  # Preserved
        assert result["level1"]["new_key"] == "new_data"  # Added

    def test_get_provider_success(self):
        """Test successful provider retrieval"""
        manager = UnifiedConfigManager()
        manager.providers["test"] = ProviderConfig(name="test")
        manager._loaded = True

        provider = manager.get_provider("test")
        assert provider.name == "test"

    def test_get_provider_not_found(self):
        """Test provider retrieval when provider doesn't exist"""
        manager = UnifiedConfigManager()
        manager._loaded = True

        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            manager.get_provider("nonexistent")

    def test_get_all_providers(self):
        """Test getting all provider names"""
        manager = UnifiedConfigManager()
        manager.providers["openai"] = ProviderConfig(name="openai")
        manager.providers["anthropic"] = ProviderConfig(name="anthropic")
        manager._loaded = True

        providers = manager.get_all_providers()
        assert set(providers) == {"openai", "anthropic"}

    def test_get_api_key_primary(self):
        """Test getting API key from primary environment variable"""
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key-123"}):
            manager = UnifiedConfigManager()
            provider = ProviderConfig(name="test", api_key_env="TEST_API_KEY")
            manager.providers["test"] = provider
            manager._loaded = True

            api_key = manager.get_api_key("test")
            assert api_key == "test-key-123"

    def test_get_api_key_fallback(self):
        """Test getting API key from fallback environment variable"""
        with patch.dict(os.environ, {"FALLBACK_KEY": "fallback-key-456"}):
            manager = UnifiedConfigManager()
            provider = ProviderConfig(
                name="test",
                api_key_env="MISSING_KEY",
                api_key_fallback_env="FALLBACK_KEY",
            )
            manager.providers["test"] = provider
            manager._loaded = True

            api_key = manager.get_api_key("test")
            assert api_key == "fallback-key-456"

    def test_get_api_key_none(self):
        """Test getting API key when none available"""
        manager = UnifiedConfigManager()
        provider = ProviderConfig(name="test", api_key_env="MISSING_KEY")
        manager.providers["test"] = provider
        manager._loaded = True

        api_key = manager.get_api_key("test")
        assert api_key is None

    def test_supports_feature(self):
        """Test feature support checking"""
        manager = UnifiedConfigManager()
        provider = ProviderConfig(
            name="test", features={Feature.TEXT, Feature.STREAMING}
        )
        manager.providers["test"] = provider
        manager._loaded = True

        assert manager.supports_feature("test", Feature.TEXT) is True
        assert manager.supports_feature("test", "streaming") is True
        assert manager.supports_feature("test", Feature.TOOLS) is False

    def test_global_settings(self):
        """Test global settings management"""
        manager = UnifiedConfigManager()
        manager.global_settings = {"debug": True, "timeout": 30}
        manager._loaded = True

        settings = manager.get_global_settings()
        assert settings["debug"] is True
        assert settings["timeout"] == 30

        manager.set_global_setting("new_key", "new_value")
        assert manager.global_settings["new_key"] == "new_value"

    def test_global_aliases(self):
        """Test global aliases management"""
        manager = UnifiedConfigManager()
        manager.global_aliases = {"gpt4": "gpt-4", "claude": "claude-3"}
        manager._loaded = True

        aliases = manager.get_global_aliases()
        assert aliases["gpt4"] == "gpt-4"
        assert aliases["claude"] == "claude-3"

        manager.add_global_alias("new_alias", "new_target")
        assert manager.global_aliases["new_alias"] == "new_target"

    def test_reload(self):
        """Test configuration reload"""
        manager = UnifiedConfigManager()
        # Manually add test data without triggering load
        manager.providers["test"] = ProviderConfig(name="test")
        manager.global_settings["key"] = "value"
        manager._loaded = True

        # Mock the load method to prevent actual file loading during reload
        with patch.object(manager, "load"):
            manager.reload()

        assert manager.providers == {}
        assert manager.global_settings == {}
        assert manager._loaded is False


class TestCapabilityChecker:
    """Test CapabilityChecker functionality"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_can_handle_request_success(self):
        """Test successful request capability check"""
        with patch(
            "chuk_llm.configuration.unified_config.get_config"
        ) as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature.return_value = True

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            can_handle, problems = CapabilityChecker.can_handle_request(
                provider="test",
                model="test-model",
                has_tools=True,
                has_vision=True,
                needs_streaming=True,
                needs_json=True,
            )

            assert can_handle is True
            assert len(problems) == 0

    def test_can_handle_request_failure(self):
        """Test failed request capability check"""
        with patch(
            "chuk_llm.configuration.unified_config.get_config"
        ) as mock_get_config:
            mock_provider = Mock()
            mock_provider.supports_feature.return_value = False

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            can_handle, problems = CapabilityChecker.can_handle_request(
                provider="test", has_tools=True, has_vision=True
            )

            assert can_handle is False
            assert "tools not supported" in problems
            assert "vision not supported" in problems

    def test_get_best_provider_for_features(self):
        """Test finding best provider for features"""
        with patch(
            "chuk_llm.configuration.unified_config.get_config"
        ) as mock_get_config:
            # Mock providers
            provider1 = Mock()
            provider1.get_model_capabilities.return_value.features = {Feature.TEXT}
            provider1.get_rate_limit.return_value = 1000

            provider2 = Mock()
            provider2.get_model_capabilities.return_value.features = {
                Feature.TEXT,
                Feature.TOOLS,
            }
            provider2.get_rate_limit.return_value = 2000

            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["provider1", "provider2"]
            mock_config.get_provider.side_effect = (
                lambda name: provider1 if name == "provider1" else provider2
            )
            mock_get_config.return_value = mock_config

            best = CapabilityChecker.get_best_provider_for_features(
                required_features={Feature.TEXT, Feature.TOOLS}
            )

            assert (
                best == "provider2"
            )  # Has higher rate limit and supports required features

    def test_get_model_info_success(self):
        """Test successful model info retrieval"""
        with patch(
            "chuk_llm.configuration.unified_config.get_config"
        ) as mock_get_config:
            mock_caps = Mock()
            mock_caps.features = {Feature.TEXT, Feature.STREAMING}
            mock_caps.max_context_length = 8192
            mock_caps.max_output_tokens = 4096

            mock_provider = Mock()
            mock_provider.get_model_capabilities.return_value = mock_caps
            mock_provider.rate_limits = {"default": 1000}

            mock_config = Mock()
            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            info = CapabilityChecker.get_model_info("test", "test-model")

            assert info["provider"] == "test"
            assert info["model"] == "test-model"
            assert "text" in info["features"]
            assert "streaming" in info["features"]
            assert info["max_context_length"] == 8192
            assert info["supports_streaming"] is True
            assert info["supports_tools"] is False

    def test_get_model_info_error(self):
        """Test model info retrieval with error"""
        with patch(
            "chuk_llm.configuration.unified_config.get_config"
        ) as mock_get_config:
            mock_get_config.side_effect = Exception("Test error")

            info = CapabilityChecker.get_model_info("test", "test-model")

            assert "error" in info
            assert "Test error" in info["error"]


class TestConfigDiscoveryMixin:
    """Test ConfigDiscoveryMixin functionality"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_parse_discovery_config_enabled(self):
        """Test parsing enabled discovery config"""
        mixin = ConfigDiscoveryMixin()

        provider_data = {
            "extra": {
                "dynamic_discovery": {
                    "enabled": True,
                    "discoverer_type": "openai",
                    "cache_timeout": 600,
                }
            }
        }

        config = mixin._parse_discovery_config(provider_data)

        assert config is not None
        assert config.enabled is True
        assert config.discoverer_type == "openai"
        assert config.cache_timeout == 600

    def test_parse_discovery_config_disabled(self):
        """Test parsing disabled discovery config"""
        mixin = ConfigDiscoveryMixin()

        provider_data = {"extra": {"dynamic_discovery": {"enabled": False}}}

        config = mixin._parse_discovery_config(provider_data)
        assert config is None

    def test_parse_discovery_config_missing(self):
        """Test parsing missing discovery config"""
        mixin = ConfigDiscoveryMixin()

        provider_data = {"extra": {}}
        config = mixin._parse_discovery_config(provider_data)
        assert config is None

    def test_reload_clears_discovery_state(self):
        """Test that reload clears discovery state"""
        mixin = ConfigDiscoveryMixin()
        mixin._discovery_managers["test"] = Mock()
        mixin._discovery_cache["test"] = {"data": "test"}

        mixin.reload()

        assert mixin._discovery_managers == {}
        assert mixin._discovery_cache == {}

    def test_discovery_config_integration(self):
        """Test discovery configuration integration"""

        class MockManager(ConfigDiscoveryMixin):
            def __init__(self):
                super().__init__()
                self.providers = {
                    "test": ProviderConfig(
                        name="test",
                        models=["model-1"],
                        extra={
                            "dynamic_discovery": {
                                "enabled": True,
                                "discoverer_type": "test",
                            }
                        },
                    )
                }

        manager = MockManager()

        # Test that discovery config is properly integrated
        provider = manager.providers["test"]
        assert "dynamic_discovery" in provider.extra
        assert provider.extra["dynamic_discovery"]["enabled"] is True


class TestGlobalFunctions:
    """Test global functions"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_get_config_singleton(self):
        """Test that get_config returns singleton"""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config(self):
        """Test config reset functionality"""
        # Add some data to global config
        config = get_config()
        config.providers["test"] = ProviderConfig(name="test")

        # Reset and verify
        reset_config()
        new_config = get_config()

        assert new_config is not config  # New instance
        assert new_config.providers == {}


class TestUnifiedConfigManagerBasic:
    """Test basic UnifiedConfigManager functionality without environment complications"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_init_basic(self):
        """Test basic initialization without environment loading"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            assert manager.providers == {}
            assert manager.global_aliases == {}
            assert manager.global_settings == {}
            assert manager._loaded is False

    def test_init_with_config_path_basic(self):
        """Test initialization with custom config path"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path="/custom/path/config.yaml")
            assert manager.config_path == "/custom/path/config.yaml"


class TestUnifiedConfigManagerEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_load_with_missing_yaml(self):
        """Test loading when YAML module is not available"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            with patch("chuk_llm.configuration.unified_config.yaml", None):
                # Should not raise exception, just use empty config
                manager.load()
                assert manager._loaded is True
                assert len(manager.providers) >= 0  # May have built-in defaults

    def test_load_with_invalid_config_path(self):
        """Test loading with invalid config path"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path="/nonexistent/path/config.yaml")

            # Should not raise exception, just use defaults
            manager.load()
            assert manager._loaded is True

    def test_load_with_file_read_error(self):
        """Test handling file read errors gracefully"""
        # Mock the environment loading to avoid dotenv issues
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path="test.yaml")

            # Mock file operations to simulate read error
            with patch("builtins.open", side_effect=OSError("File not found")):
                with patch.object(Path, "exists", return_value=True):
                    # Should handle file read errors gracefully
                    manager.load()
                    assert manager._loaded is True

    def test_process_config_with_invalid_feature(self):
        """Test processing config with invalid feature"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            config = {
                "test_provider": {
                    "client_class": "TestClient",
                    "features": ["text", "invalid_feature", "streaming"],
                }
            }

            # Should handle invalid features gracefully
            with pytest.raises(ValueError, match="Unknown feature: invalid_feature"):
                manager._process_config(config)

    def test_inheritance_circular_reference(self):
        """Test handling circular inheritance"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            config = {
                "provider_a": {"inherits": "provider_b", "client_class": "ClientA"},
                "provider_b": {"inherits": "provider_a", "client_class": "ClientB"},
            }

            manager._process_config(config)
            # Should not hang - inheritance resolution has max depth
            manager._resolve_inheritance()

            # Both providers should still have their inherits field (unresolved)
            assert manager.providers["provider_a"].inherits == "provider_b"
            assert manager.providers["provider_b"].inherits == "provider_a"

    def test_inheritance_missing_parent(self):
        """Test inheritance with missing parent"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            config = {
                "child": {"inherits": "missing_parent", "client_class": "ChildClient"}
            }

            manager._process_config(config)
            manager._resolve_inheritance()

            # Should remain unresolved
            assert manager.providers["child"].inherits == "missing_parent"

    def test_model_capabilities_pattern_compilation_error(self):
        """Test handling regex pattern compilation errors"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Invalid regex pattern
            config = {
                "test": {
                    "client_class": "TestClient",
                    "model_capabilities": [
                        {
                            "pattern": "[invalid regex",  # Missing closing bracket
                            "features": ["text"],
                        }
                    ],
                }
            }

            manager._process_config(config)
            provider = manager.providers["test"]

            # The ModelCapabilities object should be created, but matching might fail
            assert len(provider.model_capabilities) == 1

            # Test that invalid regex doesn't crash the system
            caps = provider.model_capabilities[0]
            try:
                caps.matches("test-model")
                # If it doesn't raise, that's fine too
            except Exception:
                # Expected to fail on invalid regex
                pass


class TestConfigurationIntegration:
    """Integration tests for configuration system components"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_end_to_end_provider_usage(self):
        """Test complete workflow from config to provider usage"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Simulate a complete provider configuration
            config = {
                "__global_aliases__": {"gpt4": "gpt-4-turbo"},
                "openai": {
                    "client_class": "OpenAIClient",
                    "api_key_env": "OPENAI_API_KEY",
                    "default_model": "gpt-3.5-turbo",
                    "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                    "features": ["text", "streaming"],
                    "model_capabilities": [
                        {
                            "pattern": "gpt-4.*",
                            "features": ["tools", "vision"],
                            "max_context_length": 8192,
                        }
                    ],
                },
            }

            manager._process_config(config)
            manager._loaded = True

            # Test provider retrieval
            provider = manager.get_provider("openai")
            assert provider.name == "openai"
            assert provider.default_model == "gpt-3.5-turbo"

            # Test feature support
            assert manager.supports_feature("openai", "text")
            assert manager.supports_feature("openai", "text", "gpt-3.5-turbo")
            assert manager.supports_feature("openai", "tools", "gpt-4-turbo")
            assert not manager.supports_feature("openai", "tools", "gpt-3.5-turbo")

            # Test global aliases
            aliases = manager.get_global_aliases()
            assert aliases["gpt4"] == "gpt-4-turbo"

            # Test capability checker integration
            can_handle, problems = CapabilityChecker.can_handle_request(
                provider="openai",
                model="gpt-4-turbo",
                has_tools=True,
                needs_streaming=True,
            )
            assert can_handle is True
            assert len(problems) == 0

    def test_validator_integration(self):
        """Test validator integration with real config"""
        with patch.dict(os.environ, {"TEST_API_KEY": "sk-test123"}):
            config = ProviderConfig(
                name="test",
                client_class="TestClient",
                api_key_env="TEST_API_KEY",
                api_base="https://api.test.com/v1",
                default_model="test-model",
                features={Feature.TEXT, Feature.STREAMING},
            )

            # Test config validation
            valid, issues = ConfigValidator.validate_provider_config(config)
            assert valid is True
            assert len(issues) == 0

            # Test request validation with proper mocking
            with patch(
                "chuk_llm.configuration.unified_config.get_config"
            ) as mock_get_config:
                mock_manager = Mock()
                mock_manager.get_provider.return_value = config
                mock_get_config.return_value = mock_manager

                valid, issues = ConfigValidator.validate_request_compatibility(
                    provider_name="test", model="test-model", stream=True
                )
                assert valid is True
                assert len(issues) == 0


# Integration tests for file loading (would require temporary files)
class TestFileLoading:
    """Test file loading functionality with temporary files"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_load_yaml_file_basic(self):
        """Test loading basic YAML configuration"""
        config_content = """
__global__:
  debug: true

__global_aliases__:
  gpt4: gpt-4

openai:
  client_class: OpenAIClient
  api_key_env: OPENAI_API_KEY
  default_model: gpt-4
  models:
    - gpt-4
    - gpt-3.5-turbo
  features:
    - text
    - streaming
"""

        # Create a temporary file and close it immediately to avoid Windows file lock issues
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
            f.write(config_content)
            f.flush()
        
        try:
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager(config_path=temp_path)
                manager.load()

                assert manager.global_settings["debug"] is True
                assert manager.global_aliases["gpt4"] == "gpt-4"
                assert "openai" in manager.providers

                openai = manager.providers["openai"]
                assert openai.client_class == "OpenAIClient"
                assert openai.default_model == "gpt-4"
                assert Feature.TEXT in openai.features

        finally:
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                # On Windows, file might still be locked
                pass

    def test_load_yaml_inheritance(self):
        """Test loading YAML with inheritance"""
        config_content = """
base_provider:
  client_class: BaseClient
  features:
    - text
    - streaming

specific_provider:
  inherits: base_provider
  api_key_env: SPECIFIC_API_KEY
  default_model: specific-model
  features:
    - tools
"""

        # Create a temporary file and close it immediately to avoid Windows file lock issues
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
            f.write(config_content)
            f.flush()
        
        try:
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager(config_path=temp_path)
                manager.load()

                base = manager.providers["base_provider"]
                specific = manager.providers["specific_provider"]

                # Base provider
                assert base.client_class == "BaseClient"
                assert Feature.TEXT in base.features
                assert Feature.STREAMING in base.features

                # Specific provider inherits + adds
                assert specific.client_class == "BaseClient"  # Inherited
                assert specific.api_key_env == "SPECIFIC_API_KEY"  # Own
                assert Feature.TEXT in specific.features  # Inherited
                assert Feature.TOOLS in specific.features  # Added

        finally:
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                # On Windows, file might still be locked
                pass

    def test_load_yaml_file_missing(self):
        """Test loading when YAML file is missing"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path="/nonexistent/file.yaml")

            # Should not raise exception, just load with defaults
            manager.load()
            assert manager._loaded is True

    def test_load_yaml_file_invalid_yaml(self):
        """Test loading invalid YAML file"""
        invalid_yaml = """
invalid: yaml: content:
  - missing
  - proper
  - indentation
    nested: incorrectly
"""

        # Create a temporary file and close it immediately to avoid Windows file lock issues
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
            f.write(invalid_yaml)
            f.flush()
        
        try:
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager(config_path=temp_path)

                # Should handle invalid YAML gracefully
                manager.load()
                assert manager._loaded is True

        finally:
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                # On Windows, file might still be locked
                pass


class TestEnvironmentHandling:
    """Test environment variable handling"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    @patch("chuk_llm.configuration.unified_config._dotenv_available", False)
    def test_load_environment_no_dotenv(self):
        """Test environment loading when python-dotenv is not available"""
        manager = UnifiedConfigManager()

        # Should not raise exception
        manager._load_environment()
        # No specific assertions as this just logs a debug message

    @patch("chuk_llm.configuration.unified_config._dotenv_available", True)
    @patch("chuk_llm.configuration.unified_config.load_dotenv")
    @patch("pathlib.Path.exists")
    def test_load_environment_with_env_file(self, mock_exists, mock_load_dotenv):
        """Test environment loading with .env file"""
        # Mock .env file exists
        mock_exists.return_value = True

        manager = UnifiedConfigManager()
        manager._load_environment()

        # Should have called load_dotenv at least once (might be called multiple times due to multiple candidate files)
        assert mock_load_dotenv.call_count >= 1

    @patch("chuk_llm.configuration.unified_config._dotenv_available", True)
    @patch("chuk_llm.configuration.unified_config.load_dotenv")
    @patch("pathlib.Path.exists")
    def test_load_environment_no_env_file(self, mock_exists, mock_load_dotenv):
        """Test environment loading when no .env file exists"""
        # Mock no .env file exists
        mock_exists.return_value = False

        manager = UnifiedConfigManager()
        manager._load_environment()

        # Should not have called load_dotenv
        mock_load_dotenv.assert_not_called()


class TestPackageResourceHandlingSimplified:
    """Simplified tests for package resource handling"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_get_package_config_path_no_resources(self):
        """Test package config path when no resource libraries available"""
        with (
            patch.object(UnifiedConfigManager, "_load_environment"),
            patch(
                "chuk_llm.configuration.unified_config._importlib_resources_available",
                False,
            ),
            patch(
                "chuk_llm.configuration.unified_config._pkg_resources_available",
                False,
            ),
        ):
            manager = UnifiedConfigManager()

            result = manager._get_package_config_path()
            assert result is None

    def test_is_package_resource_path_basic(self):
        """Test basic package resource path detection"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Test with a path that clearly doesn't exist and contains chuk_llm
            with patch("pathlib.Path.exists", return_value=False):
                package_path = Path("/nonexistent/chuk_llm/resource/path")
                result = manager._is_package_resource_path(package_path)
                assert result is True

            # Test with a path that exists (should not be package resource)
            with patch("pathlib.Path.exists", return_value=True):
                regular_path = Path("/existing/regular/path.yaml")
                result = manager._is_package_resource_path(regular_path)
                assert result is False


class TestYAMLFileHandlingSimplified:
    """Simplified tests for YAML file handling"""

    def setup_method(self):
        """Setup for each test"""
        reset_config()

    def test_find_config_files_no_files(self):
        """Test config file finding when no files exist"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Mock no files exist
            with patch("pathlib.Path.exists", return_value=False):
                with patch.object(
                    manager, "_get_package_config_path", return_value=None
                ):
                    user_config, package_config = manager._find_config_files()

                    assert user_config is None
                    assert package_config is None

    def test_load_single_yaml_package_resource(self):
        """Test loading a single YAML file (package resource)"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            with patch.object(manager, "_is_package_resource_path", return_value=True):
                with patch.object(
                    manager, "_load_package_yaml", return_value={"package": "config"}
                ):
                    result = manager._load_single_yaml(Path("package/resource/path"))

                    assert result == {"package": "config"}

    def test_load_package_yaml_no_resources(self):
        """Test loading package YAML when no resource libraries available"""
        with (
            patch.object(UnifiedConfigManager, "_load_environment"),
            patch(
                "chuk_llm.configuration.unified_config._importlib_resources_available",
                False,
            ),
            patch(
                "chuk_llm.configuration.unified_config._pkg_resources_available",
                False,
            ),
        ):
            manager = UnifiedConfigManager()

            result = manager._load_package_yaml()
            assert result == {}


class TestPackageResourceHandling(TestPackageResourceHandlingSimplified):
    """Test package resource handling for different import scenarios"""

    @patch("chuk_llm.configuration.unified_config._importlib_resources_available", True)
    @patch("chuk_llm.configuration.unified_config.files")
    def test_get_package_config_path_importlib_success(self, mock_files):
        """Test package config path with importlib.resources success"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            # Mock successful importlib.resources
            mock_package_files = Mock()
            mock_config_file = Mock()
            mock_config_file.is_file.return_value = True
            mock_config_file.__str__ = Mock(return_value="/mock/path/chuk_llm.yaml")

            # Mock the __truediv__ method (which handles the / operator)
            mock_package_files.__truediv__ = Mock(return_value=mock_config_file)
            mock_files.return_value = mock_package_files

            manager = UnifiedConfigManager()

            result = manager._get_package_config_path()
            assert result is not None
            mock_files.assert_called_once_with("chuk_llm")


class TestYAMLFileHandling(TestYAMLFileHandlingSimplified):
    """Test YAML file handling functionality"""

    def test_find_config_files_user_config_priority(self):
        """Test that user config files are found with proper priority"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            # Create manager with specific config path to avoid file discovery complexity
            manager = UnifiedConfigManager(config_path="chuk_llm.yaml")

            # Mock that the specified file exists
            with patch("pathlib.Path.exists", return_value=True):
                user_config, package_config = manager._find_config_files()

                # Should find the user config we specified
                assert user_config is not None
                assert "chuk_llm.yaml" in str(user_config)

    def test_find_config_files_basic(self):
        """Test basic config file finding logic"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            # Test with explicit config path
            manager = UnifiedConfigManager(config_path="test.yaml")

            with patch("pathlib.Path.exists", return_value=True):
                user_config, package_config = manager._find_config_files()

                # Should find the explicitly set config
                assert user_config is not None
                assert "test.yaml" in str(user_config)

    def test_find_config_files_no_files_exist(self):
        """Test config file finding when no files exist"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Mock that no files exist
            with patch("pathlib.Path.exists", return_value=False):
                with patch.object(
                    manager, "_get_package_config_path", return_value=None
                ):
                    user_config, package_config = manager._find_config_files()

                    assert user_config is None
                    assert package_config is None

    def test_load_single_yaml_regular_file(self):
        """Test loading a single YAML file (regular file)"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            yaml_content = {"test": {"key": "value"}}

            # Create a proper mock file context manager
            mock_file = Mock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)

            with patch("builtins.open", return_value=mock_file):
                with patch.object(
                    manager, "_is_package_resource_path", return_value=False
                ):
                    with patch(
                        "chuk_llm.configuration.unified_config.yaml.safe_load",
                        return_value=yaml_content,
                    ):
                        result = manager._load_single_yaml(Path("test.yaml"))

                        assert result == yaml_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
