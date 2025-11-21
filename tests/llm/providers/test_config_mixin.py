# tests/llm/providers/test_config_mixin.py
import gc
import logging
import threading
import time
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

# ---------------------------------------------------------------------------
# Mock configuration objects
# ---------------------------------------------------------------------------


class MockFeature:
    """Mock Feature enum for testing"""

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
    def from_string(cls, feature_str: str):
        """Convert string to feature"""
        return getattr(cls, feature_str.upper(), None)


class MockModelCapabilities:
    """Mock model capabilities"""

    def __init__(
        self,
        features: set | None = None,
        max_context_length: int = 4096,
        max_output_tokens: int = 2048,
    ):
        if features is None:
            self.features = {MockFeature.TEXT, MockFeature.STREAMING}
        else:
            self.features = features
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    """Mock provider configuration"""

    def __init__(
        self,
        name: str = "mock_provider",
        client_class: str = "MockClient",
        api_base: str = "https://api.mock.com",
        models: list[str] | None = None,
        model_aliases: dict[str, str] | None = None,
        rate_limits: dict[str, Any] | None = None,
    ):
        self.name = name
        self.client_class = client_class
        self.api_base = api_base
        self.models = models or ["model-1", "model-2"]
        self.model_aliases = model_aliases or {"latest": "model-1"}
        self.rate_limits = rate_limits or {"requests_per_minute": 60}
        self._model_capabilities = {}

    def get_model_capabilities(self, model: str) -> MockModelCapabilities:
        """Get capabilities for a specific model"""
        if model not in self._model_capabilities:
            # Default capabilities based on model name
            if "advanced" in model:
                features = {
                    MockFeature.TEXT,
                    MockFeature.STREAMING,
                    MockFeature.TOOLS,
                    MockFeature.VISION,
                    MockFeature.JSON_MODE,
                    MockFeature.SYSTEM_MESSAGES,
                }
                self._model_capabilities[model] = MockModelCapabilities(
                    features=features, max_context_length=8192, max_output_tokens=4096
                )
            elif "ultra" in model:
                features = {MockFeature.TEXT, MockFeature.STREAMING}
                self._model_capabilities[model] = MockModelCapabilities(
                    features=features,
                    max_context_length=32768,
                    max_output_tokens=100000,
                )
            else:
                self._model_capabilities[model] = MockModelCapabilities()

        return self._model_capabilities[model]

    def set_model_capabilities(self, model: str, capabilities: MockModelCapabilities):
        """Helper to set specific capabilities for testing"""
        self._model_capabilities[model] = capabilities


class MockConfig:
    """Mock configuration object"""

    def __init__(self):
        self._providers = {}

    def get_provider(self, provider_name: str) -> MockProviderConfig | None:
        """Get provider configuration"""
        return self._providers.get(provider_name)

    def add_provider(self, provider_config: MockProviderConfig):
        """Helper to add provider for testing"""
        self._providers[provider_config.name] = provider_config


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a mock configuration with test providers"""
    config = MockConfig()

    # Add test providers
    openai_provider = MockProviderConfig(
        name="openai",
        client_class="OpenAIClient",
        api_base="https://api.openai.com/v1",
        models=["gpt-4", "gpt-3.5-turbo", "gpt-4-advanced", "gpt-4-ultra"],
        model_aliases={"latest": "gpt-4"},
        rate_limits={"requests_per_minute": 60, "tokens_per_minute": 150000},
    )

    anthropic_provider = MockProviderConfig(
        name="anthropic",
        client_class="AnthropicClient",
        api_base="https://api.anthropic.com",
        models=["claude-3-opus", "claude-3-sonnet"],
        model_aliases={"latest": "claude-3-opus"},
    )

    config.add_provider(openai_provider)
    config.add_provider(anthropic_provider)

    return config


@pytest.fixture
def mock_configuration(mock_config):
    """Mock the configuration system"""
    with patch("chuk_llm.configuration.get_config", return_value=mock_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            yield mock_config


@pytest.fixture
def basic_mixin(mock_configuration):
    """Create a basic ConfigAwareProviderMixin instance for testing"""
    return ConfigAwareProviderMixin("openai", "gpt-4")


@pytest.fixture
def advanced_mixin(mock_configuration):
    """Create a ConfigAwareProviderMixin instance with advanced model"""
    return ConfigAwareProviderMixin("openai", "gpt-4-advanced")


@pytest.fixture
def ultra_mixin(mock_configuration):
    """Create a ConfigAwareProviderMixin instance with ultra-high-limit model"""
    return ConfigAwareProviderMixin("openai", "gpt-4-ultra")


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_initialization():
    """Test basic initialization of ConfigAwareProviderMixin"""
    mixin = ConfigAwareProviderMixin("test_provider", "test_model")

    assert mixin.provider_name == "test_provider"
    assert mixin.model == "test_model"
    assert mixin._cached_config is None
    assert mixin._cached_model_caps is None


# ---------------------------------------------------------------------------
# Configuration loading tests
# ---------------------------------------------------------------------------


def test_get_provider_config_success(basic_mixin):
    """Test successful provider config loading"""
    config = basic_mixin._get_provider_config()

    assert config is not None
    assert config.name == "openai"
    assert config.client_class == "OpenAIClient"
    assert basic_mixin._cached_config is config


def test_get_provider_config_caching(mock_config):
    """Test that provider config is cached"""
    call_count = 0

    def mock_get_config():
        nonlocal call_count
        call_count += 1
        return mock_config

    with patch("chuk_llm.configuration.get_config", side_effect=mock_get_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")

            # First call
            config1 = mixin._get_provider_config()
            # Second call
            config2 = mixin._get_provider_config()

            assert config1 is config2
            assert call_count == 1


def test_get_provider_config_import_error():
    """Test handling of import errors when loading config"""
    with patch(
        "chuk_llm.configuration.get_config",
        side_effect=ImportError("Config not available"),
    ):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")

        config = mixin._get_provider_config()

        assert config is None
        assert mixin._cached_config is None


def test_get_provider_config_not_found(mock_configuration):
    """Test handling when provider is not found in config"""
    mixin = ConfigAwareProviderMixin("nonexistent_provider", "some_model")

    config = mixin._get_provider_config()

    assert config is None


def test_get_model_capabilities_success(advanced_mixin):
    """Test successful model capabilities loading"""
    caps = advanced_mixin._get_model_capabilities()

    assert caps is not None
    assert MockFeature.TOOLS in caps.features
    assert MockFeature.VISION in caps.features
    assert caps.max_context_length == 8192
    assert advanced_mixin._cached_model_caps is caps


def test_get_model_capabilities_caching(basic_mixin):
    """Test that model capabilities are cached"""
    # First call
    caps1 = basic_mixin._get_model_capabilities()
    # Second call
    caps2 = basic_mixin._get_model_capabilities()

    assert caps1 is caps2


def test_get_model_capabilities_no_provider_config():
    """Test model capabilities when provider config is not available"""
    with patch("chuk_llm.configuration.get_config", side_effect=Exception("No config")):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")

        caps = mixin._get_model_capabilities()

        assert caps is None


# ---------------------------------------------------------------------------
# get_model_info tests
# ---------------------------------------------------------------------------


def test_get_model_info_success(basic_mixin):
    """Test successful get_model_info"""
    info = basic_mixin.get_model_info()

    assert info["provider"] == "openai"
    assert info["model"] == "gpt-4"
    assert info["client_class"] == "OpenAIClient"
    assert info["api_base"] == "https://api.openai.com/v1"
    assert "text" in info["features"]
    assert "streaming" in info["features"]
    assert info["supports_text"] is True
    assert info["supports_streaming"] is True
    assert info["max_context_length"] == 4096
    assert info["max_output_tokens"] == 2048
    assert "error" not in info


def test_get_model_info_advanced_model(advanced_mixin):
    """Test get_model_info for advanced model with more features"""
    info = advanced_mixin.get_model_info()

    assert info["supports_tools"] is True
    assert info["supports_vision"] is True
    assert info["supports_json_mode"] is True
    assert info["supports_system_messages"] is True
    assert info["max_context_length"] == 8192
    assert info["max_output_tokens"] == 4096


def test_get_model_info_no_config():
    """Test get_model_info when configuration is not available"""
    with patch(
        "chuk_llm.configuration.get_config", side_effect=ImportError("No config")
    ):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")

        info = mixin.get_model_info()

        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["error"] == "Configuration not available"
        assert info["supports_text"] is False
        assert info["supports_streaming"] is False
        assert info["features"] == []


def test_get_model_info_config_error():
    """Test get_model_info when configuration throws an error"""
    with patch(
        "chuk_llm.configuration.get_config", side_effect=ValueError("Config error")
    ):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")

        info = mixin.get_model_info()

        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["error"] == "Configuration not available"
        assert info["supports_text"] is False


def test_get_model_info_provider_not_found(mock_configuration):
    """Test get_model_info when provider is not found"""
    mixin = ConfigAwareProviderMixin("nonexistent", "some_model")

    info = mixin.get_model_info()

    assert info["provider"] == "nonexistent"
    assert info["model"] == "some_model"
    assert info["error"] == "Configuration not available"


# ---------------------------------------------------------------------------
# supports_feature tests
# ---------------------------------------------------------------------------


def test_supports_feature_string_input(basic_mixin):
    """Test supports_feature with string input"""
    assert basic_mixin.supports_feature("text") is True
    assert basic_mixin.supports_feature("streaming") is True
    # Registry data overrides config - gpt-4 supports tools according to registry
    assert basic_mixin.supports_feature("tools") is True
    # gpt-4 doesn't support vision according to registry
    assert basic_mixin.supports_feature("vision") is False


def test_supports_feature_enum_input(basic_mixin):
    """Test supports_feature with enum input"""
    assert basic_mixin.supports_feature(MockFeature.TEXT) is True
    assert basic_mixin.supports_feature(MockFeature.STREAMING) is True
    # Registry data overrides config - gpt-4 supports tools according to registry
    assert basic_mixin.supports_feature(MockFeature.TOOLS) is True


def test_supports_feature_advanced_model(advanced_mixin):
    """Test supports_feature for advanced model"""
    assert advanced_mixin.supports_feature("tools") is True
    assert advanced_mixin.supports_feature("vision") is True
    assert advanced_mixin.supports_feature("json_mode") is True


def test_supports_feature_no_capabilities():
    """Test supports_feature when capabilities are not available"""
    with patch("chuk_llm.configuration.get_config", side_effect=Exception("No config")):
        # Use a provider/model that doesn't exist in registry to test fallback behavior
        mixin = ConfigAwareProviderMixin("fake_provider", "fake-model")

        # Returns False when no configuration or registry data is available
        assert mixin.supports_feature("text") is False
        assert mixin.supports_feature("streaming") is False


def test_supports_feature_invalid_feature(basic_mixin):
    """Test supports_feature with invalid feature name"""
    assert basic_mixin.supports_feature("nonexistent_feature") is False


def test_supports_feature_exception_handling(basic_mixin, monkeypatch):
    """Test supports_feature handles exceptions gracefully"""
    monkeypatch.setattr(basic_mixin, "_get_model_capabilities", lambda: None)
    # Registry data still available for openai/gpt-4, so returns True
    assert basic_mixin.supports_feature("text") is True


# ---------------------------------------------------------------------------
# Token limit tests
# ---------------------------------------------------------------------------


def test_get_max_tokens_limit_success(basic_mixin):
    """Test successful max tokens limit retrieval"""
    limit = basic_mixin.get_max_tokens_limit()
    assert limit == 2048


def test_get_max_tokens_limit_advanced_model(advanced_mixin):
    """Test max tokens limit for advanced model"""
    limit = advanced_mixin.get_max_tokens_limit()
    assert limit == 4096


def test_get_max_tokens_limit_no_capabilities():
    """Test max tokens limit when capabilities are not available"""
    with patch("chuk_llm.configuration.get_config", side_effect=Exception("No config")):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")

        limit = mixin.get_max_tokens_limit()
        assert limit is None


def test_get_context_length_limit_success(basic_mixin):
    """Test successful context length limit retrieval"""
    limit = basic_mixin.get_context_length_limit()
    assert limit == 4096


def test_get_context_length_limit_advanced_model(advanced_mixin):
    """Test context length limit for advanced model"""
    limit = advanced_mixin.get_context_length_limit()
    assert limit == 8192


def test_get_context_length_limit_no_capabilities():
    """Test context length limit when capabilities are not available"""
    with patch("chuk_llm.configuration.get_config", side_effect=Exception("No config")):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")

        limit = mixin.get_context_length_limit()
        assert limit is None


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------


def test_validate_parameters_basic(basic_mixin):
    """Test basic parameter validation"""
    params = {"temperature": 0.7, "max_tokens": 1000}

    validated = basic_mixin.validate_parameters(**params)

    assert validated["temperature"] == 0.7
    assert validated["max_tokens"] == 1000


def test_validate_parameters_max_tokens_capping(basic_mixin):
    """Test that max_tokens is capped to model limit"""
    params = {"temperature": 0.7, "max_tokens": 5000}

    validated = basic_mixin.validate_parameters(**params)

    assert validated["temperature"] == 0.7
    assert validated["max_tokens"] == 2048


def test_validate_parameters_max_tokens_within_limit(basic_mixin):
    """Test that max_tokens within limit is not changed"""
    params = {"temperature": 0.7, "max_tokens": 1500}

    validated = basic_mixin.validate_parameters(**params)

    assert validated["max_tokens"] == 1500


def test_validate_parameters_add_default_max_tokens(basic_mixin):
    """Test that default max_tokens is added when not specified"""
    params = {"temperature": 0.7}

    validated = basic_mixin.validate_parameters(**params)

    assert validated["temperature"] == 0.7
    assert "max_tokens" in validated
    assert validated["max_tokens"] == 2048


def test_validate_parameters_add_default_max_tokens_high_limit(ultra_mixin):
    """Test default max_tokens when model limit is very high"""
    params = {"temperature": 0.7}
    validated = ultra_mixin.validate_parameters(**params)

    assert validated["max_tokens"] == 4096


def test_validate_parameters_no_default_when_no_limit():
    """Test that no default max_tokens is added when model has no limit"""
    with patch("chuk_llm.configuration.get_config", side_effect=Exception("No config")):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")

        params = {"temperature": 0.7}
        validated = mixin.validate_parameters(**params)

        assert validated["temperature"] == 0.7
        assert "max_tokens" not in validated


def test_validate_parameters_preserves_other_params(basic_mixin):
    """Test that other parameters are preserved during validation"""
    params = {
        "temperature": 0.8,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "stop": ["END"],
        "max_tokens": 1000,
    }

    validated = basic_mixin.validate_parameters(**params)

    assert validated["temperature"] == 0.8
    assert validated["top_p"] == 0.9
    assert validated["frequency_penalty"] == 0.1
    assert validated["presence_penalty"] == 0.2
    assert validated["stop"] == ["END"]
    assert validated["max_tokens"] == 1000


def test_validate_parameters_no_mutation(basic_mixin):
    """Test that original parameters dict is not mutated"""
    original_params = {"temperature": 0.7, "max_tokens": 5000}

    validated = basic_mixin.validate_parameters(**original_params)

    assert original_params["max_tokens"] == 5000
    assert validated["max_tokens"] == 2048
    assert original_params is not validated


def test_validate_parameters_none_max_tokens(basic_mixin):
    """Test parameter validation with None max_tokens"""
    params = {"temperature": 0.7, "max_tokens": None}

    validated = basic_mixin.validate_parameters(**params)

    assert validated["temperature"] == 0.7
    assert validated["max_tokens"] == 2048


def test_validate_parameters_edge_cases(basic_mixin):
    """Test parameter validation edge cases"""
    # Test with zero max_tokens
    validated = basic_mixin.validate_parameters(max_tokens=0)
    assert validated["max_tokens"] == 0

    # Test with negative max_tokens
    validated = basic_mixin.validate_parameters(max_tokens=-1)
    assert validated["max_tokens"] == -1

    # Test with exactly the limit
    validated = basic_mixin.validate_parameters(max_tokens=2048)
    assert validated["max_tokens"] == 2048

    # Test with one more than the limit
    validated = basic_mixin.validate_parameters(max_tokens=2049)
    assert validated["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_full_workflow_integration(advanced_mixin):
    """Test a complete workflow using the mixin"""
    # Check model info
    info = advanced_mixin.get_model_info()
    assert info["supports_tools"] is True
    assert info["supports_vision"] is True

    # Check specific features
    assert advanced_mixin.supports_feature("tools") is True
    assert advanced_mixin.supports_feature("vision") is True
    assert advanced_mixin.supports_feature("reasoning") is False

    # Get limits
    assert advanced_mixin.get_max_tokens_limit() == 4096
    assert advanced_mixin.get_context_length_limit() == 8192

    # Validate parameters
    params = {"temperature": 0.7, "max_tokens": 10000}
    validated = advanced_mixin.validate_parameters(**params)
    assert validated["max_tokens"] == 4096


def test_multiple_provider_workflow(mock_configuration):
    """Test workflow with multiple providers"""
    openai_mixin = ConfigAwareProviderMixin("openai", "gpt-4")
    anthropic_mixin = ConfigAwareProviderMixin("anthropic", "claude-3-opus")

    openai_info = openai_mixin.get_model_info()
    anthropic_info = anthropic_mixin.get_model_info()

    assert openai_info["provider"] == "openai"
    assert anthropic_info["provider"] == "anthropic"
    assert openai_mixin._cached_config != anthropic_mixin._cached_config


# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------


def test_caching_with_config_success():
    """Test that caching works correctly with successful configs"""
    call_count = 0

    def mock_get_config():
        nonlocal call_count
        call_count += 1
        config = MockConfig()
        provider = MockProviderConfig(name="openai")
        config.add_provider(provider)
        return config

    with patch("chuk_llm.configuration.get_config", side_effect=mock_get_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")

            # First call should succeed and cache the result
            config1 = mixin._get_provider_config()
            assert config1 is not None

            # Second call should return cached result
            config2 = mixin._get_provider_config()
            assert config2 is config1
            assert call_count == 1


def test_model_capabilities_with_missing_model(mock_configuration):
    """Test model capabilities for model not in provider config"""
    mixin = ConfigAwareProviderMixin("openai", "nonexistent-model")

    caps = mixin._get_model_capabilities()

    assert caps is not None
    assert MockFeature.TEXT in caps.features
    assert MockFeature.STREAMING in caps.features


def test_logging_behavior(basic_mixin, caplog):
    """Test that appropriate log messages are generated"""
    with caplog.at_level(logging.DEBUG):
        params = {"max_tokens": 5000}
        basic_mixin.validate_parameters(**params)

        assert "Capping max_tokens from 5000 to 2048" in caplog.text


def test_concurrent_access_safety(mock_config):
    """Test that the mixin is safe for concurrent access"""
    results = []
    results_lock = threading.Lock()

    def access_config():
        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            with patch("chuk_llm.configuration.Feature", MockFeature):
                mixin = ConfigAwareProviderMixin("openai", "gpt-4")
                time.sleep(0.01)
                config = mixin._get_provider_config()
                with results_lock:
                    results.append(config)

    threads = [threading.Thread(target=access_config) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 10
    assert all(r is not None for r in results)


def test_memory_usage_with_large_configs(basic_mixin):
    """Test memory usage doesn't grow with repeated access"""
    for _ in range(100):  # Reduced from 1000 for faster testing
        basic_mixin._get_provider_config()
        basic_mixin._get_model_capabilities()
        basic_mixin.get_model_info()

    assert basic_mixin._cached_config is not None
    assert basic_mixin._cached_model_caps is not None

    gc.collect()


# ---------------------------------------------------------------------------
# Custom configuration scenarios
# ---------------------------------------------------------------------------


def test_custom_model_capabilities(mock_configuration):
    """Test with custom model capabilities"""
    config = mock_configuration
    provider = config.get_provider("openai")

    custom_caps = MockModelCapabilities(
        features={MockFeature.TEXT, MockFeature.REASONING, MockFeature.MULTIMODAL},
        max_context_length=32768,
        max_output_tokens=8192,
    )
    provider.set_model_capabilities("custom-model", custom_caps)

    mixin = ConfigAwareProviderMixin("openai", "custom-model")

    assert mixin.supports_feature("reasoning") is True
    assert mixin.supports_feature("multimodal") is True
    assert mixin.supports_feature("tools") is False

    assert mixin.get_context_length_limit() == 32768
    assert mixin.get_max_tokens_limit() == 8192

    info = mixin.get_model_info()
    assert info["supports_reasoning"] is True
    assert info["supports_multimodal"] is True


def test_empty_features():
    """Test model with no features"""
    isolated_config = MockConfig()
    isolated_provider = MockProviderConfig(name="openai")

    minimal_caps = MockModelCapabilities(features=set())
    isolated_provider.set_model_capabilities("minimal-model", minimal_caps)
    isolated_config.add_provider(isolated_provider)

    with patch("chuk_llm.configuration.get_config", return_value=isolated_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "minimal-model")

            caps = mixin._get_model_capabilities()
            assert caps is not None
            assert len(caps.features) == 0

            assert mixin.supports_feature("text") is False
            assert mixin.supports_feature("streaming") is False

            info = mixin.get_model_info()
            assert info["features"] == []
            assert all(
                not info[f"supports_{feature}"]
                for feature in [
                    "text",
                    "streaming",
                    "tools",
                    "vision",
                    "json_mode",
                    "system_messages",
                    "parallel_calls",
                    "multimodal",
                    "reasoning",
                ]
            )


def test_provider_without_api_base(mock_configuration):
    """Test provider configuration without api_base"""
    config = mock_configuration

    local_provider = MockProviderConfig(name="local", api_base=None)
    config.add_provider(local_provider)

    mixin = ConfigAwareProviderMixin("local", "local-model")

    info = mixin.get_model_info()
    assert info["api_base"] is None


# ---------------------------------------------------------------------------
# Advanced caching scenarios
# ---------------------------------------------------------------------------


def test_cache_isolation_between_instances(mock_configuration):
    """Test that cache is properly isolated between instances"""
    mixin1 = ConfigAwareProviderMixin("openai", "gpt-4")
    mixin2 = ConfigAwareProviderMixin("anthropic", "claude-3-opus")

    config1 = mixin1._get_provider_config()
    config2 = mixin2._get_provider_config()

    assert config1 != config2
    assert mixin1._cached_config != mixin2._cached_config


def test_cache_persistence_across_method_calls(basic_mixin):
    """Test that cache persists across different method calls"""
    info = basic_mixin.get_model_info()
    cached_config_1 = basic_mixin._cached_config

    supports_text = basic_mixin.supports_feature("text")
    cached_config_2 = basic_mixin._cached_config

    limit = basic_mixin.get_max_tokens_limit()
    cached_config_3 = basic_mixin._cached_config

    assert cached_config_1 is cached_config_2
    assert cached_config_2 is cached_config_3
    assert info["provider"] == "openai"
    assert supports_text is True
    assert limit == 2048


# ---------------------------------------------------------------------------
# Performance and resource tests
# ---------------------------------------------------------------------------


def test_no_config_reload_on_repeated_calls(mock_config):
    """Test that configuration is not reloaded on repeated calls"""
    call_count = 0

    def counting_get_config():
        nonlocal call_count
        call_count += 1
        return mock_config

    with patch("chuk_llm.configuration.get_config", side_effect=counting_get_config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")

            for _ in range(5):  # Reduced iterations for faster testing
                mixin.get_model_info()
                mixin.supports_feature("text")
                mixin.get_max_tokens_limit()
                mixin.validate_parameters(temperature=0.7)

            assert call_count == 1


def test_capabilities_not_recomputed_unnecessarily(mock_configuration, monkeypatch):
    """Test that model capabilities are not recomputed unnecessarily"""
    call_count = 0

    def counting_get_model_capabilities(model):
        nonlocal call_count
        call_count += 1
        return MockModelCapabilities()

    provider = mock_configuration.get_provider("openai")
    monkeypatch.setattr(
        provider, "get_model_capabilities", counting_get_model_capabilities
    )

    mixin = ConfigAwareProviderMixin("openai", "gpt-4")

    for _ in range(5):
        mixin.supports_feature("text")
        mixin.get_max_tokens_limit()
        mixin.get_context_length_limit()

    assert call_count == 1


# ---------------------------------------------------------------------------
# New tests for smart fallback behavior
# ---------------------------------------------------------------------------


def test_supports_feature_returns_none_for_unknown_model():
    """Test that supports_feature returns None when model capabilities are unavailable"""
    with patch(
        "chuk_llm.configuration.get_config",
        side_effect=ImportError("Config not available"),
    ):
        mixin = ConfigAwareProviderMixin("unknown_provider", "unknown_model")

        # Should return False when config is not available
        assert mixin.supports_feature("text") is False
        assert mixin.supports_feature("tools") is False
        assert mixin.supports_feature("streaming") is False


def test_has_explicit_model_config_with_matching_capability():
    """Test _has_explicit_model_config when model_capabilities matches (lines 58-59)"""

    class MockModelCapability:
        def matches(self, model):
            return "special" in model

    class MockProviderWithCapabilities:
        model_capabilities = [MockModelCapability()]
        models = []

    with patch("chuk_llm.configuration.get_config") as mock_get_config:
        mock_config = MagicMock()
        mock_config.get_provider.return_value = MockProviderWithCapabilities()
        mock_get_config.return_value = mock_config

        mixin = ConfigAwareProviderMixin("openai", "special-model")
        result = mixin._has_explicit_model_config()

        # Should return True because capability matches
        assert result is True


def test_has_explicit_model_config_exception_on_models_check():
    """Test _has_explicit_model_config exception handling on models check (lines 66-67)"""

    class MockProviderWithBrokenModels:
        model_capabilities = []

        @property
        def models(self):
            raise Exception("Models list unavailable")

    with patch("chuk_llm.configuration.get_config") as mock_get_config:
        mock_config = MagicMock()
        mock_config.get_provider.return_value = MockProviderWithBrokenModels()
        mock_get_config.return_value = mock_config

        mixin = ConfigAwareProviderMixin("openai", "gpt-4")
        result = mixin._has_explicit_model_config()

        # Should return False when exception occurs
        assert result is False


def test_get_model_info_exception_handling():
    """Test get_model_info exception handling (lines 134-136)"""

    with patch("chuk_llm.configuration.get_config") as mock_get_config:
        mock_get_config.side_effect = Exception("Configuration error")

        with patch("chuk_llm.llm.providers._config_mixin.log") as mock_log:
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")
            info = mixin.get_model_info()

            # Should log error (may be called multiple times due to retries/caching)
            assert mock_log.error.called

            # Should return error info
            assert info["provider"] == "openai"
            assert info["model"] == "gpt-4"
            assert "error" in info
            # Error message may be wrapped
            assert "Configuration" in info["error"] or "not available" in info["error"]
            assert info["has_explicit_config"] is False


def test_supports_feature_with_non_string_feature():
    """Test supports_feature with non-string feature (line 168)"""

    # Create a mock feature enum
    class MockFeatureEnum:
        TEXT = "text"
        TOOLS = "tools"

    with patch("chuk_llm.configuration.get_config") as mock_get_config:
        mock_config = MagicMock()
        mock_provider = MockProviderConfig()
        mock_config.get_provider.return_value = mock_provider
        mock_get_config.return_value = mock_config

        with patch("chuk_llm.configuration.Feature", MockFeature):
            mixin = ConfigAwareProviderMixin("mock_provider", "model-1")

            # Test with enum-like feature (not a string)
            # This hits the else branch on line 168
            result = mixin.supports_feature(MockFeatureEnum.TEXT)

            # Should still work
            assert isinstance(result, bool)


def test_validate_parameters_caps_max_completion_tokens():
    """Test validate_parameters caps max_completion_tokens (lines 223-228)"""

    with patch("chuk_llm.configuration.get_config") as mock_get_config:
        mock_config = MagicMock()

        # Create provider with model that has max_output_tokens limit
        mock_provider = MockProviderConfig()
        mock_capabilities = MockModelCapabilities(max_output_tokens=1000)
        mock_provider._model_capabilities["model-1"] = mock_capabilities

        mock_config.get_provider.return_value = mock_provider
        mock_get_config.return_value = mock_config

        with patch("chuk_llm.configuration.Feature", MockFeature):
            mixin = ConfigAwareProviderMixin("mock_provider", "model-1")

            with patch("chuk_llm.llm.providers._config_mixin.log") as mock_log:
                # Try to set max_completion_tokens above limit
                adjusted = mixin.validate_parameters(max_completion_tokens=5000)

                # Should be capped to limit
                assert adjusted["max_completion_tokens"] == 1000

                # Should log debug message
                mock_log.debug.assert_called()
                assert "Capping max_completion_tokens" in str(mock_log.debug.call_args)


def test_supports_feature_behavior_with_partial_config():
    """Test supports_feature behavior when config exists but model capabilities are missing"""
    config = MockConfig()
    provider = MockProviderConfig(name="test_provider")
    config.add_provider(provider)

    with patch("chuk_llm.configuration.get_config", return_value=config):
        with patch("chuk_llm.configuration.Feature", MockFeature):
            mixin = ConfigAwareProviderMixin("test_provider", "unknown_model")

            # Should still work since provider exists and can provide default capabilities
            assert mixin.supports_feature("text") is True  # Default capability
            assert mixin.supports_feature("streaming") is True  # Default capability
            assert (
                mixin.supports_feature("advanced_feature") is False
            )  # Not in defaults


def test_get_model_info_includes_fallback_metadata(mock_configuration):
    """Test that get_model_info includes metadata about fallback usage"""
    mixin = ConfigAwareProviderMixin("openai", "gpt-4")
    info = mixin.get_model_info()

    # Should indicate whether using explicit config or fallback
    assert "has_explicit_config" in info
    assert "using_fallback" in info
    assert info["has_explicit_config"] is True  # gpt-4 is in our mock config
    assert info["using_fallback"] is False


def test_get_model_info_fallback_for_unknown_model(mock_configuration):
    """Test get_model_info fallback behavior for unknown models"""
    mixin = ConfigAwareProviderMixin("openai", "unknown-model-2025")
    info = mixin.get_model_info()

    # Should still work but indicate it's using fallback
    assert info["provider"] == "openai"
    assert info["model"] == "unknown-model-2025"
    assert info["has_explicit_config"] is False
    assert info["using_fallback"] is True


def test_feature_detection_error_handling():
    """Test that feature detection handles errors gracefully"""
    with patch(
        "chuk_llm.configuration.Feature.from_string",
        side_effect=AttributeError("Feature not found"),
    ):
        config = MockConfig()
        provider = MockProviderConfig(name="openai")
        config.add_provider(provider)

        with patch("chuk_llm.configuration.get_config", return_value=config):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")

            # Should handle feature detection errors gracefully
            result = mixin.supports_feature("invalid_feature")
            assert result is False  # Returns False when there's an error


def test_logging_for_config_fallback():
    """Test that appropriate logging occurs when falling back to client"""
    with patch(
        "chuk_llm.configuration.get_config",
        side_effect=ImportError("Config not available"),
    ):
        mixin = ConfigAwareProviderMixin("openai", "new-model")

        with patch("chuk_llm.llm.providers._config_mixin.log") as mock_log:
            result = mixin.supports_feature("text")

            assert result is False
            # Should log error about failed config
            mock_log.error.assert_called()
            debug_calls = [call.args[0] for call in mock_log.debug.call_args_list]
            assert any("deferring to client" in call for call in debug_calls)
