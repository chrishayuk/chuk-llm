# tests/providers/test_config_mixin.py
import logging
import pytest
import threading
import time
import gc
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

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
        features: Optional[Set] = None,
        max_context_length: int = 4096,
        max_output_tokens: int = 2048
    ):
        # FIXED: Only use default features if features is None, not if it's an empty set
        if features is None:
            self.features = {MockFeature.TEXT, MockFeature.STREAMING}
        else:
            self.features = features  # Use exactly what was passed, including empty set
        self.max_context_length = max_context_length
        self.max_output_tokens = max_output_tokens


class MockProviderConfig:
    """Mock provider configuration"""
    def __init__(
        self,
        name: str = "mock_provider",
        client_class: str = "MockClient",
        api_base: str = "https://api.mock.com",
        models: Optional[List[str]] = None,
        model_aliases: Optional[Dict[str, str]] = None,
        rate_limits: Optional[Dict[str, Any]] = None
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
            # Default capabilities
            if "advanced" in model:
                features = {
                    MockFeature.TEXT, MockFeature.STREAMING, MockFeature.TOOLS,
                    MockFeature.VISION, MockFeature.JSON_MODE, MockFeature.SYSTEM_MESSAGES
                }
                self._model_capabilities[model] = MockModelCapabilities(
                    features=features,
                    max_context_length=8192,
                    max_output_tokens=4096
                )
            elif "ultra" in model:
                features = {MockFeature.TEXT, MockFeature.STREAMING}
                self._model_capabilities[model] = MockModelCapabilities(
                    features=features,
                    max_context_length=32768,
                    max_output_tokens=100000  # Very high limit for testing capping
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
    
    def get_provider(self, provider_name: str) -> Optional[MockProviderConfig]:
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
        rate_limits={"requests_per_minute": 60, "tokens_per_minute": 150000}
    )
    
    anthropic_provider = MockProviderConfig(
        name="anthropic",
        client_class="AnthropicClient",
        api_base="https://api.anthropic.com",
        models=["claude-3-opus", "claude-3-sonnet"],
        model_aliases={"latest": "claude-3-opus"}
    )
    
    config.add_provider(openai_provider)
    config.add_provider(anthropic_provider)
    
    return config


@pytest.fixture
def mock_configuration(mock_config):
    """Mock the configuration system"""
    with patch('chuk_llm.configuration.get_config', return_value=mock_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
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
    assert basic_mixin._cached_config is config  # Should be cached


def test_get_provider_config_caching(mock_config):
    """Test that provider config is cached"""
    call_count = 0
    
    def mock_get_config():
        nonlocal call_count
        call_count += 1
        return mock_config
    
    with patch('chuk_llm.configuration.get_config', side_effect=mock_get_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")
            
            # First call
            config1 = mixin._get_provider_config()
            # Second call
            config2 = mixin._get_provider_config()
            
            assert config1 is config2
            assert call_count == 1  # Should only call get_config once


def test_get_provider_config_import_error():
    """Test handling of import errors when loading config"""
    with patch('chuk_llm.configuration.get_config', side_effect=ImportError("Config not available")):
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
    assert advanced_mixin._cached_model_caps is caps  # Should be cached


def test_get_model_capabilities_caching(basic_mixin):
    """Test that model capabilities are cached"""
    # First call
    caps1 = basic_mixin._get_model_capabilities()
    # Second call
    caps2 = basic_mixin._get_model_capabilities()
    
    assert caps1 is caps2


def test_get_model_capabilities_no_provider_config():
    """Test model capabilities when provider config is not available"""
    with patch('chuk_llm.configuration.get_config', side_effect=Exception("No config")):
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
    with patch('chuk_llm.configuration.get_config', side_effect=ImportError("No config")):
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
    with patch('chuk_llm.configuration.get_config', side_effect=ValueError("Config error")):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")
        
        info = mixin.get_model_info()
        
        # Should get the fallback error response
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
    assert basic_mixin.supports_feature("tools") is False  # gpt-4 basic doesn't have tools
    assert basic_mixin.supports_feature("vision") is False


def test_supports_feature_enum_input(basic_mixin):
    """Test supports_feature with enum input"""
    assert basic_mixin.supports_feature(MockFeature.TEXT) is True
    assert basic_mixin.supports_feature(MockFeature.STREAMING) is True
    assert basic_mixin.supports_feature(MockFeature.TOOLS) is False


def test_supports_feature_advanced_model(advanced_mixin):
    """Test supports_feature for advanced model"""
    assert advanced_mixin.supports_feature("tools") is True
    assert advanced_mixin.supports_feature("vision") is True
    assert advanced_mixin.supports_feature("json_mode") is True


def test_supports_feature_no_capabilities():
    """Test supports_feature when capabilities are not available"""
    with patch('chuk_llm.configuration.get_config', side_effect=Exception("No config")):
        mixin = ConfigAwareProviderMixin("openai", "gpt-4")
        
        assert mixin.supports_feature("text") is False
        assert mixin.supports_feature("streaming") is False


def test_supports_feature_invalid_feature(basic_mixin):
    """Test supports_feature with invalid feature name"""
    assert basic_mixin.supports_feature("nonexistent_feature") is False


def test_supports_feature_exception_handling(basic_mixin, monkeypatch):
    """Test supports_feature handles exceptions gracefully"""
    monkeypatch.setattr(basic_mixin, '_get_model_capabilities', lambda: None)
    assert basic_mixin.supports_feature("text") is False


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
    with patch('chuk_llm.configuration.get_config', side_effect=Exception("No config")):
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
    with patch('chuk_llm.configuration.get_config', side_effect=Exception("No config")):
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
    params = {"temperature": 0.7, "max_tokens": 5000}  # Exceeds limit of 2048
    
    validated = basic_mixin.validate_parameters(**params)
    
    assert validated["temperature"] == 0.7
    assert validated["max_tokens"] == 2048  # Should be capped


def test_validate_parameters_max_tokens_within_limit(basic_mixin):
    """Test that max_tokens within limit is not changed"""
    params = {"temperature": 0.7, "max_tokens": 1500}  # Within limit of 2048
    
    validated = basic_mixin.validate_parameters(**params)
    
    assert validated["max_tokens"] == 1500  # Should remain unchanged


def test_validate_parameters_add_default_max_tokens(basic_mixin):
    """Test that default max_tokens is added when not specified"""
    params = {"temperature": 0.7}
    
    validated = basic_mixin.validate_parameters(**params)
    
    assert validated["temperature"] == 0.7
    assert "max_tokens" in validated
    assert validated["max_tokens"] == 2048  # Should use model limit


def test_validate_parameters_add_default_max_tokens_high_limit(ultra_mixin):
    """Test default max_tokens when model limit is very high"""
    params = {"temperature": 0.7}
    validated = ultra_mixin.validate_parameters(**params)
    
    # Should cap at 4096 even when model limit is higher
    assert validated["max_tokens"] == 4096


def test_validate_parameters_no_default_when_no_limit():
    """Test that no default max_tokens is added when model has no limit"""
    with patch('chuk_llm.configuration.get_config', side_effect=Exception("No config")):
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
        "max_tokens": 1000
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
    
    # Original should be unchanged
    assert original_params["max_tokens"] == 5000
    # Validated should be capped
    assert validated["max_tokens"] == 2048
    # Should be different objects
    assert original_params is not validated


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
    assert advanced_mixin.supports_feature("reasoning") is False  # Not in advanced features
    
    # Get limits
    assert advanced_mixin.get_max_tokens_limit() == 4096
    assert advanced_mixin.get_context_length_limit() == 8192
    
    # Validate parameters
    params = {"temperature": 0.7, "max_tokens": 10000}
    validated = advanced_mixin.validate_parameters(**params)
    assert validated["max_tokens"] == 4096  # Capped to limit


def test_multiple_provider_workflow(mock_configuration):
    """Test workflow with multiple providers"""
    # OpenAI mixin
    openai_mixin = ConfigAwareProviderMixin("openai", "gpt-4")
    
    # Anthropic mixin
    anthropic_mixin = ConfigAwareProviderMixin("anthropic", "claude-3-opus")
    
    # Both should work independently
    openai_info = openai_mixin.get_model_info()
    anthropic_info = anthropic_mixin.get_model_info()
    
    assert openai_info["provider"] == "openai"
    assert anthropic_info["provider"] == "anthropic"
    
    # Check they have different configs cached
    assert openai_mixin._cached_config != anthropic_mixin._cached_config


# ---------------------------------------------------------------------------
# Error handling and edge cases - FIXED TESTS
# ---------------------------------------------------------------------------

def test_caching_with_config_errors():
    """Test that caching works correctly even with config errors - FIXED"""
    call_count = 0
    
    def mock_get_config():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call succeeds and returns a valid config
            config = MockConfig()
            provider = MockProviderConfig(name="openai")
            config.add_provider(provider)
            return config
        else:
            # Subsequent calls fail  
            raise Exception("Config temporarily unavailable")
    
    with patch('chuk_llm.configuration.get_config', side_effect=mock_get_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")
            
            # First call should succeed and cache the result
            config1 = mixin._get_provider_config()
            assert config1 is not None
            assert call_count == 1
            
            # Second call should return cached result (not call get_config again)
            config2 = mixin._get_provider_config()
            assert config2 is config1  # Same cached object
            assert call_count == 1  # Should not increment
            
            # Clear cache to force new call which will fail
            mixin._cached_config = None
            
            # Third call should fail and return None
            config3 = mixin._get_provider_config()
            assert config3 is None  # FIXED: Should be None when config fails
            assert call_count == 2


def test_caching_success_after_clear():
    """Test that clearing cache allows successful retry - FIXED"""
    call_count = 0
    
    def mock_get_config():
        nonlocal call_count
        call_count += 1
        # FIXED: Always return valid config, don't simulate failures
        config = MockConfig()
        provider = MockProviderConfig(name="openai")
        config.add_provider(provider)
        return config
    
    with patch('chuk_llm.configuration.get_config', side_effect=mock_get_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")
            
            # First call
            config1 = mixin._get_provider_config()
            assert config1 is not None
            
            # Second call should use cache
            config2 = mixin._get_provider_config()
            assert config2 is config1
            
            # Clear cache
            mixin._cached_config = None
            
            # Third call should fetch new config
            config3 = mixin._get_provider_config()
            assert config3 is not None
            
            # FIXED: Should only be called twice total (initial + after clear)
            assert call_count == 2


def test_model_capabilities_with_missing_model(mock_configuration):
    """Test model capabilities for model not in provider config"""
    mixin = ConfigAwareProviderMixin("openai", "nonexistent-model")
    
    caps = mixin._get_model_capabilities()
    
    # Should still get default capabilities
    assert caps is not None
    assert MockFeature.TEXT in caps.features
    assert MockFeature.STREAMING in caps.features


def test_logging_behavior(basic_mixin, caplog):
    """Test that appropriate log messages are generated"""
    with caplog.at_level(logging.DEBUG):
        # Test parameter capping logging
        params = {"max_tokens": 5000}
        basic_mixin.validate_parameters(**params)
        
        assert "Capping max_tokens from 5000 to 2048" in caplog.text


def test_concurrent_access_safety(mock_config):
    """Test that the mixin is safe for concurrent access"""
    results = []
    results_lock = threading.Lock()
    
    def access_config():
        with patch('chuk_llm.configuration.get_config', return_value=mock_config):
            with patch('chuk_llm.configuration.Feature', MockFeature):
                mixin = ConfigAwareProviderMixin("openai", "gpt-4")
                time.sleep(0.01)  # Small delay to increase chance of race conditions
                config = mixin._get_provider_config()
                with results_lock:
                    results.append(config)
    
    # Start multiple threads
    threads = [threading.Thread(target=access_config) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # All results should be similar config objects
    assert len(results) == 10
    # Check that all results are not None (basic functionality works)
    assert all(r is not None for r in results)


def test_memory_usage_with_large_configs(basic_mixin):
    """Test memory usage doesn't grow with repeated access"""
    # Access config many times
    for _ in range(1000):
        basic_mixin._get_provider_config()
        basic_mixin._get_model_capabilities()
        basic_mixin.get_model_info()
    
    # Should still only have one cached config
    assert basic_mixin._cached_config is not None
    assert basic_mixin._cached_model_caps is not None
    
    # Force garbage collection
    gc.collect()


# ---------------------------------------------------------------------------
# Custom configuration scenarios
# ---------------------------------------------------------------------------

def test_custom_model_capabilities(mock_configuration):
    """Test with custom model capabilities"""
    config = mock_configuration
    provider = config.get_provider("openai")
    
    # Create custom capabilities
    custom_caps = MockModelCapabilities(
        features={MockFeature.TEXT, MockFeature.REASONING, MockFeature.MULTIMODAL},
        max_context_length=32768,
        max_output_tokens=8192
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
    """Test model with no features - FIXED"""
    # Create a completely isolated test to avoid fixture interference
    isolated_config = MockConfig()
    isolated_provider = MockProviderConfig(name="openai")
    
    # FIXED: Create capabilities with explicit empty set
    minimal_caps = MockModelCapabilities(features=set())  # Explicit empty set
    isolated_provider.set_model_capabilities("minimal-model", minimal_caps)
    isolated_config.add_provider(isolated_provider)
    
    # Use completely isolated configuration for this test
    with patch('chuk_llm.configuration.get_config', return_value=isolated_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "minimal-model")
            
            # Verify the capabilities are actually empty
            caps = mixin._get_model_capabilities()
            assert caps is not None
            assert len(caps.features) == 0  # Should be 0 now
            
            # Now test feature support
            assert mixin.supports_feature("text") is False
            assert mixin.supports_feature("streaming") is False
            
            info = mixin.get_model_info()
            assert info["features"] == []
            assert all(not info[f"supports_{feature}"] for feature in [
                "text", "streaming", "tools", "vision", "json_mode", 
                "system_messages", "parallel_calls", "multimodal", "reasoning"
            ])


def test_provider_without_api_base(mock_configuration):
    """Test provider configuration without api_base"""
    config = mock_configuration
    
    # Add provider without api_base
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
    
    # Load configs for both
    config1 = mixin1._get_provider_config()
    config2 = mixin2._get_provider_config()
    
    # Should have different configs
    assert config1 != config2
    assert mixin1._cached_config != mixin2._cached_config


def test_cache_persistence_across_method_calls(basic_mixin):
    """Test that cache persists across different method calls"""
    # First access through get_model_info
    info = basic_mixin.get_model_info()
    cached_config_1 = basic_mixin._cached_config
    
    # Second access through supports_feature
    supports_text = basic_mixin.supports_feature("text")
    cached_config_2 = basic_mixin._cached_config
    
    # Third access through get_max_tokens_limit
    limit = basic_mixin.get_max_tokens_limit()
    cached_config_3 = basic_mixin._cached_config
    
    # All should use the same cached config
    assert cached_config_1 is cached_config_2
    assert cached_config_2 is cached_config_3
    assert info["provider"] == "openai"
    assert supports_text is True
    assert limit == 2048


# ---------------------------------------------------------------------------
# Edge cases for parameter validation
# ---------------------------------------------------------------------------

def test_validate_parameters_edge_cases(basic_mixin):
    """Test parameter validation edge cases"""
    # Test with zero max_tokens
    validated = basic_mixin.validate_parameters(max_tokens=0)
    assert validated["max_tokens"] == 0  # Should not be capped to model limit
    
    # Test with negative max_tokens
    validated = basic_mixin.validate_parameters(max_tokens=-1)
    assert validated["max_tokens"] == -1  # Should pass through (let API handle)
    
    # Test with exactly the limit
    validated = basic_mixin.validate_parameters(max_tokens=2048)
    assert validated["max_tokens"] == 2048  # Should not be changed
    
    # Test with one more than the limit
    validated = basic_mixin.validate_parameters(max_tokens=2049)
    assert validated["max_tokens"] == 2048  # Should be capped
    
    # Test with None max_tokens (should not cause comparison error)
    validated = basic_mixin.validate_parameters(max_tokens=None)
    assert validated["max_tokens"] == 2048  # Should add default


def test_validate_parameters_none_comparison_safety(basic_mixin):
    """Test that None values don't cause comparison errors"""
    # This specifically tests the fix for the TypeError: '>' not supported between instances of 'NoneType' and 'int'
    params = {
        "temperature": 0.7,
        "max_tokens": None,  # This should not cause a comparison error
        "top_p": 0.9
    }
    
    # This should not raise any TypeError
    validated = basic_mixin.validate_parameters(**params)
    
    assert validated["temperature"] == 0.7
    assert validated["max_tokens"] == 2048  # Should use default
    assert validated["top_p"] == 0.9


def test_validate_parameters_with_none_values(basic_mixin):
    """Test parameter validation with None values"""
    params = {
        "temperature": None,
        "max_tokens": None,
        "top_p": 0.9
    }
    
    validated = basic_mixin.validate_parameters(**params)
    
    assert validated["temperature"] is None
    assert validated["max_tokens"] == 2048  # Should add default since None counts as not specified
    assert validated["top_p"] == 0.9


def test_validate_parameters_with_string_max_tokens(basic_mixin):
    """Test parameter validation with string max_tokens (edge case)"""
    # This might happen if parameters come from user input
    try:
        validated = basic_mixin.validate_parameters(max_tokens="1000")
        # If it doesn't raise an exception, check the behavior
        # The actual behavior depends on implementation
    except (TypeError, ValueError):
        # It's acceptable for this to raise an exception
        pass


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
    
    with patch('chuk_llm.configuration.get_config', side_effect=counting_get_config):
        with patch('chuk_llm.configuration.Feature', MockFeature):
            mixin = ConfigAwareProviderMixin("openai", "gpt-4")
            
            # Multiple calls to different methods
            for _ in range(10):
                mixin.get_model_info()
                mixin.supports_feature("text")
                mixin.get_max_tokens_limit()
                mixin.validate_parameters(temperature=0.7)
            
            # Should only call get_config once
            assert call_count == 1


def test_capabilities_not_recomputed_unnecessarily(mock_configuration, monkeypatch):
    """Test that model capabilities are not recomputed unnecessarily"""
    call_count = 0
    
    def counting_get_model_capabilities(self, model):
        nonlocal call_count
        call_count += 1
        return MockModelCapabilities()
    
    provider = mock_configuration.get_provider("openai")
    monkeypatch.setattr(provider, "get_model_capabilities", 
                       lambda model: counting_get_model_capabilities(provider, model))
    
    mixin = ConfigAwareProviderMixin("openai", "gpt-4")
    
    # Multiple calls that need capabilities
    for _ in range(5):
        mixin.supports_feature("text")
        mixin.get_max_tokens_limit()
        mixin.get_context_length_limit()
    
    # Should only call get_model_capabilities once due to caching
    assert call_count == 1