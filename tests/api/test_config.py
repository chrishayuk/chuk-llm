"""
Comprehensive pytest tests for chuk_llm/api/config.py (updated version)

Run with:
    pytest tests/api/test_config.py -v
    pytest tests/api/test_config.py -v --tb=short
    pytest tests/api/test_config.py::TestAPIConfig::test_configure_basic -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the module under test
from chuk_llm.api import config
from chuk_llm.api.config import (
    APIConfig,
    configure,
    get_current_config,
    get_client,
    reset,
    validate_config,
    get_capabilities,
    supports_feature,
    auto_configure,
    debug_config_state,
    quick_setup,
    switch_provider,
)


class TestAPIConfigClass:
    """Test suite for the APIConfig class."""

    def test_api_config_initialization(self):
        """Test APIConfig initializes with empty overrides."""
        api_config = APIConfig()
        assert api_config.overrides == {}
        assert api_config._cached_client is None
        assert api_config._cache_key is None

    def test_api_config_set_method(self):
        """Test APIConfig.set() method updates overrides."""
        api_config = APIConfig()
        
        api_config.set(provider="anthropic", model="claude-3-sonnet", temperature=0.8)
        
        assert api_config.overrides["provider"] == "anthropic"
        assert api_config.overrides["model"] == "claude-3-sonnet"
        assert api_config.overrides["temperature"] == 0.8

    def test_api_config_set_ignores_none_values(self):
        """Test that set() ignores None values."""
        api_config = APIConfig()
        
        api_config.set(provider="openai", model=None, temperature=0.5)
        
        assert api_config.overrides["provider"] == "openai"
        assert "model" not in api_config.overrides
        assert api_config.overrides["temperature"] == 0.5

    def test_api_config_set_invalidates_cache(self):
        """Test that set() invalidates cached client."""
        api_config = APIConfig()
        
        # Set up cached client
        mock_client = Mock()
        api_config._cached_client = mock_client
        api_config._cache_key = ("test", "test", None, None)
        
        # Setting new config should invalidate cache
        api_config.set(provider="anthropic")
        
        assert api_config._cached_client is None
        assert api_config._cache_key is None

    @patch('chuk_llm.configuration.get_config')
    def test_api_config_get_current_config(self, mock_get_config):
        """Test APIConfig.get_current_config() method."""
        # Mock the unified config manager
        mock_config_manager = Mock()
        mock_config_manager.global_settings = {
            "active_provider": "openai",
            "active_model": "gpt-4o-mini"
        }
        
        mock_provider_config = Mock()
        mock_provider_config.default_model = "gpt-4o-mini"
        mock_provider_config.api_base = None
        
        # Mock model capabilities
        mock_model_caps = Mock()
        mock_model_caps.features = {Mock(value="text"), Mock(value="streaming")}
        mock_model_caps.max_context_length = 128000
        mock_model_caps.max_output_tokens = 4096
        mock_provider_config.get_model_capabilities.return_value = mock_model_caps
        
        mock_config_manager.get_provider.return_value = mock_provider_config
        mock_config_manager.get_api_key.return_value = "sk-test123"
        mock_get_config.return_value = mock_config_manager
        
        api_config = APIConfig()
        api_config.set(temperature=0.8, max_tokens=1000)
        
        current_config = api_config.get_current_config()
        
        # Should have global defaults
        assert current_config["provider"] == "openai"
        assert current_config["model"] == "gpt-4o-mini"
        assert current_config["api_key"] == "sk-test123"
        
        # Should have overrides
        assert current_config["temperature"] == 0.8
        assert current_config["max_tokens"] == 1000
        
        # Should have capabilities
        assert "_capabilities" in current_config
        assert current_config["_capabilities"]["max_context_length"] == 128000

    @patch('chuk_llm.llm.client.get_client')
    def test_api_config_get_client_creates_new(self, mock_get_client):
        """Test APIConfig.get_client() creates new client."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        api_config = APIConfig()
        
        with patch.object(api_config, 'get_current_config') as mock_get_config:
            mock_get_config.return_value = {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "sk-test",
                "api_base": None
            }
            
            result = api_config.get_client()
            
            assert result == mock_client
            mock_get_client.assert_called_once_with(
                provider="openai",
                model="gpt-4",
                api_key="sk-test",
                api_base=None
            )

    @patch('chuk_llm.llm.client.get_client')
    def test_api_config_get_client_uses_cache(self, mock_get_client):
        """Test APIConfig.get_client() uses cached client when config matches."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        api_config = APIConfig()
        
        test_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-test",
            "api_base": None
        }
        
        with patch.object(api_config, 'get_current_config', return_value=test_config):
            # First call - creates client
            result1 = api_config.get_client()
            
            # Second call - should use cached client
            result2 = api_config.get_client()
            
            assert result1 == result2 == mock_client
            # get_client should only be called once
            mock_get_client.assert_called_once()

    @patch('chuk_llm.configuration.unified_config.ConfigValidator')
    def test_api_config_validate_current_config(self, mock_validator):
        """Test APIConfig.validate_current_config() method."""
        mock_validator.validate_request_compatibility.return_value = (True, [])
        
        api_config = APIConfig()
        api_config.set(provider="openai", model="gpt-4")
        
        result = api_config.validate_current_config()
        
        assert result["valid"] is True
        assert result["issues"] == []
        assert "config" in result

    @patch('chuk_llm.llm.client.get_provider_info')
    def test_api_config_get_provider_capabilities(self, mock_get_info):
        """Test APIConfig.get_provider_capabilities() method."""
        # Mock to return an error since get_provider_info might not exist
        mock_get_info.side_effect = ImportError("Not implemented")
        
        api_config = APIConfig()
        api_config.set(provider="openai", model="gpt-4")
        
        caps = api_config.get_provider_capabilities()
        
        # Should return error dict when import fails
        assert isinstance(caps, dict)
        assert "error" in caps

    @patch('chuk_llm.configuration.unified_config.get_config')
    def test_api_config_supports_feature(self, mock_get_config):
        """Test APIConfig.supports_feature() method."""
        mock_config_manager = Mock()
        mock_config_manager.supports_feature.return_value = True
        mock_get_config.return_value = mock_config_manager
        
        api_config = APIConfig()
        api_config.set(provider="openai", model="gpt-4")
        
        result = api_config.supports_feature("streaming")
        
        assert result is True
        mock_config_manager.supports_feature.assert_called_once()

    @patch('chuk_llm.configuration.unified_config.get_config')
    def test_api_config_auto_configure_for_task(self, mock_get_config):
        """Test APIConfig.auto_configure_for_task() method."""
        mock_config_manager = Mock()
        mock_config_manager.get_all_providers.return_value = ["openai", "anthropic"]
        mock_get_config.return_value = mock_config_manager
        
        api_config = APIConfig()
        
        result = api_config.auto_configure_for_task("general")
        
        assert result is True
        assert api_config.overrides["provider"] == "openai"

    def test_api_config_reset(self):
        """Test APIConfig.reset() clears overrides and cache."""
        api_config = APIConfig()
        
        # Set some overrides and cache
        api_config.set(provider="anthropic", temperature=0.8)
        api_config._cached_client = Mock()
        api_config._cache_key = ("test", "test", None, None)
        
        api_config.reset()
        
        assert api_config.overrides == {}
        assert api_config._cached_client is None
        assert api_config._cache_key is None


class TestConfigureFunctionAPI:
    """Test suite for the module-level API functions."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    def test_configure_basic(self):
        """Test basic configure function."""
        configure(provider="anthropic", model="claude-3-sonnet")
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["model"] == "claude-3-sonnet"

    def test_configure_all_parameters(self):
        """Test configuring all available parameters."""
        configure(
            provider="anthropic",
            model="claude-3-opus",
            system_prompt="You are a helpful assistant",
            temperature=0.8,
            max_tokens=2000,
            api_key="test-key-123",
            api_base="https://api.test.com"
        )
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["model"] == "claude-3-opus"
        assert config_dict["system_prompt"] == "You are a helpful assistant"
        assert config_dict["temperature"] == 0.8
        assert config_dict["max_tokens"] == 2000
        assert config_dict["api_key"] == "test-key-123"
        assert config_dict["api_base"] == "https://api.test.com"

    def test_configure_none_values_ignored(self):
        """Test that None values are ignored during configuration."""
        configure(provider="openai", model="gpt-4", temperature=0.5)
        
        # Try to set some values to None - they should be ignored
        configure(provider="anthropic", model=None, temperature=None)
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"  # Should be updated
        # model and temperature should keep their previous values due to None being ignored

    def test_configure_multiple_calls_accumulate(self):
        """Test multiple configure calls accumulate changes."""
        configure(provider="openai", model="gpt-4")
        configure(temperature=0.7, max_tokens=1000)
        configure(api_key="test-key")
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "openai"
        assert config_dict["model"] == "gpt-4"
        assert config_dict["temperature"] == 0.7
        assert config_dict["max_tokens"] == 1000
        assert config_dict["api_key"] == "test-key"

    def test_configure_overwrites_previous_values(self):
        """Test that configure overwrites previously set values."""
        configure(provider="openai", temperature=0.5)
        configure(provider="anthropic", temperature=0.9)
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["temperature"] == 0.9


class TestGetCurrentConfig:
    """Test suite for get_current_config function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    @patch('chuk_llm.configuration.get_config')
    def test_get_current_config_with_defaults(self, mock_get_config):
        """Test get_current_config with no overrides uses defaults."""
        # Mock the unified config manager
        mock_config_manager = Mock()
        mock_config_manager.global_settings = {
            "active_provider": "openai",
            "active_model": "gpt-4o-mini"
        }
        
        mock_provider_config = Mock()
        mock_provider_config.default_model = "gpt-4o-mini"
        mock_provider_config.api_base = "https://api.openai.com"
        
        # Mock model capabilities
        mock_model_caps = Mock()
        mock_model_caps.features = set()
        mock_model_caps.max_context_length = 128000
        mock_model_caps.max_output_tokens = 4096
        mock_provider_config.get_model_capabilities.return_value = mock_model_caps
        
        mock_config_manager.get_provider.return_value = mock_provider_config
        mock_config_manager.get_api_key.return_value = "sk-default123"
        mock_get_config.return_value = mock_config_manager
        
        config_dict = get_current_config()
        
        # Should use global defaults
        assert config_dict["provider"] == "openai"
        assert config_dict["model"] == "gpt-4o-mini"
        assert config_dict["api_key"] == "sk-default123"
        assert config_dict["api_base"] == "https://api.openai.com"

    def test_get_current_config_with_overrides(self):
        """Test get_current_config reflects overrides."""
        configure(provider="anthropic", temperature=0.8)
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["temperature"] == 0.8

    @patch('chuk_llm.configuration.get_config')
    def test_get_current_config_handles_provider_error(self, mock_get_config):
        """Test get_current_config handles unknown provider gracefully."""
        # Mock config manager that raises ValueError for unknown provider
        mock_config_manager = Mock()
        mock_config_manager.global_settings = {
            "active_provider": "unknown_provider"
        }
        mock_config_manager.get_provider.side_effect = ValueError("Unknown provider")
        mock_get_config.return_value = mock_config_manager
        
        # Should not raise exception
        config_dict = get_current_config()
        
        # Should have basic structure
        assert "provider" in config_dict
        assert "model" in config_dict


class TestGetClient:
    """Test suite for get_client function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    @patch('chuk_llm.llm.client.get_client')
    def test_get_client_with_current_config(self, mock_get_client):
        """Test get_client uses current configuration."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        configure(provider="anthropic", model="claude-3-sonnet", api_key="sk-test")
        
        result = get_client()
        
        assert result == mock_client
        mock_get_client.assert_called_once()
        
        # Verify it was called with expected config
        call_args = mock_get_client.call_args
        assert call_args[1]["provider"] == "anthropic"
        assert call_args[1]["model"] == "claude-3-sonnet"
        assert call_args[1]["api_key"] == "sk-test"

    @patch('chuk_llm.llm.client.get_client')
    def test_get_client_caching_behavior(self, mock_get_client):
        """Test get_client caches clients properly."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        configure(provider="openai", model="gpt-4")
        
        # Multiple calls with same config
        result1 = get_client()
        result2 = get_client()
        result3 = get_client()
        
        assert result1 == result2 == result3 == mock_client
        # Should only create client once due to caching
        mock_get_client.assert_called_once()

    @patch('chuk_llm.llm.client.get_client')
    def test_get_client_cache_invalidation_on_config_change(self, mock_get_client):
        """Test get_client creates new client when config changes."""
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_get_client.side_effect = [mock_client1, mock_client2]
        
        # First configuration
        configure(provider="openai", model="gpt-4")
        result1 = get_client()
        
        # Change configuration - should invalidate cache
        configure(provider="anthropic", model="claude-3-sonnet")
        result2 = get_client()
        
        assert result1 == mock_client1
        assert result2 == mock_client2
        assert result1 != result2
        assert mock_get_client.call_count == 2


class TestValidateConfig:
    """Test suite for validate_config function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    @patch('chuk_llm.configuration.unified_config.ConfigValidator')
    def test_validate_config_valid(self, mock_validator):
        """Test validate_config with valid configuration."""
        mock_validator.validate_request_compatibility.return_value = (True, [])
        
        configure(provider="openai", model="gpt-4")
        
        result = validate_config()
        
        assert result["valid"] is True
        assert result["issues"] == []
        assert result["config"]["provider"] == "openai"

    @patch('chuk_llm.configuration.unified_config.ConfigValidator')
    def test_validate_config_invalid(self, mock_validator):
        """Test validate_config with invalid configuration."""
        mock_validator.validate_request_compatibility.return_value = (
            False, 
            ["Missing API key", "Invalid model"]
        )
        
        configure(provider="test", model="invalid")
        
        result = validate_config()
        
        assert result["valid"] is False
        assert len(result["issues"]) == 2
        assert "Missing API key" in result["issues"]


class TestGetCapabilities:
    """Test suite for get_capabilities function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    @patch('chuk_llm.llm.client.get_provider_info')
    def test_get_capabilities(self, mock_get_info):
        """Test get_capabilities returns provider capabilities."""
        # Mock to return an error since get_provider_info might not exist
        mock_get_info.side_effect = ImportError("Not implemented")
        
        configure(provider="openai", model="gpt-4")
        
        caps = get_capabilities()
        
        # Should return error dict when import fails
        assert isinstance(caps, dict)
        assert "error" in caps


class TestSupportsFeature:
    """Test suite for supports_feature function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    @patch('chuk_llm.configuration.get_config')
    def test_supports_feature_true(self, mock_get_config):
        """Test supports_feature returns True for supported features."""
        mock_config_manager = Mock()
        mock_config_manager.supports_feature.return_value = True
        mock_get_config.return_value = mock_config_manager
        
        configure(provider="openai", model="gpt-4")
        
        result = supports_feature("streaming")
        
        assert result is True

    @patch('chuk_llm.configuration.get_config')
    def test_supports_feature_false(self, mock_get_config):
        """Test supports_feature returns False for unsupported features."""
        mock_config_manager = Mock()
        mock_config_manager.supports_feature.return_value = False
        mock_get_config.return_value = mock_config_manager
        
        configure(provider="test", model="basic")
        
        result = supports_feature("advanced_reasoning")
        
        assert result is False


class TestAutoConfigureAndQuickSetup:
    """Test suite for auto_configure and quick_setup functions."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    @patch('chuk_llm.configuration.get_config')
    def test_auto_configure_success(self, mock_get_config):
        """Test auto_configure successfully configures for task."""
        mock_config_manager = Mock()
        mock_config_manager.get_all_providers.return_value = ["openai", "anthropic"]
        mock_get_config.return_value = mock_config_manager
        
        result = auto_configure("general")
        
        assert result is True
        config_dict = get_current_config()
        assert config_dict["provider"] == "openai"

    def test_quick_setup_valid(self):
        """Test quick_setup with valid provider/model."""
        with patch.object(config, 'validate_config') as mock_validate:
            mock_validate.return_value = {"valid": True, "issues": []}
            
            result = quick_setup("anthropic", "claude-3-sonnet")
            
            assert result is True
            config_dict = get_current_config()
            assert config_dict["provider"] == "anthropic"
            assert config_dict["model"] == "claude-3-sonnet"

    def test_quick_setup_invalid(self):
        """Test quick_setup with invalid configuration."""
        with patch.object(config, 'validate_config') as mock_validate:
            mock_validate.return_value = {
                "valid": False, 
                "issues": ["Invalid provider"]
            }
            
            result = quick_setup("invalid", "model")
            
            assert result is False


class TestSwitchProvider:
    """Test suite for switch_provider function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    def test_switch_provider_success(self):
        """Test successful provider switch."""
        # Set initial provider
        configure(provider="openai", model="gpt-4")
        
        with patch.object(config, 'validate_config') as mock_validate:
            mock_validate.return_value = {"valid": True, "issues": []}
            
            result = switch_provider("anthropic", "claude-3-sonnet")
            
            assert result is True
            config_dict = get_current_config()
            assert config_dict["provider"] == "anthropic"
            assert config_dict["model"] == "claude-3-sonnet"

    def test_switch_provider_failure_reverts(self):
        """Test failed provider switch reverts to original."""
        # Set initial provider
        configure(provider="openai", model="gpt-4")
        
        with patch.object(config, 'validate_config') as mock_validate:
            mock_validate.return_value = {
                "valid": False, 
                "issues": ["Invalid configuration"]
            }
            
            result = switch_provider("invalid", "model")
            
            assert result is False
            # Should revert to original
            config_dict = get_current_config()
            assert config_dict["provider"] == "openai"
            assert config_dict["model"] == "gpt-4"


class TestDebugConfigState:
    """Test suite for debug_config_state function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    @patch('builtins.print')
    def test_debug_config_state_output(self, mock_print):
        """Test debug_config_state prints expected information."""
        configure(provider="openai", model="gpt-4", api_key="test-key")
        
        with patch.object(config, 'validate_config') as mock_validate:
            mock_validate.return_value = {"valid": True, "issues": []}
            
            with patch.object(config, 'get_capabilities') as mock_caps:
                mock_caps.return_value = {"supports": {"streaming": True}}
                
                result = debug_config_state()
                
                # Check it returns debug info
                assert "config" in result
                assert "validation" in result
                assert result["config"]["provider"] == "openai"
                assert result["config"]["has_api_key"] is True
                
                # Check it prints
                mock_print.assert_called()
                # Should print provider info
                calls = [str(call) for call in mock_print.call_args_list]
                assert any("openai" in call for call in calls)


class TestReset:
    """Test suite for reset function."""

    def test_reset_clears_overrides(self):
        """Test reset clears all configuration overrides."""
        configure(
            provider="anthropic",
            model="claude-3-opus",
            temperature=0.8
        )
        
        # Verify config was set
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["temperature"] == 0.8
        
        # Reset
        reset()
        
        # After reset, overrides should be cleared
        config_dict = get_current_config()
        # Provider should revert to default (from global settings)
        assert config_dict["provider"] != "anthropic"
        assert config_dict.get("temperature") is None

    @patch('chuk_llm.llm.client.get_client')
    def test_reset_clears_cached_client(self, mock_get_client):
        """Test reset clears cached client."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        configure(provider="openai", model="gpt-4")
        
        # Create cached client
        get_client()
        
        # Reset should clear cache
        reset()
        
        # Next call should create new client
        get_client()
        
        # Should have been called twice (once before reset, once after)
        assert mock_get_client.call_count == 2


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    def test_configure_with_empty_kwargs(self):
        """Test configure with no arguments."""
        original_config = get_current_config()
        
        configure()  # No arguments
        
        updated_config = get_current_config()
        # Should be essentially the same (no changes made)
        assert updated_config.keys() == original_config.keys()

    def test_configure_with_zero_and_false_values(self):
        """Test that zero and False values are properly set (not treated as None)."""
        configure(temperature=0, max_tokens=0, stream=False, json_mode=False)
        
        config_dict = get_current_config()
        assert config_dict["temperature"] == 0
        assert config_dict["max_tokens"] == 0
        assert config_dict["stream"] is False
        assert config_dict["json_mode"] is False

    def test_configure_with_empty_strings(self):
        """Test configure with empty string values."""
        configure(provider="", model="", api_key="")
        
        config_dict = get_current_config()
        assert config_dict["provider"] == ""
        assert config_dict["model"] == ""
        assert config_dict["api_key"] == ""

    @patch('chuk_llm.configuration.get_config')
    def test_get_current_config_with_missing_global_settings(self, mock_get_config):
        """Test get_current_config when global settings are missing."""
        mock_config_manager = Mock()
        mock_config_manager.global_settings = {}  # Empty settings
        mock_config_manager.get_provider.side_effect = ValueError("No provider")
        mock_get_config.return_value = mock_config_manager
        
        # Should not crash, should provide reasonable defaults
        config_dict = get_current_config()
        assert isinstance(config_dict, dict)
        assert "provider" in config_dict
        assert "model" in config_dict


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])