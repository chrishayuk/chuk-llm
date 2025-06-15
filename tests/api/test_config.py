"""
Comprehensive pytest tests for chuk_llm/api/config.py (clean version)

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
    configure,
    get_current_config,
    get_client,
    reset
)


class TestAPIConfigClass:
    """Test suite for the APIConfig class."""

    def test_api_config_initialization(self):
        """Test APIConfig initializes with empty overrides."""
        api_config = config.APIConfig()
        assert api_config.overrides == {}
        assert api_config._cached_client is None
        assert api_config._cache_key is None

    def test_api_config_set_method(self):
        """Test APIConfig.set() method updates overrides."""
        api_config = config.APIConfig()
        
        api_config.set(provider="anthropic", model="claude-3-sonnet", temperature=0.8)
        
        assert api_config.overrides["provider"] == "anthropic"
        assert api_config.overrides["model"] == "claude-3-sonnet"
        assert api_config.overrides["temperature"] == 0.8

    def test_api_config_set_ignores_none_values(self):
        """Test that set() ignores None values."""
        api_config = config.APIConfig()
        
        api_config.set(provider="openai", model=None, temperature=0.5)
        
        assert api_config.overrides["provider"] == "openai"
        assert "model" not in api_config.overrides
        assert api_config.overrides["temperature"] == 0.5

    def test_api_config_set_invalidates_cache(self):
        """Test that set() invalidates cached client."""
        api_config = config.APIConfig()
        
        # Set up cached client
        mock_client = Mock()
        api_config._cached_client = mock_client
        api_config._cache_key = ("test", "test", None, None)
        
        # Setting new config should invalidate cache
        api_config.set(provider="anthropic")
        
        assert api_config._cached_client is None
        assert api_config._cache_key is None

    @patch('chuk_llm.configuration.config.get_config')
    def test_api_config_get_current_config(self, mock_get_config):
        """Test APIConfig.get_current_config() method."""
        # Mock the core config manager
        mock_config_manager = Mock()
        mock_config_manager.get_global_settings.return_value = {
            "active_provider": "openai",
            "active_model": "gpt-4o-mini"
        }
        mock_provider_config = Mock()
        mock_provider_config.default_model = "gpt-4o-mini"
        mock_provider_config.api_base = None
        mock_config_manager.get_provider.return_value = mock_provider_config
        mock_config_manager.get_api_key.return_value = "sk-test123"
        mock_get_config.return_value = mock_config_manager
        
        api_config = config.APIConfig()
        api_config.set(temperature=0.8, max_tokens=1000)
        
        current_config = api_config.get_current_config()
        
        # Should have global defaults
        assert current_config["provider"] == "openai"
        assert current_config["model"] == "gpt-4o-mini"
        assert current_config["api_key"] == "sk-test123"
        
        # Should have overrides
        assert current_config["temperature"] == 0.8
        assert current_config["max_tokens"] == 1000

    @patch('chuk_llm.llm.client.get_client')
    def test_api_config_get_client_creates_new(self, mock_get_client):
        """Test APIConfig.get_client() creates new client."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        api_config = config.APIConfig()
        
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
        
        api_config = config.APIConfig()
        
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

    def test_api_config_reset(self):
        """Test APIConfig.reset() clears overrides and cache."""
        api_config = config.APIConfig()
        
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
        # model and temperature should remain from first configure call
        # (exact behavior depends on how global defaults work)

    def test_configure_with_custom_kwargs(self):
        """Test configure with custom parameters."""
        configure(
            provider="openai",
            custom_param="custom_value",
            another_param=42,
            debug=True
        )
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "openai"
        assert config_dict["custom_param"] == "custom_value"
        assert config_dict["another_param"] == 42
        assert config_dict["debug"] is True

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

    @patch('chuk_llm.configuration.config.get_config')
    def test_get_current_config_with_defaults(self, mock_get_config):
        """Test get_current_config with no overrides uses defaults."""
        # Mock the core config manager
        mock_config_manager = Mock()
        mock_config_manager.get_global_settings.return_value = {
            "active_provider": "openai",
            "active_model": "gpt-4o-mini"
        }
        mock_provider_config = Mock()
        mock_provider_config.default_model = "gpt-4o-mini"
        mock_provider_config.api_base = "https://api.openai.com"
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
        configure(provider="anthropic", temperature=0.8, custom_param="test")
        
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["temperature"] == 0.8
        assert config_dict["custom_param"] == "test"

    @patch('chuk_llm.configuration.config.get_config')
    def test_get_current_config_handles_provider_error(self, mock_get_config):
        """Test get_current_config handles unknown provider gracefully."""
        # Mock config manager that raises ValueError for unknown provider
        mock_config_manager = Mock()
        mock_config_manager.get_global_settings.return_value = {
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

    @patch('chuk_llm.llm.client.get_client')
    def test_get_client_handles_creation_error(self, mock_get_client):
        """Test get_client handles client creation errors."""
        mock_get_client.side_effect = ValueError("Invalid provider configuration")
        
        configure(provider="invalid_provider")
        
        with pytest.raises(ValueError, match="Invalid provider configuration"):
            get_client()


class TestReset:
    """Test suite for reset function."""

    def test_reset_clears_overrides(self):
        """Test reset clears all configuration overrides."""
        configure(
            provider="anthropic",
            model="claude-3-opus",
            temperature=0.8,
            custom_param="test_value"
        )
        
        # Verify config was set
        config_dict = get_current_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["temperature"] == 0.8
        assert config_dict["custom_param"] == "test_value"
        
        # Reset
        reset()
        
        # After reset, should not have the overrides
        # (exact behavior depends on global defaults)
        config_dict = get_current_config()
        assert config_dict.get("temperature") is None
        assert "custom_param" not in config_dict

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


class TestGlobalAPIConfigInstance:
    """Test suite for the global _api_config instance behavior."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset()

    def test_module_functions_use_global_instance(self):
        """Test that module-level functions use the same global instance."""
        # This test verifies that configure, get_current_config, etc. 
        # all operate on the same underlying APIConfig instance
        
        configure(provider="test_provider", temperature=0.5)
        config1 = get_current_config()
        
        configure(model="test_model")
        config2 = get_current_config()
        
        # Both calls should reflect accumulated changes
        assert config2["provider"] == "test_provider"  # From first configure
        assert config2["model"] == "test_model"        # From second configure
        assert config2["temperature"] == 0.5          # Should persist

    def test_state_persistence_across_function_calls(self):
        """Test that state persists across multiple function calls."""
        # Set initial config
        configure(provider="openai", model="gpt-4")
        
        # Verify persistence through multiple get_current_config calls
        config1 = get_current_config()
        config2 = get_current_config()
        config3 = get_current_config()
        
        assert config1["provider"] == config2["provider"] == config3["provider"] == "openai"
        assert config1["model"] == config2["model"] == config3["model"] == "gpt-4"


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
        configure(temperature=0, max_tokens=0, debug=False, streaming=False)
        
        config_dict = get_current_config()
        assert config_dict["temperature"] == 0
        assert config_dict["max_tokens"] == 0
        assert config_dict["debug"] is False
        assert config_dict["streaming"] is False

    def test_configure_with_empty_strings(self):
        """Test configure with empty string values."""
        configure(provider="", model="", api_key="")
        
        config_dict = get_current_config()
        assert config_dict["provider"] == ""
        assert config_dict["model"] == ""
        assert config_dict["api_key"] == ""

    def test_get_current_config_returns_copy(self):
        """Test that get_current_config returns a copy that can be safely modified."""
        configure(provider="openai", temperature=0.5)
        
        config1 = get_current_config()
        config1["provider"] = "modified"
        config1["new_key"] = "new_value"
        
        # Original config should be unchanged
        config2 = get_current_config()
        assert config2["provider"] == "openai"
        assert "new_key" not in config2

    @patch('chuk_llm.configuration.config.get_config')
    def test_get_current_config_with_missing_global_settings(self, mock_get_config):
        """Test get_current_config when global settings are missing."""
        mock_config_manager = Mock()
        mock_config_manager.get_global_settings.return_value = {}  # Empty settings
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