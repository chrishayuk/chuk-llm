"""
Comprehensive pytest tests for chuk_llm/api/config.py

Run with:
    pytest tests/api/test_config.py -v
    pytest tests/api/test_config.py -v --tb=short
    pytest tests/api/test_config.py::TestConfigure::test_configure_basic -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the module under test
from chuk_llm.api import config
from chuk_llm.api.config import (
    configure,
    get_config,
    reset_config,
    get_client_for_config,
    get_current_config
)


class TestDefaultConfiguration:
    """Test suite for default configuration state."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def test_default_config_values(self):
        """Test that default configuration has expected values."""
        config_dict = get_config()
        
        expected_defaults = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "system_prompt": None,
            "temperature": None,
            "max_tokens": None,
            "api_key": None,
            "api_base": None,
        }
        
        assert config_dict == expected_defaults

    def test_default_cached_client_state(self):
        """Test that cached client is None by default."""
        assert config._cached_client is None
        assert config._cached_config_hash is None

    def test_get_config_returns_copy(self):
        """Test that get_config returns a copy, not the original dict."""
        config_dict = get_config()
        config_dict["provider"] = "modified"
        
        # Original config should be unchanged
        original_config = get_config()
        assert original_config["provider"] == "openai"
        assert original_config["provider"] != "modified"


class TestConfigure:
    """Test suite for configure function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def test_configure_basic(self):
        """Test basic configuration update."""
        configure(provider="anthropic", model="claude-3-sonnet")
        
        config_dict = get_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["model"] == "claude-3-sonnet"
        # Other values should remain default
        assert config_dict["system_prompt"] is None
        assert config_dict["temperature"] is None

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
        
        config_dict = get_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["model"] == "claude-3-opus"
        assert config_dict["system_prompt"] == "You are a helpful assistant"
        assert config_dict["temperature"] == 0.8
        assert config_dict["max_tokens"] == 2000
        assert config_dict["api_key"] == "test-key-123"
        assert config_dict["api_base"] == "https://api.test.com"

    def test_configure_none_values_ignored(self):
        """Test that None values are ignored during configuration."""
        # Set initial values
        configure(provider="openai", model="gpt-4", temperature=0.5)
        
        # Try to set some values to None - they should be ignored
        configure(provider="anthropic", model=None, temperature=None)
        
        config_dict = get_config()
        assert config_dict["provider"] == "anthropic"  # Should be updated
        assert config_dict["model"] == "gpt-4"  # Should remain unchanged
        assert config_dict["temperature"] == 0.5  # Should remain unchanged

    def test_configure_with_kwargs(self):
        """Test configuration with additional kwargs."""
        configure(
            provider="openai",
            custom_param="custom_value",
            another_param=42
        )
        
        config_dict = get_config()
        assert config_dict["provider"] == "openai"
        assert config_dict["custom_param"] == "custom_value"
        assert config_dict["another_param"] == 42

    def test_configure_invalidates_cache(self):
        """Test that configure invalidates cached client."""
        # Mock a cached client
        mock_client = Mock()
        config._cached_client = mock_client
        config._cached_config_hash = ("openai", "gpt-4", None, None)
        
        # Configure should invalidate cache
        configure(provider="anthropic")
        
        assert config._cached_client is None
        assert config._cached_config_hash is None

    def test_configure_multiple_calls(self):
        """Test multiple configure calls accumulate changes."""
        configure(provider="openai", model="gpt-4")
        configure(temperature=0.7, max_tokens=1000)
        configure(api_key="test-key")
        
        config_dict = get_config()
        assert config_dict["provider"] == "openai"
        assert config_dict["model"] == "gpt-4"
        assert config_dict["temperature"] == 0.7
        assert config_dict["max_tokens"] == 1000
        assert config_dict["api_key"] == "test-key"

    def test_configure_overwrites_previous_values(self):
        """Test that configure overwrites previously set values."""
        configure(provider="openai", temperature=0.5)
        configure(provider="anthropic", temperature=0.9)
        
        config_dict = get_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["temperature"] == 0.9


class TestResetConfig:
    """Test suite for reset_config function."""

    def test_reset_config_restores_defaults(self):
        """Test that reset_config restores default values."""
        # Modify configuration
        configure(
            provider="anthropic",
            model="claude-3-opus",
            temperature=0.8,
            api_key="test-key"
        )
        
        # Reset
        reset_config()
        
        # Should be back to defaults
        config_dict = get_config()
        expected_defaults = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "system_prompt": None,
            "temperature": None,
            "max_tokens": None,
            "api_key": None,
            "api_base": None,
        }
        assert config_dict == expected_defaults

    def test_reset_config_clears_cache(self):
        """Test that reset_config clears cached client."""
        # Set up cached client
        mock_client = Mock()
        config._cached_client = mock_client
        config._cached_config_hash = ("test", "test", None, None)
        
        reset_config()
        
        assert config._cached_client is None
        assert config._cached_config_hash is None

    def test_reset_config_removes_custom_kwargs(self):
        """Test that reset_config removes custom parameters added via kwargs."""
        configure(provider="openai", custom_param="value")
        
        config_dict = get_config()
        assert "custom_param" in config_dict
        
        reset_config()
        
        config_dict = get_config()
        assert "custom_param" not in config_dict


class TestGetCurrentConfig:
    """Test suite for get_current_config function."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def test_get_current_config_same_as_get_config(self):
        """Test that get_current_config returns same as get_config."""
        configure(provider="anthropic", temperature=0.7)
        
        current_config = get_current_config()
        public_config = get_config()
        
        assert current_config == public_config

    def test_get_current_config_reflects_changes(self):
        """Test that get_current_config reflects configuration changes."""
        original_config = get_current_config()
        assert original_config["provider"] == "openai"
        
        configure(provider="anthropic")
        
        updated_config = get_current_config()
        assert updated_config["provider"] == "anthropic"


class TestGetClientForConfig:
    """Test suite for get_client_for_config function."""

    def setup_method(self):
        """Reset configuration and cache before each test."""
        reset_config()
        config._cached_client = None
        config._cached_config_hash = None

    def test_get_client_for_config_creates_new_client(self):
        """Test that get_client_for_config creates a new client."""
        test_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "api_base": None
        }
        
        mock_client = Mock()
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            result = get_client_for_config(test_config)
            
            assert result == mock_client
            mock_get_client.assert_called_once_with(
                provider="openai",
                model="gpt-4",
                api_key="test-key",
                api_base=None
            )

    def test_get_client_for_config_caches_client(self):
        """Test that get_client_for_config caches the created client."""
        test_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "api_base": None
        }
        
        mock_client = Mock()
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            result = get_client_for_config(test_config)
            
            # Check that client is cached
            assert config._cached_client == mock_client
            assert config._cached_config_hash == ("openai", "gpt-4", "test-key", None)

    def test_get_client_for_config_returns_cached_client(self):
        """Test that get_client_for_config returns cached client when config matches."""
        test_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "api_base": None
        }
        
        mock_client = Mock()
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # First call - creates client
            result1 = get_client_for_config(test_config)
            
            # Second call with same config - should return cached client
            result2 = get_client_for_config(test_config)
            
            assert result1 == result2 == mock_client
            # get_llm_client should only be called once
            mock_get_client.assert_called_once()

    def test_get_client_for_config_creates_new_client_when_config_changes(self):
        """Test that get_client_for_config creates new client when config changes."""
        config1 = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key-1",
            "api_base": None
        }
        
        config2 = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key-2",  # Different API key
            "api_base": None
        }
        
        mock_client1 = Mock()
        mock_client2 = Mock()
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.side_effect = [mock_client1, mock_client2]
            
            # First call
            result1 = get_client_for_config(config1)
            
            # Second call with different config
            result2 = get_client_for_config(config2)
            
            assert result1 == mock_client1
            assert result2 == mock_client2
            assert result1 != result2
            # get_llm_client should be called twice
            assert mock_get_client.call_count == 2

    def test_get_client_for_config_handles_missing_config_keys(self):
        """Test that get_client_for_config handles missing config keys gracefully."""
        minimal_config = {
            "provider": "openai",
            "model": "gpt-4"
            # Missing api_key and api_base
        }
        
        mock_client = Mock()
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            result = get_client_for_config(minimal_config)
            
            assert result == mock_client
            mock_get_client.assert_called_once_with(
                provider="openai",
                model="gpt-4",
                api_key=None,  # Should default to None
                api_base=None   # Should default to None
            )

    def test_get_client_for_config_cache_key_generation(self):
        """Test that cache key is generated correctly from config."""
        test_config = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "api_key": "sk-test123",
            "api_base": "https://api.anthropic.com",
            "temperature": 0.7,  # This should not affect cache key
            "max_tokens": 1000   # This should not affect cache key
        }
        
        mock_client = Mock()
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            get_client_for_config(test_config)
            
            # Check that cache key only includes relevant fields
            expected_cache_key = ("anthropic", "claude-3-sonnet", "sk-test123", "https://api.anthropic.com")
            assert config._cached_config_hash == expected_cache_key


class TestConfigurationIntegration:
    """Integration tests for configuration functionality."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def test_full_configuration_workflow(self):
        """Test a complete configuration workflow."""
        # Start with defaults
        assert get_config()["provider"] == "openai"
        assert get_config()["model"] == "gpt-4o-mini"
        
        # Configure for Anthropic
        configure(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.8,
            api_key="sk-test123"
        )
        
        config_dict = get_config()
        assert config_dict["provider"] == "anthropic"
        assert config_dict["model"] == "claude-3-sonnet"
        assert config_dict["temperature"] == 0.8
        assert config_dict["api_key"] == "sk-test123"
        
        # Test client creation with this config
        mock_client = Mock()
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            client = get_client_for_config(config_dict)
            assert client == mock_client
            
            mock_get_client.assert_called_once_with(
                provider="anthropic",
                model="claude-3-sonnet",
                api_key="sk-test123",
                api_base=None
            )
        
        # Reset and verify defaults restored
        reset_config()
        
        reset_config_dict = get_config()
        assert reset_config_dict["provider"] == "openai"
        assert reset_config_dict["model"] == "gpt-4o-mini"
        assert reset_config_dict["temperature"] is None
        assert reset_config_dict["api_key"] is None

    def test_configuration_persistence_across_calls(self):
        """Test that configuration persists across multiple function calls."""
        configure(provider="groq", model="llama-3.3-70b-versatile")
        
        # Multiple calls to get_config should return same values
        config1 = get_config()
        config2 = get_config()
        config3 = get_current_config()
        
        assert config1["provider"] == "groq"
        assert config2["provider"] == "groq"
        assert config3["provider"] == "groq"
        
        assert config1 == config2 == config3

    def test_client_caching_across_same_config(self):
        """Test that client caching works correctly across multiple requests."""
        test_config = {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "api_base": None
        }
        
        mock_client = Mock()
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Multiple calls with same config
            client1 = get_client_for_config(test_config)
            client2 = get_client_for_config(test_config)
            client3 = get_client_for_config(test_config)
            
            # All should return same cached client
            assert client1 == client2 == client3 == mock_client
            
            # get_llm_client should only be called once
            mock_get_client.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()

    def test_configure_with_empty_kwargs(self):
        """Test configure with empty kwargs."""
        original_config = get_config()
        
        configure()  # No arguments
        
        # Config should remain unchanged
        assert get_config() == original_config

    def test_configure_with_zero_values(self):
        """Test that zero values (not None) are properly set."""
        configure(temperature=0, max_tokens=0)
        
        config_dict = get_config()
        assert config_dict["temperature"] == 0
        assert config_dict["max_tokens"] == 0

    def test_configure_with_boolean_values(self):
        """Test configure with boolean values in kwargs."""
        configure(debug=True, streaming=False)
        
        config_dict = get_config()
        assert config_dict["debug"] is True
        assert config_dict["streaming"] is False

    def test_get_client_for_config_with_exception(self):
        """Test get_client_for_config when client creation fails."""
        test_config = {
            "provider": "invalid_provider",
            "model": "invalid_model",
            "api_key": None,
            "api_base": None
        }
        
        with patch('chuk_llm.llm.llm_client.get_llm_client') as mock_get_client:
            mock_get_client.side_effect = ValueError("Invalid provider")
            
            with pytest.raises(ValueError, match="Invalid provider"):
                get_client_for_config(test_config)
            
            # Cache should not be set when creation fails
            assert config._cached_client is None
            assert config._cached_config_hash is None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])