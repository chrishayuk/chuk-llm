# tests/test_provider_config.py
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from chuk_llm.provider_config import ProviderConfig, DEFAULTS

class TestProviderConfig:
    """Tests for the ProviderConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = ProviderConfig()
        # Check providers dictionary loaded from defaults
        assert "openai" in config.providers
        assert config.providers["openai"]["default_model"] == DEFAULTS["openai"]["default_model"]
        assert "__global__" in config.providers
    
    def test_init_with_custom_config(self):
        """Test initialization with custom config dictionary."""
        custom_config = {
            "openai": {"default_model": "custom-model"},
            "custom_provider": {"client": "custom.client:Client"}
        }
        
        config = ProviderConfig(custom_config)
        
        # Check custom values were applied
        assert config.providers["openai"]["default_model"] == "custom-model"
        assert "custom_provider" in config.providers
        assert config.providers["custom_provider"]["client"] == "custom.client:Client"
    
    def test_get_provider_config_existing(self):
        """Test get_provider_config method with existing provider."""
        config = ProviderConfig()
        
        # Get config for existing provider
        openai_config = config.get_provider_config("openai")
        assert openai_config["client"] == DEFAULTS["openai"]["client"]
        assert openai_config["default_model"] == DEFAULTS["openai"]["default_model"]
    
    def test_get_provider_config_new(self):
        """Test get_provider_config method for new provider."""
        config = ProviderConfig()
        
        # For a new provider, should create an empty section
        new_config = config.get_provider_config("new_provider")
        assert isinstance(new_config, dict)
        
        # This should also add the new section to providers
        assert "new_provider" in config.providers
    
    def test_get_provider_config_with_env_var(self):
        """Test get_provider_config with environment variable for API key."""
        with patch.dict(os.environ, {"TEST_API_KEY": "key-from-env"}):
            config = ProviderConfig({
                "test_provider": {
                    "api_key_env": "TEST_API_KEY",
                    "api_key": None
                }
            })
            
            result = config.get_provider_config("test_provider")
            assert result["api_key"] == "key-from-env"
    
    def test_update_provider_config_existing(self):
        """Test update_provider_config method for existing provider."""
        config = ProviderConfig()
        
        # Update existing provider
        config.update_provider_config("openai", {"default_model": "updated-model"})
        assert config.providers["openai"]["default_model"] == "updated-model"
        
        # Original default should remain unchanged
        assert DEFAULTS["openai"]["default_model"] != "updated-model"
    
    def test_update_provider_config_new(self):
        """Test update_provider_config method for new provider."""
        config = ProviderConfig()
        
        # Update non-existing provider (should create it)
        config.update_provider_config("new_provider", {"client": "new:Client"})
        assert "new_provider" in config.providers
        assert config.providers["new_provider"]["client"] == "new:Client"
    
    def test_ensure_section_existing(self):
        """Test _ensure_section method with existing section."""
        config = ProviderConfig()
        
        # Should not change existing section
        original_openai = config.providers["openai"].copy()
        config._ensure_section("openai")
        assert config.providers["openai"] == original_openai
    
    def test_ensure_section_new(self):
        """Test _ensure_section method with new section."""
        config = ProviderConfig()
        
        # Should create empty dict for new section
        config._ensure_section("non_existent")
        assert "non_existent" in config.providers
        assert config.providers["non_existent"] == {}
    
    def test_merge_env_key_with_env_var(self):
        """Test _merge_env_key method with environment variable."""
        with patch.dict(os.environ, {"TEST_API_KEY": "key-from-env"}):
            config = ProviderConfig()
            cfg = {"api_key": None, "api_key_env": "TEST_API_KEY"}
            config._merge_env_key(cfg)
            assert cfg["api_key"] == "key-from-env"
    
    def test_merge_env_key_with_existing_key(self):
        """Test _merge_env_key method with existing key."""
        config = ProviderConfig()
        cfg = {"api_key": "existing-key", "api_key_env": "NON_EXISTENT_ENV"}
        config._merge_env_key(cfg)
        # Should not override existing key
        assert cfg["api_key"] == "existing-key"
    
    def test_merge_env_key_no_env_var(self):
        """Test _merge_env_key method with non-existent environment variable."""
        config = ProviderConfig()
        cfg = {"api_key": None, "api_key_env": "NON_EXISTENT_ENV"}
        config._merge_env_key(cfg)
        # api_key should still be None
        assert cfg["api_key"] is None
    
    def test_merge_env_key_no_env_key(self):
        """Test _merge_env_key method with no api_key_env."""
        config = ProviderConfig()
        cfg = {"api_key": None}  # No api_key_env
        config._merge_env_key(cfg)
        # Should not change anything
        assert cfg["api_key"] is None
        assert "api_key_env" not in cfg
    
    def test_global_section(self):
        """Test _glob property."""
        config = ProviderConfig()
        # Access should not fail and return the global section
        assert config._glob == config.providers["__global__"]
    
    def test_global_section_missing(self):
        """Test _glob property when global section is missing."""
        config = ProviderConfig({})  # Empty config
        # Should create global section if missing
        del config.providers["__global__"]
        assert "__global__" not in config.providers
        
        # Accessing _glob should create it
        assert config._glob == {}
        assert "__global__" in config.providers
    
    def test_get_active_provider(self):
        """Test get_active_provider method."""
        config = ProviderConfig()
        assert config.get_active_provider() == DEFAULTS["__global__"]["active_provider"]
    
    def test_get_active_provider_custom(self):
        """Test get_active_provider with custom provider."""
        config = ProviderConfig({
            "__global__": {"active_provider": "custom_provider"}
        })
        assert config.get_active_provider() == "custom_provider"
    
    def test_set_active_provider(self):
        """Test set_active_provider method."""
        config = ProviderConfig()
        config.set_active_provider("anthropic")
        assert config.get_active_provider() == "anthropic"
        assert config.providers["__global__"]["active_provider"] == "anthropic"
    
    def test_get_active_model(self):
        """Test get_active_model method."""
        config = ProviderConfig()
        assert config.get_active_model() == DEFAULTS["__global__"]["active_model"]
    
    def test_get_active_model_custom(self):
        """Test get_active_model with custom model."""
        config = ProviderConfig({
            "__global__": {"active_model": "custom-model"}
        })
        assert config.get_active_model() == "custom-model"
    
    def test_set_active_model(self):
        """Test set_active_model method."""
        config = ProviderConfig()
        config.set_active_model("custom-model")
        assert config.get_active_model() == "custom-model"
        assert config.providers["__global__"]["active_model"] == "custom-model"
    
    def test_get_api_key(self):
        """Test get_api_key method."""
        config = ProviderConfig({
            "test_provider": {"api_key": "test-key"}
        })
        assert config.get_api_key("test_provider") == "test-key"
    
    def test_get_api_key_none(self):
        """Test get_api_key method when key is None."""
        # Create a config with a custom provider that has api_key=None
        config = ProviderConfig({
            "test_provider": {"api_key": None}
        })
        # Test with our custom provider instead of "openai"
        assert config.get_api_key("test_provider") is None
    
    def test_get_api_key_missing(self):
        """Test get_api_key method with missing key."""
        config = ProviderConfig()
        # New provider with no configs
        config.providers["new_provider"] = {}
        # Should return None for missing key
        assert config.get_api_key("new_provider") is None
    
    def test_get_api_base(self):
        """Test get_api_base method."""
        config = ProviderConfig({
            "test_provider": {"api_base": "https://api.test.com"}
        })
        assert config.get_api_base("test_provider") == "https://api.test.com"
    
    def test_get_api_base_none(self):
        """Test get_api_base method when base is None."""
        config = ProviderConfig()
        # Most providers have api_base set to None in DEFAULTS
        assert config.get_api_base("openai") is None
    
    def test_get_api_base_missing(self):
        """Test get_api_base method with missing base."""
        config = ProviderConfig()
        # New provider with no configs
        config.providers["new_provider"] = {}
        # Should return None for missing base
        assert config.get_api_base("new_provider") is None
    
    def test_get_default_model(self):
        """Test get_default_model method."""
        config = ProviderConfig()
        assert config.get_default_model("openai") == DEFAULTS["openai"]["default_model"]
    
    def test_get_default_model_custom(self):
        """Test get_default_model with custom model."""
        config = ProviderConfig({
            "test_provider": {"default_model": "custom-model"}
        })
        assert config.get_default_model("test_provider") == "custom-model"
    
    def test_get_default_model_missing(self):
        """Test get_default_model with missing model."""
        config = ProviderConfig()
        # New provider with no configs
        config.providers["new_provider"] = {}
        # Should return empty string for missing model
        assert config.get_default_model("new_provider") == ""
    
    def test_multiple_operations(self):
        """Test multiple operations on the same config object."""
        config = ProviderConfig()
        
        # Change active provider
        config.set_active_provider("anthropic")
        assert config.get_active_provider() == "anthropic"
        
        # Change active model
        config.set_active_model("custom-model")
        assert config.get_active_model() == "custom-model"
        
        # Update provider config
        config.update_provider_config("openai", {"api_key": "new-key"})
        assert config.get_api_key("openai") == "new-key"
        
        # Add new provider
        config.update_provider_config("custom", {"default_model": "model-x"})
        assert config.get_provider_config("custom")["default_model"] == "model-x"
        
        # All changes should persist
        assert config.get_active_provider() == "anthropic"
        assert config.get_active_model() == "custom-model"
        assert config.get_api_key("openai") == "new-key"