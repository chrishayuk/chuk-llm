"""
Unit tests for dynamic provider registration API.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, call
from chuk_llm.api.dynamic_providers import (
    register_provider,
    register_openai_compatible,
    update_provider,
    unregister_provider,
    list_dynamic_providers,
    provider_exists,
    get_provider_config
)
from chuk_llm.configuration.unified_config import get_config, reset_config
from chuk_llm.configuration.models import ProviderConfig


class TestDynamicProviderRegistration:
    """Test dynamic provider registration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset configuration for each test
        reset_config()
        self.config = get_config()
        
        # Clean up any leftover dynamic providers
        for provider in list_dynamic_providers():
            unregister_provider(provider)
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up dynamic providers
        for provider in list_dynamic_providers():
            unregister_provider(provider)
        
        # Reset configuration
        reset_config()
    
    def test_register_basic_provider(self):
        """Test basic provider registration."""
        # Register a provider
        provider = register_provider(
            name="test_provider",
            api_base="https://test.api.com/v1",
            api_key="test-key",
            models=["model1", "model2"],
            default_model="model1",
            client_class="chuk_llm.llm.providers.openai_client.OpenAILLMClient"
        )
        
        # Check provider was registered
        assert provider is not None
        assert provider.name == "test_provider"
        assert provider.api_base == "https://test.api.com/v1"
        assert provider.models == ["model1", "model2"]
        assert provider.default_model == "model1"
        
        # Check it exists
        assert provider_exists("test_provider")
        
        # Check it's in the dynamic list
        assert "test_provider" in list_dynamic_providers()
    
    def test_register_openai_compatible(self):
        """Test OpenAI-compatible provider registration."""
        # Register without api_base (using env var)
        os.environ["TEST_OPENAI_ENDPOINT"] = "https://test.openai.com/v1"
        
        provider = register_openai_compatible(
            name="test_openai",
            api_base_env="TEST_OPENAI_ENDPOINT",
            api_key="test-key",
            models=["gpt-3.5-turbo", "gpt-4"]
        )
        
        # Check provider configuration
        assert provider is not None
        assert provider.name == "test_openai"
        assert provider.client_class == "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
        assert "text" in provider.extra.get("features", [])
        assert "streaming" in provider.extra.get("features", [])
        
        # Check API base resolution
        resolved_base = self.config.get_api_base("test_openai")
        assert resolved_base == "https://test.openai.com/v1"
        
        # Clean up
        del os.environ["TEST_OPENAI_ENDPOINT"]
    
    def test_register_with_inheritance(self):
        """Test provider registration with inheritance."""
        # Register base provider
        base = register_provider(
            name="base_provider",
            api_base="https://base.api.com/v1",
            client_class="chuk_llm.llm.providers.openai_client.OpenAILLMClient",
            models=["base-model"],
            features=["text", "streaming"]
        )
        
        # Register inheriting provider
        inherited = register_provider(
            name="inherited_provider",
            inherits_from="base_provider",
            api_base="https://inherited.api.com/v1",
            models=["inherited-model"]
        )
        
        # Check inheritance
        assert inherited.client_class == base.client_class
        assert inherited.models == ["inherited-model"]  # Should not inherit models
        assert inherited.api_base == "https://inherited.api.com/v1"  # Should override
    
    def test_update_provider(self):
        """Test updating an existing provider."""
        # Register initial provider
        provider = register_provider(
            name="update_test",
            api_base="https://initial.com/v1",
            models=["model1"],
            default_model="model1"
        )
        
        # Update provider
        updated = update_provider(
            "update_test",
            api_base="https://updated.com/v1",
            models=["model1", "model2", "model3"],
            default_model="model2",
            new_feature="test_value"
        )
        
        # Check updates
        assert updated.api_base == "https://updated.com/v1"
        assert updated.models == ["model1", "model2", "model3"]
        assert updated.default_model == "model2"
        assert updated.extra.get("new_feature") == "test_value"
    
    def test_update_nonexistent_provider(self):
        """Test updating a provider that doesn't exist."""
        with pytest.raises(ValueError, match="Provider 'nonexistent' not found"):
            update_provider("nonexistent", api_base="https://test.com/v1")
    
    def test_unregister_provider(self):
        """Test unregistering a provider."""
        # Register provider
        provider = register_provider(
            name="unregister_test",
            api_base="https://test.com/v1",
            models=["model1"]
        )
        
        # Check it exists
        assert provider_exists("unregister_test")
        
        # Unregister
        success = unregister_provider("unregister_test")
        assert success is True
        
        # Check it's gone
        assert not provider_exists("unregister_test")
        assert "unregister_test" not in list_dynamic_providers()
    
    def test_cannot_unregister_builtin(self):
        """Test that built-in providers cannot be unregistered."""
        # Try to unregister OpenAI (built-in)
        success = unregister_provider("openai")
        assert success is False
        
        # OpenAI should still exist
        assert provider_exists("openai")
    
    def test_list_dynamic_providers(self):
        """Test listing dynamic providers."""
        # Initially should be empty
        initial = list_dynamic_providers()
        initial_count = len(initial)
        
        # Register multiple providers
        register_provider(name="dynamic1", models=["m1"])
        register_provider(name="dynamic2", models=["m2"])
        register_provider(name="dynamic3", models=["m3"])
        
        # Check list
        providers = list_dynamic_providers()
        assert len(providers) == initial_count + 3
        assert "dynamic1" in providers
        assert "dynamic2" in providers
        assert "dynamic3" in providers
        
        # Built-in providers should not be in the list
        assert "openai" not in providers
        assert "anthropic" not in providers
    
    def test_get_provider_config(self):
        """Test retrieving provider configuration."""
        # Register provider
        register_provider(
            name="config_test",
            api_base="https://test.com/v1",
            models=["model1", "model2"],
            default_model="model1"
        )
        
        # Get config
        config = get_provider_config("config_test")
        
        assert config is not None
        assert config.name == "config_test"
        assert config.api_base == "https://test.com/v1"
        assert config.models == ["model1", "model2"]
        assert config.default_model == "model1"
    
    def test_provider_exists(self):
        """Test checking provider existence."""
        # Dynamic provider
        register_provider(name="exists_test", models=["m1"])
        assert provider_exists("exists_test") is True
        
        # Built-in provider
        assert provider_exists("openai") is True
        
        # Non-existent provider
        assert provider_exists("nonexistent") is False
    
    def test_api_key_storage(self):
        """Test API key storage in memory."""
        # Register with API key
        provider = register_provider(
            name="key_test",
            api_key="secret-key-123",
            models=["model1"]
        )
        
        # Check API key is stored in extra
        assert provider.extra.get("_runtime_api_key") == "secret-key-123"
        
        # Check it can be retrieved
        api_key = self.config.get_api_key("key_test")
        assert api_key == "secret-key-123"
    
    def test_api_key_env_variable(self):
        """Test API key from environment variable."""
        # Set environment variable
        os.environ["TEST_API_KEY"] = "env-key-456"
        
        # Register with env variable
        provider = register_provider(
            name="env_key_test",
            api_key_env="TEST_API_KEY",
            models=["model1"]
        )
        
        # Check API key is retrieved from env
        api_key = self.config.get_api_key("env_key_test")
        assert api_key == "env-key-456"
        
        # Clean up
        del os.environ["TEST_API_KEY"]
    
    def test_features_configuration(self):
        """Test features configuration."""
        # Register with features
        provider = register_provider(
            name="features_test",
            features=["text", "streaming", "tools", "vision"],
            models=["model1"]
        )
        
        # Check features are stored
        assert provider.extra.get("features") == ["text", "streaming", "tools", "vision"]
    
    def test_extra_kwargs_storage(self):
        """Test extra kwargs are stored properly."""
        # Register with extra kwargs
        provider = register_provider(
            name="extra_test",
            models=["model1"],
            custom_param1="value1",
            custom_param2={"nested": "value"},
            custom_param3=[1, 2, 3]
        )
        
        # Check extra kwargs are stored
        assert provider.extra.get("custom_param1") == "value1"
        assert provider.extra.get("custom_param2") == {"nested": "value"}
        assert provider.extra.get("custom_param3") == [1, 2, 3]
    
    def test_duplicate_registration_overwrites(self):
        """Test that registering with same name overwrites."""
        # Register initial
        provider1 = register_provider(
            name="duplicate_test",
            api_base="https://first.com/v1",
            models=["model1"]
        )
        
        # Register again with same name
        provider2 = register_provider(
            name="duplicate_test",
            api_base="https://second.com/v1",
            models=["model2", "model3"]
        )
        
        # Check it was overwritten
        config = get_provider_config("duplicate_test")
        assert config.api_base == "https://second.com/v1"
        assert config.models == ["model2", "model3"]
        
        # Still only one in dynamic list
        dynamics = list_dynamic_providers()
        assert dynamics.count("duplicate_test") == 1


class TestOpenAICompatibleRegistration:
    """Test OpenAI-compatible provider registration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_config()
        self.config = get_config()
        
        # Clean up
        for provider in list_dynamic_providers():
            unregister_provider(provider)
    
    def teardown_method(self):
        """Clean up after tests."""
        for provider in list_dynamic_providers():
            unregister_provider(provider)
        reset_config()
    
    def test_register_with_api_base(self):
        """Test registration with explicit api_base."""
        provider = register_openai_compatible(
            name="openai_test1",
            api_base="https://api.test.com/v1",
            api_key="test-key",
            models=["gpt-3.5-turbo", "gpt-4"]
        )
        
        assert provider.name == "openai_test1"
        assert provider.api_base == "https://api.test.com/v1"
        assert provider.client_class == "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
        assert "gpt-3.5-turbo" in provider.models
        assert "gpt-4" in provider.models
    
    def test_register_with_api_base_env(self):
        """Test registration with api_base_env."""
        os.environ["CUSTOM_ENDPOINT"] = "https://custom.api.com/v1"
        
        provider = register_openai_compatible(
            name="openai_test2",
            api_base_env="CUSTOM_ENDPOINT",
            models=["model1"]
        )
        
        # Check configuration
        assert provider.name == "openai_test2"
        assert provider.extra.get("api_base_env") == "CUSTOM_ENDPOINT"
        
        # Check resolution
        resolved = self.config.get_api_base("openai_test2")
        assert resolved == "https://custom.api.com/v1"
        
        del os.environ["CUSTOM_ENDPOINT"]
    
    def test_register_with_both_api_base_and_env(self):
        """Test registration with both api_base and api_base_env."""
        os.environ["OVERRIDE_ENDPOINT"] = "https://override.api.com/v1"
        
        provider = register_openai_compatible(
            name="openai_test3",
            api_base="https://default.api.com/v1",
            api_base_env="OVERRIDE_ENDPOINT",
            models=["model1"]
        )
        
        # Environment should take priority
        resolved = self.config.get_api_base("openai_test3")
        assert resolved == "https://override.api.com/v1"
        
        # Without env, should use configured
        del os.environ["OVERRIDE_ENDPOINT"]
        resolved = self.config.get_api_base("openai_test3")
        assert resolved == "https://default.api.com/v1"
    
    def test_default_models(self):
        """Test default models when none specified."""
        provider = register_openai_compatible(
            name="openai_test4",
            api_base="https://test.api.com/v1"
        )
        
        # Should have default model
        assert provider.models == ["gpt-3.5-turbo"]
        assert provider.default_model == "gpt-3.5-turbo" or provider.extra.get("default_model") == "gpt-3.5-turbo"
    
    def test_features_added(self):
        """Test that OpenAI features are added."""
        provider = register_openai_compatible(
            name="openai_test5",
            api_base="https://test.api.com/v1"
        )
        
        # Should have OpenAI features
        features = provider.extra.get("features", [])
        assert "text" in features
        assert "streaming" in features
        assert "system_messages" in features
        assert "tools" in features
        assert "json_mode" in features
    
    def test_extra_kwargs_passed_through(self):
        """Test that extra kwargs are passed through."""
        provider = register_openai_compatible(
            name="openai_test6",
            api_base="https://test.api.com/v1",
            custom_header="X-Custom-Value",
            rate_limit=1000,
            timeout=30
        )
        
        # Check extra kwargs
        assert provider.extra.get("custom_header") == "X-Custom-Value"
        assert provider.extra.get("rate_limit") == 1000
        assert provider.extra.get("timeout") == 30


class TestDynamicProviderIntegration:
    """Integration tests for dynamic providers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_config()
        self.config = get_config()
        
        for provider in list_dynamic_providers():
            unregister_provider(provider)
    
    def teardown_method(self):
        """Clean up after tests."""
        for provider in list_dynamic_providers():
            unregister_provider(provider)
        reset_config()
    
    def test_dynamic_provider_with_client(self):
        """Test that dynamic providers work with client creation."""
        from chuk_llm.llm.client import get_client
        
        # Register dynamic provider
        register_openai_compatible(
            name="dynamic_client_test",
            api_base="https://dynamic.api.com/v1",
            api_key="dynamic-key",
            models=["dynamic-model"],
            default_model="dynamic-model"
        )
        
        # Mock client creation
        mock_client_instance = MagicMock()
        mock_client_class = MagicMock(return_value=mock_client_instance)
        
        with patch('chuk_llm.llm.client._import_string', return_value=mock_client_class):
            # Create client
            client = get_client("dynamic_client_test")
            
            # Verify client was created with correct config
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert call_args.get("api_base") == "https://dynamic.api.com/v1"
            assert call_args.get("api_key") == "dynamic-key"
            assert call_args.get("model") == "dynamic-model"
    
    def test_persistence_across_operations(self):
        """Test that dynamic providers persist across operations."""
        # Register provider
        register_provider(
            name="persist_test",
            api_base="https://persist.api.com/v1",
            models=["model1"]
        )
        
        # Update it
        update_provider(
            "persist_test",
            models=["model1", "model2"]
        )
        
        # Check it still exists and has updates
        assert provider_exists("persist_test")
        config = get_provider_config("persist_test")
        assert config.models == ["model1", "model2"]
        
        # List should still include it
        assert "persist_test" in list_dynamic_providers()
    
    def test_multiple_providers_isolation(self):
        """Test that multiple dynamic providers don't interfere."""
        # Register multiple providers
        p1 = register_provider(
            name="multi_test1",
            api_base="https://api1.com/v1",
            api_key="key1",
            models=["model1"]
        )
        
        p2 = register_provider(
            name="multi_test2",
            api_base="https://api2.com/v1",
            api_key="key2",
            models=["model2"]
        )
        
        p3 = register_provider(
            name="multi_test3",
            api_base="https://api3.com/v1",
            api_key="key3",
            models=["model3"]
        )
        
        # Check each has correct configuration
        c1 = get_provider_config("multi_test1")
        assert c1.api_base == "https://api1.com/v1"
        assert self.config.get_api_key("multi_test1") == "key1"
        
        c2 = get_provider_config("multi_test2")
        assert c2.api_base == "https://api2.com/v1"
        assert self.config.get_api_key("multi_test2") == "key2"
        
        c3 = get_provider_config("multi_test3")
        assert c3.api_base == "https://api3.com/v1"
        assert self.config.get_api_key("multi_test3") == "key3"
        
        # Unregister one shouldn't affect others
        unregister_provider("multi_test2")
        
        assert provider_exists("multi_test1")
        assert not provider_exists("multi_test2")
        assert provider_exists("multi_test3")