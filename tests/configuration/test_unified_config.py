# test_unified_config_clean.py
"""
Clean, focused tests for the simplified unified configuration system.

Tests the NEW simplified architecture:
- Pydantic validation (automatic)
- Simple YAML loading (no complex inheritance)
- Registry integration (built-in)
- Clean, maintainable code
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from chuk_llm.configuration.models import (
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


class TestUnifiedConfigManagerSimplified:
    """Tests for simplified UnifiedConfigManager"""

    def test_init_loads_package_default(self):
        """Test that initialization loads package default config"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Should load package default, not be empty
            assert isinstance(manager.providers, dict)
            # Package default should have some providers
            # (This is the NEW behavior - it loads chuk_llm.yaml by default)

    def test_init_with_explicit_config_path(self, tmp_path):
        """Test initialization with explicit config path"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "test_provider": {
                "name": "test_provider",
                "client_class": "TestClient",
                "default_model": "test-model",
                "features": ["text"],
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            assert "test_provider" in manager.providers
            assert manager.providers["test_provider"].name == "test_provider"

    def test_process_config_data_with_pydantic(self):
        """Test that _process_config_data uses Pydantic validation"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()  # Start fresh

            config_data = {
                "openai": {
                    "name": "openai",
                    "client_class": "OpenAIClient",
                    "api_key_env": "OPENAI_API_KEY",
                    "api_base": "https://api.openai.com/v1",
                    "default_model": "gpt-4",
                    "features": ["text", "streaming", "tools"],
                }
            }

            # This should use Pydantic validation
            manager._process_config_data(config_data)

            assert "openai" in manager.providers
            provider = manager.providers["openai"]
            assert isinstance(provider, ProviderConfig)
            assert provider.name == "openai"
            assert provider.api_base == "https://api.openai.com/v1"  # Pydantic validated
            assert Feature.TEXT in provider.features

    def test_process_config_data_pydantic_rejects_invalid(self):
        """Test that Pydantic rejects invalid data"""
        from pydantic import ValidationError

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            config_data = {
                "bad_provider": {
                    "name": "bad_provider",
                    "api_base": "not-a-url",  # Invalid - no http://
                }
            }

            # Should raise ValidationError during processing
            with pytest.raises((ValidationError, ValueError)):
                manager._process_config_data(config_data)

    def test_parse_features_string(self):
        """Test parsing features from string"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            features = manager._parse_features("text")
            assert Feature.TEXT in features

    def test_parse_features_list(self):
        """Test parsing features from list"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            features = manager._parse_features(["text", "streaming", "tools"])
            assert Feature.TEXT in features
            assert Feature.STREAMING in features
            assert Feature.TOOLS in features

    def test_parse_features_empty(self):
        """Test parsing empty features"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            features = manager._parse_features(None)
            assert features == set()

            features = manager._parse_features([])
            assert features == set()

    def test_parse_features_unknown_warning(self):
        """Test that unknown features are logged as warnings"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Should not raise, just log warning
            features = manager._parse_features(["text", "unknown_feature", "streaming"])
            assert Feature.TEXT in features
            assert Feature.STREAMING in features
            # unknown_feature should be skipped

    def test_get_provider_success(self, tmp_path):
        """Test getting an existing provider"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
                "default_model": "gpt-4",
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            provider = manager.get_provider("openai")
            assert provider.name == "openai"

    def test_get_provider_not_found(self):
        """Test getting a non-existent provider"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()

            with pytest.raises(ValueError, match="Unknown provider"):
                manager.get_provider("nonexistent")

    def test_get_all_providers(self, tmp_path):
        """Test getting list of all providers"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "openai": {"name": "openai", "client_class": "OpenAIClient"},
            "anthropic": {"name": "anthropic", "client_class": "AnthropicClient"},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            providers = manager.get_all_providers()
            assert "openai" in providers
            assert "anthropic" in providers

    def test_get_api_key(self):
        """Test getting API key from environment"""
        with patch.dict(os.environ, {"TEST_API_KEY": "secret-key"}):
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager()
                manager.providers["test"] = ProviderConfig(
                    name="test",
                    client_class="TestClient",
                    api_key_env="TEST_API_KEY",
                )

                api_key = manager.get_api_key("test")
                assert api_key == "secret-key"

    def test_get_api_key_none(self):
        """Test getting API key when not set"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers["test"] = ProviderConfig(
                name="test",
                client_class="TestClient",
                api_key_env="NONEXISTENT_KEY",
            )

            api_key = manager.get_api_key("test")
            assert api_key is None

    def test_supports_feature(self):
        """Test checking feature support"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers["test"] = ProviderConfig(
                name="test",
                client_class="TestClient",
                features={Feature.TEXT, Feature.STREAMING},
            )

            assert manager.supports_feature("test", Feature.TEXT) is True
            assert manager.supports_feature("test", Feature.TOOLS) is False

    def test_global_settings(self, tmp_path):
        """Test loading global settings"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "__global__": {
                "active_provider": "openai",
                "default_temperature": 0.7,
            },
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            assert manager.global_settings["active_provider"] == "openai"
            assert manager.global_settings["default_temperature"] == 0.7

    def test_global_aliases(self, tmp_path):
        """Test loading global aliases"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "__global_aliases__": {
                "fast": "gpt-3.5-turbo",
                "smart": "gpt-4",
            },
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            aliases = manager.get_global_aliases()
            assert aliases["fast"] == "gpt-3.5-turbo"
            assert aliases["smart"] == "gpt-4"

    def test_reload(self, tmp_path):
        """Test reloading configuration"""
        config_file = tmp_path / "test_config.yaml"

        # Initial config
        config_data = {
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))
            assert "openai" in manager.providers

            # Update config file
            config_data["anthropic"] = {
                "name": "anthropic",
                "client_class": "AnthropicClient",
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Reload
            manager.reload()

            assert "openai" in manager.providers
            assert "anthropic" in manager.providers


class TestRegistryIntegration:
    """Tests for registry integration (now built into UnifiedConfigManager)"""

    @pytest.mark.asyncio
    async def test_get_registry_models_with_cache(self):
        """Test getting models from registry with caching"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Mock discover_models
            with patch("chuk_llm.api.discovery.discover_models") as mock_discover:
                mock_discover.return_value = [
                    {"name": "gpt-4"},
                    {"name": "gpt-3.5-turbo"},
                ]

                # First call - should hit registry
                models = await manager._get_registry_models("openai")
                assert "gpt-4" in models
                assert "gpt-3.5-turbo" in models
                mock_discover.assert_called_once()

                # Second call - should use cache
                mock_discover.reset_mock()
                models = await manager._get_registry_models("openai")
                assert "gpt-4" in models
                mock_discover.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_registry_models_force_refresh(self):
        """Test forcing registry refresh"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            with patch("chuk_llm.api.discovery.discover_models") as mock_discover:
                mock_discover.return_value = [{"name": "gpt-4"}]

                # First call
                await manager._get_registry_models("openai")
                assert mock_discover.call_count == 1

                # Force refresh - should bypass cache
                await manager._get_registry_models("openai", force_refresh=True)
                assert mock_discover.call_count == 2

    def test_get_discovered_models_sync(self):
        """Test synchronous wrapper for getting discovered models"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Pre-populate cache
            manager._registry_cache["openai"] = {"gpt-4", "gpt-3.5-turbo"}
            manager._registry_cache_time["openai"] = time.time()

            models = manager.get_discovered_models("openai")
            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models

    def test_get_all_available_models(self):
        """Test getting all models (static + discovered)"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers["openai"] = ProviderConfig(
                name="openai",
                client_class="OpenAIClient",
                models=["gpt-4", "gpt-3.5-turbo"],  # Static models
            )

            # Pre-populate discovered models
            manager._registry_cache["openai"] = {"gpt-4-turbo", "gpt-4-vision"}
            manager._registry_cache_time["openai"] = time.time()

            all_models = manager.get_all_available_models("openai")

            # Should have both static and discovered
            assert "gpt-4" in all_models  # Static
            assert "gpt-3.5-turbo" in all_models  # Static
            assert "gpt-4-turbo" in all_models  # Discovered
            assert "gpt-4-vision" in all_models  # Discovered

    def test_reload_clears_registry_cache(self, tmp_path):
        """Test that reload clears the registry cache"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            # Populate cache
            manager._registry_cache["openai"] = {"gpt-4"}
            manager._registry_cache_time["openai"] = time.time()

            assert len(manager._registry_cache) > 0

            # Reload should clear cache
            manager.reload()

            assert len(manager._registry_cache) == 0
            assert len(manager._registry_cache_time) == 0


class TestCapabilityCheckerSimplified:
    """Tests for CapabilityChecker helper"""

    def test_can_handle_request_success(self, tmp_path):
        """Test checking if provider can handle request"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
                "features": ["text", "streaming", "tools"],
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Reset and configure
        reset_config()
        with patch.object(UnifiedConfigManager, "_load_environment"):
            get_config(config_path=str(config_file))

        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai", features=[Feature.TEXT, Feature.TOOLS]
        )

        assert can_handle is True
        assert len(issues) == 0

    def test_can_handle_request_failure(self, tmp_path):
        """Test checking when provider cannot handle request"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "basic_provider": {
                "name": "basic_provider",
                "client_class": "BasicClient",
                "features": ["text"],  # No tools
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        reset_config()
        with patch.object(UnifiedConfigManager, "_load_environment"):
            get_config(config_path=str(config_file))

        can_handle, issues = CapabilityChecker.can_handle_request(
            "basic_provider", features=[Feature.TOOLS]
        )

        assert can_handle is False
        assert len(issues) > 0

    def test_get_model_info(self, tmp_path):
        """Test getting model information"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
                "features": ["text", "tools"],
                "max_context_length": 8192,
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        reset_config()
        with patch.object(UnifiedConfigManager, "_load_environment"):
            get_config(config_path=str(config_file))

        info = CapabilityChecker.get_model_info("openai", "gpt-4")

        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert "text" in info["features"]


class TestGlobalFunctions:
    """Tests for global helper functions"""

    def test_get_config_singleton(self):
        """Test that get_config returns singleton"""
        reset_config()

        with patch.object(UnifiedConfigManager, "_load_environment"):
            config1 = get_config()
            config2 = get_config()

            assert config1 is config2

    def test_reset_config(self):
        """Test resetting global config"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            config1 = get_config()

            reset_config()

            config2 = get_config()

            assert config1 is not config2


class TestYAMLLoading:
    """Tests for YAML file loading"""

    def test_load_yaml_file_basic(self, tmp_path):
        """Test loading a basic YAML file"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
                "default_model": "gpt-4",
                "features": ["text", "streaming"],
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            assert "openai" in manager.providers
            assert manager.providers["openai"].default_model == "gpt-4"

    def test_load_yaml_file_with_model_capabilities(self, tmp_path):
        """Test loading YAML with model capabilities"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "openai": {
                "name": "openai",
                "client_class": "OpenAIClient",
                "features": ["text"],
                "model_capabilities": [
                    {
                        "pattern": "gpt-4.*",
                        "features": ["tools", "vision"],
                        "max_context_length": 8192,
                    }
                ],
            }
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            provider = manager.providers["openai"]
            assert len(provider.model_capabilities) == 1
            assert provider.model_capabilities[0].pattern == "gpt-4.*"

    def test_load_yaml_file_missing(self):
        """Test loading with missing file"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            # Should not raise, just load package default
            manager = UnifiedConfigManager(config_path="/nonexistent/config.yaml")

            # Should still work (loads package default)
            assert isinstance(manager.providers, dict)

    def test_load_yaml_file_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML"""
        config_file = tmp_path / "bad_config.yaml"

        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [[[")

        with patch.object(UnifiedConfigManager, "_load_environment"):
            with pytest.raises(Exception):  # YAML parse error
                UnifiedConfigManager(config_path=str(config_file))


class TestEnvironmentHandling:
    """Tests for environment variable handling"""

    def test_load_environment_with_env_file(self, tmp_path):
        """Test loading .env file"""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=test_value\n")

        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            manager = UnifiedConfigManager()

            # Should have loaded .env
            assert os.getenv("TEST_KEY") == "test_value"
        finally:
            os.chdir(original_cwd)

    def test_load_environment_no_env_file(self, tmp_path):
        """Test when no .env file exists"""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Should not raise
            with patch.object(UnifiedConfigManager, "_load_config"):
                manager = UnifiedConfigManager()

                assert isinstance(manager, UnifiedConfigManager)
        finally:
            os.chdir(original_cwd)


class TestConfigValidation:
    """Tests for configuration validation with Pydantic"""

    def test_pydantic_validates_api_base_url(self):
        """Test that Pydantic validates API base URLs"""
        from pydantic import ValidationError

        # Invalid URL should be rejected
        with pytest.raises(ValidationError):
            ProviderConfig(
                name="test",
                client_class="TestClient",
                api_base="not-a-url",  # Missing http://
            )

    def test_pydantic_validates_required_fields(self):
        """Test that Pydantic validates required fields"""
        from pydantic import ValidationError

        # Missing name should be rejected
        with pytest.raises(ValidationError):
            ProviderConfig(client_class="TestClient")

    def test_pydantic_validates_on_assignment(self):
        """Test that Pydantic validates on assignment"""
        from pydantic import ValidationError

        config = ProviderConfig(
            name="test",
            client_class="TestClient",
            api_base="https://api.test.com",
        )

        # Trying to assign invalid URL should fail
        with pytest.raises(ValidationError):
            config.api_base = "invalid-url"
# test_unified_config_edge_cases.py
"""
Edge case tests for unified configuration system to achieve 90%+ coverage.

Focuses on error paths, edge cases, and fallback behaviors.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from chuk_llm.configuration.models import Feature, ProviderConfig
from chuk_llm.configuration.unified_config import (
    CapabilityChecker,
    UnifiedConfigManager,
    get_config,
    reset_config,
)


class TestImportFallbacks:
    """Test import fallback behavior"""

    def test_yaml_not_available(self):
        """Test behavior when PyYAML is not available"""
        with patch("chuk_llm.configuration.unified_config._YAML_AVAILABLE", False):
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager()

                # Should not raise, just log warning
                assert isinstance(manager, UnifiedConfigManager)
                assert isinstance(manager.providers, dict)

    def test_dotenv_not_available(self):
        """Test behavior when python-dotenv is not available"""
        with patch("chuk_llm.configuration.unified_config._DOTENV_AVAILABLE", False):
            with patch.object(UnifiedConfigManager, "_load_config"):
                manager = UnifiedConfigManager()

                # Should not raise, just skip .env loading
                assert isinstance(manager, UnifiedConfigManager)


class TestEnvironmentLoading:
    """Test environment variable loading edge cases"""

    def test_load_environment_finds_env_local(self, tmp_path):
        """Test finding .env.local file"""
        env_file = tmp_path / ".env.local"
        env_file.write_text("LOCAL_KEY=local_value\n")

        # Remove .env if it exists
        (tmp_path / ".env").unlink(missing_ok=True)

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch.object(UnifiedConfigManager, "_load_config"):
                manager = UnifiedConfigManager()

            # Should have loaded .env.local
            assert os.getenv("LOCAL_KEY") == "local_value"
        finally:
            os.chdir(original_cwd)
            # Clean up
            if "LOCAL_KEY" in os.environ:
                del os.environ["LOCAL_KEY"]

    def test_load_environment_finds_chuk_llm_env(self, tmp_path):
        """Test finding ~/.chuk_llm/.env file"""
        chuk_dir = tmp_path / ".chuk_llm"
        chuk_dir.mkdir()
        env_file = chuk_dir / ".env"
        env_file.write_text("CHUK_KEY=chuk_value\n")

        # Need to ensure .env and .env.local don't exist in tmp_path
        (tmp_path / ".env").unlink(missing_ok=True)
        (tmp_path / ".env.local").unlink(missing_ok=True)

        with patch("pathlib.Path.home", return_value=tmp_path):
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                with patch.object(UnifiedConfigManager, "_load_config"):
                    manager = UnifiedConfigManager()

                # Should have loaded ~/.chuk_llm/.env
                assert os.getenv("CHUK_KEY") == "chuk_value"
            finally:
                os.chdir(original_cwd)

        # Clean up
        if "CHUK_KEY" in os.environ:
            del os.environ["CHUK_KEY"]


class TestConfigFileDiscovery:
    """Test configuration file discovery edge cases"""

    def test_find_config_file_env_variable(self, tmp_path):
        """Test finding config via CHUK_LLM_CONFIG environment variable"""
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text("openai:\n  name: openai\n  client_class: OpenAIClient\n")

        with patch.dict(os.environ, {"CHUK_LLM_CONFIG": str(config_file)}):
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager()

                found_file = manager._find_config_file()
                assert found_file == config_file

    def test_find_config_file_providers_yaml(self, tmp_path):
        """Test finding providers.yaml as alternative"""
        config_file = tmp_path / "providers.yaml"
        config_file.write_text("openai:\n  name: openai\n  client_class: OpenAIClient\n")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager()

                found_file = manager._find_config_file()
                assert found_file and found_file.name == "providers.yaml"
        finally:
            os.chdir(original_cwd)

    def test_find_config_file_config_subdir(self, tmp_path):
        """Test finding config/chuk_llm.yaml"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "chuk_llm.yaml"
        config_file.write_text("openai:\n  name: openai\n  client_class: OpenAIClient\n")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager()

                found_file = manager._find_config_file()
                assert found_file and "config" in str(found_file)
        finally:
            os.chdir(original_cwd)

    def test_find_config_file_home_directory(self, tmp_path):
        """Test finding ~/.chuk_llm/config.yaml"""
        chuk_dir = tmp_path / ".chuk_llm"
        chuk_dir.mkdir()
        config_file = chuk_dir / "config.yaml"
        config_file.write_text("openai:\n  name: openai\n  client_class: OpenAIClient\n")

        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager()

                found_file = manager._find_config_file()
                assert found_file and found_file.name == "config.yaml"

    def test_get_package_config_error_handling(self):
        """Test error handling in _get_package_config"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Simulate error finding package config
            with patch("pathlib.Path.exists", side_effect=Exception("Test error")):
                result = manager._get_package_config()
                # Should return None, not raise
                assert result is None


class TestYAMLLoadingEdgeCases:
    """Test YAML loading edge cases"""

    def test_load_config_no_file_found(self):
        """Test loading when no config file exists"""
        with patch.object(UnifiedConfigManager, "_find_config_file", return_value=None):
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager()

                # Should not raise, just have empty providers
                assert isinstance(manager.providers, dict)

    def test_load_config_yaml_not_available(self, tmp_path):
        """Test loading when YAML library is not available"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("openai:\n  name: openai\n")

        with patch("chuk_llm.configuration.unified_config._YAML_AVAILABLE", False):
            with patch.object(UnifiedConfigManager, "_load_environment"):
                manager = UnifiedConfigManager(config_path=str(config_file))

                # Should log warning but not raise
                assert isinstance(manager.providers, dict)

    def test_load_config_empty_yaml_file(self, tmp_path):
        """Test loading empty YAML file"""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("")

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            # Should handle empty file gracefully
            assert isinstance(manager.providers, dict)

    def test_load_config_file_read_error(self, tmp_path):
        """Test handling file read errors"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("openai:\n  name: openai\n")

        with patch.object(UnifiedConfigManager, "_load_environment"):
            with patch("builtins.open", side_effect=PermissionError("Access denied")):
                with pytest.raises(PermissionError):
                    UnifiedConfigManager(config_path=str(config_file))


class TestFeatureParsing:
    """Test feature parsing edge cases"""

    def test_parse_features_with_set(self):
        """Test parsing features from set"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            features = manager._parse_features({"text", "streaming"})
            assert Feature.TEXT in features
            assert Feature.STREAMING in features

    def test_parse_features_unknown_in_list(self):
        """Test parsing list with unknown features"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Should skip unknown features, not raise
            features = manager._parse_features(["text", "bad_feature", "streaming"])
            assert Feature.TEXT in features
            assert Feature.STREAMING in features
            assert len(features) == 2  # bad_feature should be skipped

    def test_parse_features_unknown_in_set(self):
        """Test parsing set with unknown features"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            features = manager._parse_features({"text", "invalid_feature"})
            assert Feature.TEXT in features
            assert len(features) == 1

    def test_parse_model_capabilities_with_invalid_data(self):
        """Test parsing model capabilities with invalid data"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            # Invalid capability should be skipped
            caps_data = [
                {"pattern": "valid-.*", "features": ["text"]},
                {"bad": "data"},  # Missing required fields
            ]

            result = manager._parse_model_capabilities(caps_data)
            assert len(result) == 1  # Only valid one should be parsed


class TestProviderGetters:
    """Test provider getter edge cases"""

    def test_get_api_base_missing_provider(self):
        """Test getting API base for non-existent provider"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()

            result = manager.get_api_base("nonexistent")
            assert result is None

    def test_supports_feature_missing_provider(self):
        """Test checking feature support for non-existent provider"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()

            result = manager.supports_feature("nonexistent", Feature.TEXT)
            assert result is False

    def test_get_api_key_missing_provider(self):
        """Test getting API key for non-existent provider"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()

            result = manager.get_api_key("nonexistent")
            assert result is None


class TestRegistryIntegrationEdgeCases:
    """Test registry integration edge cases"""

    @pytest.mark.asyncio
    async def test_get_registry_models_discovery_returns_none(self):
        """Test when discovery returns None"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            with patch("chuk_llm.api.discovery.discover_models", return_value=None):
                models = await manager._get_registry_models("test_provider")
                assert models == set()

    @pytest.mark.asyncio
    async def test_get_registry_models_discovery_raises_error(self):
        """Test when discovery raises an error"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            with patch(
                "chuk_llm.api.discovery.discover_models",
                side_effect=Exception("Discovery failed"),
            ):
                models = await manager._get_registry_models("test_provider")
                assert models == set()  # Should return empty set, not raise

    def test_get_discovered_models_no_cache_async_error(self):
        """Test get_discovered_models when async operation fails"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()

            with patch(
                "asyncio.new_event_loop",
                side_effect=Exception("Event loop error"),
            ):
                models = manager.get_discovered_models("test_provider")
                assert models == set()  # Should return empty set, not raise

    def test_get_all_available_models_missing_provider(self):
        """Test getting all models for non-existent provider"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()

            models = manager.get_all_available_models("nonexistent")
            assert models == set()


class TestCapabilityCheckerEdgeCases:
    """Test CapabilityChecker edge cases"""

    def test_can_handle_request_provider_not_found(self, tmp_path):
        """Test when provider doesn't exist"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("openai:\n  name: openai\n  client_class: OpenAIClient\n")

        reset_config()
        with patch.object(UnifiedConfigManager, "_load_environment"):
            get_config(config_path=str(config_file))

        can_handle, issues = CapabilityChecker.can_handle_request("nonexistent")

        assert can_handle is False
        assert len(issues) > 0

    def test_get_best_provider_for_features_no_match(self, tmp_path):
        """Test when no provider supports required features"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
basic:
  name: basic
  client_class: BasicClient
  features: [text]
"""
        )

        reset_config()
        with patch.object(UnifiedConfigManager, "_load_environment"):
            get_config(config_path=str(config_file))

        # Require features that no provider has
        best = CapabilityChecker.get_best_provider_for_features(
            [Feature.VISION, Feature.TOOLS]
        )

        assert best is None

    def test_get_best_provider_for_features_with_model_constraint(self, tmp_path):
        """Test finding best provider with model constraint"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
provider1:
  name: provider1
  client_class: Client1
  features: [text, tools]
provider2:
  name: provider2
  client_class: Client2
  features: [text, vision]
"""
        )

        reset_config()
        with patch.object(UnifiedConfigManager, "_load_environment"):
            get_config(config_path=str(config_file))

        best = CapabilityChecker.get_best_provider_for_features(
            [Feature.TEXT], model="any-model"
        )

        # Should find a provider that supports text
        assert best is not None

    def test_get_model_info_error(self, tmp_path):
        """Test getting model info when provider doesn't exist"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("openai:\n  name: openai\n  client_class: OpenAIClient\n")

        reset_config()
        with patch.object(UnifiedConfigManager, "_load_environment"):
            get_config(config_path=str(config_file))

        info = CapabilityChecker.get_model_info("nonexistent", "model")

        assert "error" in info


class TestGlobalAliases:
    """Test global aliases functionality"""

    def test_get_global_aliases_with_alternatives_key(self, tmp_path):
        """Test loading global aliases using 'aliases' key"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
aliases:
  fast: gpt-3.5-turbo
  smart: gpt-4
openai:
  name: openai
  client_class: OpenAIClient
"""
        )

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            aliases = manager.get_global_aliases()
            assert aliases["fast"] == "gpt-3.5-turbo"
            assert aliases["smart"] == "gpt-4"

    def test_get_global_aliases_returns_copy(self):
        """Test that get_global_aliases returns a copy"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.global_aliases["test"] = "value"

            aliases1 = manager.get_global_aliases()
            aliases2 = manager.get_global_aliases()

            # Should be equal but not the same object
            assert aliases1 == aliases2
            assert aliases1 is not aliases2

            # Modifying one shouldn't affect the other
            aliases1["new"] = "new_value"
            assert "new" not in aliases2


class TestGlobalSettings:
    """Test global settings functionality"""

    def test_global_settings_with_alternative_key(self, tmp_path):
        """Test loading global settings using 'global' key"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
global:
  active_provider: anthropic
  timeout: 30
openai:
  name: openai
  client_class: OpenAIClient
"""
        )

        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager(config_path=str(config_file))

            assert manager.global_settings["active_provider"] == "anthropic"
            assert manager.global_settings["timeout"] == 30


class TestProcessConfigData:
    """Test _process_config_data edge cases"""

    def test_process_config_skips_special_keys(self, tmp_path):
        """Test that special keys are skipped during processing"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()

            config_data = {
                "__global__": {"setting": "value"},
                "__global_aliases__": {"alias": "model"},
                "global": {"setting2": "value2"},
                "aliases": {"alias2": "model2"},
                "__special__": {"data": "value"},
                "openai": {"name": "openai", "client_class": "OpenAIClient"},
            }

            manager._process_config_data(config_data)

            # Should only have processed openai, not special keys
            assert "openai" in manager.providers
            assert "__global__" not in manager.providers
            assert "__global_aliases__" not in manager.providers
            assert "global" not in manager.providers
            assert "aliases" not in manager.providers
            assert "__special__" not in manager.providers

    def test_process_config_adds_name_if_missing(self):
        """Test that provider name is added if not in config"""
        with patch.object(UnifiedConfigManager, "_load_environment"):
            manager = UnifiedConfigManager()
            manager.providers.clear()

            config_data = {
                "my_provider": {
                    "client_class": "MyClient",
                    # Note: no 'name' field
                }
            }

            manager._process_config_data(config_data)

            # Name should be set from key
            assert manager.providers["my_provider"].name == "my_provider"


class TestConfigAlias:
    """Test ConfigManager alias"""

    def test_config_manager_alias(self):
        """Test that ConfigManager is an alias for UnifiedConfigManager"""
        from chuk_llm.configuration.unified_config import ConfigManager

        assert ConfigManager is UnifiedConfigManager

    def test_reset_unified_config_alias(self):
        """Test that reset_unified_config is an alias"""
        from chuk_llm.configuration.unified_config import (
            reset_config,
            reset_unified_config,
        )

        assert reset_unified_config is reset_config
