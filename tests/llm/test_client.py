"""Comprehensive tests for llm/client.py module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import inspect
from typing import Any

from chuk_llm.llm.client import (
    _import_string,
    _supports_param,
    _constructor_kwargs,
    _create_client_internal,
    get_client,
    validate_request_compatibility,
    list_available_providers,
    get_provider_info,
    validate_provider_setup,
    find_best_provider_for_request,
)
from chuk_llm.configuration import Feature


class TestImportString:
    """Tests for _import_string helper function."""

    def test_import_string_colon_syntax(self):
        """Test importing with colon syntax."""
        cls = _import_string("pathlib:Path")
        from pathlib import Path
        assert cls == Path

    def test_import_string_dot_syntax(self):
        """Test importing with dot syntax."""
        cls = _import_string("pathlib.Path")
        from pathlib import Path
        assert cls == Path

    def test_import_string_invalid_module(self):
        """Test importing invalid module raises error."""
        with pytest.raises(ModuleNotFoundError):
            _import_string("invalid_module:InvalidClass")

    def test_import_string_invalid_class(self):
        """Test importing invalid class raises error."""
        with pytest.raises(AttributeError):
            _import_string("pathlib:InvalidClass")


class TestSupportsParam:
    """Tests for _supports_param helper function."""

    def test_supports_param_with_direct_param(self):
        """Test detecting direct parameter."""
        class TestClass:
            def __init__(self, model: str, api_key: str):
                pass

        assert _supports_param(TestClass, "model") is True
        assert _supports_param(TestClass, "api_key") is True

    def test_supports_param_missing_param(self):
        """Test detecting missing parameter."""
        class TestClass:
            def __init__(self, model: str):
                pass

        assert _supports_param(TestClass, "api_key") is False

    def test_supports_param_with_kwargs(self):
        """Test detecting parameter with **kwargs."""
        class TestClass:
            def __init__(self, model: str, **kwargs):
                pass

        assert _supports_param(TestClass, "api_key") is True
        assert _supports_param(TestClass, "any_param") is True

    def test_supports_param_without_kwargs(self):
        """Test without **kwargs support."""
        class TestClass:
            def __init__(self, model: str):
                pass

        assert _supports_param(TestClass, "unknown") is False


class TestConstructorKwargs:
    """Tests for _constructor_kwargs helper function."""

    def test_constructor_kwargs_basic(self):
        """Test basic constructor kwargs extraction."""
        class TestClass:
            def __init__(self, model: str, api_key: str):
                pass

        config = {"model": "gpt-4", "api_key": "test-key"}
        result = _constructor_kwargs(TestClass, config)

        assert result["model"] == "gpt-4"
        assert result["api_key"] == "test-key"

    def test_constructor_kwargs_with_api_base(self):
        """Test constructor kwargs with api_base."""
        class TestClass:
            def __init__(self, model: str, api_key: str, api_base: str):
                pass

        config = {"model": "gpt-4", "api_key": "key", "api_base": "https://api.example.com"}
        result = _constructor_kwargs(TestClass, config)

        assert result["api_base"] == "https://api.example.com"

    def test_constructor_kwargs_with_kwargs_support(self):
        """Test constructor kwargs with **kwargs support."""
        class TestClass:
            def __init__(self, model: str, **kwargs):
                pass

        config = {"model": "gpt-4", "custom_param": "value", "another": "test"}
        result = _constructor_kwargs(TestClass, config)

        assert result["model"] == "gpt-4"
        assert result["custom_param"] == "value"
        assert result["another"] == "test"

    def test_constructor_kwargs_without_kwargs_support(self):
        """Test constructor kwargs without **kwargs support."""
        class TestClass:
            def __init__(self, model: str, api_key: str):
                pass

        config = {"model": "gpt-4", "api_key": "key", "extra": "ignored"}
        result = _constructor_kwargs(TestClass, config)

        assert result["model"] == "gpt-4"
        assert result["api_key"] == "key"
        assert "extra" not in result

    def test_constructor_kwargs_filters_none(self):
        """Test that None values are filtered in kwargs mode."""
        class TestClass:
            def __init__(self, model: str, **kwargs):
                pass

        config = {"model": "gpt-4", "api_key": None, "api_base": "base"}
        result = _constructor_kwargs(TestClass, config)

        assert result["model"] == "gpt-4"
        assert "api_key" not in result
        assert result["api_base"] == "base"

    def test_constructor_kwargs_known_params_only(self):
        """Test only known params added without kwargs."""
        class TestClass:
            def __init__(self, model: str, api_key: str):
                pass

        config = {"model": "gpt-4", "api_key": "key", "api_base": "base"}
        result = _constructor_kwargs(TestClass, config)

        assert result["model"] == "gpt-4"
        assert result["api_key"] == "key"
        assert "api_base" not in result  # Not in constructor signature


class TestCreateClientInternal:
    """Tests for _create_client_internal function."""

    def test_create_client_basic(self):
        """Test basic client creation."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client._import_string') as mock_import:

            # Setup mocks
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test.module:TestClient"
            mock_provider.models = ["gpt-4"]
            mock_provider.extra = {}
            mock_provider.model_aliases = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "test-key"
            mock_config.get_api_base.return_value = "https://api.test.com"
            mock_config._ensure_model_available.return_value = True

            mock_get_config.return_value = mock_config

            # Create mock client class
            mock_client_instance = Mock()
            mock_client_class = Mock(return_value=mock_client_instance)
            mock_import.return_value = mock_client_class

            # Mock ConfigValidator
            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                client = _create_client_internal("openai", "gpt-4")

                assert client == mock_client_instance
                mock_client_class.assert_called_once()

    def test_create_client_invalid_provider(self):
        """Test client creation with invalid provider."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_get_config.return_value.get_provider.side_effect = Exception("Provider not found")

            with pytest.raises(ValueError, match="Failed to get provider"):
                _create_client_internal("invalid_provider")

    def test_create_client_azure_openai(self):
        """Test Azure OpenAI client creation with special handling."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client._import_string') as mock_import, \
             patch('chuk_llm.llm.client.os') as mock_os:

            # Setup mocks
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:AzureClient"
            mock_provider.api_base = "https://test.openai.azure.com"
            mock_provider.extra = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "azure-key"
            mock_config.get_api_base.return_value = "https://test.openai.azure.com"

            mock_get_config.return_value = mock_config

            # Mock environment variables
            mock_os.getenv.side_effect = lambda k, default=None: {
                "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
                "AZURE_OPENAI_API_VERSION": "2024-02-01"
            }.get(k, default)

            # Create mock client
            mock_client_instance = Mock()
            mock_client_class = Mock(return_value=mock_client_instance)
            mock_import.return_value = mock_client_class

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                client = _create_client_internal("azure_openai", "my-deployment")

                # Should create client successfully
                assert client == mock_client_instance

    def test_create_client_no_model_specified(self):
        """Test client creation without model uses default."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client._import_string') as mock_import:

            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "default-model"
            mock_provider.client_class = "test:Client"
            mock_provider.models = ["default-model"]
            mock_provider.extra = {}
            mock_provider.model_aliases = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "key"
            mock_config.get_api_base.return_value = "base"
            mock_config._ensure_model_available.return_value = True

            mock_get_config.return_value = mock_config

            mock_client = Mock()
            mock_import.return_value = Mock(return_value=mock_client)

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                client = _create_client_internal("test_provider")

                # Should use default model
                assert client == mock_client

    def test_create_client_model_alias(self):
        """Test client creation with model alias."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client._import_string') as mock_import:

            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:Client"
            mock_provider.models = ["gpt-4-turbo"]
            mock_provider.model_aliases = {"gpt4": "gpt-4-turbo"}
            mock_provider.extra = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "key"
            mock_config.get_api_base.return_value = "base"
            mock_config._ensure_model_available.return_value = False  # Not directly available

            mock_get_config.return_value = mock_config

            mock_client = Mock()
            mock_import.return_value = Mock(return_value=mock_client)

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                client = _create_client_internal("test_provider", "gpt4")

                # Should resolve alias and create client
                assert client == mock_client


class TestGetClient:
    """Tests for get_client function."""

    def test_get_client_with_cache(self):
        """Test get_client uses cache by default."""
        with patch('chuk_llm.llm.client.os.getenv', return_value="1"), \
             patch('chuk_llm.client_registry.get_cached_client') as mock_cached:

            mock_client = Mock()
            mock_cached.return_value = mock_client

            client = get_client("openai", "gpt-4", use_cache=True)

            assert client == mock_client
            mock_cached.assert_called_once()

    def test_get_client_without_cache(self):
        """Test get_client bypasses cache when disabled."""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:

            mock_client = Mock()
            mock_create.return_value = mock_client

            client = get_client("openai", "gpt-4", use_cache=False)

            assert client == mock_client
            mock_create.assert_called_once()

    def test_get_client_cache_env_disabled(self):
        """Test get_client respects CHUK_LLM_CACHE_CLIENTS=0."""
        with patch('chuk_llm.llm.client.os.getenv', return_value="0"), \
             patch('chuk_llm.llm.client._create_client_internal') as mock_create:

            mock_client = Mock()
            mock_create.return_value = mock_client

            client = get_client("openai", "gpt-4", use_cache=True)

            # Should not use cache
            assert client == mock_client
            mock_create.assert_called_once()


class TestValidateRequestCompatibility:
    """Tests for validate_request_compatibility function."""

    def test_validate_request_compatibility_valid(self):
        """Test validation of valid request."""
        with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
            mock_validator.validate_request_compatibility.return_value = (True, [])

            result = validate_request_compatibility(
                provider="openai",
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                stream=False
            )

            assert result["valid"] is True
            assert result["issues"] == []
            assert result["provider"] == "openai"
            assert result["model"] == "gpt-4"

    def test_validate_request_compatibility_invalid(self):
        """Test validation of invalid request."""
        with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
            mock_validator.validate_request_compatibility.return_value = (
                False,
                ["Model doesn't support tools"]
            )

            result = validate_request_compatibility(
                provider="openai",
                model="gpt-3.5",
                tools=[{"name": "test"}]
            )

            assert result["valid"] is False
            assert len(result["issues"]) > 0

    def test_validate_request_compatibility_error(self):
        """Test validation handles errors."""
        with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
            mock_validator.validate_request_compatibility.side_effect = Exception("Validation error")

            result = validate_request_compatibility(provider="openai")

            assert result["valid"] is False
            assert len(result["issues"]) > 0
            assert "Validation error" in result["issues"][0]


class TestListAvailableProviders:
    """Tests for list_available_providers function."""

    def test_list_available_providers_basic(self):
        """Test listing available providers."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai", "anthropic"]

            # Mock openai provider
            mock_openai = Mock()
            mock_openai.default_model = "gpt-4"
            mock_openai.models = ["gpt-4", "gpt-3.5-turbo"]
            mock_openai.model_aliases = {}
            mock_openai.features = [Feature.TEXT, Feature.STREAMING]
            mock_openai.api_base = "https://api.openai.com"
            mock_openai.rate_limits = {"requests_per_minute": 60}
            mock_openai.extra = {}

            mock_model_caps = Mock()
            mock_model_caps.features = [Feature.TEXT, Feature.STREAMING, Feature.TOOLS]
            mock_model_caps.max_context_length = 8192
            mock_model_caps.max_output_tokens = 4096

            mock_openai.get_model_capabilities.return_value = mock_model_caps

            mock_config.get_provider.return_value = mock_openai
            mock_config.get_api_key.return_value = "test-key"
            mock_config._discovery_cache = {}

            mock_get_config.return_value = mock_config

            result = list_available_providers()

            assert "openai" in result
            assert "anthropic" in result
            assert result["openai"]["default_model"] == "gpt-4"
            assert result["openai"]["has_api_key"] is True

    def test_list_available_providers_with_error(self):
        """Test listing providers handles errors gracefully."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai"]
            mock_config.get_provider.side_effect = Exception("Provider error")

            mock_get_config.return_value = mock_config

            result = list_available_providers()

            assert "openai" in result
            assert "error" in result["openai"]


class TestGetProviderInfo:
    """Tests for get_provider_info function."""

    def test_get_provider_info_basic(self):
        """Test getting provider info."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:Client"
            mock_provider.api_base = "https://api.test.com"
            mock_provider.models = ["gpt-4"]
            mock_provider.model_aliases = {}
            mock_provider.rate_limits = {}
            mock_provider.model_capabilities = []

            mock_caps = Mock()
            mock_caps.features = [Feature.TEXT, Feature.STREAMING]
            mock_caps.max_context_length = 8192
            mock_caps.max_output_tokens = 4096

            mock_provider.get_model_capabilities.return_value = mock_caps
            mock_provider.supports_feature.return_value = True

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "key"
            mock_config._discovery_cache = {}

            mock_get_config.return_value = mock_config

            result = get_provider_info("openai", "gpt-4")

            assert result["provider"] == "openai"
            assert result["model"] == "gpt-4"
            assert result["has_api_key"] is True
            assert "features" in result
            assert "supports" in result

    def test_get_provider_info_with_error(self):
        """Test get_provider_info handles errors."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_get_config.return_value.get_provider.side_effect = Exception("Provider not found")

            result = get_provider_info("invalid")

            assert "error" in result


class TestValidateProviderSetup:
    """Tests for validate_provider_setup function."""

    def test_validate_provider_setup_valid(self):
        """Test validating valid provider setup."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client.ConfigValidator') as mock_validator, \
             patch('chuk_llm.llm.client._import_string'):

            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.client_class = "test:Client"
            mock_provider.models = ["gpt-4"]
            mock_provider.default_model = "gpt-4"
            mock_provider.features = [Feature.TEXT]
            mock_provider.model_aliases = {}
            mock_provider.rate_limits = {}
            mock_provider.extra = {}

            mock_caps = Mock()
            mock_caps.features = [Feature.TEXT]

            mock_provider.get_model_capabilities.return_value = mock_caps

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "key"
            mock_config._discovery_cache = {}

            mock_get_config.return_value = mock_config

            mock_validator.validate_provider_config.return_value = (True, [])

            result = validate_provider_setup("openai")

            assert result["valid"] is True
            assert result["has_api_key"] is True
            assert result["client_import_ok"] is True

    def test_validate_provider_setup_not_found(self):
        """Test validating non-existent provider."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_get_config.return_value.get_provider.side_effect = Exception("Not found")

            result = validate_provider_setup("invalid")

            assert result["valid"] is False
            assert "error" in result

    def test_validate_provider_setup_no_models(self):
        """Test validating provider with no models."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client.ConfigValidator') as mock_validator, \
             patch('chuk_llm.llm.client._import_string'):

            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.client_class = "test:Client"
            mock_provider.models = []  # No models
            mock_provider.default_model = None
            mock_provider.features = []
            mock_provider.model_aliases = {}
            mock_provider.rate_limits = {}
            mock_provider.extra = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = None
            mock_config._discovery_cache = {}

            mock_get_config.return_value = mock_config

            mock_validator.validate_provider_config.return_value = (True, [])

            result = validate_provider_setup("test_provider")

            assert "No models configured" in result["warnings"]


class TestFindBestProviderForRequest:
    """Tests for find_best_provider_for_request function."""

    def test_find_best_provider_with_features(self):
        """Test finding best provider with required features."""
        with patch('chuk_llm.configuration.unified_config.CapabilityChecker') as mock_checker, \
             patch('chuk_llm.llm.client.get_provider_info') as mock_get_info:

            mock_checker.get_best_provider_for_features.return_value = "openai"
            mock_get_info.return_value = {"provider": "openai", "model": "gpt-4"}

            result = find_best_provider_for_request(required_features=["tools", "streaming"])

            assert result is not None
            assert result["provider"] == "openai"
            mock_checker.get_best_provider_for_features.assert_called_once()

    def test_find_best_provider_no_match(self):
        """Test finding best provider when no match found."""
        with patch('chuk_llm.configuration.unified_config.CapabilityChecker') as mock_checker:
            mock_checker.get_best_provider_for_features.return_value = None

            result = find_best_provider_for_request(required_features=["text", "streaming"])

            # Should call checker but find no match
            assert result is None

    def test_find_best_provider_no_requirements(self):
        """Test finding best provider with no requirements."""
        result = find_best_provider_for_request()

        assert result is None


class TestClientCreationEdgeCases:
    """Tests for edge cases in client creation."""

    def test_create_client_no_default_model(self):
        """Test client creation when no default model and none specified."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = None
            mock_provider.client_class = "test:Client"

            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            with pytest.raises(ValueError, match="No model specified"):
                _create_client_internal("test_provider")

    def test_create_client_azure_no_deployment(self):
        """Test Azure client creation without deployment."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = None
            mock_provider.client_class = "test:AzureClient"

            mock_config.get_provider.return_value = mock_provider
            mock_get_config.return_value = mock_config

            with pytest.raises(ValueError, match="No deployment specified"):
                _create_client_internal("azure_openai")

    def test_create_client_invalid_model_after_discovery(self):
        """Test client creation with invalid model after discovery."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:Client"
            mock_provider.models = ["gpt-4"]
            mock_provider.model_aliases = {}
            mock_provider.extra = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config._ensure_model_available.return_value = False

            mock_get_config.return_value = mock_config

            with pytest.raises(ValueError, match="not available"):
                _create_client_internal("test_provider", "invalid-model")

    def test_create_client_empty_client_class(self):
        """Test client creation with empty client_class."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = ""
            mock_provider.models = ["gpt-4"]
            mock_provider.extra = {}
            mock_provider.model_aliases = {}
            mock_provider.api_key_env = None  # Add this to avoid validation issues

            mock_config.get_provider.return_value = mock_provider
            mock_config._ensure_model_available.return_value = True
            mock_config.get_api_key.return_value = "key"
            mock_config.get_api_base.return_value = "base"

            mock_get_config.return_value = mock_config

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                with pytest.raises(ValueError, match="No client class configured"):
                    _create_client_internal("test_provider", "gpt-4")

    def test_create_client_import_error(self):
        """Test client creation with import error."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client._import_string') as mock_import:

            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "invalid:BadClass"
            mock_provider.models = ["gpt-4"]
            mock_provider.extra = {}
            mock_provider.model_aliases = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config._ensure_model_available.return_value = True
            mock_config.get_api_key.return_value = "key"
            mock_config.get_api_base.return_value = "base"

            mock_get_config.return_value = mock_config

            mock_import.side_effect = Exception("Import failed")

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                with pytest.raises(ValueError, match="Failed to import client class"):
                    _create_client_internal("test_provider", "gpt-4")

    def test_create_client_instantiation_error(self):
        """Test client creation with instantiation error."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client._import_string') as mock_import:

            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:Client"
            mock_provider.models = ["gpt-4"]
            mock_provider.extra = {}
            mock_provider.model_aliases = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config._ensure_model_available.return_value = True
            mock_config.get_api_key.return_value = "key"
            mock_config.get_api_base.return_value = "base"

            mock_get_config.return_value = mock_config

            # Mock client class that raises on instantiation
            mock_client_class = Mock(side_effect=Exception("Instantiation failed"))
            mock_import.return_value = mock_client_class

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                with pytest.raises(ValueError, match="Failed to create"):
                    _create_client_internal("test_provider", "gpt-4")

    def test_create_client_azure_instantiation_error(self):
        """Test Azure client creation with instantiation error."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config, \
             patch('chuk_llm.llm.client._import_string') as mock_import, \
             patch('chuk_llm.llm.client.os') as mock_os:

            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:AzureClient"
            mock_provider.api_base = "https://test.azure.com"
            mock_provider.extra = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "key"
            mock_config.get_api_base.return_value = "https://test.azure.com"

            mock_get_config.return_value = mock_config

            mock_os.getenv.return_value = None

            mock_client_class = Mock(side_effect=Exception("Azure error"))
            mock_import.return_value = mock_client_class

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (True, [])

                with pytest.raises(ValueError, match="Failed to create Azure"):
                    _create_client_internal("azure_openai", "my-deployment")

    def test_create_client_validation_failure_non_azure(self):
        """Test client creation with validation failure for non-Azure."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:Client"
            mock_provider.models = ["gpt-4"]
            mock_provider.extra = {}
            mock_provider.model_aliases = {}

            mock_config.get_provider.return_value = mock_provider
            mock_config._ensure_model_available.return_value = True
            mock_config.get_api_key.return_value = "key"
            mock_config.get_api_base.return_value = "base"

            mock_get_config.return_value = mock_config

            with patch('chuk_llm.llm.client.ConfigValidator') as mock_validator:
                mock_validator.validate_provider_config.return_value = (
                    False,
                    ["Invalid configuration"]
                )

                with pytest.raises(ValueError, match="Invalid provider configuration"):
                    _create_client_internal("test_provider", "gpt-4")


class TestListAvailableProvidersEdgeCases:
    """Tests for edge cases in list_available_providers."""

    def test_list_providers_with_discovery_enabled(self):
        """Test listing providers with discovery enabled."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get_all_providers.return_value = ["openai"]

            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.models = ["gpt-4"]
            mock_provider.model_aliases = {}
            mock_provider.features = [Feature.TEXT]
            mock_provider.api_base = "https://api.openai.com"
            mock_provider.rate_limits = {}
            mock_provider.extra = {
                "dynamic_discovery": {
                    "enabled": True
                }
            }

            mock_caps = Mock()
            mock_caps.features = [Feature.TEXT]
            mock_caps.max_context_length = 8192
            mock_caps.max_output_tokens = 4096

            mock_provider.get_model_capabilities.return_value = mock_caps

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "key"
            mock_config._discovery_cache = {
                "openai": {
                    "discovered_count": 10,
                    "new_count": 5,
                    "timestamp": 1234567890
                }
            }

            mock_get_config.return_value = mock_config

            result = list_available_providers()

            assert "openai" in result
            assert result["openai"]["discovery_enabled"] is True
            assert "discovery_stats" in result["openai"]


class TestGetProviderInfoEdgeCases:
    """Tests for edge cases in get_provider_info."""

    def test_get_provider_info_with_discovery(self):
        """Test getting provider info for discovered model."""
        with patch('chuk_llm.llm.client.get_config') as mock_get_config:
            mock_config = Mock()
            mock_provider = Mock()
            mock_provider.default_model = "gpt-4"
            mock_provider.client_class = "test:Client"
            mock_provider.api_base = "https://api.test.com"
            mock_provider.models = ["gpt-4", "discovered-model"]
            mock_provider.model_aliases = {}
            mock_provider.rate_limits = {}
            mock_provider.model_capabilities = []

            mock_caps = Mock()
            mock_caps.features = [Feature.TEXT]
            mock_caps.max_context_length = 8192
            mock_caps.max_output_tokens = 4096

            mock_provider.get_model_capabilities.return_value = mock_caps
            mock_provider.supports_feature.return_value = True

            mock_config.get_provider.return_value = mock_provider
            mock_config.get_api_key.return_value = "key"
            mock_config._discovery_cache = {
                "test_provider": {
                    "models": ["discovered-model"],
                    "timestamp": 1234567890
                }
            }

            mock_get_config.return_value = mock_config

            result = get_provider_info("test_provider", "discovered-model")

            assert result["is_discovered"] is True
            assert "discovery_info" in result
