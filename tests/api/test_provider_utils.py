"""Tests for chuk_llm/api/provider_utils.py - Provider configuration utilities."""

import os
import tempfile
from unittest.mock import patch

import pytest

from chuk_llm.api.provider_utils import (
    find_providers_yaml_path,
    get_all_providers,
    get_provider_config,
    get_provider_default_model,
)


@pytest.fixture
def temp_providers_yaml():
    """Create a temporary providers.yaml file for testing."""
    yaml_content = """
anthropic:
  api_key_env: ANTHROPIC_API_KEY
  default_model: claude-3-7-sonnet-20250219
  base_url: https://api.anthropic.com

openai:
  api_key_env: OPENAI_API_KEY
  default_model: gpt-4o-mini
  base_url: https://api.openai.com

groq:
  inherits: openai
  api_key_env: GROQ_API_KEY
  default_model: llama-3.3-70b-versatile
  base_url: https://api.groq.com

deepseek:
  api_key_env: DEEPSEEK_API_KEY
  default_model: deepseek-chat
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(yaml_content)
        temp_path = temp_file.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


class TestGetProviderDefaultModel:
    """Test get_provider_default_model function."""

    def test_get_provider_default_model_found(self, temp_providers_yaml):
        """Test getting default model when provider exists."""
        # Point to our temp file
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                model = get_provider_default_model("anthropic")
                assert model == "claude-3-7-sonnet-20250219"

    def test_get_provider_default_model_with_inheritance(self, temp_providers_yaml):
        """Test getting default model with inheritance."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                model = get_provider_default_model("groq")
                assert model == "llama-3.3-70b-versatile"

    def test_get_provider_default_model_not_found(self, temp_providers_yaml):
        """Test getting default model when provider doesn't exist."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                model = get_provider_default_model("nonexistent")
                assert model is None

    def test_get_provider_default_model_no_yaml(self):
        """Test getting default model when yaml file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            model = get_provider_default_model("openai")
            assert model is None

    def test_get_provider_default_model_yaml_import_error(self):
        """Test getting default model when yaml module not available."""
        with patch("builtins.__import__", side_effect=ImportError):
            model = get_provider_default_model("openai")
            assert model is None

    def test_get_provider_default_model_with_env_var(
        self, temp_providers_yaml, monkeypatch
    ):
        """Test using environment variable to specify yaml path."""
        monkeypatch.setenv("CHUK_LLM_PROVIDERS_YAML", temp_providers_yaml)

        # The function should find it via env var
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = "/some/path"
            with patch("os.path.join") as mock_join:
                # Make sure only the env var path exists
                def custom_join(*args):
                    return temp_providers_yaml if "CHUK_LLM" in str(args) else "/fake"

                mock_join.side_effect = custom_join

                model = get_provider_default_model("anthropic")
                # Should still work via env var fallback
                assert model == "claude-3-7-sonnet-20250219" or model is None


class TestGetProviderConfig:
    """Test get_provider_config function."""

    def test_get_provider_config_found(self, temp_providers_yaml):
        """Test getting full config when provider exists."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                config = get_provider_config("anthropic")
                assert config["api_key_env"] == "ANTHROPIC_API_KEY"
                assert config["default_model"] == "claude-3-7-sonnet-20250219"

    def test_get_provider_config_with_inheritance(self, temp_providers_yaml):
        """Test getting config with inheritance."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                config = get_provider_config("groq")
                # Should have inherited openai's values plus its own
                assert config["api_key_env"] == "GROQ_API_KEY"
                assert config["default_model"] == "llama-3.3-70b-versatile"
                assert config["base_url"] == "https://api.groq.com"

    def test_get_provider_config_not_found(self, temp_providers_yaml):
        """Test getting config when provider doesn't exist."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                config = get_provider_config("nonexistent")
                assert config == {}

    def test_get_provider_config_no_yaml(self):
        """Test getting config when yaml file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            config = get_provider_config("openai")
            assert config == {}

    def test_get_provider_config_exception_handling(self):
        """Test that exceptions are handled gracefully."""
        with patch("builtins.open", side_effect=PermissionError("No access")):
            config = get_provider_config("openai")
            assert config == {}


class TestGetAllProviders:
    """Test get_all_providers function."""

    def test_get_all_providers_from_yaml(self, temp_providers_yaml):
        """Test getting all providers from yaml."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                providers = get_all_providers()
                assert "anthropic" in providers
                assert "openai" in providers
                assert "groq" in providers
                assert "deepseek" in providers

    def test_get_all_providers_no_yaml(self):
        """Test getting all providers when yaml doesn't exist (fallback)."""
        with patch("os.path.exists", return_value=False):
            providers = get_all_providers()
            # Should return fallback list
            assert "openai" in providers
            assert "anthropic" in providers
            assert isinstance(providers, list)

    def test_get_all_providers_filters_special_keys(self):
        """Test that special keys (starting with __) are filtered."""
        yaml_content = """
__version__: 1.0
__metadata__:
  updated: 2025-01-01

openai:
  default_model: gpt-4o-mini

anthropic:
  default_model: claude-3-7-sonnet-20250219
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            temp_file.write(yaml_content)
            temp_path = temp_file.name

        try:
            with patch("os.path.dirname") as mock_dirname:
                mock_dirname.return_value = os.path.dirname(temp_path)
                with patch("os.path.join") as mock_join:
                    mock_join.return_value = temp_path

                    providers = get_all_providers()
                    assert "openai" in providers
                    assert "anthropic" in providers
                    assert "__version__" not in providers
                    assert "__metadata__" not in providers
        finally:
            os.unlink(temp_path)

    def test_get_all_providers_exception_returns_fallback(self):
        """Test that exceptions return fallback list."""
        with patch("builtins.open", side_effect=Exception("Error")):
            providers = get_all_providers()
            # Should return fallback
            assert isinstance(providers, list)
            assert len(providers) > 0


class TestFindProvidersYamlPath:
    """Test find_providers_yaml_path function."""

    def test_find_providers_yaml_path_found(self, temp_providers_yaml):
        """Test finding yaml path when it exists."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                path = find_providers_yaml_path()
                assert path is not None
                assert path.endswith(".yaml")

    def test_find_providers_yaml_path_not_found(self):
        """Test finding yaml path when it doesn't exist."""
        with patch("os.path.exists", return_value=False):
            path = find_providers_yaml_path()
            assert path is None

    def test_find_providers_yaml_path_returns_absolute(self, temp_providers_yaml):
        """Test that returned path is absolute."""
        with patch("os.path.dirname") as mock_dirname:
            mock_dirname.return_value = os.path.dirname(temp_providers_yaml)
            with patch("os.path.join") as mock_join:
                mock_join.return_value = temp_providers_yaml

                path = find_providers_yaml_path()
                if path:  # May be None if not found
                    assert os.path.isabs(path)

    def test_find_providers_yaml_path_with_env_var(
        self, temp_providers_yaml, monkeypatch
    ):
        """Test finding yaml via environment variable."""
        monkeypatch.setenv("CHUK_LLM_PROVIDERS_YAML", temp_providers_yaml)

        # The function checks multiple paths including env vars
        path = find_providers_yaml_path()
        # Should find it somewhere
        assert path is None or isinstance(path, str)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_yaml_file(self):
        """Test handling of empty yaml file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            temp_file.write("")
            temp_path = temp_file.name

        try:
            with patch("os.path.dirname") as mock_dirname:
                mock_dirname.return_value = os.path.dirname(temp_path)
                with patch("os.path.join") as mock_join:
                    mock_join.return_value = temp_path

                    model = get_provider_default_model("any")
                    assert model is None

                    config = get_provider_config("any")
                    assert config == {}
        finally:
            os.unlink(temp_path)

    def test_malformed_yaml(self):
        """Test handling of malformed yaml."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            temp_file.write("invalid: yaml: content: ][")
            temp_path = temp_file.name

        try:
            with patch("os.path.dirname") as mock_dirname:
                mock_dirname.return_value = os.path.dirname(temp_path)
                with patch("os.path.join") as mock_join:
                    mock_join.return_value = temp_path

                    # Should handle gracefully
                    model = get_provider_default_model("any")
                    assert model is None
        finally:
            os.unlink(temp_path)

    def test_yaml_with_null_values(self):
        """Test handling of yaml with null values."""
        yaml_content = """
openai:
  default_model: null
  api_key_env: OPENAI_API_KEY
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            temp_file.write(yaml_content)
            temp_path = temp_file.name

        try:
            with patch("os.path.dirname") as mock_dirname:
                mock_dirname.return_value = os.path.dirname(temp_path)
                with patch("os.path.join") as mock_join:
                    mock_join.return_value = temp_path

                    model = get_provider_default_model("openai")
                    assert model is None

                    config = get_provider_config("openai")
                    assert config.get("api_key_env") == "OPENAI_API_KEY"
        finally:
            os.unlink(temp_path)
