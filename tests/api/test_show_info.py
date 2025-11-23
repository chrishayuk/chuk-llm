"""Tests for chuk_llm/api/show_info.py - Display utilities."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_llm.api.show_info import (
    _add_show_functions_to_module,
    show_capabilities,
    show_config,
    show_functions,
    show_model_aliases,
    show_providers,
)
from chuk_llm.configuration import Feature


class TestShowProviders:
    """Test show_providers function."""

    def test_show_providers_success(self, capsys):
        """Test showing providers successfully."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.models = ["model1", "model2", "model3"]
        mock_provider.default_model = "model1"
        mock_provider.features = []

        mock_config.get_all_providers.return_value = ["openai", "anthropic"]
        mock_config.get_provider.return_value = mock_provider
        mock_config.get_api_key.return_value = "test-key"

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_providers()

            captured = capsys.readouterr()
            assert "Available LLM Providers" in captured.out
            assert "openai" in captured.out.lower()
            assert "Configured" in captured.out

    def test_show_providers_with_many_models(self, capsys):
        """Test showing providers with many models."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.models = [f"model{i}" for i in range(10)]
        mock_provider.default_model = "model0"
        mock_provider.features = []

        mock_config.get_all_providers.return_value = ["openai"]
        mock_config.get_provider.return_value = mock_provider
        mock_config.get_api_key.return_value = "key"

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_providers()

            captured = capsys.readouterr()
            assert "... and 5 more" in captured.out

    def test_show_providers_no_api_key(self, capsys):
        """Test showing providers without API key."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.models = ["model1"]
        mock_provider.default_model = "model1"
        mock_provider.features = []

        mock_config.get_all_providers.return_value = ["openai"]
        mock_config.get_provider.return_value = mock_provider
        mock_config.get_api_key.return_value = None

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_providers()

            captured = capsys.readouterr()
            assert "No API key" in captured.out

    def test_show_providers_with_features(self, capsys):
        """Test showing providers with features."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.models = ["model1"]
        mock_provider.default_model = "model1"
        mock_provider.features = [
            Feature.VISION,
            Feature.TOOLS,
            Feature.STREAMING,
            Feature.JSON_MODE,
            Feature.SYSTEM_MESSAGES,
        ]

        mock_config.get_all_providers.return_value = ["openai"]
        mock_config.get_provider.return_value = mock_provider
        mock_config.get_api_key.return_value = "key"

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_providers()

            captured = capsys.readouterr()
            assert "Features:" in captured.out

    def test_show_providers_error_handling(self, capsys):
        """Test error handling in show_providers."""
        with patch(
            "chuk_llm.configuration.get_config", side_effect=Exception("Config error")
        ):
            show_providers()

            captured = capsys.readouterr()
            assert "Error loading providers" in captured.out

    def test_show_providers_individual_provider_error(self, capsys):
        """Test error handling for individual provider."""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = ["openai", "broken"]

        def get_provider_side_effect(name):
            if name == "broken":
                raise Exception("Provider error")
            provider = MagicMock()
            provider.models = []
            provider.default_model = "gpt-4"
            provider.features = []
            return provider

        mock_config.get_provider.side_effect = get_provider_side_effect
        mock_config.get_api_key.return_value = "key"

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_providers()

            captured = capsys.readouterr()
            assert "Error:" in captured.out


class TestShowFunctions:
    """Test show_functions function."""

    def test_show_functions_success(self, capsys):
        """Test showing functions successfully."""
        mock_functions = [
            "ask_openai",
            "ask_openai_sync",
            "stream_openai",
            "ask_anthropic",
            "stream_anthropic",
        ]

        with patch("chuk_llm.api.providers.list_provider_functions", return_value=mock_functions):
            show_functions()

            captured = capsys.readouterr()
            assert "Auto-Generated Functions" in captured.out
            assert "Async Functions" in captured.out
            assert "Sync Functions" in captured.out
            assert "Streaming Functions" in captured.out

    def test_show_functions_empty(self, capsys):
        """Test showing functions when none exist."""
        with patch("chuk_llm.api.providers.list_provider_functions", return_value=[]):
            show_functions()

            captured = capsys.readouterr()
            assert "No provider functions found" in captured.out

    def test_show_functions_many_functions(self, capsys):
        """Test showing many functions (more than 20)."""
        mock_functions = [f"ask_provider{i}" for i in range(30)]

        with patch("chuk_llm.api.providers.list_provider_functions", return_value=mock_functions):
            show_functions()

            captured = capsys.readouterr()
            assert "... and 10 more" in captured.out

    def test_show_functions_error_handling(self, capsys):
        """Test error handling in show_functions."""
        with patch(
            "chuk_llm.api.providers.list_provider_functions",
            side_effect=Exception("List error"),
        ):
            show_functions()

            captured = capsys.readouterr()
            assert "Error listing functions" in captured.out

    def test_show_functions_example_usage(self, capsys):
        """Test that example usage is shown."""
        with patch("chuk_llm.api.providers.list_provider_functions", return_value=["ask_openai"]):
            show_functions()

            captured = capsys.readouterr()
            assert "Example Usage:" in captured.out
            assert "ask_openai" in captured.out


class TestShowModelAliases:
    """Test show_model_aliases function."""

    def test_show_model_aliases_success(self, capsys):
        """Test showing model aliases successfully."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.model_aliases = {"gpt4": "gpt-4-turbo", "gpt3": "gpt-3.5-turbo"}

        mock_config.get_all_providers.return_value = ["openai"]
        mock_config.get_provider.return_value = mock_provider
        mock_config.get_global_aliases.return_value = {"gpt": "openai"}

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_model_aliases()

            captured = capsys.readouterr()
            assert "Model Aliases" in captured.out
            assert "OPENAI:" in captured.out
            assert "gpt4" in captured.out
            assert "GLOBAL ALIASES:" in captured.out

    def test_show_model_aliases_no_aliases(self, capsys):
        """Test showing model aliases when provider has none."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.model_aliases = {}

        mock_config.get_all_providers.return_value = ["openai"]
        mock_config.get_provider.return_value = mock_provider
        mock_config.get_global_aliases.return_value = {}

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_model_aliases()

            captured = capsys.readouterr()
            assert "Model Aliases" in captured.out

    def test_show_model_aliases_error_handling(self, capsys):
        """Test error handling in show_model_aliases."""
        with patch(
            "chuk_llm.configuration.get_config", side_effect=Exception("Config error")
        ):
            show_model_aliases()

            captured = capsys.readouterr()
            assert "Error loading aliases" in captured.out

    def test_show_model_aliases_provider_error(self, capsys):
        """Test handling provider-specific errors."""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = ["openai", "broken"]

        def get_provider_side_effect(name):
            if name == "broken":
                raise Exception("Provider error")
            provider = MagicMock()
            provider.model_aliases = {}
            return provider

        mock_config.get_provider.side_effect = get_provider_side_effect
        mock_config.get_global_aliases.return_value = {}

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_model_aliases()
            # Should not crash, just skip the broken provider
            captured = capsys.readouterr()
            assert "Model Aliases" in captured.out


class TestShowCapabilities:
    """Test show_capabilities function."""

    def test_show_capabilities_all_providers(self, capsys):
        """Test showing capabilities for all providers."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.features = [Feature.VISION, Feature.TOOLS, Feature.STREAMING]

        mock_config.get_all_providers.return_value = ["openai"]
        mock_config.get_provider.return_value = mock_provider
        mock_config.get_api_key.return_value = "key"

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_capabilities()

            captured = capsys.readouterr()
            assert "Provider Capabilities Overview" in captured.out
            assert "openai" in captured.out.lower()

    def test_show_capabilities_specific_provider(self, capsys):
        """Test showing capabilities for specific provider."""
        mock_config = MagicMock()
        mock_provider = MagicMock()
        mock_provider.features = [Feature.VISION, Feature.TOOLS]
        mock_provider.models = ["model1", "model2"]

        mock_capability = MagicMock()
        mock_capability.max_context_length = 128000
        mock_capability.max_output_tokens = 4096
        mock_capability.features = [Feature.VISION]

        mock_provider.get_model_capabilities.return_value = mock_capability

        mock_config.get_provider.return_value = mock_provider

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_capabilities(provider="openai")

            captured = capsys.readouterr()
            assert "OPENAI Capabilities" in captured.out
            assert "Max context:" in captured.out

    def test_show_capabilities_provider_not_found(self, capsys):
        """Test showing capabilities for non-existent provider."""
        mock_config = MagicMock()
        mock_config.get_provider.side_effect = Exception("Provider not found")

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_capabilities(provider="nonexistent")

            captured = capsys.readouterr()
            assert "Error:" in captured.out

    def test_show_capabilities_error_handling(self, capsys):
        """Test error handling in show_capabilities."""
        with patch(
            "chuk_llm.configuration.get_config", side_effect=Exception("Config error")
        ):
            show_capabilities()

            captured = capsys.readouterr()
            assert "Error showing capabilities" in captured.out


class TestShowConfig:
    """Test show_config function."""

    def test_show_config_success(self, capsys):
        """Test showing config successfully."""
        mock_current_config = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        mock_config_manager = MagicMock()
        mock_config_manager.get_all_providers.return_value = ["openai", "anthropic"]
        mock_config_manager.get_api_key.side_effect = lambda p: "key" if p == "openai" else None

        with patch("chuk_llm.api.config.get_current_config", return_value=mock_current_config):
            with patch("chuk_llm.configuration.get_config", return_value=mock_config_manager):
                show_config()

                captured = capsys.readouterr()
                assert "Current Configuration" in captured.out
                assert "Active Provider: openai" in captured.out
                assert "Model: gpt-4" in captured.out

    def test_show_config_with_session_tracking(self, capsys):
        """Test showing config with session tracking enabled."""
        mock_current_config = {"provider": "openai"}
        mock_config_manager = MagicMock()
        mock_config_manager.get_all_providers.return_value = []
        mock_config_manager.get_api_key.return_value = None

        with patch("chuk_llm.api.config.get_current_config", return_value=mock_current_config):
            with patch("chuk_llm.configuration.get_config", return_value=mock_config_manager):
                with patch("chuk_llm.api.core._SESSIONS_ENABLED", True):
                    show_config()

                    captured = capsys.readouterr()
                    assert "Session Tracking:" in captured.out

    def test_show_config_error_handling(self, capsys):
        """Test error handling in show_config."""
        with patch(
            "chuk_llm.api.config.get_current_config",
            side_effect=Exception("Config error"),
        ):
            show_config()

            captured = capsys.readouterr()
            assert "Error showing config" in captured.out

    def test_show_config_default_values(self, capsys):
        """Test showing config with default/missing values."""
        mock_current_config = {}
        mock_config_manager = MagicMock()
        mock_config_manager.get_all_providers.return_value = []
        mock_config_manager.get_api_key.return_value = None

        with patch("chuk_llm.api.config.get_current_config", return_value=mock_current_config):
            with patch("chuk_llm.configuration.get_config", return_value=mock_config_manager):
                show_config()

                captured = capsys.readouterr()
                assert "not set" in captured.out or "default" in captured.out


class TestAddShowFunctionsToModule:
    """Test _add_show_functions_to_module function."""

    def test_add_show_functions_to_module(self):
        """Test that show functions are added to module."""
        # Create a mock module
        mock_module = MagicMock()

        with patch.dict("sys.modules", {"chuk_llm": mock_module}):
            _add_show_functions_to_module()

            # Verify functions were added
            assert mock_module.show_providers == show_providers
            assert mock_module.show_functions == show_functions
            assert mock_module.show_model_aliases == show_model_aliases
            assert mock_module.show_capabilities == show_capabilities
            assert mock_module.show_config == show_config


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_show_providers_empty_list(self, capsys):
        """Test showing providers with empty list."""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = []

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_providers()

            captured = capsys.readouterr()
            assert "Total: 0 providers" in captured.out

    def test_show_capabilities_all_with_provider_errors(self, capsys):
        """Test showing all capabilities with some provider errors."""
        mock_config = MagicMock()
        mock_config.get_all_providers.return_value = ["openai", "broken", "anthropic"]

        def get_provider_side_effect(name):
            if name == "broken":
                raise Exception("Broken provider")
            provider = MagicMock()
            provider.features = [Feature.TOOLS]
            return provider

        mock_config.get_provider.side_effect = get_provider_side_effect
        mock_config.get_api_key.return_value = None

        with patch("chuk_llm.configuration.get_config", return_value=mock_config):
            show_capabilities()

            captured = capsys.readouterr()
            assert "Provider Capabilities Overview" in captured.out
            # Should show openai and anthropic, skip broken

    def test_show_config_session_tracking_unknown(self, capsys):
        """Test showing config when session tracking status is unknown."""
        mock_current_config = {"provider": "openai"}
        mock_config_manager = MagicMock()
        mock_config_manager.get_all_providers.return_value = []
        mock_config_manager.get_api_key.return_value = None

        with patch("chuk_llm.api.config.get_current_config", return_value=mock_current_config):
            with patch("chuk_llm.configuration.get_config", return_value=mock_config_manager):
                # Simulate _SESSIONS_ENABLED not being available
                show_config()

                captured = capsys.readouterr()
                # Should handle gracefully
                assert "Current Configuration" in captured.out
