"""
Unit tests for base URL environment variable functionality.
"""

import os
from unittest.mock import MagicMock, patch

from chuk_llm.configuration.unified_config import get_config


class TestBaseURLEnvironment:
    """Test base URL environment variable functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_env = {}
        self.env_vars = [
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "TEST_PROVIDER_API_BASE",
            "CUSTOM_ENDPOINT",
        ]

        # Save original environment
        for var in self.env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self):
        """Restore original environment."""
        for var, value in self.original_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value

    def test_standard_env_patterns(self):
        """Test standard environment variable patterns."""
        config = get_config()

        # Test each standard pattern
        patterns = [
            ("OPENAI_API_BASE", "https://test1.com/v1"),
            ("OPENAI_BASE_URL", "https://test2.com/v1"),
            ("OPENAI_API_URL", "https://test3.com/v1"),
            ("OPENAI_ENDPOINT", "https://test4.com/v1"),
        ]

        for env_var, test_url in patterns:
            # Set environment variable
            os.environ[env_var] = test_url

            # Get resolved base URL
            resolved = config.get_api_base("openai")

            # Should match the environment variable
            assert resolved == test_url, f"Failed for {env_var}"

            # Clean up for next test
            del os.environ[env_var]

    def test_custom_env_variable(self):
        """Test custom environment variable via api_base_env."""
        config = get_config()

        # Register provider with custom env variable
        config.register_provider(
            name="test_custom", api_base_env="MY_CUSTOM_ENDPOINT", models=["test-model"]
        )

        # Without env var, should return configured base
        config.providers["test_custom"].api_base = "https://default.com/v1"
        resolved = config.get_api_base("test_custom")
        assert resolved == "https://default.com/v1"

        # With env var, should return env value
        os.environ["MY_CUSTOM_ENDPOINT"] = "https://custom.com/v1"
        resolved = config.get_api_base("test_custom")
        assert resolved == "https://custom.com/v1"

        # Clean up
        config.unregister_provider("test_custom")
        del os.environ["MY_CUSTOM_ENDPOINT"]

    def test_priority_order(self):
        """Test priority order of base URL resolution."""
        config = get_config()

        # Register provider with all options
        config.register_provider(
            name="test_priority",
            api_base="https://configured.com/v1",
            api_base_env="CUSTOM_PRIORITY_URL",
            models=["test-model"],
        )

        # 1. Only configured URL (lowest priority)
        resolved = config.get_api_base("test_priority")
        assert resolved == "https://configured.com/v1"

        # 2. Standard pattern overrides configured
        os.environ["TEST_PRIORITY_API_BASE"] = "https://standard.com/v1"
        resolved = config.get_api_base("test_priority")
        assert resolved == "https://standard.com/v1"

        # 3. Custom env overrides standard (highest priority)
        os.environ["CUSTOM_PRIORITY_URL"] = "https://custom.com/v1"
        resolved = config.get_api_base("test_priority")
        assert resolved == "https://custom.com/v1"

        # Clean up
        config.unregister_provider("test_priority")
        del os.environ["TEST_PRIORITY_API_BASE"]
        del os.environ["CUSTOM_PRIORITY_URL"]

    def test_runtime_api_base(self):
        """Test runtime API base storage."""
        config = get_config()

        # Register with runtime API base
        provider = config.register_provider(name="test_runtime", models=["test-model"])

        # Set runtime API base
        provider.extra = provider.extra or {}
        provider.extra["_runtime_api_base"] = "https://runtime.com/v1"

        # Should return runtime value (highest priority)
        resolved = config.get_api_base("test_runtime")
        assert resolved == "https://runtime.com/v1"

        # Even with env vars set
        os.environ["TEST_RUNTIME_API_BASE"] = "https://env.com/v1"
        resolved = config.get_api_base("test_runtime")
        assert resolved == "https://runtime.com/v1"  # Runtime still wins

        # Clean up
        config.unregister_provider("test_runtime")
        del os.environ["TEST_RUNTIME_API_BASE"]

    def test_fallback_to_configured(self):
        """Test fallback to configured base URL."""
        config = get_config()

        # Register provider with only configured base
        config.register_provider(
            name="test_fallback",
            api_base="https://fallback.com/v1",
            models=["test-model"],
        )

        # No env vars set, should return configured
        resolved = config.get_api_base("test_fallback")
        assert resolved == "https://fallback.com/v1"

        # Clean up
        config.unregister_provider("test_fallback")

    def test_none_when_no_base(self):
        """Test returning None when no base URL is available."""
        config = get_config()

        # Register provider with no base URL
        config.register_provider(name="test_no_base", models=["test-model"])

        # Should return None
        resolved = config.get_api_base("test_no_base")
        assert resolved is None

        # Clean up
        config.unregister_provider("test_no_base")

    def test_update_with_api_base_env(self):
        """Test updating provider with api_base_env."""
        config = get_config()

        # Register initial provider
        config.register_provider(
            name="test_update", api_base="https://initial.com/v1", models=["test-model"]
        )

        # Update to use env variable
        config.update_provider("test_update", api_base_env="UPDATED_BASE_URL")

        # Set env variable
        os.environ["UPDATED_BASE_URL"] = "https://updated.com/v1"

        # Should use env variable
        resolved = config.get_api_base("test_update")
        assert resolved == "https://updated.com/v1"

        # Clean up
        config.unregister_provider("test_update")
        del os.environ["UPDATED_BASE_URL"]

    def test_existing_provider_env_override(self):
        """Test that existing providers respect environment overrides."""
        config = get_config()

        # Test with OpenAI provider (should exist in config)
        os.environ["OPENAI_API_BASE"] = "https://proxy.openai.com/v1"

        resolved = config.get_api_base("openai")
        assert resolved == "https://proxy.openai.com/v1"

        # Clean up
        del os.environ["OPENAI_API_BASE"]

    def test_case_sensitivity(self):
        """Test that provider names are case-sensitive for env vars."""
        config = get_config()

        # Register provider with lowercase name
        config.register_provider(name="mytest", models=["test-model"])

        # Environment variable should use uppercase
        os.environ["MYTEST_API_BASE"] = "https://mytest.com/v1"

        resolved = config.get_api_base("mytest")
        assert resolved == "https://mytest.com/v1"

        # Clean up
        config.unregister_provider("mytest")
        del os.environ["MYTEST_API_BASE"]

    def test_multiple_providers_isolation(self):
        """Test that env vars for different providers don't interfere."""
        config = get_config()

        # Register two providers
        config.register_provider(
            name="provider1", api_base="https://p1.com/v1", models=["model1"]
        )

        config.register_provider(
            name="provider2", api_base="https://p2.com/v1", models=["model2"]
        )

        # Set env var for provider1 only
        os.environ["PROVIDER1_API_BASE"] = "https://p1-env.com/v1"

        # Provider1 should use env var
        assert config.get_api_base("provider1") == "https://p1-env.com/v1"

        # Provider2 should use configured value
        assert config.get_api_base("provider2") == "https://p2.com/v1"

        # Clean up
        config.unregister_provider("provider1")
        config.unregister_provider("provider2")
        del os.environ["PROVIDER1_API_BASE"]


class TestBaseURLIntegration:
    """Integration tests for base URL functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_env = {}

    def teardown_method(self):
        """Restore environment."""
        for var, value in self.original_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value

    def test_client_uses_env_base_url(self):
        """Test that LLM client uses environment base URL."""
        from chuk_llm.configuration.unified_config import get_config
        from chuk_llm.llm.client import get_client

        config = get_config()

        # Save original
        self.original_env["TEST_CLIENT_API_BASE"] = os.environ.get(
            "TEST_CLIENT_API_BASE"
        )

        # Register provider
        config.register_provider(
            name="test_client",
            client_class="chuk_llm.llm.providers.openai_client.OpenAILLMClient",
            models=["test-model"],
            default_model="test-model",
        )

        # Set environment base URL
        os.environ["TEST_CLIENT_API_BASE"] = "https://env-client.com/v1"

        # Mock the client class
        mock_client_instance = MagicMock()
        mock_client_class = MagicMock(return_value=mock_client_instance)

        with patch(
            "chuk_llm.llm.client._import_string", return_value=mock_client_class
        ):
            # Create client
            get_client("test_client")

            # Check that client was created with env base URL
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args[1]
            assert call_args.get("api_base") == "https://env-client.com/v1"

        # Clean up
        config.unregister_provider("test_client")

    def test_gateway_providers_env_support(self):
        """Test that gateway providers support environment base URLs."""
        config = get_config()

        # Test gateway providers
        gateways = [
            ("litellm", "LITELLM_API_BASE", "http://custom-litellm:4000"),
            ("openrouter", "OPENROUTER_API_BASE", "https://custom-router.ai/v1"),
            ("vllm", "VLLM_API_BASE", "http://gpu-cluster:8000/v1"),
            ("togetherai", "TOGETHERAI_API_BASE", "https://custom-together.ai/v1"),
        ]

        for provider_name, env_var, test_url in gateways:
            # Save original
            self.original_env[env_var] = os.environ.get(env_var)

            # Set environment variable
            os.environ[env_var] = test_url

            # Get resolved URL
            try:
                resolved = config.get_api_base(provider_name)
                assert resolved == test_url, (
                    f"{provider_name} failed: expected {test_url}, got {resolved}"
                )
            finally:
                # Clean up
                if self.original_env[env_var] is None:
                    if env_var in os.environ:
                        del os.environ[env_var]
                else:
                    os.environ[env_var] = self.original_env[env_var]
