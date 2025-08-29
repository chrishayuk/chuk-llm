#!/usr/bin/env python3
"""
Gateway Providers Diagnostic
============================

This diagnostic tests the configuration and connectivity of gateway providers
(LiteLLM, OpenRouter, vLLM, Together AI, etc.)
"""

import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_llm import ask_sync
from chuk_llm.configuration.unified_config import get_config
from chuk_llm.llm.client import create_client


class GatewayProviderDiagnostic:
    """Diagnostic for gateway provider configurations."""

    def __init__(self):
        self.config = get_config()
        self.test_results: list[tuple[str, bool, str]] = []
        self.gateway_providers = [
            "litellm",
            "openrouter",
            "vllm",
            "togetherai",
            "openai_compatible",
        ]

    def add_result(self, test_name: str, passed: bool, details: str):
        """Add a test result."""
        self.test_results.append((test_name, passed, details))
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if details:
            print(f"         {details}")

    def test_provider_config(self, provider_name: str):
        """Test if a gateway provider is properly configured."""
        print(f"\nTesting {provider_name.upper()}")
        print("-" * 40)

        try:
            # Get provider configuration
            provider_config = self.config.get_provider(provider_name)

            # Check basic configuration
            if provider_config.client_class:
                self.add_result(
                    f"{provider_name} client class",
                    True,
                    f"Using {provider_config.client_class.split('.')[-1]}",
                )
            else:
                self.add_result(
                    f"{provider_name} client class", False, "No client class configured"
                )

            # Check API base
            api_base = self.config.get_api_base(provider_name)
            if api_base:
                self.add_result(
                    f"{provider_name} API base", True, f"Configured: {api_base}"
                )
            else:
                self.add_result(
                    f"{provider_name} API base", False, "No API base configured"
                )

            # Check API key configuration
            api_key = self.config.get_api_key(provider_name)
            env_var = provider_config.api_key_env

            if api_key:
                self.add_result(
                    f"{provider_name} API key", True, f"Found via {env_var}"
                )
            else:
                self.add_result(
                    f"{provider_name} API key", False, f"Not set (export {env_var}=...)"
                )

            # Check models
            if provider_config.models:
                model_count = len([m for m in provider_config.models if m != "*"])
                wildcard = "*" in provider_config.models
                details = f"{model_count} specific models"
                if wildcard:
                    details += " + wildcard (*)"
                self.add_result(f"{provider_name} models", True, details)
            else:
                self.add_result(
                    f"{provider_name} models", False, "No models configured"
                )

            # Check features
            if provider_config.features:
                self.add_result(
                    f"{provider_name} features",
                    True,
                    f"{len(provider_config.features)} features supported",
                )

            return True

        except Exception as e:
            self.add_result(f"{provider_name} configuration", False, f"Error: {e}")
            return False

    def test_environment_overrides(self):
        """Test environment variable overrides for gateways."""
        print("\nTesting Environment Overrides")
        print("-" * 40)

        test_cases = [
            ("LITELLM_API_BASE", "litellm", "http://custom-litellm:4000"),
            ("OPENROUTER_API_BASE", "openrouter", "https://custom-router.ai/v1"),
            ("VLLM_API_BASE", "vllm", "http://gpu-server:8000/v1"),
            ("TOGETHERAI_API_BASE", "togetherai", "https://custom-together.ai/v1"),
        ]

        original_values = {}

        for env_var, provider, test_url in test_cases:
            # Save original value
            original_values[env_var] = os.getenv(env_var)

            # Set test value
            os.environ[env_var] = test_url

            # Check if it's picked up
            try:
                resolved = self.config.get_api_base(provider)
                if resolved == test_url:
                    self.add_result(
                        f"{env_var} override", True, f"Correctly resolved to {test_url}"
                    )
                else:
                    self.add_result(
                        f"{env_var} override",
                        False,
                        f"Expected {test_url}, got {resolved}",
                    )
            except Exception as e:
                self.add_result(f"{env_var} override", False, f"Error: {e}")

            # Restore original value
            if original_values[env_var] is None:
                del os.environ[env_var]
            else:
                os.environ[env_var] = original_values[env_var]

    def test_connectivity(self, provider_name: str):
        """Test actual connectivity to a gateway provider."""
        print(f"\nTesting {provider_name.upper()} Connectivity")
        print("-" * 40)

        # Skip if no API key
        api_key = self.config.get_api_key(provider_name)
        if not api_key:
            print("  ⚠️  Skipping: No API key configured")
            return

        try:
            # Try to create a client
            create_client(provider_name)
            self.add_result(
                f"{provider_name} client creation", True, "Client created successfully"
            )

            # Try a simple request
            response = ask_sync(
                "Say 'OK' in one word",
                provider=provider_name,
                max_tokens=5,
                temperature=0,
            )

            if response:
                self.add_result(
                    f"{provider_name} inference", True, f"Response: {response[:20]}..."
                )
            else:
                self.add_result(f"{provider_name} inference", False, "Empty response")

        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg:
                self.add_result(
                    f"{provider_name} connectivity", False, "Service not running"
                )
            elif "401" in error_msg or "Unauthorized" in error_msg:
                self.add_result(
                    f"{provider_name} connectivity", False, "Invalid API key"
                )
            else:
                self.add_result(
                    f"{provider_name} connectivity",
                    False,
                    f"Error: {error_msg[:50]}...",
                )

    def test_model_routing(self):
        """Test model routing for gateway providers."""
        print("\nTesting Model Routing")
        print("-" * 40)

        routing_examples = {
            "litellm": ["gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"],
            "openrouter": ["openai/gpt-3.5-turbo", "anthropic/claude-3-opus"],
            "togetherai": [
                "deepseek-ai/deepseek-v3",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            ],
        }

        for provider, models in routing_examples.items():
            try:
                provider_config = self.config.get_provider(provider)

                for model in models:
                    # Check if model is accepted
                    if "*" in provider_config.models or model in provider_config.models:
                        self.add_result(
                            f"{provider} accepts {model}",
                            True,
                            "Model routing configured",
                        )
                    else:
                        self.add_result(
                            f"{provider} accepts {model}",
                            False,
                            "Model not in configured list",
                        )

            except Exception as e:
                self.add_result(f"{provider} model routing", False, f"Error: {e}")

    def print_setup_instructions(self):
        """Print setup instructions for each gateway."""
        print("\n" + "=" * 60)
        print("GATEWAY SETUP INSTRUCTIONS")
        print("=" * 60)

        instructions = {
            "litellm": """
LiteLLM Setup:
1. Install: pip install litellm
2. Create config.yaml with your providers
3. Start: litellm --config config.yaml --port 4000
4. Set: export LITELLM_API_KEY=your-key
5. (Optional) export LITELLM_API_BASE=http://your-server:4000
""",
            "openrouter": """
OpenRouter Setup:
1. Sign up at https://openrouter.ai
2. Get API key from https://openrouter.ai/keys
3. Set: export OPENROUTER_API_KEY=sk-or-...
4. Use models with provider prefix: openai/gpt-4
""",
            "vllm": """
vLLM Setup:
1. Install: pip install vllm
2. Start server:
   python -m vllm.entrypoints.openai.api_server \\
     --model meta-llama/Llama-3-8b-hf \\
     --port 8000
3. (Optional) export VLLM_API_BASE=http://localhost:8000/v1
""",
            "togetherai": """
Together AI Setup:
1. Sign up at https://api.together.xyz
2. Get API key from https://api.together.xyz/settings/api-keys
3. Set: export TOGETHER_API_KEY=...
4. Use models like: deepseek-ai/deepseek-v3
""",
            "openai_compatible": """
OpenAI Compatible Setup:
For any OpenAI-compatible service:
1. Set: export OPENAI_COMPATIBLE_API_BASE=http://your-service/v1
2. Set: export OPENAI_COMPATIBLE_API_KEY=your-key
3. Works with LocalAI, FastChat, LM Studio, etc.
""",
        }

        for provider, instruction in instructions.items():
            print(f"\n{provider.upper()}:")
            print(instruction)

    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("\n" + "=" * 60)
        print("GATEWAY PROVIDERS DIAGNOSTIC")
        print("=" * 60)

        # Test each gateway provider configuration
        for provider in self.gateway_providers:
            self.test_provider_config(provider)

        # Test environment overrides
        self.test_environment_overrides()

        # Test model routing
        self.test_model_routing()

        # Test connectivity for configured providers
        print("\n" + "=" * 60)
        print("CONNECTIVITY TESTS")
        print("=" * 60)

        for provider in self.gateway_providers:
            if self.config.get_api_key(provider):
                self.test_connectivity(provider)

        # Summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, p, _ in self.test_results if p)
        total = len(self.test_results)

        print(f"\nTests Passed: {passed}/{total}")

        if passed == total:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n⚠️  SOME TESTS FAILED OR SKIPPED")

            # Group failures by provider
            failures_by_provider = {}
            for name, passed, details in self.test_results:
                if not passed:
                    provider = name.split()[0]
                    if provider not in failures_by_provider:
                        failures_by_provider[provider] = []
                    failures_by_provider[provider].append((name, details))

            if failures_by_provider:
                print("\nFailed tests by provider:")
                for provider, failures in failures_by_provider.items():
                    print(f"\n{provider}:")
                    for name, details in failures:
                        print(f"  - {name}: {details}")

        # Print setup instructions
        self.print_setup_instructions()


def main():
    """Run the diagnostic."""
    diagnostic = GatewayProviderDiagnostic()
    diagnostic.run_all_tests()


if __name__ == "__main__":
    main()
