#!/usr/bin/env python
# diagnostics/capabilities/llm_diagnostics_main.py
"""
llm_diagnostics_main.py
======================
Main diagnostics script updated for the new chuk-llm architecture.
Integrates seamlessly with the new configuration and provider system.

Usage:
    python llm_diagnostics_main.py
    python llm_diagnostics_main.py --providers openai anthropic
    python llm_diagnostics_main.py --model "openai:text=gpt-4o-mini"
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our utilities (these should exist in utils/)
try:
    from utils.display_utils import DiagnosticDisplay
    from utils.provider_configs import get_provider_config
    from utils.result_models import ProviderResult
    from utils.test_runners import CapabilityTester
except ImportError as e:
    print(f"Warning: Could not import utilities: {e}")
    print("Creating minimal fallback implementations...")

    # Minimal fallback implementations
    class CapabilityTester:
        async def test_text_completion(self, provider, model, result, tick):
            pass

        async def test_streaming(self, provider, model, result, tick):
            pass

        async def test_tools(self, provider, model, result, tick):
            pass

        async def test_streaming_tools(self, provider, model, result, tick):
            pass

        async def test_vision(self, provider, model, result, tick):
            pass

    class DiagnosticDisplay:
        def print_live_tick(self, provider, stage, success, timing):
            pass

        def print_provider_header(self, provider):
            print(f"\n🔍 Testing {provider}")

        def print_summary(self, results):
            print(f"\n📊 Summary: {len(results)} providers tested")

        def print_quick_summary(self, results):
            print(f"Quick: {len(results)} providers")

        def print_capabilities_matrix(self, results):
            print("Matrix view not available")

    def get_provider_config(provider):
        return {}

    class ProviderResult:
        def __init__(self, provider, models):
            self.provider = provider
            self.models = models
            self.errors = {}
            self.timings = {}
            self.successful_capabilities = 0
            self.total_capabilities = 5
            self.feature_set = set()

        def record(self, capability, result):
            if result is not None:
                self.successful_capabilities += 1
                if result:
                    self.feature_set.add(capability)

        def has_errors(self):
            return len(self.errors) > 0


# chuk-llm imports - FIXED
try:
    import chuk_llm
    from chuk_llm.configuration.unified_config import get_config
    from chuk_llm.llm.client import get_client
except ImportError as e:
    print(f"Error importing chuk_llm: {e}")
    print("Make sure you're running from the project root directory")
    print("Also ensure you've added REASONING to the Feature enum in unified_config.py")
    sys.exit(1)

# Optional Rich imports for progress display
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


class DiagnosticRunner:
    """Main orchestrator for running LLM provider diagnostics"""

    def __init__(self):
        self.tester = CapabilityTester()
        self.display = DiagnosticDisplay()
        self.config_manager = get_config()

    def parse_model_overrides(self, override_arg: str) -> dict[str, dict[str, str]]:
        """Parse model override CLI argument into structured format"""
        override_pattern = re.compile(
            r"^(?P<provider>[\w-]+)(:(?P<capability>text|vision|tools|streaming|stream_tools))?=(?P<model>.+)$",
            re.I,
        )

        mapping: dict[str, dict[str, str]] = {}
        for fragment in override_arg.split(","):
            fragment = fragment.strip()
            if not fragment:
                continue

            match = override_pattern.match(fragment)
            if not match:
                raise ValueError(f"Invalid --model fragment: '{fragment}'")

            provider = match.group("provider").lower()
            capability = match.group("capability") or "*"
            model = match.group("model")

            mapping.setdefault(provider, {})[capability] = model

        return mapping

    def build_provider_models(
        self, provider: str, overrides: dict[str, dict[str, str]]
    ) -> dict[str, str]:
        """Build model mapping with intelligent capability-based selection"""
        provider_lower = provider.lower()
        provider_overrides = overrides.get(provider_lower, {})

        try:
            provider_config = self.config_manager.get_provider(provider)
            default_model = provider_config.default_model

            # Smart model selection based on capabilities
            models = {}
            capabilities = [
                "text",
                "streaming",
                "tools",
                "streaming_tools",
                "vision",
            ]  # FIXED: Define capabilities here

            for capability in capabilities:
                # Check override first
                if provider_overrides.get(capability) or provider_overrides.get("*"):
                    models[capability] = provider_overrides.get(
                        capability
                    ) or provider_overrides.get("*")
                    continue

                # Find best model for this capability
                best_model = self._find_best_model_for_capability(
                    provider_config, capability
                )
                models[capability] = best_model or default_model

            return models

        except Exception as e:
            # FIXED: Ensure capabilities is defined in all code paths
            print(f"Warning: Error building models for {provider}: {e}")
            default_model = "default"
            capabilities = [
                "text",
                "streaming",
                "tools",
                "streaming_tools",
                "vision",
            ]  # FIXED: Define here too
            return dict.fromkeys(capabilities, default_model)

    def _find_best_model_for_capability(
        self, provider_config, capability: str
    ) -> str | None:
        """Find the best model for a specific capability"""
        from chuk_llm.configuration.unified_config import Feature

        capability_map = {
            "text": Feature.STREAMING,
            "streaming": Feature.STREAMING,
            "tools": Feature.TOOLS,
            "streaming_tools": Feature.TOOLS,
            "vision": Feature.VISION,
        }

        required_feature = capability_map.get(capability)
        if not required_feature:
            return provider_config.default_model

        # Check if default model supports the capability
        if provider_config.supports_feature(
            required_feature, provider_config.default_model
        ):
            return provider_config.default_model

        # Search through available models
        for model in provider_config.models:
            if provider_config.supports_feature(required_feature, model):
                return model

        # Check model aliases
        for _alias, actual_model in provider_config.model_aliases.items():
            if provider_config.supports_feature(required_feature, actual_model):
                return actual_model

        return None

    def get_available_providers(
        self, requested_providers: list[str] = None
    ) -> list[str]:
        """Get list of available providers, optionally filtered by request"""
        try:
            all_providers = self.config_manager.get_all_providers()
        except Exception:
            # Fallback to hardcoded list if config fails
            all_providers = [
                "openai",
                "anthropic",
                "groq",
                "gemini",
                "mistral",
                "deepseek",
                "perplexity",
            ]

        if not requested_providers:
            return all_providers

        # Filter to requested providers (case-insensitive)
        requested_lower = {p.lower() for p in requested_providers}
        filtered = [p for p in all_providers if p.lower() in requested_lower]

        if not filtered:
            available = ", ".join(all_providers)
            raise ValueError(f"No matching providers found. Available: {available}")

        return filtered

    def validate_provider_config(self, provider: str) -> tuple[bool, list[str]]:
        """Validate provider configuration"""
        try:
            provider_config = self.config_manager.get_provider(provider)
            issues = []

            # Check if we can get an API key
            api_key = self.config_manager.get_api_key(provider)
            if not api_key and provider not in [
                "ollama"
            ]:  # ollama doesn't need API key
                env_var = getattr(
                    provider_config, "api_key_env", f"{provider.upper()}_API_KEY"
                )
                issues.append(
                    f"No API key found for {provider} (set {env_var} environment variable)"
                )

            # Check if client class is configured
            if not provider_config.client_class:
                issues.append(f"No client class configured for {provider}")

            # Test client creation
            try:
                get_client(provider, model=provider_config.default_model)
            except Exception as e:
                issues.append(f"Failed to create client: {str(e)}")

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Configuration error: {str(e)}"]

    async def run_single_provider(
        self, provider: str, models: dict[str, str], args
    ) -> ProviderResult:
        """Run all capability tests for a single provider"""
        result = ProviderResult(provider, models)

        # Validate configuration first
        config_valid, config_issues = self.validate_provider_config(provider)
        if not config_valid:
            if args.verbose:
                print(
                    f"❌ {provider}: Configuration issues - {', '.join(config_issues)}"
                )
                for issue in config_issues:
                    print(f"   - {issue}")
            # Still record the config issues but don't return immediately
            for issue in config_issues:
                result.errors[f"config_{len(result.errors)}"] = issue

        # Create tick function for progress feedback
        def tick(stage: str, success: bool | None) -> None:
            if args.verbose:
                timing = result.timings.get(stage, 0)
                self.display.print_live_tick(provider, stage, success, timing)

        # Always test basic text completion
        await self.tester.test_text_completion(provider, models["text"], result, tick)

        # Test streaming if not skipped
        if not args.skip_streaming:
            await self.tester.test_streaming(
                provider, models["streaming"], result, tick
            )
        else:
            result.record("streaming_text", None)
            if args.verbose:
                tick("stream", None)

        # Test function calling if not skipped
        if not args.skip_tools:
            await self.tester.test_tools(provider, models["tools"], result, tick)

            # Test streaming + tools if streaming not skipped
            if not args.skip_streaming:
                await self.tester.test_streaming_tools(
                    provider, models["streaming_tools"], result, tick
                )
            else:
                result.record("streaming_function_call", None)
                if args.verbose:
                    tick("stream_tools", None)
        else:
            result.record("function_call", None)
            result.record("streaming_function_call", None)
            if args.verbose:
                tick("tools", None)
                tick("stream_tools", None)

        # Test vision if not skipped
        if not args.skip_image:
            await self.tester.test_vision(provider, models["vision"], result, tick)
        else:
            result.record("vision", None)
            if args.verbose:
                tick("vision", None)

        return result

    async def run_diagnostics(self, args) -> list[ProviderResult]:
        """Run diagnostics for all requested providers"""
        try:
            providers = self.get_available_providers(args.providers)
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)

        # Parse model overrides if provided
        overrides = {}
        if args.model:
            try:
                overrides = self.parse_model_overrides(args.model)
            except ValueError as e:
                print(f"❌ Model override error: {e}")
                sys.exit(1)

        # Print configuration summary
        if HAS_RICH:
            summary_text = (
                f"Testing {len(providers)} provider(s): {', '.join(providers)}"
            )
            if args.skip_streaming:
                summary_text += "\n⏭️  Skipping streaming tests"
            if args.skip_tools:
                summary_text += "\n⏭️  Skipping tool tests"
            if args.skip_image:
                summary_text += "\n⏭️  Skipping vision tests"

            console.print(
                Panel(
                    summary_text,
                    title="🚀 Diagnostic Configuration",
                    border_style="blue",
                )
            )
        else:
            print(f"🚀 Testing {len(providers)} provider(s): {', '.join(providers)}")
            if args.skip_streaming:
                print("⏭️  Skipping streaming tests")
            if args.skip_tools:
                print("⏭️  Skipping tool tests")
            if args.skip_image:
                print("⏭️  Skipping vision tests")
            print()

        results = []

        # Run tests with optional progress display
        if HAS_RICH and not args.verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Testing providers...", total=len(providers))

                for provider in providers:
                    models = self.build_provider_models(provider, overrides)
                    progress.update(task, description=f"Testing {provider}")

                    result = await self.run_single_provider(provider, models, args)
                    results.append(result)
                    progress.advance(task)
        else:
            # Verbose mode or no Rich - show detailed output
            for i, provider in enumerate(providers, 1):
                models = self.build_provider_models(provider, overrides)

                if args.verbose:
                    self.display.print_provider_header(provider)
                else:
                    print(f"[{i}/{len(providers)}] Testing {provider}...")

                result = await self.run_single_provider(provider, models, args)
                results.append(result)

                if not args.verbose:
                    # Show quick result
                    score = (
                        f"{result.successful_capabilities}/{result.total_capabilities}"
                    )
                    features = ", ".join(sorted(result.feature_set)) or "none"
                    print(f"   Result: {score} capabilities, features: {features}")

        return results

    def print_final_summary(self, results: list[ProviderResult]) -> None:
        """Print final diagnostic summary and statistics"""
        print()

        # Print the main summary
        self.display.print_summary(results)

        # Calculate and display overall statistics
        total_tests = sum(result.total_capabilities for result in results)
        successful_tests = sum(result.successful_capabilities for result in results)

        if HAS_RICH:
            stats_text = f"📊 Overall: {successful_tests}/{total_tests} tests passed"

            # Highlight fully-featured providers
            full_providers = [
                result.provider for result in results if len(result.feature_set) >= 4
            ]
            if full_providers:
                stats_text += (
                    f"\n🎯 Full-featured providers: {', '.join(full_providers)}"
                )

            # Show providers with issues
            error_providers = [
                result.provider for result in results if result.has_errors()
            ]
            if error_providers:
                stats_text += (
                    f"\n⚠️  Providers with errors: {', '.join(error_providers)}"
                )

            console.print(
                Panel(stats_text, title="📊 Final Statistics", border_style="green")
            )
        else:
            print(f"📊 Overall: {successful_tests}/{total_tests} tests passed")

            # Highlight fully-featured providers
            full_providers = [
                result.provider for result in results if len(result.feature_set) >= 4
            ]
            if full_providers:
                print(f"🎯 Full-featured providers: {', '.join(full_providers)}")

            # Show providers with issues
            error_providers = [
                result.provider for result in results if result.has_errors()
            ]
            if error_providers:
                print(f"⚠️  Providers with errors: {', '.join(error_providers)}")


# Simple test function for basic validation
async def test_basic_chuk_llm():
    """Test basic ChukLLM functionality"""
    print("🧪 Testing basic ChukLLM functionality...")

    try:
        # Test configuration loading
        config = get_config()
        providers = config.get_all_providers()
        print(f"✅ Configuration loaded: {len(providers)} providers found")

        # Test provider access
        for provider in providers[:3]:  # Test first 3 providers
            try:
                provider_config = config.get_provider(provider)
                print(f"✅ {provider}: {provider_config.default_model}")
            except Exception as e:
                print(f"❌ {provider}: {e}")

        # Test basic ask function
        try:
            response = await chuk_llm.ask(
                "Hello, can you respond with just 'OK'?", provider="openai"
            )
            print(f"✅ Basic ask test: {response[:50]}...")
        except Exception as e:
            print(f"❌ Basic ask test failed: {e}")

    except Exception as e:
        print(f"❌ Basic test failed: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Provider Diagnostics - Test capabilities across providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_diagnostics_main.py
  python llm_diagnostics_main.py --providers openai anthropic
  python llm_diagnostics_main.py --model "openai:text=gpt-4o-mini,anthropic:vision=claude-sonnet-4-20250514"
  python llm_diagnostics_main.py --skip-tools --skip-image
  python llm_diagnostics_main.py --quick
  python llm_diagnostics_main.py --test-basic
        """,
    )

    parser.add_argument(
        "--providers",
        nargs="*",
        help="Only test these providers (default: all configured)",
    )
    parser.add_argument(
        "--model", help="Override model(s) e.g. 'openai:text=gpt-4o,ollama=llama3'"
    )
    parser.add_argument(
        "--skip-streaming", action="store_true", help="Skip all streaming tests"
    )
    parser.add_argument(
        "--skip-tools", action="store_true", help="Skip function-calling tests"
    )
    parser.add_argument("--skip-image", action="store_true", help="Skip vision tests")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output during testing",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick summary only (minimal output)"
    )
    parser.add_argument(
        "--matrix", action="store_true", help="Show capabilities matrix view"
    )
    parser.add_argument(
        "--test-basic",
        action="store_true",
        help="Run basic ChukLLM functionality test only",
    )

    return parser


async def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # If --test-basic is specified, just run the basic test
    if args.test_basic:
        await test_basic_chuk_llm()
        return

    runner = DiagnosticRunner()

    try:
        start_time = time.time()
        results = await runner.run_diagnostics(args)
        total_time = time.time() - start_time

        # Print results based on requested format
        if args.quick:
            runner.display.print_quick_summary(results)
        elif args.matrix:
            runner.display.print_capabilities_matrix(results)
        else:
            runner.print_final_summary(results)

        # Always show total time
        if HAS_RICH:
            console.print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
        else:
            print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")

    except KeyboardInterrupt:
        print("\n❌ Diagnostics cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
