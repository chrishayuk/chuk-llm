#!/usr/bin/env python
# diagnostics/capabilities/llm_diagnostics.py
"""
llm_diagnostics.py
=====================================================
Orchestrates capability testing across all configured LLM providers.
Uses modular components for clean separation of concerns.

Usage:
    python llm_diagnostics_refactored.py
    python llm_diagnostics_refactored.py --providers openai anthropic
    python llm_diagnostics_refactored.py --model "openai:text=gpt-4o-mini"
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import modular components
from utils.result_models import ProviderResult
from utils.display_utils import DiagnosticDisplay
from utils.test_runners import CapabilityTester
from utils.provider_configs import get_provider_config

# Optional Rich imports for progress display
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# chuk-llm imports
try:
    from chuk_llm.llm.configuration.provider_config import DEFAULTS, ProviderConfig
except ImportError as e:
    print(f"Error importing chuk_llm: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class DiagnosticRunner:
    """Main orchestrator for running LLM provider diagnostics"""
    
    def __init__(self):
        self.tester = CapabilityTester()
        self.display = DiagnosticDisplay()
        self.config = ProviderConfig()
    
    def parse_model_overrides(self, override_arg: str) -> Dict[str, Dict[str, str]]:
        """Parse model override CLI argument into structured format"""
        override_pattern = re.compile(
            r"^(?P<provider>[\w-]+)(:(?P<capability>text|vision|tools|streaming|stream_tools))?=(?P<model>.+)$", 
            re.I
        )
        
        mapping: Dict[str, Dict[str, str]] = {}
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
        self, 
        provider: str, 
        overrides: Dict[str, Dict[str, str]]
    ) -> Dict[str, str]:
        """Build model mapping for a provider, respecting CLI overrides"""
        provider_lower = provider.lower()
        provider_overrides = overrides.get(provider_lower, {})
        default_model = self.config.get_default_model(provider)
        
        capabilities = ["text", "streaming", "tools", "streaming_tools", "vision"]
        models = {}
        
        for capability in capabilities:
            # Check for specific capability override, then wildcard, then default
            models[capability] = (
                provider_overrides.get(capability) or 
                provider_overrides.get("*") or 
                default_model
            )
        
        return models
    
    def get_available_providers(self, requested_providers: List[str] = None) -> List[str]:
        """Get list of available providers, optionally filtered by request"""
        all_providers = [p for p in DEFAULTS if p != "__global__"]
        
        if not requested_providers:
            return all_providers
        
        # Filter to requested providers (case-insensitive)
        requested_lower = {p.lower() for p in requested_providers}
        filtered = [p for p in all_providers if p.lower() in requested_lower]
        
        if not filtered:
            available = ", ".join(all_providers)
            raise ValueError(f"No matching providers found. Available: {available}")
        
        return filtered
    
    async def run_single_provider(
        self, 
        provider: str, 
        models: Dict[str, str], 
        args
    ) -> ProviderResult:
        """Run all capability tests for a single provider"""
        result = ProviderResult(provider, models)
        prefix = f"[{provider}]"
        
        def tick(stage: str, success) -> None:
            """Progress callback for individual tests"""
            status_icon = {True: "✅", False: "❌", None: "—"}[success]
            print(f"{prefix} {stage:<12} {status_icon}")
        
        # Always test basic text completion
        await self.tester.test_text_completion(provider, models["text"], result, tick)
        
        # Test streaming if not skipped
        if not args.skip_streaming:
            await self.tester.test_streaming(provider, models["streaming"], result, tick)
        else:
            result.record("streaming_text", None)
            tick("stream", None)
        
        # Test function calling if not skipped
        if not args.skip_tools:
            await self.tester.test_tools(provider, models["tools"], result, tick)
            
            # Test streaming + tools if streaming not skipped
            if not args.skip_streaming:
                await self.tester.test_streaming_tools(provider, models["streaming_tools"], result, tick)
            else:
                result.record("streaming_function_call", None)
                tick("stream_tools", None)
        else:
            result.record("function_call", None)
            result.record("streaming_function_call", None)
            tick("tools", None)
            tick("stream_tools", None)
        
        # Test vision if not skipped
        if not args.skip_image:
            await self.tester.test_vision(provider, models["vision"], result, tick)
        else:
            result.record("vision", None)
            tick("vision", None)
        
        return result
    
    async def run_diagnostics(self, args) -> List[ProviderResult]:
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
            with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
                task = progress.add_task("Diagnostics", total=len(providers))
                
                for provider in providers:
                    models = self.build_provider_models(provider, overrides)
                    progress.update(task, description=f"Testing {provider}")
                    
                    result = await self.run_single_provider(provider, models, args)
                    results.append(result)
                    progress.advance(task)
        else:
            # Verbose mode or no Rich - show detailed output
            for provider in providers:
                models = self.build_provider_models(provider, overrides)
                print(f"== {provider.upper()} ==")
                
                result = await self.run_single_provider(provider, models, args)
                results.append(result)
                print()
        
        return results
    
    def print_final_summary(self, results: List[ProviderResult]) -> None:
        """Print final diagnostic summary and statistics"""
        print()
        self.display.print_summary(results)
        
        # Calculate and display overall statistics
        total_tests = sum(result.total_capabilities for result in results)
        successful_tests = sum(result.successful_capabilities for result in results)
        
        print(f"\n📊 Overall: {successful_tests}/{total_tests} tests passed")
        
        # Highlight fully-featured providers
        full_providers = [
            result.provider for result in results 
            if len(result.feature_set) >= 4
        ]
        if full_providers:
            print(f"🎯 Full-featured providers: {', '.join(full_providers)}")
        
        # Show providers with issues
        error_providers = [
            result.provider for result in results 
            if result.has_errors()
        ]
        if error_providers:
            print(f"⚠️  Providers with errors: {', '.join(error_providers)}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Provider Diagnostics - Test capabilities across providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_diagnostics_refactored.py
  python llm_diagnostics_refactored.py --providers openai anthropic
  python llm_diagnostics_refactored.py --model "openai:text=gpt-4o-mini,anthropic:vision=claude-3-5-sonnet-20241022"
  python llm_diagnostics_refactored.py --skip-tools --skip-image
        """
    )
    
    parser.add_argument(
        "--providers", 
        nargs="*", 
        help="Only test these providers (default: all configured)"
    )
    parser.add_argument(
        "--model", 
        help="Override model(s) e.g. 'openai:text=gpt-4o,ollama=llama3'"
    )
    parser.add_argument(
        "--skip-streaming", 
        action="store_true", 
        help="Skip all streaming tests"
    )
    parser.add_argument(
        "--skip-tools", 
        action="store_true", 
        help="Skip function-calling tests"
    )
    parser.add_argument(
        "--skip-image", 
        action="store_true", 
        help="Skip vision tests"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Show detailed output"
    )
    
    return parser


async def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    runner = DiagnosticRunner()
    
    try:
        results = await runner.run_diagnostics(args)
        runner.print_final_summary(results)
    except KeyboardInterrupt:
        print("\n❌ Diagnostics cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())