# benchmarks/compare_models.py
"""
Enhanced Live Model Comparison - TPS-based Performance Benchmarking
Simplified version using external JSON configuration files
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import pathlib
import sys
from pathlib import Path
from random import shuffle
from time import perf_counter
from typing import Any

# ── local imports (repo-relative) ─────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
# noqa: E402 - imports must come after sys.path modification
from benchmarks.utils.benchmark_utils import (
    calculate_sustained_tps,
    get_accurate_token_count,
    get_token_count,
)
from benchmarks.utils.config_loader import (
    get_available_suites,
    get_suite_info,
    load_test_suite,
)
from benchmarks.utils.results_display import CompellingResultsDisplay

# add (or colocate) a single source of truth for the threshold
MIN_LARGE_TOKENS = 400

# ── keep the console tidy ─────────────────────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ── Validation utilities ──────────────────────────────────────────────────────────
def validate_response_quality(
    response_text: str, test_type: str, expected_min_tokens: int = 50
) -> tuple[str, list[str]]:
    """Validate response quality with reasonable thresholds"""
    issues = []

    if not response_text or not response_text.strip():
        return "failed", ["Empty response"]

    text = response_text.strip()
    word_count = len(text.split())

    # Speed tests - be lenient
    if test_type == "speed":
        if word_count < 5:
            issues.append(f"Speed test response too short: {word_count} words")
            return "truncated", issues
        completion_phrases = ["complete", "success", "finished", "done"]
        if not any(phrase in text.lower() for phrase in completion_phrases):
            issues.append("Speed test response doesn't indicate completion")
            return "good", issues
        return "excellent", []

    # For longer tests - only flag if obviously too short
    if word_count < expected_min_tokens * 0.3:
        issues.append(
            f"Response significantly too short: {word_count} words < {expected_min_tokens}"
        )
        return "failed", issues

    # Detect obvious truncation
    obvious_truncation = [
        text.endswith("...") or text.endswith(".."),
        len(text) > 50 and len(text.split()[-10:]) < 3,
        any(
            text.lower().strip().endswith(pattern)
            for pattern in [" and then", " but then", " however"]
        ),
    ]

    if any(obvious_truncation):
        issues.append("Response appears clearly truncated")
        return "truncated", issues

    # Test-specific quality checks
    if test_type == "creative" and word_count < 100:
        issues.append("Creative response very brief")
        return "poor", issues
    elif test_type == "math" and word_count < 50:
        issues.append("Math response very brief")
        return "poor", issues

    # Length-based quality assessment
    if word_count >= expected_min_tokens:
        return "excellent", []
    elif word_count >= expected_min_tokens * 0.7:
        return "good", []
    elif word_count >= expected_min_tokens * 0.5:
        return "good", ["Response shorter than expected but acceptable"]
    else:
        return "poor", ["Response significantly shorter than expected"]


def validate_timing_metrics(
    total_time: float,
    first_token_time: float | None,
    token_count: int,
    test_type: str,
) -> list[str]:
    """Validate timing metrics for reasonableness"""
    issues = []

    if total_time <= 0:
        issues.append("Invalid total time")

    if first_token_time is not None:
        if first_token_time <= 0:
            issues.append("Invalid first token time")
        elif first_token_time > total_time:
            issues.append("First token time > total time")

    # Performance checks
    if total_time > 0 and token_count > 0:
        tokens_per_second = token_count / total_time
        if tokens_per_second > 500:
            issues.append(
                f"Suspiciously high throughput: {tokens_per_second:.0f} tok/s"
            )
        elif tokens_per_second < 1 and test_type != "speed":
            issues.append(f"Suspiciously low throughput: {tokens_per_second:.1f} tok/s")

    if test_type in ["creative", "math"] and total_time < 0.5 and token_count > 100:
        issues.append("Suspiciously fast for substantial response")

    return issues


# ══════════════════════════════════════════════════════════════════════════════════
# Simplified Live benchmark runner
# ══════════════════════════════════════════════════════════════════════════════════
class LiveBenchmarkRunner:
    """Simplified benchmark runner using external config files"""

    def __init__(self) -> None:
        self.display: CompellingResultsDisplay | None = None
        self.client_cache: dict[tuple[str, str], Any] = {}
        self.validation_stats = {
            "total_tests": 0,
            "quality_distribution": {},
            "common_issues": {},
            "suspicious_results": [],
        }

    async def get_client(self, provider: str, model: str):
        """Get or create a cached client"""
        key = (provider, model)
        if key not in self.client_cache:
            from chuk_llm.llm.client import get_client

            try:
                self.client_cache[key] = get_client(provider=provider, model=model)
            except Exception as e:
                log.error(f"Failed to create client for {provider}:{model} - {e}")
                raise
        return self.client_cache[key]

    def _adjust_config_for_model(
        self, config: dict[str, Any], model: str
    ) -> dict[str, Any]:
        """Adjust test configuration for specific model quirks"""
        adjusted_config = config.copy()

        # GPT-3.5-turbo specific adjustments
        if "gpt-3.5" in model.lower() and config["name"] == "math":
            adjusted_config["messages"] = [
                {
                    "role": "user",
                    "content": (
                        "Please solve this step by step using text explanation only: "
                        "An investment of $10,000 grows at 5% annual interest rate, "
                        "compounded monthly for 10 years. Show your mathematical "
                        "work and provide the final dollar amount. Write out all steps."
                    ),
                }
            ]

        return adjusted_config

    async def warmup_model(self, provider: str, model: str) -> None:
        """Warm up model to avoid cold-start effects"""
        try:
            client = await self.get_client(provider, model)
            await client.create_completion(
                [{"role": "user", "content": "ping"}], max_tokens=1, temperature=0
            )
            print(f"    🔥 {model} warmed up")
        except Exception as exc:
            print(f"    ⚠️  {model} warm-up failed: {exc}")

    async def run_benchmark_battle(
        self,
        provider: str,
        models: list[str],
        test_suite: str = "quick",
        runs_per_model: int = 3,
        enable_validation: bool = True,
    ) -> None:
        """Run the benchmark battle using external test configurations"""

        # Load test configurations from JSON
        try:
            test_configs = load_test_suite(test_suite)
            suite_info = get_suite_info(test_suite)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            available = get_available_suites()
            print(f"Available test suites: {', '.join(available)}")
            return
        except Exception as e:
            print(f"❌ Error loading test suite: {e}")
            return

        self.display = CompellingResultsDisplay(models)

        print(f"🔥 BENCHMARK BATTLE: {provider.upper()}")
        print(f"⚔️  Contenders: {' vs '.join(models)}")
        print(f"🎯 Challenge: {suite_info['description']}")
        print(
            f"📋 Tests: {len(test_configs)} tests × {runs_per_model} runs = {len(test_configs) * runs_per_model} per model"
        )

        if enable_validation:
            print(
                "🏆 Victory Condition: HIGHEST AVERAGE THROUGHPUT (TOKENS PER SECOND)!"
            )
            print("📊 Enhanced failure detection excludes broken models from rankings")
        else:
            print("🏆 Victory Condition: THROUGHPUT-FOCUSED PERFORMANCE!")

        print("\n🔥 Warming up models...")
        for model in models:
            await self.warmup_model(provider, model)
        await asyncio.sleep(1)

        self.display.display_live_standings()

        # Create randomized test schedule
        test_schedule = self._create_test_schedule(models, test_configs, runs_per_model)

        print(f"\n🎲 Randomized schedule: {len(test_schedule)} total tests")
        print("🚀 Focus on measuring sustained throughput (tokens per second)\n")

        # Execute tests
        model_test_counts = dict.fromkeys(models, 0)

        for i, schedule_item in enumerate(test_schedule, 1):
            model = schedule_item["model"]
            test_config = schedule_item["test_config"]
            test_id = schedule_item["test_id"]

            model_test_counts[model] += 1

            print(
                f"  🥊 {test_id} • {model} • {test_config['description']}...",
                end=" ",
                flush=True,
            )

            result = await self._execute_test(
                provider, model, test_config, enable_validation
            )

            self._print_test_result(result, enable_validation)

            # Update display
            self.display.update_model(
                model,
                test_config["name"],
                result["total_time"],
                result.get("first_token_time"),
                result.get("end_to_end_tps"),
                result.get("sustained_tps"),
                result.get("display_tokens", 0),
                result.get("extra_metrics", {}),
                result["success"] and result.get("quality_valid", True),
                model_test_counts[model],
                len(test_configs) * runs_per_model,
            )

            if enable_validation:
                self._update_validation_stats(result)

            # Show leaderboard periodically
            if i % (len(models) * len(test_configs)) == 0:
                round_num = i // (len(models) * len(test_configs))
                print(f"\n📊 Round {round_num} complete!")
                self.display.display_live_standings()
                print()

            await asyncio.sleep(0.3)  # Rate limiting

        print(f"\n🏁 All {len(models)} models completed!")

        # Save and show results
        self._save_results(provider, models, test_suite, enable_validation)
        self.display.show_final_battle_results()

        if enable_validation:
            self._show_validation_summary()

    def _create_test_schedule(
        self, models: list[str], test_configs: list[dict[str, Any]], runs_per_model: int
    ) -> list[dict[str, Any]]:
        """Create randomized test schedule for fair execution"""
        test_schedule = []

        for round_num in range(runs_per_model):
            round_tests = test_configs[:]
            shuffle(round_tests)

            for test_config in round_tests:
                round_models = models[:]
                shuffle(round_models)

                for model in round_models:
                    test_schedule.append(
                        {
                            "model": model,
                            "test_config": test_config,
                            "round": round_num + 1,
                            "test_id": f"R{round_num + 1}_{test_config['name']}",
                        }
                    )

        return test_schedule

    async def _execute_test(
        self, provider: str, model: str, config: dict[str, Any], enable_validation: bool
    ) -> dict[str, Any]:
        """
        Stream one test, calculate TPS, and return a rich result dict.
        A run counts as quality-valid when:
            • validator says “excellent” or “good”, OR
            • validator says “truncated” *and* token_count ≥ MIN_LARGE_TOKENS
        """

        try:
            # ── setup ──────────────────────────────────────────────────────────
            client = await self.get_client(provider, model)
            config = self._adjust_config_for_model(config, model)

            kwargs = {}
            if config.get("max_tokens"):
                kwargs["max_tokens"] = config["max_tokens"]
            if config.get("temperature") is not None:
                kwargs["temperature"] = config["temperature"]
            if "gpt-3.5" in model.lower():
                kwargs["tool_choice"] = "none"

            # ── stream ─────────────────────────────────────────────────────────
            start = perf_counter()
            first_tok = None
            full_response = ""
            cumulative_timeline = []
            cum_tokens = 0
            chunk_count = 0
            meaningful_chunks = 0
            error_encountered = False

            try:
                async for chunk in client.create_completion(
                    config["messages"], tools=None, stream=True, **kwargs
                ):
                    now = perf_counter()
                    chunk_count += 1

                    if chunk.get("error"):
                        error_encountered = True
                        break

                    chunk_text = chunk.get("response", "")
                    if not chunk_text:
                        continue

                    # first-token latency
                    if first_tok is None:
                        first_tok = now - start

                    meaningful_chunks += 1
                    full_response += chunk_text

                    # token accounting
                    delta = (
                        get_accurate_token_count
                        if enable_validation
                        else get_token_count
                    )(chunk_text, model)
                    cum_tokens += delta
                    cumulative_timeline.append((now - start, cum_tokens))

                    # 2-min safety timeout
                    if now - start > 120:
                        break

            except Exception as se:
                error_encountered = True
                log.error(f"Streaming error for {model}: {se}")

            # ── basic metrics ──────────────────────────────────────────────────
            total_time = perf_counter() - start
            first_tok = first_tok or 0.0
            response_length = len(full_response.strip())

            # failure heuristics unrelated to validation
            is_failed = (
                error_encountered
                or response_length < 20
                or meaningful_chunks == 0
                or (total_time < 1.0 and response_length < 50)
                or (total_time > 0.5 and cum_tokens == 0)
                or (
                    config.get("test_type") in ["creative", "math"]
                    and response_length < 100
                    and total_time < 2.0
                )
            )

            if is_failed:
                reasons = []
                if error_encountered:
                    reasons.append("streaming error")
                if response_length < 20:
                    reasons.append(f"insufficient content ({response_length} chars)")
                if meaningful_chunks == 0:
                    reasons.append("no meaningful chunks")
                if total_time < 1.0 and response_length < 50:
                    reasons.append("early termination")

                return {
                    "success": False,
                    "total_time": total_time,
                    "first_token_time": first_tok,
                    "end_to_end_tps": None,
                    "sustained_tps": None,
                    "stream_tokens": cum_tokens,
                    "accurate_tokens": 0,
                    "display_tokens": 0,
                    "response_text": full_response,
                    "model": model,
                    "test_name": config["name"],
                    "error": f"Test failure: {', '.join(reasons)}",
                    "debug_info": {
                        "chunk_count": chunk_count,
                        "meaningful_chunks": meaningful_chunks,
                        "response_length": response_length,
                    },
                }

            # ── throughput metrics ─────────────────────────────────────────────
            e2e_tps = cum_tokens / total_time if total_time > 0 else 0
            sust_tps = calculate_sustained_tps(cumulative_timeline)

            accurate_tokens = (
                get_accurate_token_count(full_response, model)
                if enable_validation
                else cum_tokens
            )

            result = {
                "success": True,
                "total_time": total_time,
                "first_token_time": first_tok,
                "end_to_end_tps": e2e_tps,
                "sustained_tps": sust_tps,
                "stream_tokens": cum_tokens,
                "accurate_tokens": accurate_tokens,
                "display_tokens": accurate_tokens,
                "response_text": full_response,
                "model": model,
                "test_name": config["name"],
            }

            # ── optional validation ───────────────────────────────────────────
            if enable_validation:
                quality, issues = validate_response_quality(
                    full_response,
                    config.get("test_type", "unknown"),
                    config.get("expected_min_tokens", 50),
                )

                timing_issues = validate_timing_metrics(
                    total_time,
                    first_tok,
                    accurate_tokens,
                    config.get("test_type", "unknown"),
                )

                result.update(
                    {
                        "quality": quality,
                        "issues": issues,
                        "timing_issues": timing_issues,
                        "quality_valid": (
                            quality in ["excellent", "good"]
                            or (
                                quality == "truncated"
                                and accurate_tokens >= MIN_LARGE_TOKENS
                            )
                        ),
                        "token_accuracy": (
                            abs(accurate_tokens - cum_tokens) / max(accurate_tokens, 1)
                            if accurate_tokens
                            else 0
                        ),
                    }
                )

            return result

        # ── outer exception guard ─────────────────────────────────────────────
        except Exception as exc:
            elapsed = perf_counter() - start if "start" in locals() else 0
            log.error(f"Exception for {model}: {exc}")
            return {
                "success": False,
                "total_time": elapsed,
                "first_token_time": None,
                "end_to_end_tps": None,
                "sustained_tps": None,
                "stream_tokens": 0,
                "accurate_tokens": 0,
                "display_tokens": 0,
                "response_text": "",
                "quality": "failed",
                "issues": [f"Exception: {exc}"],
                "timing_issues": [],
                "quality_valid": False,
                "error": str(exc),
                "model": model,
                "test_name": config["name"],
            }

    def _print_test_result(
        self, result: dict[str, Any], validation_enabled: bool
    ) -> None:
        """Print test result with TPS emphasis"""

        if result["success"]:
            if validation_enabled and result.get("quality"):
                quality_emoji = {
                    "excellent": "✅",
                    "good": "👍",
                    "poor": "⚠️",
                    "truncated": "✂️",
                    "failed": "❌",
                }.get(result["quality"], "?")

                pieces = [f"{quality_emoji} {result['total_time']:.2f}s"]
            else:
                pieces = [f"✅ {result['total_time']:.2f}s"]

            if result.get("first_token_time"):
                pieces.append(f"🚀 {result['first_token_time']:.2f}s")

            # TPS display
            if result.get("sustained_tps"):
                tps = result["sustained_tps"]
                if tps > 100:
                    pieces.append(f"⚡ {tps:.0f}/s")
                elif tps > 50:
                    pieces.append(f"📈 {tps:.0f}/s")
                else:
                    pieces.append(f"🐌 {tps:.0f}/s")
            elif result.get("end_to_end_tps"):
                pieces.append(f"⚡ {result['end_to_end_tps']:.0f}/s")

            if result.get("display_tokens"):
                pieces.append(f"📝 {result['display_tokens']}tok")

            print(" | ".join(pieces))

            # TPS achievements
            tps = result.get("sustained_tps") or result.get("end_to_end_tps", 0)
            if tps > 150:
                print("    🚀 BLAZING THROUGHPUT! >150 tok/s!")
            elif tps > 100:
                print("    💨 Excellent throughput! >100 tok/s!")
            elif tps > 50:
                print("    📈 Good throughput! >50 tok/s")

        else:
            print(f"❌ {result.get('error', 'Unknown error')}")

    def _update_validation_stats(self, result: dict[str, Any]) -> None:
        """Update validation statistics"""
        self.validation_stats["total_tests"] += 1

        quality = result.get("quality", "unknown")
        self.validation_stats["quality_distribution"][quality] = (
            self.validation_stats["quality_distribution"].get(quality, 0) + 1
        )

        for issue in result.get("issues", []):
            self.validation_stats["common_issues"][issue] = (
                self.validation_stats["common_issues"].get(issue, 0) + 1
            )

    def _save_results(
        self, provider: str, models: list[str], suite: str, validation: bool
    ) -> None:
        """Save benchmark results to JSON"""
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        pathlib.Path("benchmark_runs").mkdir(exist_ok=True)

        results_data = {
            "timestamp": ts,
            "provider": provider,
            "models": models,
            "test_suite": suite,
            "validation_enabled": validation,
            "leadership_metric": "avg_throughput_tps",
            "results": self.display.test_results,
            "validation_summary": self.validation_stats if validation else None,
        }

        suffix = "_tps_focused" if validation else "_tps_basic"
        filename = f"benchmark_runs/{ts}_{provider}_{suite}{suffix}.json"

        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 Results saved to: {filename}")

    def _show_validation_summary(self) -> None:
        """Show validation statistics summary"""
        if self.validation_stats["total_tests"] == 0:
            return

        print("\n📊 VALIDATION SUMMARY")
        print("=" * 50)

        stats = self.validation_stats
        total = stats["total_tests"]

        # Quality distribution
        print("Quality Distribution:")
        for quality, count in sorted(stats["quality_distribution"].items()):
            percentage = count / total * 100
            emoji = {
                "excellent": "✅",
                "good": "👍",
                "poor": "⚠️",
                "truncated": "✂️",
                "failed": "❌",
            }.get(quality, "?")
            print(f"  {emoji} {quality}: {count} ({percentage:.1f}%)")

        # TPS reliability
        excellent_good = stats["quality_distribution"].get("excellent", 0) + stats[
            "quality_distribution"
        ].get("good", 0)
        quality_rate = excellent_good / total

        print("\n💡 TPS Measurement Reliability:")
        if quality_rate >= 0.8:
            print("  ✅ Throughput measurements are highly reliable")
        elif quality_rate >= 0.6:
            print("  ⚠️ Some quality issues may affect TPS accuracy")
        else:
            print("  ❌ Significant quality issues detected")


# ══════════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ══════════════════════════════════════════════════════════════════════════════════
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Model Benchmark Battle with External Test Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔥 EXTERNAL TEST CONFIGURATION:
Test suites are loaded from JSON files in benchmarks/test_configs/:
  • lightning.json - Ultra-fast 2-test sprint
  • quick.json     - Fast 3-test battle (default)
  • standard.json  - Full 4-test championship

🚀 TPS LEADERSHIP FEATURES:
  • Rankings based on highest tokens per second
  • Enhanced failure detection excludes broken models
  • Fair round-robin execution for unbiased comparison
  • Quality validation ensures meaningful measurements

⚡ CUSTOM TEST SUITES:
Create your own test suite by adding a JSON file to test_configs/
with the following structure:
{
  "name": "my_suite",
  "description": "My custom test suite",
  "tests": [
    {
      "name": "test1",
      "description": "Test description",
      "messages": [{"role": "user", "content": "..."}],
      "max_tokens": 1000,
      "temperature": 0.7,
      "expected_min_tokens": 100,
      "test_type": "creative"
    }
  ]
}

Examples:
  python compare_models.py openai "gpt-4o-mini,gpt-4o" --suite quick
  python compare_models.py gemini "gemini-1.5-flash,gemini-2.0-flash" --runs 5
  python compare_models.py anthropic "claude-3-5-sonnet-20241022" --suite lightning
        """,
    )
    parser.add_argument(
        "provider", help="Provider (openai, groq, anthropic, gemini, ollama)"
    )
    parser.add_argument("models", help="Comma-separated models to battle")
    parser.add_argument(
        "--suite",
        default="quick",
        help="Test suite name (default: quick). Use --list-suites to see available options",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Rounds per model (default: 3)"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation features (faster but less accurate)",
    )
    parser.add_argument(
        "--list-suites", action="store_true", help="List available test suites and exit"
    )

    args = parser.parse_args()

    # Handle --list-suites
    if args.list_suites:
        try:
            available = get_available_suites()
            if not available:
                print("❌ No test suites found in benchmarks/test_configs/")
                return

            print("📋 Available Test Suites:")
            print("=" * 50)

            for suite_name in available:
                try:
                    info = get_suite_info(suite_name)
                    print(f"\n🎯 {suite_name}")
                    print(f"   Description: {info['description']}")
                    print(
                        f"   Tests: {info['test_count']} ({', '.join(info['test_names'])})"
                    )
                except Exception as e:
                    print(f"\n❌ {suite_name}: Error loading ({e})")

        except Exception as e:
            print(f"❌ Error listing suites: {e}")
        return

    # Validate arguments
    models = [m.strip() for m in args.models.split(",")]
    if len(models) < 2:
        print("❌ Need at least 2 models for a proper benchmark battle!")
        return

    # Run benchmark
    runner = LiveBenchmarkRunner()
    await runner.run_benchmark_battle(
        provider=args.provider,
        models=models,
        test_suite=args.suite,
        runs_per_model=args.runs,
        enable_validation=not args.no_validation,
    )


if __name__ == "__main__":
    asyncio.run(main())
