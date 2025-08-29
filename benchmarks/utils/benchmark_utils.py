# benchmarks/utils/benchmark_utils.py
"""
Enhanced benchmarking utilities with better accuracy and validation
"""

from __future__ import annotations

import logging
import re
import statistics as _stats
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

__all__ = [
    "get_token_count",
    "get_accurate_token_count",  # Added missing function
    "approx_tokens",
    "calculate_prompt_tokens",
    "calculate_sustained_tps",
    "filter_stable_runs",
    "validate_benchmark_result",
    "ResponseQuality",
    "BenchmarkValidator",
    "ImprovedTokenCounter",
    "StreamingMetrics",
]

logger = logging.getLogger(__name__)


class ResponseQuality(Enum):
    """Response quality assessment"""

    EXCELLENT = "excellent"
    GOOD = "good"
    POOR = "poor"
    TRUNCATED = "truncated"
    FAILED = "failed"


@dataclass
class ValidationResult:
    """Result of benchmark validation"""

    is_valid: bool
    quality: ResponseQuality
    issues: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


class StreamMetrics(NamedTuple):
    """Streaming metrics container"""

    total_time: float
    first_token_time: float | None
    end_to_end_tps: float | None
    sustained_tps: float | None
    token_count: int
    chunk_count: int
    avg_chunk_interval: float | None


# ────────────────────────────────────────────────────────────────────────────
# Enhanced token counting
# ────────────────────────────────────────────────────────────────────────────


class ImprovedTokenCounter:
    """Enhanced token counting with model-specific accuracy"""

    _encodings_cache = {}

    @classmethod
    def get_token_count(cls, text: str, model: str) -> int:
        """Get accurate token count with caching and fallbacks"""
        if not text:
            return 0

        try:
            import tiktoken

            # Cache encodings for performance
            if model not in cls._encodings_cache:
                try:
                    cls._encodings_cache[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Unknown model, use cl100k_base (GPT-4 encoding)
                    cls._encodings_cache[model] = tiktoken.get_encoding("cl100k_base")

            encoding = cls._encodings_cache[model]
            return len(encoding.encode(text))

        except ImportError:
            logger.warning("tiktoken not available, using approximation")
            return cls.approx_tokens(text)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using approximation")
            return cls.approx_tokens(text)

    @staticmethod
    def approx_tokens(text: str) -> int:
        """Enhanced token approximation with better accuracy"""
        if not text:
            return 0

        # More sophisticated approximation
        # 1. Split on whitespace and count words
        words = len(text.split())

        # 2. Count punctuation separately (often separate tokens)
        punctuation = len(re.findall(r"[^\w\s]", text))

        # 3. Estimate subword tokens (longer words often split)
        long_words = len([w for w in text.split() if len(w) > 6])
        subword_estimate = long_words * 0.3  # 30% of long words split

        # 4. Factor in common prefixes/suffixes
        affixes = len(
            re.findall(r"\b(?:un|re|pre|de|over|under|out)\w+", text, re.IGNORECASE)
        )
        affix_tokens = affixes * 0.2

        return int(words + punctuation + subword_estimate + affix_tokens)

    @classmethod
    def calculate_prompt_tokens(cls, messages: list[dict], model: str) -> int:
        """Calculate tokens in prompt with message formatting overhead"""
        total = 0

        for message in messages:
            # Content tokens
            content = message.get("content", "")
            if isinstance(content, str):
                total += cls.get_token_count(content, model)
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += cls.get_token_count(item.get("text", ""), model)

            # Role and formatting overhead (roughly 3-4 tokens per message)
            total += 4

        # Chat completion overhead
        total += 3

        return total


# ────────────────────────────────────────────────────────────────────────────
# MISSING FUNCTION - This is what was causing the ImportError!
# ────────────────────────────────────────────────────────────────────────────


def get_accurate_token_count(text: str, model: str) -> int:
    """Get accurate token count with tiktoken fallback - FIXED MISSING FUNCTION"""
    if not text:
        return 0

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Unknown model, use cl100k_base (GPT-4 encoding)
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to existing method
        return get_token_count(text, model)
    except Exception as e:
        logger.warning(f"Accurate token counting failed: {e}, using fallback")
        return get_token_count(text, model)


# ────────────────────────────────────────────────────────────────────────────
# Enhanced streaming metrics
# ────────────────────────────────────────────────────────────────────────────


class StreamingMetrics:
    """Enhanced streaming metrics calculation"""

    @staticmethod
    def calculate_sustained_tps(
        timeline: list[tuple[float, int]],
        tail_fraction: float = 0.3,
        min_points: int = 10,
    ) -> float | None:
        """
        Calculate sustained tokens per second using robust linear regression

        Args:
            timeline: List of (timestamp, cumulative_tokens) pairs
            tail_fraction: Fraction of timeline to use for sustained calculation
            min_points: Minimum points needed for calculation
        """
        if len(timeline) < min_points:
            logger.debug(
                f"Insufficient points for sustained TPS: {len(timeline)} < {min_points}"
            )
            return None

        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available for sustained TPS calculation")
            return StreamingMetrics._fallback_sustained_tps(timeline, tail_fraction)

        # Use tail portion to avoid first-token latency effects
        start_idx = int(len(timeline) * (1 - tail_fraction))
        tail_timeline = timeline[start_idx:]

        if len(tail_timeline) < 5:
            return None

        times, tokens = zip(*tail_timeline, strict=False)

        # Check for valid time progression
        if max(times) - min(times) < 0.1:  # Less than 100ms range
            logger.debug("Insufficient time range for sustained TPS calculation")
            return None

        # Robust linear regression (handles outliers better)
        try:
            # Simple linear regression: slope = tokens/second
            slope, _ = np.polyfit(times, tokens, 1)

            # Validation: slope should be positive and reasonable
            if slope <= 0:
                logger.debug(f"Invalid slope for sustained TPS: {slope}")
                return None

            if slope > 1000:  # Sanity check: > 1000 tokens/sec seems unrealistic
                logger.warning(f"Suspiciously high sustained TPS: {slope}")
                return min(slope, 1000)  # Cap at 1000

            return float(slope)

        except Exception as e:
            logger.warning(f"Linear regression failed: {e}")
            return None

    @staticmethod
    def _fallback_sustained_tps(
        timeline: list[tuple[float, int]], tail_fraction: float
    ) -> float | None:
        """Fallback calculation without numpy"""
        start_idx = int(len(timeline) * (1 - tail_fraction))
        tail_timeline = timeline[start_idx:]

        if len(tail_timeline) < 3:
            return None

        start_time, start_tokens = tail_timeline[0]
        end_time, end_tokens = tail_timeline[-1]

        time_diff = end_time - start_time
        token_diff = end_tokens - start_tokens

        if time_diff <= 0 or token_diff <= 0:
            return None

        return token_diff / time_diff

    @classmethod
    def analyze_stream(
        cls, chunk_timeline: list[tuple[float, str]], model: str
    ) -> StreamMetrics:
        """
        Comprehensive streaming analysis

        Args:
            chunk_timeline: List of (timestamp, chunk_text) pairs
            model: Model name for token counting
        """
        if not chunk_timeline:
            return StreamMetrics(0, None, None, None, 0, 0, None)

        start_time = chunk_timeline[0][0]
        end_time = chunk_timeline[-1][0]
        total_time = end_time - start_time

        # Find first meaningful chunk
        first_token_time = None
        cumulative_tokens = 0
        cumulative_timeline = []

        for timestamp, chunk_text in chunk_timeline:
            if chunk_text.strip() and first_token_time is None:
                first_token_time = timestamp - start_time

            # Count tokens in this chunk
            chunk_tokens = ImprovedTokenCounter.get_token_count(chunk_text, model)
            cumulative_tokens += chunk_tokens

            cumulative_timeline.append((timestamp, cumulative_tokens))

        # Calculate metrics
        end_to_end_tps = cumulative_tokens / total_time if total_time > 0 else None
        sustained_tps = cls.calculate_sustained_tps(cumulative_timeline)

        # Chunk timing analysis
        chunk_intervals = []
        for i in range(1, len(chunk_timeline)):
            interval = chunk_timeline[i][0] - chunk_timeline[i - 1][0]
            chunk_intervals.append(interval)

        avg_chunk_interval = _stats.mean(chunk_intervals) if chunk_intervals else None

        return StreamMetrics(
            total_time=total_time,
            first_token_time=first_token_time,
            end_to_end_tps=end_to_end_tps,
            sustained_tps=sustained_tps,
            token_count=cumulative_tokens,
            chunk_count=len(chunk_timeline),
            avg_chunk_interval=avg_chunk_interval,
        )


# ────────────────────────────────────────────────────────────────────────────
# Enhanced validation
# ────────────────────────────────────────────────────────────────────────────


class BenchmarkValidator:
    """Validates benchmark results for accuracy and fairness"""

    @staticmethod
    def validate_response_quality(
        response_text: str, test_type: str, expected_min_tokens: int = 50
    ) -> tuple[ResponseQuality, list[str]]:
        """Assess response quality and detect issues"""
        issues = []

        if not response_text or not response_text.strip():
            return ResponseQuality.FAILED, ["Empty response"]

        text = response_text.strip()
        token_count = ImprovedTokenCounter.approx_tokens(text)

        # Check length appropriateness
        if token_count < expected_min_tokens and test_type in [
            "creative",
            "math",
            "reasoning",
        ]:
            issues.append(
                f"Response too short: {token_count} tokens < {expected_min_tokens}"
            )
            if token_count < 20:
                return ResponseQuality.FAILED, issues
            else:
                return ResponseQuality.TRUNCATED, issues

        # Check for truncation patterns
        truncation_indicators = [
            not text.endswith((".", "!", "?", '"', "'", ")", "]", "}", "\n")),
            text.endswith("..."),
            text.endswith(" and"),
            text.endswith(" or"),
            text.endswith(" but"),
            len(text) > 10 and text[-10:].count(" ") < 2,  # Ends mid-word
        ]

        if any(truncation_indicators):
            issues.append("Response appears truncated")
            return ResponseQuality.TRUNCATED, issues

        # Test-specific quality checks
        if test_type == "creative":
            if token_count < 100:
                issues.append("Creative response unusually brief")
                return ResponseQuality.POOR, issues

            # Check for narrative elements
            narrative_words = ["story", "character", "scene", "plot", "narrative"]
            if not any(word in text.lower() for word in narrative_words):
                issues.append("Creative response lacks narrative elements")

        elif test_type == "math":
            # Check for mathematical reasoning
            math_indicators = [
                "answer",
                "solution",
                "result",
                "therefore",
                "thus",
                "=",
                "calculate",
            ]
            if not any(indicator in text.lower() for indicator in math_indicators):
                issues.append("Math response lacks clear solution indicators")
                return ResponseQuality.POOR, issues

        elif test_type == "speed":
            # Speed tests should be brief but complete
            if token_count > 100:
                issues.append("Speed test response unnecessarily verbose")

        # Overall quality assessment
        if not issues:
            return ResponseQuality.EXCELLENT, []
        elif len(issues) == 1 and "unusually brief" in issues[0]:
            return ResponseQuality.GOOD, issues
        else:
            return ResponseQuality.POOR, issues

    @staticmethod
    def validate_timing_metrics(
        total_time: float,
        first_token_time: float | None,
        token_count: int,
        test_type: str,
    ) -> list[str]:
        """Validate timing metrics for reasonableness"""
        issues = []

        # Basic sanity checks
        if total_time <= 0:
            issues.append("Invalid total time <= 0")

        if first_token_time is not None:
            if first_token_time <= 0:
                issues.append("Invalid first token time <= 0")
            elif first_token_time > total_time:
                issues.append("First token time > total time")

        # Performance reasonableness checks
        if total_time > 0 and token_count > 0:
            tokens_per_second = token_count / total_time

            if tokens_per_second > 500:
                issues.append(
                    f"Suspiciously high throughput: {tokens_per_second:.0f} tok/s"
                )
            elif tokens_per_second < 1 and test_type != "speed":
                issues.append(
                    f"Suspiciously low throughput: {tokens_per_second:.1f} tok/s"
                )

        # Test-specific timing expectations
        if test_type == "speed":
            if total_time > 5.0:
                issues.append("Speed test took too long")
        elif (
            test_type in ["creative", "math"] and total_time < 0.5 and token_count > 100
        ):
            issues.append("Suspiciously fast for substantial response")

        return issues

    @classmethod
    def validate_benchmark_result(
        cls,
        response_text: str,
        total_time: float,
        first_token_time: float | None,
        token_count: int,
        test_type: str,
        model: str,
        expected_min_tokens: int = 50,
    ) -> ValidationResult:
        """Comprehensive validation of a benchmark result"""
        all_issues = []
        all_warnings = []

        # Validate response quality
        quality, quality_issues = cls.validate_response_quality(
            response_text, test_type, expected_min_tokens
        )
        all_issues.extend(quality_issues)

        # Validate timing
        timing_issues = cls.validate_timing_metrics(
            total_time, first_token_time, token_count, test_type
        )
        all_issues.extend(timing_issues)

        # Calculate additional metrics
        actual_tokens = ImprovedTokenCounter.get_token_count(response_text, model)
        token_accuracy = abs(actual_tokens - token_count) / max(actual_tokens, 1)

        if token_accuracy > 0.2:  # 20% difference
            all_warnings.append(
                f"Token count discrepancy: estimated {token_count}, actual {actual_tokens}"
            )

        metrics = {
            "actual_tokens": actual_tokens,
            "estimated_tokens": token_count,
            "token_accuracy": 1 - token_accuracy,
            "tokens_per_second": actual_tokens / total_time if total_time > 0 else None,
            "response_length": len(response_text),
            "first_token_latency": first_token_time,
        }

        # Overall validity
        is_valid = (
            quality not in [ResponseQuality.FAILED, ResponseQuality.TRUNCATED]
            and len(
                [
                    issue
                    for issue in all_issues
                    if "Suspiciously" in issue or "Invalid" in issue
                ]
            )
            == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            quality=quality,
            issues=all_issues,
            warnings=all_warnings,
            metrics=metrics,
        )


# ────────────────────────────────────────────────────────────────────────────
# Enhanced utility functions
# ────────────────────────────────────────────────────────────────────────────


def filter_stable_runs(
    run_times: list[float], variation: float = 0.15, min_runs: int = 2
) -> list[float]:
    """Enhanced outlier filtering with better statistics"""
    if len(run_times) < min_runs:
        return run_times

    if len(run_times) <= 3:
        # For small samples, just remove extreme outliers
        median = _stats.median(run_times)
        mad = _stats.median(
            [abs(x - median) for x in run_times]
        )  # Median Absolute Deviation
        threshold = 3 * mad  # 3-MAD rule

        if threshold > 0:
            return [t for t in run_times if abs(t - median) <= threshold]
        else:
            return run_times

    # For larger samples, use the original method
    median = _stats.median(run_times)
    low, high = median * (1 - variation), median * (1 + variation)
    filtered = [t for t in run_times if low <= t <= high]

    # Ensure we don't filter too aggressively
    if len(filtered) < len(run_times) * 0.5:  # Don't remove more than 50%
        return run_times

    return filtered


# Backwards compatibility aliases
get_token_count = ImprovedTokenCounter.get_token_count
approx_tokens = ImprovedTokenCounter.approx_tokens
calculate_prompt_tokens = ImprovedTokenCounter.calculate_prompt_tokens
calculate_sustained_tps = StreamingMetrics.calculate_sustained_tps
validate_benchmark_result = BenchmarkValidator.validate_benchmark_result

# ────────────────────────────────────────────────────────────────────────────
# Integration helper for existing benchmark code
# ────────────────────────────────────────────────────────────────────────────


class EnhancedBenchmarkRunner:
    """Helper class to integrate improved utilities with existing benchmark code"""

    def __init__(self, models: list[str]):
        self.models = models
        self.validator = BenchmarkValidator()
        self.token_counter = ImprovedTokenCounter()
        self.results_log = []

    async def run_validated_test(
        self, client, model: str, test_config: dict[str, Any], run_number: int
    ) -> dict[str, Any]:
        """Run a test with comprehensive validation"""
        start_time = time.time()
        chunk_timeline = []

        try:
            # Stream the response and collect timing data
            full_response = ""
            async for chunk in client.create_completion(
                test_config["messages"],
                tools=test_config.get("tools"),
                stream=True,
                max_tokens=test_config.get("max_tokens", 1000),
                temperature=test_config.get("temperature", 0.7),
            ):
                timestamp = time.time()
                chunk_text = chunk.get("response", "")

                if chunk_text:
                    chunk_timeline.append((timestamp, chunk_text))
                    full_response += chunk_text

            # Analyze streaming metrics
            stream_metrics = StreamingMetrics.analyze_stream(chunk_timeline, model)

            # Validate the result
            validation = self.validator.validate_benchmark_result(
                response_text=full_response,
                total_time=stream_metrics.total_time,
                first_token_time=stream_metrics.first_token_time,
                token_count=stream_metrics.token_count,
                test_type=test_config.get("type", "unknown"),
                model=model,
                expected_min_tokens=test_config.get("expected_min_tokens", 50),
            )

            result = {
                "model": model,
                "test_name": test_config["name"],
                "run_number": run_number,
                "success": validation.is_valid,
                "response_text": full_response,
                "total_time": stream_metrics.total_time,
                "first_token_time": stream_metrics.first_token_time,
                "end_to_end_tps": stream_metrics.end_to_end_tps,
                "sustained_tps": stream_metrics.sustained_tps,
                "token_count": stream_metrics.token_count,
                "actual_tokens": validation.metrics["actual_tokens"],
                "chunk_count": stream_metrics.chunk_count,
                "quality": validation.quality.value,
                "issues": validation.issues,
                "warnings": validation.warnings,
                "validation_metrics": validation.metrics,
            }

            self.results_log.append(result)
            return result

        except Exception as e:
            error_result = {
                "model": model,
                "test_name": test_config["name"],
                "run_number": run_number,
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time,
                "first_token_time": None,
                "end_to_end_tps": None,
                "sustained_tps": None,
                "token_count": 0,
                "actual_tokens": 0,
                "chunk_count": 0,
                "quality": ResponseQuality.FAILED.value,
                "issues": [f"Exception: {str(e)}"],
                "warnings": [],
                "validation_metrics": {},
            }

            self.results_log.append(error_result)
            return error_result

    def generate_validation_report(self) -> str:
        """Generate a report on validation results"""
        if not self.results_log:
            return "No results to validate"

        report = ["# Benchmark Validation Report", ""]

        # Quality summary
        quality_counts = {}
        for result in self.results_log:
            quality = result.get("quality", "unknown")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        report.extend(
            [
                "## Response Quality Summary",
                "",
                "| Quality | Count | Percentage |",
                "|---------|-------|------------|",
            ]
        )

        total_results = len(self.results_log)
        for quality, count in sorted(quality_counts.items()):
            percentage = count / total_results * 100
            report.append(f"| {quality} | {count} | {percentage:.1f}% |")

        report.append("")

        # Issue summary
        all_issues = []
        for result in self.results_log:
            all_issues.extend(result.get("issues", []))

        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            report.extend(["## Common Issues", ""])

            for issue, count in sorted(
                issue_counts.items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"- {issue} ({count} occurrences)")

            report.append("")

        # Recommendations
        recommendations = []

        if quality_counts.get("truncated", 0) > 0:
            recommendations.append("Increase max_tokens for affected tests")

        if quality_counts.get("failed", 0) > 0:
            recommendations.append("Investigate failed responses")

        if any("Suspiciously high throughput" in issue for issue in all_issues):
            recommendations.append("Verify token counting accuracy")

        if recommendations:
            report.extend(["## Recommendations", ""])
            for rec in recommendations:
                report.append(f"- {rec}")

        return "\n".join(report)
