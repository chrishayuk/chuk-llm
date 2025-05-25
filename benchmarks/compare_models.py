# benchmarks/compare_models.py
"""
Enhanced Live Model Comparison - TPS-based Performance Benchmarking
"""
from __future__ import annotations

import asyncio
import argparse
import datetime as dt
import json
import logging
import pathlib
import sys
from pathlib import Path
from random import shuffle
from typing import List, Dict, Any, Tuple, Optional
from time import perf_counter

# ── keep the console tidy ─────────────────────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── local imports (repo-relative) ─────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
from llm_benchmark import BenchmarkConfig
from benchmarks.utils.benchmark_utils import (
    calculate_prompt_tokens,
    calculate_sustained_tps,
    filter_stable_runs,
    get_token_count,
)
from benchmarks.utils.results_display import CompellingResultsDisplay

# ── Enhanced validation utilities (inline to avoid imports) ───────────────────────
def get_accurate_token_count(text: str, model: str) -> int:
    """Get accurate token count with tiktoken fallback"""
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

def validate_response_quality(
    response_text: str, 
    test_type: str, 
    expected_min_tokens: int = 50
) -> Tuple[str, List[str]]:
    """More reasonable validation that doesn't over-flag good responses"""
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
        # Check for completion phrases
        completion_phrases = ["complete", "success", "finished", "done"]
        if not any(phrase in text.lower() for phrase in completion_phrases):
            issues.append("Speed test response doesn't indicate completion")
            return "good", issues  # Changed from "poor" to "good"
        return "excellent", []
    
    # For longer tests - only flag if obviously too short
    if word_count < expected_min_tokens * 0.3:  # Only flag if less than 30% of expected
        issues.append(f"Response significantly too short: {word_count} words < {expected_min_tokens}")
        return "failed", issues
    
    # More reasonable truncation detection - only check for obvious problems
    obvious_truncation = [
        # Ends with clear incomplete indicators
        text.endswith('...') or text.endswith('..'),
        # Extremely abrupt ending (very few words in last portion)
        len(text) > 50 and len(text.split()[-10:]) < 3,  # Last 10 words have less than 3 words
        # Ends with obvious incomplete patterns
        any(text.lower().strip().endswith(pattern) for pattern in [' and then', ' but then', ' however']),
    ]
    
    # Only flag truncation for obvious cases
    if any(obvious_truncation):
        issues.append("Response appears clearly truncated")
        return "truncated", issues
    
    # Test-specific quality checks - be more lenient
    if test_type == "creative":
        if word_count < 100:  # Only flag if very short
            issues.append("Creative response very brief")
            return "poor", issues
        # Don't require specific story elements - let creative responses be creative
    
    elif test_type == "math":
        # Only check for obvious math content
        if word_count < 50:  # Only flag very short math responses
            issues.append("Math response very brief")
            return "poor", issues
        # Don't be too strict about specific indicators
    
    # Length-based quality assessment
    if word_count >= expected_min_tokens:
        return "excellent", []
    elif word_count >= expected_min_tokens * 0.7:  # 70% of expected
        return "good", []
    elif word_count >= expected_min_tokens * 0.5:  # 50% of expected
        return "good", ["Response shorter than expected but acceptable"]
    else:
        return "poor", ["Response significantly shorter than expected"]

def validate_timing_metrics(
    total_time: float,
    first_token_time: Optional[float],
    token_count: int,
    test_type: str
) -> List[str]:
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
            issues.append(f"Suspiciously high throughput: {tokens_per_second:.0f} tok/s")
        elif tokens_per_second < 1 and test_type != "speed":
            issues.append(f"Suspiciously low throughput: {tokens_per_second:.1f} tok/s")
    
    if test_type in ["creative", "math"] and total_time < 0.5 and token_count > 100:
        issues.append("Suspiciously fast for substantial response")
    
    return issues

# ══════════════════════════════════════════════════════════════════════════════════
# Enhanced Live benchmark runner with TPS-focused validation
# ══════════════════════════════════════════════════════════════════════════════════
class EnhancedLiveBenchmarkRunner:
    """Run an on-screen 'battle' between several provider models with TPS-focused validation."""

    def __init__(self) -> None:
        self.display: CompellingResultsDisplay | None = None
        self.validation_stats = {
            "total_tests": 0,
            "quality_distribution": {},
            "common_issues": {},
            "suspicious_results": []
        }

    # ───────────────────────────────────────────────────────────────────────────────
    # Enhanced TEST SUITE DEFS with validation parameters
    # ───────────────────────────────────────────────────────────────────────────────
    def create_test_configs(self, suite: str) -> List[Dict[str, Any]]:
        """Build test configs optimized for TPS measurement"""
        
        if suite == "quick":
            return [
                {
                    "name": "speed",
                    "description": "Speed test (short)",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Please respond with: 'Speed test completed successfully. This demonstrates a complete response without truncation.' Include this exact phrase and nothing else.",
                        }
                    ],
                    "max_tokens": 100,
                    "temperature": 0,
                    "expected_min_tokens": 12,
                    "test_type": "speed"
                },
                {
                    "name": "math",
                    "description": "Math reasoning (optimized for TPS)",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Calculate compound interest: $10,000 invested at 5% annual rate, "
                                "compounded monthly for 10 years. Show your work step by step and "
                                "provide the final answer. Be thorough but efficient in your explanation."
                            ),
                        }
                    ],
                    "max_tokens": 1500,  # Optimized for good TPS measurement
                    "temperature": 0,
                    "expected_min_tokens": 100,
                    "test_type": "math"
                },
                {
                    "name": "creative",
                    "description": "Creative generation (TPS-optimized)",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Write a creative short story (400-600 words) about an AI discovering "
                                "emotions for the first time. Include dialogue, descriptive scenes, "
                                "and a satisfying conclusion. Focus on producing consistent, flowing text "
                                "that showcases your generation speed and quality."
                            ),
                        }
                    ],
                    "max_tokens": 2000,  # Optimized for sustained TPS measurement
                    "temperature": 0.7,
                    "expected_min_tokens": 250,  # Reasonable for TPS measurement
                    "test_type": "creative"
                },
            ]

        elif suite == "lightning":
            return [
                {
                    "name": "instant",
                    "description": "One-liner (TPS burst)",
                    "messages": [
                        {
                            "role": "user", 
                            "content": "Say 'Hello! This is a complete response.' exactly."
                        }
                    ],
                    "max_tokens": 50,
                    "temperature": 0,
                    "expected_min_tokens": 8,
                    "test_type": "speed"
                },
                {
                    "name": "sustained_generation",
                    "description": "Sustained TPS measurement",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Generate a detailed explanation of how neural networks learn, "
                                "including forward propagation, backpropagation, and gradient descent. "
                                "Aim for 300-400 words with consistent detail throughout."
                            ),
                        }
                    ],
                    "max_tokens": 1200,
                    "temperature": 0,
                    "expected_min_tokens": 200,
                    "test_type": "technical"
                },
            ]

        else:  # standard
            return [
                {
                    "name": "speed",
                    "description": "Short ping",
                    "messages": [
                        {
                            "role": "user", 
                            "content": "Respond with 'Pong! Response completed successfully.'"
                        }
                    ],
                    "max_tokens": 50,
                    "temperature": 0,
                    "expected_min_tokens": 5,
                    "test_type": "speed"
                },
                {
                    "name": "reasoning",
                    "description": "Logical reasoning (TPS measurement)",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Explain the traveling salesman problem, why it's computationally difficult, "
                                "and describe three different approaches to solving it. Provide examples "
                                "and be thorough in your explanation while maintaining good flow."
                            ),
                        }
                    ],
                    "max_tokens": 1800,
                    "temperature": 0,
                    "expected_min_tokens": 150,
                    "test_type": "reasoning"
                },
                {
                    "name": "creative",
                    "description": "Creative writing (sustained TPS)",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Write an engaging technical blog post about the future of quantum computing. "
                                "Make it accessible to a general audience while being technically accurate. "
                                "Aim for 500-700 words with consistent quality throughout."
                            ),
                        }
                    ],
                    "max_tokens": 2200,
                    "temperature": 0.3,
                    "expected_min_tokens": 300,
                    "test_type": "creative"
                },
                {
                    "name": "code",
                    "description": "Code generation (TPS evaluation)",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Create a Python class for a binary search tree with insert, delete, "
                                "search, and in-order traversal methods. Include proper documentation "
                                "and error handling. Focus on producing clean, consistent code."
                            ),
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0,
                    "expected_min_tokens": 200,
                    "test_type": "code"
                },
            ]

    def _adjust_config_for_model(self, config: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Adjust test configuration for specific model quirks"""
        adjusted_config = config.copy()
        
        # GPT-3.5-turbo specific adjustments
        if "gpt-3.5" in model.lower():
            if config["name"] == "math":
                # Use a prompt that doesn't trigger function calling
                adjusted_config["messages"] = [
                    {
                        "role": "user",
                        "content": (
                            "Please solve this step by step using text explanation only: "
                            "An investment of $10,000 grows at 5% annual interest rate, "
                            "compounded monthly for 10 years. Show your mathematical "
                            "work and provide the final dollar amount. Write out all steps."
                        )
                    }
                ]
        
        return adjusted_config

    # ───────────────────────────────────────────────────────────────────────────────
    # WARM-UP
    # ───────────────────────────────────────────────────────────────────────────────
    async def warmup_model(self, provider: str, model: str) -> None:
        """One tiny call to avoid cold-start skew."""
        try:
            from chuk_llm.llm.llm_client import get_llm_client

            client = get_llm_client(provider, model=model)
            await client.create_completion(
                [{"role": "user", "content": "ping"}], max_tokens=1, temperature=0
            )
            print(f"    🔥 {model} warmed up")
        except Exception as exc:
            print(f"    ⚠️  {model} warm-up failed: {exc}")

    # ───────────────────────────────────────────────────────────────────────────────
    # MAIN DRIVER with TPS-focused validation
    # ───────────────────────────────────────────────────────────────────────────────
    async def run_benchmark_battle(
        self,
        provider: str,
        models: List[str],
        test_suite: str = "quick",
        runs_per_model: int = 3,
        enable_validation: bool = True,
    ) -> None:
        """Run the full on-screen benchmark with TPS-focused leadership."""
        
        self.display = CompellingResultsDisplay(models)
        test_configs = self.create_test_configs(test_suite)

        print(f"🔥 BENCHMARK BATTLE: {provider.upper()}")
        print(f"⚔️  Contenders: {' vs '.join(models)}")
        print(
            f"🎯 Challenge: {test_suite.upper()} "
            f"({len(test_configs)} tests × {runs_per_model} runs)"
        )
        
        if enable_validation:
            print("🏆 Victory Condition: HIGHEST THROUGHPUT (TOKENS PER SECOND)!")
            print("📋 Using accurate token counting, TPS measurement, and quality filtering")
            print("⚖️ Round-robin execution for fair timing conditions")
        else:
            print("🏆 Victory Condition: THROUGHPUT-FOCUSED PERFORMANCE!")
            print("📋 Using real token counting, TPS optimization, and sustained throughput")
            
        print("\n🔥 Warming up models...")
        for m in models:
            await self.warmup_model(provider, m)
        await asyncio.sleep(1)

        self.display.display_live_standings()
        
        total_tests = len(test_configs) * runs_per_model * len(models)
        test_counter = 0
        
        # Create test schedule with round-robin approach
        test_schedule = []
        
        # For each round
        for round_num in range(runs_per_model):
            # Shuffle test order for this round to avoid test-order bias
            round_tests = test_configs[:]
            shuffle(round_tests)
            
            # For each test in this round
            for test_config in round_tests:
                # Shuffle model order for this test to avoid model-order bias
                round_models = models[:]
                shuffle(round_models)
                
                # Add each model for this test
                for model in round_models:
                    test_schedule.append({
                        "model": model,
                        "test_config": test_config,
                        "round": round_num + 1,
                        "test_id": f"R{round_num + 1}_{test_config['name']}"
                    })
        
        print(f"\n🎲 Randomized schedule created: {len(test_schedule)} total tests")
        print("📊 Fair round-robin execution ensures equal timing conditions for all models")
        print("🚀 Focus on measuring sustained throughput (tokens per second)\n")
        
        # Execute the round-robin schedule
        model_test_counts = {model: 0 for model in models}
        
        for schedule_item in test_schedule:
            model = schedule_item["model"]
            test_config = schedule_item["test_config"]
            round_num = schedule_item["round"]
            test_id = schedule_item["test_id"]
            
            test_counter += 1
            model_test_counts[model] += 1
            
            print(f"  🥊 {test_id} • {model} • {test_config['description']}...", end=" ", flush=True)
            
            result = await self._execute_enhanced_test(provider, model, test_config, enable_validation)
            
            # Print results with TPS emphasis
            self._print_test_result(result, enable_validation)
            
            # Update display with TPS emphasis
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
                len(test_configs) * runs_per_model,  # Total tests per model
            )
            
            # Track validation stats
            if enable_validation:
                self._update_validation_stats(result)
            
            # Show updated leaderboard every few tests (emphasizing TPS leaders)
            if test_counter % (len(models) * len(test_configs)) == 0:
                print(f"\n📊 Round {test_counter // (len(models) * len(test_configs))} complete!")
                self.display.display_live_standings()
                print()
            
            # Small delay between tests to avoid overwhelming APIs
            await asyncio.sleep(0.3)
        
        print(f"\n🏁 All {len(models)} models complete their challenges!")
        
        # Save and show final results
        self._save_enhanced_results(provider, models, test_suite, enable_validation)
        self.display.show_final_battle_results()
        
        if enable_validation:
            self._show_validation_summary()

    def _print_test_result(self, result: Dict[str, Any], validation_enabled: bool) -> None:
        """Print test result with TPS emphasis"""
        
        if result["success"]:
            if validation_enabled and result.get("quality"):
                quality_emoji = {
                    "excellent": "✅",
                    "good": "👍", 
                    "poor": "⚠️",
                    "truncated": "✂️",
                    "failed": "❌"
                }.get(result["quality"], "?")
                
                pieces = [f"{quality_emoji} {result['total_time']:.2f}s"]
            else:
                pieces = [f"✅ {result['total_time']:.2f}s"]
            
            if result.get("first_token_time"):
                pieces.append(f"🚀 {result['first_token_time']:.2f}s")
            
            # Emphasize TPS in output
            if result.get("sustained_tps"):
                tps = result["sustained_tps"]
                if tps > 100:
                    pieces.append(f"⚡ {tps:.0f}/s")  # High TPS
                elif tps > 50:
                    pieces.append(f"📈 {tps:.0f}/s")  # Good TPS
                else:
                    pieces.append(f"🐌 {tps:.0f}/s")  # Lower TPS
            elif result.get("end_to_end_tps"):
                pieces.append(f"⚡ {result['end_to_end_tps']:.0f}/s")
            
            if result.get("display_tokens"):
                pieces.append(f"📝 {result['display_tokens']}tok")
            
            print(" | ".join(pieces))
            
            # Show TPS achievements
            tps = result.get("sustained_tps") or result.get("end_to_end_tps", 0)
            if tps > 150:
                print("    🚀 BLAZING THROUGHPUT! >150 tok/s!")
            elif tps > 100:
                print("    💨 Excellent throughput! >100 tok/s!")
            elif tps > 50:
                print("    📈 Good throughput! >50 tok/s")
            
            # Show warnings for quality issues
            if validation_enabled and result.get("issues"):
                if result["quality"] in ["poor", "truncated"]:
                    issues_summary = "; ".join(result["issues"][:2])
                    print(f"    ⚠️ Quality issues: {issues_summary}")
            
        else:
            print(f"❌ {result.get('error', 'Unknown error')}")

    def _update_validation_stats(self, result: Dict[str, Any]) -> None:
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
        
        if result.get("timing_issues"):
            self.validation_stats["suspicious_results"].append({
                "model": result.get("model"),
                "test": result.get("test_name"),
                "issues": result["timing_issues"]
            })

    # ───────────────────────────────────────────────────────────────────────────────
    # Enhanced JSON SAVE
    # ───────────────────────────────────────────────────────────────────────────────
    def _save_enhanced_results(self, provider: str, models: List[str], suite: str, validation: bool) -> None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        pathlib.Path("benchmark_runs").mkdir(exist_ok=True)

        results_data = {
            "timestamp": ts,
            "provider": provider,
            "models": models,
            "test_suite": suite,
            "validation_enabled": validation,
            "leadership_metric": "throughput_tps",  # NEW: indicates TPS-based leadership
            "results": self.display.test_results,
            "validation_summary": self.validation_stats if validation else None
        }

        suffix = "_tps_focused" if validation else "_tps_basic"
        filename = f"benchmark_runs/{ts}_{provider}_{suite}{suffix}.json"
        
        with open(filename, "w") as fh:
            json.dump(results_data, fh, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 Results saved to: {filename}")

    # ───────────────────────────────────────────────────────────────────────────────
    # Enhanced SINGLE TEST EXEC with TPS focus and GPT-3.5-turbo fixes
    # ───────────────────────────────────────────────────────────────────────────────
    async def _execute_enhanced_test(
        self, provider: str, model: str, config: Dict[str, Any], enable_validation: bool
    ) -> Dict[str, Any]:
        """Execute single test with ENHANCED debugging for GPT-3.5-turbo."""
        
        try:
            from chuk_llm.llm.llm_client import get_llm_client

            client = get_llm_client(provider, model=model)
            
            # ADJUST CONFIG FOR MODEL-SPECIFIC QUIRKS
            config = self._adjust_config_for_model(config, model)

            # Prepare request parameters
            kwargs = {}
            if config.get("max_tokens"):
                kwargs["max_tokens"] = config["max_tokens"]
            if config.get("temperature") is not None:
                kwargs["temperature"] = config["temperature"]
            
            # EXPLICITLY DISABLE FUNCTION CALLING FOR GPT-3.5-turbo
            if "gpt-3.5" in model.lower():
                kwargs["tool_choice"] = "none"  # Explicitly disable function calling

            # # SPECIAL DEBUGGING FOR GPT-3.5-turbo
            # if "gpt-3.5" in model.lower():
            #     print(f"\n🔍 DEBUG {model}: Starting execution...")
            #     print(f"    Config: {config['name']}")
            #     print(f"    Kwargs: {kwargs}")
            #     print(f"    Message length: {len(config['messages'][0]['content'])}")
            #     print(f"    Tools disabled: {kwargs.get('tool_choice') == 'none'}")

            # Execute with streaming and TPS-focused timing
            start = perf_counter()
            first_tok = None
            full_response = ""
            cumulative_timeline = []
            cum_tokens = 0
            chunk_count = 0
            error_encountered = False

            try:
                async for chunk in client.create_completion(
                    config["messages"], 
                    tools=None,  # Explicitly no tools
                    stream=True, 
                    **kwargs
                ):
                    current_time = perf_counter()
                    chunk_count += 1
                    
                    # # ENHANCED DEBUGGING FOR GPT-3.5-turbo
                    # if "gpt-3.5" in model.lower() and chunk_count <= 10:
                    #     print(f"    Chunk {chunk_count}: {chunk}")
                    
                    # Check for errors in chunk
                    if chunk.get("error"):
                        error_encountered = True
                        error_msg = chunk.get("response", "Unknown streaming error")
                        if "gpt-3.5" in model.lower():
                            print(f"    ❌ Error in chunk {chunk_count}: {error_msg}")
                        break
                    
                    # Skip chunks that only contain tool calls (shouldn't happen now)
                    if not chunk.get("response") and chunk.get("tool_calls"):
                        if "gpt-3.5" in model.lower():
                            print(f"    🔧 Skipping tool call chunk {chunk_count}")
                        continue
                    
                    if "response" not in chunk:
                        if "gpt-3.5" in model.lower() and chunk_count <= 5:
                            print(f"    ⚠️ Chunk {chunk_count} missing 'response' key")
                        continue
                        
                    chunk_text = chunk["response"]
                    if chunk_text and first_tok is None:
                        first_tok = current_time - start
                        if "gpt-3.5" in model.lower():
                            print(f"    🚀 First token at {first_tok:.2f}s")

                    full_response += chunk_text
                    
                    # Calculate tokens (use accurate counting if validation enabled)
                    if enable_validation:
                        from benchmarks.utils.benchmark_utils import get_accurate_token_count
                        delta_tokens = get_accurate_token_count(chunk_text, model)
                    else:
                        from benchmarks.utils.benchmark_utils import get_token_count
                        delta_tokens = get_token_count(chunk_text, model)
                        
                    cum_tokens += delta_tokens
                    cumulative_timeline.append((current_time - start, cum_tokens))
                    
                    # Safety timeout for debugging
                    if current_time - start > 60:  # 60 second timeout
                        if "gpt-3.5" in model.lower():
                            print(f"    ⏰ Timeout after 60s")
                        break

            except Exception as stream_error:
                error_encountered = True
                elapsed = perf_counter() - start
                if "gpt-3.5" in model.lower():
                    print(f"    ❌ Streaming exception after {elapsed:.2f}s: {stream_error}")
                    print(f"    Exception type: {type(stream_error).__name__}")
                    import traceback
                    traceback.print_exc()

            total_time = perf_counter() - start
            first_tok = first_tok or 0.0

            # # Enhanced logging for GPT-3.5-turbo
            # if "gpt-3.5" in model.lower():
            #     print(f"    📊 Results: {total_time:.2f}s total, {chunk_count} chunks, {len(full_response)} chars")
            #     print(f"    Error encountered: {error_encountered}")
            #     if full_response:
            #         print(f"    Response preview: {full_response[:100]}...")

            # Check if this is the characteristic failure pattern
            if error_encountered or (total_time < 2 and len(full_response) < 50):
                return {
                    "success": False,
                    "total_time": total_time,
                    "first_token_time": first_tok,
                    "end_to_end_tps": None,
                    "sustained_tps": None,
                    "stream_tokens": cum_tokens,
                    "accurate_tokens": len(full_response.split()) if full_response else 0,
                    "display_tokens": 0,
                    "response_text": full_response,
                    "model": model,
                    "test_name": config["name"],
                    "error": "Early termination or streaming failure",
                    "debug_info": {
                        "chunk_count": chunk_count,
                        "error_encountered": error_encountered,
                        "response_length": len(full_response)
                    }
                }

            # Calculate TPS metrics with enhanced accuracy
            from benchmarks.utils.benchmark_utils import calculate_sustained_tps
            e2e_tps = cum_tokens / total_time if total_time > 0 else 0
            sust_tps = calculate_sustained_tps(cumulative_timeline)
            
            # Get accurate token count for validation
            if enable_validation:
                from benchmarks.utils.benchmark_utils import get_accurate_token_count
                accurate_tokens = get_accurate_token_count(full_response, model)
            else:
                accurate_tokens = cum_tokens

            result = {
                "success": True,
                "total_time": total_time,
                "first_token_time": first_tok,
                "end_to_end_tps": e2e_tps,
                "sustained_tps": sust_tps,
                "stream_tokens": cum_tokens,
                "accurate_tokens": accurate_tokens,
                "display_tokens": accurate_tokens,  # Use accurate count for display
                "response_text": full_response,
                "model": model,
                "test_name": config["name"]
            }

            # Add validation if enabled
            if enable_validation:
                quality, issues = validate_response_quality(
                    full_response, 
                    config.get("test_type", "unknown"),
                    config.get("expected_min_tokens", 50)
                )
                
                timing_issues = validate_timing_metrics(
                    total_time, first_tok, accurate_tokens, config.get("test_type", "unknown")
                )
                
                result.update({
                    "quality": quality,
                    "issues": issues,
                    "timing_issues": timing_issues,
                    "quality_valid": quality in ["excellent", "good"],
                    "token_accuracy": abs(accurate_tokens - cum_tokens) / max(accurate_tokens, 1) if accurate_tokens > 0 else 0
                })

            return result

        except Exception as exc:
            elapsed = perf_counter() - start if 'start' in locals() else 0
            
            # Enhanced error logging for GPT-3.5-turbo
            if "gpt-3.5" in model.lower():
                print(f"🔍 DEBUG {model}: Top-level exception after {elapsed:.2f}s")
                print(f"    Exception: {exc}")
                print(f"    Exception type: {type(exc).__name__}")
                import traceback
                traceback.print_exc()
            
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
                "issues": [f"Exception: {str(exc)}"],
                "timing_issues": [],
                "quality_valid": False,
                "error": str(exc),
                "model": model,
                "test_name": config["name"]
            }
        
    def _show_validation_summary(self) -> None:
        """Show validation statistics summary with TPS focus"""
        if self.validation_stats["total_tests"] == 0:
            return
            
        print(f"\n📊 VALIDATION SUMMARY (TPS-FOCUSED)")
        print("=" * 60)
        
        stats = self.validation_stats
        total = stats["total_tests"]
        
        # Quality distribution
        print("Quality Distribution:")
        for quality, count in sorted(stats["quality_distribution"].items()):
            percentage = count / total * 100
            emoji = {"excellent": "✅", "good": "👍", "poor": "⚠️", "truncated": "✂️", "failed": "❌"}.get(quality, "?")
            print(f"  {emoji} {quality}: {count} ({percentage:.1f}%)")
        
        # TPS-focused insights
        excellent_good = (stats["quality_distribution"].get("excellent", 0) + 
                         stats["quality_distribution"].get("good", 0))
        quality_rate = excellent_good / total
        
        print(f"\n💡 TPS Measurement Reliability:")
        if quality_rate >= 0.8:
            print("  ✅ Throughput measurements are highly reliable")
            print("  🚀 TPS comparisons are valid and meaningful")
        elif quality_rate >= 0.6:
            print("  ⚠️ Some quality issues may affect TPS accuracy")
            print("  💡 Consider increasing max_tokens for better TPS measurement")
        else:
            print("  ❌ Significant quality issues detected")
            print("  💡 TPS measurements may be unreliable - increase max_tokens")


# ══════════════════════════════════════════════════════════════════════════════════
# CLI entry-point with TPS focus
# ══════════════════════════════════════════════════════════════════════════════════
async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced Live Model Benchmark Battle with TPS-based Leadership",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔥 TPS-FOCUSED BATTLE MODES:
  lightning  - Ultra-fast 2-test sprint focused on burst and sustained TPS
  quick      - Fast 3-test battle optimized for TPS measurement
  standard   - Full 4-test championship with comprehensive TPS analysis

🚀 TPS LEADERSHIP FEATURES:
  • Rankings based on highest tokens per second (sustained TPS preferred)
  • Real-time TPS measurement during streaming
  • Optimized test lengths for accurate throughput measurement
  • Quality validation to ensure TPS measurements are meaningful
  • Fair round-robin execution for unbiased TPS comparison

⚡ THROUGHPUT METRICS:
  • Sustained TPS: Measured from stable streaming portion (most accurate)
  • End-to-end TPS: Total tokens divided by total time
  • Peak TPS: Highest sustained rate achieved across all tests
  • Consistent TPS: Models with stable throughput across tests

Examples:
  python compare_models.py openai "gpt-4o-mini,gpt-4o" --suite quick --runs 3
  python compare_models.py anthropic "claude-3-5-sonnet-20241022" --suite lightning
  python compare_models.py openai "gpt-4o,gpt-4o-mini" --no-validation  # Disable validation
        """,
    )
    parser.add_argument("provider", help="Provider (openai, groq, anthropic, gemini, ollama)")
    parser.add_argument("models", help="Comma-separated models to battle")
    parser.add_argument(
        "--suite",
        choices=["lightning", "quick", "standard"],
        default="quick",
        help="Battle intensity optimized for TPS measurement (default: quick)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Rounds per model (default: 3)"
    )
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Disable TPS validation features (faster but less accurate)"
    )
    
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    if len(models) < 2:
        print("❌ Need at least 2 models for a proper TPS battle!")
        return

    runner = EnhancedLiveBenchmarkRunner()
    await runner.run_benchmark_battle(
        provider=args.provider,
        models=models,
        test_suite=args.suite,
        runs_per_model=args.runs,
        enable_validation=not args.no_validation
    )


if __name__ == "__main__":
    asyncio.run(_main())