# benchmarks/utils/results_display.py
"""
Results display and analysis system - Enhanced with backward compatibility
Fixed to exclude failed runs from leaderboard rankings
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

MIN_TOKENS_FOR_THROUGHPUT = 400

class DisplayTheme(Enum):
    """Different display themes for the leaderboard"""
    COMPACT = "compact"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    COLORFUL = "colorful"

@dataclass
class ModelResult:
    """Structured model result data"""
    model: str
    rank: int
    best_tps: Optional[float]
    best_time: float
    first_token: Optional[float]
    avg_tps: Optional[float]
    success_rate: float
    tests_completed: int
    total_tests: int
    status: str
    trend_emoji: str
    recent_test: str

class CompellingResultsDisplay:
    """Enhanced results display with better formatting and visual appeal"""

    def __init__(self, models: List[str], theme: str = "detailed"):
        self.models = models
        self.theme = self._parse_theme(theme)
        self.results = {m: self._empty_result() for m in models}
        self.test_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
        self.start_time = time.time()
        self.current_leader: Optional[str] = None
        
        # Display configuration
        self.config = self._get_display_config(self.theme)

    def _parse_theme(self, theme: str) -> DisplayTheme:
        """Parse theme string to DisplayTheme enum"""
        theme_map = {
            "compact": DisplayTheme.COMPACT,
            "detailed": DisplayTheme.DETAILED,
            "minimal": DisplayTheme.MINIMAL,
            "colorful": DisplayTheme.COLORFUL
        }
        return theme_map.get(theme.lower(), DisplayTheme.DETAILED)

    def _get_display_config(self, theme: DisplayTheme) -> Dict[str, Any]:
        """Get display configuration for the selected theme"""
        # Calculate total width based on fixed column widths (NO EMOJIS)
        base_width = 4 + 25 + 12 + 10 + 10 + 15 + 6  # rank + model + tps + time + token + status + spacing
        progress_width = 12 + 1  # progress column + spacing
        avg_tps_width = 12 + 1   # avg tps column + spacing
        
        configs = {
            DisplayTheme.COMPACT: {
                "show_progress": False,
                "show_trend": True,
                "show_avg_tps": False,
                "model_width": 25,
                "total_width": base_width,
                "use_unicode": True,
                "show_detailed_status": False
            },
            DisplayTheme.DETAILED: {
                "show_progress": True,
                "show_trend": True,
                "show_avg_tps": True,
                "model_width": 25,
                "total_width": base_width + progress_width + avg_tps_width,
                "use_unicode": True,
                "show_detailed_status": True
            },
            DisplayTheme.MINIMAL: {
                "show_progress": False,
                "show_trend": False,
                "show_avg_tps": False,
                "model_width": 25,
                "total_width": base_width,
                "use_unicode": False,
                "show_detailed_status": False
            },
            DisplayTheme.COLORFUL: {
                "show_progress": True,
                "show_trend": True,
                "show_avg_tps": True,
                "model_width": 25,
                "total_width": base_width + progress_width + avg_tps_width,
                "use_unicode": True,
                "show_detailed_status": True
            }
        }
        return configs[theme]

    @staticmethod
    def _empty_result():
        return {
            "status": "⏳ Pending",
            "tests_completed": 0,
            "best_time": float("inf"),
            "best_first_token": float("inf"),
            "best_throughput": 0,
            "avg_throughput": 0,
            "success_rate": 0,
            "recent_test": "",
            "performance_trend": "📊",
        }

    def update_model(
        self,
        model: str,
        test_name: str,
        response_time: float,
        first_token: Optional[float],
        end_to_end_tps: Optional[float],
        sustained_tps: Optional[float],
        token_count: int,
        extra_metrics: Dict[str, Any],
        success: bool,
        tests_completed: int,
        total_tests: int
    ):
        """Update model results, **counting only runs that emit ≥ 400 tokens** for any
        throughput-related statistic or ranking."""

        MIN_TOKENS_FOR_THROUGHPUT = 400          # ← single point of truth

        tps_alias = sustained_tps or end_to_end_tps or 0

        # ── 1. log raw run (unchanged) ─────────────────────────────────────────
        self.test_results[model].append({
            "test_name": test_name,
            "response_time": response_time,
            "first_token": first_token,
            "end_to_end_tps": end_to_end_tps,
            "sustained_tps": sustained_tps,
            "throughput": tps_alias,
            "token_count": token_count,
            "extra_metrics": extra_metrics,
            "success": success,
            "timestamp": time.time(),
            "run_id": f"{test_name}_{len([r for r in self.test_results[model] if r['test_name']==test_name])+1}",
        })

        res = self.results[model]
        res["tests_completed"] = tests_completed
        res["recent_test"]     = test_name

        # ── 2. update best-of metrics  (only if big enough) ───────────────────
        if success:
            if response_time < res["best_time"]:
                res["best_time"] = response_time
            if first_token and first_token < res["best_first_token"]:
                res["best_first_token"] = first_token
            if (token_count >= MIN_TOKENS_FOR_THROUGHPUT
                    and tps_alias
                    and tps_alias > res["best_throughput"]):
                res["best_throughput"] = tps_alias

        # ── 3. compute avg TPS from   **big, successful runs**  ────────────────
        big_successes = [
            r for r in self.test_results[model]
            if r["success"]
            and r.get("throughput")
            and r["token_count"] >= MIN_TOKENS_FOR_THROUGHPUT
        ]
        if big_successes:
            res["avg_throughput"] = (
                sum(r["throughput"] for r in big_successes) / len(big_successes)
            )

        # ── 4. success-rate & status  (unchanged logic) ───────────────────────
        total_runs = len(self.test_results[model])
        succ_cnt   = sum(1 for r in self.test_results[model] if r["success"])
        res["success_rate"] = succ_cnt / total_runs if total_runs else 0

        if tests_completed == total_tests:
            if res["success_rate"] == 1:
                res["status"] = "✅ Complete"
            elif res["success_rate"] == 0:
                res["status"] = "❌ All Failed"
            else:
                res["status"] = f"⚠️ {res['success_rate']:.0%} Success"
            res["performance_trend"] = self._get_performance_emoji(
                res["best_time"], res["best_throughput"]
            )
        else:
            res["status"] = f"🔄 Testing ({tests_completed}/{total_tests})"
            if success:
                res["performance_trend"] = self._get_live_trend_emoji(
                    response_time, tps_alias
                )


    def _get_performance_emoji(self, best_time: float, best_tps: float) -> str:
        """Get performance emoji based on time and TPS"""
        if best_tps > 100:
            return "🚀"  # Blazing fast TPS
        elif best_tps > 50:
            return "⚡"  # Fast TPS
        elif best_time < 1:
            return "🏃"  # Fast time
        elif best_time < 3:
            return "🚶"  # OK time
        else:
            return "🐢"  # Slow

    def _get_live_trend_emoji(self, response_time: float, tps: float) -> str:
        """Get live trend emoji for ongoing tests"""
        if tps > 100:
            return "🚀"
        elif tps > 50:
            return "⚡"
        elif response_time < 1:
            return "🏃"
        elif response_time < 2:
            return "🚶"
        else:
            return "🐌"

    def display_live_standings(self, highlight_model: str = None):
        """Display improved live leaderboard with better formatting"""
        print("\n")
        elapsed = time.time() - self.start_time
        
        # Header with better formatting
        self._print_header(elapsed)
        
        # Collect and sort model results
        model_results = self._collect_model_results()
        
        # Display the formatted table
        self._print_leaderboard_table(model_results, highlight_model)
        
        # Show insights
        self._print_insights(model_results)
        
        print()

    def _print_header(self, elapsed: float):
        """Print formatted header"""
        config = self.config
        width = config["total_width"]
        
        print("═" * width)
        title_line = f"🏁 LIVE LEADERBOARD ⏱️  {elapsed:.0f}s"
        padding = (width - len(title_line)) // 2
        print(" " * padding + title_line)
        print("═" * width)
        
        if self.theme != DisplayTheme.MINIMAL:
            subtitle = "📊 Rankings by AVERAGE THROUGHPUT (tokens/sec) • Real-time Performance Tracking"
            sub_padding = (width - len(subtitle)) // 2
            print(" " * sub_padding + subtitle)
            print("─" * width)

    def _collect_model_results(self) -> List[ModelResult]:
        """
        Assemble the leaderboard.

        • A run is eligible only when `success` is True **and**
        `token_count ≥ MIN_TOKENS_FOR_THROUGHPUT`.
        • Sorting: highest **average TPS** first; if two models differ
        by < 5 % in avg-TPS, we fall back to peak TPS, then success-rate.
        """
        throughput_best: Dict[str, Dict[str, Any]] = {}

        for model, runs in self.test_results.items():
            big_successes = [
                r for r in runs
                if r["success"]
                and r.get("throughput")
                and r["token_count"] >= MIN_TOKENS_FOR_THROUGHPUT
            ]
            if not big_successes:
                continue  # exclude model with no qualifying runs

            # per-model stats
            best_run = max(big_successes, key=lambda r: r["throughput"])
            avg_tps  = sum(r["throughput"] for r in big_successes) / len(big_successes)

            throughput_best[model] = {
                **best_run,
                "avg_tps": avg_tps,
                "successful_runs_count": len(big_successes)
            }

        # ── ranking key: avg TPS → peak TPS → success-rate ───────────────────
        def _sort_key(item):
            model, data  = item
            avg          = data["avg_tps"]
            peak         = data["throughput"]
            total_runs   = len(self.test_results[model])
            succ_runs    = throughput_best[model]["successful_runs_count"]
            success_rate = succ_runs / total_runs if total_runs else 0
            return (avg, peak, success_rate)   # all descending

        sorted_models = sorted(throughput_best.items(),
                            key=_sort_key,
                            reverse=True)

        # ── build ModelResult objects ─────────────────────────────────────────
        results: List[ModelResult] = []
        for rank, (model, data) in enumerate(sorted_models, 1):
            all_runs   = self.test_results[model]
            total_runs = len(all_runs)
            successful = sum(1 for r in all_runs if r["success"])

            results.append(
                ModelResult(
                    model           = model,
                    rank            = rank,
                    best_tps        = data["throughput"],
                    best_time       = data["response_time"],
                    first_token     = data.get("first_token"),
                    avg_tps         = data["avg_tps"],
                    success_rate    = successful / total_runs if total_runs else 0,
                    tests_completed = successful,
                    total_tests     = total_runs,
                    status          = self._get_status_for_model(model,
                                                                successful,
                                                                total_runs),
                    trend_emoji     = self.results[model]["performance_trend"],
                    recent_test     = self.results[model]["recent_test"],
                )
            )

        return results

    def _get_status_for_model(self, model: str, successful_runs: int, total_runs: int) -> str:
        """Get appropriate status for model based on success rate"""
        if total_runs == 0:
            return "🔄 Pending"
        
        success_rate = successful_runs / total_runs
        
        if successful_runs == 0:
            return "❌ All Failed"
        elif success_rate == 1.0:
            return "✅ Complete"
        elif success_rate >= 0.5:
            return f"⚠️ {success_rate:.0%} Success"
        else:
            return f"❌ {success_rate:.0%} Success"

    def _print_leaderboard_table(
        self,
        results: List[ModelResult],
        highlight_model: str = None
    ):
        """Main leaderboard table — now sorted and shown by *average* TPS first."""
        cfg = self.config

        # fixed column widths
        rank_w, model_w = 4, 25
        tps_w, time_w   = 12, 10
        tok_w           = 10
        prog_w, stat_w  = 12, 15

        # ── build header ──────────────────────────────────────────────────────
        if cfg["show_avg_tps"] and cfg["show_progress"]:
            header = (
                f"{'Rank':<{rank_w}} {'Model':<{model_w}} "
                f"{'Avg TPS':<{tps_w}} {'Peak TPS':<{tps_w}} "
                f"{'Time':<{time_w}} {'First Tok':<{tok_w}} "
                f"{'Progress':<{prog_w}} {'Status':<{stat_w}}"
            )
        elif cfg["show_progress"]:
            header = (
                f"{'Rank':<{rank_w}} {'Model':<{model_w}} "
                f"{'Avg TPS':<{tps_w}} "
                f"{'Time':<{time_w}} {'First Tok':<{tok_w}} "
                f"{'Progress':<{prog_w}} {'Status':<{stat_w}}"
            )
        elif cfg["show_avg_tps"]:
            header = (
                f"{'Rank':<{rank_w}} {'Model':<{model_w}} "
                f"{'Avg TPS':<{tps_w}} {'Peak TPS':<{tps_w}} "
                f"{'Time':<{time_w}} {'First Tok':<{tok_w}} "
                f"{'Status':<{stat_w}}"
            )
        else:
            header = (
                f"{'Rank':<{rank_w}} {'Model':<{model_w}} "
                f"{'Time':<{time_w}} {'First Tok':<{tok_w}} "
                f"{'Status':<{stat_w}}"
            )

        print(header)
        print("─" * len(header))

        for res in results:
            self._print_model_row(res, highlight_model, cfg)


    def _print_model_row(self, result: ModelResult, highlight_model: str, config: Dict[str, Any]):
        """Print a single model row with proper fixed-width formatting"""
        
        # Fixed column widths - these MUST match the header exactly
        rank_width = 4
        model_width = 25
        tps_width = 12
        time_width = 10
        token_width = 10
        progress_width = 12
        status_width = 15

        # Rank - NO EMOJIS, just clean numbers for perfect alignment
        if result.rank == 1:
            rank_display = "1st"
        elif result.rank == 2:
            rank_display = "2nd"
        elif result.rank == 3:
            rank_display = "3rd"
        else:
            rank_display = f"{result.rank}th"
        
        # Model name - strict truncation to fit exactly in 25 characters
        model_display = result.model
        if highlight_model and result.model == highlight_model:
            # Add indicator but ensure total fits in model_width
            if len(result.model) > model_width - 4:  # Reserve space for "» "
                truncated = result.model[:model_width - 6] + "..."
                model_display = f"» {truncated}"
            else:
                model_display = f"» {result.model}"
        else:
            if len(model_display) > model_width:
                model_display = model_display[:model_width - 3] + "..."
        
        # Ensure model name is exactly model_width characters
        model_display = f"{model_display:<{model_width}}"[:model_width]
        
        # Performance metrics - fixed width, NO EMOJIS
        tps_display = self._format_tps_fixed_clean(result.best_tps, tps_width)
        time_display = self._format_time_fixed(result.best_time, time_width)
        token_display = self._format_time_fixed(result.first_token, token_width)
        
        # Progress - fixed width
        if config["show_progress"] and result.total_tests > 0:
            progress_pct = result.tests_completed / result.total_tests
            progress_text = f"{result.tests_completed}/{result.total_tests} ({progress_pct:.0%})"
            progress_display = f"{progress_text:<{progress_width}}"[:progress_width]
        else:
            progress_display = f"{'─':<{progress_width}}"[:progress_width]
        
        # Status - fixed width, minimal emojis
        status_parts = result.status.split()
        if len(status_parts) > 1:
            status_text = status_parts[1]  # "Complete", "Pending", etc.
        else:
            status_text = result.status
        # Use simple text-based indicators
        if "Complete" in status_text:
            status_display = f"{'Complete':<{status_width}}"[:status_width]
        elif "Pending" in status_text:
            status_display = f"{'Pending':<{status_width}}"[:status_width]
        else:
            status_display = f"{status_text:<{status_width}}"[:status_width]

        # Print with exact spacing
        avg_tps_display  = self._format_tps_fixed_clean(result.avg_tps,  tps_width)
        peak_tps_display = self._format_tps_fixed_clean(result.best_tps, tps_width)

        if config["show_avg_tps"] and config["show_progress"]:
            print(f"{rank_display:<{rank_width}} {model_display} "
                f"{avg_tps_display} {peak_tps_display} "
                f"{time_display} {token_display} {progress_display} {status_display}")
        elif config["show_progress"]:
            print(f"{rank_display:<{rank_width}} {model_display} "
                f"{avg_tps_display} "
                f"{time_display} {token_display} {progress_display} {status_display}")
        elif config["show_avg_tps"]:
            print(f"{rank_display:<{rank_width}} {model_display} "
                f"{avg_tps_display} {peak_tps_display} "
                f"{time_display} {token_display} {status_display}")
        else:
            print(f"{rank_display:<{rank_width}} {model_display} "
                f"{time_display} {token_display} {status_display}")


    def _format_tps_fixed_clean(self, tps: Optional[float], width: int) -> str:
        """Format TPS with fixed width - NO EMOJIS for perfect alignment"""
        if tps is None or tps == 0:
            return f"{'─':<{width}}"[:width]
        
        # Clean numeric display only
        text = f"{tps:.0f}/s"
        return f"{text:<{width}}"[:width]

    def _format_tps_fixed(self, tps: Optional[float], width: int) -> str:
        """Format TPS with fixed width"""
        if tps is None or tps == 0:
            return f"{'─':<{width}}"[:width]
        
        if tps >= 100:
            text = f"⚡{tps:.0f}/s"
        elif tps >= 50:
            text = f"🔥{tps:.0f}/s"
        elif tps >= 25:
            text = f"📈{tps:.0f}/s"
        else:
            text = f"{tps:.0f}/s"
        
        return f"{text:<{width}}"[:width]

    def _format_time_fixed(self, time_val: Optional[float], width: int) -> str:
        """Format time with fixed width"""
        if time_val is None or time_val == float("inf"):
            return f"{'─':<{width}}"[:width]
        
        if time_val < 1:
            text = f"{time_val:.2f}s"
        elif time_val < 10:
            text = f"{time_val:.1f}s"
        else:
            text = f"{time_val:.0f}s"
        
        return f"{text:<{width}}"[:width]

    def _format_tps(self, tps: Optional[float]) -> str:
        """Format TPS with appropriate styling"""
        if tps is None or tps == 0:
            return "─"
        
        if tps >= 100:
            return f"⚡{tps:.0f}/s"  # Lightning for 100+
        elif tps >= 50:
            return f"🔥{tps:.0f}/s"  # Fire for 50+
        elif tps >= 25:
            return f"📈{tps:.0f}/s"  # Chart for 25+
        else:
            return f"{tps:.0f}/s"    # Plain for low TPS

    def _format_time(self, time_val: Optional[float]) -> str:
        """Format time values consistently"""
        if time_val is None or time_val == float("inf"):
            return "─"
        
        if time_val < 1:
            return f"{time_val:.2f}s"
        elif time_val < 10:
            return f"{time_val:.1f}s"
        else:
            return f"{time_val:.0f}s"

    def _print_insights(self, results: List[ModelResult]):
        """Print performance insights and leader detection - updated for filtered results"""
        config = self.config
        
        if not results or self.theme == DisplayTheme.MINIMAL:
            return
            
        print("─" * config["total_width"])
        
        # Current leader (only among successful models)
        if results:
            leader = results[0]
            if leader.model != self.current_leader:
                self.current_leader = leader.model
                print(f"👑 CURRENT LEADER: {leader.model} • {leader.best_tps:.0f} tok/s peak throughput")
            
            # Performance insights
            insights = []
            if leader.best_tps >= 400:
                insights.append("🚀 EXTREME SPEED! (>400 tok/s)")
            elif leader.best_tps >= 200:
                insights.append("🚀 BLAZING SPEED! (>200 tok/s)")
            elif leader.best_tps >= 100:
                insights.append("⚡ EXCELLENT! (>100 tok/s)")
            elif leader.best_tps >= 50:
                insights.append("🔥 VERY GOOD! (>50 tok/s)")
            
            if leader.best_time and leader.best_time < 2:
                insights.append(f"🏃 SUB-2S RESPONSE!")
            elif leader.first_token and leader.first_token < 0.5:
                insights.append(f"⚡ INSTANT FIRST TOKEN!")
            
            if insights:
                print("💡 " + " • ".join(insights))
            
            # Competition status (only among successful models)
            if len(results) > 1:
                runner_up = results[1]
                gap = ((leader.best_tps - runner_up.best_tps) / runner_up.best_tps) * 100
                if gap < 5:
                    print(f"🔥 TIGHT RACE! {leader.model} leads {runner_up.model} by only {gap:.1f}%")
                elif gap > 50:
                    print(f"🎯 DOMINATING! {leader.model} is {gap:.0f}% faster than {runner_up.model}")
        
        # Show excluded models summary
        all_models = set(self.test_results.keys())
        successful_models = set(r.model for r in results)
        failed_models = all_models - successful_models
        
        if failed_models:
            print(f"⚠️ EXCLUDED FROM RANKINGS: {', '.join(sorted(failed_models))} (no successful runs)")

        print("═" * config["total_width"])

    def show_final_battle_results(self):
        """Show final results - leaderboard is ordered by *average* TPS."""
        print("\n🎉 BENCHMARK BATTLE COMPLETE!")
        print("═" * self.config["total_width"])

        # only models with at least one qualifying (large, successful) run
        model_results = self._collect_model_results()

        if not model_results:
            print("❌ No models completed successfully!")

            # list every model and its failure count
            for model in sorted(self.test_results):
                runs = self.test_results[model]
                failed = len([r for r in runs if not r["success"]])
                print(f"   • {model}: {failed}/{len(runs)} failed")
            print("═" * self.config["total_width"])
            return

        # ── Overall champion ─────────────────────────────────────────────────
        winner = model_results[0]
        print(f"🏆 THROUGHPUT CHAMPION: {winner.model}")
        print(f"   📊 **Average Throughput:** {winner.avg_tps:.0f} tokens/sec")
        print(f"   ⚡ Peak TPS observed:      {winner.best_tps:.0f} tokens/sec")
        print(f"   🚀 Best Time:              {winner.best_time:.2f}s")
        print(f"   ⏱️  Best First Token:       {winner.first_token:.2f}s"
            if winner.first_token else "")
        print(f"   🎯 Success Rate:           {winner.success_rate:.0%}")

        # advantage over runner-up
        if len(model_results) > 1:
            runner = model_results[1]
            adv = (winner.avg_tps - runner.avg_tps) / runner.avg_tps * 100
            print(f"   🥊 {adv:.0f}% higher avg TPS than {runner.model}")

        # ── Podium (top 3) ───────────────────────────────────────────────────
        print("\n🏅 FINAL PODIUM:")
        medals = ["🥇", "🥈", "🥉"]
        for i, r in enumerate(model_results[:3]):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            line  = (f"   {medal} {r.model}: "
                    f"{r.avg_tps:.0f} avg / {r.best_tps:.0f} peak tok/s "
                    f"({r.best_time:.2f}s) "
                    f"[{r.success_rate:.0%} success]")
            print(line)

        # ── Performance categories ───────────────────────────────────────────
        print("\n🎖️ PERFORMANCE CATEGORIES:")

        # fastest overall time
        speed = min(model_results, key=lambda r: r.best_time)
        print(f"   ⚡ Speed Demon: {speed.model} ({speed.best_time:.2f}s)")

        # most consistent (highest success-rate among good-throughput models)
        consistent = max(
            (r for r in model_results if r.avg_tps and r.avg_tps > 50),
            key=lambda r: r.success_rate,
            default=None
        )
        if consistent:
            print(f"   🎯 Consistency Champ: {consistent.model} ({consistent.success_rate:.0%} success)")

        # fastest first token
        token = min(
            (r for r in model_results if r.first_token),
            key=lambda r: r.first_token,
            default=None
        )
        if token:
            print(f"   🚀 First Token Flash: {token.model} ({token.first_token:.2f}s)")

        # ── list models that never produced a qualifying run ────────────────
        failed_models = set(self.test_results) - {r.model for r in model_results}
        if failed_models:
            print("\n❌ MODELS WITH NO SUCCESSFUL RUNS:")
            for model in sorted(failed_models):
                runs = self.test_results[model]
                fails = len([r for r in runs if not r["success"]])
                print(f"   • {model}: {fails}/{len(runs)} failed attempts")

        print("═" * self.config["total_width"])


    # Legacy methods for backward compatibility
    @staticmethod
    def _perf_emoji(best_time: float) -> str:
        """Legacy method for backward compatibility"""
        if best_time < 0.5:
            return "🚀"
        if best_time < 1:
            return "⚡"
        if best_time < 2:
            return "🏃"
        return "🐢"

    def _get_throughput_leader(self) -> Optional[Dict[str, Any]]:
        """Get the model with the highest throughput across all tests"""
        best_tps = 0
        leader_model = None
        leader_test = None
        
        for model, runs in self.test_results.items():
            for run in runs:
                if run["success"] and run.get("throughput", 0) > best_tps:
                    best_tps = run["throughput"]
                    leader_model = model
                    leader_test = run["test_name"]
        
        if leader_model:
            return {"model": leader_model, "tps": best_tps, "test": leader_test}
        return None

# Factory function for easy integration
def create_results_display(models: List[str], theme: str = "detailed") -> CompellingResultsDisplay:
    """Create a results display with the specified theme"""
    return CompellingResultsDisplay(models, theme)

# Legacy alias for backward compatibility
ImprovedResultsDisplay = CompellingResultsDisplay