# benchmarks/utils/results_display.py
"""
Results display and analysis system - Enhanced with backward compatibility
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

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

    def update_model(self, model: str, test_name: str, response_time: float,
                    first_token: Optional[float], end_to_end_tps: Optional[float],
                    sustained_tps: Optional[float], token_count: int,
                    extra_metrics: Dict[str, Any], success: bool,
                    tests_completed: int, total_tests: int):
        """Update model results with better data tracking"""
        
        tps_alias = sustained_tps or end_to_end_tps or 0

        # Store the raw run with enhanced metadata
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
        res["recent_test"] = test_name

        if success:
            if response_time < res["best_time"]:
                res["best_time"] = response_time
            if first_token and first_token < res["best_first_token"]:
                res["best_first_token"] = first_token
            if token_count >= 20 and tps_alias and tps_alias > res["best_throughput"]:
                res["best_throughput"] = tps_alias

        # Calculate average TPS from successful runs
        successful_runs = [r for r in self.test_results[model] if r["success"] and r.get("throughput")]
        if successful_runs:
            res["avg_throughput"] = sum(r["throughput"] for r in successful_runs) / len(successful_runs)

        # Update success rate
        succ_cnt = sum(1 for r in self.test_results[model] if r["success"])
        res["success_rate"] = succ_cnt / tests_completed if tests_completed else 0

        # Update status and trend
        if tests_completed == total_tests:
            if res["success_rate"] == 1:
                res["status"] = "✅ Complete"
            else:
                res["status"] = f"⚠️ {res['success_rate']:.0%} Success"
            res["performance_trend"] = self._get_performance_emoji(res["best_time"], res["best_throughput"])
        else:
            res["status"] = f"🔄 Testing ({tests_completed}/{total_tests})"
            if success:
                res["performance_trend"] = self._get_live_trend_emoji(response_time, tps_alias)

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
            subtitle = "📊 Rankings by PEAK THROUGHPUT (tokens/sec) • Real-time Performance Tracking"
            sub_padding = (width - len(subtitle)) // 2
            print(" " * sub_padding + subtitle)
            print("─" * width)

    def _collect_model_results(self) -> List[ModelResult]:
        """Collect and structure model results for display"""
        throughput_best: Dict[str, Dict[str, Any]] = {}
        
        for model, runs in self.test_results.items():
            successful_runs = [r for r in runs if r["success"] and r.get("throughput")]
            if successful_runs:
                best = max(successful_runs, key=lambda r: r["throughput"])
                avg_tps = sum(r["throughput"] for r in successful_runs) / len(successful_runs)
                throughput_best[model] = {
                    **best,
                    "avg_tps": avg_tps
                }
            else:
                throughput_best[model] = {
                    "throughput": 0,
                    "avg_tps": 0,
                    "response_time": float("inf"),
                    "first_token": None,
                    "test_name": "none"
                }

        # Sort by throughput (descending)
        sorted_models = sorted(
            throughput_best.items(),
            key=lambda x: x[1]["throughput"],
            reverse=True
        )

        # Create structured results
        results = []
        for rank, (model, data) in enumerate(sorted_models, 1):
            model_state = self.results[model]
            
            # Estimate total tests more accurately
            unique_tests = len(set(r["test_name"] for r in self.test_results[model]))
            runs_per_test = 3  # From the benchmark configuration
            estimated_total = unique_tests * runs_per_test if unique_tests > 0 else runs_per_test
            
            results.append(ModelResult(
                model=model,
                rank=rank,
                best_tps=data["throughput"] if data["throughput"] > 0 else None,
                best_time=data["response_time"] if data["response_time"] != float("inf") else None,
                first_token=data.get("first_token"),
                avg_tps=data.get("avg_tps") if data.get("avg_tps", 0) > 0 else None,
                success_rate=model_state["success_rate"],
                tests_completed=model_state["tests_completed"],
                total_tests=estimated_total,
                status=model_state["status"],
                trend_emoji=model_state["performance_trend"],
                recent_test=model_state["recent_test"]
            ))
        
        return results

    def _print_leaderboard_table(self, results: List[ModelResult], highlight_model: str = None):
        """Print the main leaderboard table with perfect fixed-width formatting"""
        config = self.config
        
        # FIXED column widths - these ensure perfect alignment
        rank_width = 4
        model_width = 25
        tps_width = 12
        time_width = 10
        token_width = 10
        progress_width = 12
        status_width = 15

        # Print header with exact same spacing as data rows
        if config["show_avg_tps"] and config["show_progress"]:
            header = f"{'Rank':<{rank_width}} {'Model':<{model_width}} {'Peak TPS':<{tps_width}} {'Avg TPS':<{tps_width}} {'Time':<{time_width}} {'First Tok':<{token_width}} {'Progress':<{progress_width}} {'Status':<{status_width}}"
        elif config["show_progress"]:
            header = f"{'Rank':<{rank_width}} {'Model':<{model_width}} {'Peak TPS':<{tps_width}} {'Time':<{time_width}} {'First Tok':<{token_width}} {'Progress':<{progress_width}} {'Status':<{status_width}}"
        elif config["show_avg_tps"]:
            header = f"{'Rank':<{rank_width}} {'Model':<{model_width}} {'Peak TPS':<{tps_width}} {'Avg TPS':<{tps_width}} {'Time':<{time_width}} {'First Tok':<{token_width}} {'Status':<{status_width}}"
        else:
            header = f"{'Rank':<{rank_width}} {'Model':<{model_width}} {'Peak TPS':<{tps_width}} {'Time':<{time_width}} {'First Tok':<{token_width}} {'Status':<{status_width}}"
            
        print(header)
        print("─" * len(header))

        # Print model rows
        for result in results:
            self._print_model_row(result, highlight_model, config)

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
        if config["show_avg_tps"] and config["show_progress"]:
            avg_tps_display = self._format_tps_fixed_clean(result.avg_tps, tps_width)
            print(f"{rank_display:<{rank_width}} {model_display} {tps_display} {avg_tps_display} {time_display} {token_display} {progress_display} {status_display}")
        elif config["show_progress"]:
            print(f"{rank_display:<{rank_width}} {model_display} {tps_display} {time_display} {token_display} {progress_display} {status_display}")
        elif config["show_avg_tps"]:
            avg_tps_display = self._format_tps_fixed_clean(result.avg_tps, tps_width)
            print(f"{rank_display:<{rank_width}} {model_display} {tps_display} {avg_tps_display} {time_display} {token_display} {status_display}")
        else:
            print(f"{rank_display:<{rank_width}} {model_display} {tps_display} {time_display} {token_display} {status_display}")

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
        """Print performance insights and leader detection"""
        config = self.config
        
        if not results or self.theme == DisplayTheme.MINIMAL:
            return
            
        print("─" * config["total_width"])
        
        # Current leader
        if results:
            leader = results[0]
            if leader.best_tps and leader.best_tps > 0:
                if leader.model != self.current_leader:
                    self.current_leader = leader.model
                    print(f"👑 CURRENT LEADER: {leader.model} • {leader.best_tps:.0f} tok/s peak throughput")
                
                # Performance insights
                insights = []
                if leader.best_tps >= 100:
                    insights.append("🚀 BLAZING SPEED! (>100 tok/s)")
                elif leader.best_tps >= 75:
                    insights.append("⚡ EXCELLENT! (>75 tok/s)")
                elif leader.best_tps >= 50:
                    insights.append("🔥 VERY GOOD! (>50 tok/s)")
                
                if leader.best_time and leader.best_time < 2:
                    insights.append(f"🏃 SUB-2S RESPONSE!")
                elif leader.first_token and leader.first_token < 0.5:
                    insights.append(f"⚡ INSTANT FIRST TOKEN!")
                
                if insights:
                    print("💡 " + " • ".join(insights))
                
                # Competition status
                if len(results) > 1:
                    runner_up = results[1]
                    if runner_up.best_tps and runner_up.best_tps > 0:
                        gap = ((leader.best_tps - runner_up.best_tps) / runner_up.best_tps) * 100
                        if gap < 5:
                            print(f"🔥 TIGHT RACE! {leader.model} leads {runner_up.model} by only {gap:.1f}%")
                        elif gap > 50:
                            print(f"🎯 DOMINATING! {leader.model} is {gap:.0f}% faster than {runner_up.model}")

        print("═" * config["total_width"])

    def show_final_battle_results(self):
        """Show final results with enhanced formatting"""
        print("\n🎉 BENCHMARK BATTLE COMPLETE!")
        print("═" * self.config["total_width"])

        # Get final results
        model_results = self._collect_model_results()
        
        if not model_results:
            print("No results to display.")
            return

        # Throughput champion
        winner = model_results[0]
        if winner.best_tps and winner.best_tps > 0:
            print(f"🏆 THROUGHPUT CHAMPION: {winner.model}")
            print(f"   ⚡ Peak Throughput: {winner.best_tps:.0f} tokens/sec")
            if winner.avg_tps:
                print(f"   📊 Average TPS: {winner.avg_tps:.0f} tokens/sec")
            if winner.best_time:
                print(f"   🚀 Best Time: {winner.best_time:.2f}s")
            if winner.first_token:
                print(f"   ⏱️ Best First Token: {winner.first_token:.2f}s")
            print(f"   🎯 Success Rate: {winner.success_rate:.0%}")

            # Performance advantage
            if len(model_results) > 1:
                runner_up = model_results[1]
                if runner_up.best_tps and runner_up.best_tps > 0:
                    advantage = ((winner.best_tps - runner_up.best_tps) / runner_up.best_tps) * 100
                    print(f"   🎯 {advantage:.0f}% faster than {runner_up.model}")

        # Podium
        print(f"\n🏅 FINAL PODIUM:")
        medals = ["🥇", "🥈", "🥉"]
        for i, result in enumerate(model_results[:3]):
            medal = medals[i] if i < len(medals) else f"{i+1}."
            tps_text = f"{result.best_tps:.0f} tok/s" if result.best_tps else "No data"
            time_text = f"({result.best_time:.2f}s)" if result.best_time else ""
            print(f"   {medal} {result.model}: {tps_text} {time_text}")

        # Performance categories
        print(f"\n🎖️ PERFORMANCE CATEGORIES:")
        
        # Speed demon (best time)
        speed_winner = min((r for r in model_results if r.best_time), 
                          key=lambda r: r.best_time, default=None)
        if speed_winner:
            print(f"   ⚡ Speed Demon: {speed_winner.model} ({speed_winner.best_time:.2f}s)")
        
        # Consistency king (best success rate among top performers)
        consistent_winner = max((r for r in model_results if r.best_tps and r.best_tps > 20), 
                               key=lambda r: r.success_rate, default=None)
        if consistent_winner:
            print(f"   🎯 Consistency King: {consistent_winner.model} ({consistent_winner.success_rate:.0%} success)")
        
        # First token flash (best first token time)
        token_winner = min((r for r in model_results if r.first_token), 
                          key=lambda r: r.first_token, default=None)
        if token_winner:
            print(f"   🚀 First Token Flash: {token_winner.model} ({token_winner.first_token:.2f}s)")

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