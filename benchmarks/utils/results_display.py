# benchmarks/utils/results_display.py
"""
Results display and analysis system - Modified for TPS-based leadership
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional


class CompellingResultsDisplay:
    """Creates a compelling live results experience with TPS-based leadership"""

    # ────────────────────────────────────────────────────────────────────────
    # constructor & helpers
    # ────────────────────────────────────────────────────────────────────────
    def __init__(self, models: List[str]):
        self.models = models
        self.results = {m: self._empty_result() for m in models}
        self.test_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
        self.start_time = time.time()
        self.current_leader: Optional[str] = None

    @staticmethod
    def _empty_result():
        return {
            "status": "⏳ Pending",
            "tests_completed": 0,
            "best_time": float("inf"),
            "best_first_token": float("inf"),
            "best_throughput": 0,
            "success_rate": 0,
            "recent_test": "",
            "performance_trend": "📊",
        }

    # ────────────────────────────────────────────────────────────────────────
    # public  – record a single run
    # ────────────────────────────────────────────────────────────────────────
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
        total_tests: int,
    ):
        """Store a run, tracking best-of stats and adding a `throughput` alias."""

        tps_alias = sustained_tps or end_to_end_tps

        # store the raw run
        self.test_results[model].append(
            {
                "test_name": test_name,
                "response_time": response_time,
                "first_token": first_token,
                "end_to_end_tps": end_to_end_tps,
                "sustained_tps": sustained_tps,
                "throughput": tps_alias,  # <── legacy key used by print code
                "token_count": token_count,
                "extra_metrics": extra_metrics,
                "success": success,
                "timestamp": time.time(),
                "run_id": f"{test_name}_{len([r for r in self.test_results[model] if r['test_name']==test_name])+1}",
            }
        )

        res = self.results[model]
        res["tests_completed"] = tests_completed
        res["recent_test"] = test_name

        if success:
            if response_time < res["best_time"]:
                res["best_time"] = response_time
            if first_token and first_token < res["best_first_token"]:
                res["best_first_token"] = first_token
            # ignore tiny completions when picking max throughput
            if token_count >= 20 and tps_alias and tps_alias > res["best_throughput"]:
                res["best_throughput"] = tps_alias

        succ_cnt = tests_completed if success else tests_completed - 1
        res["success_rate"] = succ_cnt / tests_completed if tests_completed else 0

        if tests_completed == total_tests:
            res["status"] = "✅ Complete" if res["success_rate"] == 1 else f"⚠️ {res['success_rate']:.0%} Success"
            res["performance_trend"] = self._perf_emoji(res["best_time"])
        else:
            res["status"] = f"🔄 Testing ({tests_completed}/{total_tests})"
            if success:
                res["performance_trend"] = (
                    "🚀" if response_time < 1 else "⚡" if response_time < 2 else "🐌"
                )

    # ────────────────────────────────────────────────────────────────────────
    # live leaderboard - MODIFIED FOR TPS-BASED LEADERSHIP
    # ────────────────────────────────────────────────────────────────────────
    def display_live_standings(self, highlight_model: str | None = None):
        """Print the live leaderboard, ranked by BEST THROUGHPUT (TPS)."""
        print("\n")
        elapsed = time.time() - self.start_time
        print(f"🏁 LIVE LEADERBOARD ⏱️  {elapsed:.0f}s")
        print("=" * 85)
        print("📋 Rankings based on BEST THROUGHPUT (tokens/sec) for performance comparison")

        # collect best throughput per model (across all successful runs)
        throughput_best: Dict[str, Dict[str, Any]] = {}
        for model, runs in self.test_results.items():
            successful_runs = [r for r in runs if r["success"] and r.get("throughput")]
            if successful_runs:
                best = max(successful_runs, key=lambda r: r["throughput"])
                throughput_best[model] = best
            else:
                throughput_best[model] = {
                    "throughput": 0,
                    "response_time": float("inf"),
                    "first_token": None,
                    "test_name": "none"
                }

        # sort by throughput (descending - highest TPS first)
        sorted_rows = sorted(
            throughput_best.items(),
            key=lambda x: x[1]["throughput"],
            reverse=True
        )

        # header
        print(f"{'Rank':<6} {'Model':<18} {'Best TPS':<10} {'Time':<10} {'First Tok':<12} {'Status'}")
        print("-" * 85)

        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
        for idx, (model, run) in enumerate(sorted_rows):
            best_tps = run["throughput"]
            rank = "⏳" if best_tps == 0 else (rank_emojis[idx] if idx < len(rank_emojis) else f"{idx+1}️⃣")

            tps_str = "-" if best_tps == 0 else f"{best_tps:.0f}/s"
            time_str = "-" if run["response_time"] == float("inf") else f"{run['response_time']:.2f}s"
            first_tok = run.get("first_token")
            ft_str = "-" if first_tok is None else f"{first_tok:.2f}s"

            model_disp = f"👉 {model}" if model == highlight_model else model
            status_disp = f"{self.results[model]['performance_trend']} {self.results[model]['status']}"

            print(f"{rank:<6} {model_disp:<18} {tps_str:<10} {time_str:<10} {ft_str:<12} {status_disp}")

        print("=" * 85)

        # Update leader detection to use TPS
        leader = self._get_throughput_leader()
        if leader and leader["model"] != self.current_leader:
            self.current_leader = leader["model"]
            print(f"👑 NEW LEADER: {leader['model']} ({leader['tps']:.0f} tok/s best throughput)")
        print()

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

    def _get_creative_test_leader(self) -> Optional[Dict[str, Any]]:
        """Legacy method - kept for compatibility"""
        creative_times = {
            model: min(
                (r["response_time"] for r in runs if r["test_name"] == "creative" and r["success"]),
                default=None,
            )
            for model, runs in self.test_results.items()
        }
        creative_times = {m: t for m, t in creative_times.items() if t is not None}
        if not creative_times:
            return None
        leader = min(creative_times.items(), key=lambda x: x[1])
        return {"model": leader[0], "time": leader[1]}

    # ────────────────────────────────────────────────────────────────────────
    # final output - MODIFIED FOR TPS-BASED LEADERSHIP
    # ────────────────────────────────────────────────────────────────────────
    def show_final_battle_results(self):
        print("\n🎉 BENCHMARK BATTLE COMPLETE!")
        print("=" * 110)

        # TPS-based podium (changed from creative test)
        tps_runs = []
        for model, runs in self.test_results.items():
            successful_runs = [r for r in runs if r["success"] and r.get("throughput")]
            if successful_runs:
                best_run = max(successful_runs, key=lambda r: r["throughput"])
                tps_runs.append((model, best_run))
        
        tps_runs.sort(key=lambda x: x[1]["throughput"], reverse=True)

        if tps_runs:
            winner_model, winner_run = tps_runs[0]
            print(f"🏆 THROUGHPUT CHAMPION: {winner_model}")
            print(f"   ⚡ Best Throughput: {winner_run['throughput']:.0f} tokens/sec")
            print(f"   🚀 Response Time: {winner_run['response_time']:.2f}s")
            print(
                f"   ⏱️ First Token: {winner_run['first_token']:.2f}s"
                if winner_run["first_token"] is not None
                else "   ⏱️ First Token: -"
            )
            print(f"   📝 Test: {winner_run['test_name']}")

            if len(tps_runs) > 1:
                runner_up_tps = tps_runs[1][1]["throughput"]
                if runner_up_tps > 0:
                    speed_adv = (winner_run["throughput"] - runner_up_tps) / runner_up_tps * 100
                    print(f"   🎯 {speed_adv:.0f}% faster than {tps_runs[1][0]} in throughput")

            print(f"\n💡 Recommendation: Use {winner_model} for best throughput performance!")

        print("\n🏅 THROUGHPUT PODIUM:")
        medals = ["🥇", "🥈", "🥉"]
        for idx, (model, run) in enumerate(tps_runs[:3]):
            medal = medals[idx]
            time_info = f" ({run['response_time']:.2f}s)" if run["response_time"] != float("inf") else ""
            print(f"   {medal} {model}: {run['throughput']:.0f} tok/s{time_info}")

        # Also show creative test results for comparison
        creative_runs = [
            (model, min((r for r in runs if r["test_name"] == "creative" and r["success"]), key=lambda r: r["response_time"]))
            for model, runs in self.test_results.items()
            if any(r["test_name"] == "creative" and r["success"] for r in runs)
        ]
        
        if creative_runs:
            creative_runs.sort(key=lambda x: x[1]["response_time"])
            print("\n🎨 CREATIVE TEST SPEED RANKING (for comparison):")
            for idx, (model, run) in enumerate(creative_runs[:3]):
                medal = medals[idx] if idx < 3 else f"{idx+1}."
                tps_info = f" ({run['throughput']:.0f} tok/s)" if run.get("throughput") else ""
                print(f"   {medal} {model}: {run['response_time']:.2f}s{tps_info}")

        print("=" * 110)

        # Call analysis methods if they exist (monkey-patched)
        if hasattr(self, '_show_test_by_test_leaderboards'):
            self._show_test_by_test_leaderboards()
        if hasattr(self, '_show_comprehensive_analysis'):
            self._show_comprehensive_analysis()

    # ────────────────────────────────────────────────────────────────────────
    # tiny helper
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _perf_emoji(best_time: float) -> str:
        if best_time < 0.5:
            return "🚀"
        if best_time < 1:
            return "⚡"
        if best_time < 2:
            return "🏃"
        return "🐢"