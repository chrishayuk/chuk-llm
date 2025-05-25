# benchmarks/utils/analysis_methods.py
"""
Analysis methods for CompellingResultsDisplay class - Modified for TPS focus
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Any
from collections import Counter

# helper from benchmark_utils
from benchmarks.utils.benchmark_utils import filter_stable_runs

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _fmt_time(t: float | None) -> str:
    return "-" if t is None or t == float("inf") else f"{t:.2f}s"

def _fmt_tps(tps: float | None) -> str:
    return "-" if tps is None or tps == 0 else f"{tps:.0f}/s"

# ────────────────────────────────────────────────────────────────────────────
# 1.  Per-test leaderboards - ENHANCED FOR TPS
# ────────────────────────────────────────────────────────────────────────────
def show_test_by_test_leaderboards(self):
    """Print a detailed per-test leaderboard for every benchmarked test with TPS emphasis."""
    print("\n📊 TEST-BY-TEST LEADERBOARDS")
    print("=" * 110)
    print("🎯 Each test reveals different model strengths – TPS and speed combined!")
    print("📋 Metrics are from the SAME best-performing run per model per test\n")

    tests = {
        r["test_name"]
        for runs in self.test_results.values()
        for r in runs
        if r["success"]
    }
    tests = sorted(tests)

    descriptions = {
        "speed": "Quick response – raw latency",
        "math": "Mathematical reasoning – accuracy + speed",
        "creative": "Creative generation – quality + throughput",
        "reasoning": "Logical thinking – complex problem solving",
        "code": "Code generation – technical accuracy",
        "instant": "Ultra-fast response – minimal latency",
        "quick_math": "Basic calculations – speed + accuracy",
    }

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]

    for test in tests:
        desc = descriptions.get(test, "Performance test")
        print(f"🧪 {test.upper()} TEST: {desc}")
        print("-" * 100)

        # gather best-run stats per model for this test
        rows: List[Dict[str, Any]] = []
        for model, runs in self.test_results.items():
            relevant = [r for r in runs if r["test_name"] == test and r["success"]]
            if not relevant:
                continue
            
            # For this test, find the run with best TPS (if available), fallback to best time
            if any(r.get("throughput") for r in relevant):
                best_run = max(relevant, key=lambda r: r.get("throughput", 0))
            else:
                best_run = min(relevant, key=lambda r: r["response_time"])
            
            rows.append(
                {
                    "model": model,
                    "best_time": best_run["response_time"],
                    "avg_time": statistics.mean(r["response_time"] for r in relevant),
                    "first_tok": best_run["first_token"],
                    "tps": best_run.get("throughput", 0),
                    "runs": len(relevant),
                    "best_tps": max((r.get("throughput", 0) for r in relevant), default=0),
                }
            )

        # Sort by TPS (descending), then by time (ascending) as tiebreaker
        rows.sort(key=lambda r: (-r["best_tps"], r["best_time"]))

        # header
        print(f"{'Rank':<6}{'Model':<18}{'Best TPS':<12}{'Best Time':<10}{'Avg Time':<10}{'First Token':<12}{'Runs'}")
        print("-" * 100)

        for idx, r in enumerate(rows):
            rank = medals[idx] if idx < len(medals) else f"{idx+1}️⃣"
            tps_disp = _fmt_tps(r["best_tps"])
            print(
                f"{rank:<6}{r['model']:<18}"
                f"{tps_disp:<12}"
                f"{_fmt_time(r['best_time']):<10}"
                f"{_fmt_time(r['avg_time']):<10}"
                f"{_fmt_time(r['first_tok']):<12}"
                f"{r['runs']}"
            )

        if len(rows) > 1:
            win, runner = rows[0], rows[1]
            insights = []
            
            if win["best_tps"] > 0:
                if runner["best_tps"] > 0:
                    tps_adv = (win["best_tps"] - runner["best_tps"]) / runner["best_tps"] * 100
                    insights.append(f"🎯 {win['model']} leads by {tps_adv:.0f}% in TPS")
                else:
                    insights.append(f"🎯 {win['model']} dominates with {win['best_tps']:.0f} tok/s")
            
            time_adv = (runner["best_time"] - win["best_time"]) / win["best_time"] * 100
            insights.append(f"⚡ {time_adv:.0f}% faster in time")
            
            if win["best_time"] < 0.5:
                insights.append("🚀 Sub-500ms!")
            elif win["best_time"] < 1:
                insights.append("⚡ Sub-1s!")
            
            if win["best_tps"] > 100:
                insights.append(f"📈 {win['best_tps']:.0f} tok/s!")
            
            print(" • ".join(insights))

        print()

    # summary - MODIFIED FOR TPS FOCUS
    print("🏆 TEST CHAMPIONS SUMMARY")
    print("-" * 70)
    tps_winners: Dict[str, str] = {}
    time_winners: Dict[str, str] = {}
    
    for test in tests:
        # TPS winners
        best_tps, best_tps_model = 0, None
        best_time, best_time_model = float("inf"), None
        
        for model, runs in self.test_results.items():
            for r in runs:
                if r["test_name"] == test and r["success"]:
                    if r.get("throughput", 0) > best_tps:
                        best_tps, best_tps_model = r["throughput"], model
                    if r["response_time"] < best_time:
                        best_time, best_time_model = r["response_time"], model
        
        if best_tps_model and best_tps > 0:
            tps_winners[test] = best_tps_model
            print(f"🥇 {test.title()} TPS: {best_tps_model} ({best_tps:.0f} tok/s)")
        
        if best_time_model:
            time_winners[test] = best_time_model
            print(f"⚡ {test.title()} Speed: {best_time_model} ({best_time:.2f}s)")

    print()
    if tps_winners:
        champ, wins = Counter(tps_winners.values()).most_common(1)[0]
        print(f"👑 MOST DOMINANT (TPS): {champ} ({wins}/{len(tests)} test wins)")
    
    if time_winners:
        speed_champ, speed_wins = Counter(time_winners.values()).most_common(1)[0]
        print(f"⚡ SPEED DEMON: {speed_champ} ({speed_wins}/{len(tests)} fastest times)")


# ────────────────────────────────────────────────────────────────────────────
# 2.  Comprehensive cross-test summary - ENHANCED FOR TPS
# ────────────────────────────────────────────────────────────────────────────
def show_comprehensive_analysis(self):
    """Print cross-test statistics and recommendations with TPS emphasis."""
    print("\n📈 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 110)
    print("📋 Statistics use metrics from each model's best run per test for fair comparison")

    stats: Dict[str, Dict[str, Any]] = {}
    for model, runs in self.test_results.items():
        if not runs:
            continue

        best_runs: Dict[str, Any] = {}
        for r in runs:
            if not r["success"]:
                continue
            t = r["test_name"]
            # Choose best run by TPS if available, otherwise by time
            if t not in best_runs:
                best_runs[t] = r
            else:
                current_tps = best_runs[t].get("throughput", 0)
                new_tps = r.get("throughput", 0)
                if new_tps > current_tps or (new_tps == current_tps and r["response_time"] < best_runs[t]["response_time"]):
                    best_runs[t] = r

        resp_times = [r["response_time"] for r in best_runs.values()]
        resp_times = filter_stable_runs(resp_times) or resp_times

        first_toks = [r["first_token"] for r in best_runs.values() if r["first_token"]]
        tps_vals = [r["throughput"] for r in best_runs.values() if r.get("throughput")]

        stats[model] = dict(
            success=len(best_runs) / len({r['test_name'] for r in runs}),
            best_time=min(resp_times),
            avg_time=statistics.mean(resp_times),
            median_time=statistics.median(resp_times),
            std_dev=statistics.stdev(resp_times) if len(resp_times) > 1 else 0,
            cv=(statistics.stdev(resp_times) / statistics.mean(resp_times)) if len(resp_times) > 1 else 0,
            best_token=min(first_toks) if first_toks else None,
            avg_token=statistics.mean(first_toks) if first_toks else None,
            best_tps=max(tps_vals) if tps_vals else None,
            avg_tps=statistics.mean(tps_vals) if tps_vals else None,
            median_tps=statistics.median(tps_vals) if tps_vals else None,
        )

    # table - ENHANCED WITH TPS COLUMNS
    print("\n🎯 OVERALL PERFORMANCE SUMMARY (Best Run Per Test):")
    print("-" * 130)
    print(f"{'Model':<18}{'Success':<9}{'Best TPS':<10}{'Avg TPS':<10}{'Best Time':<10}{'Avg Time':<10}{'Std Dev':<9}{'Best Tok':<10}")
    print("-" * 130)

    # Sort by average TPS (descending), then by average time (ascending)
    for model, s in sorted(stats.items(), key=lambda x: (-(x[1]["avg_tps"] or 0), x[1]["avg_time"])):
        best_tps_disp = f"{s['best_tps']:.0f}/s" if s["best_tps"] else "-"
        avg_tps_disp = f"{s['avg_tps']:.0f}/s" if s["avg_tps"] else "-"
        best_tok_disp = f"{s['best_token']:.2f}s" if s["best_token"] else "-"
        
        print(f"{model:<18}{s['success']:.0%}      {best_tps_disp:<10}{avg_tps_disp:<10}"
              f"{s['best_time']:.2f}s   {s['avg_time']:.2f}s   {s['std_dev']:.2f}s   {best_tok_disp:<10}")

    # winners - ENHANCED FOR TPS
    print("\n🏆 PERFORMANCE CATEGORY WINNERS:")
    print("-" * 70)
    if stats:
        # TPS-based winners
        best_peak_tps = max(
            (m for m in stats.items() if m[1]["best_tps"]),
            key=lambda x: x[1]["best_tps"],
            default=None,
        )
        best_avg_tps = max(
            (m for m in stats.items() if m[1]["avg_tps"]),
            key=lambda x: x[1]["avg_tps"],
            default=None,
        )
        
        # Time-based winners
        best_avg_time = min(stats.items(), key=lambda x: x[1]["avg_time"])
        most_consistent = min(stats.items(), key=lambda x: x[1]["cv"])
        fastest_token = min(
            (m for m in stats.items() if m[1]["best_token"] is not None),
            key=lambda x: x[1]["best_token"],
            default=None,
        )

        print(f"🥇 Peak Throughput: {best_peak_tps[0]} ({best_peak_tps[1]['best_tps']:.0f} tok/s)" if best_peak_tps else "🥇 Peak Throughput: None")
        print(f"📈 Best Average TPS: {best_avg_tps[0]} ({best_avg_tps[1]['avg_tps']:.0f} tok/s)" if best_avg_tps else "📈 Best Average TPS: None")
        print(f"⚡ Best Average Time: {best_avg_time[0]} ({best_avg_time[1]['avg_time']:.2f}s)")
        print(f"🎯 Most Consistent: {most_consistent[0]} (CV: {most_consistent[1]['cv']:.1%})")
        if fastest_token:
            print(f"🚀 Fastest First Token: {fastest_token[0]} ({fastest_token[1]['best_token']:.2f}s)")

    # recommendations - ENHANCED FOR TPS
    print("\n💡 USAGE RECOMMENDATIONS:")
    print("-" * 50)
    if stats:
        # Get the model with best balanced performance (good TPS + reasonable time)
        tps_models = [(m, s) for m, s in stats.items() if s.get("avg_tps")]
        if tps_models:
            throughput_model = max(tps_models, key=lambda x: x[1]["avg_tps"])[0]
            print(f"🚀 For High Throughput: Use {throughput_model}")
        
        speed_model = min(stats.items(), key=lambda x: x[1]["best_time"])[0]
        consistent_model = min(stats.items(), key=lambda x: x[1]["cv"])[0]
        
        print(f"⚡ For Speed: Use {speed_model}")
        print(f"🎯 For Reliability: Use {consistent_model}")
        
        # Balanced recommendation
        if tps_models:
            # Score models on both TPS and consistency
            balanced_scores = []
            for model, s in stats.items():
                if s.get("avg_tps") and s["cv"] < 1.0:  # Has TPS data and reasonable consistency
                    # Normalize TPS (0-1) and consistency (1-CV, 0-1), then combine
                    max_tps = max(x[1]["avg_tps"] for x in tps_models)
                    tps_score = s["avg_tps"] / max_tps
                    consistency_score = 1 - min(s["cv"], 1.0)  # Cap CV at 1.0
                    balanced_score = (tps_score * 0.7) + (consistency_score * 0.3)  # 70% TPS, 30% consistency
                    balanced_scores.append((model, balanced_score))
            
            if balanced_scores:
                balanced_model = max(balanced_scores, key=lambda x: x[1])[0]
                print(f"⚖️ For Balanced Performance: Use {balanced_model}")

    print("=" * 110)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Monkey-patch the new methods onto the display class
# ────────────────────────────────────────────────────────────────────────────
from benchmarks.utils.results_display import CompellingResultsDisplay  # noqa: E402

CompellingResultsDisplay._show_test_by_test_leaderboards = show_test_by_test_leaderboards
CompellingResultsDisplay._show_comprehensive_analysis = show_comprehensive_analysis