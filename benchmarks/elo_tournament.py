# benchmarks/elo_tournament.py
"""
ELO-style Tournament System for LLM Benchmarking - FIXED INTEGRATION
Provides fair, adaptive matchmaking with resilient network/rate limit handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BattleResult(Enum):
    WIN = 1.0
    DRAW = 0.5
    LOSS = 0.0


@dataclass
class ModelStats:
    name: str
    elo_rating: float = 1500.0  # Standard ELO starting rating
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_tps: float = 0.0
    best_tps: float = 0.0
    avg_response_time: float = 0.0
    reliability_score: float = 1.0  # Success rate
    current_streak: int = 0
    last_battle_time: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def avg_tps(self) -> float:
        return self.total_tps / self.games_played if self.games_played > 0 else 0.0

    @property
    def k_factor(self) -> float:
        """Dynamic K-factor based on games played and rating"""
        if self.games_played < 10:
            return 40  # High volatility for new models
        elif self.elo_rating < 1200:
            return 32  # Higher adjustment for lower-rated models
        elif self.elo_rating > 1800:
            return 16  # Lower adjustment for highly-rated models
        else:
            return 24  # Standard adjustment


@dataclass
class BattleRecord:
    round_num: int
    model_a: str
    model_b: str
    test_name: str
    winner: str | None
    model_a_metrics: dict[str, Any]
    model_b_metrics: dict[str, Any]
    battle_result: BattleResult
    elo_change_a: float
    elo_change_b: float
    timestamp: float


class ELOTournament:
    """ELO-based tournament system with adaptive matchmaking and resilience"""

    def __init__(self, models: list[str], test_configs: list[dict]):
        self.models = {name: ModelStats(name) for name in models}
        self.test_configs = test_configs
        self.battle_history: list[BattleRecord] = []
        self.current_round = 0
        self.rate_limit_tracker = {}
        self.network_failure_tracker = {}

        # Tournament parameters
        self.rounds_per_tournament = 50  # Adjustable
        self.battles_per_round = len(models) // 2  # Simultaneous battles
        self.max_retries = 3
        self.base_cooldown = 2.0  # Base cooldown between battles

    def calculate_elo_change(
        self, rating_a: float, rating_b: float, result: BattleResult, k_factor: float
    ) -> tuple[float, float]:
        """Calculate ELO rating changes based on battle result"""
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        change_a = k_factor * (result.value - expected_a)
        change_b = k_factor * ((1 - result.value) - expected_b)

        return change_a, change_b

    def determine_battle_winner(
        self, metrics_a: dict, metrics_b: dict, test_type: str
    ) -> tuple[str | None, BattleResult]:
        """Determine winner based on TPS performance with outlier filtering"""

        # Check if both battles succeeded
        success_a = metrics_a.get("success", False)
        success_b = metrics_b.get("success", False)

        if not success_a and not success_b:
            return None, BattleResult.DRAW
        elif not success_a:
            return metrics_b["model"], BattleResult.LOSS
        elif not success_b:
            return metrics_a["model"], BattleResult.WIN

        # Both succeeded - compare TPS performance (primary metric)
        tps_a = metrics_a.get("sustained_tps") or metrics_a.get("end_to_end_tps", 0)
        tps_b = metrics_b.get("sustained_tps") or metrics_b.get("end_to_end_tps", 0)

        # Filter out outliers (suspiciously high TPS > 500 tok/s)
        if tps_a > 500:
            tps_a = 0
        if tps_b > 500:
            tps_b = 0

        quality_a = self._quality_score(
            metrics_a.get("quality", "excellent")
        )  # Default to excellent if not provided
        quality_b = self._quality_score(metrics_b.get("quality", "excellent"))

        # TPS-focused scoring: 85% TPS, 15% quality (heavily favor TPS)
        score_a = (tps_a * 0.85) + (
            quality_a * 50 * 0.15
        )  # Scale quality to match TPS range
        score_b = (tps_b * 0.85) + (quality_b * 50 * 0.15)

        # Smaller threshold for more decisive TPS-based wins
        diff_threshold = (
            max(score_a, score_b) * 0.03
        )  # 3% difference needed for decisive win

        if abs(score_a - score_b) < diff_threshold:
            return None, BattleResult.DRAW
        elif score_a > score_b:
            return metrics_a["model"], BattleResult.WIN
        else:
            return metrics_b["model"], BattleResult.LOSS

    def _quality_score(self, quality: str) -> float:
        """Convert quality rating to numeric score"""
        quality_map = {
            "excellent": 1.0,
            "good": 0.8,
            "poor": 0.4,
            "truncated": 0.2,
            "failed": 0.0,
            "unknown": 0.5,
        }
        return quality_map.get(quality, 0.5)

    def select_battle_pairs(self) -> list[tuple[str, str]]:
        """Smart matchmaking based on ELO ratings and recent activity"""
        available_models = list(self.models.keys())

        # Sort by ELO rating for balanced matchmaking
        available_models.sort(key=lambda m: self.models[m].elo_rating, reverse=True)

        pairs = []
        used_models = set()

        # Create balanced pairs
        while len(available_models) >= 2 and len(pairs) < self.battles_per_round:
            # Try to match models with similar ratings
            model_a = None

            # Find the highest-rated available model
            for _i, model in enumerate(available_models):
                if model not in used_models:
                    model_a = model
                    break

            if model_a is None:
                break

            # Find best opponent (similar rating, but not too close in recent history)
            rating_a = self.models[model_a].elo_rating
            best_opponent = None
            best_score = float("inf")

            for model in available_models:
                if model == model_a or model in used_models:
                    continue

                rating_diff = abs(self.models[model].elo_rating - rating_a)
                recent_battles = self._count_recent_battles(model_a, model)

                # Prefer similar ratings but avoid recent rematches
                score = rating_diff + (
                    recent_battles * 50
                )  # Penalty for recent battles

                if score < best_score:
                    best_score = score
                    best_opponent = model

            if best_opponent:
                pairs.append((model_a, best_opponent))
                used_models.add(model_a)
                used_models.add(best_opponent)
            else:
                break

        return pairs

    def _count_recent_battles(
        self, model_a: str, model_b: str, recent_rounds: int = 5
    ) -> int:
        """Count battles between two models in recent rounds"""
        count = 0
        recent_threshold = self.current_round - recent_rounds

        for battle in self.battle_history:
            if battle.round_num >= recent_threshold and (
                (battle.model_a == model_a and battle.model_b == model_b)
                or (battle.model_a == model_b and battle.model_b == model_a)
            ):
                count += 1

        return count

    async def execute_battle(
        self, model_a: str, model_b: str, test_config: dict, benchmark_runner
    ) -> BattleRecord | None:
        """Execute a single battle between two models with retry logic"""

        for attempt in range(self.max_retries):
            try:
                # Add progressive delay for rate limiting
                if attempt > 0:
                    delay = self.base_cooldown * (2**attempt) + random.uniform(0, 2)
                    await asyncio.sleep(delay)

                # Execute tests for both models concurrently
                tasks = [
                    benchmark_runner._execute_enhanced_test(
                        "openai", model_a, test_config, True
                    ),
                    benchmark_runner._execute_enhanced_test(
                        "openai", model_b, test_config, True
                    ),
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions
                metrics_a = (
                    results[0]
                    if not isinstance(results[0], Exception)
                    else {"success": False, "error": str(results[0]), "model": model_a}
                )
                metrics_b = (
                    results[1]
                    if not isinstance(results[1], Exception)
                    else {"success": False, "error": str(results[1]), "model": model_b}
                )

                # Determine winner
                winner, battle_result = self.determine_battle_winner(
                    metrics_a, metrics_b, test_config.get("test_type", "unknown")
                )

                # Calculate ELO changes
                k_factor_a = self.models[model_a].k_factor
                k_factor_b = self.models[model_b].k_factor

                # Use average K-factor for the battle
                avg_k_factor = (k_factor_a + k_factor_b) / 2

                elo_change_a, elo_change_b = self.calculate_elo_change(
                    self.models[model_a].elo_rating,
                    self.models[model_b].elo_rating,
                    battle_result,
                    avg_k_factor,
                )

                # Create battle record
                battle_record = BattleRecord(
                    round_num=self.current_round,
                    model_a=model_a,
                    model_b=model_b,
                    test_name=test_config["name"],
                    winner=winner,
                    model_a_metrics=metrics_a,
                    model_b_metrics=metrics_b,
                    battle_result=battle_result,
                    elo_change_a=elo_change_a,
                    elo_change_b=elo_change_b,
                    timestamp=time.time(),
                )

                # Update model stats
                self._update_model_stats(
                    model_a, metrics_a, battle_result, elo_change_a
                )
                self._update_model_stats(
                    model_b,
                    metrics_b,
                    BattleResult(1.0 - battle_result.value),
                    elo_change_b,
                )

                return battle_record

            except Exception as e:
                logger.warning(f"Battle attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Battle failed after {self.max_retries} attempts: {model_a} vs {model_b}"
                    )
                    return None

        return None

    def _update_model_stats(
        self, model_name: str, metrics: dict, result: BattleResult, elo_change: float
    ):
        """Update model statistics after a battle with outlier filtering"""
        stats = self.models[model_name]

        # Update ELO rating
        stats.elo_rating += elo_change
        stats.games_played += 1

        # Update win/loss/draw record
        if result == BattleResult.WIN:
            stats.wins += 1
            stats.current_streak = max(0, stats.current_streak) + 1
        elif result == BattleResult.LOSS:
            stats.losses += 1
            stats.current_streak = min(0, stats.current_streak) - 1
        else:
            stats.draws += 1
            stats.current_streak = 0

        # Update performance metrics with outlier filtering
        if metrics.get("success", False):
            tps = metrics.get("sustained_tps") or metrics.get("end_to_end_tps", 0)

            # Filter outliers - ignore suspiciously high TPS (> 500 tok/s)
            if tps > 0 and tps <= 500:
                stats.total_tps += tps
                stats.best_tps = max(stats.best_tps, tps)

            response_time = metrics.get("total_time", 0)
            if response_time > 0:
                stats.avg_response_time = (
                    (stats.avg_response_time * (stats.games_played - 1)) + response_time
                ) / stats.games_played

        # Update reliability score (exponential moving average)
        success = 1.0 if metrics.get("success", False) else 0.0
        stats.reliability_score = 0.9 * stats.reliability_score + 0.1 * success

        stats.last_battle_time = time.time()

    async def run_tournament_round(self, benchmark_runner) -> list[BattleRecord]:
        """Execute one round of the tournament"""
        self.current_round += 1
        print(f"\n🏟️ TOURNAMENT ROUND {self.current_round}")
        print("=" * 60)

        # Select test for this round
        test_config = random.choice(self.test_configs)
        print(f"📋 Test: {test_config['name']} - {test_config['description']}")

        # Create battle pairs
        pairs = self.select_battle_pairs()
        print(f"⚔️ Battles: {len(pairs)} simultaneous matches")

        # Display matchups
        for i, (model_a, model_b) in enumerate(pairs, 1):
            rating_a = self.models[model_a].elo_rating
            rating_b = self.models[model_b].elo_rating
            print(f"   {i}. {model_a} ({rating_a:.0f}) vs {model_b} ({rating_b:.0f})")

        # Execute battles
        round_battles = []
        battle_tasks = []

        for model_a, model_b in pairs:
            task = self.execute_battle(model_a, model_b, test_config, benchmark_runner)
            battle_tasks.append(task)

        # Wait for all battles to complete
        battle_results = await asyncio.gather(*battle_tasks)

        # Process results
        for battle in battle_results:
            if battle:
                round_battles.append(battle)
                self.battle_history.append(battle)

                # Print battle result
                winner_str = battle.winner if battle.winner else "DRAW"
                print(f"   🥊 {battle.model_a} vs {battle.model_b}: {winner_str}")
                print(
                    f"      ELO: {battle.model_a} {battle.elo_change_a:+.1f}, {battle.model_b} {battle.elo_change_b:+.1f}"
                )

        return round_battles

    def get_leaderboard(self) -> list[tuple[str, ModelStats]]:
        """Get current leaderboard sorted by BEST TPS (primary) then ELO rating (secondary)"""
        return sorted(
            self.models.items(),
            key=lambda x: (x[1].best_tps, x[1].elo_rating),
            reverse=True,
        )

    def display_leaderboard(self):
        """Display current tournament standings ranked by BEST TPS"""
        print(
            f"\n🏆 TOURNAMENT LEADERBOARD (Round {self.current_round}) - RANKED BY BEST TPS"
        )
        print("=" * 110)
        print(
            f"{'Rank':<5} {'Model':<20} {'Best TPS':<10} {'ELO':<8} {'W-L-D':<10} {'Win%':<8} {'Avg TPS':<10} {'Reliability':<12}"
        )
        print("-" * 110)

        leaderboard = self.get_leaderboard()

        for rank, (model_name, stats) in enumerate(leaderboard, 1):
            rank_emoji = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}."

            record = f"{stats.wins}-{stats.losses}-{stats.draws}"
            win_pct = f"{stats.win_rate:.1%}"
            avg_tps = f"{stats.avg_tps:.0f}" if stats.avg_tps > 0 else "-"
            best_tps = f"{stats.best_tps:.0f}" if stats.best_tps > 0 else "-"
            reliability = f"{stats.reliability_score:.1%}"

            # Add streak indicator
            streak_indicator = ""
            if abs(stats.current_streak) >= 3:
                streak_indicator = (
                    f" 🔥{stats.current_streak}"
                    if stats.current_streak > 0
                    else f" ❄️{stats.current_streak}"
                )

            print(
                f"{rank_emoji:<5} {model_name:<20} {best_tps:<10} {stats.elo_rating:.0f}    {record:<10} {win_pct:<8} {avg_tps:<10} {reliability:<12} {streak_indicator}"
            )

        # Highlight the TPS champion
        if leaderboard:
            champion = leaderboard[0]
            print(
                f"\n👑 TPS CHAMPION: {champion[0]} with {champion[1].best_tps:.0f} tokens/second!"
            )

    def generate_tournament_report(self) -> dict[str, Any]:
        """Generate comprehensive tournament report"""
        report = {
            "tournament_summary": {
                "total_rounds": self.current_round,
                "total_battles": len(self.battle_history),
                "models": list(self.models.keys()),
            },
            "final_standings": {},
            "battle_history": [asdict(battle) for battle in self.battle_history],
            "model_stats": {},
            "insights": {},
        }

        # Final standings
        leaderboard = self.get_leaderboard()
        for rank, (model_name, stats) in enumerate(leaderboard, 1):
            report["final_standings"][model_name] = {
                "rank": rank,
                "elo_rating": stats.elo_rating,
                "record": f"{stats.wins}-{stats.losses}-{stats.draws}",
                "win_rate": stats.win_rate,
                "avg_tps": stats.avg_tps,
                "best_tps": stats.best_tps,
                "reliability": stats.reliability_score,
            }

        # Individual model stats
        for model_name, stats in self.models.items():
            report["model_stats"][model_name] = asdict(stats)

        # Generate insights
        if len(leaderboard) > 0:
            champion = leaderboard[0]
            report["insights"] = {
                "champion": champion[0],
                "champion_elo": champion[1].elo_rating,
                "most_reliable": max(
                    self.models.items(), key=lambda x: x[1].reliability_score
                )[0],
                "highest_tps": max(self.models.items(), key=lambda x: x[1].best_tps)[0],
                "most_active": max(
                    self.models.items(), key=lambda x: x[1].games_played
                )[0],
                "closest_rivals": self._find_closest_rivals(),
            }

        return report

    def _find_closest_rivals(self) -> list[tuple[str, str, float]]:
        """Find models with closest ELO ratings"""
        models_list = list(self.models.items())
        rivals = []

        for i in range(len(models_list)):
            for j in range(i + 1, len(models_list)):
                name_a, stats_a = models_list[i]
                name_b, stats_b = models_list[j]
                elo_diff = abs(stats_a.elo_rating - stats_b.elo_rating)
                rivals.append((name_a, name_b, elo_diff))

        rivals.sort(key=lambda x: x[2])
        return rivals[:3]  # Top 3 closest matchups


class ELOBenchmarkRunner:
    """Integration class for running ELO tournaments with existing benchmark infrastructure"""

    def __init__(self, models: list[str], test_configs: list[dict]):
        self.tournament = ELOTournament(models, test_configs)
        self.benchmark_runner = None  # Will be set when running

    async def run_elo_tournament(self, provider: str, rounds: int = 20):
        """Run a complete ELO tournament with FIXED integration"""
        print(f"🏟️ ELO TOURNAMENT: {provider.upper()}")
        print(f"⚔️ Models: {', '.join(self.tournament.models.keys())}")
        print(f"🎯 Tournament: {rounds} rounds of adaptive battles")
        print("🏆 Victory: Highest ELO rating after all rounds\n")

        # Initialize benchmark runner - FIXED: Import and create properly
        from compare_models import EnhancedLiveBenchmarkRunner

        self.benchmark_runner = EnhancedLiveBenchmarkRunner()

        # Run tournament rounds
        for _round_num in range(rounds):
            await self.tournament.run_tournament_round(self.benchmark_runner)

            # Display updated leaderboard
            self.tournament.display_leaderboard()

            # Add delay between rounds
            await asyncio.sleep(2)

        # Generate final report
        print("\n🎉 TOURNAMENT COMPLETE!")
        print("=" * 60)

        report = self.tournament.generate_tournament_report()

        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"elo_tournament_{provider}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Tournament report saved: {filename}")

        return report


# FIXED: Example usage with proper model names and CLI integration
async def run_example_tournament():
    """Example of how to run an ELO tournament"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ELO Tournament for LLM Models")
    parser.add_argument("provider", help="Provider (openai, anthropic, etc.)")
    parser.add_argument("models", help="Comma-separated list of models")
    parser.add_argument(
        "--suite", choices=["quick", "standard", "lightning"], default="quick"
    )
    parser.add_argument(
        "--rounds", type=int, default=15, help="Number of tournament rounds"
    )

    args = parser.parse_args()

    # Parse models
    models = [m.strip() for m in args.models.split(",")]

    # Filter out invalid models for OpenAI
    if args.provider == "openai":
        valid_openai_models = [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
        ]
        filtered_models = [m for m in models if m in valid_openai_models]

        if len(filtered_models) < len(models):
            removed = set(models) - set(filtered_models)
            print(f"⚠️ Removed invalid models: {', '.join(removed)}")
            print(f"✅ Valid models: {', '.join(filtered_models)}")

        models = filtered_models

    if len(models) < 2:
        print("❌ Need at least 2 valid models for a tournament!")
        return

    # Create test configurations based on suite
    if args.suite == "quick":
        test_configs = [
            {
                "name": "speed",
                "description": "Speed test",
                "messages": [{"role": "user", "content": "Say hello briefly."}],
                "max_tokens": 50,
                "temperature": 0,
                "test_type": "speed",
            },
            {
                "name": "reasoning",
                "description": "Logic test",
                "messages": [
                    {
                        "role": "user",
                        "content": "Solve: If A > B and B > C, what's the relationship between A and C?",
                    }
                ],
                "max_tokens": 200,
                "temperature": 0,
                "test_type": "reasoning",
            },
        ]
    elif args.suite == "lightning":
        test_configs = [
            {
                "name": "instant",
                "description": "Ultra-fast response",
                "messages": [{"role": "user", "content": "Ping"}],
                "max_tokens": 10,
                "temperature": 0,
                "test_type": "speed",
            }
        ]
    else:  # standard
        test_configs = [
            {
                "name": "speed",
                "description": "Speed test",
                "messages": [{"role": "user", "content": "Say hello briefly."}],
                "max_tokens": 50,
                "temperature": 0,
                "test_type": "speed",
            },
            {
                "name": "reasoning",
                "description": "Logic test",
                "messages": [
                    {
                        "role": "user",
                        "content": "Solve: If A > B and B > C, what's the relationship between A and C?",
                    }
                ],
                "max_tokens": 200,
                "temperature": 0,
                "test_type": "reasoning",
            },
            {
                "name": "creative",
                "description": "Creative writing",
                "messages": [
                    {"role": "user", "content": "Write a short story about AI."}
                ],
                "max_tokens": 500,
                "temperature": 0.7,
                "test_type": "creative",
            },
        ]

    runner = ELOBenchmarkRunner(models, test_configs)
    await runner.run_elo_tournament(args.provider, rounds=args.rounds)


if __name__ == "__main__":
    asyncio.run(run_example_tournament())
