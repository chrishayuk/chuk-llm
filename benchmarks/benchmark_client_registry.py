#!/usr/bin/env python3
"""
Benchmark client registry performance improvement
=================================================

Measures the performance improvement from using client caching vs recreating clients.
"""
import os
import time

# Set test API key
os.environ["OPENAI_API_KEY"] = "test-key-123"

from chuk_llm.llm.client import get_client
from chuk_llm.client_registry import (
    clear_cache,
    get_cache_stats,
    print_registry_stats,
)


def format_time(seconds: float) -> str:
    """Format time in appropriate unit"""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def benchmark_without_cache(iterations: int = 100) -> tuple[float, float]:
    """Benchmark client creation without caching"""
    print(f"\n{'=' * 70}")
    print(f"WITHOUT CACHING ({iterations} iterations)")
    print(f"{'=' * 70}")

    times = []
    for i in range(iterations):
        start = time.perf_counter()
        client = get_client("openai", model="gpt-4o", use_cache=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    total_time = sum(times)

    print(f"  Average time:  {format_time(avg_time)}")
    print(f"  Min time:      {format_time(min_time)}")
    print(f"  Max time:      {format_time(max_time)}")
    print(f"  Total time:    {format_time(total_time)}")
    print(f"  Operations/s:  {iterations / total_time:.0f}")

    return avg_time, total_time


def benchmark_with_cache(iterations: int = 100) -> tuple[float, float]:
    """Benchmark client creation with caching"""
    print(f"\n{'=' * 70}")
    print(f"WITH CACHING ({iterations} iterations)")
    print(f"{'=' * 70}")

    # Clear cache first
    clear_cache()

    times = []
    for i in range(iterations):
        start = time.perf_counter()
        client = get_client("openai", model="gpt-4o", use_cache=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    total_time = sum(times)

    print(f"  Average time:  {format_time(avg_time)}")
    print(f"  Min time:      {format_time(min_time)}")
    print(f"  Max time:      {format_time(max_time)}")
    print(f"  Total time:    {format_time(total_time)}")
    print(f"  Operations/s:  {iterations / total_time:.0f}")

    # Show cache stats
    print_registry_stats()

    return avg_time, total_time


def benchmark_mixed_workload(
    iterations: int = 100, num_unique_configs: int = 5
) -> tuple[float, float]:
    """Benchmark realistic workload with multiple configs"""
    print(f"\n{'=' * 70}")
    print(f"MIXED WORKLOAD ({iterations} iterations, {num_unique_configs} unique configs)")
    print(f"{'=' * 70}")

    # Clear cache first
    clear_cache()

    # Create multiple model configurations
    models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"]

    times = []
    for i in range(iterations):
        model = models[i % num_unique_configs]
        start = time.perf_counter()
        client = get_client("openai", model=model, use_cache=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    total_time = sum(times)

    print(f"  Average time:  {format_time(avg_time)}")
    print(f"  Min time:      {format_time(min_time)}")
    print(f"  Max time:      {format_time(max_time)}")
    print(f"  Total time:    {format_time(total_time)}")
    print(f"  Operations/s:  {iterations / total_time:.0f}")

    # Show cache stats
    print_registry_stats()

    return avg_time, total_time


def benchmark_cache_overhead() -> float:
    """Measure overhead of cache checking for hits"""
    print(f"\n{'=' * 70}")
    print(f"CACHE HIT OVERHEAD (1000 iterations)")
    print(f"{'=' * 70}")

    # Warm cache
    clear_cache()
    get_client("openai", model="gpt-4o", use_cache=True)

    # Measure cache hits
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        get_client("openai", model="gpt-4o", use_cache=True)
    total_time = time.perf_counter() - start

    avg_time = total_time / iterations

    print(f"  Average time:  {format_time(avg_time)}")
    print(f"  Total time:    {format_time(total_time)}")
    print(f"  Operations/s:  {iterations / total_time:.0f}")

    return avg_time


def main():
    """Run all benchmarks and show comparison"""
    print("\n" + "=" * 70)
    print("CLIENT REGISTRY PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Run benchmarks
    nocache_avg, nocache_total = benchmark_without_cache(iterations=100)
    cache_avg, cache_total = benchmark_with_cache(iterations=100)
    mixed_avg, mixed_total = benchmark_mixed_workload(iterations=100, num_unique_configs=5)
    overhead = benchmark_cache_overhead()

    # Calculate improvements
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    speedup = nocache_avg / cache_avg
    time_saved = nocache_avg - cache_avg
    total_speedup = nocache_total / cache_total

    print(f"\nPer-operation improvement:")
    print(f"  Without cache: {format_time(nocache_avg)}")
    print(f"  With cache:    {format_time(cache_avg)}")
    print(f"  Speedup:       {speedup:.0f}x faster")
    print(f"  Time saved:    {format_time(time_saved)}")

    print(f"\nTotal time improvement (100 operations):")
    print(f"  Without cache: {format_time(nocache_total)}")
    print(f"  With cache:    {format_time(cache_total)}")
    print(f"  Speedup:       {total_speedup:.2f}x faster")
    print(f"  Time saved:    {format_time(nocache_total - cache_total)}")

    print(f"\nMixed workload (5 unique configs, 20 hits each):")
    print(f"  Average time:  {format_time(mixed_avg)}")
    print(f"  Total time:    {format_time(mixed_total)}")

    print(f"\nCache overhead:")
    print(f"  Hit latency:   {format_time(overhead)}")
    print(f"  Negligible:    {overhead < nocache_avg / 1000}")

    # Calculate ROI for different scenarios
    print(f"\n{'=' * 70}")
    print("ROI ANALYSIS")
    print(f"{'=' * 70}")

    scenarios = [
        ("10 clients", 10),
        ("100 clients", 100),
        ("1000 clients", 1000),
        ("10000 clients", 10000),
    ]

    print(f"\nTime savings for identical configs (vs no caching):")
    for name, count in scenarios:
        nocache_time = nocache_avg * count
        cache_time = nocache_avg + (overhead * (count - 1))  # 1 miss + (n-1) hits
        saved = nocache_time - cache_time
        print(f"  {name:20s} saves {format_time(saved):>10s}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
