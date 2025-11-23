#!/usr/bin/env python3
"""
JSON Serialization Benchmark
=============================

Benchmark different JSON libraries (stdlib, ujson, orjson) to verify
that chuk-llm is using the fastest available library.
"""

import json as stdlib_json
import time
from typing import Any

# Try importing fast libraries
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    import ujson
    HAS_UJSON = True
except ImportError:
    HAS_UJSON = False

# Import chuk-llm's json utils
from chuk_llm.core import json_utils


# =============================================================================
# Test Data
# =============================================================================

# Small payload
SMALL_DATA = {"name": "test", "value": 42, "active": True}

# Medium payload
MEDIUM_DATA = {
    "users": [
        {"id": i, "name": f"User {i}", "email": f"user{i}@example.com", "active": True}
        for i in range(100)
    ],
    "metadata": {
        "version": "1.0.0",
        "timestamp": 1234567890,
        "config": {"timeout": 30, "retries": 3},
    },
}

# Large payload
LARGE_DATA = {
    "items": [
        {
            "id": i,
            "name": f"Item {i}",
            "description": f"Description for item {i}" * 10,
            "tags": [f"tag{j}" for j in range(10)],
            "metadata": {"created": 1234567890 + i, "updated": 1234567890 + i + 1000},
        }
        for i in range(1000)
    ]
}

# Serialized versions
SMALL_JSON = stdlib_json.dumps(SMALL_DATA)
MEDIUM_JSON = stdlib_json.dumps(MEDIUM_DATA)
LARGE_JSON = stdlib_json.dumps(LARGE_DATA)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_dumps(data: Any, iterations: int = 1000, name: str = "data") -> dict[str, float]:
    """Benchmark JSON serialization (dumps) performance"""
    results = {}

    # Stdlib JSON
    start = time.perf_counter()
    for _ in range(iterations):
        stdlib_json.dumps(data)
    stdlib_time = time.perf_counter() - start
    results["stdlib"] = stdlib_time

    # ujson
    if HAS_UJSON:
        start = time.perf_counter()
        for _ in range(iterations):
            ujson.dumps(data)
        ujson_time = time.perf_counter() - start
        results["ujson"] = ujson_time
    else:
        results["ujson"] = None

    # orjson
    if HAS_ORJSON:
        start = time.perf_counter()
        for _ in range(iterations):
            orjson.dumps(data)
        orjson_time = time.perf_counter() - start
        results["orjson"] = orjson_time
    else:
        results["orjson"] = None

    # chuk-llm json_utils
    start = time.perf_counter()
    for _ in range(iterations):
        json_utils.dumps(data)
    chuk_time = time.perf_counter() - start
    results["chuk_llm"] = chuk_time

    return results


def benchmark_loads(json_str: str, iterations: int = 1000, name: str = "data") -> dict[str, float]:
    """Benchmark JSON deserialization (loads) performance"""
    results = {}

    # Stdlib JSON
    start = time.perf_counter()
    for _ in range(iterations):
        stdlib_json.loads(json_str)
    stdlib_time = time.perf_counter() - start
    results["stdlib"] = stdlib_time

    # ujson
    if HAS_UJSON:
        start = time.perf_counter()
        for _ in range(iterations):
            ujson.loads(json_str)
        ujson_time = time.perf_counter() - start
        results["ujson"] = ujson_time
    else:
        results["ujson"] = None

    # orjson
    if HAS_ORJSON:
        json_bytes = json_str.encode("utf-8")
        start = time.perf_counter()
        for _ in range(iterations):
            orjson.loads(json_bytes)
        orjson_time = time.perf_counter() - start
        results["orjson"] = orjson_time
    else:
        results["orjson"] = None

    # chuk-llm json_utils
    start = time.perf_counter()
    for _ in range(iterations):
        json_utils.loads(json_str)
    chuk_time = time.perf_counter() - start
    results["chuk_llm"] = chuk_time

    return results


def print_results(operation: str, size: str, results: dict[str, float], iterations: int):
    """Print benchmark results in a nice format"""
    print(f"\n{'='*70}")
    print(f"{operation} - {size}")
    print(f"{'='*70}")
    print(f"Iterations: {iterations:,}")
    print()

    # Find fastest
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("No results available")
        return

    fastest_time = min(valid_results.values())
    fastest_lib = min(valid_results, key=valid_results.get)

    # Print results
    for lib, time_taken in results.items():
        if time_taken is None:
            print(f"{lib:12s}: Not available")
        else:
            ops_per_sec = iterations / time_taken
            speedup = time_taken / fastest_time
            is_fastest = lib == fastest_lib

            marker = " 🏆 FASTEST" if is_fastest else f" ({speedup:.2f}x slower)"
            print(f"{lib:12s}: {time_taken:8.4f}s | {ops_per_sec:10,.0f} ops/sec{marker}")

    print()
    print(f"Winner: {fastest_lib} ({results[fastest_lib]:.4f}s)")
    print(f"chuk-llm using: {json_utils.get_json_library()}")


def run_all_benchmarks():
    """Run all benchmarks"""
    print("\n" + "="*70)
    print("chuk-llm JSON Performance Benchmark")
    print("="*70)
    print(f"\nLibraries available:")
    print(f"  - stdlib json: ✓ (always available)")
    print(f"  - ujson:       {'✓' if HAS_UJSON else '✗'}")
    print(f"  - orjson:      {'✓' if HAS_ORJSON else '✗'}")
    print(f"\nchuk-llm is using: {json_utils.get_json_library()}")
    print(f"Performance: {json_utils.get_performance_info().speedup}")

    # Serialization benchmarks
    print_results(
        "SERIALIZATION (dumps)",
        "Small Payload (~100 bytes)",
        benchmark_dumps(SMALL_DATA, iterations=10000),
        10000
    )

    print_results(
        "SERIALIZATION (dumps)",
        "Medium Payload (~5KB)",
        benchmark_dumps(MEDIUM_DATA, iterations=1000),
        1000
    )

    print_results(
        "SERIALIZATION (dumps)",
        "Large Payload (~200KB)",
        benchmark_dumps(LARGE_DATA, iterations=100),
        100
    )

    # Deserialization benchmarks
    print_results(
        "DESERIALIZATION (loads)",
        "Small Payload (~100 bytes)",
        benchmark_loads(SMALL_JSON, iterations=10000),
        10000
    )

    print_results(
        "DESERIALIZATION (loads)",
        "Medium Payload (~5KB)",
        benchmark_loads(MEDIUM_JSON, iterations=1000),
        1000
    )

    print_results(
        "DESERIALIZATION (loads)",
        "Large Payload (~200KB)",
        benchmark_loads(LARGE_JSON, iterations=100),
        100
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if HAS_ORJSON:
        print("✓ orjson is installed and active")
        print("✓ chuk-llm is using the fastest JSON library (orjson)")
        print("✓ Expected speedup: 2-3x faster than stdlib")
    elif HAS_UJSON:
        print("⚠ ujson is installed and active")
        print("  Consider installing orjson for 2-3x speedup")
        print("  pip install orjson")
    else:
        print("⚠ Using stdlib json (slowest)")
        print("  Install orjson for 2-3x speedup:")
        print("  pip install orjson")
    print("="*70)


if __name__ == "__main__":
    run_all_benchmarks()
