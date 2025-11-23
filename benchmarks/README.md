# chuk-llm Performance Benchmarks

This directory contains benchmarks for identifying and eliminating performance bottlenecks in chuk-llm.

## Available Benchmarks

### 1. JSON Serialization (`benchmark_json.py`)

Measures the performance of JSON operations across different libraries:
- **stdlib json**: Python's standard library (baseline)
- **ujson**: Ultra-fast JSON encoder/decoder (1.5-2x faster)
- **orjson**: Rust-based JSON library (2-3x faster)
- **chuk-llm**: Our wrapper with fast-path optimizations

**Key Findings:**
- ✅ chuk-llm uses orjson when available (2-3x faster than stdlib)
- ✅ Within 1.02-1.64x of raw orjson performance
- ✅ Optimized with fast paths for common cases (no kwargs)
- ✅ For large payloads, chuk-llm can be faster than raw orjson

**Run:**
```bash
uv run python benchmarks/benchmark_json.py
```

### 2. Message Building (`benchmark_message_building.py`)

Measures the overhead of creating message objects using Pydantic V2:
- Simple text messages
- Multimodal messages (text + images)
- Tool call messages
- Conversations of varying lengths
- Message to dict conversion

**Key Findings:**
- ✅ Simple messages: ~1.9M ops/sec (0.52µs per message)
- ✅ Multimodal messages: ~450K ops/sec (2.23µs)
- ✅ Tool calls: ~700K ops/sec (1.41µs)
- ✅ Dict conversion: ~1.7M ops/sec (0.59µs)
- ✅ Pydantic V2 overhead is negligible

**Run:**
```bash
uv run python benchmarks/benchmark_message_building.py
```

### 3. API to Provider Analysis (`benchmark_api_to_provider.py`)

Traces a complete request from API layer through provider and back:
- Message preparation (API layer)
- Provider initialization
- Message format conversion
- Request parameter building
- Response parsing and transformation
- Streaming chunk processing

**Key Findings:**
- ✅ Message preparation: ~500K-2M ops/sec (FAST)
- ✅ Request building: ~21M ops/sec (VERY FAST)
- ✅ Response parsing: ~1.6M ops/sec with orjson (FAST)
- ✅ Result transformation: ~12M ops/sec (VERY FAST)
- ✅ Streaming chunks: ~21M ops/sec (VERY FAST)
- ✅ Provider initialization: 12ms per client (was 25ms, now 2x faster!)
- ✅ Full request cycle overhead: ~50-140µs

**Optimization Applied:**
- Eliminated duplicate sync client (now async-native only)
- Cut provider initialization time in half (25ms → 12ms)

**Run:**
```bash
uv run python benchmarks/benchmark_api_to_provider.py
```

## Benchmark Results Summary

| Component | Performance | Bottleneck? |
|-----------|-------------|-------------|
| JSON serialization | 2-3x faster than stdlib (orjson) | ✅ Optimized |
| JSON deserialization | 2-3x faster than stdlib (orjson) | ✅ Optimized |
| Message creation | ~2M ops/sec | ✅ Not a bottleneck |
| Message conversion | ~1.7M ops/sec | ✅ Not a bottleneck |
| Provider init | 12ms (was 25ms) | ✅ Optimized (2x faster) |
| Request building | ~21M ops/sec | ✅ Not a bottleneck |
| Response parsing | ~1.6M ops/sec | ✅ Not a bottleneck |
| Full request cycle | ~50-140µs overhead | ✅ Minimal |

## Installation

To run all benchmarks with fast JSON libraries:

```bash
# Install dev dependencies including orjson and ujson
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

## Creating New Benchmarks

When creating a new benchmark:

1. Create a new file `benchmark_<name>.py`
2. Include comprehensive test data
3. Measure both small and large workloads
4. Compare against baseline/alternatives
5. Print clear, actionable results
6. Document findings in this README

## Goals

chuk-llm aims to be the **fastest LLM library** by:

1. ✅ Using the fastest available JSON library (orjson > ujson > stdlib)
2. ✅ Minimizing overhead in message building (Pydantic V2)
3. 🔄 Efficient connection pooling (in progress)
4. 🔄 Optimized streaming performance (to benchmark)
5. 🔄 Fast provider initialization (to benchmark)
6. 🔄 Minimal discovery overhead (to benchmark)

Legend:
- ✅ = Completed and optimized
- 🔄 = In progress or to be benchmarked
