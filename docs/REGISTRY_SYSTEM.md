# Dynamic Model Registry System

## Overview

The **Model Registry** is chuk-llm's dynamic capability resolution system. It eliminates the need for constant library updates whenever new models are released by:

1. **Dynamically discovering** available models from various sources
2. **Resolving capabilities** through layered resolvers (static data, provider APIs, GGUF introspection)
3. **Intelligently selecting** the best model for a task based on requirements

This transforms chuk-llm from a static client library into **the dynamic capability brain of the CHUK stack**.

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    ModelRegistry                        │
│  Orchestrates discovery + capability resolution         │
└─────────────────────────────────────────────────────────┘
                    ▲                 ▲
                    │                 │
        ┌───────────┴──────┐   ┌─────┴──────────┐
        │  ModelSources    │   │  Resolvers     │
        │  (Discovery)     │   │  (Capabilities)│
        └──────────────────┘   └────────────────┘
             ▲  ▲  ▲                ▲  ▲  ▲
             │  │  │                │  │  │
     ┌───────┘  │  └──────┐    ┌────┘  │  └────┐
     │          │         │    │       │       │
  EnvProvider Ollama  Custom Static Ollama  Custom
   Source     Source  Source  Resolver Resolver Resolver
```

### 1. **ModelSpec** (Identity)

The raw identity of a discovered model:

```python
ModelSpec(
    provider="openai",
    name="gpt-4o-mini",
    family="gpt-4o",
    aliases=["gpt4o-mini"]
)
```

### 2. **ModelCapabilities** (Metadata)

Enriched metadata about what a model can do:

```python
ModelCapabilities(
    max_context=128_000,
    supports_tools=True,
    supports_vision=True,
    supports_json_mode=True,
    known_params={"temperature", "top_p", "max_tokens"},
    input_cost_per_1m=0.15,
    quality_tier=QualityTier.BALANCED,
)
```

### 3. **ModelWithCapabilities**

The complete model information returned by the registry:

```python
ModelWithCapabilities(
    spec=ModelSpec(...),
    capabilities=ModelCapabilities(...)
)
```

---

## Discovery System

### ModelSource Protocol

Sources discover **ModelSpec** objects from various locations.

#### Built-in Sources

**1. EnvProviderSource**

Discovers providers based on environment variables (API keys):

```python
EnvProviderSource(include_ollama=False)
# Checks OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
# Returns default models for each available provider
```

**2. OllamaSource**

Queries the Ollama API for locally installed models:

```python
OllamaSource(base_url="http://localhost:11434")
# Calls GET /api/tags
# Returns all installed Ollama models
```

**3. Custom Sources**

Implement the `ModelSource` protocol:

```python
class MyCustomSource:
    async def discover(self) -> list[ModelSpec]:
        # Your discovery logic
        return [ModelSpec(...), ...]
```

---

## Capability Resolution

### CapabilityResolver Protocol

Resolvers enrich **ModelSpec** objects with **ModelCapabilities**.

Resolvers are **layered** — later resolvers override earlier ones.

#### Built-in Resolvers

**1. StaticCapabilityResolver**

Provides baseline capabilities for ~30 well-known models:

```python
StaticCapabilityResolver()
# OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
# Anthropic: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
# Groq: llama-3.3-70b-versatile, llama-3.1-8b-instant
# Gemini: gemini-2.0-flash-exp, gemini-1.5-pro
# Mistral, DeepSeek, Watsonx, Azure OpenAI
```

**2. OllamaCapabilityResolver**

Queries Ollama's `/api/show` endpoint for GGUF metadata:

```python
OllamaCapabilityResolver()
# Extracts:
# - Context length from num_ctx parameter
# - Vision support from model families (clip, llava)
# - Quality tier from parameter size (7b, 70b, etc.)
```

**3. Custom Resolvers**

Implement the `CapabilityResolver` protocol:

```python
class MyCustomResolver:
    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        if spec.provider != "my_provider":
            return ModelCapabilities()  # No opinion

        # Your resolution logic
        return ModelCapabilities(
            max_context=...,
            supports_tools=...,
        )
```

---

## Using the Registry

### Basic Usage

```python
from chuk_llm import get_registry

# Get the global registry instance
registry = await get_registry()

# Discover all available models
models = await registry.get_models()

for model in models:
    print(f"{model.spec.provider}:{model.spec.name}")
    print(f"  Context: {model.capabilities.max_context}")
    print(f"  Tools: {model.capabilities.supports_tools}")
```

### Intelligent Model Selection

#### Find Best Model by Criteria

```python
# Best model with vision + large context
best = await registry.find_best(
    requires_vision=True,
    min_context=128_000,
    quality_tier="any"
)

# Cheapest model with tools
cheap = await registry.find_best(
    requires_tools=True,
    quality_tier="cheap"
)

# Fastest model (Groq)
fast = await registry.find_best(
    requires_tools=True,
    provider="groq"
)
```

#### Custom Queries

```python
from chuk_llm.registry import ModelQuery

query = ModelQuery(
    requires_tools=True,
    requires_json_mode=True,
    min_context=100_000,
    max_cost_per_1m_input=1.00,
    quality_tier="balanced"
)

results = await registry.query(query)
```

### Find Specific Model

```python
model = await registry.find_model("openai", "gpt-4o-mini")

if model:
    print(f"Context: {model.capabilities.max_context}")
    print(f"Cost: ${model.capabilities.input_cost_per_1m}/1M")
```

---

## Quality Tiers

Models are classified into quality tiers:

```python
class QualityTier(str, Enum):
    BEST = "best"          # Frontier: GPT-4o, Claude 3.5 Sonnet, Gemini Pro
    BALANCED = "balanced"  # Mid-tier: GPT-4o-mini, Claude 3.5 Haiku, Llama 3.3 70B
    CHEAP = "cheap"        # Budget: GPT-3.5, Llama 3.1 8B, local models
    UNKNOWN = "unknown"    # Not yet classified
```

---

## Advanced Usage

### Custom Registry Configuration

```python
from chuk_llm.registry import (
    ModelRegistry,
    StaticCapabilityResolver,
    OllamaCapabilityResolver,
    EnvProviderSource,
    OllamaSource,
)

# Create custom registry with your own sources/resolvers
registry = ModelRegistry(
    sources=[
        EnvProviderSource(),
        OllamaSource(),
        MyCustomSource(),
    ],
    resolvers=[
        StaticCapabilityResolver(),
        OllamaCapabilityResolver(),
        MyCustomResolver(),
    ]
)

models = await registry.get_models()
```

### Force Refresh

```python
# Bypass cache and re-discover
registry = await get_registry(force_refresh=True)
models = await registry.get_models(force_refresh=True)
```

---

## Integration with CHUK Stack

### chuk-ai-planner

```python
# Instead of hardcoding model names:
model = "gpt-4o-mini"

# Use capability-based selection:
registry = await get_registry()
model = await registry.find_best(
    requires_tools=True,
    min_context=128_000,
    cost_tier="cheap"
)
```

### chuk-acp-agent

```python
# Agents declare requirements
agent_llm = await registry.find_best(
    requires_tools=True,
    requires_streaming=True,
    provider="groq"  # Fast for agent loops
)
```

### chuk-tool-processor

```python
# Find models that support specific tool semantics
compatible_models = await registry.query(
    ModelQuery(requires_tools=True, requires_json_mode=True)
)
```

---

## Future Enhancements

### 1. **Performance Lab Integration**

Add measured capabilities from CHUK Performance Lab:

```python
class PerfLabResolver(CapabilityResolver):
    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        # Load measured data from ~/.cache/chuk-llm/perf.json
        perf_data = await self._load_perf_data(spec)

        return ModelCapabilities(
            speed_hint_tps=perf_data.measured_tps,
            input_cost_per_1m=perf_data.actual_cost,  # Cost drift
        )
```

### 2. **Remote Registry (MCP)**

Central capability server for organizations:

```python
class MCPRegistrySource(ModelSource):
    def __init__(self, registry_url: str):
        self.url = registry_url

    async def discover(self) -> list[ModelSpec]:
        # Fetch from central registry
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.url}/models")
            return [ModelSpec(**m) for m in response.json()]
```

### 3. **User Overrides**

```yaml
# ~/.config/chuk-llm/model_overrides.yaml
models:
  - provider: custom_proxy
    name: llama-3-custom
    capabilities:
      max_context: 100000
      supports_tools: true
      input_cost_per_1m: 0.0  # Internal proxy
```

### 4. **Provider API Resolvers**

Dynamic resolution from provider APIs:

```python
class OpenAICapabilityResolver(CapabilityResolver):
    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        if spec.provider != "openai":
            return ModelCapabilities()

        # Fetch from OpenAI /v1/models
        meta = await self._fetch_openai_model_metadata(spec.name)

        return ModelCapabilities(
            max_context=meta.context_length,
            supports_tools=meta.tools,
            supports_vision=meta.vision,
            input_cost_per_1m=meta.pricing.input,
        )
```

---

## Benefits

### ✅ No More Constant Updates

New models are discovered automatically. No need to update chuk-llm.

### ✅ Intelligent Selection

Pick models based on capabilities, not names.

### ✅ Provider Agnostic

Code doesn't depend on specific providers or model names.

### ✅ Extensible

Add custom sources and resolvers without modifying chuk-llm.

### ✅ Cache-Friendly

Registry caches results for fast repeated access.

### ✅ CHUK Design Philosophy

Capabilities = design tokens
Models = components
Resolvers = providers
Registry = theme builder

---

## Example: Complete Workflow

```python
import asyncio
from chuk_llm import get_registry, ask

async def main():
    # 1. Get registry
    registry = await get_registry()

    # 2. Find best model for task
    model = await registry.find_best(
        requires_tools=True,
        min_context=128_000,
        quality_tier="balanced"
    )

    if not model:
        print("No suitable model found")
        return

    print(f"Selected: {model.spec.provider}:{model.spec.name}")
    print(f"  Context: {model.capabilities.max_context:,} tokens")
    print(f"  Cost: ${model.capabilities.input_cost_per_1m}/1M")

    # 3. Use the model
    response = await ask(
        "What is the capital of France?",
        provider=model.spec.provider,
        model=model.spec.name
    )

    print(f"\nResponse: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Testing

```bash
# Run registry tests
pytest tests/test_registry.py -v

# Run demo
python examples/registry_demo.py
```

---

## Summary

The **Model Registry** elevates chuk-llm from a client library to:

> **The intelligent model capability registry of the entire CHUK ecosystem**

It provides:
- Dynamic model discovery
- Layered capability resolution
- Intelligent model selection
- No-update-required extensibility

This aligns perfectly with the **design-system-first** philosophy of CHUK.
