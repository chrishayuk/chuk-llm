# chuk-llm Registry Philosophy: Zero-Maintenance Intelligence

## The Problem with Traditional Approaches

Most LLM libraries suffer from the **hardcoding nightmare**:

❌ **Hardcoded model lists** - Need updates for every new release
❌ **Hardcoded capabilities** - Break when providers change features
❌ **Hardcoded costs** - Become stale quickly
❌ **Manual maintenance** - 2700+ lines of YAML/JSON to maintain
❌ **Always outdated** - Can't keep up with weekly model releases

## The chuk-llm Solution: Dynamic Discovery + Smart Inference

### **Tier 1: Provider APIs (Zero Maintenance)** ✅

Query provider endpoints dynamically:

```python
# OpenAI: /v1/models
models = await openai.models.list()

# Gemini: /v1beta/models/{name}
metadata = await gemini.get_model("gemini-2.0-flash")
# Returns: inputTokenLimit, outputTokenLimit, supportedGenerationMethods

# Ollama: /api/tags, /api/show
metadata = await ollama.show("llama3.3")
# Returns: context_length, parameters, families, quantization
```

**Result**: New models automatically discovered, no code changes needed.

### **Tier 2: Smart Inference (Minimal Maintenance)** ✅

Infer capabilities from model name patterns:

```python
# Name: "gpt-4o-mini"
→ Family: GPT-4
→ Context: 128k (GPT-4 standard)
→ Features: tools, vision, streaming, JSON mode
→ Quality: CHEAP (has "mini")
→ Cost: Low (mini variant)

# Name: "claude-3-5-sonnet-20241022"
→ Family: Claude 3
→ Context: 200k (Claude standard)
→ Features: tools, vision, streaming
→ Quality: BEST (sonnet tier)

# Name: "gemini-1.5-flash"
→ Family: Gemini 1.5
→ Context: 1M (Gemini 1.5 standard)
→ Features: tools, vision, JSON mode
→ Quality: CHEAP (flash tier)
```

**Result**: Reasonable defaults for any model, even unknown ones.

### **Tier 3: User Overrides (Optional)** ✅

Only for special cases:

```yaml
# ~/.config/chuk-llm/models.yaml
overrides:
  - provider: "openai"
    pattern: "gpt-5-custom"
    capabilities:
      max_context: 500_000
      cost_per_1m_input: 1.50
      quality_tier: "best"
```

**Result**: Users can patch/extend without touching code.

---

## Resolution Priority (Layered)

```
┌─────────────────────────────────────┐
│  1. HeuristicCapabilityResolver     │  ← Guesses from name patterns
│     (Lowest Priority)                │     Always provides something
└─────────────────────────────────────┘
              ↓ Overrides
┌─────────────────────────────────────┐
│  2. OllamaCapabilityResolver        │  ← Queries /api/show
│     (Medium Priority)                │     For local models only
└─────────────────────────────────────┘
              ↓ Overrides
┌─────────────────────────────────────┐
│  3. GeminiCapabilityResolver        │  ← Queries Google API
│     (High Priority)                  │     For Gemini models only
└─────────────────────────────────────┘
              ↓ Overrides
┌─────────────────────────────────────┐
│  4. YamlCapabilityResolver          │  ← User overrides
│     (Highest Priority - Optional)    │     Only when needed
└─────────────────────────────────────┘
```

Each resolver only overrides fields it **actually knows** about.
Empty/None values don't override previous resolvers.

---

## Maintenance Burden Comparison

### Traditional Approach (LangChain, LiteLLM style)

```python
# models.json - Must update for every release
{
  "gpt-4o": {"context": 128000, "cost": 2.50, ...},
  "gpt-4o-2024-08-06": {"context": 128000, "cost": 2.50, ...},
  "gpt-4o-2024-11-20": {"context": 128000, "cost": 2.50, ...},
  "gpt-4o-mini": {"context": 128000, "cost": 0.15, ...},
  "gpt-4o-mini-2024-07-18": {"context": 128000, "cost": 0.15, ...},
  // ... 300+ more entries
}
```

**Maintenance**: Update file for EVERY new model release (weekly)

### chuk-llm Approach

```python
# No hardcoding - just patterns
"gpt-4o*" → GPT-4o family → context=128k, vision=true, tools=true
```

**Maintenance**: Update patterns only when provider changes architecture (yearly)

---

## Real-World Example: GPT-5 Release

### Traditional Library (Manual Update Required)

```python
# Step 1: Wait for library maintainer to update
# Step 2: Library releases new version
# Step 3: You update your dependency
# Step 4: Finally can use GPT-5
# Total time: Days to weeks
```

### chuk-llm (Automatic)

```python
# Day 1: OpenAI releases GPT-5
registry = await get_registry()  # Queries /v1/models
models = await registry.get_models()  # GPT-5 automatically discovered!

model = await registry.find_model("openai", "gpt-5")
# Capabilities inferred from name:
#   - Family: GPT-5
#   - Context: 200k+ (inferred from GPT-5 generation)
#   - Features: tools, vision, reasoning (all GPT-5 features)
#   - Quality: BEST (flagship model)

# Total time: Zero (immediate)
```

---

## Anti-Patterns We Avoid

❌ **Don't maintain giant model lists**
```python
# Bad
MODELS = [
    "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
    "gpt-4o-mini", "gpt-4o-mini-2024-07-18", ...
]
```

❌ **Don't hardcode capabilities**
```python
# Bad
CAPABILITIES = {
    "gpt-4o": {"context": 128000, "vision": True, ...},
    "gpt-4o-2024-08-06": {"context": 128000, "vision": True, ...},
    ...
}
```

❌ **Don't hardcode costs**
```python
# Bad (stale immediately)
COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    ...
}
```

✅ **Do use patterns**
```python
# Good
if "gpt-4o" in model_name:
    return {"context": 128000, "vision": True, "tools": True}
```

✅ **Do query APIs**
```python
# Good
metadata = await provider.get_model_info(model_name)
```

✅ **Do allow overrides**
```python
# Good
user_config = load_user_overrides()
capabilities.update(user_config)
```

---

## Philosophy Summary

> **"Don't maintain what you can discover.
> Don't hardcode what you can infer.
> Don't update what updates itself."**

1. **Dynamic Discovery** - Query provider APIs
2. **Smart Inference** - Pattern matching for unknowns
3. **Layered Resolution** - Override only when needed
4. **User Extensions** - Allow custom patches

This transforms chuk-llm from "a library that needs constant updates" to **"a self-updating capability intelligence system."**

---

## For Contributors

When adding a new provider:

1. ✅ **Create a ModelSource** - Queries provider's model list API
2. ✅ **Optional: Create a CapabilityResolver** - If provider exposes metadata API
3. ✅ **Add inference patterns** - For when API doesn't provide metadata
4. ❌ **Don't add to YAML** - Only for user overrides

Example:

```python
# 1. Source (discover models)
class NewProviderSource(BaseModelSource):
    async def discover(self) -> list[ModelSpec]:
        # Query provider's /models endpoint
        ...

# 2. Resolver (get capabilities) - OPTIONAL
class NewProviderCapabilityResolver(BaseCapabilityResolver):
    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        # Query provider's /models/{id} endpoint if available
        ...

# 3. Inference patterns (fallback)
# Add to HeuristicCapabilityResolver:
if "newprovider-pro" in name:
    return ModelCapabilities(quality_tier=QualityTier.BEST, ...)
```

That's it. No YAML maintenance required.

---

## Result: Zero-Maintenance Intelligence

- ✅ **New OpenAI models**: Auto-discovered → Auto-inferred
- ✅ **New Anthropic models**: Auto-discovered → Auto-inferred
- ✅ **New Gemini models**: Auto-discovered → API-queried
- ✅ **New Ollama models**: Auto-discovered → API-queried
- ✅ **Price changes**: User's problem, not ours (query provider APIs)
- ✅ **Feature updates**: Inferred from model generation

**The registry maintains itself.**

🚀 **This is the capability brain CHUK deserves.**
