# Discovery System Migration Guide

## Old Discovery Module (Removed)

The old `chuk_llm.llm.discovery` module has been replaced with a new registry-based discovery system.

### Old Files (Removed)
The following example files used the deprecated discovery system and have been removed:
- `universal_discovery.py` ❌ Removed
- `discovery_enhanced.py` ❌ Removed
- `discovery_inference.py` ❌ Removed
- `discovery_inference_diagnostic.py` ❌ Removed
- `demo_azure_discovery_inference.py` ❌ Removed
- `azure_openai_discovery.py` ❌ Removed
- `azure_openai_explorer.py` ❌ Removed
- `ollama_discovery.py` ❌ Removed
- `demo_openai_compatible.py` ❌ Removed

**If you were using these files, please migrate to the new discovery API below.**

## New Discovery System

### Using the New API

```python
from chuk_llm.api.discovery import discover_models

# Discover models for a provider
models = await discover_models("openai", force_refresh=True)

# Models are cached in ~/.cache/chuk-llm/registry_cache.json
# Use force_refresh=False to use cached data
models = await discover_models("openai", force_refresh=False)
```

### Current Example Files

- **`registry_provider_discovery.py`** - Shows the new registry-based discovery
- **`providers/*_usage_examples.py`** - Provider-specific examples using new system
- **`common_demos.py`** - Updated to use new discovery API

### Benefits of New System

1. **Unified API** - Single `discover_models()` function for all providers
2. **Automatic Caching** - Models cached automatically in registry
3. **Capability Detection** - Automatically detects model capabilities (tools, vision, streaming, etc.)
4. **Integration with Config** - Models from registry auto-load into configuration when `models: ["*"]` is used in YAML

### Migration Examples

#### Old Way:
```python
from chuk_llm.llm.discovery.openai_discoverer import OpenAIModelDiscoverer

discoverer = OpenAIModelDiscoverer(
    provider_name="openai",
    api_key=os.getenv("OPENAI_API_KEY")
)
models = await discoverer.discover_models()
```

#### New Way:
```python
from chuk_llm.api.discovery import discover_models

models = await discover_models("openai", force_refresh=True)
```

### Configuration Integration

The new system integrates with the YAML configuration:

```yaml
# src/chuk_llm/chuk_llm.yaml
openai:
  client_class: "chuk_llm.llm.providers.openai_client:OpenAILLMClient"
  api_key_env: "OPENAI_API_KEY"
  api_base: "https://api.openai.com/v1"
  default_model: "gpt-4o-mini"
  models: ["*"]  # Loads from registry cache automatically
```

When `models: ["*"]` is specified, the configuration system automatically loads:
- Model names from the registry cache
- Model capabilities (tools, vision, streaming, JSON mode, etc.)
- Context lengths and other metadata

This enables automatic feature detection without hardcoding model capabilities.
