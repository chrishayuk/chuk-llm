# Registry Migration Analysis

## Current State: Two Discovery Systems

### Old System (`/llm/discovery/` + `configuration/discovery.py`)
**Location:** `src/chuk_llm/llm/discovery/` + `src/chuk_llm/configuration/discovery.py`

**What it does:**
- Provider-specific discoverers (openai_discoverer.py, anthropic_discoverer.py, etc.)
- Integrated with UnifiedConfigManager via ConfigDiscoveryMixin
- Used by CLI (`chuk-llm discover openai`)
- Used by API (`from chuk_llm.api import discover_models`)
- Reads from `chuk_llm.yaml` for static configuration
- Dynamically discovers models from provider APIs
- Caches discovery results in memory

**Files:**
- `src/chuk_llm/llm/discovery/` (entire directory)
- `src/chuk_llm/configuration/discovery.py`
- `src/chuk_llm/chuk_llm.yaml` (static config)
- Tests in `tests/llm/discovery/`

**Entry points:**
- `chuk_llm.api.discovery.discover_models()`
- `chuk_llm.cli` - `discover_models` command
- `UnifiedConfigManager.get_discovered_models()`

### New System (`/registry/`)
**Location:** `src/chuk_llm/registry/`

**What it does:**
- Clean pydantic-native architecture
- Provider-agnostic sources (OpenAIModelSource, AnthropicModelSource, etc.)
- Layered capability resolvers (Inference → API → YAML cache)
- Persistent disk caching with TTL
- Test-based capability discovery script
- YAML capability caches per provider

**Files:**
- `src/chuk_llm/registry/` (core, sources, resolvers, models, cache)
- `src/chuk_llm/registry/capabilities/*.yaml` (generated caches)
- `scripts/update_capabilities.py` (capability testing script)

**Entry points:**
- `from chuk_llm.registry import get_registry`
- `registry.get_models()`
- `registry.find_best(requires_tools=True, ...)`

## Key Differences

| Feature | Old System | New Registry |
|---------|-----------|--------------|
| **Architecture** | Mixed dict/class | Pure Pydantic |
| **Config** | chuk_llm.yaml (hardcoded) | Dynamic + YAML cache |
| **Capabilities** | Static YAML | Tested + inferred |
| **Caching** | Memory only | Memory + disk (TTL) |
| **Discovery** | Provider-specific classes | Unified sources |
| **Integration** | Tight coupling to config | Standalone |
| **Testing** | No automated testing | scripts/update_capabilities.py |

## Migration Options

### Option 1: Full Replacement (Recommended)
**Replace old discovery system entirely with registry**

Pros:
- ✅ Single source of truth
- ✅ Cleaner architecture
- ✅ Test-based capabilities
- ✅ No duplicate code
- ✅ Better maintainability

Cons:
- ⚠️ Breaking change for existing code
- ⚠️ Need to migrate CLI
- ⚠️ Need to migrate API

**Migration steps:**
1. Update `chuk_llm.api.discovery` to use registry
2. Update `chuk_llm.cli` discover commands to use registry
3. Update `UnifiedConfigManager` to use registry
4. Deprecate old discovery system
5. Remove `chuk_llm/llm/discovery/` after deprecation period
6. Remove `chuk_llm.yaml` (or repurpose for user overrides)

### Option 2: Parallel Systems
**Keep both, use registry internally**

Pros:
- ✅ No breaking changes
- ✅ Gradual migration

Cons:
- ❌ Duplicate maintenance
- ❌ Confusing for users
- ❌ Two sources of truth

**Not recommended** - adds complexity

### Option 3: Registry as Backend
**Replace old discoverer backends with registry calls**

Pros:
- ✅ Minimal API changes
- ✅ Clean internal refactor

Cons:
- ⚠️ Still maintaining old interfaces
- ⚠️ Some duplication

**Compromise approach:**
1. Keep `api/discovery.py` API surface
2. Rewrite internals to use registry
3. Mark old interfaces as deprecated
4. Eventually remove old code

## Recommendation: Option 1 (Full Replacement)

The registry is superior in every way:
- More accurate (test-based)
- Better architecture (pydantic native)
- More maintainable (single system)
- More flexible (layered resolvers)

## Migration Checklist

### Phase 1: API Compatibility Layer
- [ ] Add registry-based implementation to `api/discovery.py`
- [ ] Keep existing API surface (`discover_models()`, etc.)
- [ ] Use registry internally instead of old discoverers

### Phase 2: CLI Migration
- [ ] Update `chuk-llm discover` to use registry
- [ ] Update `chuk-llm list-models` to use registry
- [ ] Add new commands: `chuk-llm cache update`

### Phase 3: Config Integration
- [ ] Add registry to UnifiedConfigManager
- [ ] Deprecate ConfigDiscoveryMixin
- [ ] Keep `chuk_llm.yaml` for user overrides only (optional)

### Phase 4: Cleanup
- [ ] Mark old discovery system as deprecated
- [ ] Add deprecation warnings
- [ ] Remove in next major version
- [ ] Delete `src/chuk_llm/llm/discovery/`

### Phase 5: Documentation
- [ ] Update docs to reference registry
- [ ] Migration guide for users
- [ ] Examples using registry

## Next Steps

1. **Immediate:** Implement Option 1, Phase 1 (API compat layer)
2. **Short-term:** Complete Phases 2-3
3. **Long-term:** Cleanup (Phase 4)

The registry is production-ready and superior - let's migrate!
