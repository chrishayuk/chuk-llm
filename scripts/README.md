# Capability Cache Update Script

## Overview

`update_capabilities.py` discovers models from provider APIs and optionally tests their capabilities, saving results to YAML cache files.

## Quick Start

### Before Each Release (Recommended)

Discover all models without testing (free, fast):

```bash
python scripts/update_capabilities.py
```

This updates the YAML caches with newly discovered models but doesn't test them (testing costs API credits).

### Testing New Models (Optional)

To test capabilities of new models (costs API credits):

```bash
python scripts/update_capabilities.py --test
```

### Specific Provider

Update only one provider:

```bash
python scripts/update_capabilities.py --provider openai
python scripts/update_capabilities.py --provider anthropic --test
python scripts/update_capabilities.py --provider gemini
python scripts/update_capabilities.py --provider ollama
```

## What It Does

### Discovery Mode (default)
- Queries provider APIs for available models
- Updates `src/chuk_llm/registry/capabilities/{provider}.yaml`
- Preserves existing tested capability data
- **Free** - no API calls beyond model listing

### Testing Mode (`--test`)
For each **new** model discovered, tests:
- ✅ Tool calling support
- ✅ Vision/multimodal support
- ✅ JSON mode support
- ✅ Streaming support
- ✅ Maximum context length (via progressive testing)
- ✅ Supported parameters
- ✅ Speed (tokens/second benchmark)

**⚠️ Testing costs API credits!** Only use when needed.

## Requirements

Set environment variables for providers you want to update:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
# Ollama: ensure running locally at http://localhost:11434
```

Missing keys are handled gracefully - providers without keys are skipped.

## CI/CD Integration

For automated updates:

```bash
python scripts/update_capabilities.py --ci --auto-commit
```

This will:
1. Discover models from all providers
2. Skip testing (no API credits used)
3. Auto-commit changes to git

## Output Files

Cache files are saved to:
```
src/chuk_llm/registry/capabilities/
├── openai.yaml
├── anthropic.yaml
├── gemini.yaml
└── ollama.yaml
```

## Example Workflow

Before a release:

```bash
# 1. Discover all models (free)
python scripts/update_capabilities.py

# 2. Check git diff to see new models
git diff src/chuk_llm/registry/capabilities/

# 3. Optionally test new models for one provider
python scripts/update_capabilities.py --provider openai --test

# 4. Commit the updated caches
git add src/chuk_llm/registry/capabilities/
git commit -m "chore: update capability caches"
```

## Cache Structure

Each YAML file contains:

```yaml
provider: openai
last_updated: '2025-01-20T00:00:00.000000'

families:
  gpt-4o:
    max_context: 128000
    supports_tools: true
    # ... family defaults

models:
  gpt-4o:
    inherits_from: gpt-4o
    tested_at: '2025-01-20T00:00:00.000000'
    max_context: 128000
    tokens_per_second: 45.2
    # ... tested capabilities
```

## Troubleshooting

### "Failed to discover {provider} models"
- Check that API key is set in environment
- For Ollama, ensure it's running: `ollama serve`

### No models being tested
- Use `--test` flag to enable testing
- Without `--test`, only discovery runs (preserves existing tests)

### Tests timing out
- Context length tests may be slow for large contexts
- Speed benchmarks require model response generation
