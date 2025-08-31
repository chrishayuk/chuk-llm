# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
This project uses `uv` (modern Python package manager) or `pip`:
```bash
# Install dependencies
uv sync  # or pip install -e .

# Install with all features (Redis, enhanced CLI)
uv add chuk_llm[all]  # or pip install -e .[all]
```

### CLI Usage with Dynamic Provider Configuration
The CLI supports dynamic provider configuration via command-line arguments:
```bash
# Override base URL for any provider
chuk-llm ask "Hello" --provider openai --base-url https://api.custom.com/v1 --api-key sk-custom-key

# Use a remote Ollama server
chuk-llm ask "What is Python?" --provider ollama --base-url http://remote-server:11434

# Override API key for testing different accounts
chuk-llm ask "Test prompt" --provider anthropic --api-key sk-test-key-123

# Combine with other options
chuk-llm ask "Generate JSON" --provider openai --base-url https://proxy.api.com/v1 --json --no-stream
```

### Testing
```bash
# Run tests
make test  # or uv run pytest

# Run tests with coverage
make test-cov  # or uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/path/to/test_file.py

# Run tests matching a pattern
uv run pytest -k "test_openai"  # Runs all tests with "test_openai" in the name

# Run specific test class or method
uv run pytest tests/llm/providers/test_openai_client.py::TestOpenAIClient::test_ask
```

### Linting & Formatting
```bash
# Check code quality with ruff
make lint  # or uv run ruff check .

# Auto-format code
make format  # or uv run ruff format .

# Type checking
make typecheck  # or uv run mypy src

# Run all checks (lint, typecheck, test)
make check
```

### Building & Publishing
```bash
# Build the package
make build  # or uv build

# Publish to PyPI
make publish  # or twine upload dist/*

# Clean build artifacts
make clean-all
```

## Architecture Overview

ChukLLM is a unified Python library for Large Language Model providers with a modular, extensible architecture:

### Core Components

1. **API Layer** (`src/chuk_llm/api/`)
   - `core.py`: Main async functions (`ask`, `stream`, `ask_with_tools`)
   - `sync.py`: Synchronous wrappers for all async functions
   - `providers.py`: Dynamic provider function generation (creates 200+ auto-generated functions)
   - `conversation.py`: Stateful conversation management with branching
   - `config.py`: Configuration management and provider switching

2. **LLM Client System** (`src/chuk_llm/llm/`)
   - `client.py`: Factory for creating provider clients
   - `providers/`: Individual provider implementations (OpenAI, Anthropic, Azure, Ollama, etc.)
   - Each provider inherits from `BaseLLMClient` with standardized interfaces for `ask()` and `stream()`
   - `system_prompt_generator.py`: Intelligent prompt generation based on provider/model capabilities

3. **Discovery System** (`src/chuk_llm/llm/discovery/`)
   - `engine.py`: Core discovery engine that coordinates model detection
   - `ollama_discoverer.py`: Discovers local Ollama models and generates functions
   - `azure_openai_discoverer.py`: Discovers Azure OpenAI deployments
   - Automatically creates convenience functions like `ask_ollama_gpt_oss()` for discovered models

4. **Configuration** (`src/chuk_llm/configuration/`)
   - `unified_config.py`: Central configuration loading from YAML
   - `discovery.py`: Manages discovered models and function generation
   - `chuk_llm.yaml`: Provider configurations, model aliases, and feature flags

5. **Session Tracking**
   - Integrated with `chuk-ai-session-manager` for automatic usage tracking
   - Supports memory (default) or Redis storage
   - Tracks messages, tokens, costs automatically

### Key Design Patterns

- **Provider Abstraction**: All providers implement `BaseLLMClient` interface
- **Dynamic Function Generation**: Functions are auto-generated from configuration and discovery
- **Streaming Architecture**: Unified streaming with tool call support across all providers
- **Middleware System**: Request/response middleware for logging, retries, etc.
- **Connection Pooling**: Efficient connection management for high-throughput scenarios

### Provider-Specific Considerations

- **OpenAI/Azure**: Full function calling, vision, JSON mode support. GPT-5 models require special parameter handling (no temperature, max_completion_tokens)
- **Anthropic**: Claude 4 family with enhanced reasoning. Different tool call format than OpenAI
- **Ollama**: Local models with automatic discovery. Special handling for reasoning models like GPT-OSS
- **Google Gemini**: Multimodal support with different parameter names (e.g., `max_output_tokens`)
- **Groq**: Ultra-fast inference but limited model selection
- **Watsonx**: IBM's enterprise solution with Granite models

### Testing Strategy

- Unit tests in `tests/` mirror source structure
- Provider-specific tests in `tests/llm/providers/`
- Diagnostic scripts in `diagnostics/` for debugging provider issues
- Benchmark suite in `benchmarks/` for performance testing

### Common Development Tasks

When adding new features:
1. Check existing provider implementations for patterns
2. Use diagnostic scripts to test provider-specific behavior
3. Ensure compatibility across providers via base class interface
4. Add unit tests following existing test structure
5. Update discovery if adding new model detection logic