# chuk-llm

**The intelligent model capability engine.** Production-ready Python library with dynamic model discovery, capability-based selection, real-time streaming, and Pydantic-native architecture.

```python
from chuk_llm import quick_question
print(quick_question("What is 2+2?"))  # "2 + 2 equals 4."
```

## ✨ What's New in v0.14

**Revolutionary Registry System:**
- 🧠 **Dynamic Model Discovery** - No more hardcoded model lists, automatic capability detection
- 🎯 **Intelligent Selection** - Find models by capabilities, cost, and quality tier
- 🔍 **Smart Queries** - `find_best(requires_tools=True, quality_tier="cheap")`
- 🏗️ **Pydantic V2 Native** - Type-safe models throughout, no dictionary goop
- ⚡ **Async-First Architecture** - True async/await with sync wrappers for convenience
- 📊 **Layered Capability Resolution** - Heuristics → YAML cache → Provider APIs
- 🚀 **Zero-Config** - Pull a new Ollama model, use it immediately

**Latest Models (December 2025):**
- 🤖 **Gemini 2.5/3 Pro** - 1M token context, adaptive thinking, multimodal (`gemini-2.5-flash`, `gemini-3-pro-preview`)
- 🚀 **Mistral Large 3** - 675B MoE, 41B active, Apache 2.0 (`mistral-large-2512`, `ministral-8b-2512`, `ministral-14b-2512`)
- 💡 **DeepSeek V3.2** - 671B MoE, ultra-efficient at $0.27/M tokens (`deepseek-chat`, `deepseek-reasoner`)

**Performance:**
- ⚡ **52x faster imports** - Lazy loading reduces import time from 735ms to 14ms
- 🚀 **112x faster client creation** - Automatic thread-safe caching
- 📊 **<0.015% overhead** - Negligible library overhead vs API latency

See [REGISTRY_COMPLETE.md](REGISTRY_COMPLETE.md) for architecture details.

## Why chuk-llm?

- **🧠 Intelligent**: Dynamic registry selects models by capabilities, not names
- **🔍 Auto-Discovery**: Pull new models, use immediately - no configuration needed
- **⚡ Lightning Fast**: Massive performance improvements (see [Performance](#performance))
- **🛠️ Clean Tools API**: Function calling without complexity - tools are just parameters
- **🏗️ Type-Safe**: Pydantic V2 models throughout, no dictionary goop
- **⚡ Async-Native**: True async/await with sync wrappers when needed
- **📊 Built-in Analytics**: Automatic cost and usage tracking with session isolation
- **🎯 Production-Ready**: Thread-safe caching, connection pooling, negligible overhead

## Quick Start

### Installation

```bash
# Core functionality
pip install chuk_llm

# Or with extras
pip install chuk_llm[redis]  # Persistent sessions
pip install chuk_llm[cli]    # Enhanced CLI experience
pip install chuk_llm[all]    # Everything
```

### Basic Usage

```python
# Simplest approach - auto-detects available providers
from chuk_llm import quick_question
answer = quick_question("Explain quantum computing in one sentence")

# Provider-specific (auto-generated functions!)
from chuk_llm import ask_openai_sync, ask_claude_sync, ask_ollama_llama3_2_sync

response = ask_openai_sync("Tell me a joke")
response = ask_claude_sync("Write a haiku")
response = ask_ollama_llama3_2_sync("Explain Python")  # Auto-discovered!
```

### Latest Models (December 2025)

```python
from chuk_llm import ask

# Gemini 3 Pro - Advanced reasoning with 1M context
response = await ask(
    "Explain consciousness vs intelligence in AI",
    provider="gemini",
    model="gemini-3-pro-preview"
)

# Mistral Large 3 - 675B MoE, Apache 2.0
response = await ask(
    "Write a Python function for binary search",
    provider="mistral",
    model="mistral-large-2512"
)

# Ministral 8B - Fast, efficient, cost-effective
response = await ask(
    "Summarize this text",
    provider="mistral",
    model="ministral-8b-2512"
)

# DeepSeek V3.2 - Ultra-efficient at $0.27/M tokens
response = await ask(
    "Solve this math problem step by step",
    provider="deepseek",
    model="deepseek-chat"
)
```

### Async & Streaming

```python
import asyncio
from chuk_llm import ask, stream

async def main():
    # Async call
    response = await ask("What's the capital of France?")
    
    # Real-time streaming
    async for chunk in stream("Write a story"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Function Calling (Tools)

```python
from chuk_llm import ask
from chuk_llm.api.tools import tools_from_functions

def get_weather(location: str) -> dict:
    return {"temp": 22, "location": location, "condition": "sunny"}

# Tools are just a parameter!
toolkit = tools_from_functions(get_weather)
response = await ask(
    "What's the weather in Paris?",
    tools=toolkit.to_openai_format()
)
print(response)  # Returns dict with tool_calls when tools provided
```

### CLI Usage

```bash
# Quick commands with global aliases
chuk-llm ask_gpt "What is Python?"
chuk-llm ask_claude "Explain quantum computing"

# Auto-discovered Ollama models work instantly
chuk-llm ask_ollama_gemma3 "Hello world"
chuk-llm stream_ollama_mistral "Write a long story"

# llama.cpp with automatic model resolution
chuk-llm ask "What is Python?" --provider llamacpp --model qwen3
chuk-llm ask "Count to 5" --provider llamacpp --model llama3.2

# Discover new models
chuk-llm discover ollama
```

## 🧠 Dynamic Registry System

The **registry** is the intelligent core of chuk-llm. Instead of hardcoding model names, it dynamically discovers models and their capabilities, then selects the best one for your needs.

### Intelligent Model Selection

```python
from chuk_llm.registry import get_registry
from chuk_llm import ask

# Get the registry (auto-discovers all available models)
registry = await get_registry()

# Find the best cheap model with tool support
model = await registry.find_best(
    requires_tools=True,
    quality_tier="cheap"
)
print(f"Selected: {model.spec.provider}:{model.spec.name}")
# Selected: groq:llama-3.3-70b-versatile

# Use the selected model with ask()
response = await ask(
    "Summarize this document",
    provider=model.spec.provider,
    model=model.spec.name
)

# Find best model for vision with large context
model = await registry.find_best(
    requires_vision=True,
    min_context=128_000,
    quality_tier="balanced"
)
# Returns: openai:gpt-4o-mini or gemini:gemini-2.0-flash-exp

# Custom queries with multiple requirements
from chuk_llm.registry import ModelQuery

results = await registry.query(ModelQuery(
    requires_tools=True,
    requires_vision=True,
    min_context=100_000,
    max_cost_per_1m_input=2.0,
    quality_tier="balanced"
))
```

### How It Works

**3-Tier Capability Resolution:**

1. **Heuristic Resolver** - Infers capabilities from model name patterns (e.g., "gpt-4" → likely supports tools)
2. **YAML Cache** - Tested capabilities stored in `registry/capabilities/*.yaml` for fast, reliable access
3. **Provider APIs** - Queries provider APIs dynamically (Ollama `/api/tags`, Gemini models API, etc.)

**Dynamic Discovery Sources:**
- OpenAI `/v1/models` API
- Anthropic known models
- Google Gemini models API
- Ollama `/api/tags` (local models)
- llama.cpp `/v1/models` (local GGUF + Ollama bridge)
- DeepSeek `/v1/models` API
- Moonshot AI `/v1/models` API
- Groq, Mistral, Perplexity, and more

Provider APIs are cached on disk and refreshed periodically (or via `chuk-llm discover`), so new models appear without needing a chuk-llm release.

**Benefits:**
- ✅ **No hardcoded model lists** - Pull new Ollama models, use immediately
- ✅ **Capability-based selection** - Declare requirements, not model names
- ✅ **Cost-aware** - Find cheapest model that meets requirements
- ✅ **Quality tiers** - BEST, BALANCED, CHEAP classification
- ✅ **Extensible** - Add custom sources and resolvers via protocols

## Key Features

### 🔍 Automatic Model Discovery

Pull new Ollama models and use them immediately - no configuration needed:

```bash
# Terminal 1: Pull a new model
ollama pull llama3.2
ollama pull mistral-small:latest

# Terminal 2: Use immediately in Python
from chuk_llm import ask_ollama_llama3_2_sync, ask_ollama_mistral_small_latest_sync
response = ask_ollama_llama3_2_sync("Hello!")

# Or via CLI
chuk-llm ask_ollama_mistral_small_latest "Tell me a joke"
```

### 🦙 llama.cpp Integration

Run local GGUF models with advanced control via llama.cpp server. **Reuse Ollama's downloaded models** without re-downloading!

**CLI Usage** (✨ Now fully supported!):

```bash
# Simple usage - model names automatically resolve to GGUF files
chuk-llm ask "What is Python?" --provider llamacpp --model qwen3
chuk-llm ask "Count to 5" --provider llamacpp --model llama3.2

# Streaming (default)
chuk-llm ask "Write a story" --provider llamacpp --model qwen3

# Non-streaming
chuk-llm ask "Quick question" --provider llamacpp --model qwen3 --no-stream
```

**Python API** (Simple - Recommended):

```python
from chuk_llm import ask

# Model names automatically resolve to Ollama's GGUF files!
response = await ask(
    "What is Python?",
    provider="llamacpp",
    model="qwen3"  # Auto-resolves to ~/.ollama/models/blobs/sha256-xxx
)
print(response)

# Streaming
from chuk_llm import stream
async for chunk in stream("Tell me a story", provider="llamacpp", model="llama3.2"):
    print(chunk, end="", flush=True)
```

**Python API** (Advanced - Full Control):

```python
from chuk_llm.registry.resolvers.llamacpp_ollama import discover_ollama_models
from chuk_llm.llm.providers.llamacpp_client import LlamaCppLLMClient
from chuk_llm.core import Message, MessageRole

# Discover Ollama models (finds GGUF blobs in ~/.ollama/models/blobs/)
models = discover_ollama_models()
print(f"Found {len(models)} Ollama models")  # e.g., "Found 48 Ollama models"

# Create client with auto-managed server
client = LlamaCppLLMClient(
    model=str(models[0].gguf_path),  # Reuse Ollama's GGUF!
    ctx_size=8192,
    n_gpu_layers=-1,  # Use all GPU layers
)

messages = [Message(role=MessageRole.USER, content="Hello!")]
result = await client.create_completion(messages=messages)
print(result["response"])

# Cleanup
await client.stop_server()
```

**Key Features:**
- ✅ **CLI Support** - Full integration with chuk-llm CLI (model name resolution)
- ✅ **Ollama Bridge** - Automatically discovers and reuses Ollama's downloaded models (no re-download!)
- ✅ **Auto-Resolution** - Model names (qwen3, llama3.2) resolve to GGUF file paths automatically
- ✅ **Process Management** - Auto-managed server lifecycle (start/stop/health checks)
- ✅ **OpenAI-Compatible** - Uses standard OpenAI client (streaming, tools, etc.)
- ✅ **High Performance** - Benchmarks show llama.cpp is 1.53x faster than Ollama (311 vs 204 tok/s)
- ✅ **Advanced Control** - Custom sampling, grammars, GPU layers, context size
- ✅ **Cross-Platform** - Works on macOS, Linux, Windows

**Performance Comparison** (same GGUF file, qwen3:0.6b):
- llama.cpp: 311.4 tok/s
- Ollama: 204.2 tok/s
- **llama.cpp is 1.53x faster!**

See `examples/providers/llamacpp_ollama_usage_examples.py` and `examples/providers/benchmark_ollama_vs_llamacpp.py` for full examples.

### 📊 Automatic Session Tracking

Every call is automatically tracked for analytics:

```python
from chuk_llm import ask_sync, get_session_stats

ask_sync("What's the capital of France?")
ask_sync("What's 2+2?")

stats = get_session_stats()
print(f"Total cost: ${stats['estimated_cost']:.6f}")
print(f"Total tokens: {stats['total_tokens']}")
```

### 🎭 Stateful Conversations

Build conversational AI with memory:

```python
from chuk_llm import conversation

async with conversation() as chat:
    await chat.ask("My name is Alice")
    response = await chat.ask("What's my name?")
    # AI responds: "Your name is Alice"
```

### ⚡ Concurrent Execution

Run multiple queries in parallel for massive speedups:

```python
import asyncio
from chuk_llm import ask

# 3-7x faster than sequential!
responses = await asyncio.gather(
    ask("What is AI?"),
    ask("Capital of Japan?"),
    ask("Meaning of life?")
)
```

## Supported Providers

All providers are **dynamically discovered** via the registry system - no hardcoded model lists!

| Provider | Discovery Method | Special Features | Status |
|----------|-----------------|-----------------|--------|
| **OpenAI** | `/v1/models` API | GPT-5 / GPT-5.1, o3-family reasoning, industry standard | ✅ Dynamic |
| **Azure OpenAI** | Deployment config | SOC2, HIPAA compliant, VNet, multi-region | ✅ Dynamic |
| **Anthropic** | Known models† | Claude 3.5 Sonnet, advanced reasoning, 200K context | ✅ Static |
| **Google Gemini** | Models API | Gemini 2.5/3 Pro, 1M token context, adaptive thinking, multimodal | ✅ Dynamic |
| **Groq** | `/v1/models` API | Llama 3.3, ultra-fast (our benchmarks: ~526 tok/s) | ✅ Dynamic |
| **Ollama** | `/api/tags` | Any local model, auto-discovery, offline, privacy | ✅ Dynamic |
| **llama.cpp** | `/v1/models` | Local GGUF models, Ollama bridge, advanced control | ✅ Dynamic |
| **IBM watsonx** | Known models† | Granite 3.3, enterprise, on-prem, compliance | ✅ Static |
| **Perplexity** | Known models† | Sonar, real-time web search, citations | ✅ Static |
| **Mistral** | Known models† | Large 3 (675B MoE), Ministral 3 (3B/8B/14B), Apache 2.0 | ✅ Static |
| **DeepSeek** | `/v1/models` API | DeepSeek V3.2 (671B MoE), ultra-efficient, $0.27/M tokens | ✅ Dynamic |
| **Moonshot AI** | `/v1/models` API | Kimi K2, 256K context, coding, Chinese language | ✅ Dynamic |
| **OpenRouter** | Known models† | Access to 100+ models via single API | ✅ Static |

† Static = discovered from curated model list + provider docs, not via `/models` endpoint

**Capabilities** (auto-detected by registry):
- ✅ Streaming responses
- ✅ Function calling / tool use
- ✅ Vision / multimodal inputs
- ✅ JSON mode / structured outputs
- ✅ Async and sync interfaces
- ✅ Automatic client caching
- ✅ Session tracking
- ✅ Conversation management

## Configuration

### Environment Variables

```bash
# API Keys - Cloud Providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."        # For Gemini 2.5/3 models
export GROQ_API_KEY="..."
export DEEPSEEK_API_KEY="..."      # For DeepSeek V3.2 (chat/reasoner)
export MOONSHOT_API_KEY="..."
export MISTRAL_API_KEY="..."       # For Mistral Large 3 & Ministral 3

# Azure Configuration
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# Local Servers
# (No API keys needed for Ollama or llama.cpp)

# Session Storage (optional)
export SESSION_PROVIDER=redis  # Default: memory
export SESSION_REDIS_URL=redis://localhost:6379/0

# Performance Settings
export CHUK_LLM_CACHE_CLIENTS=1      # Enable client caching (default: 1)
export CHUK_LLM_AUTO_DISCOVER=true   # Auto-discover new models (default: true)
```

### Python Configuration

```python
from chuk_llm import configure

configure(
    provider="azure_openai",
    model="gpt-4o-mini",
    temperature=0.7
)

# All subsequent calls use these settings
response = ask_sync("Hello!")
```

### Client Caching (Advanced)

Automatic client caching is enabled by default for maximum performance:

```python
from chuk_llm.llm.client import get_client

# First call creates client (~12ms)
client1 = get_client("openai", model="gpt-4o")

# Subsequent calls return cached instance (~125µs)
client2 = get_client("openai", model="gpt-4o")
assert client1 is client2  # Same instance!

# Disable caching for specific call
client3 = get_client("openai", model="gpt-4o", use_cache=False)

# Monitor cache performance
from chuk_llm.client_registry import print_registry_stats
print_registry_stats()
# Cache statistics:
# - Total clients: 1
# - Cache hits: 1
# - Cache misses: 1
# - Hit rate: 50.0%
```

## Advanced Features

### 🛠️ Function Calling / Tool Use

ChukLLM provides a clean, unified API for function calling. **Recommended approach**: Use the `Tools` class for automatic execution.

```python
from chuk_llm import Tools, tool

# Recommended: Class-based tools with auto-execution
class MyTools(Tools):
    @tool(description="Get weather for a city")
    def get_weather(self, location: str) -> dict:
        return {"temp": 22, "location": location, "condition": "sunny"}

    @tool  # Description auto-extracted from docstring
    def calculate(self, expr: str) -> float:
        """Evaluate a mathematical expression"""
        return eval(expr)

# Auto-executes tools and returns final response
tools = MyTools()
response = await tools.ask("What's the weather in Paris and what's 2+2?")
print(response)  # "The weather in Paris is 22°C and sunny. 2+2 equals 4."

# Sync version
response = tools.ask_sync("Calculate 15 * 4")
print(response)  # "15 * 4 equals 60"
```

**Alternative: Direct API usage** (for more control):

```python
from chuk_llm import ask
from chuk_llm.api.tools import tools_from_functions

def get_weather(location: str) -> dict:
    """Get weather information for a location"""
    return {"temp": 22, "location": location}

# Create toolkit
toolkit = tools_from_functions(get_weather)

# Returns dict with tool_calls - you handle execution
response = await ask(
    "What's the weather in Paris?",
    tools=toolkit.to_openai_format()
)
print(response)  # {"response": "...", "tool_calls": [...]}
```

#### Streaming with Tools

```python
from chuk_llm import stream

# Streaming with tools
async for chunk in stream(
    "What's the weather in Tokyo?", 
    tools=toolkit.to_openai_format(),
    return_tool_calls=True  # Include tool calls in stream
):
    if isinstance(chunk, dict):
        print(f"Tool call: {chunk['tool_calls']}")
    else:
        print(chunk, end="", flush=True)
```

<details>
<summary><b>🌳 Conversation Branching</b></summary>

```python
async with conversation() as chat:
    await chat.ask("Planning a vacation")
    
    # Explore different options
    async with chat.branch() as japan_branch:
        await japan_branch.ask("Tell me about Japan")
    
    async with chat.branch() as italy_branch:
        await italy_branch.ask("Tell me about Italy")
    
    # Main conversation unaffected by branches
    await chat.ask("I'll go with Japan!")
```
</details>

<details>
<summary><b>📈 Provider Comparison</b></summary>

```python
from chuk_llm import compare_providers

results = compare_providers(
    "Explain quantum computing",
    ["openai", "anthropic", "groq", "ollama"]
)

for provider, response in results.items():
    print(f"{provider}: {response[:100]}...")
```
</details>

<details>
<summary><b>🎯 Intelligent System Prompts</b></summary>

ChukLLM automatically generates optimized system prompts based on provider capabilities:

```python
# Each provider gets optimized prompts
response = ask_claude_sync("Help me code", tools=tools)
# Claude gets: "You are Claude, an AI assistant created by Anthropic..."

response = ask_openai_sync("Help me code", tools=tools)  
# OpenAI gets: "You are a helpful assistant with function calling..."
```
</details>

## CLI Commands

```bash
# Quick access to any model
chuk-llm ask_gpt "Your question"
chuk-llm ask_claude "Your question"
chuk-llm ask_ollama_llama3_2 "Your question"

# llama.cpp with automatic model resolution
chuk-llm ask "Your question" --provider llamacpp --model qwen3
chuk-llm ask "Your question" --provider llamacpp --model llama3.2

# Discover and test
chuk-llm discover ollama        # Find new models
chuk-llm test llamacpp          # Test llamacpp provider
chuk-llm test azure_openai      # Test connection
chuk-llm providers              # List all providers
chuk-llm models ollama          # Show available models
chuk-llm functions              # List all generated functions

# Advanced usage
chuk-llm ask "Question" --provider azure_openai --model gpt-4o-mini --json
chuk-llm ask "Question" --provider llamacpp --model qwen3 --no-stream
chuk-llm ask "Question" --stream --verbose

# Function calling / Tool use from CLI
chuk-llm ask "Calculate 15 * 4" --tools calculator_tools.py
chuk-llm stream "What's the weather?" --tools weather_tools.py --return-tool-calls

# Zero-install with uvx
uvx chuk-llm ask_claude "Hello world"
uvx chuk-llm ask "Question" --provider llamacpp --model qwen3
```

## Performance

chuk-llm is designed for production with negligible overhead:

### Key Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| Import | 14ms | 52x faster than eager loading |
| Client creation (cached) | 125µs | 112x faster, thread-safe |
| Request overhead | 50-140µs | <0.015% of typical API call |

### Production Features

- **Automatic client caching** - Thread-safe, 112x faster repeated operations
- **Lazy imports** - Only load what you use
- **Connection pooling** - Efficient HTTP/2 reuse
- **Async-native** - Built on asyncio for maximum throughput
- **Smart caching** - Model discovery results cached on disk

### Benchmarks

Run comprehensive benchmarks:
```bash
uv run python benchmarks/benchmark_client_registry.py
uv run python benchmarks/llm_benchmark.py
```

**See [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) for detailed analysis and micro-benchmarks.**

## Architecture

ChukLLM uses a **registry-driven, async-native architecture** optimized for production use:

### 🏗️ Core Design Principles

1. **Dynamic Registry** - Models discovered and selected by capabilities, not names
2. **Pydantic V2 Native** - Type-safe models throughout, no dictionary goop
3. **Async-First** - Built on asyncio with sync wrappers for convenience
4. **Stateless Clients** - Clients don't store conversation history; your application manages state
5. **Lazy Loading** - Modules load on-demand for instant imports (14ms)
6. **Automatic Caching** - Thread-safe client registry eliminates duplicate initialization

### 🔄 Request Flow

```
User Code
    ↓
import chuk_llm (14ms - lazy loading)
    ↓
get_client() (2µs - cached registry lookup)
    ↓
[Cached Client Instance]
    ↓
async ask() (~50µs - minimal overhead)
    ↓
Provider SDK (~50µs - efficient request building)
    ↓
HTTP Request (50-500ms - network I/O)
    ↓
Response Parsing (~50µs - orjson)
    ↓
Return to User

Total chuk-llm Overhead: ~150µs (<0.015% of API call)
```

### 🔐 Session Isolation

**Important:** Conversation history is **NOT** shared between calls. Each conversation is independent:

```python
from chuk_llm.llm.client import get_client
from chuk_llm.core.models import Message

client = get_client("openai", model="gpt-4o")

# Conversation 1
conv1 = [Message(role="user", content="My name is Alice")]
response1 = await client.create_completion(conv1)

# Conversation 2 (completely separate)
conv2 = [Message(role="user", content="What's my name?")]
response2 = await client.create_completion(conv2)
# AI won't know the name - conversations are isolated!
```

**Key Insights:**
- ✅ Clients are stateless (safe to cache and share)
- ✅ Conversation state lives in YOUR application
- ✅ HTTP sessions shared for performance (connection pooling)
- ✅ No cross-conversation or cross-user leakage
- ✅ Thread-safe for concurrent use

See [CONVERSATION_ISOLATION.md](CONVERSATION_ISOLATION.md) for detailed architecture.

### 📦 Module Organization

```
chuk-llm/
├── api/                      # Public API (ask, stream, conversation)
├── registry/                 # ⭐ Dynamic model registry (THE BRAIN)
│   ├── core.py              # ModelRegistry orchestrator
│   ├── models.py            # Pydantic models (ModelSpec, ModelCapabilities)
│   ├── sources/             # Discovery sources (OpenAI, Ollama, Gemini, etc.)
│   └── resolvers/           # Capability resolvers (Heuristic, YAML, APIs)
├── core/                     # Pydantic V2 models (Message, Tool, ContentPart)
│   ├── models.py            # Core Pydantic models
│   ├── enums.py             # Type-safe enums (Provider, Feature, etc.)
│   └── constants.py         # Constants
├── llm/
│   ├── providers/           # 15+ provider implementations
│   ├── client.py            # Client factory with registry integration
│   └── features.py          # Feature detection
├── configuration/           # Unified configuration system
└── client_registry.py       # Thread-safe client caching
```

## Used by the CHUK Stack

chuk-llm is the **canonical LLM layer** for the entire CHUK ecosystem:

- **chuk-ai-planner** uses the registry to select planning vs drafting models by capability
- **chuk-acp-agent** uses capability-based policies per agent (e.g., "requires tools + 128k context")
- **chuk-mcp-remotion** uses it to pick video-script models with vision + long context

Instead of hardcoding "use GPT-4o", CHUK components declare **what they need**, and the registry finds the best available model.

## Documentation

- 📚 [Full Documentation](https://github.com/chrishayuk/chuk-llm/wiki)
- 🎯 [Examples (33)](https://github.com/chrishayuk/chuk-llm/tree/main/examples)
- ⚡ [Performance Optimizations](PERFORMANCE_OPTIMIZATIONS.md)
- 🗄️ [Client Registry](CLIENT_REGISTRY.md)
- 🔄 [Lazy Imports](LAZY_IMPORTS.md)
- 🔐 [Conversation Isolation](CONVERSATION_ISOLATION.md)
- 📊 [Registry System](REGISTRY_COMPLETE.md)
- 🔧 [Debug Tools](examples/debug/README.md) - Test OpenAI-compatible API capabilities
- 🏗️ [Migration Guide](https://github.com/chrishayuk/chuk-llm/wiki/migration)
- 🤝 [Contributing](https://github.com/chrishayuk/chuk-llm/blob/main/CONTRIBUTING.md)

## Quick Comparison

| Feature | chuk-llm | LangChain | LiteLLM | OpenAI SDK |
|---------|----------|-----------|---------|------------|
| Import speed | ⚡ 14ms | 🐌 1-2s | 🐌 500ms+ | ⚡ Fast |
| Client caching | ✅ Auto (112x) | ❌ | ❌ | ❌ |
| Auto-discovery | ✅ | ❌ | ❌ | ❌ |
| Native streaming | ✅ | ⚠️ | ✅ | ✅ |
| Function calling | ✅ Clean API | ✅ Complex | ⚠️ Basic | ✅ |
| Session tracking | ✅ Built-in | ⚠️ Manual | ❌ | ❌ |
| Session isolation | ✅ Guaranteed | ⚠️ Varies | ⚠️ Unclear | ⚠️ Manual |
| CLI included | ✅ | ❌ | ⚠️ Basic | ❌ |
| Provider functions | ✅ Auto-generated | ❌ | ❌ | ❌ |
| Conversations | ✅ Branching | ✅ | ❌ | ⚠️ Manual |
| Thread-safe | ✅ | ⚠️ Varies | ⚠️ | ✅ |
| Async-native | ✅ | ⚠️ Mixed | ✅ | ✅ |
| Setup complexity | Simple | Complex | Simple | Simple |
| Dependencies | Minimal | Heavy | Moderate | Minimal |
| Performance overhead | <0.015% | ~2-5% | ~1-2% | Minimal |

## Installation Options

| Command | Features | Use Case |
|---------|----------|----------|
| `pip install chuk_llm` | Core + Session tracking | Development |
| `pip install chuk_llm[redis]` | + Redis persistence | Production |
| `pip install chuk_llm[cli]` | + Rich CLI formatting | CLI tools |
| `pip install chuk_llm[all]` | Everything | Full features |

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 🐛 [Issues](https://github.com/chrishayuk/chuk-llm/issues)
- 💬 [Discussions](https://github.com/chrishayuk/chuk-llm/discussions)
- 📧 [Email](mailto:chrishayuk@somejunkmailbox.com)

---

**Built with ❤️ for developers who just want their LLMs to work.**