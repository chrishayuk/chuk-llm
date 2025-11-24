# chuk-llm Examples

Comprehensive examples demonstrating all features of chuk-llm with modern Pydantic-native architecture.

## 🚀 Quick Start - New User Path

**Start here if you're new to chuk-llm!**

### Beginner Examples (5 minutes)
1. **[00_quick_start.py](00_quick_start.py)** - One line to ask a question
2. **[01_basic_ask.py](01_basic_ask.py)** - Basic async and sync patterns
3. **[02_streaming.py](02_streaming.py)** - Real-time streaming responses

### Registry System (10 minutes) - ⭐ THE KEY DIFFERENTIATOR
4. **[03_registry_discovery.py](03_registry_discovery.py)** - Intelligent model selection
5. **[registry_provider_discovery.py](registry_provider_discovery.py)** - Deep dive into discovery

### Intermediate Examples (15 minutes)
6. **[04_tools_basic.py](04_tools_basic.py)** - Function calling basics
7. **[05_tools_advanced.py](05_tools_advanced.py)** - Auto-execution with Tools class
8. **[06_conversations.py](06_conversations.py)** - Stateful conversations

### Advanced Examples (20 minutes)
9. **[07_json_mode.py](07_json_mode.py)** - Structured outputs and data extraction
10. **[08_multimodal.py](08_multimodal.py)** - Vision and image understanding

## 📁 Directory Structure

The examples directory is now clean and well-organized:

```
examples/
├── README.md                                   # This file
│
├── 🆕 CORE EXAMPLES (Root directory - Start Here!)
│   ├── 00_quick_start.py                      # Simplest possible example
│   ├── 01_basic_ask.py                        # Basic usage patterns
│   ├── 02_streaming.py                        # Real-time streaming
│   ├── 03_registry_discovery.py               # 🧠 Registry-based selection
│   ├── 04_tools_basic.py                      # Tool calling basics
│   ├── 05_tools_advanced.py                   # Auto-execution
│   ├── 06_conversations.py                    # Stateful chatbots
│   ├── 07_json_mode.py                        # Structured outputs
│   ├── 08_multimodal.py                       # Vision/images
│   └── registry_provider_discovery.py         # Registry deep dive
│
├── providers/                                  # Provider-specific examples (17 files)
│   ├── run_all_providers.py                   # ⭐ Test all providers
│   ├── openai_usage_examples.py               # OpenAI (GPT-5, o1, GPT-4o)
│   ├── openai_chat_completions_example.py     # Chat completions format
│   ├── openai_responses_example.py            # Responses API (stateful)
│   ├── openai_compatible_example.py           # OpenAI-compatible providers
│   ├── anthropic_usage_examples.py            # Claude 3.5
│   ├── gemini_usage_examples.py               # Gemini 2.0
│   ├── groq_usage_examples.py                 # Ultra-fast inference
│   ├── mistral_usage_examples.py              # Mistral AI
│   ├── azure_usage_examples.py                # Enterprise Azure
│   ├── watsonx_usage_examples.py              # IBM Watsonx
│   ├── advantage_usage_examples.py            # IBM Advantage
│   ├── deepseek_usage_examples.py             # DeepSeek V3
│   ├── perplexity_usage_examples.py           # Web search + citations
│   ├── openrouter_usage_examples.py           # 100+ models via one API
│   ├── conversation_isolation_demo.py         # Architecture deep dive
│   └── session_isolation_demo.py              # Session management
│
└── advanced/                                   # Advanced features (6 files)
    ├── registry_demo.py                       # Registry system demo
    ├── performance_demo.py                    # Performance benchmarks
    ├── dynamic_provider_workflow.py           # Dynamic provider selection
    ├── streaming_usage.py                     # Advanced streaming
    ├── tools_execution_demo.py                # Tool execution patterns
    └── common_demos.py                        # Common usage patterns
```

**Total: 33 focused, well-organized examples** (10 core + 17 provider + 6 advanced)

### 🧹 Recent Cleanup (2025-01-21)

The examples directory was massively cleaned up to remove:
- ❌ 34+ duplicate, debug, and outdated examples
- ❌ Pirate-themed examples (fun but not core)
- ❌ Legacy discovery examples (replaced by registry)
- ❌ Test and scratch files

**Result**: Crystal clear learning path with only essential, well-documented examples!

## Test All Providers

Run the master test suite to verify all providers work:

```bash
python providers/run_all_providers.py
python providers/run_all_providers.py --provider openai --verbose
```

## OpenAI Examples Overview 🆕

We provide **3 comprehensive OpenAI examples** covering different use cases:

### 1. OpenAI Chat Completions API

**File**: `openai_chat_completions_example.py`

OpenAI's traditional Chat Completions API (`/v1/chat/completions`):

```bash
python examples/providers/openai_chat_completions_example.py
python examples/providers/openai_chat_completions_example.py --model gpt-5
python examples/providers/openai_chat_completions_example.py --demo 3  # Function calling
```

**Features** (11 demos):
- ✅ Basic completion, streaming, function calling
- ✅ Vision, JSON mode, structured outputs
- ✅ GPT-5 and GPT-5-mini support
- ✅ Model comparison (4 models)
- ✅ Manual conversation history management
- ⚠️ You manage conversation state yourself

**Use this for**: Standard OpenAI requests, maximum control, custom history management

### 2. OpenAI Responses API 🆕

**File**: `openai_responses_example.py`

OpenAI's next-generation Responses API (`/v1/responses`) with built-in conversation state:

```bash
python examples/providers/openai_responses_example.py
python examples/providers/openai_responses_example.py --model gpt-5
python examples/providers/openai_responses_example.py --demo 3  # Stateful conversation
```

**Features** (13 demos):
- ✅ **Stateful conversations** with `previous_response_id`
- ✅ **Automatic history** management by OpenAI (`store=true`)
- ✅ Response retrieval and deletion by ID
- ✅ Background processing mode
- ✅ JSON mode with full json_schema validation
- ✅ GPT-5 with reasoning configuration
- ✅ Vision, function calling, structured outputs

**Use this for**: Multi-turn conversations, complex workflows, automatic state management

### 3. OpenAI-Compatible Providers

**File**: `openai_compatible_example.py`

Using other providers (Groq, DeepSeek, Together, Mistral, Perplexity) with the same API:

```bash
python examples/providers/openai_compatible_example.py
python examples/providers/openai_compatible_example.py --provider groq
python examples/providers/openai_compatible_example.py --demo 4  # Provider comparison
```

**Features** (6 demos):
- ✅ **5 providers** (Groq, DeepSeek, Together, Mistral, Perplexity)
- ✅ Same API across all providers
- ✅ Basic completion, streaming, function calling
- ✅ Provider comparison and multi-provider apps
- ✅ Easy provider switching (just change base_url + api_key)

**Use this for**: Provider flexibility, cost optimization, avoiding vendor lock-in

---

## API Comparison

| Feature | Chat Completions | Responses API | Compatible Providers |
|---------|------------------|---------------|---------------------|
| **Endpoint** | `/v1/chat/completions` | `/v1/responses` | `/v1/chat/completions` |
| **History** | Manual | ✅ Automatic | Manual |
| **Storage** | None | ✅ Built-in | None |
| **Stateful** | No | ✅ Yes | No |
| **Providers** | OpenAI only | OpenAI only | ✅ Multi-provider |
| **JSON Schema** | JSON mode only | ✅ Full validation | Varies |
| **Background** | No | ✅ Yes | No |
| **Use Case** | Standard requests | Complex conversations | Provider flexibility |

## All 12 Providers

ChukLLM supports 12 LLM providers with modern Pydantic clients (100% coverage):

### Core Providers

| Provider | API Key Env Var | Default Model | Status |
|----------|----------------|---------------|---------|
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o-mini` | ✅ Modern client |
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-20241022` | ✅ Modern client |

### OpenAI-Compatible Providers

| Provider | API Key Env Var | Default Model | Status |
|----------|----------------|---------------|---------|
| **Groq** | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | ✅ Modern client |
| **DeepSeek** | `DEEPSEEK_API_KEY` | `deepseek-chat` | ✅ Modern client |
| **Together** | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | ✅ Modern client |
| **Perplexity** | `PERPLEXITY_API_KEY` | `llama-3.1-sonar-small-128k-online` | ✅ Modern client |
| **Mistral** | `MISTRAL_API_KEY` | `mistral-small-latest` | ✅ Modern client |
| **Ollama** | `OLLAMA_BASE_URL` (optional) | `llama3.2:latest` | ✅ Modern client |

### Enterprise Providers

| Provider | API Key Env Var | Default Model | Status |
|----------|----------------|---------------|---------|
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `gpt-4o` | ✅ Modern client |
| **IBM Advantage** | `ADVANTAGE_API_KEY` + `ADVANTAGE_API_BASE` | `meta-llama/llama-3-3-70b-instruct` | ✅ Modern client |

### Specialized Providers

| Provider | API Key Env Var | Default Model | Status |
|----------|----------------|---------------|---------|
| **Gemini** | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-2.0-flash-exp` | ✅ Modern client |
| **Watsonx** | `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` | `ibm/granite-3-8b-instruct` | ✅ Modern client |

## Features Tested

Each provider example demonstrates:

- ✅ **Basic Completion** - Standard text generation
- ✅ **Streaming** - Real-time response streaming
- ✅ **Tool Calling** - Function/tool invocation (where supported)
- ✅ **System Prompts** - Custom system instructions
- ✅ **Vision** - Multi-modal image understanding (where supported)
- ✅ **JSON Mode** - Structured output (where supported)
- ✅ **Error Handling** - Graceful error management

## Setup

### 1. Install ChukLLM

```bash
pip install -e .
# or with all features
pip install -e .[all]
```

### 2. Set API Keys

Create a `.env` file in the project root:

```bash
# Core providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI-compatible
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
TOGETHER_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
MISTRAL_API_KEY=...

# Ollama (optional, defaults to localhost)
OLLAMA_BASE_URL=http://localhost:11434

# Enterprise
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# IBM providers
ADVANTAGE_API_KEY=...
ADVANTAGE_API_BASE=...
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...

# Google
GEMINI_API_KEY=...
```

### 3. Run Examples

```bash
# Test all available providers
python providers/run_all_providers.py

# Test specific provider
python providers/run_all_providers.py --provider openai --verbose

# Run comprehensive provider examples
python providers/openai_usage_examples.py
python providers/anthropic_usage_examples.py
python providers/gemini_usage_examples.py
```

## Usage Patterns

### Basic Completion

```python
from chuk_llm.api import ask

response = await ask(
    "Explain quantum computing in simple terms",
    provider="openai",
    model="gpt-4o-mini"
)
print(response)
```

### Streaming

```python
from chuk_llm.api import stream

async for chunk in stream(
    "Write a haiku about AI",
    provider="anthropic",
    model="claude-3-5-haiku-20241022"
):
    print(chunk, end="", flush=True)
```

### Tool Calling

```python
from chuk_llm.api import ask

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        }
    }
}]

response = await ask(
    "What's the weather in Tokyo?",
    provider="openai",
    tools=tools
)
```

### Vision (Multi-modal)

```python
from chuk_llm.api import ask

response = await ask(
    [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://..."}}
    ],
    provider="gemini",
    model="gemini-2.0-flash-exp"
)
```

## Provider-Specific Notes

### OpenAI
- **Latest models**: GPT-5, GPT-5-mini (newest, most capable)
- Full support for GPT-4, GPT-4o, O-series models
- Vision supported with `gpt-5`, `gpt-4o`, `gpt-4-turbo`
- JSON mode available
- Function calling fully supported

### Anthropic (Claude)
- Claude 3.5 Sonnet/Haiku with enhanced reasoning
- Vision supported with Claude 3+ models
- Different tool format than OpenAI (auto-converted)

### Groq
- Ultra-fast inference (<1s responses)
- Llama 3.3, Mixtral, Gemma models
- Tool calling supported

### Gemini
- Google's latest models with multimodal support
- Vision natively integrated
- Uses REST API (no heavy SDK)

### Azure OpenAI
- Enterprise OpenAI deployment
- Requires endpoint + deployment name
- Same models as OpenAI

### Watsonx
- IBM's enterprise LLM platform
- Granite models optimized for business
- Wraps IBM SDK with modern patterns

### Ollama
- Local model hosting
- No API key needed for localhost
- Supports Llama, Mistral, CodeLlama, etc.

## Troubleshooting

### API Key Not Set

```
❌ OPENAI_API_KEY not set
   export OPENAI_API_KEY='sk-...'
```

**Solution**: Set the required environment variable or add to `.env`

### Model Not Found

```
❌ Model 'gpt-5' not found
```

**Solution**: Check available models for the provider in their documentation

### Rate Limit Errors

```
❌ Rate limit exceeded
```

**Solution**: Reduce request frequency or upgrade API tier

### Connection Errors

```
❌ Connection failed to http://localhost:11434
```

**Solution**: For Ollama, ensure server is running: `ollama serve`

## Testing

Run the comprehensive test suite:

```bash
# Test all providers
python providers/run_all_providers.py

# Test specific provider
python providers/run_all_providers.py --provider gemini --verbose

# Quick test (basic features only)
python providers/run_all_providers.py --quick
```

## Migration Status

✅ **100% Complete** - All 12 providers migrated to modern Pydantic clients

- Type-safe with Pydantic V2
- Fast JSON with orjson
- Connection pooling with httpx
- Clean architecture (no fallbacks)
- Zero breaking changes

## Contributing

When adding new examples:

1. Place provider-specific examples in `providers/`
2. Name files descriptively: `{provider}_usage_examples.py`
3. Include comprehensive tests for all features
4. Add provider to `run_all_providers.py`
5. Update this README

## See Also

- [Migration Progress](../MIGRATION_PROGRESS.md) - 100% migration details
- [Session Summary](../SESSION_SUMMARY.md) - What was accomplished
- [Documentation](../docs/) - Full API documentation

---

**Last Updated**: 2025-11-19
**Migration Status**: 100% Complete (12/12 providers)
**Modern Clients**: OpenAIClient, AnthropicClient, AzureOpenAIClient, OpenAICompatibleClient, GeminiClient, WatsonxClient
