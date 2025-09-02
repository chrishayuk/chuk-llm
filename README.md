# chuk-llm

**One library, all LLMs.** Production-ready Python library with automatic model discovery, real-time streaming, and zero-config session tracking.

```python
from chuk_llm import quick_question
print(quick_question("What is 2+2?"))  # "2 + 2 equals 4."
```

## Why chuk-llm?

- **🚀 Instant Setup**: Works out of the box with any LLM provider
- **🔍 Auto-Discovery**: Detects new models automatically (especially Ollama)
- **🛠️ Clean Tools API**: Function calling without the complexity - tools are just parameters
- **⚡ 5-7x Faster**: Groq achieves 526 tokens/sec vs OpenAI's 68 tokens/sec
- **📊 Built-in Analytics**: Automatic cost and usage tracking
- **🎯 Developer-First**: Clean API, great CLI, sensible defaults

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

# Discover new models
chuk-llm discover ollama
```

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

| Provider | Models | Special Features |
|----------|--------|-----------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-3.5 | Industry standard |
| **Azure OpenAI** | GPT-4o, GPT-3.5 (Enterprise) | SOC2, HIPAA compliant, VNet |
| **Anthropic** | Claude 3.5 Sonnet, Haiku | Advanced reasoning |
| **Google** | Gemini 2.0 Flash, 1.5 Pro | Multimodal support |
| **Groq** | Llama 3.3, Mixtral | Ultra-fast (526 tokens/sec) |
| **Ollama** | Any local model | Auto-discovery, offline |
| **IBM watsonx** | Granite 3.3, Llama 4 | Enterprise features |
| **Perplexity** | Sonar models | Real-time web search |
| **Mistral** | Large, Medium, Small | European sovereignty |

## Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# Session Storage (optional)
export SESSION_PROVIDER=redis  # Default: memory
export SESSION_REDIS_URL=redis://localhost:6379/0

# Discovery Settings
export CHUK_LLM_AUTO_DISCOVER=true  # Auto-discover new models
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

## Advanced Features

### 🛠️ Function Calling / Tool Use

ChukLLM provides a clean, unified API for function calling. Tools are just another parameter - no special functions needed!

> 🚀 **New in v0.9+**: Simplified API! Use `ask(prompt, tools=tools_list)` instead of `ask_with_tools()`. The response format automatically adapts: dict when tools are provided, string otherwise.

```python
from chuk_llm import ask, ask_sync
from chuk_llm.api.tools import tool, Tools, tools_from_functions

# Method 1: Direct API usage
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather information for a location"""
    return {"temp": 22, "location": location, "unit": unit, "condition": "sunny"}

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression"""
    return eval(expression)

# Create toolkit
toolkit = tools_from_functions(get_weather, calculate)

# With tools parameter - returns dict with tool_calls
response = await ask(
    "What's the weather in Paris and what's 15 * 4?",
    tools=toolkit.to_openai_format()
)
print(response)  # {"response": "...", "tool_calls": [...]}

# Without tools - returns just string
response = await ask("Hello there!")
print(response)  # "Hello! How can I help you today?"

# Method 2: Class-based tools (auto-execution)
class MyTools(Tools):
    @tool(description="Get weather for a city")
    def get_weather(self, location: str) -> dict:
        return {"temp": 22, "location": location}
    
    @tool  # Description auto-extracted from docstring
    def calculate(self, expr: str) -> float:
        "Evaluate a math expression"
        return eval(expr)

# Auto-executes tools and returns final response
tools = MyTools()
response = await tools.ask("What's the weather in Paris and what's 2+2?")
print(response)  # "The weather in Paris is 22°C and sunny. 2+2 equals 4."

# Method 3: Sync versions work identically
response = ask_sync("Calculate 15 * 4", tools=toolkit.to_openai_format())
print(response)  # {"response": "60", "tool_calls": [...]}
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

# Discover and test
chuk-llm discover ollama        # Find new models
chuk-llm test azure_openai      # Test connection
chuk-llm providers              # List all providers
chuk-llm models ollama          # Show available models
chuk-llm functions              # List all generated functions

# Advanced usage
chuk-llm ask "Question" --provider azure_openai --model gpt-4o-mini --json
chuk-llm ask "Question" --stream --verbose

# Function calling / Tool use from CLI
chuk-llm ask "Calculate 15 * 4" --tools calculator_tools.py
chuk-llm stream "What's the weather?" --tools weather_tools.py --return-tool-calls

# Zero-install with uvx
uvx chuk-llm ask_claude "Hello world"
```

## Performance

ChukLLM is designed for production use with:

- **Connection pooling** for efficient HTTP management
- **Automatic retries** with exponential backoff
- **Concurrent execution** for parallel processing
- **Smart caching** for discovered models
- **Zero-overhead** session tracking (can be disabled)

### Real Benchmark Results

```bash
# Run benchmarks yourself
uv run benchmarks/llm_benchmark.py

# Results show:
# - Groq: 526 tokens/sec, 0.15s first token
# - OpenAI: 68 tokens/sec, 0.58s first token
```

## Documentation

- 📚 [Full Documentation](https://github.com/chrishayuk/chuk-llm/wiki)
- 🎯 [Examples](https://github.com/chrishayuk/chuk-llm/tree/main/examples)
- 🔄 [Migration Guide](https://github.com/chrishayuk/chuk-llm/wiki/migration)
- 📊 [Benchmarks](https://github.com/chrishayuk/chuk-llm/wiki/benchmarks)
- 🤝 [Contributing](https://github.com/chrishayuk/chuk-llm/blob/main/CONTRIBUTING.md)

## Quick Comparison

| Feature | chuk-llm | LangChain | LiteLLM | OpenAI SDK |
|---------|----------|-----------|---------|------------|
| Auto-discovery | ✅ | ❌ | ❌ | ❌ |
| Native streaming | ✅ | ⚠️ | ✅ | ✅ |
| Function calling | ✅ Clean API | ✅ Complex | ⚠️ Basic | ✅ |
| Session tracking | ✅ Built-in | ⚠️ Manual | ❌ | ❌ |
| CLI included | ✅ | ❌ | ⚠️ Basic | ❌ |
| Provider functions | ✅ Auto-generated | ❌ | ❌ | ❌ |
| Conversations | ✅ | ✅ | ❌ | ⚠️ Manual |
| Setup complexity | Simple | Complex | Simple | Simple |
| Dependencies | Minimal | Heavy | Moderate | Minimal |

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