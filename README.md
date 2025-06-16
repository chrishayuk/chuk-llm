# chuk_llm

A unified, production-ready Python library for Large Language Model (LLM) providers with real-time streaming, function calling, middleware support, and comprehensive provider management.

## 🚀 QuickStart

### Installation

```bash
pip install chuk_llm
```

### API Keys Setup

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
# Add other provider keys as needed
```

### Simple API - Perfect for Scripts & Prototypes

```python
from chuk_llm import ask_sync, quick_question, configure

# Ultra-simple one-liner
answer = quick_question("What is 2+2?")
print(answer)  # "2 + 2 equals 4."

# Provider-specific functions (103 auto-generated!)
from chuk_llm import ask_openai_sync, ask_claude_sync, ask_groq_sync

openai_response = ask_openai_sync("Tell me a joke")
claude_response = ask_claude_sync("Explain quantum computing") 
groq_response = ask_groq_sync("What's the weather like?")

# Configure once, use everywhere
configure(provider="anthropic", temperature=0.7)
response = ask_sync("Write a creative story opening")

# Compare multiple providers
from chuk_llm import compare_providers
results = compare_providers("What is AI?", ["openai", "anthropic"])
for provider, response in results.items():
    print(f"{provider}: {response}")
```

### Async API - Production Performance (3-7x faster!)

```python
import asyncio
from chuk_llm import ask, stream, conversation

async def main():
    # Basic async call
    response = await ask("Hello!")
    
    # Provider-specific async functions
    from chuk_llm import ask_openai, ask_claude, ask_groq
    
    openai_response = await ask_openai("Tell me a joke")
    claude_response = await ask_claude("Explain quantum computing")
    groq_response = await ask_groq("What's the weather like?")
    
    # Real-time streaming (token by token)
    print("Streaming: ", end="", flush=True)
    async for chunk in stream("Write a haiku about coding"):
        print(chunk, end="", flush=True)
    
    # Conversations with memory
    async with conversation(provider="anthropic") as chat:
        await chat.say("My name is Alice")
        response = await chat.say("What's my name?")
        # Remembers: "Your name is Alice"
    
    # Concurrent requests (massive speedup!)
    tasks = [
        ask("Capital of France?"),
        ask("What is 2+2?"), 
        ask("Name a color")
    ]
    responses = await asyncio.gather(*tasks)
    # 3-7x faster than sequential!

asyncio.run(main())
```

### 103 Auto-Generated Functions

ChukLLM automatically creates functions for every provider and model:

```python
# Base provider functions
from chuk_llm import ask_openai, ask_anthropic, ask_groq

# Model-specific functions  
from chuk_llm import ask_openai_gpt4o, ask_claude_sonnet

# Convenient aliases
from chuk_llm import ask_gpt4o, ask_claude

# All with sync, async, and streaming variants!
from chuk_llm import (
    ask_openai_sync,      # Synchronous
    stream_anthropic,     # Async streaming  
    ask_groq_sync         # Sync version
)
```

### Discovery & Utilities

```python
import chuk_llm

# See all available providers and models
chuk_llm.show_providers()

# See all 103 auto-generated functions
chuk_llm.show_functions()

# Comprehensive diagnostics
chuk_llm.print_diagnostics()

# Test connections
from chuk_llm import test_connection_sync
result = test_connection_sync("anthropic")
print(f"✅ {result['provider']}: {result['duration']:.2f}s")
```

### Performance Demo

```python
# Sequential vs Concurrent Performance Test
import time
import asyncio
from chuk_llm import ask

async def performance_demo():
    questions = ["What is AI?", "Capital of Japan?", "2+2=?"]
    
    # Sequential (slow)
    start = time.time()
    for q in questions:
        await ask(q)
    sequential_time = time.time() - start
    
    # Concurrent (fast!)
    start = time.time()
    await asyncio.gather(*[ask(q) for q in questions])
    concurrent_time = time.time() - start
    
    print(f"🐌 Sequential: {sequential_time:.2f}s")
    print(f"🚀 Concurrent: {concurrent_time:.2f}s") 
    print(f"⚡ Speedup: {sequential_time/concurrent_time:.1f}x faster!")
    # Typical result: 3-7x speedup!

asyncio.run(performance_demo())
```

## 🌟 Why ChukLLM?

✅ **103 Auto-Generated Functions** - Every provider & model gets functions  
✅ **3-7x Performance Boost** - Concurrent requests vs sequential  
✅ **Real-time Streaming** - Token-by-token output as it's generated  
✅ **Memory Management** - Stateful conversations with context  
✅ **Production Ready** - Error handling, retries, connection pooling  
✅ **Developer Friendly** - Simple sync for scripts, powerful async for apps  

## 🚀 Features

### Multi-Provider Support
- **OpenAI** - GPT-4, GPT-3.5 with full API support
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Haiku
- **Google Gemini** - Gemini 2.0 Flash, Gemini 1.5 Pro  
- **Groq** - Lightning-fast inference with Llama models
- **Perplexity** - Real-time web search with Sonar models
- **Ollama** - Local model deployment and management

### Core Capabilities
- 🌊 **Real-time Streaming** - True streaming without buffering
- 🛠️ **Function Calling** - Standardized tool/function execution
- 🔧 **Middleware Stack** - Logging, metrics, caching, retry logic
- 📊 **Performance Monitoring** - Built-in benchmarking and metrics
- 🔄 **Error Handling** - Automatic retries with exponential backoff
- 🎯 **Type Safety** - Full Pydantic validation and type hints
- 🧩 **Extensible Architecture** - Easy to add new providers

### Advanced Features
- **Vision Support** - Image analysis across compatible providers
- **JSON Mode** - Structured output generation
- **Real-time Web Search** - Live information retrieval with citations
- **Parallel Function Calls** - Execute multiple tools simultaneously
- **Connection Pooling** - Efficient HTTP connection management
- **Configuration Management** - Environment-based provider setup
- **Capability Detection** - Automatic feature detection per provider

## 📦 Installation

```bash
pip install chuk_llm
```

### Optional Dependencies
```bash
# For all providers
pip install chuk_llm[all]

# For specific providers
pip install chuk_llm[openai]       # OpenAI support
pip install chuk_llm[anthropic]    # Anthropic support  
pip install chuk_llm[google]       # Google Gemini support
pip install chuk_llm[groq]         # Groq support
pip install chuk_llm[perplexity]   # Perplexity support
pip install chuk_llm[ollama]       # Ollama support
```

## 🚀 Quick Start

### Basic Usage

```python
from chuk_llm import ask_sync, quick_question

# Ultra-simple one-liner
answer = quick_question("What is 2+2?")
print(answer)  # "2 + 2 equals 4."

# Basic sync usage
response = ask_sync("Tell me a joke")
print(response)

# Provider-specific functions
from chuk_llm import ask_openai_sync, ask_claude_sync
openai_joke = ask_openai_sync("Tell me a dad joke")
claude_explanation = ask_claude_sync("Explain quantum computing")
```

### Async Usage

```python
import asyncio
from chuk_llm import ask, stream, conversation

async def main():
    # Basic async call
    response = await ask("Hello!")
    
    # Real-time streaming
    print("Streaming: ", end="", flush=True)
    async for chunk in stream("Write a haiku about coding"):
        print(chunk, end="", flush=True)
    
    # Conversations with memory
    async with conversation(provider="anthropic") as chat:
        await chat.say("My name is Alice")
        response = await chat.say("What's my name?")
        # Remembers: "Your name is Alice"

asyncio.run(main())
```

### Real-time Web Search with Perplexity

```python
# Sync version
from chuk_llm import ask_perplexity_sync

response = ask_perplexity_sync("What are the latest AI developments this week?")
print(response)  # Includes real-time web search results with citations

# Async version  
import asyncio
from chuk_llm import ask_perplexity

async def search_example():
    response = await ask_perplexity("What are the latest AI developments this week?")
    print(response)
    
asyncio.run(search_example())
```

### Streaming Responses

```python
import asyncio
from chuk_llm import stream

async def streaming_example():
    print("Assistant: ", end="", flush=True)
    async for chunk in stream("Write a short story about AI"):
        print(chunk, end="", flush=True)
    print()  # New line when done

asyncio.run(streaming_example())
```

### Function Calling

```python
# Sync version
from chuk_llm import ask_sync

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = ask_sync("What's the weather in Paris?", tools=tools)
print(response)  # ChukLLM handles tool calling automatically

# Async version
import asyncio
from chuk_llm import ask

async def function_calling_example():
    response = await ask("What's the weather in Paris?", tools=tools)
    print(response)

asyncio.run(function_calling_example())
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export PERPLEXITY_API_KEY="your-perplexity-key"

# Custom endpoints
export OPENAI_API_BASE="https://api.openai.com/v1"
export PERPLEXITY_API_BASE="https://api.perplexity.ai"
export OLLAMA_API_BASE="http://localhost:11434"
```

### Simple API Configuration

```python
from chuk_llm import configure, get_config

# Simple configuration
configure(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022", 
    temperature=0.7
)

# All subsequent calls use these settings
from chuk_llm import ask_sync
response = ask_sync("Tell me about AI")

# Check current configuration
config = get_config()
print(f"Using {config['provider']} with {config['model']}")
```

### Low-Level API Configuration

```python
import asyncio
from chuk_llm.llm.client import get_client
from chuk_llm.llm.configuration.provider_config import ProviderConfig

async def low_level_example():
    # Method 1: Direct client creation
    client = get_client("openai", model="gpt-4o-mini")
    
    # Method 2: Custom configuration
    config = ProviderConfig({
        "openai": {
            "api_key": "your-key",
            "api_base": "https://custom-endpoint.com",
            "default_model": "gpt-4o"
        },
        "anthropic": {
            "api_key": "your-anthropic-key",
            "default_model": "claude-3-5-sonnet-20241022"
        }
    })
    
    client = get_client("openai", config=config)
    
    # Use the client directly
    response = await client.create_completion([
        {"role": "user", "content": "Hello! How are you?"}
    ])
    
    print(response["response"])

asyncio.run(low_level_example())
```

## 🛠️ Advanced Usage

### Low-Level API

For maximum control, use the low-level client API:

```python
import asyncio
from chuk_llm.llm.client import get_client

async def low_level_examples():
    # Get a client for any provider
    client = get_client("openai", model="gpt-4o-mini")
    
    # Basic completion with full control
    response = await client.create_completion([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ])
    print(response["response"])
    
    # Streaming with low-level control
    messages = [
        {"role": "user", "content": "Write a short story about AI"}
    ]
    
    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)
    
    # Function calling with full control
    tools = [
        {
            "type": "function", 
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = await client.create_completion(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        temperature=0.7,
        max_tokens=150
    )
    
    if response.get("tool_calls"):
        for tool_call in response["tool_calls"]:
            print(f"Function: {tool_call['function']['name']}")
            print(f"Arguments: {tool_call['function']['arguments']}")

asyncio.run(low_level_examples())
```

### Perplexity Web Search (Low-Level)

```python
import asyncio
from chuk_llm.llm.client import get_client

async def perplexity_low_level():
    # Use Perplexity for real-time web information
    client = get_client("perplexity", model="sonar-pro")
    
    messages = [
        {"role": "user", "content": "What are the latest developments in AI today?"}
    ]
    
    response = await client.create_completion(messages)
    print(response["response"])  # Includes real-time web search results with citations

asyncio.run(perplexity_low_level())
```

### Middleware Stack

```python
# ChukLLM automatically includes production-ready middleware
from chuk_llm import ask_sync, get_metrics, health_check_sync

# Use normally - middleware runs automatically
response = ask_sync("Hello!")

# Access built-in metrics
metrics = get_metrics()
print(f"Total requests: {metrics.get('total_requests', 0)}")

# Health monitoring
health = health_check_sync()
print(f"Status: {health.get('status', 'unknown')}")

# For advanced middleware control, use the low-level API:
from chuk_llm.llm.core.enhanced_base import get_enhanced_client

async def advanced_middleware():
    client = get_enhanced_client(
        provider="openai",
        model="gpt-4o-mini",
        enable_logging=True,
        enable_metrics=True,
        enable_caching=True
    )
    
    # Use with full middleware stack
    response = await client.create_completion([
        {"role": "user", "content": "Hello!"}
    ])
    
    # Access detailed metrics
    if hasattr(client, 'middleware_stack'):
        for middleware in client.middleware_stack.middlewares:
            if hasattr(middleware, 'get_metrics'):
                print(middleware.get_metrics())

asyncio.run(advanced_middleware())
```

### Multi-Provider Comparison

```python
from chuk_llm import compare_providers

# Compare responses across providers
results = compare_providers(
    "Explain quantum computing",
    providers=["openai", "anthropic", "perplexity", "groq"]
)

for provider, response in results.items():
    print(f"{provider}: {response[:100]}...")
```

### Real-time Information with Perplexity

```python
import asyncio
from chuk_llm import ask_perplexity

async def current_events_example():
    # Perplexity excels at current information
    response = await ask_perplexity(
        "What are the latest tech industry layoffs this week?",
        model="sonar-reasoning-pro"
    )
    print("Real-time information with citations:")
    print(response)

asyncio.run(current_events_example())
```

### Performance Monitoring

```python
import asyncio
from chuk_llm import test_all_providers

async def monitor_performance():
    # Test all providers concurrently
    results = await test_all_providers()
    
    for provider, result in results.items():
        status = "✅" if result["success"] else "❌"
        duration = result.get("duration", 0)
        print(f"{status} {provider}: {duration:.2f}s")

asyncio.run(monitor_performance())
```

## 📊 Benchmarking

```python
import asyncio
from chuk_llm import test_all_providers, compare_providers

async def benchmark_providers():
    # Quick performance test
    results = await test_all_providers()
    
    print("Provider Performance:")
    for provider, result in results.items():
        if result["success"]:
            print(f"✅ {provider}: {result['duration']:.2f}s")
        else:
            print(f"❌ {provider}: {result['error']}")
    
    # Quality comparison
    comparison = compare_providers(
        "Explain machine learning in simple terms",
        ["openai", "anthropic", "groq"]
    )
    
    print("\nQuality Comparison:")
    for provider, response in comparison.items():
        print(f"{provider}: {response[:100]}...")

asyncio.run(benchmark_providers())
```

## 🔍 Provider Capabilities

```python
import chuk_llm

# Discover available providers and models
chuk_llm.show_providers()

# See all auto-generated functions
chuk_llm.show_functions()

# Get comprehensive diagnostics
chuk_llm.print_diagnostics()

# Test specific provider capabilities
from chuk_llm import test_connection_sync
result = test_connection_sync("anthropic")
print(f"✅ {result['provider']}: {result['duration']:.2f}s")
```

## 🌐 Provider Models

### OpenAI
- **GPT-4** - gpt-4o, gpt-4o-mini, gpt-4-turbo
- **GPT-3.5** - gpt-3.5-turbo

### Anthropic  
- **Claude 3.5** - claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- **Claude 3** - claude-3-opus-20240229, claude-3-sonnet-20240229

### Google Gemini
- **Gemini 2.0** - gemini-2.0-flash-exp
- **Gemini 1.5** - gemini-1.5-pro, gemini-1.5-flash

### Groq
- **Llama 3.3** - llama-3.3-70b-versatile
- **Llama 3.1** - llama-3.1-70b-versatile, llama-3.1-8b-instant
- **Mixtral** - mixtral-8x7b-32768

### Perplexity 🔍
Perplexity offers specialized models optimized for real-time web search and reasoning with citations.

#### Search Models (Online)
- **sonar-pro** - Premier search model built on Llama 3.3 70B, optimized for answer quality and speed (1200 tokens/sec)
- **sonar** - Cost-effective model for quick factual queries and current events
- **llama-3.1-sonar-small-128k-online** - 8B parameter model with 128k context, web search enabled
- **llama-3.1-sonar-large-128k-online** - 70B parameter model with 128k context, web search enabled

#### Reasoning Models  
- **sonar-reasoning-pro** - Expert reasoning with Chain of Thought (CoT) and search capabilities
- **sonar-reasoning** - Fast real-time reasoning model for quick problem-solving

#### Research Models
- **sonar-research** - Deep research model conducting exhaustive searches and comprehensive reports

#### Chat Models (No Search)
- **llama-3.1-sonar-small-128k-chat** - 8B parameter chat model without web search
- **llama-3.1-sonar-large-128k-chat** - 70B parameter chat model without web search

### Ollama
- **Local Models** - Any compatible GGUF model (Llama, Mistral, CodeLlama, etc.)

## 🏗️ Architecture

### Core Components

- **`BaseLLMClient`** - Abstract interface for all providers
- **`MiddlewareStack`** - Request/response processing pipeline
- **`ProviderConfig`** - Configuration management system
- **`ConnectionPool`** - HTTP connection optimization
- **`SystemPromptGenerator`** - Dynamic prompt generation

### Provider Implementations

Each provider implements the `BaseLLMClient` interface with:
- Standardized message format (ChatML)
- Real-time streaming support
- Function calling normalization
- Error handling and retries

### Middleware System

```python
# Custom middleware example
from chuk_llm.llm.middleware import Middleware

class CustomMiddleware(Middleware):
    async def process_request(self, messages, tools=None, **kwargs):
        # Pre-process request
        return messages, tools, kwargs
    
    async def process_response(self, response, duration, is_streaming=False):
        # Post-process response
        return response
```

## 🧪 Testing & Diagnostics

```python
import asyncio
from chuk_llm import test_connection, health_check, print_diagnostics

async def run_diagnostics():
    # Test connection to specific provider
    result = await test_connection("anthropic")
    print(f"Connection test: {result['success']}")
    
    # System health check
    health = await health_check()
    print(f"System status: {health['status']}")
    
    # Comprehensive diagnostics
    print_diagnostics()

asyncio.run(run_diagnostics())
```

## 📈 Performance

### Streaming Performance
- **Zero-buffering streaming** - Chunks delivered in real-time
- **Parallel requests** - Multiple concurrent streams
- **Connection pooling** - Reduced latency

### Benchmarks
```
Provider Comparison (avg response time):
├── Groq: 0.8s (ultra-fast inference)
├── Perplexity: 1.0s (real-time search + generation)
├── OpenAI: 1.2s (balanced performance)
├── Anthropic: 1.5s (high quality)
├── Gemini: 1.8s (multimodal)
└── Ollama: 2.5s (local processing)
```

### Real-time Web Search Performance
Perplexity's Sonar models deliver blazing fast search results at 1200 tokens per second, nearly 10x faster than comparable models like Gemini 2.0 Flash.

## 🔒 Security & Safety

- **API key management** - Environment variable support
- **Request validation** - Input sanitization
- **Error handling** - No sensitive data leakage
- **Rate limiting** - Built-in provider limit awareness
- **Tool name sanitization** - Safe function calling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Adding New Providers

```python
# ChukLLM automatically detects new providers from configuration
# Just add to your providers.yaml:

# providers.yaml
newprovider:
  api_key_env: NEWPROVIDER_API_KEY
  api_base: https://api.newprovider.com
  default_model: new-model-name

# ChukLLM will automatically generate:
# - ask_newprovider()
# - ask_newprovider_sync() 
# - stream_newprovider()
# - ask_newprovider_new_model_name()
```

## 📚 Documentation

- [API Reference](docs/api.md)
- [Provider Guide](docs/providers.md)
- [Middleware Development](docs/middleware.md)
- [Configuration Guide](docs/configuration.md)
- [Benchmarking Guide](docs/benchmarking.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the ChatML format and function calling standards
- Anthropic for advanced reasoning capabilities
- Google for multimodal AI innovations
- Groq for ultra-fast inference
- Perplexity for real-time web search and information retrieval
- Ollama for local AI deployment

---

**chuk_llm** - Unified LLM interface for production applications