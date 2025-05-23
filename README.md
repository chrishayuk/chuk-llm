# chuk_llm

A unified, production-ready Python library for Large Language Model (LLM) providers with real-time streaming, function calling, middleware support, and comprehensive provider management.

## 🚀 Features

### Multi-Provider Support
- **OpenAI** - GPT-4, GPT-3.5 with full API support
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Haiku
- **Google Gemini** - Gemini 2.0 Flash, Gemini 1.5 Pro  
- **Groq** - Lightning-fast inference with Llama models
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
pip install chuk_llm[openai]     # OpenAI support
pip install chuk_llm[anthropic]  # Anthropic support  
pip install chuk_llm[google]     # Google Gemini support
pip install chuk_llm[groq]       # Groq support
pip install chuk_llm[ollama]     # Ollama support
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from chuk_llm.llm.llm_client import get_llm_client

async def main():
    # Get a client for any provider
    client = get_llm_client("openai", model="gpt-4o-mini")
    
    # Simple completion
    response = await client.create_completion([
        {"role": "user", "content": "Hello! How are you?"}
    ])
    
    print(response["response"])

asyncio.run(main())
```

### Streaming Responses

```python
async def streaming_example():
    client = get_llm_client("openai", model="gpt-4o-mini")
    
    messages = [
        {"role": "user", "content": "Write a short story about AI"}
    ]
    
    async for chunk in client.create_completion(messages, stream=True):
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)

asyncio.run(streaming_example())
```

### Function Calling

```python
async def function_calling_example():
    client = get_llm_client("openai", model="gpt-4o-mini")
    
    # Define tools
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
    
    response = await client.create_completion(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools
    )
    
    if response.get("tool_calls"):
        for tool_call in response["tool_calls"]:
            print(f"Function: {tool_call['function']['name']}")
            print(f"Arguments: {tool_call['function']['arguments']}")

asyncio.run(function_calling_example())
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"

# Custom endpoints
export OPENAI_API_BASE="https://api.openai.com/v1"
export OLLAMA_API_BASE="http://localhost:11434"
```

### Provider Configuration

```python
from chuk_llm.llm.configuration.provider_config import ProviderConfig

# Custom configuration
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

client = get_llm_client("openai", config=config)
```

## 🛠️ Advanced Usage

### Middleware Stack

```python
from chuk_llm.llm.middleware import LoggingMiddleware, MetricsMiddleware
from chuk_llm.llm.core.enhanced_base import get_enhanced_llm_client

# Create client with middleware
client = get_enhanced_llm_client(
    provider="openai",
    model="gpt-4o-mini",
    enable_logging=True,
    enable_metrics=True,
    enable_caching=True
)

# Use normally - middleware runs automatically
response = await client.create_completion(messages)

# Access metrics
if hasattr(client, 'middleware_stack'):
    for middleware in client.middleware_stack.middlewares:
        if hasattr(middleware, 'get_metrics'):
            print(middleware.get_metrics())
```

### Multi-Provider Chat

```python
from chuk_llm.llm.features import multi_provider_chat

# Compare responses across providers
responses = await multi_provider_chat(
    message="Explain quantum computing",
    providers=["openai", "anthropic", "groq"],
    model_map={
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "groq": "llama-3.3-70b-versatile"
    }
)

for provider, response in responses.items():
    print(f"{provider}: {response[:100]}...")
```

### Unified Interface

```python
from chuk_llm.llm.features import UnifiedLLMInterface

# High-level interface
interface = UnifiedLLMInterface("openai", "gpt-4o-mini")

# Simple chat
response = await interface.simple_chat("Hello!")

# Chat with options
response = await interface.chat(
    messages=[{"role": "user", "content": "Explain AI"}],
    temperature=0.7,
    max_tokens=500,
    json_mode=True
)
```

### System Prompt Generation

```python
from chuk_llm.llm.system_prompt_generator import (
    SystemPromptGenerator, 
    PromptStyle, 
    PromptContext
)

# Create generator
generator = SystemPromptGenerator(PromptStyle.FUNCTION_FOCUSED)

# Define tools
tools = {
    "functions": [
        {
            "name": "calculate",
            "description": "Perform calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
    ]
}

# Generate optimized prompt
prompt = generator.generate_for_provider(
    provider="openai",
    model="gpt-4o",
    tools=tools,
    user_instructions="You are a math tutor."
)

# Use in completion
messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": "What is 15 * 23?"}
]
```

## 📊 Benchmarking

```python
from benchmarks.llm_benchmark import LLMBenchmark

# Create benchmark
benchmark = LLMBenchmark()

# Test multiple providers
results = await benchmark.benchmark_multiple([
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
    ("groq", "llama-3.3-70b-versatile")
])

# Generate report
report = benchmark.generate_report(results)
print(report)
```

## 🔍 Provider Capabilities

```python
from chuk_llm.llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature

# Check what a provider supports
openai_caps = PROVIDER_CAPABILITIES["openai"]
print(f"Supports streaming: {openai_caps.supports(Feature.STREAMING)}")
print(f"Supports vision: {openai_caps.supports(Feature.VISION)}")
print(f"Max context: {openai_caps.max_context_length}")

# Find best provider for requirements
from chuk_llm.llm.configuration.capabilities import CapabilityChecker

best = CapabilityChecker.get_best_provider({
    Feature.STREAMING, 
    Feature.TOOLS, 
    Feature.VISION
})
print(f"Best provider: {best}")
```

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
# Extended streaming test
from diagnostics.streaming_extended import test_extended_streaming

await test_extended_streaming()

# Health check
from chuk_llm.llm.connection_pool import get_llm_health_status

health = await get_llm_health_status()
print(health)
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
├── OpenAI: 1.2s (balanced performance)
├── Anthropic: 1.5s (high quality)
├── Gemini: 1.8s (multimodal)
└── Ollama: 2.5s (local processing)
```

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
# Implement BaseLLMClient
class NewProviderClient(BaseLLMClient):
    def create_completion(self, messages, tools=None, *, stream=False, **kwargs):
        # Implementation here
        pass

# Add to provider config
DEFAULTS["newprovider"] = {
    "client": "chuk_llm.llm.providers.newprovider_client:NewProviderClient",
    "api_key_env": "NEWPROVIDER_API_KEY",
    "default_model": "default-model"
}
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
- Ollama for local AI deployment

---

**chuk_llm** - Unified LLM interface for production applications