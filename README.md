# chuk_llm

A unified, production-ready Python library for Large Language Model (LLM) providers with real-time streaming, function calling, middleware support, automatic session tracking, dynamic model discovery, intelligent system prompt generation, and a powerful CLI.

## 🌟 Why ChukLLM?

✅ **🔌 OpenAI-Compatible Everything** - Support for 100+ providers via OpenAI API compatibility  
✅ **🎯 Dynamic Provider Registration** - Add new providers at runtime without config files  
✅ **🛠️ Advanced Tool Streaming** - Real-time tool calls with incremental JSON parsing  
✅ **200+ Auto-Generated Functions** - Every provider & model + discovered models  
✅ **🚀 GPT-5 & Reasoning Models** - Full support for GPT-5, O1, O3+ series, Claude 4, and GPT-OSS  
✅ **🤖 Smart Sync/Async Detection** - Functions auto-detect context, no more coroutine confusion  
✅ **3-7x Performance Boost** - Concurrent requests vs sequential  
✅ **Real-time Streaming** - Token-by-token output as it's generated  
✅ **Memory Management** - Stateful conversations with context  
✅ **Automatic Session Tracking** - Zero-config usage analytics & cost monitoring  
✅ **✨ Dynamic Model Discovery** - Automatically detect and generate functions for new models  
✅ **🧠 Intelligent System Prompts** - Provider-optimized prompts with tool integration  
✅ **🖥️ Enhanced CLI** - Terminal access with streaming, discovery, and convenience functions  
✅ **🏢 Enterprise Ready** - Error handling, retries, connection pooling, compliance features  
✅ **👨‍💻 Developer Friendly** - Simple sync for scripts, powerful async for apps  

## 🚀 QuickStart

### Installation

```bash
# Install uv (if you haven't already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Core functionality with session tracking (memory storage)
uv add chuk_llm

# With Redis for persistent sessions
uv add chuk_llm[redis]

# With enhanced CLI experience
uv add chuk_llm[cli]

# Full installation
uv add chuk_llm[all]

# Alternative: use pip if you prefer
pip install chuk_llm[all]
```

### 30-Second Demo

```bash
# Zero installation required - try it instantly with uv!
uvx chuk-llm stream_ollama_gpt_oss "What is Python?"

# Or use convenience functions directly
uvx chuk-llm ask_ollama_granite3_3 "What is machine learning?"
```

Or in Python:
```python
from chuk_llm import quick_question, ask_sync

# Ultra-simple one-liner
answer = quick_question("What is 2+2?")
print(answer)  # "2 + 2 equals 4."

# Simple synchronous usage
response = ask_sync("Tell me a joke")
print(response)
```

### 🎯 Getting Started - Pick Your Style

```python
# 1️⃣ Simplest - Let ChukLLM choose the provider
from chuk_llm import quick_question
answer = quick_question("What is the capital of France?")

# 2️⃣ Direct Provider Functions (Auto-generated)
from chuk_llm import ask_ollama_granite3_3, ask_openai_gpt4o_mini
response = ask_ollama_granite3_3("Explain Python in one sentence")
response = ask_openai_gpt4o_mini("What's the weather like?")

# 3️⃣ Flexible API with provider selection
from chuk_llm import ask_sync
response = ask_sync("Tell me a story", provider="openai", model="gpt-4o")
response = ask_sync("Explain quantum physics", provider="anthropic", model="claude-3-sonnet")

# 4️⃣ Async for high-performance apps
import asyncio
from chuk_llm import ask

async def main():
    response = await ask("What is AI?", provider="gemini")
    print(response)

asyncio.run(main())

# 5️⃣ Streaming for real-time output (standalone)
from chuk_llm import stream_sync_iter

# Note: This is for one-off streaming. For conversations, use async conversation API
for chunk in stream_sync_iter("Write a poem about code", provider="ollama"):
    print(chunk, end="", flush=True)
```

## 🤖 NEW: Smart Sync/Async Auto-Detection

ChukLLM's `ask_*` functions now automatically detect whether they're being called from sync or async context:

```python
from chuk_llm import ask_ollama_granite

# ✅ Sync context - works without _sync suffix!
result = ask_ollama_granite("What is Python?")
print(result)  # Direct result, not a coroutine!

# ✅ Async context - same function with await
async def my_async_function():
    result = await ask_ollama_granite("What is Python?")
    return result

# No more confusion! The same function works in both contexts
```

### Before vs After

```python
# ❌ OLD WAY - Confusing coroutine errors
from chuk_llm import ask_ollama_granite
print(ask_ollama_granite("Hello"))  # <coroutine object...> 😕

# ❌ OLD WAY - Need to remember _sync suffix
from chuk_llm import ask_ollama_granite_sync
print(ask_ollama_granite_sync("Hello"))  # Works but verbose

# ✅ NEW WAY - Just works!
from chuk_llm import ask_ollama_granite
print(ask_ollama_granite("Hello"))  # "Hello! How can I help?" 🎉
```

**Note:** Real-time streaming in conversations requires the async API. For standalone streaming, use `stream_sync_iter` (shown above) or async `stream()` functions.

## 🔌 NEW: OpenAI-Compatible Providers

ChukLLM now supports **ANY OpenAI-compatible API** with zero configuration changes. This includes gateways, proxies, and self-hosted services.

### Built-in OpenAI-Compatible Providers

```python
from chuk_llm import ask_sync

# LiteLLM Gateway - Universal LLM Gateway (100+ providers)
response = ask_sync("Hello!", provider="litellm", model="gpt-3.5-turbo")
response = ask_sync("Hello!", provider="litellm", model="claude-3-opus")
response = ask_sync("Hello!", provider="litellm", model="gemini-pro")

# OpenRouter - Unified API for LLMs
response = ask_sync("Hello!", provider="openrouter", model="openai/gpt-4")
response = ask_sync("Hello!", provider="openrouter", model="anthropic/claude-3-opus")
response = ask_sync("Hello!", provider="openrouter", model="meta-llama/llama-3-70b-instruct")

# vLLM - High-performance inference
response = ask_sync("Hello!", provider="vllm", model="meta-llama/Llama-3-70b-hf")

# Together AI - Scalable inference
response = ask_sync("Hello!", provider="togetherai", model="deepseek-ai/deepseek-v3")
response = ask_sync("Hello!", provider="togetherai", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

# Generic OpenAI-compatible endpoint
response = ask_sync("Hello!", provider="openai_compatible", 
                   base_url="https://your-service.com/v1",
                   api_key="your-key")
```

### Environment Configuration

```bash
# LiteLLM Gateway
export LITELLM_API_BASE=http://localhost:4000
export LITELLM_API_KEY=your-litellm-key

# OpenRouter
export OPENROUTER_API_KEY=sk-or-...

# vLLM Server
export VLLM_API_BASE=http://localhost:8000/v1

# Together AI
export TOGETHER_API_KEY=...
export TOGETHERAI_API_BASE=https://api.together.xyz/v1

# Generic OpenAI-compatible
export OPENAI_COMPATIBLE_API_BASE=https://your-service.com/v1
export OPENAI_COMPATIBLE_API_KEY=your-key
```

## 🎯 NEW: Dynamic Provider Registration

Register new providers at runtime without modifying configuration files:

### Simple Registration

```python
from chuk_llm import register_provider, register_openai_compatible

# Register a simple OpenAI-compatible provider
register_openai_compatible(
    name="my_service",
    api_base="https://api.myservice.com/v1",
    api_key="sk-abc123",
    models=["model-a", "model-b"],
    default_model="model-a"
)

# Now use it immediately!
from chuk_llm import ask_sync
response = ask_sync("Hello!", provider="my_service")
```

### Advanced Registration

```python
# Register with environment variables
register_provider(
    name="custom_llm",
    api_key_env="CUSTOM_API_KEY",        # Uses environment variable
    api_base_env="CUSTOM_API_BASE",      # Uses environment variable
    models=["gpt-3.5-turbo", "gpt-4"],
    client_class="chuk_llm.llm.providers.openai_client.OpenAILLMClient",
    features=["streaming", "tools", "vision", "json_mode"]
)

# Inherit from existing provider
register_provider(
    name="my_openai",
    inherits_from="openai",              # Inherit OpenAI's configuration
    api_base="https://proxy.company.com/v1",
    api_key="company-key"
)

# Register LocalAI
register_openai_compatible(
    name="localai",
    api_base="http://localhost:8080/v1",
    models=["llama", "mistral", "phi"]
)

# Register FastChat
register_openai_compatible(
    name="fastchat",
    api_base="http://localhost:8000/v1",
    models=["vicuna-13b", "chatglm2-6b"]
)

# Register LM Studio
register_openai_compatible(
    name="lmstudio",
    api_base="http://localhost:1234/v1",
    models=["*"]  # Accept any model
)

# Register Anyscale Endpoints
register_openai_compatible(
    name="anyscale",
    api_base="https://api.endpoints.anyscale.com/v1",
    api_key_env="ANYSCALE_API_KEY",
    models=["meta-llama/Llama-2-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
)

# Register Ollama with OpenAI compatibility
register_openai_compatible(
    name="ollama_openai",
    api_base="http://localhost:11434/v1",
    models=["llama3.3", "mistral", "phi3"]
)
```

### Managing Dynamic Providers

```python
from chuk_llm import (
    update_provider,
    unregister_provider,
    list_dynamic_providers,
    get_provider_config,
    provider_exists
)

# Update a provider's configuration
update_provider(
    "my_service",
    api_base="https://new-endpoint.com/v1",
    models=["model-a", "model-b", "model-c"]
)

# Check if a provider exists
if provider_exists("my_service"):
    response = ask_sync("Hello", provider="my_service")

# List all dynamically registered providers
dynamic = list_dynamic_providers()
print(f"Dynamic providers: {dynamic}")

# Get provider configuration
config = get_provider_config("my_service")
print(f"API base: {config.api_base}")
print(f"Models: {config.models}")

# Remove a dynamic provider
success = unregister_provider("my_service")
```

## 🖥️ Enhanced CLI with Dynamic Configuration

The CLI now supports dynamic provider configuration on-the-fly:

```bash
# Use any provider with custom endpoints
chuk-llm ask "Hello" --provider openai --base-url https://api.custom.com/v1 --api-key sk-custom-key
chuk-llm ask "Test" --provider ollama --base-url http://remote-server:11434
chuk-llm ask "Query" --provider anthropic --api-key sk-test-key

# Use OpenAI-compatible providers
chuk-llm ask "Hello" --provider litellm --model claude-3-opus
chuk-llm ask "Hello" --provider openrouter --model openai/gpt-4
chuk-llm ask "Hello" --provider vllm --model meta-llama/Llama-3-70b-hf
chuk-llm ask "Hello" --provider togetherai --model deepseek-ai/deepseek-v3

# Quick questions using global aliases
chuk-llm ask_granite "What is Python?"
chuk-llm ask_claude "Explain quantum computing"
chuk-llm ask_gpt "Write a haiku about code"

# Convenience functions for discovered models
chuk-llm ask_ollama_gpt_oss "Think through this step by step"
chuk-llm ask_ollama_mistral_small_latest "Tell me a joke"
chuk-llm stream_ollama_llama3_2 "Write a long explanation"

# Dot notation is automatically converted to underscores
chuk-llm ask_ollama_granite3.3 "What is AI?"  # Works with dots!
chuk-llm ask_ollama_llama3.2 "Explain quantum computing"

# JSON responses for structured output
chuk-llm ask "List 3 Python libraries as json " --json --provider openai --model gpt-5

# Set system prompts to control AI personality
chuk-llm ask "What is coding?" -p ollama -m granite3.3:latest -s "You are a pirate. Use pirate speak."
chuk-llm ask "Explain databases" --provider openai --system-prompt "You are a 5-year-old. Use simple words."
chuk-llm ask "Review my code" -p anthropic -s "Be extremely critical and thorough."

# Provider and model management
chuk-llm providers              # Show all available providers
chuk-llm models openai          # Show models for OpenAI
chuk-llm test openai            # Test OpenAI connection
chuk-llm discover ollama        # Discover new Ollama models

# Configuration and diagnostics
chuk-llm config                 # Show current configuration
chuk-llm functions              # List all auto-generated functions
chuk-llm functions ollama       # Filter functions by provider
chuk-llm help                   # Comprehensive help

# Use with uv for zero-install usage
uvx chuk-llm ask "What is AI?" --provider openai
uvx chuk-llm ask_ollama_gpt_oss "Reasoning problem"
uvx chuk-llm ask "Test GPT-5" --provider openai --model gpt-5
```

## 📊 Supported Providers

### Major Cloud Providers
| Provider | Models | Special Features |
|----------|---------|------------------|
| **OpenAI** | GPT-5, GPT-4o, GPT-3.5-turbo, O1/O3 series | Reasoning models, function calling, JSON mode |
| **Azure OpenAI** | Enterprise GPT-5, GPT-4, Custom deployments | Private endpoints, compliance, auto-discovery |
| **Anthropic** | Claude 4.1 Opus, Claude 4 Sonnet, Claude 3.7 | Advanced reasoning, 200K context, vision |
| **Google Gemini** | Gemini 2.5 Flash/Pro, 2.0 Flash, 1.5 Pro | Multimodal, 2M context, thinking capabilities |
| **Groq** | Llama 3.3, Mixtral, GPT-OSS | Ultra-fast inference (245+ tokens/sec), 131K context |
| **Perplexity** | Sonar models | Real-time web search, citations |
| **Mistral AI** | Magistral (reasoning), Codestral, Pixtral | European, <think> tags, vision models |
| **DeepSeek** | DeepSeek-Reasoner, DeepSeek-Chat | Complex reasoning (30-60s), OpenAI-compatible |
| **IBM watsonx** | Granite 3.3, Llama 4, Custom models | Enterprise compliance, IBM Cloud integration |

### OpenAI-Compatible Gateways
| Provider | Description | Use Case |
|----------|-------------|----------|
| **LiteLLM** | Universal gateway for 100+ providers | Multi-provider apps |
| **OpenRouter** | Unified API for all major LLMs | Provider switching |
| **vLLM** | High-performance OpenAI-compatible | Self-hosted inference |
| **Together AI** | Scalable inference platform | Production workloads |
| **openai_compatible** | Generic OpenAI API | Any compatible service |

### Local/Self-Hosted
| Provider | Description | Use Case |
|----------|-------------|----------|
| **Ollama** | Local models with discovery | Privacy, offline |
| **LocalAI** | OpenAI-compatible local API | Self-hosted |
| **FastChat** | Multi-model serving | Research |
| **LM Studio** | Desktop model server | Personal use |
| **Text Generation WebUI** | Gradio-based UI | Experimentation |

## 🎭 System Prompts - Control AI Personality

System prompts let you define the AI's personality, behavior, and response style:

```python
from chuk_llm import ask_sync, ask_ollama_granite3_3

# Make the AI respond like a pirate
pirate_prompt = """You are a pirate captain. Always speak like a pirate with 'arr', 
'matey', 'ahoy', and other pirate expressions. Be dramatic and mention the sea."""

response = ask_ollama_granite3_3(
    "What is Python programming?",
    system_prompt=pirate_prompt
)
print(response)  # "Arr matey! Python be a mighty fine programming language..."

# Professional technical writer
tech_prompt = "You are a technical documentation expert. Be precise, clear, and use proper terminology."
response = ask_sync(
    "Explain REST APIs",
    provider="openai",
    system_prompt=tech_prompt
)

# Streaming with system prompt
async def stream_with_personality():
    async for chunk in stream(
        "Tell me about databases",
        provider="ollama",
        model="granite",
        system_prompt="You are a database expert. Use SQL examples in your explanations."
    ):
        print(chunk, end="", flush=True)
```

CLI with system prompts:
```bash
# Short form
chuk-llm ask "What is coding?" -p ollama -m granite3.3 -s "You are a pirate"

# Long form  
chuk-llm ask "Explain databases" --provider openai --system-prompt "You are a 5-year-old"

# With convenience functions
chuk-llm ask_ollama_granite "What is AI?" --system-prompt "Be extremely technical"
```

## 🛠️ Advanced Features

### Real-time Tool Streaming

ChukLLM implements cutting-edge real-time tool call streaming:

```python
import asyncio
from chuk_llm import stream

async def stream_with_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }
        }
    ]
    
    print("🛠️  Streaming with tool calls:")
    async for chunk in stream(
        "Calculate 15% of 2,847",
        provider="openai",
        model="gpt-5",
        tools=tools
    ):
        if chunk.get("tool_calls"):
            for tool_call in chunk["tool_calls"]:
                func_name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                print(f"🔧 TOOL CALL: {func_name}({args})")
        
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)

asyncio.run(stream_with_tools())
```

### Conversations with Memory & Branching

#### Async Conversation (Recommended for performance)

```python
from chuk_llm import conversation
import asyncio

async def chat_example():
    # Start a conversation with automatic session tracking
    async with conversation(provider="openai", model="gpt-4o-mini") as chat:
        # Basic ask
        response = await chat.ask("My name is Alice and I'm learning Python")
        print(response)
        
        # Memory persists
        response = await chat.ask("What's my name?")
        print(response)  # "Your name is Alice"
        
        # Stream responses for real-time output
        async for chunk in chat.stream("Tell me about Python"):
            print(chunk, end="", flush=True)
        
        # Branch to explore alternatives without affecting main conversation
        async with chat.branch() as branch:
            await branch.ask("Actually, I want to learn Rust instead")
            response = await branch.ask("What makes Rust special?")
            print(response)  # Rust information
        
        # Main conversation still remembers Python context
        response = await chat.ask("What Python framework should I learn?")
        print(response)  # Python frameworks, not Rust
        
        # Save for later
        conversation_id = await chat.save()
    
    # Resume conversation later
    async with conversation(resume_from=conversation_id) as chat:
        response = await chat.ask("What was I learning?")
        print(response)  # "You were learning Python"

asyncio.run(chat_example())
```

#### Sync Conversation (Simpler for scripts)

```python
from chuk_llm import conversation_sync

# Synchronous API - great for simple scripts
with conversation_sync(provider="openai", model="gpt-4o-mini") as chat:
    # Basic conversation
    chat.ask("I'm planning a dinner party")
    response = chat.ask("What's a good appetizer?")
    print(response)
    
    # Note: Real-time streaming requires async conversation
    # In sync mode, use ask() for complete responses
    full_response = chat.ask("Give me the recipe")
    print(full_response)
    
    # Branching in sync mode
    with chat.branch() as branch:
        branch.ask("What if I have vegetarian guests?")
        veg_option = branch.ask("Suggest vegetarian appetizers")
        print(veg_option)
    
    # Main conversation unaffected by branch
    response = chat.ask("What wine pairs well with the appetizer?")
    print(response)  # Pairs with original appetizer, not vegetarian
```

### Dynamic Model Discovery

ChukLLM automatically discovers and generates functions for available models:

```python
# Ollama models are discovered automatically
# ollama pull gpt-oss
# ollama pull llama3.2

from chuk_llm import (
    ask_ollama_gpt_oss,          # Auto-generated with smart detection!
    ask_ollama_llama3_2,         # Works in both sync and async!
)

# No need for _sync suffix anymore!
response = ask_ollama_gpt_oss("Think through this problem")

# Trigger discovery manually
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
new_functions = trigger_ollama_discovery_and_refresh()
print(f"Discovered {len(new_functions)} new functions")
```

#### Provider Discovery Examples

Run the discovery examples to see available models for each provider:

```bash
# Discover OpenAI models
uv run examples/llm_provider_examples/openai_usage_examples.py

# Discover Anthropic models  
uv run examples/llm_provider_examples/anthropic_usage_examples.py

# Discover local Ollama models
uv run examples/llm_provider_examples/ollama_usage_examples.py

# Discover Azure OpenAI deployments
uv run examples/llm_provider_examples/azure_usage_examples.py
```

Available provider examples:
- `anthropic_usage_examples.py` - Claude models with Opus/Sonnet/Haiku families
- `azure_usage_examples.py` - Azure OpenAI deployments with auto-discovery
- `deepseek_usage_examples.py` - DeepSeek reasoning and chat models
- `gemini_usage_examples.py` - Google Gemini models with multimodal support
- `groq_usage_examples.py` - Groq's ultra-fast inference models
- `mistral_usage_examples.py` - Mistral AI models including Magistral
- `openai_usage_examples.py` - OpenAI GPT models including O1/O3 reasoning
- `openrouter_usage_examples.py` - OpenRouter's model marketplace
- `perplexity_usage_examples.py` - Perplexity's search-enhanced models
- `watsonx_usage_examples.py` - IBM Watsonx Granite models

### Session Analytics

```python
from chuk_llm import ask, get_session_stats, get_session_history

# All calls are automatically tracked
await ask("What's the capital of France?")
await ask("What's 2+2?")

# Get comprehensive analytics
stats = await get_session_stats()
print(f"📊 Tracked {stats['total_messages']} messages")
print(f"💰 Total cost: ${stats['estimated_cost']:.6f}")
print(f"🔤 Total tokens: {stats['total_tokens']}")

# View complete history
history = await get_session_history()
for msg in history:
    print(f"{msg['role']}: {msg['content'][:50]}...")
```

## 📚 Conversation API Reference

The conversation API provides stateful chat with memory, branching, and streaming:

### Async API (Full Features)

```python
from chuk_llm import conversation
import asyncio

async with conversation(provider="openai", model="gpt-4o-mini") as chat:
    # ask() - Send a message and get a response
    response = await chat.ask("Hello!")
    
    # stream() - Stream the response in real-time (async only)
    async for chunk in chat.stream("Tell me a story"):
        print(chunk, end="", flush=True)
    
    # branch() - Create isolated exploration paths
    async with chat.branch() as branch:
        await branch.ask("What if...")  # Doesn't affect main conversation
    
    # save() - Save conversation state
    conversation_id = await chat.save()
    
    # Other methods
    history = chat.get_history()
    stats = chat.get_stats()
    chat.clear()  # Clear history (keeps system prompt)
    chat.pop_last()  # Remove last exchange
```

### Sync API (Simpler Interface)

```python
from chuk_llm import conversation_sync

with conversation_sync(provider="openai", model="gpt-4o-mini") as chat:
    # ask() - Send message and get response
    response = chat.ask("Hello!")
    
    # branch() - Works in sync mode too
    with chat.branch() as branch:
        branch.ask("Alternative path...")
    
    # save() and other methods work the same
    conversation_id = chat.save()
    
    # All methods available: ask(), branch(), save(), get_history(), etc.
    # Note: Real-time streaming requires async API
```

### Advanced Features

```python
# Resume a saved conversation (async or sync)
async with conversation(resume_from=conversation_id) as chat:
    await chat.ask("Continue where we left off")

# Multi-modal support (images)
response = await chat.ask("What's in this image?", image="path/to/image.jpg")

# Automatic session tracking (when chuk-ai-session-manager is installed)
stats = await chat.get_session_stats()  # Includes tokens, costs, duration

# Extract insights
summary = await chat.summarize(max_length=500)
key_points = await chat.extract_key_points()
```

## 🏷️ Model Aliases - Simplify Your Life

ChukLLM supports global and provider-specific aliases to make model names easier to remember:

```python
from chuk_llm import ask_sync

# Global aliases work across providers
response = ask_sync("Hello", provider="ollama", model="granite")  # Resolves to granite3.3
response = ask_sync("Hello", provider="openai", model="gpt")      # Resolves to gpt-4o-mini  
response = ask_sync("Hello", provider="anthropic", model="claude") # Resolves to claude-3-sonnet

# Direct alias functions (auto-generated)
from chuk_llm import ask_granite, ask_claude, ask_gpt
response = ask_granite("What is Python?")  # Uses ollama/granite3.3
response = ask_claude("Explain REST APIs")  # Uses anthropic/claude-3-sonnet
response = ask_gpt("Tell me a joke")       # Uses openai/gpt-4o-mini
```

CLI with aliases:
```bash
# Global aliases as commands
chuk-llm ask_granite "What is machine learning?"
chuk-llm ask_claude "Explain quantum computing"
chuk-llm ask_gpt "Write a haiku"

# Using aliases with providers
chuk-llm ask "Hello" --provider ollama --model granite
chuk-llm ask "Test" --provider openai --model gpt
```

Configure your own aliases in `providers.yaml`:
```yaml
__global_aliases__:
  # Your custom aliases
  fast: ollama/granite3.3
  smart: openai/gpt-4o
  creative: anthropic/claude-3-opus
  
ollama:
  model_aliases:
    # Provider-specific aliases
    local: granite3.3
    quick: llama3.2
```

## 🔧 Configuration

### Environment Variables

```bash
# Core API Keys
export OPENAI_API_KEY="your-openai-key"
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"

# OpenAI-Compatible Services
export LITELLM_API_BASE="http://localhost:4000"
export OPENROUTER_API_KEY="sk-or-..."
export VLLM_API_BASE="http://localhost:8000/v1"
export TOGETHER_API_KEY="..."

# Custom endpoints (override defaults)
export OPENAI_API_BASE="https://api.openai.com/v1"
export PERPLEXITY_API_BASE="https://api.perplexity.ai"
export OLLAMA_API_BASE="http://localhost:11434"

# Session tracking
export CHUK_LLM_DISABLE_SESSIONS="false"
export SESSION_PROVIDER="redis"  # or "memory"
export SESSION_REDIS_URL="redis://localhost:6379/0"

# Discovery settings
export CHUK_LLM_DISCOVERY_ENABLED="true"
export CHUK_LLM_OLLAMA_DISCOVERY="true"
export CHUK_LLM_AUTO_DISCOVER="true"
export CHUK_LLM_DISCOVERY_TIMEOUT="5"
```

### Programmatic Configuration

```python
from chuk_llm import configure, get_current_config

# Simple configuration
configure(
    provider="openai",
    model="gpt-5",
    temperature=0.7
)

# Check current configuration
config = get_current_config()
print(f"Using {config['provider']} with {config['model']}")

# Quick setup helpers
from chuk_llm import quick_setup, switch_provider

# Setup a provider quickly
quick_setup("openai", model="gpt-5")

# Switch between providers
switch_provider("anthropic", model="claude-4-sonnet")
```

## 📦 Installation Options

| Command | Features | Use Case |
|---------|----------|----------|
| `uv add chuk_llm` | Core + Memory sessions | Development |
| `uv add chuk_llm[redis]` | Core + Redis sessions | Production |
| `uv add chuk_llm[cli]` | Core + Enhanced CLI | CLI tools |
| `uv add chuk_llm[all]` | Everything | Full features |
| `pip install chuk_llm[all]` | Everything (alternative) | If not using uv |

## 🔍 Troubleshooting

### Common Issues and Solutions

```python
# Issue: "Provider not found"
# Solution: Check available providers
from chuk_llm import list_providers
print(list_providers())  # Shows all configured providers

# Issue: "Model not found"  
# Solution: Check available models for a provider
from chuk_llm import list_models
print(list_models("openai"))  # Shows ['gpt-4o', 'gpt-4o-mini', ...]

# Issue: Ollama functions not appearing
# Solution: Trigger discovery
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
trigger_ollama_discovery_and_refresh()

# Issue: API key not configured
# Solution: Set environment variable or use config
import os
os.environ['OPENAI_API_KEY'] = 'your-key'
# Or use dynamic registration
from chuk_llm import register_provider
register_provider("openai", api_key="your-key")

# Issue: Connection timeouts
# Solution: Increase timeout
from chuk_llm import ask_sync
response = ask_sync("Hello", timeout=60)  # 60 second timeout

# Issue: Rate limiting
# Solution: Add retry logic
import time
for attempt in range(3):
    try:
        response = ask_sync("Hello")
        break
    except Exception as e:
        if "rate limit" in str(e).lower():
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
```

### Debugging Tips

```bash
# Enable verbose mode in CLI
chuk-llm ask "test" --provider openai --verbose

# Check your configuration
chuk-llm config

# Test provider connectivity
chuk-llm test openai
chuk-llm test ollama

# List all available functions
chuk-llm functions

# Check discovered models
chuk-llm discovered ollama
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🚀 What's Next?

- **🔌 More Providers**: Cohere, AI21, Replicate integration
- **🧠 Advanced Reasoning**: Support for O6, O7 series and Claude 5
- **🌐 Multi-Modal**: Enhanced image, audio, and document processing
- **🔧 Tool Orchestration**: Advanced workflows with tool dependencies
- **📊 Analytics Dashboard**: Web UI for session analytics
- **🏢 Enterprise Features**: SSO, audit logs, compliance tools
- **⚡ Performance**: WebSocket streaming, connection pooling optimizations
- **🔐 Security**: End-to-end encryption for sensitive workloads

## 📊 Performance Benchmarks

```python
# Concurrent vs Sequential Performance
import asyncio
import time
from chuk_llm import ask

async def benchmark():
    questions = ["Question 1", "Question 2", "Question 3"]
    
    # Sequential
    start = time.time()
    for q in questions:
        await ask(q, provider="openai")
    sequential_time = time.time() - start
    
    # Concurrent
    start = time.time()
    await asyncio.gather(*[ask(q, provider="openai") for q in questions])
    concurrent_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.1f}x")

asyncio.run(benchmark())
```

Typical results:
- Sequential: 6.2s
- Concurrent: 1.8s  
- **Speedup: 3.4x faster!**

## 🧪 Testing

```python
# Test provider connectivity
from chuk_llm import test_connection_sync, test_all_providers_sync

# Test single provider
result = test_connection_sync("openai", model="gpt-4o")
print(f"Response time: {result['duration']:.2f}s")

# Test all configured providers
results = test_all_providers_sync()
for provider, result in results.items():
    if result["success"]:
        print(f"✅ {provider}: {result['duration']:.2f}s")
    else:
        print(f"❌ {provider}: {result['error']}")
```

## 📚 Examples

### 🖼️ Vision - Analyze Images

```python
from chuk_llm import ask_openai_gpt4o, ask_anthropic_sonnet, ask_gemini_flash

# Analyze local image file
response = ask_openai_gpt4o(
    "What objects do you see in this image?",
    "photo.png"  # Supports PNG, JPG, etc.
)

# With image bytes
from PIL import Image
import io

img = Image.new("RGB", (200, 200), color="blue")
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG")

response = ask_anthropic_sonnet(
    "What color is this image?",
    img_bytes.getvalue()
)

# Compare vision models
providers = [
    ("GPT-4o", ask_openai_gpt4o),
    ("Claude", ask_anthropic_sonnet),
    ("Gemini", ask_gemini_flash),
]

for name, func in providers:
    response = func("Describe this chart", "sales_chart.png")
    print(f"{name}: {response[:100]}...")
```

### 🔄 Multi-Provider Workflows

```python
from chuk_llm import ask_sync

# Use different providers for different tasks
def analyze_document(text):
    # Use GPT for extraction
    summary = ask_sync(
        f"Summarize this text: {text}",
        provider="openai",
        model="gpt-4o-mini"
    )
    
    # Use Claude for analysis
    analysis = ask_sync(
        f"Analyze the sentiment of: {summary}",
        provider="anthropic",
        model="claude-3-sonnet"
    )
    
    # Use local model for classification
    category = ask_sync(
        f"Categorize this into one word: {summary}",
        provider="ollama",
        model="granite3.3"
    )
    
    return {
        "summary": summary,
        "sentiment": analysis,
        "category": category
    }
```

### 🎯 Model-Specific Features

```python
# GPT-5 and O1/O3 Reasoning Models
from chuk_llm import ask_sync

# GPT-5 - No temperature or max_tokens needed
response = ask_sync(
    "Solve this complex problem step by step...",
    provider="openai",
    model="gpt-5"
)

# O1 Reasoning - Longer thinking time
response = ask_sync(
    "Explain the solution to this puzzle...",
    provider="openai",
    model="o1-preview",
    # O1 models think for 10-30 seconds
)

# DeepSeek Reasoner - Extended reasoning
response = ask_sync(
    "Prove this mathematical theorem...",
    provider="deepseek",
    model="deepseek-reasoner",
    # Can take 30-60 seconds for complex reasoning
)

# Mistral Magistral with thinking tags
response = ask_sync(
    "Debug this code and explain your reasoning...",
    provider="mistral",
    model="mistral-large-2411"
    # Returns <think> tags with reasoning process
)
```

### 📝 Structured Output with JSON Mode

```python
from chuk_llm import ask_json

# Get structured JSON responses
recipe = ask_json(
    "Give me a recipe for chocolate cake with ingredients and steps",
    provider="openai"
)

print(recipe["ingredients"])  # List of ingredients
print(recipe["steps"])  # List of steps

# Works with CLI too
# chuk-llm ask "List 3 colors" --json --provider openai
```

### Compare Multiple Providers

```python
from chuk_llm import compare_providers

results = compare_providers(
    "Explain quantum computing in one sentence",
    ["openai", "anthropic", "gemini", "groq"]
)

for provider, response in results.items():
    print(f"{provider}: {response}")
```

### Stream with Multiple Providers

```python
import asyncio
from chuk_llm import stream

async def multi_stream():
    providers = ["openai", "anthropic", "ollama"]
    prompt = "Write a haiku about coding"
    
    async def stream_provider(provider):
        print(f"\n{provider.upper()}:")
        async for chunk in stream(prompt, provider=provider):
            print(chunk, end="", flush=True)
    
    await asyncio.gather(*[stream_provider(p) for p in providers])

asyncio.run(multi_stream())
```

### Use with Pandas DataFrames

```python
import pandas as pd
from chuk_llm import ask_sync

# Process DataFrame with LLM
df = pd.DataFrame({
    'product': ['laptop', 'phone', 'tablet'],
    'review': ['Great device!', 'Battery issues', 'Perfect for reading']
})

# Add sentiment analysis
df['sentiment'] = df['review'].apply(
    lambda x: ask_sync(f"Sentiment of '{x}' (positive/negative/neutral):", 
                      provider="openai", model="gpt-4o-mini")
)

print(df)
```

### Build a Simple Chatbot

```python
from chuk_llm import conversation
import asyncio

async def chatbot():
    print("Chatbot ready! Type 'quit' to exit.\n")
    
    async with conversation(provider="openai", model="gpt-4o") as chat:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            
            response = await chat.ask(user_input)
            print(f"Bot: {response}\n")

asyncio.run(chatbot())
```

### Use System Prompts for Custom Personalities

```python
from chuk_llm import ask_sync, ask_ollama_granite

# Make the AI respond with different personalities
pirate_prompt = "You are a pirate captain. Speak in pirate dialect with 'arr' and 'matey'."
response = ask_sync("What is Python?", provider="ollama", model="granite3.3:latest", 
                   system_prompt=pirate_prompt)
print(f"🏴‍☠️ {response}")

# Works with auto-detection too!
professor_prompt = "You are a university professor. Be academic and thorough."
response = ask_ollama_granite("Explain recursion", system_prompt=professor_prompt)
print(f"👨‍🏫 {response}")

# CLI examples
# chuk-llm ask "What is AI?" -p ollama -m granite3.3:latest -s "Explain like I'm 5"
# chuk-llm ask "Debug this" --provider openai --system-prompt "You are a senior developer"
```

### Function Calling Example

```python
from chuk_llm import ask_sync
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = ask_sync(
    "What's the weather in Paris?",
    provider="openai",
    model="gpt-4o",
    tools=tools
)

print(response)
# The model will call the get_weather function with location="Paris"
```

## 🛠️ Troubleshooting

### Common Issues

**Issue: "No API key found"**
```bash
# Set your API key
export OPENAI_API_KEY="sk-..."
# Or use a .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

**Issue: "Provider not found"**
```python
# Check available providers
from chuk_llm import list_available_providers
print(list_available_providers())

# Register a new provider
from chuk_llm import register_openai_compatible
register_openai_compatible(
    name="my_provider",
    api_base="https://api.example.com/v1",
    api_key="your-key"
)
```

**Issue: "Connection timeout"**
```python
# Use a different endpoint
from chuk_llm import ask_sync
response = ask_sync(
    "Hello",
    provider="ollama",
    base_url="http://192.168.1.100:11434"  # Custom Ollama server
)
```

**Issue: "Model not available"**
```python
# Discover available models
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
trigger_ollama_discovery_and_refresh()

# Or check provider models
from chuk_llm import get_provider_config
config = get_provider_config("ollama")
print(config.models)
```

## 📞 Support

- **Documentation**: [docs.chukai.io](https://docs.chuk-llm.dev)
- **Issues**: [GitHub Issues](https://github.com/chuk-llm/chuk-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chuk-llm/chuk-llm/discussions)
- **Email**: support@chukai.io

## 🌟 Why Choose ChukLLM?

1. **Universal Compatibility**: Works with 100+ LLM providers through OpenAI-compatible APIs
2. **Zero Lock-in**: Switch providers with one line of code
3. **Production Ready**: Built-in retries, connection pooling, error handling
4. **Developer Friendly**: Auto-generated functions, great documentation
5. **Cost Tracking**: Automatic session analytics and cost estimation
6. **Enterprise Features**: Azure OpenAI support, compliance, audit logs
7. **Active Development**: Regular updates with new providers and features
8. **Community Driven**: Open source with active community contributions

---

**⭐ Star us on GitHub if ChukLLM helps your AI projects!**

Built with ❤️ by the ChukLLM team