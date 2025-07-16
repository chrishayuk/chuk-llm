# chuk_llm

A unified, production-ready Python library for Large Language Model (LLM) providers with real-time streaming, function calling, middleware support, automatic session tracking, dynamic model discovery, intelligent system prompt generation, and a powerful CLI.

## 🚀 QuickStart

### Installation

```bash
# Core functionality with session tracking (memory storage)
pip install chuk_llm

# With Redis for persistent sessions
pip install chuk_llm[redis]

# With watsonx
pip install chuk_llm[watsonx]

# With enhanced CLI experience
pip install chuk_llm[cli]

# Full installation
pip install chuk_llm[all]
```

### API Keys Setup

```bash
export OPENAI_API_KEY="your-openai-key"
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
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

# Provider-specific functions (auto-generated!)
from chuk_llm import ask_openai_sync, ask_azure_openai_sync, ask_claude_sync, ask_groq_sync, ask_watsonx_sync

openai_response = ask_openai_sync("Tell me a joke")
azure_response = ask_azure_openai_sync("Explain quantum computing")
claude_response = ask_claude_sync("Write a Python function") 
groq_response = ask_groq_sync("What's the weather like?")
watsonx_response = ask_watsonx_sync("Write enterprise Python code")

# ✨ NEW: Dynamic convenience functions for discovered models
from chuk_llm import ask_ollama_llama3_2_sync, ask_ollama_gemma3_sync, ask_ollama_qwen3_sync
local_response = ask_ollama_llama3_2_sync("Write Python code")
gemma_response = ask_ollama_gemma3_sync("Explain machine learning")

# Configure once, use everywhere
configure(provider="azure_openai", temperature=0.7)
response = ask_sync("Write a creative story opening")

# Compare multiple providers
from chuk_llm import compare_providers
results = compare_providers("What is AI?", ["openai", "azure_openai", "anthropic"])
for provider, response in results.items():
    print(f"{provider}: {response}")
```

### 🖥️ Command Line Interface (CLI)

ChukLLM includes a powerful CLI for quick AI interactions from your terminal:

```bash
# Quick questions using global aliases
chuk-llm ask_granite "What is Python?"
chuk-llm ask_claude "Explain quantum computing"
chuk-llm ask_gpt "Write a haiku about code"
chuk-llm ask_azure "Deploy models to Azure"

# ✨ NEW: Convenience functions for discovered models
chuk-llm ask_ollama_gemma3 "Hello world"
chuk-llm ask_ollama_mistral_small_latest "Tell me a joke"
chuk-llm ask_ollama_qwen3_latest "Explain Python"
chuk-llm stream_ollama_llama3_2 "Write a long explanation"

# General ask command with provider selection
chuk-llm ask "What is machine learning?" --provider azure_openai --model gpt-4o-mini

# JSON responses for structured output
chuk-llm ask "List 3 Python libraries" --json --provider azure_openai

# Provider and model management
chuk-llm providers              # Show all available providers
chuk-llm models azure_openai    # Show models for Azure OpenAI
chuk-llm test azure_openai      # Test Azure connection
chuk-llm discover ollama        # Discover new Ollama models ✨ NEW

# Configuration and diagnostics
chuk-llm config                 # Show current configuration
chuk-llm functions              # List all auto-generated functions ✨ NEW
chuk-llm aliases                # Show available global aliases
chuk-llm help                   # Comprehensive help

# Use with uvx for zero-install usage
uvx chuk-llm ask_azure "What is Azure OpenAI?"
uvx chuk-llm ask_ollama_gemma3 "Hello"  # ✨ NEW
uvx chuk-llm providers
```

#### CLI Features

- **🎯 Global Aliases**: Quick commands like `ask_granite`, `ask_claude`, `ask_gpt`, `ask_azure`
- **✨ Dynamic Convenience Functions**: Auto-generated functions like `ask_ollama_gemma3`, `ask_ollama_mistral_small_latest`
- **🌊 Real-time Streaming**: See responses as they're generated
- **🔧 Provider Management**: Test, discover, and configure providers
- **📊 Rich Output**: Beautiful tables and formatting (with `[cli]` extra)
- **🔍 Discovery Integration**: Find and use new Ollama models instantly
- **⚡ Fast Feedback**: Immediate responses with connection testing
- **🎨 Quiet/Verbose Modes**: Control output detail with `--quiet` or `--verbose`

#### CLI Installation Options

| Command | CLI Features | Use Case |
|---------|-------------|----------|
| `pip install chuk_llm` | Basic CLI | Quick terminal usage |
| `pip install chuk_llm[cli]` | Enhanced CLI with rich formatting | Beautiful terminal experience |
| `pip install chuk_llm[all]` | Enhanced CLI + Redis + All features | Complete installation |

### Async API - Production Performance (3-7x faster!)

```python
import asyncio
from chuk_llm import ask, stream, conversation

async def main():
    # Basic async call
    response = await ask("Hello!")
    
    # Provider-specific async functions
    from chuk_llm import ask_openai, ask_azure_openai, ask_claude, ask_groq, ask_watsonx
    
    openai_response = await ask_openai("Tell me a joke")
    azure_response = await ask_azure_openai("Explain quantum computing")
    claude_response = await ask_claude("Write a Python function")
    groq_response = await ask_groq("What's the weather like?")
    watsonx_response = await ask_watsonx("Generate enterprise documentation")
    
    # ✨ NEW: Dynamic async functions for discovered models
    from chuk_llm import ask_ollama_llama3_2, ask_ollama_gemma3, stream_ollama_qwen3
    local_response = await ask_ollama_llama3_2("Write Python code")
    gemma_response = await ask_ollama_gemma3("Explain AI")
    
    # Real-time streaming (token by token)
    print("Streaming: ", end="", flush=True)
    async for chunk in stream_ollama_qwen3("Write a haiku about coding"):
        print(chunk, end="", flush=True)
    
    # Conversations with memory
    async with conversation(provider="azure_openai") as chat:
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

### 🔍 Dynamic Model Discovery for Ollama - ✨ ENHANCED!

ChukLLM automatically discovers and generates functions for Ollama models in real-time. **NEW**: Now with seamless CLI integration!

```python
# Start Ollama with some models
# ollama pull llama3.2
# ollama pull qwen2.5:14b
# ollama pull deepseek-coder:6.7b
# ollama pull mistral-small:latest

# ChukLLM automatically discovers them and generates functions!
from chuk_llm import (
    ask_ollama_llama3_2_sync,          # Auto-generated!
    ask_ollama_qwen2_5_14b_sync,       # Auto-generated!
    ask_ollama_deepseek_coder_6_7b_sync, # Auto-generated!
    ask_ollama_mistral_small_latest_sync  # Auto-generated!
)

# Use immediately without any configuration
response = ask_ollama_llama3_2_sync("Write a Python function to sort a list")

# Trigger discovery manually to refresh available models
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
new_functions = trigger_ollama_discovery_and_refresh()
print(f"Discovered {len(new_functions)} new functions!")

# ✨ NEW: CLI discovery with instant function availability
# chuk-llm discover ollama
# chuk-llm ask_ollama_mistral_small_latest "Hello"  # Works immediately!
# chuk-llm functions  # See all available functions
```

#### Discovery Features

- **🔍 Real-time Detection**: Automatically finds new Ollama models as they're pulled
- **⚡ Instant Functions**: Generates `ask_*` and `stream_*` functions immediately
- **🖥️ CLI Integration**: New models work instantly in CLI with convenience syntax
- **🧠 Smart Caching**: Remembers discovered models between sessions
- **📊 Environment Controls**: Fine-grained control over discovery behavior
- **🔄 Live Updates**: Refreshes available models without restart

#### Discovery Environment Controls

```bash
# Enable/disable discovery globally
export CHUK_LLM_DISCOVERY_ENABLED=true

# Provider-specific discovery control
export CHUK_LLM_OLLAMA_DISCOVERY=true
export CHUK_LLM_AUTO_DISCOVER=true

# Performance tuning
export CHUK_LLM_DISCOVERY_TIMEOUT=5
export CHUK_LLM_DISCOVERY_CACHE_TIMEOUT=300
```

### Discovery Integration Examples

```python
# Example: Pull a new model and use immediately
# Terminal 1: ollama pull phi3:latest
# Terminal 2: Python code (no restart needed!)

from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh

# Refresh to pick up new model
trigger_ollama_discovery_and_refresh()

# New function is now available!
from chuk_llm import ask_ollama_phi3_latest_sync
response = ask_ollama_phi3_latest_sync("Explain quantum computing")

# Or use via CLI immediately:
# chuk-llm ask_ollama_phi3_latest "Explain quantum computing"
```

### 🧠 Intelligent System Prompt Generation - NEW!

ChukLLM features an advanced system prompt generator that automatically creates optimized prompts based on provider capabilities, tools, and context:

```python
from chuk_llm import ask_sync

# Basic example - ChukLLM automatically generates appropriate system prompts
response = ask_sync("Help me write a Python function")
# Automatically gets system prompt optimized for code generation

# With function calling - system prompt automatically includes tool usage instructions
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                }
            }
        }
    }
]

response = ask_sync("What's 15% of 250?", tools=tools)
# System prompt automatically includes function calling guidelines optimized for the provider
```

#### Provider-Optimized System Prompts

The system prompt generator creates different prompts based on provider capabilities:

```python
# Anthropic gets Claude-optimized prompts
from chuk_llm import ask_claude_sync
response = ask_claude_sync("Explain quantum physics", tools=tools)
# System prompt: "You are Claude, an AI assistant created by Anthropic. You have access to tools..."

# OpenAI gets GPT-optimized prompts  
from chuk_llm import ask_openai_sync
response = ask_openai_sync("Explain quantum physics", tools=tools)
# System prompt: "You are a helpful assistant with access to function calling capabilities..."

# Azure OpenAI gets enterprise-optimized prompts
from chuk_llm import ask_azure_openai_sync
response = ask_azure_openai_sync("Explain quantum physics", tools=tools)
# System prompt: "You are an enterprise AI assistant powered by Azure OpenAI. You have access to function calling capabilities..."

# Groq gets ultra-fast inference optimized prompts
from chuk_llm import ask_groq_sync
response = ask_groq_sync("Explain quantum physics", tools=tools)
# System prompt: "You are an intelligent assistant with function calling capabilities. Take advantage of ultra-fast inference..."

# IBM Granite gets enterprise-optimized prompts
from chuk_llm import ask_watsonx_sync
response = ask_watsonx_sync("Explain quantum physics", tools=tools)
# System prompt: "You are an enterprise AI assistant with function calling capabilities. Provide professional, accurate responses..."
```

### 🎯 Automatic Session Tracking

ChukLLM includes automatic session tracking powered by `chuk-ai-session-manager`. Every API call is automatically tracked for complete observability:

```python
from chuk_llm import ask, get_session_stats, get_session_history

# All calls are automatically tracked
await ask("What's the capital of France?")
await ask("What's 2+2?")

# Get comprehensive analytics
stats = await get_session_stats()
print(f"📊 Tracked {stats['total_messages']} messages")
print(f"💰 Total cost: ${stats['estimated_cost']:.6f}")

# View complete history
history = await get_session_history()
for msg in history:
    print(f"{msg['role']}: {msg['content'][:50]}...")
```

**Session Storage Options:**
- **Memory (default)**: Fast, no persistence, no extra dependencies
- **Redis**: Persistent, requires `pip install chuk_llm[redis]`

```bash
# Configure session storage
export SESSION_PROVIDER=memory  # Default
export SESSION_PROVIDER=redis   # Requires [redis] extra
export SESSION_REDIS_URL=redis://localhost:6379/0
```

### 🎭 Enhanced Conversations

ChukLLM supports advanced conversation features for building sophisticated dialogue systems:

#### 1. Conversation Branching
```python
async with conversation() as chat:
    await chat.say("Let's plan a vacation")
    
    # Branch to explore Japan
    async with chat.branch() as japan_branch:
        await japan_branch.say("Tell me about visiting Japan")
        # This conversation stays isolated
    
    # Main conversation doesn't know about branches
    await chat.say("I've decided on Japan!")
```

#### 2. Conversation Persistence
```python
# Start a conversation
async with conversation() as chat:
    await chat.say("I'm learning Python")
    conversation_id = await chat.save()

# Resume days later
async with conversation(resume_from=conversation_id) as chat:
    response = await chat.say("What should I learn next?")
    # AI remembers your background!
```

### 150+ Auto-Generated Functions ✨ EXPANDED

ChukLLM automatically creates functions for every provider and model, **including dynamically discovered ones**:

```python
# Base provider functions
from chuk_llm import ask_openai, ask_azure_openai, ask_anthropic, ask_groq, ask_ollama, ask_watsonx

# Model-specific functions (auto-generated from config + discovery)
from chuk_llm import ask_openai_gpt4o, ask_azure_openai_gpt4o, ask_claude_sonnet, ask_watsonx_granite

# ✨ NEW: Dynamically discovered functions (auto-generated from Ollama)
from chuk_llm import (
    ask_ollama_llama3_2,              # Discovered from ollama pull llama3.2
    ask_ollama_mistral_small_latest,  # Discovered from ollama pull mistral-small:latest
    ask_ollama_gemma3,                # Discovered from ollama pull gemma3
    ask_ollama_qwen3_latest,          # Discovered from ollama pull qwen3:latest
    stream_ollama_phi3,               # Stream version auto-generated too!
)

# All with sync, async, and streaming variants!
from chuk_llm import (
    ask_openai_sync,                  # Synchronous
    ask_azure_openai_sync,            # Azure OpenAI sync
    stream_anthropic,                 # Async streaming  
    ask_groq_sync,                    # Sync version
    ask_ollama_llama3_2_sync,         # Auto-discovered local model (sync)
    stream_ollama_mistral_small_latest # Auto-discovered streaming function
)
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

✅ **150+ Auto-Generated Functions** - Every provider & model + discovered models  
✅ **3-7x Performance Boost** - Concurrent requests vs sequential  
✅ **Real-time Streaming** - Token-by-token output as it's generated  
✅ **Memory Management** - Stateful conversations with context  
✅ **Automatic Session Tracking** - Zero-config usage analytics & cost monitoring  
✅ **✨ Dynamic Model Discovery** - Automatically detect and generate functions for new models  
✅ **Intelligent System Prompts** - Provider-optimized prompts with tool integration  
✅ **✨ Enhanced CLI** - Terminal access with streaming, discovery, and convenience functions  
✅ **Production Ready** - Error handling, retries, connection pooling  
✅ **Developer Friendly** - Simple sync for scripts, powerful async for apps  

## 📦 Installation

### Basic Installation

```bash
# Core functionality with session tracking (memory storage)
pip install chuk_llm

# Session tracking is included by default!
# chuk-ai-session-manager is a core dependency
```

### Production Installation

```bash
# With Redis support for persistent session storage
pip install chuk_llm[redis]

# This adds Redis support to the included session manager
```

### Enhanced Installation

```bash
# Enhanced CLI experience with rich formatting
pip install chuk_llm[cli]

# Full installation with all features
pip install chuk_llm[all]

# Development installation
pip install chuk_llm[dev]
```

### Installation Matrix

| Command | Session Storage | CLI Features | Use Case |
|---------|----------------|--------------|----------|
| `pip install chuk_llm` | Memory (included) | Basic | Development, scripting |
| `pip install chuk_llm[redis]` | Memory + Redis | Basic | Production apps |
| `pip install chuk_llm[cli]` | Memory (included) | Enhanced | CLI tools |
| `pip install chuk_llm[all]` | Memory + Redis | Enhanced | Full features |

### Session Storage Configuration

```bash
# Default: Memory storage (fast, no persistence)
export SESSION_PROVIDER=memory

# Production: Redis storage (persistent, requires redis extra)
export SESSION_PROVIDER=redis
export SESSION_REDIS_URL=redis://localhost:6379/0

# Disable session tracking entirely
export CHUK_LLM_DISABLE_SESSIONS=true
```

### Provider-Specific Dependencies

All major LLM providers are included by default. Optional dependencies are only for storage and CLI enhancements:

```bash
# Providers included out of the box:
# ✅ OpenAI (GPT-4, GPT-3.5)
# ✅ Azure OpenAI (Enterprise GPT models with Azure security)
# ✅ Anthropic (Claude 3.5 Sonnet, Haiku)  
# ✅ Google Gemini (2.0 Flash, 1.5 Pro)
# ✅ Groq (Llama models with ultra-fast inference)
# ✅ Perplexity (Real-time web search)
# ✅ Ollama (Local models with dynamic discovery) ✨ ENHANCED
# ✅ IBM watsonx & Granite (Enterprise models)
# ✅ Mistral AI
```

## 🚀 Features

### Multi-Provider Support
- **OpenAI** - GPT-4, GPT-3.5 with full API support
- **Azure OpenAI** - Enterprise GPT models with Azure security and compliance features
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Haiku
- **Google Gemini** - Gemini 2.0 Flash, Gemini 1.5 Pro  
- **Groq** - Lightning-fast inference with Llama models
- **Perplexity** - Real-time web search with Sonar models
- **Ollama** - Local model deployment with **✨ dynamic discovery and CLI integration**
- **IBM Watson** - Enterprise Granite models with SOC2 compliance
- **Mistral AI** - Efficient European LLM models

### Core Capabilities
- 🌊 **Real-time Streaming** - True streaming without buffering
- 🛠️ **Function Calling** - Standardized tool/function execution
- 📊 **Automatic Session Tracking** - Usage analytics with zero configuration
- 💰 **Cost Monitoring** - Real-time spend tracking across all providers
- 🔧 **Middleware Stack** - Logging, metrics, caching, retry logic
- 📈 **Performance Monitoring** - Built-in benchmarking and metrics
- 🔄 **Error Handling** - Automatic retries with exponential backoff
- 🎯 **Type Safety** - Full Pydantic validation and type hints
- 🧩 **Extensible Architecture** - Easy to add new providers

### Advanced Features
- **🧠 Intelligent System Prompts** - Provider-optimized prompt generation
- **✨ Dynamic Model Discovery** - Automatic detection of new models with seamless integration
- **🖥️ Enhanced CLI** - Terminal interface with streaming, discovery, and convenience functions
- **Vision Support** - Image analysis across compatible providers
- **JSON Mode** - Structured output generation
- **Real-time Web Search** - Live information retrieval with citations
- **Parallel Function Calls** - Execute multiple tools simultaneously
- **Connection Pooling** - Efficient HTTP connection management
- **Configuration Management** - Environment-based provider setup
- **Capability Detection** - Automatic feature detection per provider
- **Infinite Context** - Automatic conversation segmentation for long chats
- **Conversation History** - Full audit trail of all interactions

### Enhanced Conversation Features
- **🌿 Conversation Branching** - Explore multiple paths without affecting main thread
- **💾 Conversation Persistence** - Save and resume conversations across sessions
- **🖼️ Multi-Modal Support** - Send images with text in conversations
- **📊 Built-in Utilities** - Summarize, extract key points, get statistics
- **🎯 Stateless Context** - Add context to one-off questions without state

### ✨ Enhanced Dynamic Discovery Features
- **🔍 Real-time Model Detection** - Automatically discover new Ollama models
- **⚡ Function Generation** - Create provider functions on-demand
- **🧠 Capability Inference** - Automatically detect model features and limitations
- **📦 Universal Discovery** - Support for multiple discovery sources (Ollama, HuggingFace, etc.)
- **🔄 Cache Management** - Intelligent caching with automatic refresh
- **📊 Discovery Analytics** - Statistics and insights about discovered models
- **🖥️ CLI Integration** - Discovered models work instantly in CLI with convenience syntax
- **🎯 Environment Controls** - Fine-grained control over discovery behavior
- **⚡ Live Updates** - Refresh available models without restart
- **🔧 Provider Integration** - Discovered models work with all chuk_llm APIs

## 🖥️ CLI Guide

### Quick Commands

```bash
# Global alias commands (fastest way)
chuk-llm ask_granite "What is Python?"
chuk-llm ask_claude "Explain quantum computing"
chuk-llm ask_gpt "Write a haiku about programming"
chuk-llm ask_azure "Deploy models to Azure"
chuk-llm ask_llama "What is machine learning?"

# ✨ NEW: Convenience functions for discovered models
chuk-llm ask_ollama_gemma3 "Hello world"
chuk-llm ask_ollama_mistral_small_latest "Tell me a joke"
chuk-llm ask_ollama_qwen3_latest "Explain Python"
chuk-llm stream_ollama_llama3_2 "Write a long explanation"

# General ask with provider selection
chuk-llm ask "Your question" --provider azure_openai --model gpt-4o-mini

# JSON responses for structured data
chuk-llm ask "List 3 Python libraries" --json --provider azure_openai
```

### Discovery and Provider Management

```bash
# List all available providers
chuk-llm providers

# Show models for a specific provider (includes discovered models)
chuk-llm models azure_openai
chuk-llm models ollama

# Test provider connection
chuk-llm test azure_openai
chuk-llm test ollama

# ✨ NEW: Discover new models (especially useful for Ollama)
chuk-llm discover ollama

# ✨ NEW: List all available functions (includes discovered ones)
chuk-llm functions
```

### Configuration and Diagnostics

```bash
# Show current configuration
chuk-llm config

# ✨ NEW: List all auto-generated functions (updates with discovery)
chuk-llm functions

# Show global aliases
chuk-llm aliases

# Comprehensive help
chuk-llm help
```

### CLI Options

```bash
# Verbose mode (show provider details)
chuk-llm ask_claude "Hello" --verbose
chuk-llm ask_ollama_gemma3 "Hello" --verbose  # ✨ NEW

# Quiet mode (minimal output)
chuk-llm providers --quiet

# Disable streaming
chuk-llm ask "Question" --provider azure_openai --no-stream
```

### Examples with uvx (Zero Install)

```bash
# Use without installing globally
uvx chuk-llm ask_azure "What is Azure OpenAI?"
uvx chuk-llm ask_ollama_gemma3 "Hello"  # ✨ NEW
uvx chuk-llm providers
uvx chuk-llm discover ollama  # ✨ NEW
uvx chuk-llm test azure_openai
```

### ✨ Enhanced CLI Features

- **🎯 Global Aliases**: Pre-configured shortcuts for popular models
- **⚡ Dynamic Convenience Functions**: Auto-generated functions like `ask_ollama_gemma3`
- **🌊 Real-time Streaming**: See responses as they're generated
- **🔧 Provider Testing**: Verify connections and configurations
- **🔍 Model Discovery**: Find and use new Ollama models instantly
- **📊 Rich Formatting**: Beautiful tables and output (with `[cli]` extra)
- **⚡ Fast Feedback**: Quick responses and error reporting
- **🎨 Output Control**: Verbose, quiet, and no-stream modes
- **🔄 Function Listing**: See all available auto-generated functions
- **🎯 Smart Error Messages**: Helpful suggestions when functions don't exist

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

# Provider-specific functions (including auto-discovered Ollama models)
from chuk_llm import (
    ask_openai_sync, ask_azure_openai_sync, ask_claude_sync, ask_watsonx_sync,
    ask_ollama_llama3_2_sync, ask_ollama_gemma3_sync  # ✨ Auto-discovered!
)
openai_joke = ask_openai_sync("Tell me a dad joke")
azure_explanation = ask_azure_openai_sync("Explain Azure security features")
claude_explanation = ask_claude_sync("Explain quantum computing")
local_response = ask_ollama_llama3_2_sync("Write Python code to read a CSV")
enterprise_response = ask_watsonx_sync("Generate enterprise security documentation")
gemma_response = ask_ollama_gemma3_sync("Explain machine learning")  # ✨ Auto-discovered!
```

### Async Usage

```python
import asyncio
from chuk_llm import ask, stream, conversation

async def main():
    # Basic async call
    response = await ask("Hello!")
    
    # ✨ NEW: Dynamic async functions for discovered models
    from chuk_llm import ask_ollama_gemma3, stream_ollama_qwen3
    gemma_response = await ask_ollama_gemma3("Explain AI")
    
    # Real-time streaming
    print("Streaming: ", end="", flush=True)
    async for chunk in stream_ollama_qwen3("Write a haiku about coding"):
        print(chunk, end="", flush=True)
    
    # Conversations with memory
    async with conversation(provider="azure_openai") as chat:
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

# CLI version
# chuk-llm ask_perplexity "What are the latest AI developments?"
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
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export PERPLEXITY_API_KEY="your-perplexity-key"

# Custom endpoints
export OPENAI_API_BASE="https://api.openai.com/v1"
export PERPLEXITY_API_BASE="https://api.perplexity.ai"
export OLLAMA_API_BASE="http://localhost:11434"

# Session tracking
export CHUK_LLM_DISABLE_SESSIONS="false"  # Set to "true" to disable

# Session storage
export SESSION_PROVIDER="memory"  # or "redis"
export SESSION_REDIS_URL="redis://localhost:6379/0"

# ✨ NEW: Discovery settings
export CHUK_LLM_DISCOVERY_ENABLED="true"       # Enable discovery globally
export CHUK_LLM_OLLAMA_DISCOVERY="true"        # Enable Ollama discovery
export CHUK_LLM_AUTO_DISCOVER="true"           # Enable auto-discovery
export CHUK_LLM_DISCOVERY_TIMEOUT="5"          # Discovery timeout (seconds)
export CHUK_LLM_DISCOVERY_CACHE_TIMEOUT="300"  # Cache timeout (seconds)
```

### Simple API Configuration

```python
from chuk_llm import configure, get_current_config

# Simple configuration
configure(
    provider="azure_openai",
    model="gpt-4o-mini", 
    temperature=0.7
)

# All subsequent calls use these settings
from chuk_llm import ask_sync
response = ask_sync("Tell me about AI")

# Check current configuration
config = get_current_config()
print(f"Using {config['provider']} with {config['model']}")
```

## 🛠️ Advanced Usage

### Session Analytics & Monitoring

```python
import asyncio
from chuk_llm import ask, get_session_stats, get_session_history

async def analytics_example():
    # Use the API normally
    await ask("Explain machine learning")
    await ask("What are neural networks?")
    await ask("How does backpropagation work?")
    
    # Get detailed analytics
    stats = await get_session_stats()
    print("📊 Session Analytics:")
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Estimated cost: ${stats['estimated_cost']:.6f}")
    
    # Get conversation history
    history = await get_session_history()
    print("\n📜 Conversation History:")
    for i, msg in enumerate(history[-6:]):  # Last 6 messages
        role = msg['role']
        content = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
        print(f"{i+1}. {role}: {content}")

asyncio.run(analytics_example())
```

### Multi-Provider Comparison

```python
from chuk_llm import compare_providers

# Compare responses across providers (including local Ollama models)
results = compare_providers(
    "Explain quantum computing",
    providers=["openai", "azure_openai", "anthropic", "perplexity", "groq", "ollama", "watsonx"]
)

for provider, response in results.items():
    print(f"{provider}: {response[:100]}...")
```

### Performance Monitoring

```python
import asyncio
from chuk_llm import test_all_providers

async def monitor_performance():
    # Test all providers concurrently (including discovered Ollama models)
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
        ["openai", "azure_openai", "anthropic", "groq", "ollama", "watsonx"]
    )
    
    print("\nQuality Comparison:")
    for provider, response in comparison.items():
        print(f"{provider}: {response[:100]}...")

asyncio.run(benchmark_providers())
```

## 🔍 Provider Capabilities

```python
import chuk_llm

# Discover available providers and models (including discovered ones)
chuk_llm.show_providers()

# ✨ NEW: See all auto-generated functions (updates with discovery)
chuk_llm.show_functions()

# Get comprehensive diagnostics (including session info)
chuk_llm.print_full_diagnostics()

# Test specific provider capabilities
from chuk_llm import test_connection_sync
result = test_connection_sync("azure_openai")
print(f"✅ {result['provider']}: {result['duration']:.2f}s")

# ✨ NEW: Trigger Ollama discovery and see new functions
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
new_functions = trigger_ollama_discovery_and_refresh()
print(f"🔍 Generated {len(new_functions)} new Ollama functions")

# ✨ NEW: Environment controls for discovery
from chuk_llm.api.providers import disable_discovery, enable_discovery
disable_discovery("ollama")  # Disable Ollama discovery
enable_discovery("ollama")   # Re-enable Ollama discovery
```

## 🌐 Provider Models

### OpenAI
- **GPT-4** - gpt-4o, gpt-4o-mini, gpt-4-turbo
- **GPT-3.5** - gpt-3.5-turbo

### Azure OpenAI 🏢
Azure OpenAI provides enterprise-grade access to OpenAI models with enhanced security, compliance, and enterprise features:

#### Enterprise GPT Models
- **GPT-4** - gpt-4o, gpt-4o-mini, gpt-4-turbo (with enterprise security)
- **GPT-3.5** - gpt-3.5-turbo (enterprise deployment)

#### Azure-Specific Features
- **🔒 Enterprise Security**: Private endpoints, VNet integration, data residency controls
- **📊 Compliance**: SOC 2, HIPAA, PCI DSS, ISO 27001 certified
- **🎯 Custom Deployments**: Deploy specific model versions with dedicated capacity
- **📈 Advanced Monitoring**: Detailed usage analytics and audit logs
- **🔧 Fine-tuning**: Custom model training on your enterprise data
- **🌍 Global Availability**: Multiple Azure regions with data residency

#### Model Aliases
ChukLLM provides convenient aliases for Azure OpenAI:
```python
# These automatically use your Azure deployment:
ask_azure_openai_gpt4o()     # → Your gpt-4o deployment
ask_azure_openai_gpt4_mini() # → Your gpt-4o-mini deployment
ask_azure_openai_gpt35()     # → Your gpt-3.5-turbo deployment
```

*Enterprise features: Private networking, audit logs, data residency, compliance certifications, custom deployments*

### Anthropic  
- **Claude 3.5** - claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- **Claude 3** - claude-3-opus-20240229, claude-3-sonnet-20240229

### Google Gemini
- **Gemini 2.0** - gemini-2.0-flash
- **Gemini 1.5** - gemini-1.5-pro, gemini-1.5-flash

### Groq
- **Llama 3.3** - llama-3.3-70b-versatile
- **Llama 3.1** - llama-3.1-70b-versatile, llama-3.1-8b-instant
- **Mixtral** - mixtral-8x7b-32768

### ✨ Ollama (Local Models with Dynamic Discovery)
Ollama provides local model deployment with **automatic discovery and function generation**:

#### Static Models (Configured)
- **llama3.3** - Latest Llama 3.3 model
- **qwen3** - Qwen 3 series
- **granite3.3** - IBM Granite 3.3
- **mistral** - Mistral base model
- **gemma3** - Google Gemma 3
- **phi3** - Microsoft Phi-3
- **codellama** - Code-specialized Llama

#### ✨ Dynamic Discovery Examples
When you pull new models, ChukLLM automatically discovers them:

```bash
# Pull new models in Ollama
ollama pull llama3.2
ollama pull mistral-small:latest
ollama pull qwen2.5:14b
ollama pull deepseek-coder:6.7b
ollama pull phi3:latest
```

ChukLLM automatically generates functions:
```python
# These functions are auto-generated after discovery:
ask_ollama_llama3_2_sync()
ask_ollama_mistral_small_latest_sync()
ask_ollama_qwen2_5_14b_sync()
ask_ollama_deepseek_coder_6_7b_sync()
ask_ollama_phi3_latest_sync()

# And CLI commands work immediately:
# chuk-llm ask_ollama_llama3_2 "Hello"
# chuk-llm ask_ollama_mistral_small_latest "Write code"
```

#### Discovery Features
- **🔍 Real-time Detection**: Automatically finds new models as they're pulled
- **⚡ Instant Generation**: Creates functions immediately without restart
- **🖥️ CLI Integration**: New models work instantly in CLI
- **🧠 Smart Naming**: Converts model names to valid function names
- **🔄 Live Updates**: Refreshes available models without configuration changes

### IBM watsonx & Granite 🏢
IBM's enterprise-grade AI models optimized for business applications with 131K context length:

#### Granite Models
- **ibm/granite-3-3-8b-instruct** - Latest Granite 3.3 8B instruction model (default)
- **ibm/granite-3-2b-instruct** - Efficient 2B parameter model for lightweight deployment
- **ibm/granite-vision-3-2-2b-instruct** - Vision-capable Granite for multimodal tasks

#### Meta Llama Models (via IBM)
- **meta-llama/llama-4-maverick-17b-128e-instruct-fp8** - Latest Llama 4 Maverick (17B params)
- **meta-llama/llama-4-scout-17b-16e-instruct** - Llama 4 Scout specialized model
- **meta-llama/llama-3-3-70b-instruct** - Llama 3.3 70B for complex reasoning
- **meta-llama/llama-3-405b-instruct** - Massive 405B parameter model
- **meta-llama/llama-3-2-90b-vision-instruct** - 90B vision-capable model
- **meta-llama/llama-3-2-11b-vision-instruct** - 11B vision model
- **meta-llama/llama-3-2-3b-instruct** - Efficient 3B model
- **meta-llama/llama-3-2-1b-instruct** - Ultra-lightweight 1B model

#### Mistral Models (via IBM)
- **mistralai/mistral-large-2** - Latest Mistral Large with advanced capabilities
- **mistralai/mistral-medium-2505** - Balanced performance and efficiency
- **mistralai/mistral-small-3-1-24b-instruct-2503** - Compact 24B instruction model
- **mistralai/pixtral-12b** - Vision-capable Mistral model

#### Model Aliases
ChukLLM provides convenient aliases for IBM watsonx models:
```python
# These are equivalent:
ask_watsonx_granite()  # → ibm/granite-3-3-8b-instruct
ask_watsonx_vision()   # → ibm/granite-vision-3-2-2b-instruct  
ask_watsonx_llama4()   # → meta-llama/llama-4-maverick-17b-128e-instruct-fp8
ask_watsonx_mistral()  # → mistralai/mistral-large-2
```