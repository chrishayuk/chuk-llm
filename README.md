# chuk_llm

A unified, production-ready Python library for Large Language Model (LLM) providers with real-time streaming, function calling, middleware support, automatic session tracking, dynamic model discovery, and intelligent system prompt generation.

## 🚀 QuickStart

### Installation

```bash
pip install chuk_llm chuk-ai-session-manager
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

# Provider-specific functions (auto-generated!)
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

### 🧠 Intelligent System Prompt Generation - NEW!

ChukLLM now features an advanced system prompt generator that automatically creates optimized prompts based on provider capabilities, tools, and context:

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

# Groq gets ultra-fast inference optimized prompts
from chuk_llm import ask_groq_sync
response = ask_groq_sync("Explain quantum physics", tools=tools)
# System prompt: "You are an intelligent assistant with function calling capabilities. Take advantage of ultra-fast inference..."
```

#### Custom System Prompt Templates

```python
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator

# Create custom templates
generator = SystemPromptGenerator(provider="openai", model="gpt-4o")

# Generate context-aware prompts
prompt = generator.generate_prompt(
    tools=tools,
    user_system_prompt="You are a helpful coding assistant specializing in Python.",
    json_mode=True
)

print(prompt)
# Output: Comprehensive system prompt optimized for OpenAI, with tool instructions, 
# JSON mode guidance, and custom specialization
```

### 🔍 Dynamic Model Discovery for Ollama - NEW!

ChukLLM now automatically discovers and generates functions for Ollama models in real-time:

```python
# Start Ollama with some models
# ollama pull llama3.2
# ollama pull qwen2.5:14b
# ollama pull deepseek-coder:6.7b

# ChukLLM automatically discovers them and generates functions!
from chuk_llm import (
    ask_ollama_llama3_2_sync,          # Auto-generated!
    ask_ollama_qwen2_5_14b_sync,       # Auto-generated!
    ask_ollama_deepseek_coder_6_7b_sync # Auto-generated!
)

# Use immediately without any configuration
response = ask_ollama_llama3_2_sync("Write a Python function to sort a list")
vision_response = ask_ollama_llama3_2_sync("Describe this image", image="photo.jpg")

# Trigger discovery manually to refresh available models
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
new_functions = trigger_ollama_discovery_and_refresh()
print(f"Discovered {len(new_functions)} new functions!")

# Universal discovery for any provider
from chuk_llm.api.discovery import discover_models_sync, show_discovered_models_sync

# Discover Ollama models with capability inference
models = discover_models_sync("ollama")
for model in models:
    print(f"📦 {model['name']}: {', '.join(model['features'])}")

# Show detailed discovery results
show_discovered_models_sync("ollama")
# Output:
# 🔍 Discovered 8 Ollama Models
# ========================
# 📁 Llama Models (3):
#   • llama3.2:latest
#     Size: 2.0GB | Context: 128000 | Features: text, streaming, tools
#   • llama3.2:1b  
#     Size: 1.3GB | Context: 128000 | Features: text, streaming
```

#### Discovery Configuration

```python
# Configure discovery behavior in chuk_llm.yaml
# ollama:
#   dynamic_discovery:
#     enabled: true
#     cache_timeout: 300
#     auto_update_on_startup: true
#     inference_config:
#       default_features: ["text", "streaming"]
#       family_rules:
#         llama:
#           features: ["text", "streaming", "tools", "reasoning"]
#           context_rules:
#             "llama.*3\.[23]": 128000
#         qwen:
#           features: ["text", "streaming", "tools", "reasoning"]

# Manual discovery with custom configuration
from chuk_llm.api.discovery import update_provider_configuration_sync

result = update_provider_configuration_sync(
    "ollama",
    inference_config={
        "family_rules": {
            "llama": {
                "features": ["text", "streaming", "tools", "vision", "reasoning"],
                "patterns": ["llama.*"]
            }
        }
    }
)

print(f"✅ Updated {result['text_models']} models")
```

### 🎭 Enhanced Conversations - NEW!

ChukLLM now supports advanced conversation features for building sophisticated dialogue systems:

#### 1. Conversation Branching
Explore different conversation paths without affecting the main thread:

```python
async with conversation() as chat:
    await chat.say("Let's plan a vacation")
    
    # Branch to explore Japan
    async with chat.branch() as japan_branch:
        await japan_branch.say("Tell me about visiting Japan")
        await japan_branch.say("What about costs?")
        # This conversation stays isolated
    
    # Branch to explore Italy
    async with chat.branch() as italy_branch:
        await italy_branch.say("Tell me about visiting Italy")
        await italy_branch.say("Best time to visit?")
    
    # Main conversation doesn't know about branches
    await chat.say("I've decided on Japan!")  # AI won't know why!
```

#### 2. Conversation Persistence
Save and resume conversations across sessions:

```python
# Start a conversation
async with conversation() as chat:
    await chat.say("I'm learning Python")
    await chat.say("I know JavaScript already")
    
    # Save for later
    conversation_id = await chat.save()
    print(f"Saved as: {conversation_id}")

# Resume days later
async with conversation(resume_from=conversation_id) as chat:
    response = await chat.say("What should I learn next?")
    # AI remembers your background!
```

#### 3. Multi-Modal Conversations
Send images along with text (requires vision-capable models):

```python
async with conversation(model="gpt-4o") as chat:
    # Send an image
    await chat.say("What's in this diagram?", image="architecture.png")
    
    # Continue with context
    await chat.say("Can you explain the database layer?")
    # AI remembers the image context!
```

#### 4. Conversation Utilities
Built-in tools for analysis and summarization:

```python
async with conversation() as chat:
    # Have a detailed discussion
    await chat.say("Let's discuss machine learning")
    await chat.say("Explain neural networks")
    await chat.say("What about transformers?")
    await chat.say("How do they relate to LLMs?")
    
    # Get a summary
    summary = await chat.summarize(max_length=200)
    print(f"Summary: {summary}")
    
    # Extract key points
    points = await chat.extract_key_points()
    for point in points:
        print(f"• {point}")
    
    # Get statistics
    stats = await chat.get_session_stats()
    print(f"Cost so far: ${stats['estimated_cost']:.6f}")
```

#### 5. Stateless Context
Add context to one-off questions without maintaining conversation state:

```python
# Quick context for a single question
response = await ask(
    "What's the capital?",
    context="We're discussing France and its major cities"
)

# Provide conversation history for context
response = await ask(
    "What should I do next?",
    previous_messages=[
        {"role": "user", "content": "I'm making a cake"},
        {"role": "assistant", "content": "Great! First, preheat your oven to 350°F..."},
        {"role": "user", "content": "Done, and I've mixed the dry ingredients"}
    ]
)
```

### 103+ Auto-Generated Functions

ChukLLM automatically creates functions for every provider and model, including dynamically discovered ones:

```python
# Base provider functions
from chuk_llm import ask_openai, ask_anthropic, ask_groq, ask_ollama

# Model-specific functions (auto-generated from config + discovery)
from chuk_llm import ask_openai_gpt4o, ask_claude_sonnet, ask_ollama_llama3_2

# Convenient aliases
from chuk_llm import ask_gpt4o, ask_claude

# All with sync, async, and streaming variants!
from chuk_llm import (
    ask_openai_sync,          # Synchronous
    stream_anthropic,         # Async streaming  
    ask_groq_sync,           # Sync version
    ask_ollama_llama3_2_sync # Auto-discovered local model
)

# Real-time function generation for new models
# When you pull a new Ollama model:
# ollama pull mistral:7b-instruct

# ChukLLM automatically generates:
from chuk_llm import ask_ollama_mistral_7b_instruct_sync  # Available immediately!
```

### Discovery & Utilities

```python
import chuk_llm

# See all available providers and models (including discovered)
chuk_llm.show_providers()

# See all 103+ auto-generated functions (updates dynamically)
chuk_llm.show_functions()

# Comprehensive diagnostics
chuk_llm.print_diagnostics()

# Test connections
from chuk_llm import test_connection_sync
result = test_connection_sync("anthropic")
print(f"✅ {result['provider']}: {result['duration']:.2f}s")

# Trigger Ollama model discovery
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
new_functions = trigger_ollama_discovery_and_refresh()
print(f"🔍 Discovered {len(new_functions)} new Ollama functions")
```

### 🎯 Automatic Session Tracking (NEW!)

ChukLLM now includes automatic session tracking powered by `chuk-ai-session-manager`. Every API call is automatically tracked for complete observability - no code changes needed!

```python
from chuk_llm import ask, conversation, get_session_stats, get_session_history

# Example 1: ask() is stateless but still tracked
await ask("What's the capital of France?")  # Response: "The capital of France is Paris"
await ask("What's 2+2?")                    # Response: "2+2 equals 4"

# These are independent calls, but both are tracked!
stats = await get_session_stats()
print(f"📊 Tracked {stats['total_messages']} messages")
print(f"💰 Total cost: ${stats['estimated_cost']:.6f}")

# Example 2: For contextual conversations, use conversation()
async with conversation() as conv:
    await conv.say("What's the capital of France?")
    await conv.say("Tell me more about it")  # This WILL understand "it" = Paris
    
    # Session tracking happens automatically here too!
    if conv.has_session:
        stats = await conv.get_session_stats()
        print(f"💬 Conversation cost: ${stats['estimated_cost']:.6f}")
```

**Key Points:**
- ✅ `ask()` - Stateless calls, each independent, but ALL tracked
- ✅ `conversation()` - Stateful dialogue, maintains context, also tracked
- ✅ Both contribute to the same session for analytics

**Automatic Features:**
- ✅ **Zero Configuration** - Works out of the box
- 📊 **Token Tracking** - Monitor usage in real-time
- 💰 **Cost Estimation** - Track spending across providers
- 🔄 **Infinite Context** - Automatic conversation segmentation
- 📝 **Full History** - Complete audit trail
- 🛠️ **Tool Tracking** - Function calls logged automatically

**Session Analytics:**
```python
# After using ask() or conversation()
stats = await get_session_stats()
print(f"📊 Session {stats['session_id'][:8]}...")
print(f"   Messages: {stats['total_messages']}")
print(f"   Tokens: {stats['total_tokens']}")
print(f"   Cost: ${stats['estimated_cost']:.6f}")

# View the history (you'll see all calls)
history = await get_session_history()
for msg in history:
    print(f"{msg['role']}: {msg['content'][:50]}...")
```

**Session Management:**
```python
from chuk_llm import (
    get_current_session_id,  # Get current session ID
    reset_session,           # Start a new session
    disable_sessions,        # Turn off tracking
    enable_sessions          # Turn on tracking
)

# Check current session
session_id = get_current_session_id()
print(f"Current session: {session_id}")

# Start fresh
reset_session()

# Disable if needed (or set CHUK_LLM_DISABLE_SESSIONS=true)
disable_sessions()
```

### 🌳 Hierarchical Sessions - Branched Conversations

ChukLLM supports hierarchical sessions for complex conversation flows, A/B testing, and parallel exploration:

```python
from chuk_llm import conversation
from chuk_ai_session_manager import SessionManager

# Start main conversation
async with conversation() as main_conv:
    await main_conv.say("I need help planning a vacation")
    main_session_id = main_conv.session_id
    
    # Branch 1: Explore Japan
    japan_session = SessionManager(parent_session_id=main_session_id)
    async with conversation(session_id=japan_session.session_id) as japan_conv:
        await japan_conv.say("Tell me about visiting Japan")
        japan_cost = (await japan_conv.get_session_stats())['estimated_cost']
    
    # Branch 2: Explore Italy  
    italy_session = SessionManager(parent_session_id=main_session_id)
    async with conversation(session_id=italy_session.session_id) as italy_conv:
        await italy_conv.say("Tell me about visiting Italy")
        italy_cost = (await italy_conv.get_session_stats())['estimated_cost']
    
    # Continue main thread with decision
    await main_conv.say("I'll choose Japan! Plan a 10-day trip.")

# Session hierarchy:
# └── Main conversation (vacation planning)
#     ├── Branch 1 (Japan exploration)
#     └── Branch 2 (Italy exploration)
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

✅ **103+ Auto-Generated Functions** - Every provider & model gets functions  
✅ **3-7x Performance Boost** - Concurrent requests vs sequential  
✅ **Real-time Streaming** - Token-by-token output as it's generated  
✅ **Memory Management** - Stateful conversations with context  
✅ **Enhanced Conversations** - Branching, persistence, multi-modal support  
✅ **Automatic Session Tracking** - Zero-config usage analytics & cost monitoring  
✅ **Dynamic Model Discovery** - Automatically detect and generate functions for new models  
✅ **Intelligent System Prompts** - Provider-optimized prompts with tool integration  
✅ **Production Ready** - Error handling, retries, connection pooling  
✅ **Developer Friendly** - Simple sync for scripts, powerful async for apps  

## 🚀 Features

### Multi-Provider Support
- **OpenAI** - GPT-4, GPT-3.5 with full API support
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Haiku
- **Google Gemini** - Gemini 2.0 Flash, Gemini 1.5 Pro  
- **Groq** - Lightning-fast inference with Llama models
- **Perplexity** - Real-time web search with Sonar models
- **Ollama** - Local model deployment with dynamic discovery

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
- **🔍 Dynamic Model Discovery** - Automatic detection of new models (Ollama, HuggingFace, etc.)
- **Vision Support** - Image analysis across compatible providers
- **JSON Mode** - Structured output generation
- **Real-time Web Search** - Live information retrieval with citations
- **Parallel Function Calls** - Execute multiple tools simultaneously
- **Connection Pooling** - Efficient HTTP connection management
- **Configuration Management** - Environment-based provider setup
- **Capability Detection** - Automatic feature detection per provider
- **Infinite Context** - Automatic conversation segmentation for long chats
- **Conversation History** - Full audit trail of all interactions

### Enhanced Conversation Features (NEW!)
- **🌿 Conversation Branching** - Explore multiple paths without affecting main thread
- **💾 Conversation Persistence** - Save and resume conversations across sessions
- **🖼️ Multi-Modal Support** - Send images with text in conversations
- **📊 Built-in Utilities** - Summarize, extract key points, get statistics
- **🎯 Stateless Context** - Add context to one-off questions without state

### Dynamic Discovery Features (NEW!)
- **🔍 Real-time Model Detection** - Automatically discover new Ollama models
- **⚡ Function Generation** - Create provider functions on-demand
- **🧠 Capability Inference** - Automatically detect model features and limitations
- **📦 Universal Discovery** - Support for multiple discovery sources (Ollama, HuggingFace, etc.)
- **🔄 Cache Management** - Intelligent caching with automatic refresh
- **📊 Discovery Analytics** - Statistics and insights about discovered models

## 📦 Installation

```bash
pip install chuk_llm
```

### With Session Tracking (Recommended)
```bash
pip install chuk_llm chuk-ai-session-manager
```

### Optional Dependencies
```bash
# For all providers with session tracking
pip install chuk_llm[all] chuk-ai-session-manager

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

# Provider-specific functions (including auto-discovered Ollama models)
from chuk_llm import ask_openai_sync, ask_claude_sync, ask_ollama_llama3_2_sync
openai_joke = ask_openai_sync("Tell me a dad joke")
claude_explanation = ask_claude_sync("Explain quantum computing")
local_response = ask_ollama_llama3_2_sync("Write Python code to read a CSV")
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

### Dynamic Ollama Discovery

```python
# Start with a fresh Ollama installation
# ollama pull llama3.2
# ollama pull qwen2.5:7b
# ollama pull deepseek-coder

# ChukLLM automatically discovers and generates functions!
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh

# Manually trigger discovery (happens automatically too)
new_functions = trigger_ollama_discovery_and_refresh()
print(f"🔍 Generated {len(new_functions)} new functions")

# Use the auto-generated functions immediately
from chuk_llm import (
    ask_ollama_llama3_2_sync,        # Auto-generated!
    ask_ollama_qwen2_5_7b_sync,      # Auto-generated!
    ask_ollama_deepseek_coder_sync   # Auto-generated!
)

response = ask_ollama_llama3_2_sync("Explain machine learning")
code = ask_ollama_deepseek_coder_sync("Write a Python sorting function")
translation = ask_ollama_qwen2_5_7b_sync("Translate 'hello' to Chinese")

# Show all discovered models
from chuk_llm.api.discovery import show_discovered_models_sync
show_discovered_models_sync("ollama")
```

### Smart System Prompts

```python
from chuk_llm import ask_sync

# Tools example - system prompt automatically optimized for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                }
            }
        }
    }
]

# Each provider gets optimized system prompts automatically
claude_response = ask_claude_sync("What's the latest AI news?", tools=tools)
# Gets Claude-optimized system prompt with tool instructions

openai_response = ask_openai_sync("What's the latest AI news?", tools=tools) 
# Gets OpenAI-optimized system prompt with function calling guidelines

groq_response = ask_groq_sync("What's the latest AI news?", tools=tools)
# Gets Groq-optimized system prompt emphasizing fast inference
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

# Session tracking
export CHUK_LLM_DISABLE_SESSIONS="false"  # Set to "true" to disable

# Discovery settings
export CHUK_LLM_DISCOVERY_CACHE_TIMEOUT="300"  # Cache timeout in seconds
```

### Simple API Configuration

```python
from chuk_llm import configure, get_current_config

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
config = get_current_config()
print(f"Using {config['provider']} with {config['model']}")
```

### YAML Configuration

Create a `chuk_llm.yaml` file for advanced configuration:

```yaml
# chuk_llm.yaml
__global__:
  active_provider: anthropic
  default_timeout: 30

anthropic:
  api_key_env: ANTHROPIC_API_KEY
  default_model: claude-3-5-sonnet-20241022
  models:
    - claude-3-5-sonnet-20241022
    - claude-3-5-haiku-20241022
  features: [text, streaming, tools, vision, json_mode]

ollama:
  api_base: http://localhost:11434
  default_model: llama3.2
  features: [text, streaming, system_messages]
  # Enable dynamic discovery
  dynamic_discovery:
    enabled: true
    cache_timeout: 300
    auto_update_on_startup: true
    inference_config:
      default_features: [text, streaming]
      family_rules:
        llama:
          features: [text, streaming, tools, reasoning]
          patterns: ["llama.*"]
          context_rules:
            "llama.*3\.[23]": 128000
        qwen:
          features: [text, streaming, tools, reasoning]
          patterns: ["qwen.*"]
          context_length: 32768
```

## 🛠️ Advanced Usage

### Enhanced Conversation Examples

```python
import asyncio
from chuk_llm import conversation

async def advanced_conversation_demo():
    # 1. Branching conversations
    async with conversation() as chat:
        await chat.say("Let's discuss AI")
        
        # Explore a tangent
        async with chat.branch() as tangent:
            await tangent.say("What about AI ethics?")
            ethics_response = await tangent.say("Tell me more")
        
        # Main thread continues unaware
        await chat.say("What are the main applications?")
    
    # 2. Persistent conversations
    conversation_id = None
    async with conversation() as chat:
        await chat.say("I'm building a web app")
        await chat.say("Using Python and React")
        conversation_id = await chat.save()
    
    # Resume later
    async with conversation(resume_from=conversation_id) as chat:
        response = await chat.say("What database should I use?")
        # AI remembers your tech stack!
    
    # 3. Multi-modal with utilities
    async with conversation(model="gpt-4o") as chat:
        await chat.say("Analyze this UI", image="screenshot.png")
        await chat.say("How can I improve the layout?")
        
        # Get insights
        summary = await chat.summarize()
        key_points = await chat.extract_key_points()
        
        print(f"Summary: {summary}")
        for point in key_points:
            print(f"• {point}")

asyncio.run(advanced_conversation_demo())
```

### Dynamic Discovery Examples

```python
import asyncio
from chuk_llm.api.discovery import discover_models, update_provider_configuration

async def discovery_examples():
    # 1. Discover all available Ollama models
    models = await discover_models("ollama")
    print(f"Found {len(models)} Ollama models:")
    for model in models[:5]:  # Show first 5
        print(f"  📦 {model['name']}: {', '.join(model['features'])}")
    
    # 2. Update provider configuration with discovered models
    result = await update_provider_configuration("ollama")
    if result['success']:
        print(f"✅ Updated configuration with {result['text_models']} models")
    
    # 3. Custom discovery configuration
    custom_result = await update_provider_configuration(
        "ollama",
        inference_config={
            "family_rules": {
                "reasoning": {
                    "features": ["text", "streaming", "reasoning"],
                    "patterns": [".*reasoning.*", ".*qwq.*", ".*o1.*"]
                }
            }
        }
    )

asyncio.run(discovery_examples())

# Sync versions available too
from chuk_llm.api.discovery import discover_models_sync, show_discovered_models_sync

models = discover_models_sync("ollama")
show_discovered_models_sync("ollama")
```

### System Prompt Customization

```python
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator

# Create a custom generator
generator = SystemPromptGenerator(provider="openai", model="gpt-4o")

# Generate different types of prompts
tools = [{"type": "function", "function": {"name": "calculate", "description": "Do math"}}]

# Default prompt
basic_prompt = generator.generate_prompt()

# With tools
tools_prompt = generator.generate_prompt(tools=tools)

# With custom system message
custom_prompt = generator.generate_prompt(
    tools=tools,
    user_system_prompt="You are a math tutor specializing in calculus.",
    json_mode=True
)

# Minimal template
minimal_prompt = generator.generate_prompt(
    template_name="minimal",
    user_system_prompt="Be concise."
)

print("Generated prompts:", custom_prompt)
```

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
    print(f"   Average tokens/message: {stats['average_tokens_per_message']}")
    
    # Get conversation history
    history = await get_session_history()
    print("\n📜 Conversation History:")
    for i, msg in enumerate(history[-6:]):  # Last 6 messages
        role = msg['role']
        content = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
        print(f"{i+1}. {role}: {content}")

asyncio.run(analytics_example())
```

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

# Compare responses across providers (including local Ollama models)
results = compare_providers(
    "Explain quantum computing",
    providers=["openai", "anthropic", "perplexity", "groq", "ollama"]
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
        ["openai", "anthropic", "groq", "ollama"]
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

# See all auto-generated functions (updates with discovery)
chuk_llm.show_functions()

# Get comprehensive diagnostics
chuk_llm.print_diagnostics()

# Test specific provider capabilities
from chuk_llm import test_connection_sync
result = test_connection_sync("anthropic")
print(f"✅ {result['provider']}: {result['duration']:.2f}s")

# Trigger Ollama discovery and see new functions
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
new_functions = trigger_ollama_discovery_and_refresh()
print(f"🔍 Generated {len(new_functions)} new Ollama functions")
```

## 🌐 Provider Models

### OpenAI
- **GPT-4** - gpt-4o, gpt-4o-mini, gpt-4-turbo
- **GPT-3.5** - gpt-3.5-turbo

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

### Ollama 🔍
- **Dynamic Discovery** - Any compatible GGUF model automatically detected
- **Popular Models** - Llama 3.2, Qwen 2.5, DeepSeek-Coder, Mistral, Phi, Code Llama, etc.
- **Automatic Functions** - ChukLLM generates functions for all discovered models

Example discovered Ollama models:
```
📦 llama3.2:latest - Features: text, streaming, tools, reasoning
📦 qwen2.5:14b - Features: text, streaming, tools, reasoning, vision
📦 deepseek-coder:6.7b - Features: text, streaming, tools
📦 mistral:7b-instruct - Features: text, streaming, tools
📦 phi3:mini - Features: text, streaming, reasoning
```

## 🏗️ Architecture

### Core Components

- **`BaseLLMClient`** - Abstract interface for all providers
- **`MiddlewareStack`** - Request/response processing pipeline
- **`UnifiedConfigManager`** - Configuration management with discovery support
- **`ConnectionPool`** - HTTP connection optimization
- **`SystemPromptGenerator`** - Dynamic, provider-optimized prompt generation
- **`SessionManager`** - Automatic conversation tracking (via chuk-ai-session-manager)
- **`ConversationContext`** - Advanced conversation state management
- **`UniversalModelDiscoveryManager`** - Dynamic model detection and capability inference

### Provider Implementations

Each provider implements the `BaseLLMClient` interface with:
- Standardized message format (ChatML)
- Real-time streaming support
- Function calling normalization
- Error handling and retries
- Automatic session tracking
- Provider-optimized system prompts

### Discovery System

```python
# Discovery architecture example
from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
from chuk_llm.llm.discovery.providers import DiscovererFactory

# Create a discoverer for any provider
discoverer = DiscovererFactory.create_discoverer("ollama", api_base="http://localhost:11434")

# Universal manager handles capability inference
manager = UniversalModelDiscoveryManager("ollama", discoverer)

# Discover and infer capabilities
models = await manager.discover_models()
for model in models:
    print(f"{model.name}: {[f.value for f in model.capabilities]}")
```

### System Prompt Generation

```python
# System prompt generation example
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator

# Provider-aware generation
generator = SystemPromptGenerator(provider="anthropic", model="claude-3-5-sonnet")

# Automatic optimization based on provider and context
prompt = generator.generate_prompt(
    tools=tools,
    json_mode=True,
    user_system_prompt="You are a helpful coding assistant."
)
# Returns Claude-optimized prompt with tool instructions and JSON mode guidance
```

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
    
    # Test Ollama discovery
    from chuk_llm.api.discovery import discover_models
    models = await discover_models("ollama")
    print(f"Discovered {len(models)} Ollama models")

asyncio.run(run_diagnostics())
```

## 📈 Performance

### Streaming Performance
- **Zero-buffering streaming** - Chunks delivered in real-time
- **Parallel requests** - Multiple concurrent streams
- **Connection pooling** - Reduced latency
- **Dynamic discovery** - Real-time model availability

### Benchmarks
```
Provider Comparison (avg response time):
├── Groq: 0.8s (ultra-fast inference)
├── Perplexity: 1.0s (real-time search + generation)
├── OpenAI: 1.2s (balanced performance)
├── Anthropic: 1.5s (high quality)
├── Gemini: 1.8s (multimodal)
├── Ollama: 2.5s (local processing, varies by model)
└── Discovery: <0.1s (cached model detection)
```

### Real-time Web Search Performance
Perplexity's Sonar models deliver blazing fast search results at 1200 tokens per second, nearly 10x faster than comparable models like Gemini 2.0 Flash.

### Dynamic Discovery Performance
- **Ollama Model Detection**: ~100ms (cached), ~500ms (fresh API call)
- **Function Generation**: ~50ms per model
- **Capability Inference**: ~10ms per model
- **Cache Refresh**: 5 minutes default, configurable

## 🔒 Security & Safety

- **API key management** - Environment variable support
- **Request validation** - Input sanitization
- **Error handling** - No sensitive data leakage
- **Rate limiting** - Built-in provider limit awareness
- **Tool name sanitization** - Safe function calling
- **Session data privacy** - All tracking data stays local
- **Discovery security** - Safe model detection without code execution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Adding New Providers

```python
# ChukLLM automatically detects new providers from configuration
# Just add to your chuk_llm.yaml:

# chuk_llm.yaml
newprovider:
  api_key_env: NEWPROVIDER_API_KEY
  api_base: https://api.newprovider.com
  default_model: new-model-name
  # Optional: Enable discovery
  dynamic_discovery:
    enabled: true
    discoverer_type: openai  # Use OpenAI-compatible discovery

# ChukLLM will automatically generate:
# - ask_newprovider()
# - ask_newprovider_sync() 
# - stream_newprovider()
# - ask_newprovider_new_model_name()
```

### Adding New Discoverers

```python
# Add a new discovery provider
from chuk_llm.llm.discovery.providers import DiscovererFactory
from chuk_llm.llm.discovery.engine import BaseModelDiscoverer

class CustomDiscoverer(BaseModelDiscoverer):
    async def discover_models(self):
        # Your discovery logic here
        return models

# Register it
DiscovererFactory.register_discoverer("custom", CustomDiscoverer)
```

## 📚 Documentation

- [API Reference](docs/api.md)
- [Provider Guide](docs/providers.md)
- [Discovery System](docs/discovery.md)
- [System Prompts](docs/system-prompts.md)
- [Middleware Development](docs/middleware.md)
- [Configuration Guide](docs/configuration.md)
- [Benchmarking Guide](docs/benchmarking.md)
- [Session Tracking Guide](docs/sessions.md)
- [Enhanced Conversations Guide](docs/conversations.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the ChatML format and function calling standards
- CHUK AI for session management and analytics

---

**chuk_llm** - Unified LLM interface for production applications with intelligent system prompts, dynamic discovery, and automatic session tracking