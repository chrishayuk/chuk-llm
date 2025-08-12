# chuk_llm

A unified, production-ready Python library for Large Language Model (LLM) providers with real-time streaming, function calling, middleware support, automatic session tracking, dynamic model discovery, intelligent system prompt generation, and a powerful CLI.

## 🌟 Why ChukLLM?

✅ **🛠️ Advanced Tool Streaming** - Real-time tool calls with incremental JSON parsing  
✅ **200+ Auto-Generated Functions** - Every provider & model + discovered models  
✅ **🚀 GPT-5 & Reasoning Models** - Full support for GPT-5, O1, O3+ series, and GPT-OSS  
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
# Core functionality with session tracking (memory storage)
pip install chuk_llm
# OR with modern package manager
uv add chuk_llm

# With Redis for persistent sessions
pip install chuk_llm[redis]
uv add chuk_llm[redis]

# With enhanced CLI experience
pip install chuk_llm[cli]
uv add chuk_llm[cli]

# Full installation
pip install chuk_llm[all]
uv add chuk_llm[all]
```

### 30-Second Demo

```bash
# Zero installation required - try it instantly with uv!
uvx chuk-llm stream_ollama_gpt_oss "What is Python?"
```

**Live Streaming Output** (see the AI thinking in real-time!):
```
The user asks: "What is Python?" Provide a clear, accurate, concise response. It's a general question. Likely want to describe Python programming language: its nature, usage, etc. Should be succinct. Could mention its design, general-purpose, high-level, interpreted, dynamic, etc. Use clear language. Probably mention its creators, history, uses, features. Let's produce a short answer.

Python is a high‑level, general‑purpose programming language known for its readability and simplicity.  
* **Designed by** Guido van Rossum (first released in 1991).  
* **Interpreted, dynamically typed** – you run code directly without compiling.  
* **Extensible** – can call C/C++ libraries and embed in other applications.  
* **Wide ecosystem** – thousands of packages on PyPI for data science, web development, automation, AI, etc.  
* **Community‑driven** – strong support, regular updates (currently Python 3.x).  
In short, Python lets developers write clear, concise code that can tackle everything from quick scripts to large, complex systems.
```

**🧠 Notice**: You can see the AI's thinking process first, then the polished answer - this is the power of reasoning models!

Or in Python:
```python
from chuk_llm import quick_question

# Ultra-simple one-liner
answer = quick_question("What is 2+2?")
print(answer)  # "2 + 2 equals 4."
```

### Simple API - Perfect for Scripts & Prototypes

```python
from chuk_llm import ask_sync, configure

# Provider-specific functions (auto-generated!)
from chuk_llm import ask_openai_sync, ask_azure_openai_sync, ask_claude_sync, ask_groq_sync

# 🚀 NEW: GPT-5 models with unified reasoning architecture
gpt5_response = ask_openai_sync("Explain quantum computing", model="gpt-5")
gpt5_mini_response = ask_openai_sync("Quick summary of AI", model="gpt-5-mini")

# 🧠 NEW: Claude 4 family with enhanced reasoning
claude4_response = ask_claude_sync("Complex analysis task", model="claude-4-sonnet")
claude41_response = ask_claude_sync("Advanced reasoning problem", model="claude-4-1-opus")

openai_response = ask_openai_sync("Tell me a joke")
azure_response = ask_azure_openai_sync("Explain quantum computing")
claude_response = ask_claude_sync("Write a Python function") 
groq_response = ask_groq_sync("What's the weather like?")

# ✨ NEW: Dynamic convenience functions for discovered models (including GPT-OSS)
from chuk_llm import ask_ollama_llama3_2_sync, ask_ollama_gpt_oss_sync
local_response = ask_ollama_llama3_2_sync("Write Python code")
reasoning_response = ask_ollama_gpt_oss_sync("Think through this problem step by step")

# Configure once, use everywhere
configure(provider="openai", model="gpt-5", temperature=0.7)  # GPT-5 ready!
response = ask_sync("Write a creative story opening")

# Compare multiple providers including GPT-5
from chuk_llm import compare_providers
results = compare_providers("What is AI?", ["openai", "azure_openai", "anthropic"])
for provider, response in results.items():
    print(f"{provider}: {response}")
```

### API Keys Setup

```bash
export OPENAI_API_KEY="your-openai-key"
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
export GEMINI_API_KEY="your-google-key"
export PERPLEXITY_API_KEY="your-perplexity-key"
# Add other provider keys as needed
```

### 🖥️ Command Line Interface (CLI)

ChukLLM includes a powerful CLI for quick AI interactions from your terminal:

```bash
# Quick questions using global aliases
chuk-llm ask_granite "What is Python?"
chuk-llm ask_claude "Explain quantum computing"
chuk-llm ask_gpt "Write a haiku about code"
chuk-llm ask_azure "Deploy models to Azure"

# 🚀 NEW: GPT-5 models via CLI
chuk-llm ask "Solve this complex problem" --provider openai --model gpt-5
chuk-llm ask "Quick answer" --provider openai --model gpt-5-mini

# 🧠 NEW: Claude 4 models via CLI
chuk-llm ask "Complex reasoning task" --provider anthropic --model claude-4-sonnet
chuk-llm ask "Advanced analysis" --provider anthropic --model claude-4-1-opus

# ✨ NEW: Convenience functions for discovered models (including reasoning models)
chuk-llm ask_ollama_gpt_oss "Think through this step by step"
chuk-llm ask_ollama_mistral_small_latest "Tell me a joke"
chuk-llm stream_ollama_llama3_2 "Write a long explanation"

# General ask command with provider selection
chuk-llm ask "What is machine learning?" --provider openai --model gpt-5

# JSON responses for structured output
chuk-llm ask "List 3 Python libraries" --json --provider openai --model gpt-5

# Provider and model management
chuk-llm providers              # Show all available providers
chuk-llm models openai          # Show models for OpenAI (includes GPT-5)
chuk-llm test openai            # Test OpenAI connection
chuk-llm discover ollama        # Discover new Ollama models ✨ NEW

# Configuration and diagnostics
chuk-llm config                 # Show current configuration
chuk-llm functions              # List all auto-generated functions ✨ NEW
chuk-llm help                   # Comprehensive help

# Use with uv for zero-install usage (modern package manager)
uvx chuk-llm ask "What is Azure OpenAI?" --provider azure_openai
uvx chuk-llm ask_ollama_gpt_oss "Reasoning problem"  # ✨ NEW
uvx chuk-llm ask "Test GPT-5" --provider openai --model gpt-5  # 🚀 NEW
```

#### CLI Features

- **🎯 Global Aliases**: Quick commands like `ask_granite`, `ask_claude`, `ask_gpt`, `ask_azure`
- **🚀 GPT-5 Support**: Full CLI support for GPT-5 family models with reasoning capabilities
- **✨ Dynamic Convenience Functions**: Auto-generated functions like `ask_ollama_gpt_oss`, `ask_ollama_mistral_small_latest`
- **🛠️ Real-time Tool Streaming**: See function calls and responses as they're generated
- **🌊 Content Streaming**: See text responses as they're generated token by token
- **🔧 Provider Management**: Test, discover, and configure providers
- **📊 Rich Output**: Beautiful tables and formatting (with `[cli]` extra)
- **🔍 Discovery Integration**: Find and use new Ollama models instantly (including GPT-OSS)
- **⚡ Fast Feedback**: Immediate responses with connection testing
- **🎨 Quiet/Verbose Modes**: Control output detail with `--quiet` or `--verbose`

### Async API - Production Performance (3-7x faster!)

```python
import asyncio
from chuk_llm import ask, stream, conversation

async def main():
    # Basic async call
    response = await ask("Hello!")
    
    # 🚀 NEW: GPT-5 family with reasoning capabilities
    from chuk_llm import ask_openai
    gpt5_response = await ask_openai("Complex reasoning task", model="gpt-5")
    gpt5_mini_response = await ask_openai("Quick question", model="gpt-5-mini")
    
# 🧠 NEW: Claude 4 family with enhanced reasoning
    from chuk_llm import ask_claude
    claude4_response = await ask_claude("Complex reasoning task", model="claude-4-sonnet")
    claude41_response = await ask_claude("Advanced analysis", model="claude-4-1-opus")
    
    # Provider-specific async functions
    from chuk_llm import ask_azure_openai, ask_claude, ask_groq
    
    azure_response = await ask_azure_openai("Explain quantum computing")
    claude_response = await ask_claude("Write a Python function")
    groq_response = await ask_groq("What's the weather like?")
    
    # ✨ NEW: Dynamic async functions for discovered models (including reasoning models)
    from chuk_llm import ask_ollama_llama3_2, ask_ollama_gpt_oss, stream_ollama_qwen3
    local_response = await ask_ollama_llama3_2("Write Python code")
    reasoning_response = await ask_ollama_gpt_oss("Think through this problem")
    
    # Real-time streaming (token by token) - works with GPT-5 and reasoning models
    print("Streaming GPT-OSS thinking: ", end="", flush=True)
    async for chunk in stream_ollama_qwen3("Write a haiku about coding"):
        print(chunk, end="", flush=True)
    
    # 🛠️ NEW: Stream tool calls in real-time
    print("\n🛠️ Streaming with tools:")
    tools = [{"type": "function", "function": {"name": "calculate", ...}}]
    async for chunk in stream("Calculate compound interest", tools=tools):
        if chunk.get("tool_calls"):
            for tc in chunk["tool_calls"]:
                print(f"🔧 {tc['function']['name']}({tc['function']['arguments']})")
        if chunk.get("response"):
            print(chunk["response"], end="", flush=True)
    
    # Conversations with memory - GPT-5 compatible
    async with conversation(provider="openai", model="gpt-5") as chat:
        await chat.say("My name is Alice")
        response = await chat.say("What's my name?")
        # Remembers: "Your name is Alice"
    
    # Concurrent requests (massive speedup!) - works with all models including GPT-5
    tasks = [
        ask("Capital of France?", provider="openai", model="gpt-5"),
        ask("What is 2+2?", provider="openai", model="gpt-5-mini"), 
        ask("Name a color", provider="claude")
    ]
    responses = await asyncio.gather(*tasks)
    # 3-7x faster than sequential!

asyncio.run(main())
```

## 🌟 Core Features

### Multi-Provider Support

ChukLLM supports **9 major LLM providers** with unified APIs:

| Provider | Models | Special Features |
|----------|---------|------------------|
| **OpenAI** | 🚀 GPT-5, GPT-5-mini, GPT-4o, GPT-3.5-turbo | Reasoning models, function calling, vision, JSON mode |
| **Azure OpenAI** 🏢 | 🚀 Enterprise GPT-5, GPT-4 models | Private endpoints, compliance, audit logs |
| **Anthropic** 🧠 | Claude 4.1 Opus, Claude 4 Sonnet, Claude 3.5 Sonnet | Advanced reasoning, long context, strong analysis |
| **Google Gemini** | Gemini 2.0 Flash, Gemini 1.5 Pro | Multimodal, fast inference |
| **Groq** ⚡ | Llama models | Ultra-fast inference (500+ tokens/sec) |
| **Perplexity** 🌐 | Sonar models | Real-time web search with citations |
| **Ollama** 🏠 | 🧠 GPT-OSS, Local models + discovery | Privacy, reasoning models, offline usage, custom models |
| **IBM watsonx** 🏢 | Granite, Llama 4 | Enterprise compliance, 131K context |
| **Mistral AI** 🇪🇺 | Mistral Large, Medium | European, efficient models |

### 🛠️ Advanced Tool Streaming - BREAKTHROUGH FEATURE!

ChukLLM implements cutting-edge **real-time tool call streaming** that solves one of the hardest problems in LLM streaming: how to stream function calls as they're being generated.

#### The Challenge
Traditional streaming only handles text content, but modern LLMs also generate tool/function calls with complex JSON parameters. Most libraries force you to wait for the entire response before seeing tool calls.

#### ChukLLM's Solution
- **🔄 Incremental JSON Parsing**: Streams tool calls as JSON arguments are built up token by token
- **⚡ Immediate Tool Detection**: Shows function names as soon as they're available
- **🧠 Smart Deduplication**: Prevents duplicate tool calls during streaming
- **🔧 Universal Compatibility**: Works across all providers (OpenAI, Anthropic, Ollama, etc.)
- **💫 Reasoning Model Support**: Special handling for models like GPT-5 and Claude 4 that interleave thinking and tool calls

#### Live Tool Streaming Examples

```python
import asyncio
from chuk_llm import stream

# Stream tool calls in real-time
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
                        "expression": {"type": "string"},
                        "precision": {"type": "integer"}
                    }
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"}
                    }
                }
            }
        }
    ]
    
    print("🛠️  Streaming with tool calls:")
    async for chunk in stream(
        "Calculate 15% of 2,847 and then search for information about compound interest",
        provider="openai",
        model="gpt-5",
        tools=tools
    ):
        if chunk.get("tool_calls"):
            # Tool calls stream in real-time!
            for tool_call in chunk["tool_calls"]:
                func_name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                print(f"🔧 TOOL CALL: {func_name}({args})")
        
        if chunk.get("response"):
            # Text content also streams
            print(chunk["response"], end="", flush=True)

asyncio.run(stream_with_tools())
```

#### Streaming Output Example
```bash
🧠 Thinking: The user asks: "What is Python?" Provide a clear, accurate, concise response...

🛠️  Streaming with tool calls:
I'll help you calculate that and find information about compound interest.

🔧 TOOL CALL: calculate({"expression": "2847 * 0.15", "precision": 2})
The calculation shows that 15% of 2,847 is 427.05.

🔧 TOOL CALL: search_web({"query": "compound interest explanation", "max_results": 5})
Based on the search results, compound interest is...
```

#### Advanced Streaming Features

```python
# Stream reasoning models with tool calls and see the thinking process live
async for chunk in stream(
    "Think through this problem step by step, then use tools as needed",
    provider="anthropic",
    model="claude-4-sonnet",
    tools=tools
):
    # Get both thinking process AND tool calls in real-time
    if chunk.get("reasoning"):
        print(f"💭 Thinking: {chunk['reasoning']['thinking_content']}")
    
    if chunk.get("tool_calls"):
        print(f"🔧 Tool: {chunk['tool_calls'][0]['function']['name']}")
    
    if chunk.get("response"):
        print(chunk["response"], end="")

# Stream GPT-OSS reasoning with complete thinking visibility
async for chunk in stream_ollama_gpt_oss(
    "Analyze this data and create a report",
    tools=analysis_tools
):
    # See the model's complete reasoning process live:
    # "The user asks for data analysis. I should first understand the data structure, 
    #  then identify key patterns, calculate relevant metrics, and present findings..."
    if chunk.get("reasoning", {}).get("is_thinking"):
        print(f"🧠 {chunk['reasoning']['thinking_content']}", end="")
    elif chunk.get("tool_calls"):
        print(f"\n🛠️  Using: {chunk['tool_calls'][0]['function']['name']}")
    else:
        print(chunk["response"], end="")
```

#### Why This Matters

1. **⚡ Immediate Feedback**: See what the AI is doing as it works
2. **🔧 Tool Transparency**: Watch function calls happen in real-time  
3. **🧠 Reasoning Visibility**: For reasoning models, see the thinking process live
4. **📊 Better UX**: Users don't wait in silence for complex operations
5. **🐛 Easier Debugging**: Spot issues with tool calls immediately
6. **🔄 Interactive Workflows**: Build responsive AI applications

#### Technical Implementation

ChukLLM's streaming engine handles:
- **JSON Fragment Assembly**: Builds complete JSON from streaming tokens
- **Tool Call Deduplication**: Prevents the same tool call from being yielded multiple times
- **Provider Differences**: Handles OpenAI's delta format vs Anthropic's event format vs Ollama's chunks
- **Error Recovery**: Graceful handling of malformed JSON during streaming
- **Context Preservation**: Maintains conversation context across streaming tool calls

This breakthrough makes ChukLLM ideal for building responsive AI applications where users need to see what's happening in real-time, especially with complex multi-tool workflows.

### 🧠 Claude 4 & Advanced Reasoning - NEW!

ChukLLM provides first-class support for Anthropic's latest Claude 4 family alongside GPT-5:

#### Claude 4 Family Models
- **claude-4-1-opus** - Flagship model with the most advanced reasoning capabilities
- **claude-4-sonnet** - Balanced model with strong reasoning and fast performance
- **claude-4-haiku** - Efficient model with reasoning capabilities (coming soon)

#### Claude 4 Features
- **Enhanced Reasoning**: Advanced analytical and logical thinking capabilities
- **Long Context**: Up to 200K tokens for extensive document analysis
- **Tool Calling**: Sophisticated function calling with complex parameter handling
- **Vision Support**: Advanced image analysis and understanding
- **Code Generation**: Superior programming assistance and debugging

```python
# Claude 4 usage examples
from chuk_llm import ask_sync

# Claude 4 Sonnet for balanced reasoning
response = ask_sync("Analyze this complex business problem step by step", 
                   provider="anthropic", model="claude-4-sonnet")

# Claude 4.1 Opus for the most advanced reasoning
advanced_response = ask_sync("Provide a detailed strategic analysis", 
                           provider="anthropic", model="claude-4-1-opus")

# Claude 4 with vision capabilities
vision_response = ask_sync("Analyze this chart and provide insights", 
                          provider="anthropic", 
                          model="claude-4-sonnet",
                          messages=[{
                              "role": "user",
                              "content": [
                                  {"type": "text", "text": "Analyze this chart"},
                                  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                              ]
                          }])

# Claude 4 with complex tool calling
tools = [{"type": "function", "function": {"name": "data_analysis", ...}}]
response = ask_sync("Analyze this dataset", 
                   provider="anthropic", model="claude-4-1-opus", tools=tools)
```

### 🚀 GPT-5 & Reasoning Model Support - NEW!

ChukLLM provides first-class support for OpenAI's latest GPT-5 family and reasoning models:

#### GPT-5 Family Models
- **gpt-5** - Full-scale GPT-5 with unified reasoning architecture
- **gpt-5-mini** - Efficient GPT-5 variant for faster responses
- **gpt-5-nano** - Ultra-lightweight GPT-5 for simple tasks
- **gpt-5-chat** - Conversation-optimized GPT-5

#### Reasoning Model Support
- **O1 Series** - o1-mini (legacy support)
- **O3 Series** - o3, o3-mini with advanced reasoning
- **O4/O5 Series** - Next-generation reasoning models
- **Claude 4 Series** - claude-4-sonnet, claude-4-1-opus with enhanced reasoning
- **GPT-OSS** - Open-source reasoning model via Ollama discovery

```python
# GPT-5 usage examples
from chuk_llm import ask_sync

# GPT-5 with automatic parameter optimization
response = ask_sync("Solve this complex reasoning problem", 
                   provider="openai", model="gpt-5")

# GPT-5-mini for efficiency
quick_response = ask_sync("Quick question", 
                         provider="openai", model="gpt-5-mini")

# Claude 4 usage examples
response = ask_sync("Complex reasoning problem", 
                   provider="anthropic", model="claude-4-sonnet")

# Claude 4.1 Opus for advanced analysis
advanced_response = ask_sync("Detailed analysis task", 
                           provider="anthropic", model="claude-4-1-opus")

# GPT-OSS via Ollama (automatically discovered)
from chuk_llm import ask_ollama_gpt_oss_sync
reasoning_response = ask_ollama_gpt_oss_sync("Think through this step by step")

# All reasoning models support tool calling and streaming
tools = [{"type": "function", "function": {"name": "calculate", ...}}]
response = ask_sync("What's 15% of 250?", 
                   provider="anthropic", model="claude-4-sonnet", tools=tools)
```

#### Automatic Parameter Handling

ChukLLM automatically handles reasoning model requirements:

```python
# Automatic parameter conversion for reasoning models
response = ask_sync("Complex task", 
                   provider="openai", 
                   model="gpt-5",
                   max_tokens=1000)  # Automatically converted to max_completion_tokens

# Temperature restrictions handled automatically
response = ask_sync("Task", 
                   provider="openai", 
                   model="gpt-5",
                   temperature=0.7)  # Automatically removed (GPT-5 uses fixed temperature)
```

### 🔍 Dynamic Model Discovery - ✨ ENHANCED!

ChukLLM automatically discovers and generates functions for Ollama models in real-time, including reasoning models:

```python
# Start Ollama with reasoning models
# ollama pull gpt-oss
# ollama pull llama3.2
# ollama pull qwen2.5:14b
# ollama pull deepseek-coder:6.7b

# ChukLLM automatically discovers them and generates functions!
from chuk_llm import (
    ask_ollama_gpt_oss_sync,           # 🧠 Reasoning model - Auto-generated!
    ask_ollama_llama3_2_sync,          # Auto-generated!
    ask_ollama_qwen2_5_14b_sync,       # Auto-generated!
    ask_ollama_deepseek_coder_6_7b_sync, # Auto-generated!
)

# Use reasoning models immediately
reasoning_response = ask_ollama_gpt_oss_sync("Think through this problem step by step")

# ✨ NEW: CLI discovery with instant function availability
# chuk-llm discover ollama
# chuk-llm ask_ollama_gpt_oss "Reasoning problem"  # Works immediately!
```

#### Discovery Features

- **🔍 Real-time Detection**: Automatically finds new Ollama models including reasoning models
- **🧠 Reasoning Model Detection**: Automatically identifies models like GPT-OSS with thinking capabilities
- **⚡ Instant Functions**: Generates `ask_*` and `stream_*` functions immediately
- **🖥️ CLI Integration**: New models work instantly in CLI with convenience syntax
- **🧠 Smart Caching**: Remembers discovered models between sessions
- **📊 Environment Controls**: Fine-grained control over discovery behavior

### 🧠 Intelligent System Prompt Generation - NEW!

ChukLLM features an advanced system prompt generator that automatically creates optimized prompts:

```python
from chuk_llm import ask_sync

# GPT-5 gets optimized reasoning prompts
response = ask_sync("Complex problem", provider="openai", model="gpt-5")
# Automatically gets system prompt optimized for reasoning

# Claude 4 gets optimized reasoning prompts
response = ask_sync("Complex problem", provider="anthropic", model="claude-4-sonnet")
# Automatically gets system prompt optimized for advanced reasoning

# GPT-OSS gets thinking-focused prompts
response = ask_ollama_gpt_oss_sync("Analyze this situation")
# Automatically gets system prompt optimized for step-by-step thinking

# With function calling - works with GPT-5 and reasoning models
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

response = ask_sync("What's 15% of 250?", provider="anthropic", model="claude-4-sonnet", tools=tools)
# System prompt automatically includes function calling guidelines optimized for Claude 4
```

### 🎯 Automatic Session Tracking

ChukLLM includes automatic session tracking powered by `chuk-ai-session-manager`:

```python
from chuk_llm import ask, get_session_stats, get_session_history

# All calls are automatically tracked (including GPT-5)
await ask("What's the capital of France?", provider="openai", model="gpt-5")
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

### 🎭 Enhanced Conversations

```python
# Conversation branching with Claude 4
async with conversation(provider="anthropic", model="claude-4-sonnet") as chat:
    await chat.say("Let's plan a vacation")
    
    # Branch to explore Japan
    async with chat.branch() as japan_branch:
        await japan_branch.say("Tell me about visiting Japan")
        # This conversation stays isolated
    
    # Main conversation doesn't know about branches
    await chat.say("I've decided on Japan!")

# Conversation persistence with reasoning models
async with conversation(provider="ollama", model="gpt-oss") as chat:
    await chat.say("I'm learning Python")
    conversation_id = await chat.save()

# Resume days later with full reasoning context
async with conversation(resume_from=conversation_id) as chat:
    response = await chat.say("What should I learn next?")
    # AI remembers your background and thinks through the answer!
```

### 200+ Auto-Generated Functions ✨ EXPANDED

ChukLLM automatically creates functions for every provider and model:

```python
# Base provider functions
from chuk_llm import ask_openai, ask_azure_openai, ask_anthropic, ask_groq, ask_ollama

# 🚀 NEW: GPT-5 family functions (auto-generated from config)
from chuk_llm import ask_openai_gpt5, ask_openai_gpt5_mini, ask_azure_openai_gpt5

# Model-specific functions (auto-generated from config + discovery)
from chuk_llm import ask_openai_gpt4o, ask_azure_openai_gpt4o, ask_claude_4_sonnet, ask_claude_4_1_opus

# ✨ NEW: Dynamically discovered functions (including reasoning models)
from chuk_llm import (
    ask_ollama_gpt_oss,               # 🧠 Reasoning model - Auto-discovered!
    ask_ollama_llama3_2,              # Discovered from ollama pull llama3.2
    ask_ollama_mistral_small_latest,  # Discovered from ollama pull mistral-small:latest
    stream_ollama_gpt_oss,            # 🧠 Reasoning stream version auto-generated!
)

# All with sync, async, and streaming variants!
```

## 📦 Installation

### Installation Matrix

| Command | Session Storage | CLI Features | Use Case |
|---------|----------------|--------------|----------|
| `pip install chuk_llm` | Memory (included) | Basic | Development, scripting |
| `uv add chuk_llm` | Memory (included) | Basic | Modern development, scripting |
| `pip install chuk_llm[redis]` | Memory + Redis | Basic | Production apps |
| `uv add chuk_llm[redis]` | Memory + Redis | Basic | Modern production apps |
| `pip install chuk_llm[cli]` | Memory (included) | Enhanced | CLI tools |
| `uv add chuk_llm[cli]` | Memory (included) | Enhanced | Modern CLI tools |
| `pip install chuk_llm[all]` | Memory + Redis | Enhanced | Full features |
| `uv add chuk_llm[all]` | Memory + Redis | Enhanced | Modern full features |

### Why UV?

`uv` is the modern, fast Python package manager that's becoming the new standard:

```bash
# Install uv (if you haven't already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Use chuk_llm with uv (much faster than pip!)
uv add chuk_llm[all]

# Run without installation using uvx - perfect for trying ChukLLM
uvx chuk-llm ask "What is GPT-5?" --provider openai --model gpt-5
uvx chuk-llm ask_ollama_gpt_oss "Think through this problem"

# Real example - zero installation required:
$ uv run chuk-llm stream_ollama_gpt_oss "What is Python?"
🧠 Thinking: The user asks: "What is Python?" Provide a clear, accurate, concise response...
Python is a high‑level, general‑purpose programming language known for its readability and simplicity.  
* **Designed by** Guido van Rossum (first released in 1991).  
* **Interpreted, dynamically typed** – you run code directly without compiling...

# Create a new project with chuk_llm
uv init my-ai-project
cd my-ai-project
uv add chuk_llm[all]
```

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

## 🚀 Advanced Usage

### GPT-5 Advanced Examples

```python
# GPT-5 with complex reasoning tasks
from chuk_llm import ask_sync

# GPT-5 automatically uses optimized parameters
response = ask_sync("""
Analyze this complex business scenario:
A startup has $100k runway, 5 employees, growing 20% MoM but burning $15k/month.
They have 3 potential funding offers. What should they do?
""", provider="openai", model="gpt-5")

# GPT-5 with function calling (tools work seamlessly)
tools = [
    {
        "type": "function",
        "function": {
            "name": "financial_calculator",
            "description": "Calculate financial metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue": {"type": "number"},
                    "expenses": {"type": "number"},
                    "growth_rate": {"type": "number"}
                }
            }
        }
    }
]

response = ask_sync("Calculate runway for this startup", 
                   provider="openai", model="gpt-5", tools=tools)
```

### Reasoning Model Comparison

```python
# Compare reasoning models across providers
from chuk_llm import compare_providers

reasoning_prompt = """
Think through this step by step:
If a train leaves New York at 3 PM traveling at 80 mph toward Chicago (800 miles away),
and another train leaves Chicago at 4 PM traveling at 70 mph toward New York,
at what time will they meet?
"""

# Compare different reasoning approaches
results = compare_providers(reasoning_prompt, [
    "openai:gpt-5",           # 🚀 GPT-5 unified reasoning
    "openai:o3-mini",         # O3 series reasoning
    "ollama:gpt-oss",         # 🧠 Open-source reasoning
    "anthropic:claude-3-5-sonnet"  # Claude's reasoning
])

for provider_model, response in results.items():
    print(f"\n{provider_model}:")
    print(response[:200] + "...")
```

### Performance Demo

```python
# Sequential vs Concurrent Performance Test with GPT-5
import time
import asyncio
from chuk_llm import ask

async def performance_demo():
    questions = [
        "What is quantum computing?",
        "Explain machine learning",
        "What is the future of AI?"
    ]
    
    # Sequential (slow)
    start = time.time()
    for q in questions:
        await ask(q, provider="openai", model="gpt-5")
    sequential_time = time.time() - start
    
    # Concurrent (fast!) - works great with GPT-5
    start = time.time()
    await asyncio.gather(*[
        ask(q, provider="openai", model="gpt-5") for q in questions
    ])
    concurrent_time = time.time() - start
    
    print(f"🐌 Sequential GPT-5: {sequential_time:.2f}s")
    print(f"🚀 Concurrent GPT-5: {concurrent_time:.2f}s") 
    print(f"⚡ Speedup: {sequential_time/concurrent_time:.1f}x faster!")
    # GPT-5 typically shows 3-7x speedup with concurrent requests!

asyncio.run(performance_demo())
```

## 🌐 Provider Models

### OpenAI 🚀
- **GPT-5 Family** - gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat (unified reasoning architecture)
- **GPT-4** - gpt-4o, gpt-4o-mini, gpt-4-turbo
- **GPT-3.5** - gpt-3.5-turbo
- **Reasoning Models** - o1-mini (legacy), o3, o3-mini, o4, o5 series

### Azure OpenAI 🏢
Enterprise-grade access to OpenAI models with enhanced security and compliance:

#### Enterprise Features
- **🔒 Enterprise Security**: Private endpoints, VNet integration, data residency controls
- **📊 Compliance**: SOC 2, HIPAA, PCI DSS, ISO 27001 certified
- **🎯 Custom Deployments**: Deploy specific model versions with dedicated capacity
- **📈 Advanced Monitoring**: Detailed usage analytics and audit logs
- **🔧 Fine-tuning**: Custom model training on your enterprise data
- **🌍 Global Availability**: Multiple Azure regions with data residency

#### GPT-5 Enterprise Models
- **🚀 gpt-5** - Enterprise-grade GPT-5 with full reasoning capabilities
- **🚀 gpt-5-mini** - Efficient enterprise GPT-5 variant
- **Enterprise O3+ Models** - Advanced reasoning models for enterprise

#### Model Aliases
```python
# These automatically use your Azure deployment:
ask_azure_openai_gpt5()      # → Your gpt-5 deployment  🚀 NEW
ask_azure_openai_gpt4o()     # → Your gpt-4o deployment
ask_azure_openai_gpt4_mini() # → Your gpt-4o-mini deployment
ask_azure_openai_gpt35()     # → Your gpt-3.5-turbo deployment
```

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

#### ✨ Dynamic Discovery Examples (Including Reasoning Models)
When you pull new models, ChukLLM automatically discovers them:

```bash
# Pull reasoning and standard models in Ollama
ollama pull gpt-oss          # 🧠 Open-source reasoning model
ollama pull llama3.2
ollama pull mistral-small:latest
ollama pull qwen2.5:14b
ollama pull deepseek-coder:6.7b
```

ChukLLM automatically generates functions:
```python
# These functions are auto-generated after discovery:
ask_ollama_gpt_oss_sync()                  # 🧠 Reasoning model
ask_ollama_llama3_2_sync()
ask_ollama_mistral_small_latest_sync()
ask_ollama_qwen2_5_14b_sync()
ask_ollama_deepseek_coder_6_7b_sync()

# And CLI commands work immediately:
# chuk-llm ask_ollama_gpt_oss "Think through this step by step"  🧠
# chuk-llm ask_ollama_llama3_2 "Hello"
# chuk-llm ask_ollama_mistral_small_latest "Write code"
```

#### Reasoning Model Features in Ollama
- **🧠 GPT-OSS Support**: Full support for open-source reasoning models
- **💭 Thinking Streams**: Stream the reasoning process in real-time
- **🔍 Automatic Detection**: Recognizes reasoning models and optimizes prompts
- **📊 Context Preservation**: Maintains thinking context across conversations

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"  # Includes GPT-5 access
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-google-key"
export GROQ_API_KEY="your-groq-key"
export PERPLEXITY_API_KEY="your-perplexity-key"

# Custom endpoints
export OPENAI_API_BASE="https://api.openai.com/v1"
export PERPLEXITY_API_BASE="https://api.perplexity.ai"
export OLLAMA_API_BASE="http://localhost:11434"  # For GPT-OSS and other local models

# Session tracking
export CHUK_LLM_DISABLE_SESSIONS="false"  # Set to "true" to disable

# ✨ NEW: Discovery settings (includes reasoning model detection)
export CHUK_LLM_DISCOVERY_ENABLED="true"       # Enable discovery globally
export CHUK_LLM_OLLAMA_DISCOVERY="true"        # Enable Ollama discovery (includes GPT-OSS)
export CHUK_LLM_AUTO_DISCOVER="true"           # Enable auto-discovery
export CHUK_LLM_DISCOVERY_TIMEOUT="5"          # Discovery timeout (seconds)
export CHUK_LLM_DISCOVERY_CACHE_TIMEOUT="300"  # Cache timeout (seconds)

# 🚀 NEW: GPT-5 optimization settings
export CHUK_LLM_GPT5_OPTIMIZATION="true"       # Enable GPT-5 optimizations
export CHUK_LLM_REASONING_MODEL_DETECTION="true"  # Auto-detect reasoning models
```

### Simple API Configuration

```python
from chuk_llm import configure, get_current_config

# Simple configuration with GPT-5
configure(
    provider="openai",
    model="gpt-5",           # 🚀 GPT-5 ready!
    # temperature removed automatically for GPT-5
)

# Configuration for reasoning models
configure(
    provider="ollama",
    model="gpt-oss"          # 🧠 Reasoning model ready!
)

# All subsequent calls use these settings
from chuk_llm import ask_sync
response = ask_sync("Complex reasoning task")

# Check current configuration
config = get_current_config()
print(f"Using {config['provider']} with {config['model']}")
```

## 📊 Benchmarking

```python
import asyncio
from chuk_llm import test_all_providers, compare_providers

async def benchmark_providers():
    # Quick performance test including GPT-5
    results = await test_all_providers()
    
    print("Provider Performance (including GPT-5):")
    for provider, result in results.items():
        if result["success"]:
            print(f"✅ {provider}: {result['duration']:.2f}s")
        else:
            print(f"❌ {provider}: {result['error']}")
    
    # Quality comparison including reasoning models
    comparison = compare_providers(
        "Explain machine learning step by step",
        ["openai:gpt-5", "openai:gpt-4o", "anthropic:claude-4-sonnet", "anthropic:claude-4-1-opus", "ollama:gpt-oss"]
    )
    
    print("\nQuality Comparison (including reasoning models):")
    for provider, response in comparison.items():
        print(f"{provider}: {response[:100]}...")

asyncio.run(benchmark_providers())
```

## 🔍 Provider Capabilities

```python
import chuk_llm

# Discover available providers and models (including GPT-5 and reasoning models)
chuk_llm.show_providers()

# ✨ NEW: See all auto-generated functions (includes GPT-5 and discovered models)
chuk_llm.show_functions()

# Get comprehensive diagnostics (including session info)
chuk_llm.print_full_diagnostics()

# ✨ NEW: Trigger Ollama discovery and see new functions (including reasoning models)
from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh
new_functions = trigger_ollama_discovery_and_refresh()
print(f"🔍 Generated {len(new_functions)} new Ollama functions")

# 🚀 NEW: Test GPT-5 capabilities
from chuk_llm import test_connection_sync
result = test_connection_sync("openai", model="gpt-5")
print(f"✅ GPT-5 test: {result['duration']:.2f}s")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🚀 What's Next?

- **🛠️ Enhanced Tool Streaming**: Even more sophisticated real-time function call capabilities
- **🚀 More Claude 4 Features**: Enhanced reasoning optimizations and new model variants
- **🧠 Advanced Reasoning Models**: Support for upcoming O6, O7 series and Claude 5
- **More Providers**: Adding support for Cohere, AI21, and others
- **Advanced Discovery**: Support for HuggingFace model discovery
- **Multi-Modal**: Enhanced image and document processing
- **Enterprise Features**: Advanced audit logging and compliance tools
- **Performance**: Further optimizations for high-throughput scenarios
- **🔧 Tool Orchestration**: Advanced workflows with tool dependencies and error handling

## 📞 Support

- **Documentation**: [docs.chuk-llm.dev](https://docs.chuk-llm.dev)
- **Issues**: [GitHub Issues](https://github.com/chuk-llm/chuk-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chuk-llm/chuk-llm/discussions)
- **Email**: support@chuk-llm.dev

---

**⭐ Star us on GitHub if ChukLLM helps your AI projects!**

**🚀 Try GPT-5 and reasoning models today with ChukLLM's seamless integration!**