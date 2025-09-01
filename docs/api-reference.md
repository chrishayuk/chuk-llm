# API Reference

Complete API documentation for ChukLLM.

## Core Functions

### `ask(prompt, **kwargs)`
Async function to get a response from an LLM.

**Parameters:**
- `prompt` (str): The question or prompt
- `provider` (str, optional): LLM provider to use
- `model` (str, optional): Specific model
- `temperature` (float, optional): Creativity (0.0-2.0)
- `max_tokens` (int, optional): Maximum response length
- `system_prompt` (str, optional): System instructions
- `tools` (list, optional): Function calling tools
- `response_format` (dict, optional): JSON mode settings

**Returns:** `str` - The LLM response

**Example:**
```python
response = await ask("What is Python?", temperature=0.7)
```

### `ask_sync(prompt, **kwargs)`
Synchronous version of `ask()`.

### `stream(prompt, **kwargs)`
Async generator for streaming responses.

**Parameters:** Same as `ask()`

**Yields:** `str` - Response chunks as they arrive

**Example:**
```python
async for chunk in stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### `stream_sync_iter(prompt, **kwargs)`
Synchronous iterator for streaming (uses threading).

## Conversation API

### `conversation(**kwargs)`
Async context manager for stateful conversations.

**Parameters:**
- `provider` (str, optional): LLM provider
- `model` (str, optional): Model to use
- `system_prompt` (str, optional): System instructions
- `session_id` (str, optional): Session tracking ID
- `resume_from` (str, optional): Resume saved conversation

**Methods:**
- `ask(prompt)`: Send message and get response
- `stream(prompt)`: Stream response
- `branch()`: Create conversation branch
- `save()`: Save conversation state
- `get_history()`: Get message history
- `get_stats()`: Get conversation statistics
- `clear()`: Clear history
- `pop_last()`: Remove last exchange

**Example:**
```python
async with conversation() as chat:
    await chat.ask("Hello")
    response = await chat.ask("What did I just say?")
```

### `conversation_sync(**kwargs)`
Synchronous version of conversation API.

## Provider-Specific Functions

Auto-generated functions for each provider:

### OpenAI
- `ask_openai(prompt, **kwargs)` - Async
- `ask_openai_sync(prompt, **kwargs)` - Sync
- `stream_openai(prompt, **kwargs)` - Async stream
- Model-specific: `ask_openai_gpt_4o()`, `ask_openai_gpt_4o_mini()`

### Anthropic
- `ask_anthropic(prompt, **kwargs)`
- `ask_anthropic_sync(prompt, **kwargs)`
- `stream_anthropic(prompt, **kwargs)`
- Model-specific: `ask_anthropic_claude_3_5_sonnet()`

### Ollama (Auto-discovered)
- `ask_ollama(prompt, **kwargs)`
- `ask_ollama_sync(prompt, **kwargs)`
- Model-specific: `ask_ollama_llama3_2()`, `ask_ollama_mistral()`

## Configuration

### `configure(**kwargs)`
Set default configuration for all subsequent calls.

**Parameters:**
- `provider` (str): Default provider
- `model` (str): Default model
- `temperature` (float): Default temperature
- `max_tokens` (int): Default max tokens
- Any provider-specific settings

**Example:**
```python
configure(provider="openai", model="gpt-4o-mini", temperature=0.5)
```

### `get_current_config()`
Get current configuration settings.

**Returns:** `dict` - Current configuration

### `reset()`
Reset configuration to defaults.

## Session Management

### `get_session_stats()`
Get current session statistics.

**Returns:** `dict` with:
- `total_messages`: Message count
- `total_tokens`: Token usage
- `estimated_cost`: Cost estimate
- `duration`: Session duration

### `get_session_history()`
Get full session history.

**Returns:** `list[dict]` - All messages in session

### `reset_session()`
Start a new session.

### `disable_sessions()`
Disable session tracking.

### `enable_sessions()`
Enable session tracking.

## Utility Functions

### `quick_question(prompt)`
Ask any available provider (auto-detects).

**Example:**
```python
answer = quick_question("What is 2+2?")
```

### `compare_providers(prompt, providers)`
Compare responses from multiple providers.

**Parameters:**
- `prompt` (str): Question to ask
- `providers` (list[str]): Providers to compare

**Returns:** `dict` - Provider responses

### `test_connection(provider)`
Test if a provider is configured correctly.

**Returns:** `bool` - Connection status

### `health_check()`
Check system health and provider status.

**Returns:** `dict` - Health status

## Discovery Functions

### `discover(provider="ollama")`
Discover available models and generate functions.

**Example:**
```python
from chuk_llm import discover
models = discover("ollama")
print(f"Found {len(models)} models")
```

## CLI Functions

These functions are primarily for CLI use but can be called programmatically:

### `list_providers()`
List all available providers.

### `list_models(provider)`
List models for a specific provider.

### `list_functions()`
List all generated functions.

### `show_config()`
Display current configuration.

## Advanced Features

### Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

response = await ask_with_tools("What's the weather?", tools=tools)
```

### JSON Mode

```python
response = await ask_json("Generate a user object")
# Returns parsed JSON object
```

### Multi-modal (Images)

```python
response = await ask(
    "What's in this image?",
    image="path/to/image.jpg"
)
```

## Error Handling

All functions may raise these exceptions:

- `ProviderError`: Provider-specific errors
- `ConfigurationError`: Missing configuration
- `ValidationError`: Invalid parameters
- `NetworkError`: Connection issues
- `RateLimitError`: Rate limit exceeded

**Example:**
```python
from chuk_llm.llm.core.errors import ProviderError

try:
    response = await ask("Hello")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Type Hints

ChukLLM is fully typed. Common types:

```python
from typing import AsyncIterator, Iterator, Optional, Dict, Any, List

Message = Dict[str, Any]
Messages = List[Message]
ProviderResponse = str
StreamChunk = str
```

## Environment Variables

Recognized environment variables:

```bash
# Provider API Keys
OPENAI_API_KEY
ANTHROPIC_API_KEY
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
GOOGLE_API_KEY
GROQ_API_KEY
MISTRAL_API_KEY
PERPLEXITY_API_KEY

# Session Configuration
SESSION_PROVIDER=redis|memory
SESSION_REDIS_URL=redis://localhost:6379/0

# Discovery
CHUK_LLM_AUTO_DISCOVER=true|false
CHUK_LLM_DISCOVER_ON_STARTUP=true|false

# Debug
CHUK_LLM_DEBUG=true|false
CHUK_LLM_DEBUG_PROVIDERS=true|false
```

## Constants

```python
from chuk_llm import (
    DEFAULT_TEMPERATURE,  # 0.7
    DEFAULT_MAX_TOKENS,   # 2000
    DEFAULT_MODEL,        # "gpt-4o-mini"
    DEFAULT_PROVIDER,     # "openai"
)
```

## Version Information

```python
from chuk_llm import __version__
print(f"ChukLLM version: {__version__}")
```