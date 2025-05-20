# chuk_llm

A flexible and unified Python library for interacting with multiple Large Language Model (LLM) providers through a consistent interface.

## Features

- **Provider-agnostic API**: Use the same code to interact with multiple LLM providers
- **Supported providers**:
  - OpenAI (gpt-4o-mini, etc.)
  - Groq (llama-3.3-70b-versatile, etc.)
  - Ollama (locally-hosted models)
  - Google Gemini (gemini-2.0-flash, etc.)
  - Anthropic Claude (claude-3-7-sonnet-20250219, etc.)
- **Streaming support**: Async iterators for streaming completions from supported providers
- **Tool/function calling**: Consistent interface for tool/function calling across providers
- **Environment variables**: Load API keys and configurations from `.env` files
- **Configurable defaults**: Easy to customize default providers and models

## Installation

```bash
pip install chuk-llm
```

## Quick Start

```python
import asyncio
from chuk_llm.llm.llm_client import get_llm_client

async def main():
    # Get a client for the default provider (OpenAI)
    client = get_llm_client()
    
    # Or specify a provider and model
    # client = get_llm_client(provider="anthropic", model="claude-3-7-sonnet-20250219")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Non-streaming completion
    result = await client.create_completion(messages)
    print(result["response"])
    
    # Streaming completion
    async for chunk in client.create_completion(messages, stream=True):
        print(chunk["response"], end="", flush=True)

asyncio.run(main())
```

## Configuration

### Environment Variables

Create a `.env` file in your project directory:

```
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Groq
GROQ_API_KEY=your_groq_api_key

# Google/Gemini
GOOGLE_API_KEY=your_google_api_key
# or
GEMINI_API_KEY=your_gemini_api_key
```

### Provider Configuration

You can configure providers programmatically:

```python
from chuk_llm.llm.provider_config import ProviderConfig

config = ProviderConfig()

# Update a provider's configuration
config.update_provider_config("openai", {
    "api_base": "https://your-proxy.com/v1",
    "default_model": "gpt-4-turbo"
})

# Set active provider and model
config.set_active_provider("anthropic")
config.set_active_model("claude-3-7-sonnet-20250219")

# Use this configuration when getting a client
from chuk_llm.llm.llm_client import get_llm_client
client = get_llm_client(config=config)
```

## Tool/Function Calling Example

```python
import asyncio
import json
from chuk_llm.llm.llm_client import get_llm_client

async def main():
    client = get_llm_client(provider="openai", model="gpt-4o")
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in London today?"}
    ]
    
    result = await client.create_completion(messages, tools=tools)
    
    if result["tool_calls"]:
        for tool_call in result["tool_calls"]:
            if tool_call["function"]["name"] == "get_weather":
                args = json.loads(tool_call["function"]["arguments"])
                location = args["location"]
                print(f"The assistant wants to check the weather in {location}")
                
                # In a real application, you would call your weather API here
                weather_data = {"temperature": 22, "condition": "Partly cloudy"}
                
                # Add the function response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": "get_weather",
                    "content": json.dumps(weather_data)
                })
        
        # Get the final response with the tool results
        final_result = await client.create_completion(messages)
        print(final_result["response"])
    else:
        print(result["response"])

asyncio.run(main())
```

## System Prompt Generation

The library includes a `SystemPromptGenerator` class to help create structured system prompts:

```python
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator

# Define available tools
tools_json = {
    "functions": [
        {
            "name": "search_database",
            "description": "Search for information in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

# Generate a system prompt
generator = SystemPromptGenerator()
system_prompt = generator.generate_prompt(
    tools=tools_json,
    user_system_prompt="You are a database assistant that helps users find information.",
    tool_config="Use the search_database function whenever the user asks for information."
)

# Use in messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Find information about renewable energy"}
]
```

## Supported Provider Models

### OpenAI
- gpt-4o
- gpt-4o-mini
- gpt-3.5-turbo
- And others from OpenAI's model lineup

### Anthropic
- claude-3-7-sonnet-20250219
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku

### Groq
- llama-3.3-70b-versatile
- llama-3-8b
- And other models available on Groq

### Google Gemini
- gemini-2.0-flash
- gemini-1.5-pro
- And other Gemini models

### Ollama
- qwen3 (default)
- llama3
- mistral
- and any other model available in your Ollama installation

## Advanced Usage

### Using the OpenAIStyleMixin

Some providers share similar API patterns to OpenAI. The `OpenAIStyleMixin` provides helper methods for these providers:

- `_sanitize_tool_names`: Ensures tool names are valid
- `_call_blocking`: Runs blocking SDK calls in a thread
- `_normalise_message`: Converts provider-specific responses to a standard format
- `_stream_from_blocking`: Wraps blocking SDK generators into async iterators

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.