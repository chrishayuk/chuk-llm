# Chuk-LLM

A unified interface for interacting with multiple Large Language Model providers with a consistent API.

## Overview

Chuk-LLM provides a lightweight, unified interface to interact with various LLM providers including:

- OpenAI
- Anthropic (Claude)
- Groq
- Google (Gemini)
- Ollama (local models)

The library handles the differences between provider APIs, allowing you to switch models or providers with minimal code changes.

## Installation

```bash
pip install chuk-llm
```

## Features

- **Unified API**: Common interface across all supported providers
- **Streaming Support**: Stream tokens as they're generated
- **Function/Tool Calling**: Execute tools and functions across providers
- **Multimodal Support**: Process images with compatible models
- **Provider Configuration**: Centralized configuration management
- **Async Architecture**: Modern async/await design for efficient processing

## Quick Start

```python
import asyncio
from chuk_llm.llm_client import get_llm_client

async def main():
    # Get a client for any supported provider
    client = get_llm_client(provider="openai", model="gpt-4o-mini")
    
    # Simple completion
    response = await client.create_completion([
        {"role": "user", "content": "Hello! How are you today?"}
    ])
    
    print(response["response"])
    
    # With streaming
    stream = await client.create_completion([
        {"role": "user", "content": "Count to 5 slowly"}
    ], stream=True)
    
    async for chunk in stream:
        print(chunk["response"], end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## Tool Calling

```python
import asyncio
from chuk_llm.llm_client import get_llm_client

# Define a tool
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
}

async def main():
    client = get_llm_client(provider="openai", model="gpt-4o")
    
    # Request using tools
    response = await client.create_completion([
        {"role": "user", "content": "What's the weather like in London today?"}
    ], tools=[WEATHER_TOOL])
    
    # Check if the model wants to call a tool
    if tool_calls := response.get("tool_calls"):
        for tool_call in tool_calls:
            print(f"Tool call: {tool_call['function']['name']}")
            print(f"Arguments: {tool_call['function']['arguments']}")
            
            # Here you would actually call your tool/function...
            weather_data = {"temperature": 22, "condition": "Partly Cloudy"}
            
            # Send the tool's response back to the model
            final_response = await client.create_completion([
                {"role": "user", "content": "What's the weather like in London today?"},
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {"role": "tool", "content": str(weather_data), "tool_call_id": tool_call["id"]}
            ])
            
            print(final_response["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Multimodal Support

```python
import asyncio
import base64
from chuk_llm.llm_client import get_llm_client

async def main():
    # Use a model with vision capabilities
    client = get_llm_client(provider="openai", model="gpt-4o")
    
    # Load an image (base64 encoded)
    with open("image.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Create a multimodal message
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    }]
    
    response = await client.create_completion(messages)
    print(response["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Provider Configuration

Create a custom configuration with your preferred settings:

```python
from chuk_llm.llm.provider_config import ProviderConfig
from chuk_llm.llm_client import get_llm_client

# Create a configuration with custom settings
config = ProviderConfig({
    "openai": {
        "api_key": "your-key-here",
        "default_model": "gpt-4-turbo"
    },
    "anthropic": {
        "api_key": "your-anthropic-key",
        "default_model": "claude-3-sonnet-20240229"
    }
})

# Use this configuration when getting a client
client = get_llm_client(provider="anthropic", config=config)
```

## Diagnostics Tool

The package includes a diagnostics script to test various providers:

```bash
# Run basic diagnostics on all providers
python -m chuk_llm.diagnostics

# Check specific providers
python -m chuk_llm.diagnostics --providers openai anthropic

# List models available in your Ollama installation
python -m chuk_llm.diagnostics --show-ollama-models

# Use specific models
python -m chuk_llm.diagnostics --model "ollama:llama3.2"
```

## Advanced Usage

### Environment Variables

The library respects provider-specific environment variables:

- `OPENAI_API_KEY` - For OpenAI
- `ANTHROPIC_API_KEY` - For Anthropic
- `GROQ_API_KEY` - For Groq
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` - For Google Gemini

### Custom API Base URLs

```python
client = get_llm_client(
    provider="openai",
    api_base="https://your-custom-endpoint.com/v1"
)
```

## Debugging

Set the `LOGLEVEL` environment variable to adjust logging verbosity:

```bash
LOGLEVEL=DEBUG python your_script.py
```

## License

MIT

## Acknowledgements

This library was created to simplify working with multiple LLM providers in Python applications.