"""
Compatibility Layer Usage Example
==================================

Shows how to use the compatibility layer to bridge old dict-based API
and new Pydantic-based API.
"""

from chuk_llm.compat import (
    completion_response_to_dict,
    completion_streaming_chunk_to_dict,
    convert_messages,
    convert_tools,
    dict_to_completion_request,
)
from chuk_llm.core import CompletionResponse, FinishReason, TokenUsage

# ================================================================
# Inbound: Convert old dict-based requests to Pydantic
# ================================================================

# Old-style message dicts
old_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

# Convert to Pydantic Message models
pydantic_messages = convert_messages(old_messages)
print(f"Converted {len(pydantic_messages)} messages to Pydantic models")
print(f"First message: {pydantic_messages[0]}")

# Old-style tool dicts
old_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# Convert to Pydantic Tool models
pydantic_tools = convert_tools(old_tools)
print(f"\nConverted {len(pydantic_tools)} tools to Pydantic models")
print(f"First tool: {pydantic_tools[0]}")

# Old-style full request dict
old_request = {
    "messages": old_messages,
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 100,
    "tools": old_tools,
}

# Convert to Pydantic CompletionRequest
pydantic_request = dict_to_completion_request(old_request)
print(f"\nConverted request to Pydantic model")
print(f"Model: {pydantic_request.model}")
print(f"Temperature: {pydantic_request.temperature}")
print(f"Has tools: {bool(pydantic_request.tools)}")

# ================================================================
# Outbound: Convert Pydantic responses back to dicts for API
# ================================================================

# Pydantic response (from new client)
pydantic_response = CompletionResponse(
    content="Python is a high-level programming language.",
    finish_reason=FinishReason.STOP,
    usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    model="gpt-4o-mini",
)

# Convert to old-style dict for API compatibility
response_dict = completion_response_to_dict(pydantic_response)
print(f"\nConverted response to dict:")
print(f"Content: {response_dict['content']}")
print(f"Finish reason: {response_dict['finish_reason']}")
print(f"Usage: {response_dict['usage']}")

# ================================================================
# Usage in real code
# ================================================================


async def example_api_endpoint(request_dict: dict):
    """
    Example API endpoint that accepts old dict format
    but uses new Pydantic clients internally.
    """
    # Convert inbound dict to Pydantic
    request = dict_to_completion_request(request_dict, default_model="gpt-4o-mini")

    # Use with modern Pydantic client
    # client = OpenAIClient(...)
    # response = await client.complete(request)

    # For this example, create mock response
    from chuk_llm.core import CompletionResponse, FinishReason

    response = CompletionResponse(
        content="Mock response",
        finish_reason=FinishReason.STOP,
    )

    # Convert outbound Pydantic to dict for API
    return completion_response_to_dict(response)


# This allows gradual migration - external API stays dict-based
# while internal code uses type-safe Pydantic models
print("\n✅ Compatibility layer ready for gradual migration!")
