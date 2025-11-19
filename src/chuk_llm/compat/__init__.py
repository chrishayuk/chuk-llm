"""
Compatibility Layer
===================

Two-way compatibility for gradual migration to Pydantic-based API.

- **Inbound**: Convert dict-based requests to Pydantic models
- **Outbound**: Convert Pydantic responses back to dicts for API compatibility

Usage:
    # Convert inbound dict messages to Pydantic
    messages = [{"role": "user", "content": "Hello"}]
    pydantic_messages = convert_messages(messages)

    # Convert inbound dict request to Pydantic CompletionRequest
    request_dict = {"messages": [...], "temperature": 0.7}
    request = dict_to_completion_request(request_dict)

    # Convert outbound Pydantic response to dict for API
    response = CompletionResponse(content="Hi!", ...)
    response_dict = completion_response_to_dict(response)
"""

from .converters import (
    completion_response_to_dict,
    completion_streaming_chunk_to_dict,
    convert_message,
    convert_messages,
    convert_tool,
    convert_tools,
    dict_to_completion_request,
)

__all__ = [
    # Inbound converters (dict → Pydantic)
    "convert_message",
    "convert_messages",
    "convert_tool",
    "convert_tools",
    "dict_to_completion_request",
    # Outbound converters (Pydantic → dict)
    "completion_response_to_dict",
    "completion_streaming_chunk_to_dict",
]
