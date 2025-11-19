"""
Type Converters
===============

Convert between old dict-based API and new Pydantic models.

Handles:
- Message format conversion
- Tool/function call conversion
- Request/response conversion
- Streaming chunk conversion
"""

from __future__ import annotations

from typing import Any

from chuk_llm.core import (
    CompletionRequest,
    CompletionResponse,
    ContentPart,
    ContentType,
    FinishReason,
    FunctionCall,
    ImageUrlContent,
    Message,
    MessageRole,
    StreamChunk,
    TextContent,
    Tool,
    ToolCall,
    ToolFunction,
    ToolType,
)

# ================================================================
# Inbound Converters (dict → Pydantic)
# ================================================================


def convert_message(msg_dict: dict[str, Any]) -> Message:
    """
    Convert dict message to Pydantic Message model.

    Handles:
    - Simple text messages
    - Multimodal content (text + images)
    - Tool call messages
    - Tool result messages

    Args:
        msg_dict: Message dict with 'role', 'content', optional 'tool_calls', etc.

    Returns:
        Validated Message model
    """
    # Parse role
    role_str = msg_dict.get("role", "user")
    try:
        role = MessageRole(role_str)
    except ValueError:
        role = MessageRole.USER

    # Parse content
    content = msg_dict.get("content")
    parsed_content: str | list[ContentPart] | None = None

    if content is not None:
        if isinstance(content, str):
            parsed_content = content
        elif isinstance(content, list):
            # Multimodal content (text + images)
            parts: list[ContentPart] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(
                            TextContent(
                                type=ContentType.TEXT, text=part.get("text", "")
                            )
                        )
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        if isinstance(image_url, str):
                            parts.append(
                                ImageUrlContent(
                                    type=ContentType.IMAGE_URL,
                                    image_url={"url": image_url},
                                )
                            )
                        else:
                            parts.append(
                                ImageUrlContent(
                                    type=ContentType.IMAGE_URL, image_url=image_url
                                )
                            )
            parsed_content = parts if parts else None

    # Parse tool calls
    tool_calls = None
    if "tool_calls" in msg_dict and msg_dict["tool_calls"]:
        tool_calls = []
        for tc in msg_dict["tool_calls"]:
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    type=ToolType.FUNCTION,
                    function=FunctionCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    ),
                )
            )

    # Parse tool_call_id (for tool result messages)
    tool_call_id = msg_dict.get("tool_call_id")

    # Parse name
    name = msg_dict.get("name")

    return Message(
        role=role,
        content=parsed_content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
    )


def convert_messages(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert list of message dicts to Pydantic Message models."""
    return [convert_message(msg) for msg in messages]


def convert_tool(tool_dict: dict[str, Any]) -> Tool:
    """
    Convert dict tool definition to Pydantic Tool model.

    Args:
        tool_dict: Tool dict with 'type' and 'function'

    Returns:
        Validated Tool model
    """
    function_def = tool_dict.get("function", {})

    # ToolFunction expects parameters as dict[str, Any] (JSON Schema)
    parameters = function_def.get("parameters", {})

    function = ToolFunction(
        name=function_def.get("name", ""),
        description=function_def.get("description", ""),  # Required field
        parameters=parameters,  # Pass JSON Schema dict directly
    )

    return Tool(function=function)


def convert_tools(tools: list[dict[str, Any]]) -> list[Tool]:
    """Convert list of tool dicts to Pydantic Tool models."""
    return [convert_tool(tool) for tool in tools]


def dict_to_completion_request(
    request_dict: dict[str, Any], default_model: str | None = None
) -> CompletionRequest:
    """
    Convert dict request to Pydantic CompletionRequest model.

    Args:
        request_dict: Request dict with messages, temperature, etc.
        default_model: Default model if not in request_dict

    Returns:
        Validated CompletionRequest model
    """
    # Parse messages
    messages_list = request_dict.get("messages", [])
    messages = convert_messages(messages_list)

    # Parse tools
    tools = None
    if "tools" in request_dict and request_dict["tools"]:
        tools = convert_tools(request_dict["tools"])

    # Build request
    return CompletionRequest(
        messages=messages,
        model=request_dict.get("model", default_model or ""),
        temperature=request_dict.get("temperature"),
        max_tokens=request_dict.get("max_tokens"),
        top_p=request_dict.get("top_p"),
        frequency_penalty=request_dict.get("frequency_penalty"),
        presence_penalty=request_dict.get("presence_penalty"),
        stop=request_dict.get("stop"),
        tools=tools,
        response_format=request_dict.get("response_format"),
        stream=request_dict.get("stream", False),
    )


# ================================================================
# Outbound Converters (Pydantic → dict)
# ================================================================


def completion_response_to_dict(response: CompletionResponse) -> dict[str, Any]:
    """
    Convert Pydantic CompletionResponse to dict for API output.

    Args:
        response: CompletionResponse model

    Returns:
        Dict matching expected API response format
    """
    result: dict[str, Any] = {}

    if response.content is not None:
        result["content"] = response.content

    if response.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type.value,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in response.tool_calls
        ]

    # Finish reason
    if isinstance(response.finish_reason, FinishReason):
        result["finish_reason"] = response.finish_reason.value
    else:
        result["finish_reason"] = response.finish_reason

    # Usage
    if response.usage:
        from chuk_llm.core.constants import ResponseKey

        result[ResponseKey.USAGE.value] = {
            ResponseKey.PROMPT_TOKENS.value: response.usage.prompt_tokens,
            ResponseKey.COMPLETION_TOKENS.value: response.usage.completion_tokens,
            ResponseKey.TOTAL_TOKENS.value: response.usage.total_tokens,
        }
        if response.usage.reasoning_tokens is not None:
            result[ResponseKey.USAGE.value][
                ResponseKey.REASONING_TOKENS.value
            ] = response.usage.reasoning_tokens

    # Model
    if response.model:
        result["model"] = response.model

    return result


def completion_streaming_chunk_to_dict(chunk: StreamChunk) -> dict[str, Any]:
    """
    Convert Pydantic StreamChunk to dict for streaming API output.

    Args:
        chunk: StreamChunk model

    Returns:
        Dict matching expected streaming chunk format
    """
    result: dict[str, Any] = {}

    if chunk.content is not None:
        result["content"] = chunk.content

    if chunk.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type.value,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in chunk.tool_calls
        ]

    if chunk.finish_reason:
        if isinstance(chunk.finish_reason, FinishReason):
            result["finish_reason"] = chunk.finish_reason.value
        else:
            result["finish_reason"] = chunk.finish_reason

    return result
