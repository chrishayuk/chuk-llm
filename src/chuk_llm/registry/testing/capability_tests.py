"""
Shared capability testing functions.

Common testing logic used by both:
- scripts/update_capabilities.py (offline testing)
- registry/runtime_tester.py (runtime testing)

This ensures consistent testing behavior across both systems.
"""

from __future__ import annotations

import logging
from typing import Any

from chuk_llm.core.enums import ContentType, MessageRole
from chuk_llm.core.models import (
    ImageDataContent,
    ImageUrlContent,
    Message,
    TextContent,
)

log = logging.getLogger(__name__)

# Small 16x16 red square PNG for vision testing
RED_SQUARE_PNG = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAPklEQVR42mP8z8DwnxGIGZAw+v8MCA4DMSM2TrJtYPzPyAiXx2YIThtgtqPbjm47RoMBZjvMdjQbcNqAbjs2GwBfBgphg7NNPQAAAABJRU5ErkJggg=="


async def test_tools(client: Any) -> bool:
    """
    Test if model supports tool calling.

    Args:
        client: LLM client instance

    Returns:
        True if model supports tools, False otherwise
    """
    try:
        messages = [
            Message(
                role=MessageRole.USER,
                content="What's the weather in Paris?",
            )
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await client.create_completion(
            messages, tools=tools, max_tokens=50, timeout=10.0
        )

        # Check for error response
        if isinstance(response, dict) and response.get("error"):
            return False

        # Check if response has tool calls
        if response and isinstance(response, dict) and response.get("tool_calls"):
            return True

        return False

    except Exception as e:
        error_msg = str(e).lower()
        if (
            "tool" in error_msg
            or "function" in error_msg
            or "not supported" in error_msg
        ):
            return False
        # Other errors might not be tool-related
        return False


async def test_vision(client: Any) -> bool:
    """
    Test if model supports vision/image inputs.

    Tests both image_url and image_data formats.

    Args:
        client: LLM client instance

    Returns:
        True if model supports vision, False otherwise
    """
    # Test 1: Try image_url with data URL
    try:
        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    TextContent(
                        type=ContentType.TEXT, text="What color is this image?"
                    ),
                    ImageUrlContent(
                        type=ContentType.IMAGE_URL,
                        image_url={"url": f"data:image/png;base64,{RED_SQUARE_PNG}"},
                    ),
                ],
            )
        ]

        response = await client.create_completion(messages, max_tokens=50, timeout=10.0)

        # Check if response contains an error
        if isinstance(response, dict) and response.get("error"):
            pass  # Try alternative format
        elif (
            response
            and isinstance(response, dict)
            and (response.get("response") or response.get("choices"))
        ):
            return True  # Success with image_url format
        else:
            pass  # Try alternative

    except Exception as e:
        error_msg = str(e).lower()
        if any(
            x in error_msg
            for x in ["image", "vision", "not supported", "invalid content"]
        ):
            pass  # Try alternative
        else:
            return False

    # Test 2: Try image_data format (alternative)
    try:
        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    TextContent(
                        type=ContentType.TEXT, text="What color is this image?"
                    ),
                    ImageDataContent(
                        type=ContentType.IMAGE_DATA,
                        image_data=RED_SQUARE_PNG,
                        mime_type="image/png",
                    ),
                ],
            )
        ]

        response = await client.create_completion(messages, max_tokens=50, timeout=10.0)

        if isinstance(response, dict) and response.get("error"):
            return False

        if (
            response
            and isinstance(response, dict)
            and (response.get("response") or response.get("choices"))
        ):
            return True

        return False

    except Exception as e:
        error_msg = str(e).lower()
        if any(
            x in error_msg
            for x in ["image", "vision", "not supported", "invalid content"]
        ):
            return False
        return False


async def test_json_mode(client: Any) -> bool:
    """
    Test if model supports JSON mode.

    Args:
        client: LLM client instance

    Returns:
        True if model supports JSON mode, False otherwise
    """
    try:
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant that outputs JSON.",
            ),
            Message(
                role=MessageRole.USER,
                content="Return JSON with a greeting field.",
            ),
        ]

        response = await client.create_completion(
            messages,
            response_format={"type": "json_object"},
            max_tokens=50,
            timeout=10.0,
        )

        # Check if response contains an error
        if isinstance(response, dict) and response.get("error"):
            return False

        return True

    except Exception as e:
        error_msg = str(e).lower()
        if any(
            x in error_msg
            for x in [
                "json",
                "response_format",
                "not supported",
                "invalid parameter",
            ]
        ):
            return False
        return False


async def test_structured_outputs(client: Any) -> bool:
    """
    Test if model supports structured outputs with JSON Schema.

    Args:
        client: LLM client instance

    Returns:
        True if model supports structured outputs, False otherwise
    """
    try:
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant that outputs structured data.",
            ),
            Message(
                role=MessageRole.USER,
                content="Generate a person with name and age.",
            ),
        ]

        # Define a JSON schema for structured output
        json_schema = {
            "name": "person",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        }

        response = await client.create_completion(
            messages,
            response_format={"type": "json_schema", "json_schema": json_schema},
            max_tokens=50,
            timeout=10.0,
        )

        # Check if response contains an error
        if isinstance(response, dict) and response.get("error"):
            return False

        return True

    except Exception as e:
        error_msg = str(e).lower()
        if any(
            x in error_msg
            for x in [
                "json_schema",
                "structured",
                "response_format",
                "not supported",
                "invalid parameter",
                "strict",
            ]
        ):
            return False
        return False


async def test_streaming(client: Any) -> bool:
    """
    Test if model supports streaming.

    Args:
        client: LLM client instance

    Returns:
        True if model supports streaming, False otherwise
    """
    try:
        messages = [Message(role=MessageRole.USER, content="Say hello")]

        chunks = 0
        async for chunk in client.create_completion(
            messages, stream=True, max_tokens=10, timeout=10.0
        ):
            if chunk.get("response"):
                chunks += 1
                if chunks > 0:
                    return True

        return chunks > 0

    except Exception:
        return False


async def test_text(client: Any) -> bool:
    """
    Test basic text input/output capability.

    Args:
        client: LLM client instance

    Returns:
        True if model supports basic text I/O, False otherwise
    """
    try:
        messages = [Message(role=MessageRole.USER, content="Say hello")]

        response = await client.create_completion(messages, max_tokens=50, timeout=10.0)

        # Check for error response
        if isinstance(response, dict) and response.get("error"):
            return False

        # Check if response has content
        if (
            response
            and isinstance(response, dict)
            and (response.get("response") or response.get("choices"))
        ):
            return True

        return False

    except Exception:
        return False


async def test_chat_model(client: Any) -> bool:
    """
    Test if model is a chat model (vs completion model).

    Args:
        client: LLM client instance

    Returns:
        True if model is a chat model, False otherwise
    """
    try:
        messages = [Message(role=MessageRole.USER, content="test")]

        response = await client.create_completion(messages, max_tokens=5, timeout=10.0)

        # Check for chat model error
        if isinstance(response, dict) and response.get("error"):
            error_msg = str(response.get("error", "")).lower()
            if "chat" in error_msg or "completion" in error_msg:
                return False

        # If we got a response, it's a chat model
        return True

    except Exception as e:
        error_msg = str(e).lower()
        if "chat" in error_msg and "not supported" in error_msg:
            return False
        return True  # Assume chat model unless explicitly not
