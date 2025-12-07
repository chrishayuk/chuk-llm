# chuk_llm/llm/core/base.py
"""
Common abstract interface for every LLM adapter.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chuk_llm.core.models import Message, Tool


def _ensure_pydantic_messages(messages: list) -> list:
    """Convert dict messages to Pydantic Message objects if needed (backward compatibility)."""
    if not messages or not isinstance(messages[0], dict):
        return messages  # Already Pydantic objects

    from chuk_llm.core.enums import ContentType, MessageRole, ToolType
    from chuk_llm.core.models import (
        FunctionCall,
        ImageUrlContent,
        Message,
        TextContent,
        ToolCall,
    )

    pydantic_messages = []
    for msg in messages:
        role = (
            MessageRole(msg["role"])
            if isinstance(msg.get("role"), str)
            else msg.get("role")
        )
        content = msg.get("content")

        # Convert content if it's a list of dicts
        if isinstance(content, list) and content and isinstance(content[0], dict):
            content_parts: list[TextContent | ImageUrlContent] = []
            for part in content:
                if part.get("type") == "text":
                    content_parts.append(
                        TextContent(type=ContentType.TEXT, text=part.get("text", ""))
                    )
                elif part.get("type") == "image_url":
                    content_parts.append(
                        ImageUrlContent(
                            type=ContentType.IMAGE_URL,
                            image_url=part.get("image_url", {}),
                        )
                    )
            content = content_parts if content_parts else content

        # Convert tool_calls if present
        tool_calls_list = None
        if msg.get("tool_calls"):
            tool_calls_list = []
            for tc in msg["tool_calls"]:
                # Skip invalid tool calls
                if not tc or not tc.get("function"):
                    continue

                # Get type, default to "function"
                tc_type = tc.get("type") or "function"
                if isinstance(tc_type, str):
                    tc_type = ToolType(tc_type)

                tool_calls_list.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc_type,
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"].get("arguments", "{}"),
                        ),
                    )
                )

        pydantic_messages.append(
            Message(
                role=role,
                content=content,
                tool_calls=tool_calls_list,
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
                reasoning_content=msg.get(
                    "reasoning_content"
                ),  # Preserve for DeepSeek reasoner
            )
        )
    return pydantic_messages


def _ensure_pydantic_tools(tools: list | None) -> list | None:
    """Convert dict tools to Pydantic Tool objects if needed (backward compatibility)."""
    if not tools or not isinstance(tools[0], dict):
        return tools  # Already Pydantic objects or None

    from chuk_llm.core.enums import ToolType
    from chuk_llm.core.models import Tool, ToolFunction

    pydantic_tools = []
    for tool in tools:
        # Skip invalid tools
        if not tool or not tool.get("function"):
            continue

        # Get type, default to "function"
        tool_type = tool.get("type") or "function"
        if isinstance(tool_type, str):
            tool_type = ToolType(tool_type)

        pydantic_tools.append(
            Tool(
                type=tool_type,
                function=ToolFunction(
                    name=tool["function"]["name"],
                    description=tool["function"].get("description", ""),
                    parameters=tool["function"].get("parameters", {}),
                ),
            )
        )
    return pydantic_tools


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM chat clients."""

    @abc.abstractmethod
    def create_completion(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]] | Any:
        """
        Generate (or continue) a chat conversation.

        Parameters
        ----------
        messages
            List of Pydantic Message objects (not dicts).
        tools
            Optional list of Pydantic Tool objects (not dicts).
        stream
            Whether to stream the response or return complete response.
        **kwargs
            Additional parameters to pass to the underlying LLM.

        Returns
        -------
        When stream=True: AsyncIterator that yields chunks as they arrive
        When stream=False: Awaitable that resolves to standardised payload
                          with keys ``response`` and ``tool_calls``.

        CRITICAL: When stream=True, this method MUST NOT be async and
                 MUST return the async iterator directly (no awaiting).
        """
        ...

    async def close(self):
        """
        Cleanup resources and invalidate client cache.

        This default implementation handles cache invalidation to prevent
        returning closed clients from get_client(). Subclasses should call
        super().close() and then close their own resources.

        Example:
            async def close(self):
                await super().close()  # Invalidate cache
                await self.client.close()  # Close provider client
        """
        # Invalidate cache entry to prevent returning closed clients
        try:
            from chuk_llm.client_registry import invalidate_client

            invalidate_client(self)
        except ImportError:
            # client_registry not available (shouldn't happen, but be safe)
            pass
