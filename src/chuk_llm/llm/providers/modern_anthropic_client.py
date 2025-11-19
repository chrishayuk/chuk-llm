"""
Modern Anthropic Provider Adapter
==================================

Adapter that wraps the modern Pydantic-based Anthropic client to implement
the legacy BaseLLMClient interface.

This provides backward compatibility while using type-safe Pydantic models internally.

Features:
- Zero dict[str, Any] usage internally
- All enums instead of magic strings
- Proper async/await patterns
- Type-safe with Pydantic models
- Fast JSON with orjson/ujson
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from chuk_llm.clients.anthropic import AnthropicClient
from chuk_llm.compat import (
    completion_response_to_dict,
    completion_streaming_chunk_to_dict,
    dict_to_completion_request,
)
from chuk_llm.core import CompletionRequest, Provider
from chuk_llm.llm.core.base import BaseLLMClient

logger = logging.getLogger(__name__)


class ModernAnthropicLLMClient(BaseLLMClient):
    """
    Modern Anthropic provider adapter using Pydantic models internally.

    Implements BaseLLMClient interface for backward compatibility while
    using the modern type-safe AnthropicClient internally.

    Architecture:
        External API: dict-based (backward compatible)
        Internal: Pydantic models (type-safe)
        Conversion: Uses compatibility layer
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize modern Anthropic client.

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            api_base: API base URL (defaults to https://api.anthropic.com/v1)
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        self.model = model

        # Create modern Pydantic-based client
        self._modern_client = AnthropicClient(
            model=model,
            api_key=api_key or "",  # AnthropicClient will use env var if empty
            base_url=api_base or "https://api.anthropic.com/v1",
            **kwargs,
        )

        logger.info(f"Initialized modern Anthropic client: model={model}")

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]] | Any:
        """
        Create completion using legacy dict-based interface.

        This method maintains backward compatibility while using
        Pydantic models internally.

        Args:
            messages: List of message dicts (legacy format)
            tools: Optional list of tool dicts (legacy format)
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            If stream=True: AsyncIterator yielding dict chunks
            If stream=False: Coroutine resolving to dict response
        """
        # Convert legacy dict request to Pydantic model
        request_dict = {
            "messages": messages,
            "model": self.model,
            "tools": tools,
            "stream": stream,
            **kwargs,
        }

        request: CompletionRequest = dict_to_completion_request(
            request_dict, default_model=self.model
        )

        if stream:
            return self._stream_completion(request)
        else:
            return self._create_completion(request)

    async def _create_completion(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Non-streaming completion using Pydantic models internally.

        Args:
            request: Validated CompletionRequest model

        Returns:
            Legacy dict response for backward compatibility
        """
        # Use modern Pydantic client
        response = await self._modern_client.complete(request)

        # Convert Pydantic response back to dict for backward compatibility
        result = completion_response_to_dict(response)

        # Add legacy "response" field if content exists
        if "content" in result:
            result["response"] = result["content"]

        return result

    async def _stream_completion(
        self, request: CompletionRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Streaming completion using Pydantic models internally.

        Args:
            request: Validated CompletionRequest model

        Yields:
            Legacy dict chunks for backward compatibility
        """
        # Use modern Pydantic client's streaming
        async for chunk in self._modern_client.stream(request):
            # Convert Pydantic StreamChunk to legacy dict
            chunk_dict = completion_streaming_chunk_to_dict(chunk)

            # Add legacy "response" field for backward compatibility
            if "content" in chunk_dict:
                chunk_dict["response"] = chunk_dict["content"]

            yield chunk_dict

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model information.

        Returns dict for backward compatibility while using
        Pydantic ModelInfo internally.
        """
        # Get Pydantic ModelInfo from modern client
        model_info = self._modern_client.get_model_info()

        # Convert to dict for backward compatibility
        return {
            "provider": model_info.provider,
            "model": model_info.model,
            "is_reasoning": model_info.is_reasoning,
            "supports_tools": model_info.supports_tools,
            "supports_streaming": model_info.supports_streaming,
            "supports_vision": model_info.supports_vision,
        }

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._modern_client.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModernAnthropicLLMClient(model={self.model}, "
            f"provider={Provider.ANTHROPIC.value})"
        )
