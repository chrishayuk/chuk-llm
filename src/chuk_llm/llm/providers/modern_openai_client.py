"""
Modern OpenAI Provider Adapter
===============================

Adapter that wraps the modern Pydantic-based OpenAI client to implement
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

from chuk_llm.clients.openai import OpenAIClient
from chuk_llm.compat import (
    completion_response_to_dict,
    completion_streaming_chunk_to_dict,
    dict_to_completion_request,
)
from chuk_llm.core import CompletionRequest, Provider
from chuk_llm.llm.core.base import BaseLLMClient

logger = logging.getLogger(__name__)


class ModernOpenAILLMClient(BaseLLMClient):
    """
    Modern OpenAI provider adapter using Pydantic models internally.

    Implements BaseLLMClient interface for backward compatibility while
    using the modern type-safe OpenAIClient internally.

    Architecture:
        External API: dict-based (backward compatible)
        Internal: Pydantic models (type-safe)
        Conversion: Uses compatibility layer
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize modern OpenAI client.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o", "o1", "gpt-5")
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            api_base: API base URL (defaults to https://api.openai.com/v1)
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        self.model = model

        # Create modern Pydantic-based client
        self._modern_client = OpenAIClient(
            model=model,
            api_key=api_key or "",  # OpenAIClient will use env var if empty
            base_url=api_base or "https://api.openai.com/v1",
            **kwargs,
        )

        logger.info(
            f"Initialized modern OpenAI client: model={model}, "
            f"reasoning={self._modern_client._is_reasoning}"
        )

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
            f"ModernOpenAILLMClient(model={self.model}, "
            f"provider={Provider.OPENAI.value})"
        )
