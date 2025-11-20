# chuk_llm/llm/providers/perplexity_client.py
"""
Perplexity API client with proper response_format translation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extends OpenAI client to handle Perplexity-specific API differences:
1. Translates OpenAI-style response_format to Perplexity's format
2. Removes tool support (Perplexity doesn't support function calling)
3. Handles Perplexity-specific parameters
"""

from __future__ import annotations

import logging
from typing import Any

from chuk_llm.llm.providers.openai_client import OpenAILLMClient

log = logging.getLogger(__name__)


class PerplexityLLMClient(OpenAILLMClient):
    """
    Perplexity-specific client that handles API differences from OpenAI.

    Key differences:
    - response_format uses json_schema type instead of json_object
    - No tool/function calling support
    - Perplexity-specific parameters (search_recency_filter, etc.)
    """

    def __init__(
        self,
        model: str = "sonar-pro",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        # Set default api_base for Perplexity if not provided
        if api_base is None:
            api_base = "https://api.perplexity.ai"

        super().__init__(model=model, api_key=api_key, api_base=api_base)

        # Override detected provider to perplexity
        self.detected_provider = "perplexity"

        log.debug(f"Perplexity client initialized: model={self.model}")

    def _translate_response_format(
        self, response_format: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """
        Translate OpenAI-style response_format to Perplexity format.

        OpenAI format: {"type": "json_object"}
        Perplexity format: {"type": "json_schema", "json_schema": {...}}

        Args:
            response_format: OpenAI-style response format

        Returns:
            Perplexity-compatible response format or None
        """
        if not response_format:
            return None

        format_type = response_format.get("type")

        if format_type == "json_object":
            # Convert to Perplexity's json_schema format
            # Perplexity requires a json_schema field with actual schema
            # For generic JSON mode, we'll use a permissive schema
            log.debug("Translating json_object to Perplexity json_schema format")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "json_response",
                    "schema": {"type": "object", "additionalProperties": True},
                    "strict": False,
                },
            }
        elif format_type == "json_schema":
            # Already in Perplexity format
            return response_format
        else:
            # Text or other formats - pass through
            return response_format

    async def _regular_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        name_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Override to handle Perplexity-specific parameters.

        Key changes:
        1. Remove tools (not supported by Perplexity)
        2. Translate response_format
        """
        # Perplexity doesn't support tools - warn and remove
        if tools:
            log.warning(
                f"Perplexity model {self.model} does not support tool calling. "
                "Tools will be ignored."
            )
            tools = None

        # Translate response_format if present
        if "response_format" in kwargs:
            log.debug(f"Original response_format: {kwargs['response_format']}")
            kwargs["response_format"] = self._translate_response_format(
                kwargs["response_format"]
            )
            log.debug(f"Translated response_format: {kwargs['response_format']}")
            if kwargs["response_format"] is None:
                del kwargs["response_format"]

        # Call parent implementation
        return await super()._regular_completion(
            messages=messages,
            tools=None,  # Always None for Perplexity
            name_mapping=name_mapping,
            **kwargs,
        )

    async def _stream_completion_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        name_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """
        Override to handle Perplexity-specific parameters for streaming.
        """
        # Perplexity doesn't support tools
        if tools:
            log.warning(
                f"Perplexity model {self.model} does not support tool calling. "
                "Tools will be ignored."
            )
            tools = None

        # Translate response_format if present
        if "response_format" in kwargs:
            log.debug(
                f"[Streaming] Original response_format: {kwargs['response_format']}"
            )
            kwargs["response_format"] = self._translate_response_format(
                kwargs["response_format"]
            )
            log.debug(
                f"[Streaming] Translated response_format: {kwargs['response_format']}"
            )
            if kwargs["response_format"] is None:
                del kwargs["response_format"]

        # Call parent implementation
        async for chunk in super()._stream_completion_async(
            messages=messages,
            tools=None,  # Always None for Perplexity
            name_mapping=name_mapping,
            **kwargs,
        ):
            yield chunk

    def supports_feature(self, feature: str) -> bool:
        """
        Override to accurately report Perplexity capabilities.

        Perplexity supports:
        - text, streaming, json_mode, vision, system_messages

        Perplexity does NOT support:
        - tools (function calling)
        """
        if feature == "tools":
            return False

        # Delegate to parent for other features
        return super().supports_feature(feature)
