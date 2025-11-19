"""
Advantage Client
================

Modern Advantage client using the OpenAI-compatible API.
Includes workarounds for Advantage-specific API quirks.
"""

from __future__ import annotations

import logging
from typing import Any

from chuk_llm.core import (
    CompletionRequest,
    Message,
    MessageRole,
    ModelInfo,
    Provider,
)

from .openai_compatible import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class AdvantageClient(OpenAICompatibleClient):
    """
    Advantage API client with enhanced function calling support.

    Advantage is an OpenAI-compatible API but requires specific handling:
    - Strict parameter must be added to function definitions
    - Enhanced function calling prompts may be needed
    - Custom API base URL required

    Features:
    - Type-safe with Pydantic models
    - Fast JSON with orjson/ujson
    - Connection pooling with httpx
    - Zero-copy streaming
    - Proper error handling
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,  # Required for Advantage
        **kwargs: Any,
    ):
        """
        Initialize Advantage client.

        Args:
            model: Model name (e.g., "global/gpt-5-chat")
            api_key: Advantage API key
            base_url: Advantage API base URL (required)
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        if not base_url:
            raise ValueError(
                "base_url is required for Advantage client. "
                "Set ADVANTAGE_API_BASE environment variable or provide base_url parameter."
            )

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.provider = Provider.ADVANTAGE

        logger.info(
            f"Initialized Advantage client: model={model}, base_url={base_url}"
        )

    def get_model_info(self) -> ModelInfo:
        """
        Get model information with Advantage-specific metadata.

        Returns:
            ModelInfo with Advantage capabilities
        """
        return ModelInfo(
            provider=self.provider.value,
            model=self.model,
            is_reasoning=False,  # Advantage doesn't have reasoning models
            supports_tools=True,
            supports_streaming=True,
            supports_vision=False,  # Not supported
            supports_temperature=True,
            supports_top_p=True,
            supports_max_tokens=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
            supports_logit_bias=False,
            supports_logprobs=False,
        )

    def _prepare_request(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Prepare request with Advantage-specific adjustments.

        Args:
            request: Validated completion request

        Returns:
            API request parameters
        """
        # Get base request from parent
        params = super()._prepare_request(request)

        # Add strict parameter to tool definitions if tools are present
        if params.get("tools"):
            params["tools"] = self._add_strict_parameter_to_tools(params["tools"])

        # Enhance system message for function calling if needed
        if params.get("tools") and params.get("messages"):
            params["messages"] = self._inject_function_calling_prompt(
                params["messages"], params["tools"]
            )

        return params

    def _add_strict_parameter_to_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Add strict parameter to function definitions.

        The Advantage API requires the strict parameter to be present
        and be a boolean value.

        Args:
            tools: Original tool definitions

        Returns:
            Modified tools with strict parameter
        """
        modified_tools = []
        for tool in tools:
            tool_copy = tool.copy()
            if tool_copy.get("type") == "function" and "function" in tool_copy:
                # Make a copy of the function dict to avoid modifying the original
                func_copy = tool_copy["function"].copy()
                if "strict" not in func_copy:
                    func_copy["strict"] = False
                    logger.debug(
                        f"Added strict=False to tool: {func_copy.get('name', 'unknown')}"
                    )
                tool_copy["function"] = func_copy
            modified_tools.append(tool_copy)
        return modified_tools

    def _create_function_calling_system_prompt(self) -> str:
        """
        Create system prompt that guides the model to return function calls
        in the correct JSON format.

        Returns:
            System prompt for function calling
        """
        return (
            "When you need to call a function, respond with ONLY a JSON object in this exact format: "
            '{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}. '
            "Do not include markdown code blocks, explanations, or any other text. "
            "Just return the raw JSON object. "
            "If the user's request requires a function call, you MUST use this format."
        )

    def _inject_function_calling_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """
        Inject system prompt to guide function calling.

        This modifies the messages list to include instructions for the model
        to return function calls in a parseable JSON format.

        Args:
            messages: Original conversation messages
            tools: Tool/function definitions (if None, no modification)

        Returns:
            New list of messages with system prompt added/modified
        """
        # Only inject if tools are provided
        if not tools:
            return messages

        function_prompt = self._create_function_calling_system_prompt()

        # Copy messages to avoid mutating original
        new_messages = [msg.copy() for msg in messages]

        # If first message is already system, prepend to it
        if new_messages and new_messages[0].get("role") == "system":
            existing_content = new_messages[0]["content"]
            new_messages[0]["content"] = f"{function_prompt}\n\n{existing_content}"
            logger.debug(
                "Prepended function calling prompt to existing system message"
            )
        else:
            # Add new system message at the start
            new_messages.insert(0, {"role": "system", "content": function_prompt})
            logger.debug("Added function calling system message")

        return new_messages
