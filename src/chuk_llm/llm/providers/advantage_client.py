# chuk_llm/llm/providers/advantage_client.py
"""
Advantage API Client

This client extends OpenAILLMClient with enhanced function calling support.
"""

import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from chuk_llm.core.enums import MessageRole

from .openai_client import OpenAILLMClient

if TYPE_CHECKING:
    from chuk_llm.core.models import Message, Tool

log = logging.getLogger(__name__)


class AdvantageClient(OpenAILLMClient):
    """
    Advantage API client with enhanced function calling support.

    Extends OpenAILLMClient with optimized handling for function/tool calling.
    """

    def __init__(self, model: str, api_key: str, api_base: str | None = None, **kwargs):
        """
        Initialize Advantage client.

        Args:
            model: Model name (e.g., "global/gpt-5-chat")
            api_key: Advantage API key
            api_base: API base URL (should be provided via config or env var)
            **kwargs: Additional arguments passed to OpenAILLMClient
        """
        # api_base should come from configuration or environment variable
        # It's passed in by the client factory from the config
        if not api_base:
            raise ValueError(
                "api_base is required for Advantage client. "
                "Set ADVANTAGE_API_BASE environment variable or provide api_base parameter."
            )

        # Filter out config-only parameters that OpenAILLMClient doesn't accept
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["api_base_env", "api_key_fallback_env"]
        }

        # Call parent constructor
        super().__init__(model, api_key, api_base, **filtered_kwargs)

        # Override detected provider to be 'advantage' for proper configuration lookup
        # This needs to be done AFTER super().__init__() but we also need to reload
        # the config with the correct provider name
        self.detected_provider = "advantage"

        # Reinitialize the config mixin with correct provider
        from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

        ConfigAwareProviderMixin.__init__(self, "advantage", model)

        log.info(
            f"[advantage] Initialized Advantage client for model={model}, "
            f"base={api_base}"
        )

    # ================================================================
    # FUNCTION CALLING WORKAROUND
    # ================================================================

    def _add_strict_parameter_to_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:  # type: ignore[override]
        """
        Override to ensure strict parameter is added as a boolean.

        Pydantic native - converts Tool models to dicts with strict parameter.

        The Advantage API requires the strict parameter to be present and
        be a boolean value. This returns dicts because the parent OpenAI client
        expects dicts for the final API call.
        """

        modified_tools = []
        for tool in tools:
            # Convert Tool to dict
            tool_dict = tool.model_dump()

            # Add strict parameter to function
            if tool_dict.get("type") == "function" and "function" in tool_dict:
                if "strict" not in tool_dict["function"]:
                    tool_dict["function"]["strict"] = False
                    log.debug(
                        f"[advantage] Added strict=False to tool: {tool_dict['function'].get('name', 'unknown')}"
                    )

            modified_tools.append(tool_dict)
        return modified_tools

    def _create_function_calling_system_prompt(self) -> str:
        """
        Create system prompt that guides the model to return function calls
        in the correct JSON format.

        The Advantage API doesn't return proper function_call/tool_calls fields,
        so we need to instruct the model to format its response as JSON that
        we can then parse.
        """
        return (
            "When you need to call a function, respond with ONLY a JSON object in this exact format: "
            '{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}. '
            "Do not include markdown code blocks, explanations, or any other text. "
            "Just return the raw JSON object. "
            "If the user's request requires a function call, you MUST use this format."
        )

    def _inject_function_calling_prompt(
        self, messages: list[Message], tools: list[Tool] | None
    ) -> list[dict[str, Any]]:
        """
        Inject system prompt to guide function calling.

        Pydantic native - works with Message models and returns dicts for parent client.

        This modifies the messages list to include instructions for the model
        to return function calls in a parseable JSON format.

        Args:
            messages: Original Pydantic Message objects
            tools: Tool/function definitions (if None, no modification)

        Returns:
            List of message dicts with system prompt added/modified
        """

        # Only inject if tools are provided
        if not tools:
            # Convert to dicts for parent client
            return [msg.model_dump() for msg in messages]

        function_prompt = self._create_function_calling_system_prompt()

        # Convert messages to dicts for manipulation
        new_messages = [msg.model_dump() for msg in messages]

        # If first message is already system, prepend to it
        if new_messages and new_messages[0].get("role") == MessageRole.SYSTEM.value:
            existing_content = new_messages[0]["content"]
            new_messages[0]["content"] = f"{function_prompt}\n\n{existing_content}"
            log.debug(
                "[advantage] Prepended function calling prompt to existing system message"
            )
        else:
            # Add new system message at the start
            new_messages.insert(0, {"role": "system", "content": function_prompt})
            log.debug("[advantage] Added function calling system prompt")

        return new_messages

    def _parse_function_call_from_content(self, content: str) -> dict[str, Any] | None:
        """
        Parse function call JSON from content field.

        The Advantage API returns function calls as JSON strings in the
        content field instead of in proper function_call/tool_calls fields.

        Args:
            content: The response content string

        Returns:
            Dict with 'name' and 'arguments' if function call found, None otherwise
        """
        if not content:
            return None

        # Try direct JSON parse
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                log.debug(
                    f"[advantage] Parsed function call from content: {parsed['name']}"
                )
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        code_block_pattern = r"```(?:json)?\s*(\{[^`]+\})\s*```"
        match = re.search(code_block_pattern, content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if (
                    isinstance(parsed, dict)
                    and "name" in parsed
                    and "arguments" in parsed
                ):
                    log.debug(
                        f"[advantage] Parsed function call from code block: {parsed['name']}"
                    )
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try to find JSON object pattern in content
        json_pattern = (
            r'\{[^}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        )
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if (
                    isinstance(parsed, dict)
                    and "name" in parsed
                    and "arguments" in parsed
                ):
                    log.debug(
                        f"[advantage] Parsed function call from pattern: {parsed['name']}"
                    )
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _convert_content_to_tool_calls(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert function calls in content field to standard tool_calls format.

        This is the core of the Advantage workaround - it detects function calls
        that were returned as JSON in the content field and converts them to
        the standard OpenAI tool_calls format.

        Handles both raw API response format (with choices array) and normalized
        format (with response and tool_calls fields).

        Args:
            response: API response (raw or normalized)

        Returns:
            Response with tool_calls field populated if function call detected
        """
        # Handle normalized format (response, tool_calls keys)
        if "response" in response and "tool_calls" in response:
            # Skip if already has tool_calls
            if response.get("tool_calls"):
                return response

            # Try to parse function call from response content
            content = response.get("response", "")
            function_call = self._parse_function_call_from_content(content)

            if function_call:
                # Convert to tool_calls format
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call["arguments"]),
                    },
                }

                response["tool_calls"] = [tool_call]
                # Set content to None (standard for tool calls)
                response["response"] = None

                log.info(
                    f"[advantage] Converted content to tool_call: "
                    f"{function_call['name']}({function_call['arguments']})"
                )

            return response

        # Handle raw API response format (choices array)
        if "choices" not in response:
            return response

        for choice in response["choices"]:
            message = choice.get("message", {})

            # Skip if already has tool_calls
            if message.get("tool_calls"):
                continue

            # Try to parse function call from content
            content = message.get("content", "")
            function_call = self._parse_function_call_from_content(content)

            if function_call:
                # Convert to tool_calls format
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call["arguments"]),
                    },
                }

                message["tool_calls"] = [tool_call]
                # Set content to None (standard for tool calls)
                message["content"] = None

                log.info(
                    f"[advantage] Converted content to tool_call: "
                    f"{function_call['name']}({function_call['arguments']})"
                )

        return response

    # ================================================================
    # OVERRIDDEN METHODS
    # ================================================================

    def create_completion(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        *,
        stream: bool = False,
        **kwargs,
    ):
        """
        Create a completion with Advantage API.

        This overrides the parent method to inject the function calling
        system prompt and parse the response.

        Args:
            messages: Conversation messages
            tools: Tool/function definitions
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Response with tool_calls properly formatted (or AsyncIterator for streaming)
        """
        # Pydantic native - auto-convert dicts to Pydantic if needed
        from chuk_llm.core.models import Message as MessageModel
        from chuk_llm.core.models import Tool as ToolModel

        pydantic_messages = [
            msg if isinstance(msg, MessageModel) else MessageModel.model_validate(msg)
            for msg in messages
        ]
        pydantic_tools = (
            [
                tool if isinstance(tool, ToolModel) else ToolModel.model_validate(tool)
                for tool in tools
            ]
            if tools
            else None
        )

        messages_dicts = None
        tools_dicts = None

        # Inject function calling prompt if tools provided
        if pydantic_tools:
            # This returns dicts after injection
            messages_dicts = self._inject_function_calling_prompt(
                pydantic_messages, pydantic_tools
            )
            log.debug("[advantage] Injected function calling prompt")

            # Add strict parameter to tools (returns dicts)
            tools_dicts = self._add_strict_parameter_to_tools(pydantic_tools)
        else:
            # No tools, convert messages to dicts
            messages_dicts = [msg.model_dump() for msg in pydantic_messages]
            tools_dicts = None

        # Convert modified dicts back to Pydantic for parent client
        from chuk_llm.llm.core.base import (
            _ensure_pydantic_messages,
            _ensure_pydantic_tools,
        )

        messages_for_parent = _ensure_pydantic_messages(messages_dicts)
        tools_for_parent = _ensure_pydantic_tools(tools_dicts) if tools_dicts else None

        # Call parent implementation with Pydantic models
        # For streaming, parent returns AsyncIterator, for non-streaming returns coroutine
        result = super().create_completion(
            messages_for_parent, tools=tools_for_parent, stream=stream, **kwargs
        )

        # If streaming, wrap iterator to convert tool calls
        if stream:
            return self._stream_and_convert(result, tools_dicts)

        # For non-streaming, wrap in an async function to convert tool calls
        return self._complete_and_convert(result, tools_dicts)

    async def _stream_and_convert(
        self, stream_iterator, tools: list[dict[str, Any]] | None
    ):
        """Wrap streaming iterator to convert tool calls at the end"""
        import json
        import uuid

        accumulated_content = ""

        async for chunk in stream_iterator:
            # Accumulate content to detect function calls
            if isinstance(chunk, dict):
                response = chunk.get("response", "")
                if response:
                    accumulated_content += response

            yield chunk

        # After streaming completes, check if accumulated content is a function call
        if tools and accumulated_content:
            function_call = self._parse_function_call_from_content(accumulated_content)
            if function_call:
                log.info(
                    f"[advantage] Detected function call in streamed content: "
                    f"{function_call['name']}"
                )
                # Yield a final chunk with the tool_calls
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call["arguments"]),
                    },
                }

                yield {"response": None, "tool_calls": [tool_call]}

    async def _complete_and_convert(
        self, completion_coro, tools: list[dict[str, Any]] | None
    ):
        """Helper to await completion and convert tool calls"""
        response = await completion_coro

        # Convert content-based function calls to tool_calls format
        if tools:
            response = self._convert_content_to_tool_calls(response)

        return response

    async def stream_completion(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs,
    ):
        """
        Stream a completion with Advantage API.

        Pydantic native - auto-converts dicts to Pydantic models.

        Note: Function calling with streaming is complex with Advantage's
        non-standard implementation. The function call JSON needs to be
        accumulated before it can be parsed.

        Args:
            messages: Conversation messages (Pydantic Message objects or dicts)
            tools: Tool/function definitions (Pydantic Tool objects or dicts)
            **kwargs: Additional parameters

        Yields:
            Completion chunks
        """
        # Pydantic native - auto-convert dicts if needed
        from chuk_llm.core.models import Message as MessageModel
        from chuk_llm.core.models import Tool as ToolModel

        pydantic_messages = [
            msg if isinstance(msg, MessageModel) else MessageModel.model_validate(msg)
            for msg in messages
        ]
        pydantic_tools = (
            [
                tool if isinstance(tool, ToolModel) else ToolModel.model_validate(tool)
                for tool in tools
            ]
            if tools
            else None
        )

        # Inject function calling prompt if tools provided
        messages_dicts = None
        tools_dicts = None

        if pydantic_tools:
            messages_dicts = self._inject_function_calling_prompt(
                pydantic_messages, pydantic_tools
            )
            log.debug("[advantage] Injected function calling prompt for streaming")
            tools_dicts = [tool.model_dump() for tool in pydantic_tools]
        else:
            messages_dicts = [msg.model_dump() for msg in pydantic_messages]
            tools_dicts = None

        # For streaming, we need to accumulate content to detect function calls
        accumulated_content = ""

        async for chunk in super().stream_completion(
            messages_dicts, tools=tools_dicts, **kwargs
        ):
            # Accumulate content chunks
            if "choices" in chunk:
                for choice in chunk["choices"]:
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        accumulated_content += delta["content"]

            yield chunk

        # After streaming completes, check if accumulated content is a function call
        if tools and accumulated_content:
            function_call = self._parse_function_call_from_content(accumulated_content)
            if function_call:
                log.info(
                    f"[advantage] Detected function call in streamed content: "
                    f"{function_call['name']}"
                )
                # Yield a final chunk with the tool_calls in the expected format
                import json
                import uuid

                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call["arguments"]),
                    },
                }

                # Yield final chunk with tool_calls
                yield {
                    "response": None,  # Clear response when tool call is made
                    "tool_calls": [tool_call],
                }
