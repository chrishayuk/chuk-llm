# chuk_llm/llm/providers/advantage_client.py
"""
Advantage API Client

This client extends OpenAILLMClient with enhanced function calling support.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from .openai_client import OpenAILLMClient

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
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['api_base_env', 'api_key_fallback_env']}

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

    def _add_strict_parameter_to_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Override to ensure strict parameter is added as a boolean.

        The Advantage API requires the strict parameter to be present and
        be a boolean value.
        """
        modified_tools = []
        for tool in tools:
            tool_copy = tool.copy()
            if tool_copy.get("type") == "function" and "function" in tool_copy:
                # Make a copy of the function dict to avoid modifying the original
                func_copy = tool_copy["function"].copy()
                if "strict" not in func_copy:
                    func_copy["strict"] = False
                    log.debug(
                        f"[advantage] Added strict=False to tool: {func_copy.get('name', 'unknown')}"
                    )
                tool_copy["function"] = func_copy
            modified_tools.append(tool_copy)
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
            "When calling a function, respond with ONLY a JSON object in this format: "
            '{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}. '
            "Do not include markdown code blocks, explanations, or any other text. "
            "Just return the raw JSON object."
        )

    def _inject_function_calling_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
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
        new_messages = []
        for msg in messages:
            new_messages.append(msg.copy())

        # If first message is already system, prepend to it
        if new_messages and new_messages[0].get("role") == "system":
            existing_content = new_messages[0]["content"]
            new_messages[0]["content"] = f"{function_prompt}\n\n{existing_content}"
            log.debug("[advantage] Prepended function calling prompt to existing system message")
        else:
            # Add new system message at the start
            new_messages.insert(0, {
                "role": "system",
                "content": function_prompt
            })
            log.debug("[advantage] Added function calling system prompt")

        return new_messages

    def _parse_function_call_from_content(
        self,
        content: str
    ) -> Optional[Dict[str, Any]]:
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
                log.debug(f"[advantage] Parsed function call from content: {parsed['name']}")
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        match = re.search(code_block_pattern, content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    log.debug(f"[advantage] Parsed function call from code block: {parsed['name']}")
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try to find JSON object pattern in content
        json_pattern = r'\{[^}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    log.debug(f"[advantage] Parsed function call from pattern: {parsed['name']}")
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _convert_content_to_tool_calls(
        self,
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                        "arguments": json.dumps(function_call["arguments"])
                    }
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
                        "arguments": json.dumps(function_call["arguments"])
                    }
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
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs
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
        # Inject function calling prompt if tools provided
        if tools:
            messages = self._inject_function_calling_prompt(messages, tools)
            log.debug("[advantage] Injected function calling prompt")

            # Add strict parameter to tools (required by Advantage API)
            tools = self._add_strict_parameter_to_tools(tools)

        # Call parent implementation
        # For streaming, parent returns AsyncIterator, for non-streaming returns coroutine
        result = super().create_completion(messages, tools=tools, stream=stream, **kwargs)

        # If streaming, return the iterator directly
        if stream:
            return result

        # For non-streaming, wrap in an async function to convert tool calls
        return self._complete_and_convert(result, tools)

    async def _complete_and_convert(
        self,
        completion_coro,
        tools: Optional[List[Dict[str, Any]]]
    ):
        """Helper to await completion and convert tool calls"""
        response = await completion_coro

        # Convert content-based function calls to tool_calls format
        if tools:
            response = self._convert_content_to_tool_calls(response)

        return response

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Stream a completion with Advantage API.

        Note: Function calling with streaming is complex with Advantage's
        non-standard implementation. The function call JSON needs to be
        accumulated before it can be parsed.

        Args:
            messages: Conversation messages
            tools: Tool/function definitions
            **kwargs: Additional parameters

        Yields:
            Completion chunks
        """
        # Inject function calling prompt if tools provided
        if tools:
            messages = self._inject_function_calling_prompt(messages, tools)
            log.debug("[advantage] Injected function calling prompt for streaming")

        # For streaming, we need to accumulate content to detect function calls
        accumulated_content = ""

        async for chunk in super().stream_completion(messages, tools=tools, **kwargs):
            # Accumulate content chunks
            if "choices" in chunk:
                for choice in chunk["choices"]:
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        accumulated_content += delta["content"]

            yield chunk

        # After streaming completes, check if accumulated content is a function call
        # This is a limitation - we can't detect mid-stream
        if tools and accumulated_content:
            function_call = self._parse_function_call_from_content(accumulated_content)
            if function_call:
                log.info(
                    f"[advantage] Detected function call in streamed content: "
                    f"{function_call['name']}"
                )
                # Note: At this point streaming is done, so we can't modify
                # the already-yielded chunks. The user will need to parse
                # the accumulated content themselves, or use non-streaming mode.
