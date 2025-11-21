# chuk_llm/llm/providers/mistral_client.py - FIXED VERSION WITH DUPLICATION PREVENTION

"""
Mistral Le Plateforme chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Features
--------
* Configuration-driven capabilities from YAML instead of hardcoded patterns
* Full support for Mistral's API including vision, function calling, and streaming
* Real async streaming without buffering
* Vision capabilities for supported models
* Function calling support for compatible models
* Universal tool name compatibility with bidirectional mapping
* CRITICAL FIX: Tool call duplication prevention in streaming
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

# Import Mistral SDK
try:
    from mistralai import Mistral
except ImportError as e:
    raise ImportError(
        "mistralai package is required for Mistral provider. "
        "Install with: pip install mistralai"
    ) from e

# Base imports
from chuk_llm.core.enums import ContentType, MessageRole
from chuk_llm.llm.core.base import BaseLLMClient
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

log = logging.getLogger(__name__)


class MistralLLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, BaseLLMClient):
    """
    Configuration-aware adapter for Mistral Le Plateforme API.

    Gets all capabilities from unified YAML configuration instead of
    hardcoded model patterns for better maintainability.

    CRITICAL FIX: Now includes tool call duplication prevention using the same
    pattern that was successfully implemented for Groq.

    Uses universal tool name compatibility system to handle any naming convention:
    - stdio.read_query -> stdio_read_query
    - web.api:search -> web_api_search
    - database.sql.execute -> database_sql_execute
    - service:method -> service_method
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        # Initialize mixins
        ConfigAwareProviderMixin.__init__(self, "mistral", model)
        ToolCompatibilityMixin.__init__(self, "mistral")

        self.model = model
        self.provider_name = "mistral"

        # Initialize Mistral client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["server_url"] = api_base

        # Create client with proper kwargs handling
        if client_kwargs:
            if "api_key" in client_kwargs and "server_url" in client_kwargs:
                self.client = Mistral(
                    api_key=client_kwargs["api_key"],
                    server_url=client_kwargs["server_url"],
                )
            elif "api_key" in client_kwargs:
                self.client = Mistral(api_key=client_kwargs["api_key"])
            elif "server_url" in client_kwargs:
                self.client = Mistral(server_url=client_kwargs["server_url"])
            else:
                self.client = Mistral()
        else:
            self.client = Mistral()

        log.info(f"MistralLLMClient initialized with model: {model}")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model info using configuration, with Mistral-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()

        # Add tool compatibility info
        tool_compatibility = self.get_tool_compatibility_info()

        # Add Mistral-specific metadata only if no error occurred
        if not info.get("error"):
            info.update(
                {
                    "mistral_specific": {
                        "supports_magistral_reasoning": "magistral"
                        in self.model.lower(),
                        "supports_code_generation": any(
                            pattern in self.model.lower()
                            for pattern in ["codestral", "devstral"]
                        ),
                        "is_multilingual": "saba" in self.model.lower(),
                        "is_edge_model": "ministral" in self.model.lower(),
                        "duplication_fix": "enabled",  # NEW: Indicates duplication fix is active
                    },
                    # Universal tool compatibility info
                    **tool_compatibility,
                    "parameter_mapping": {
                        "temperature": "temperature",
                        "max_tokens": "max_tokens",
                        "top_p": "top_p",
                        "stream": "stream",
                        "tool_choice": "tool_choice",
                    },
                    "unsupported_parameters": [
                        "frequency_penalty",
                        "presence_penalty",
                        "stop",
                        "logit_bias",
                        "user",
                        "n",
                        "best_of",
                        "top_k",
                        "seed",
                    ],
                }
            )

        return info

    async def _convert_messages_to_mistral_format(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert ChatML messages to Mistral format with configuration-aware vision handling and URL downloading"""
        mistral_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Handle different message types
            if role == MessageRole.SYSTEM.value:
                # Always try system messages - let API handle if not supported
                mistral_messages.append(
                    {"role": MessageRole.SYSTEM.value, "content": content}
                )

            elif role == MessageRole.USER.value:
                if isinstance(content, str):
                    # Simple text message
                    mistral_messages.append(
                        {"role": MessageRole.USER.value, "content": content}
                    )
                elif isinstance(content, list):
                    # Permissive approach: Process all multimodal content (text, vision, audio)
                    # Let Mistral API handle unsupported cases rather than filtering
                    # Process multimodal content normally (both dict and Pydantic)
                    mistral_content = []
                    for item in content:
                        if isinstance(item, dict):
                            # Dict-based content
                            if item.get("type") == ContentType.TEXT.value:
                                mistral_content.append(
                                    {
                                        "type": ContentType.TEXT.value,
                                        "text": item.get("text", ""),
                                    }
                                )
                            elif item.get("type") == ContentType.IMAGE_URL.value:
                                image_url = item.get("image_url", {})
                                url = (
                                    image_url.get("url", "")
                                    if isinstance(image_url, dict)
                                    else str(image_url)
                                )

                                # Download HTTP(S) URLs and convert to base64
                                if url.startswith(("http://", "https://")):
                                    try:
                                        log.debug(
                                            f"Downloading image from URL for Mistral: {url[:100]}..."
                                        )
                                        (
                                            encoded_data,
                                            image_format,
                                        ) = await OpenAIStyleMixin._download_and_encode_image(
                                            url
                                        )
                                        url = f"data:image/{image_format};base64,{encoded_data}"
                                        log.debug(
                                            "Image downloaded and encoded successfully"
                                        )
                                    except Exception as e:
                                        log.error(
                                            f"Failed to download image from {url}: {e}"
                                        )
                                        # Keep original URL and let Mistral handle the error

                                mistral_content.append(
                                    {
                                        "type": ContentType.IMAGE_URL.value,
                                        "image_url": url,
                                    }
                                )
                        else:
                            # Pydantic object-based content
                            if hasattr(item, "type") and item.type == ContentType.TEXT:
                                mistral_content.append(
                                    {
                                        "type": ContentType.TEXT.value,
                                        "text": item.text,
                                    }
                                )
                            elif (
                                hasattr(item, "type")
                                and item.type == ContentType.IMAGE_URL
                            ):
                                image_url_data = item.image_url
                                url = (
                                    image_url_data.get("url")
                                    if isinstance(image_url_data, dict)
                                    else image_url_data
                                )

                                # Download HTTP(S) URLs and convert to base64
                                if url and url.startswith(("http://", "https://")):
                                    try:
                                        log.debug(
                                            f"Downloading image from URL for Mistral: {url[:100]}..."
                                        )
                                        (
                                            encoded_data,
                                            image_format,
                                        ) = await OpenAIStyleMixin._download_and_encode_image(
                                            url
                                        )
                                        url = f"data:image/{image_format};base64,{encoded_data}"
                                        log.debug(
                                            "Image downloaded and encoded successfully"
                                        )
                                    except Exception as e:
                                        log.error(
                                            f"Failed to download image from {url}: {e}"
                                        )
                                        # Keep original URL and let Mistral handle the error

                                mistral_content.append(
                                    {
                                        "type": ContentType.IMAGE_URL.value,
                                        "image_url": url,
                                    }
                                )

                    mistral_messages.append(
                        {"role": MessageRole.USER.value, "content": mistral_content}
                    )

            elif role == MessageRole.ASSISTANT.value:
                # Handle assistant messages with potential tool calls - permissive approach
                if msg.get("tool_calls"):
                    # Convert tool calls to Mistral format
                    # IMPORTANT: Sanitize tool names in conversation history
                    tool_calls = []
                    for tc in msg["tool_calls"]:
                        original_name = tc["function"]["name"]

                        # Apply sanitization to tool names in conversation history
                        # Use inherited sanitizer from ToolCompatibilityMixin
                        sanitized_name = self._sanitizer.sanitize_for_provider(
                            original_name, self.provider_name
                        )

                        tool_calls.append(
                            {
                                "id": tc.get("id"),
                                "type": tc.get("type", MessageRole.FUNCTION.value),
                                "function": {
                                    "name": sanitized_name,  # Use sanitized name for API
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                        )

                    mistral_messages.append(
                        {
                            "role": MessageRole.ASSISTANT.value,
                            "content": content or "",
                            "tool_calls": tool_calls,
                        }
                    )
                else:
                    mistral_messages.append(
                        {"role": MessageRole.ASSISTANT.value, "content": content or ""}
                    )

            elif role == MessageRole.TOOL.value:
                # Tool response messages - permissive approach
                # IMPORTANT: Sanitize tool name to match what was sent in assistant message
                original_name = msg.get("name", "")
                # Use inherited sanitizer from ToolCompatibilityMixin
                sanitized_name = self._sanitizer.sanitize_for_provider(
                    original_name, self.provider_name
                )

                mistral_messages.append(
                    {
                        "role": MessageRole.TOOL.value,
                        "name": sanitized_name,  # Use sanitized name
                        "content": content or "",
                        "tool_call_id": msg.get("tool_call_id", ""),
                    }
                )

        return mistral_messages

    def _normalize_mistral_response(
        self, response: Any, name_mapping: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Convert Mistral response to standard format and restore tool names"""
        # Handle both response types
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message

            content = getattr(message, "content", "") or ""
            tool_calls = []

            # Extract tool calls if present - permissive approach
            if hasattr(message, "tool_calls") and message.tool_calls:
                from chuk_llm.core.enums import ToolType

                for tc in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": ToolType.FUNCTION,  # Always use enum value
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )

            # Extract usage information if present
            usage_info = None
            if hasattr(response, "usage") and response.usage:
                from chuk_llm.core.constants import ResponseKey

                usage = response.usage
                # Mistral may include reasoning_tokens for magistral models
                reasoning_tokens = getattr(
                    usage, ResponseKey.REASONING_TOKENS.value, None
                )

                usage_info = {
                    ResponseKey.PROMPT_TOKENS.value: getattr(
                        usage, ResponseKey.PROMPT_TOKENS.value, 0
                    ),
                    ResponseKey.COMPLETION_TOKENS.value: getattr(
                        usage, ResponseKey.COMPLETION_TOKENS.value, 0
                    ),
                    ResponseKey.TOTAL_TOKENS.value: getattr(
                        usage, ResponseKey.TOTAL_TOKENS.value, 0
                    ),
                }
                if reasoning_tokens is not None:
                    usage_info[ResponseKey.REASONING_TOKENS.value] = reasoning_tokens

            # Create response
            result = {
                "response": content if content else None,
                "tool_calls": tool_calls,
            }

            if usage_info:
                result["usage"] = usage_info

            # Restore original tool names using universal restoration
            if name_mapping and tool_calls:
                result = self._restore_tool_names_in_response(result, name_mapping)

            return result

        # Fallback for unexpected response format
        return {"response": str(response), "tool_calls": []}

    def _validate_request_with_config(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, bool, dict[str, Any]]:
        """
        Validate request against configuration before processing.
        """
        validated_messages = messages
        validated_tools = tools
        validated_stream = stream
        validated_kwargs = kwargs.copy()

        # Permissive approach: Let API handle streaming support
        # Don't block based on capability checks

        # Permissive approach: Let Mistral API handle tool support
        # Don't block based on capability checks
        # Permissive approach: Pass all content to API (vision, audio, etc.)

        # Validate parameters using configuration
        validated_kwargs = self.validate_parameters(**validated_kwargs)

        return validated_messages, validated_tools, validated_stream, validated_kwargs

    def create_completion(
        self,
        messages: list,  # Pydantic Message objects (or dicts for backward compat)
        tools: (
            list | None
        ) = None,  # Pydantic Tool objects (or dicts for backward compat)
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]] | Any:
        """
        Configuration-aware completion with Mistral API and universal tool name compatibility.

        CRITICAL FIX: Now includes tool call duplication prevention using the same
        successful pattern from Groq.

        Args:
            messages: List of Pydantic Message objects (or dicts for backward compatibility)
            tools: List of Pydantic Tool objects (or dicts for backward compatibility)
            stream: Whether to stream response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            AsyncIterator for streaming, awaitable for non-streaming
        """
        # Handle backward compatibility
        from chuk_llm.llm.core.base import (
            _ensure_pydantic_messages,
            _ensure_pydantic_tools,
        )

        messages = _ensure_pydantic_messages(messages)
        tools = _ensure_pydantic_tools(tools)

        # Convert Pydantic to dicts
        dict_messages = [msg.to_dict() for msg in messages]
        dict_tools = [tool.to_dict() for tool in tools] if tools else None

        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            self._validate_request_with_config(
                dict_messages, dict_tools, stream, **kwargs
            )
        )

        # Apply universal tool name sanitization (stores mapping for restoration)
        # Keep tools as Pydantic objects throughout
        name_mapping = {}
        if validated_tools:
            validated_tools = self._sanitize_tool_names(validated_tools)
            name_mapping = self._current_name_mapping
            log.debug(
                f"Tool sanitization: {len(name_mapping)} tools processed for Mistral compatibility"
            )

        # Convert messages to Mistral format (with configuration-aware processing)
        # Note: This needs to be async for image downloading, so we'll handle it in the async methods

        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": validated_messages,  # Will be converted in async methods
            **validated_kwargs,
        }

        # Add tools if provided and supported
        if validated_tools:
            request_params["tools"] = validated_tools
            # Set tool_choice to "auto" by default if not specified
            if "tool_choice" not in validated_kwargs:
                request_params["tool_choice"] = "auto"

        if validated_stream:
            return self._stream_completion_async(request_params, name_mapping)
        else:
            return self._regular_completion(request_params, name_mapping)

    async def _stream_completion_async(
        self, request_params: dict[str, Any], name_mapping: dict[str, str] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        COMPLETELY FIXED: Mistral streaming with proper JSON completion testing.

        Uses the same successful completion-based approach from OpenAI/Azure,
        not the problematic signature-based approach.

        Key fixes:
        - Only yield tool calls when JSON arguments are complete and parseable
        - Added completion status tracking (like OpenAI/Azure fix)
        - Removed signature tracking system entirely
        - Prevents both JSON parsing errors and tool call duplication
        """
        try:
            log.debug(f"Starting Mistral streaming for model: {self.model}")

            # Convert messages to Mistral format (with async image downloading)
            request_params["messages"] = await self._convert_messages_to_mistral_format(
                request_params["messages"]
            )

            # Convert tools to dicts only at this final step (Pydantic-native until SDK call)
            if "tools" in request_params and request_params["tools"]:
                request_params["tools"] = self._tools_to_dicts(request_params["tools"])

            # Use Mistral's streaming endpoint
            stream = self.client.chat.stream(**request_params)

            # FIXED: Simple completion-based tracking (like OpenAI/Azure)
            accumulated_tool_calls = {}  # {index: {id, name, arguments, complete}}
            chunk_count = 0
            total_content = ""

            # Process streaming response
            for chunk in stream:
                chunk_count += 1

                content = ""
                completed_tool_calls = []  # Only completed tool calls this chunk

                try:
                    if hasattr(chunk, "data") and hasattr(chunk.data, "choices"):
                        choices = chunk.data.choices
                        if choices:
                            choice = choices[0]

                            if hasattr(choice, "delta"):
                                delta = choice.delta

                                # Handle content - this works fine
                                if hasattr(delta, "content") and delta.content:
                                    content = delta.content
                                    total_content += content

                                # FIXED: Handle tool calls with proper completion testing
                                if (
                                    hasattr(delta, "tool_calls")
                                    and delta.tool_calls
                                    and self.supports_feature("tools")
                                ):
                                    for tc in delta.tool_calls:
                                        try:
                                            tc_index = getattr(tc, "index", 0)

                                            # Initialize accumulator with completion tracking
                                            if tc_index not in accumulated_tool_calls:
                                                accumulated_tool_calls[tc_index] = {
                                                    "id": getattr(
                                                        tc,
                                                        "id",
                                                        f"call_{uuid.uuid4().hex[:8]}",
                                                    ),
                                                    "name": "",
                                                    "arguments": "",
                                                    "complete": False,  # ADDED: Track completion status
                                                }

                                            tool_call_data = accumulated_tool_calls[
                                                tc_index
                                            ]

                                            # Update data
                                            if hasattr(tc, "id") and tc.id:
                                                tool_call_data["id"] = tc.id

                                            if hasattr(tc, "function") and tc.function:
                                                if (
                                                    hasattr(tc.function, "name")
                                                    and tc.function.name
                                                ):
                                                    tool_call_data["name"] += (
                                                        tc.function.name
                                                    )

                                                if (
                                                    hasattr(tc.function, "arguments")
                                                    and tc.function.arguments
                                                ):
                                                    tool_call_data["arguments"] += (
                                                        tc.function.arguments
                                                    )

                                            # CRITICAL FIX: Only yield when JSON is complete and valid
                                            if (
                                                tool_call_data["name"]
                                                and tool_call_data["arguments"]
                                                and not tool_call_data["complete"]
                                            ):
                                                try:
                                                    # Test if JSON is complete and valid
                                                    # Ensure arguments is a string before parsing
                                                    args_str = str(
                                                        tool_call_data["arguments"]
                                                    )
                                                    parsed_args = json.loads(args_str)

                                                    # Mark as complete and add to current chunk
                                                    tool_call_data["complete"] = True

                                                    from chuk_llm.core.enums import (
                                                        ToolType,
                                                    )

                                                    tool_call = {
                                                        "id": tool_call_data["id"],
                                                        "type": ToolType.FUNCTION,  # Always use enum value
                                                        "function": {
                                                            "name": tool_call_data[
                                                                "name"
                                                            ],
                                                            "arguments": json.dumps(
                                                                parsed_args
                                                            ),
                                                        },
                                                    }

                                                    completed_tool_calls.append(
                                                        tool_call
                                                    )
                                                    log.debug(
                                                        f"Mistral tool call {tc_index} completed: {tool_call_data['name']}"
                                                    )

                                                except json.JSONDecodeError:
                                                    # JSON incomplete - keep accumulating
                                                    log.debug(
                                                        f"Mistral tool call {tc_index} JSON incomplete, continuing accumulation"
                                                    )
                                                    pass

                                        except Exception as e:
                                            log.debug(
                                                f"Error processing Mistral streaming tool call chunk: {e}"
                                            )
                                            continue

                except Exception as chunk_error:
                    log.warning(
                        f"Error processing Mistral chunk {chunk_count}: {chunk_error}"
                    )
                    content = ""

                # Create chunk response
                chunk_response = {
                    "response": content,
                    "tool_calls": (
                        completed_tool_calls if completed_tool_calls else None
                    ),
                }

                # Restore tool names using universal restoration
                if name_mapping and completed_tool_calls:
                    chunk_response = self._restore_tool_names_in_response(
                        chunk_response, name_mapping
                    )

                # Only yield if we have content or completed tool calls
                if content or completed_tool_calls:
                    yield chunk_response

                # Allow other async tasks to run
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)

            log.debug(
                f"Mistral streaming completed with {chunk_count} chunks, "
                f"{len(total_content)} total characters, {len(accumulated_tool_calls)} tool calls"
            )

        except Exception as e:
            log.error(f"Error in Mistral streaming: {e}")

            # Check if it's a tool name validation error
            if "Function name" in str(e) and "must be a-z, A-Z, 0-9" in str(e):
                log.error(
                    f"Tool name validation error (this should not happen with universal compatibility): {e}"
                )
                log.error(
                    f"Request tools: {[t.get('function', {}).get('name') for t in request_params.get('tools', []) if t.get('type') == 'function']}"
                )

            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True,
            }

    async def _regular_completion(
        self, request_params: dict[str, Any], name_mapping: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Non-streaming completion using async execution with tool name restoration."""
        try:
            log.debug(f"Starting Mistral completion for model: {self.model}")

            # Convert messages to Mistral format (with async image downloading)
            request_params["messages"] = await self._convert_messages_to_mistral_format(
                request_params["messages"]
            )

            # Convert tools to dicts only at this final step (Pydantic-native until SDK call)
            if "tools" in request_params and request_params["tools"]:
                request_params["tools"] = self._tools_to_dicts(request_params["tools"])

            def _sync_completion():
                return self.client.chat.complete(**request_params)

            # Run sync call in thread to avoid blocking
            response = await asyncio.to_thread(_sync_completion)

            # Normalize response and restore tool names
            result = self._normalize_mistral_response(response, name_mapping)

            log.debug(
                f"Mistral completion result: "
                f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                f"tool_calls={len(result.get('tool_calls', []))}"
            )

            return result

        except Exception as e:
            log.error(f"Error in Mistral completion: {e}")

            # Check if it's a tool name validation error
            if "Function name" in str(e) and "must be a-z, A-Z, 0-9" in str(e):
                log.error(
                    f"Tool name validation error (this should not happen with universal compatibility): {e}"
                )
                log.error(
                    f"Request tools: {[t.get('function', {}).get('name') for t in request_params.get('tools', []) if t.get('type') == 'function']}"
                )

            return {"response": f"Error: {str(e)}", "tool_calls": [], "error": True}

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping
        self._current_name_mapping = {}
        # Mistral client doesn't require explicit cleanup
        pass
