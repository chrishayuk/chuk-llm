"""
OpenAI Responses API Client
============================

Type-safe client for OpenAI's new Responses API.
Supports stateful conversations, built-in tools, and streaming.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from chuk_llm.core import (
    LLMError,
    ResponsesRequestParam,
    loads,
)
from chuk_llm.core.models import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesMessage,
    ResponsesOutputText,
)
from chuk_llm.core.model_capabilities import model_supports_parameter

logger = logging.getLogger(__name__)


class OpenAIResponsesClient:
    """
    Modern OpenAI Responses API client.

    Features:
    - Stateful conversations with previous_response_id
    - Built-in tools (web search, file search, computer use)
    - Background processing mode
    - Type-safe with Pydantic models
    - Fast JSON with orjson

    Note: This client is for the Responses API, not Chat Completions.
    It does not inherit from AsyncLLMClient because the API is fundamentally different.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_connections: int = 100,
        max_keepalive: int = 20,
    ):
        """
        Initialize OpenAI Responses API client.

        Args:
            model: Model name (e.g., "gpt-4.1", "gpt-5")
            api_key: OpenAI API key
            base_url: API base URL
            timeout: Request timeout in seconds
            max_connections: Maximum number of connections in pool
            max_keepalive: Maximum number of keep-alive connections
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        # Create async HTTP client with connection pooling
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        )

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=limits,
        )

        self._closed = False

        logger.info(f"Initialized OpenAI Responses API client: model={model}")

    def _messages_to_responses_input(
        self, messages: list
    ) -> list[dict[str, Any]]:
        """
        Convert Message objects to Responses API input format.

        Responses API uses different content type names:
        - "text" -> "input_text"
        - "image_url" -> "input_image" (with image_url field for URLs/data URLs)

        Args:
            messages: List of Message objects

        Returns:
            List of input items in Responses API format
        """
        from chuk_llm.core.enums import ContentType, MessageRole
        from chuk_llm.core.models import Message

        input_items = []

        for msg in messages:
            # Build content array
            content_parts = []

            if isinstance(msg.content, str):
                # Simple string content
                content_parts.append({"type": "input_text", "text": msg.content})
            elif isinstance(msg.content, list):
                # Multi-modal content (text + images)
                for part in msg.content:
                    if isinstance(part, dict):
                        # Already a dict
                        part_type = part.get("type")
                        if part_type == ContentType.TEXT.value:
                            content_parts.append(
                                {"type": "input_text", "text": part.get("text", "")}
                            )
                        elif part_type == ContentType.IMAGE_URL.value:
                            # Extract URL from image_url dict or string
                            image_url_data = part.get("image_url", {})
                            if isinstance(image_url_data, dict):
                                url = image_url_data.get("url", "")
                            else:
                                url = image_url_data
                            content_parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": url,
                                }
                            )
                    else:
                        # Pydantic object
                        if hasattr(part, "type") and part.type == ContentType.TEXT:
                            content_parts.append(
                                {"type": "input_text", "text": part.text}
                            )
                        elif hasattr(part, "type") and part.type == ContentType.IMAGE_URL:
                            # Extract URL from image_url dict or string
                            image_url_data = part.image_url
                            if isinstance(image_url_data, dict):
                                url = image_url_data.get("url", "")
                            else:
                                url = image_url_data
                            content_parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": url,
                                }
                            )

            # Wrap in message envelope
            input_items.append(
                {
                    "type": "message",
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": content_parts,
                }
            )

        return input_items

    async def create_response(self, request: ResponsesRequest) -> ResponsesResponse:
        """
        Create a model response using the Responses API.

        Args:
            request: Validated responses request

        Returns:
            Type-safe responses response

        Raises:
            LLMError: If the API request fails
        """
        # Build request body
        model_name = request.model or self.model

        # Convert messages to Responses API input format if provided
        if request.messages is not None:
            input_data = self._messages_to_responses_input(request.messages)
        elif request.input is not None:
            input_data = request.input
        else:
            raise ValueError("Either 'messages' or 'input' must be provided")

        body: dict[str, Any] = {
            ResponsesRequestParam.MODEL.value: model_name,
            ResponsesRequestParam.INPUT.value: input_data,
            ResponsesRequestParam.STORE.value: request.store,
        }

        # Add optional parameters
        if request.instructions is not None:
            body[ResponsesRequestParam.INSTRUCTIONS.value] = request.instructions

        if request.previous_response_id is not None:
            body[ResponsesRequestParam.PREVIOUS_RESPONSE_ID.value] = request.previous_response_id

        # Check model capabilities before adding parameters
        if request.temperature is not None and model_supports_parameter(model_name, "temperature"):
            body[ResponsesRequestParam.TEMPERATURE.value] = request.temperature

        if request.max_output_tokens is not None and model_supports_parameter(model_name, "max_tokens"):
            body[ResponsesRequestParam.MAX_OUTPUT_TOKENS.value] = request.max_output_tokens

        if request.max_tool_calls is not None:
            body[ResponsesRequestParam.MAX_TOOL_CALLS.value] = request.max_tool_calls

        if request.top_p is not None and model_supports_parameter(model_name, "top_p"):
            body[ResponsesRequestParam.TOP_P.value] = request.top_p

        if request.tools is not None:
            # Convert Tool objects to Responses API format (flat, not nested)
            body[ResponsesRequestParam.TOOLS.value] = [
                {
                    "type": tool.type.value,
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                }
                for tool in request.tools
            ]

        if request.tool_choice is not None:
            body[ResponsesRequestParam.TOOL_CHOICE.value] = request.tool_choice

        if request.parallel_tool_calls is not None:
            body[ResponsesRequestParam.PARALLEL_TOOL_CALLS.value] = request.parallel_tool_calls

        if request.text is not None:
            body[ResponsesRequestParam.TEXT.value] = {
                "format": {
                    "type": request.text.format.type,
                }
            }
            if request.text.format.name is not None:
                body[ResponsesRequestParam.TEXT.value]["format"]["name"] = request.text.format.name
            if request.text.format.schema_ is not None:
                body[ResponsesRequestParam.TEXT.value]["format"]["schema"] = request.text.format.schema_

        if request.reasoning is not None:
            body[ResponsesRequestParam.REASONING.value] = {}
            if request.reasoning.effort is not None:
                body[ResponsesRequestParam.REASONING.value]["effort"] = request.reasoning.effort
            if request.reasoning.summary is not None:
                body[ResponsesRequestParam.REASONING.value]["summary"] = request.reasoning.summary

        if request.metadata is not None:
            body[ResponsesRequestParam.METADATA.value] = request.metadata

        if request.background:
            body[ResponsesRequestParam.BACKGROUND.value] = request.background

        if request.truncation != "disabled":
            body[ResponsesRequestParam.TRUNCATION.value] = request.truncation

        # Make API request
        try:
            response = await self.client.post(
                f"{self.base_url}/responses",
                json=body,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            # Parse response
            return self._parse_response(data)

        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get("error", {}).get("message", str(e))
            logger.error(f"API error response: {error_data}")
            logger.error(f"Request body: {body}")
            raise LLMError(
                error_type="api_error",
                error_message=error_msg,
            ) from e

        except Exception as e:
            raise LLMError(
                error_type="unknown_error",
                error_message=str(e),
            ) from e

    async def stream_response(
        self, request: ResponsesRequest
    ) -> AsyncIterator[ResponsesResponse]:
        """
        Stream a model response using the Responses API.

        Args:
            request: Validated responses request

        Yields:
            Partial responses as they arrive

        Raises:
            LLMError: If the streaming request fails
        """
        # Build request body (same as create_response)
        model_name = request.model or self.model

        # Convert messages to Responses API input format if provided
        if request.messages is not None:
            input_data = self._messages_to_responses_input(request.messages)
        elif request.input is not None:
            input_data = request.input
        else:
            raise ValueError("Either 'messages' or 'input' must be provided")

        body: dict[str, Any] = {
            ResponsesRequestParam.MODEL.value: model_name,
            ResponsesRequestParam.INPUT.value: input_data,
            ResponsesRequestParam.STORE.value: request.store,
            ResponsesRequestParam.STREAM.value: True,
        }

        # Add optional parameters (check model capabilities)
        if request.instructions is not None:
            body[ResponsesRequestParam.INSTRUCTIONS.value] = request.instructions
        if request.previous_response_id is not None:
            body[ResponsesRequestParam.PREVIOUS_RESPONSE_ID.value] = request.previous_response_id
        if request.temperature is not None and model_supports_parameter(model_name, "temperature"):
            body[ResponsesRequestParam.TEMPERATURE.value] = request.temperature
        if request.max_output_tokens is not None and model_supports_parameter(model_name, "max_tokens"):
            body[ResponsesRequestParam.MAX_OUTPUT_TOKENS.value] = request.max_output_tokens
        if request.top_p is not None and model_supports_parameter(model_name, "top_p"):
            body[ResponsesRequestParam.TOP_P.value] = request.top_p

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/responses",
                json=body,
                headers=self._get_headers(),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or line.startswith(":"):
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = loads(data_str)
                            logger.debug(f"SSE chunk data: {chunk}")

                            # Responses API uses event-based streaming
                            # Events have format: {"type": "event.name", "response": {...}}
                            event_type = chunk.get("type", "")

                            # Extract response data from the event
                            if "response" in chunk:
                                # Events like response.created, response.in_progress have nested response
                                data = chunk["response"]
                            elif event_type.startswith("response.output_item"):
                                # Output item events - skip for now, could aggregate later
                                continue
                            else:
                                # Unknown event type, skip
                                continue

                            yield self._parse_response(data)

                        except Exception as e:
                            logger.warning(f"Failed to parse SSE chunk: {e}, data: {data_str[:200]}")
                            continue

        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get("error", {}).get("message", str(e))
            raise LLMError(
                error_type="api_error",
                error_message=error_msg,
            ) from e

        except Exception as e:
            raise LLMError(
                error_type="unknown_error",
                error_message=str(e),
            ) from e

    def _parse_response(self, data: dict[str, Any]) -> ResponsesResponse:
        """
        Parse API response into ResponsesResponse model.

        Args:
            data: Raw API response data

        Returns:
            Validated responses response
        """
        # Extract output_text from output array if present
        output_text = None
        if "output" in data and data["output"]:
            # Aggregate text from all output_text items
            text_parts = []
            for item in data["output"]:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            text_parts.append(content.get("text", ""))
            if text_parts:
                output_text = "".join(text_parts)

        # Build response
        response_data = {
            "id": data["id"],
            "object": data.get("object", "response"),
            "created_at": data["created_at"],
            "status": data["status"],
            "model": data["model"],
            "output": data.get("output", []),
            "output_text": output_text,
            "usage": data.get("usage"),
            "error": data.get("error"),
            "incomplete_details": data.get("incomplete_details"),
            "previous_response_id": data.get("previous_response_id"),
            "temperature": data.get("temperature"),
            "top_p": data.get("top_p"),
            "max_output_tokens": data.get("max_output_tokens"),
            "store": data.get("store", True),
            "reasoning": data.get("reasoning"),
            "metadata": data.get("metadata"),
        }

        return ResponsesResponse(**response_data)

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def retrieve_response(self, response_id: str) -> ResponsesResponse:
        """
        Retrieve a response by ID.

        Args:
            response_id: The ID of the response to retrieve

        Returns:
            The response object

        Raises:
            LLMError: If the API request fails
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/responses/{response_id}",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)

        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get("error", {}).get("message", str(e))
            raise LLMError(
                error_type="api_error",
                error_message=error_msg,
            ) from e

        except Exception as e:
            raise LLMError(
                error_type="unknown_error",
                error_message=str(e),
            ) from e

    async def delete_response(self, response_id: str) -> dict[str, Any]:
        """
        Delete a response by ID.

        Args:
            response_id: The ID of the response to delete

        Returns:
            Deletion confirmation

        Raises:
            LLMError: If the API request fails
        """
        try:
            response = await self.client.delete(
                f"{self.base_url}/responses/{response_id}",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get("error", {}).get("message", str(e))
            raise LLMError(
                error_type="api_error",
                error_message=error_msg,
            ) from e

        except Exception as e:
            raise LLMError(
                error_type="unknown_error",
                error_message=str(e),
            ) from e

    async def close(self):
        """Close the HTTP client and clean up resources."""
        if not self._closed:
            await self.client.aclose()
            self._closed = True
            logger.debug("OpenAI Responses API client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
