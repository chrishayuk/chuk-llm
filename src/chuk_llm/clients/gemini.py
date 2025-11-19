"""
Modern Gemini Client
====================

Type-safe async client for Google Gemini using Pydantic V2.
Uses REST API directly for better control and performance.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from chuk_llm.clients.base import AsyncLLMClient
from chuk_llm.core import (
    CompletionRequest,
    CompletionResponse,
    ContentType,
    ErrorType,
    FinishReason,
    FunctionCall,
    LLMError,
    MessageRole,
    StreamChunk,
    ToolCall,
    ToolType,
    loads,
)

logger = logging.getLogger(__name__)


class GeminiClient(AsyncLLMClient):
    """
    Modern async client for Google Gemini.

    Uses Google's Generative Language API (REST) directly with httpx.

    Features:
    - Type-safe with Pydantic models
    - Fast JSON with orjson/ujson
    - Connection pooling with httpx
    - Zero-copy streaming
    - Proper error handling
    - Multimodal support (vision)

    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        api_key: Google API key
        **kwargs: Additional httpx client options
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        **kwargs: Any,
    ):
        """
        Initialize Gemini client.

        Args:
            model: Model name
            api_key: Google API key
            **kwargs: Additional client options
        """
        # Gemini uses API key as query parameter, not in headers
        # So we use a dummy base_url and override _post_json
        super().__init__(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            **kwargs,
        )
        self.model = model

        # Override headers - Gemini doesn't use Authorization header
        # API key is passed as query parameter
        if "Authorization" in self._client.headers:
            del self._client.headers["Authorization"]

        logger.info(f"Initialized Gemini client: model={model}")

    def _prepare_gemini_request(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Convert CompletionRequest to Gemini API format.

        Gemini API has different structure:
        - Uses "contents" instead of "messages"
        - Uses "parts" for message content
        - Uses "generationConfig" for parameters
        - Different parameter names (max_output_tokens vs max_tokens)

        Args:
            request: Pydantic completion request

        Returns:
            Dict in Gemini API format
        """
        # Convert messages to Gemini "contents" format
        contents = []
        system_instruction = None

        for msg in request.messages:
            # Extract system message separately (Gemini handles it differently)
            if msg.role == MessageRole.SYSTEM:
                if isinstance(msg.content, list):
                    # Extract text from content parts
                    text_parts = [
                        p.text for p in msg.content if p.type == ContentType.TEXT
                    ]
                    system_instruction = " ".join(text_parts)
                else:
                    system_instruction = msg.content
                continue

            # Convert role
            if msg.role == MessageRole.USER:
                role = "user"
            elif msg.role == MessageRole.ASSISTANT:
                role = "model"  # Gemini uses "model" instead of "assistant"
            elif msg.role == MessageRole.TOOL:
                # Tool responses handled specially
                role = "user"  # Gemini treats tool responses as user messages
            else:
                role = "user"

            # Convert content to parts
            parts: list[dict[str, Any]] = []
            if isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == ContentType.TEXT:
                        parts.append({"text": part.text})
                    elif part.type == ContentType.IMAGE_URL and part.image_url:
                        # Gemini vision format
                        # image_url can be string or dict
                        if isinstance(part.image_url, dict):
                            image_url = part.image_url.get("url", "")
                        else:
                            image_url = part.image_url

                        if image_url.startswith("data:"):
                            # Extract base64 data
                            if ";base64," in image_url:
                                mime_type, data = image_url.split(";base64,", 1)
                                mime_type = mime_type.replace("data:", "")
                                inline_data: dict[str, str] = {
                                    "mime_type": mime_type,
                                    "data": data,
                                }
                                parts.append({"inline_data": inline_data})
                        else:
                            # URL reference - not supported directly by Gemini
                            # Skip or convert to text
                            parts.append({"text": f"[Image: {image_url}]"})
            else:
                # Simple string content
                if msg.content:
                    parts.append({"text": msg.content})

            # Add tool calls if present
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    # Parse arguments JSON string to dict
                    args_dict = loads(tc.function.arguments)
                    func_call: dict[str, Any] = {
                        "name": tc.function.name,
                        "args": args_dict,
                    }
                    parts.append({"functionCall": func_call})

            contents.append({"role": role, "parts": parts})

        # Build generation config
        generation_config = {}

        if request.temperature is not None:
            # Gemini temperature range: 0.0-2.0
            generation_config["temperature"] = min(request.temperature, 2.0)

        if request.max_tokens is not None:
            # Gemini uses different parameter name
            generation_config["maxOutputTokens"] = request.max_tokens

        if request.top_p is not None:
            generation_config["topP"] = request.top_p

        # Build tools if present
        tools = None
        if request.tools:
            tools = []
            for tool in request.tools:
                if tool.type == ToolType.FUNCTION:
                    tools.append(
                        {
                            "functionDeclarations": [
                                {
                                    "name": tool.function.name,
                                    "description": tool.function.description or "",
                                    "parameters": tool.function.parameters or {},
                                }
                            ]
                        }
                    )

        # Build final request
        gemini_request: dict[str, Any] = {"contents": contents}

        if system_instruction:
            system_part: dict[str, str] = {"text": system_instruction}
            gemini_request["systemInstruction"] = {"parts": [system_part]}

        if generation_config:
            gemini_request["generationConfig"] = generation_config

        if tools:
            gemini_request["tools"] = tools

        return gemini_request

    def _parse_gemini_response(self, response: dict[str, Any]) -> CompletionResponse:
        """
        Parse Gemini API response to CompletionResponse.

        Gemini response structure:
        {
          "candidates": [{
            "content": {
              "parts": [
                {"text": "..."},
                {"functionCall": {"name": "...", "args": {...}}}
              ],
              "role": "model"
            },
            "finishReason": "STOP"
          }]
        }

        Args:
            response: Raw Gemini API response

        Returns:
            Pydantic CompletionResponse

        Raises:
            LLMError: On parsing errors
        """
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                raise LLMError(
                    error_type=ErrorType.API_ERROR.value,
                    error_message="No candidates in Gemini response",
                )

            candidate = candidates[0]
            content_obj = candidate.get("content", {})
            parts = content_obj.get("parts", [])

            # Extract text and tool calls from parts
            text_parts = []
            tool_calls = []

            for part in parts:
                # Text part
                if "text" in part:
                    text_parts.append(part["text"])

                # Function call part
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    import json

                    # Convert args dict to JSON string
                    args_str = json.dumps(fc.get("args", {}))

                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type=ToolType.FUNCTION,
                            function=FunctionCall(
                                name=fc.get("name", ""),
                                arguments=args_str,
                            ),
                        )
                    )

            # Combine text
            content = "".join(text_parts) if text_parts else None

            # Map finish reason
            finish_reason_map = {
                "STOP": FinishReason.STOP,
                "MAX_TOKENS": FinishReason.LENGTH,
                "SAFETY": FinishReason.CONTENT_FILTER,
                "RECITATION": FinishReason.CONTENT_FILTER,
                "OTHER": FinishReason.STOP,
            }
            gemini_reason = candidate.get("finishReason", "STOP")
            finish_reason = finish_reason_map.get(gemini_reason, FinishReason.STOP)

            # Build response
            return CompletionResponse(
                content=content,
                finish_reason=finish_reason,
                tool_calls=tool_calls,  # Already a list, empty if no tool calls
            )

        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Failed to parse Gemini response: {e}",
            ) from e

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Complete a request using Gemini.

        Args:
            request: Pydantic completion request

        Returns:
            Pydantic completion response

        Raises:
            LLMError: On API errors
        """
        # Prepare Gemini-format request
        gemini_request = self._prepare_gemini_request(request)

        # Build URL with API key
        url = f"{self.base_url}/models/{self.model}:generateContent"

        try:
            # Make request
            response = await self._client.post(
                url,
                json=gemini_request,
                params={"key": self.api_key},
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            return self._parse_gemini_response(data)

        except httpx.HTTPStatusError as e:
            error_data = {}
            with contextlib.suppress(Exception):
                error_data = e.response.json()

            error_message = error_data.get("error", {}).get("message", str(e))
            status_code = e.response.status_code

            # Map status codes to error types
            if status_code == 400:
                error_type_str = ErrorType.VALIDATION_ERROR.value
            elif status_code == 401 or status_code == 403:
                error_type_str = ErrorType.AUTHENTICATION_ERROR.value
            elif status_code == 429:
                error_type_str = ErrorType.RATE_LIMIT_ERROR.value
            elif status_code >= 500:
                error_type_str = ErrorType.SERVER_ERROR.value
            else:
                error_type_str = ErrorType.API_ERROR.value

            raise LLMError(
                error_type=error_type_str,
                error_message=f"Gemini API error (HTTP {status_code}): {error_message}",
            ) from e

        except httpx.RequestError as e:
            raise LLMError(
                error_type=ErrorType.NETWORK_ERROR.value,
                error_message=f"Gemini request failed: {e}",
            ) from e

        except LLMError:
            raise

        except Exception as e:
            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Gemini completion failed: {e}",
            ) from e

    async def stream(  # type: ignore[misc,override]
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a request using Gemini.

        Gemini uses Server-Sent Events (SSE) for streaming.

        Args:
            request: Pydantic completion request

        Yields:
            Stream chunks

        Raises:
            LLMError: On API errors
        """
        # Prepare Gemini-format request
        gemini_request = self._prepare_gemini_request(request)

        # Build URL with API key (streaming endpoint)
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent"

        try:
            # Make streaming request
            async with self._client.stream(
                "POST",
                url,
                json=gemini_request,
                params={"key": self.api_key, "alt": "sse"},
            ) as response:
                response.raise_for_status()

                # Parse SSE stream
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    # Extract JSON from SSE line
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break

                    try:
                        chunk_data = loads(json_str)

                        # Parse chunk similar to complete response
                        candidates = chunk_data.get("candidates", [])
                        if not candidates:
                            continue

                        candidate = candidates[0]
                        content_obj = candidate.get("content", {})
                        parts = content_obj.get("parts", [])

                        # Extract text from parts
                        for part in parts:
                            if "text" in part:
                                text = part["text"]
                                if text:
                                    yield StreamChunk(
                                        content=text,
                                    )

                    except Exception as e:
                        logger.warning(f"Failed to parse Gemini chunk: {e}")
                        continue

        except httpx.HTTPStatusError as e:
            error_data = {}
            with contextlib.suppress(Exception):
                error_data = e.response.json()

            error_message = error_data.get("error", {}).get("message", str(e))

            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Gemini streaming error (HTTP {e.response.status_code}): {error_message}",
            ) from e

        except LLMError:
            raise

        except Exception as e:
            raise LLMError(
                error_type=ErrorType.STREAMING_ERROR.value,
                error_message=f"Gemini streaming failed: {e}",
            ) from e
