"""
Modern Anthropic Client
========================

Fast, type-safe Anthropic client using Pydantic models and httpx.
Zero-copy streaming, proper async, no magic strings.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from chuk_llm.clients.base import AsyncLLMClient
from chuk_llm.core import (
    CompletionRequest,
    CompletionResponse,
    ErrorType,
    FinishReason,
    FunctionCall,
    ImageDataContent,
    ImageUrlContent,
    LLMError,
    Message,
    MessageRole,
    ModelInfo,
    Provider,
    RequestParam,
    ResponseKey,
    StreamChunk,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolType,
    dumps,
    loads,
)

logger = logging.getLogger(__name__)


class AnthropicClient(AsyncLLMClient):
    """
    Modern Anthropic-compatible client.

    Features:
    - Type-safe with Pydantic models
    - Fast JSON with orjson/ujson
    - Connection pooling with httpx
    - Zero-copy streaming
    - Proper error handling
    - Vision support (base64 + URLs)
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        **kwargs: Any,
    ):
        """
        Initialize Anthropic client.

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key
            base_url: API base URL
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.model = model
        self.provider = Provider.ANTHROPIC

        logger.info(f"Initialized Anthropic client: model={model}")

    def _get_default_headers(self) -> dict[str, str]:
        """Get Anthropic-specific headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _prepare_request(
        self, request: CompletionRequest
    ) -> tuple[dict[str, Any], str]:
        """
        Convert Pydantic request to Anthropic API format.

        Args:
            request: Validated completion request

        Returns:
            Tuple of (params dict, system message string)
        """
        # Anthropic uses separate system parameter
        system_message = ""
        messages = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                # Extract system content
                if isinstance(msg.content, str):
                    system_message = msg.content
                elif isinstance(msg.content, list):
                    # Join text parts
                    system_message = " ".join(
                        part.text
                        for part in msg.content
                        if isinstance(part, TextContent)
                    )
            else:
                messages.append(self._message_to_dict(msg))

        # Build Anthropic request
        params: dict[str, Any] = {
            RequestParam.MODEL.value: request.model or self.model,
            RequestParam.MESSAGES.value: messages,
        }

        if system_message:
            params["system"] = system_message

        # Add optional parameters
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        else:
            # Anthropic requires max_tokens
            params["max_tokens"] = 4096

        if request.temperature is not None:
            params[RequestParam.TEMPERATURE.value] = request.temperature

        if request.top_p is not None:
            params[RequestParam.TOP_P.value] = request.top_p

        if request.stop:
            params[RequestParam.STOP.value] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        if request.tools:
            # Convert to Anthropic tool format
            params[RequestParam.TOOLS.value] = [
                {
                    ResponseKey.NAME.value: tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters,
                }
                for tool in request.tools
            ]

        if request.stream:
            params[RequestParam.STREAM.value] = True

        return params, system_message

    def _message_to_dict(self, msg: Message) -> dict[str, Any]:
        """Convert Pydantic Message to Anthropic format."""
        msg_dict: dict[str, Any] = {ResponseKey.ROLE.value: self._map_role(msg.role)}

        if msg.content:
            if isinstance(msg.content, str):
                msg_dict[ResponseKey.CONTENT.value] = msg.content
            elif isinstance(msg.content, list):
                # Multimodal content
                content_blocks: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        content_blocks.append(
                            {ResponseKey.TYPE.value: "text", "text": part.text}
                        )
                    elif isinstance(part, ImageUrlContent):
                        # Convert to Anthropic image format
                        image_url = (
                            part.image_url
                            if isinstance(part.image_url, str)
                            else part.image_url.get("url", "")
                        )
                        # Anthropic requires base64, will need to download if URL
                        content_blocks.append(self._convert_image_url(image_url))
                    elif isinstance(part, ImageDataContent):
                        content_blocks.append(
                            {
                                ResponseKey.TYPE.value: "image",
                                "source": {
                                    ResponseKey.TYPE.value: "base64",
                                    "media_type": part.mime_type,
                                    "data": part.image_data,
                                },
                            }
                        )
                msg_dict[ResponseKey.CONTENT.value] = content_blocks

        if msg.tool_calls:
            # Anthropic tool use format
            tool_use_blocks = []
            for tc in msg.tool_calls:
                tool_use_blocks.append(
                    {
                        ResponseKey.TYPE.value: "tool_use",
                        ResponseKey.ID.value: tc.id,
                        ResponseKey.NAME.value: tc.function.name,
                        "input": loads(tc.function.arguments),
                    }
                )
            msg_dict[ResponseKey.CONTENT.value] = tool_use_blocks

        if msg.tool_call_id:
            # Tool result message
            msg_dict[ResponseKey.CONTENT.value] = [
                {
                    ResponseKey.TYPE.value: "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    ResponseKey.CONTENT.value: msg.content or "",
                }
            ]

        return msg_dict

    def _map_role(self, role: MessageRole) -> str:
        """Map MessageRole to Anthropic role."""
        if role == MessageRole.ASSISTANT:
            return "assistant"
        elif role == MessageRole.TOOL:
            return "user"  # Tool results go as user messages
        else:
            return "user"

    def _convert_image_url(self, url: str) -> dict[str, Any]:
        """
        Convert image URL to Anthropic format.

        For now, returns placeholder. In production, should download
        and convert to base64.
        """
        # TODO: Download image and convert to base64
        logger.warning(f"Image URL conversion not yet implemented: {url}")
        return {
            ResponseKey.TYPE.value: "image",
            "source": {
                ResponseKey.TYPE.value: "base64",
                "media_type": "image/jpeg",
                "data": "",  # Placeholder
            },
        }

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Create a non-streaming completion.

        Args:
            request: Validated completion request

        Returns:
            Validated completion response
        """
        params, _ = self._prepare_request(request)
        params[RequestParam.STREAM.value] = False

        logger.debug(
            f"Creating completion: model={params[RequestParam.MODEL.value]}, "
            f"messages={len(params[RequestParam.MESSAGES.value])}"
        )

        try:
            response = await self._post_json("/messages", params)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Completion failed: {str(e)}",
            ) from e

        return self._parse_completion_response(response)

    def _parse_completion_response(
        self, response: dict[str, Any]
    ) -> CompletionResponse:
        """Parse Anthropic API response to CompletionResponse."""
        content_blocks = response.get(ResponseKey.CONTENT.value, [])

        # Extract text content
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            block_type = block.get(ResponseKey.TYPE.value)

            if block_type == "text":
                text_content = block.get("text", "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get(
                            ResponseKey.ID.value, f"call_{uuid.uuid4().hex[:8]}"
                        ),
                        type=ToolType.FUNCTION,
                        function=FunctionCall(
                            name=block.get(ResponseKey.NAME.value, ""),
                            arguments=dumps(block.get("input", {})),
                        ),
                    )
                )

        # Extract usage
        usage = None
        if ResponseKey.USAGE.value in response:
            usage_data = response[ResponseKey.USAGE.value]
            usage = TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0),
            )

        # Parse finish reason
        stop_reason = response.get("stop_reason", "end_turn")
        finish_reason = self._map_finish_reason(stop_reason)

        return CompletionResponse(
            content=text_content if text_content else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=response.get(ResponseKey.MODEL.value),
        )

    def _map_finish_reason(self, stop_reason: str) -> FinishReason | str:
        """Map Anthropic stop_reason to FinishReason."""
        mapping = {
            "end_turn": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "stop_sequence": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
        }
        return mapping.get(stop_reason, stop_reason)

    async def stream(  # type: ignore[misc,override]
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Create a streaming completion.

        Args:
            request: Validated completion request

        Yields:
            Stream chunks with incremental content
        """
        params, _ = self._prepare_request(request)
        params[RequestParam.STREAM.value] = True

        logger.debug(
            f"Starting stream: model={params[RequestParam.MODEL.value]}, "
            f"messages={len(params[RequestParam.MESSAGES.value])}"
        )

        tool_use_acc: dict[str, dict[str, Any]] = {}

        try:
            async for chunk_bytes in self._stream_post("/messages", params):
                chunk_str = chunk_bytes.decode("utf-8").strip()

                if not chunk_str or chunk_str.startswith("event:"):
                    continue

                if chunk_str.startswith("data:"):
                    chunk_str = chunk_str[5:].strip()

                if chunk_str == "[DONE]":
                    break

                try:
                    chunk_data = loads(chunk_str)
                except Exception as e:
                    logger.debug(f"Failed to parse chunk: {e}")
                    continue

                stream_chunk = self._parse_stream_chunk(chunk_data, tool_use_acc)

                if stream_chunk:
                    yield stream_chunk

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                error_type=ErrorType.STREAMING_ERROR.value,
                error_message=f"Streaming failed: {str(e)}",
            ) from e

    def _parse_stream_chunk(
        self, chunk: dict[str, Any], tool_use_acc: dict[str, dict[str, Any]]
    ) -> StreamChunk | None:
        """Parse Anthropic streaming chunk."""
        chunk_type = chunk.get(ResponseKey.TYPE.value)

        if chunk_type == "content_block_delta":
            delta = chunk.get("delta", {})
            delta_type = delta.get(ResponseKey.TYPE.value)

            if delta_type == "text_delta":
                return StreamChunk(content=delta.get("text"))

            elif delta_type == "input_json_delta":
                # Tool use input accumulation
                index = chunk.get("index", 0)
                if index not in tool_use_acc:
                    tool_use_acc[index] = {"name": "", "input": ""}

                tool_use_acc[index]["input"] += delta.get("partial_json", "")

        elif chunk_type == "content_block_start":
            block = chunk.get("content_block", {})
            if block.get(ResponseKey.TYPE.value) == "tool_use":
                index = chunk.get("index", 0)
                tool_use_acc[index] = {
                    "id": block.get(ResponseKey.ID.value),
                    "name": block.get(ResponseKey.NAME.value),
                    "input": "",
                }

        elif chunk_type == "content_block_stop":
            # Tool use complete
            index = chunk.get("index", 0)
            if index in tool_use_acc and tool_use_acc[index].get("input"):
                try:
                    # Validate JSON is complete
                    loads(tool_use_acc[index]["input"])

                    tool_call = ToolCall(
                        id=tool_use_acc[index].get(
                            "id", f"call_{uuid.uuid4().hex[:8]}"
                        ),
                        type=ToolType.FUNCTION,
                        function=FunctionCall(
                            name=tool_use_acc[index]["name"],
                            arguments=tool_use_acc[index]["input"],
                        ),
                    )

                    return StreamChunk(tool_calls=[tool_call])
                except Exception:
                    # JSON not complete yet
                    pass

        elif chunk_type == "message_delta":
            delta = chunk.get("delta", {})
            if "stop_reason" in delta:
                finish_reason = self._map_finish_reason(delta["stop_reason"])
                return StreamChunk(finish_reason=finish_reason)

        return None

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            provider=self.provider.value,
            model=self.model,
            is_reasoning=False,
            supports_tools=True,
            supports_streaming=True,
            supports_vision="claude-3" in self.model.lower(),
        )
