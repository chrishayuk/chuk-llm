"""
Modern OpenAI Client
====================

Fast, type-safe OpenAI client using Pydantic models and httpx.
Zero-copy streaming, proper async, no magic strings.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from chuk_llm.core import (
    APIRequest,
    CompletionRequest,
    CompletionResponse,
    ErrorType,
    FinishReason,
    FunctionCall,
    LLMError,
    Message,
    MessageRole,
    ModelInfo,
    OpenAIEndpoint,
    Provider,
    ReasoningGeneration,
    RequestParam,
    ResponseKey,
    SSEEvent,
    SSEPrefix,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolType,
    loads,
)
from chuk_llm.core.model_capabilities import get_model_capabilities, model_supports_parameter

from .base import AsyncLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(AsyncLLMClient):
    """
    Modern OpenAI-compatible client.

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
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            api_key: OpenAI API key
            base_url: API base URL
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.model = model
        self.provider = Provider.OPENAI

        # Detect reasoning models
        self._is_reasoning = self._detect_reasoning_model(model)
        self._reasoning_generation = (
            self._detect_generation(model) if self._is_reasoning else None
        )

        logger.info(
            f"Initialized OpenAI client: model={model}, "
            f"reasoning={self._is_reasoning}, generation={self._reasoning_generation}"
        )

    @staticmethod
    def _detect_reasoning_model(model: str) -> bool:
        """Check if model is a reasoning model."""
        model_lower = model.lower()
        return any(
            pattern in model_lower for pattern in ["o1-", "o3-", "o4-", "o5-", "gpt-5"]
        )

    @staticmethod
    def _detect_generation(model: str) -> ReasoningGeneration:
        """Detect reasoning model generation."""
        model_lower = model.lower()
        if "o1" in model_lower:
            return ReasoningGeneration.O1
        elif "o3" in model_lower:
            return ReasoningGeneration.O3
        elif "o4" in model_lower:
            return ReasoningGeneration.O4
        elif "o5" in model_lower:
            return ReasoningGeneration.O5
        elif "gpt-5" in model_lower:
            return ReasoningGeneration.GPT5
        return ReasoningGeneration.UNKNOWN

    def _prepare_request(self, request: CompletionRequest) -> APIRequest:
        """
        Convert Pydantic request to OpenAI API format.

        Args:
            request: Validated completion request

        Returns:
            Type-safe API request model
        """
        # Convert messages to OpenAI format
        messages = [self._message_to_dict(msg) for msg in request.messages]

        # Handle reasoning models
        if self._is_reasoning:
            messages = self._prepare_reasoning_messages(messages)

        # Build OpenAI request parameters
        params: dict[str, Any] = {
            RequestParam.MODEL.value: request.model or self.model,
            RequestParam.MESSAGES.value: messages,
        }

        # Get model name for capability checks
        model_name = request.model or self.model

        # Add optional parameters (check model capabilities)
        if request.temperature is not None and model_supports_parameter(model_name, "temperature"):
            params[RequestParam.TEMPERATURE.value] = request.temperature

        if request.max_tokens is not None and model_supports_parameter(model_name, "max_tokens"):
            # Reasoning models use max_completion_tokens instead of max_tokens
            if self._is_reasoning:
                params[RequestParam.MAX_COMPLETION_TOKENS.value] = request.max_tokens
            else:
                params[RequestParam.MAX_TOKENS.value] = request.max_tokens

        if request.top_p is not None and model_supports_parameter(model_name, "top_p"):
            params[RequestParam.TOP_P.value] = request.top_p

        if request.frequency_penalty is not None and model_supports_parameter(model_name, "frequency_penalty"):
            params[RequestParam.FREQUENCY_PENALTY.value] = request.frequency_penalty

        if request.presence_penalty is not None and model_supports_parameter(model_name, "presence_penalty"):
            params[RequestParam.PRESENCE_PENALTY.value] = request.presence_penalty

        if request.stop:
            params[RequestParam.STOP.value] = request.stop

        if request.tools:
            # Convert Pydantic tools to OpenAI format
            tools_supported = self._supports_tools()
            if tools_supported:
                params[RequestParam.TOOLS.value] = [
                    loads(tool.model_dump_json()) for tool in request.tools
                ]

        if request.response_format:
            params[RequestParam.RESPONSE_FORMAT.value] = request.response_format

        if request.stream:
            params[RequestParam.STREAM.value] = True

        # Return validated API request model
        return APIRequest(**params)

    def _supports_tools(self) -> bool:
        """Check if current model supports tools."""
        # O1 models don't support tools
        if self._reasoning_generation == ReasoningGeneration.O1:
            return False
        # All other models support tools
        return True

    def _message_to_dict(self, msg: Message) -> dict[str, Any]:
        """Convert Pydantic Message to dict."""
        msg_dict: dict[str, Any] = {ResponseKey.ROLE.value: msg.role.value}

        if msg.content:
            msg_dict[ResponseKey.CONTENT.value] = msg.content

        if msg.tool_calls:
            msg_dict[ResponseKey.TOOL_CALLS.value] = [
                {
                    ResponseKey.ID.value: tc.id,
                    ResponseKey.TYPE.value: ToolType.FUNCTION.value,
                    ResponseKey.FUNCTION.value: {
                        ResponseKey.NAME.value: tc.function.name,
                        ResponseKey.ARGUMENTS.value: tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        if msg.name:
            msg_dict[ResponseKey.NAME.value] = msg.name

        return msg_dict

    def _prepare_reasoning_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare messages for reasoning models (O1 doesn't support system messages)."""
        if self._reasoning_generation != ReasoningGeneration.O1:
            return messages

        # O1 models: convert system messages to user messages
        adjusted = []
        system_parts = []

        for msg in messages:
            if msg.get(ResponseKey.ROLE.value) == MessageRole.SYSTEM.value:
                system_parts.append(msg[ResponseKey.CONTENT.value])
            else:
                adjusted.append(msg)

        # Prepend system content to first user message
        if system_parts and adjusted:
            for i, msg in enumerate(adjusted):
                if msg.get(ResponseKey.ROLE.value) == MessageRole.USER.value:
                    combined = "\n".join(system_parts)
                    adjusted[i][ResponseKey.CONTENT.value] = (
                        f"System: {combined}\n\nUser: {msg[ResponseKey.CONTENT.value]}"
                    )
                    break

        return adjusted

    def _prepare_reasoning_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Remove unsupported parameters for reasoning models."""
        unsupported = [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]

        for param in unsupported:
            params.pop(param, None)

        return params

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Create a non-streaming completion.

        Args:
            request: Validated completion request

        Returns:
            Validated completion response
        """
        # Prepare OpenAI API request
        api_request_model = self._prepare_request(request)
        api_request = api_request_model.model_dump(exclude_none=True)
        api_request["stream"] = False

        logger.debug(
            f"Creating completion: model={api_request['model']}, "
            f"messages={len(api_request['messages'])}, "
            f"tools={len(api_request.get('tools', []))}"
        )

        # Make API request
        try:
            response = await self._post_json(
                OpenAIEndpoint.CHAT_COMPLETIONS.value, api_request
            )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                error_type=ErrorType.API_ERROR.value,
                error_message=f"Completion failed: {str(e)}",
            ) from e

        # Parse response
        return self._parse_completion_response(response)

    def _parse_completion_response(
        self, response: dict[str, Any]
    ) -> CompletionResponse:
        """Parse OpenAI API response to CompletionResponse."""
        choice = response[ResponseKey.CHOICES.value][0]
        message = choice[ResponseKey.MESSAGE.value]

        # Extract content
        content = message.get(ResponseKey.CONTENT.value)

        # Extract tool calls
        tool_calls = []
        if (
            ResponseKey.TOOL_CALLS.value in message
            and message[ResponseKey.TOOL_CALLS.value]
        ):
            for tc in message[ResponseKey.TOOL_CALLS.value]:
                tool_calls.append(
                    ToolCall(
                        id=tc[ResponseKey.ID.value],
                        type=ToolType.FUNCTION,
                        function=FunctionCall(
                            name=tc[ResponseKey.FUNCTION.value][ResponseKey.NAME.value],
                            arguments=tc[ResponseKey.FUNCTION.value][
                                ResponseKey.ARGUMENTS.value
                            ],
                        ),
                    )
                )

        # Extract usage
        usage = None
        if ResponseKey.USAGE.value in response:
            usage_data = response[ResponseKey.USAGE.value]
            # Extract reasoning tokens from completion_tokens_details if available
            reasoning_tokens = None
            if ResponseKey.COMPLETION_TOKENS_DETAILS.value in usage_data:
                details = usage_data[ResponseKey.COMPLETION_TOKENS_DETAILS.value]
                reasoning_tokens = details.get(ResponseKey.REASONING_TOKENS.value)

            usage = TokenUsage(
                prompt_tokens=usage_data.get(ResponseKey.PROMPT_TOKENS.value, 0),
                completion_tokens=usage_data.get(
                    ResponseKey.COMPLETION_TOKENS.value, 0
                ),
                total_tokens=usage_data.get(ResponseKey.TOTAL_TOKENS.value, 0),
                reasoning_tokens=reasoning_tokens,
            )

        # Parse finish reason
        finish_reason = choice.get(
            ResponseKey.FINISH_REASON.value, FinishReason.STOP.value
        )
        try:
            finish_reason_enum = FinishReason(finish_reason)
        except ValueError:
            finish_reason_enum = finish_reason  # type: ignore[assignment]

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason_enum,
            usage=usage,
            model=response.get(ResponseKey.MODEL.value),
        )

    async def stream(  # type: ignore[misc,override]
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Create a streaming completion.

        Zero-copy streaming: yields chunks immediately without accumulation.

        Args:
            request: Validated completion request

        Yields:
            Stream chunks with incremental content
        """
        # Prepare OpenAI API request
        api_request_model = self._prepare_request(request)
        api_request = api_request_model.model_dump(exclude_none=True)
        api_request["stream"] = True

        logger.debug(
            f"Starting stream: model={api_request['model']}, "
            f"messages={len(api_request['messages'])}"
        )

        # Track tool call accumulation
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        try:
            # Stream response - accumulate bytes for SSE parsing
            buffer = b""
            chunk_num = 0

            async for chunk_bytes in self._stream_post(
                OpenAIEndpoint.CHAT_COMPLETIONS.value, api_request
            ):
                # Accumulate bytes
                buffer += chunk_bytes

                # Process complete SSE events (delimited by double newline)
                while b"\n\n" in buffer:
                    event_bytes, buffer = buffer.split(b"\n\n", 1)
                    event_str = event_bytes.decode("utf-8").strip()

                    if not event_str:
                        continue

                    # Parse each line in the event
                    for line in event_str.split("\n"):
                        line = line.strip()

                        if not line or line.startswith(":"):
                            continue

                        if line == f"{SSEPrefix.DATA.value}{SSEEvent.DONE.value}":
                            continue

                        if line.startswith(SSEPrefix.DATA.value):
                            chunk_num += 1
                            data_str = line[len(SSEPrefix.DATA.value) :]

                            try:
                                chunk_data = loads(data_str)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to parse chunk #{chunk_num}: {e}, data: {data_str[:200]}"
                                )
                                continue

                            # Parse chunk
                            stream_chunk = self._parse_stream_chunk(
                                chunk_data, tool_calls_acc
                            )

                            # Yield only if there's content or complete tool calls
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
        self, chunk: dict[str, Any], tool_calls_acc: dict[int, dict[str, Any]]
    ) -> StreamChunk | None:
        """
        Parse streaming chunk.

        Zero-copy approach: only yield when we have complete data.
        Tool calls are accumulated until arguments are complete JSON.
        """
        if (
            ResponseKey.CHOICES.value not in chunk
            or not chunk[ResponseKey.CHOICES.value]
        ):
            return None

        choice = chunk[ResponseKey.CHOICES.value][0]

        # Extract delta
        delta = choice.get(ResponseKey.DELTA.value, {})

        # Content delta (yield immediately)
        content = delta.get(ResponseKey.CONTENT.value)

        # Tool calls (accumulate until complete)
        complete_tool_calls: list[ToolCall] | None = None

        if (
            ResponseKey.TOOL_CALLS.value in delta
            and delta[ResponseKey.TOOL_CALLS.value]
        ):
            for tc_delta in delta[ResponseKey.TOOL_CALLS.value]:
                idx = tc_delta.get(ResponseKey.INDEX.value, 0)

                # Initialize accumulator
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        ResponseKey.ID.value: tc_delta.get(
                            ResponseKey.ID.value, f"call_{uuid.uuid4().hex[:8]}"
                        ),
                        ResponseKey.NAME.value: "",
                        ResponseKey.ARGUMENTS.value: "",
                    }

                # Update ID
                if ResponseKey.ID.value in tc_delta:
                    tool_calls_acc[idx][ResponseKey.ID.value] = tc_delta[
                        ResponseKey.ID.value
                    ]

                # Update function data
                if ResponseKey.FUNCTION.value in tc_delta:
                    func = tc_delta[ResponseKey.FUNCTION.value]
                    if ResponseKey.NAME.value in func:
                        tool_calls_acc[idx][ResponseKey.NAME.value] = func[
                            ResponseKey.NAME.value
                        ]
                    if ResponseKey.ARGUMENTS.value in func:
                        tool_calls_acc[idx][ResponseKey.ARGUMENTS.value] += func[
                            ResponseKey.ARGUMENTS.value
                        ]

                # Check if JSON is complete
                if (
                    tool_calls_acc[idx][ResponseKey.NAME.value]
                    and tool_calls_acc[idx][ResponseKey.ARGUMENTS.value]
                ):
                    try:
                        # Try to parse - if succeeds, it's complete
                        loads(tool_calls_acc[idx][ResponseKey.ARGUMENTS.value])

                        # Mark as complete by converting to ToolCall
                        if complete_tool_calls is None:
                            complete_tool_calls = []

                        complete_tool_calls.append(
                            ToolCall(
                                id=tool_calls_acc[idx][ResponseKey.ID.value],
                                type=ToolType.FUNCTION,
                                function=FunctionCall(
                                    name=tool_calls_acc[idx][ResponseKey.NAME.value],
                                    arguments=tool_calls_acc[idx][
                                        ResponseKey.ARGUMENTS.value
                                    ],
                                ),
                            )
                        )
                    except Exception:
                        # JSON not complete yet
                        pass

        # Finish reason
        finish_reason_str = choice.get(ResponseKey.FINISH_REASON.value)
        finish_reason = None
        if finish_reason_str:
            try:
                finish_reason = FinishReason(finish_reason_str)
            except ValueError:
                finish_reason = finish_reason_str  # type: ignore[assignment]

        # Extract usage if present (final chunk may include usage)
        usage = None
        if ResponseKey.USAGE.value in chunk:
            usage_data = chunk[ResponseKey.USAGE.value]
            # Extract reasoning tokens from completion_tokens_details if available
            reasoning_tokens = None
            if ResponseKey.COMPLETION_TOKENS_DETAILS.value in usage_data:
                details = usage_data[ResponseKey.COMPLETION_TOKENS_DETAILS.value]
                reasoning_tokens = details.get(ResponseKey.REASONING_TOKENS.value)

            usage = TokenUsage(
                prompt_tokens=usage_data.get(ResponseKey.PROMPT_TOKENS.value, 0),
                completion_tokens=usage_data.get(
                    ResponseKey.COMPLETION_TOKENS.value, 0
                ),
                total_tokens=usage_data.get(ResponseKey.TOTAL_TOKENS.value, 0),
                reasoning_tokens=reasoning_tokens,
            )

        # Only yield if we have content or complete tool calls or usage
        if content or complete_tool_calls or finish_reason or usage:
            return StreamChunk(
                content=content,
                tool_calls=complete_tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )

        return None

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            provider=self.provider.value,
            model=self.model,
            is_reasoning=self._is_reasoning,
            supports_tools=self._supports_tools(),
            supports_streaming=True,
            supports_vision="vision" in self.model.lower()
            or "gpt-4" in self.model.lower(),
        )
