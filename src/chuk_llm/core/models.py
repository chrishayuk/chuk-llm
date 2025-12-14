"""
Core Pydantic Models
====================

Type-safe, validated data models for all LLM interactions.
Uses Pydantic V2 for speed and validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import (
    ContentType,
    FinishReason,
    MessageRole,
    ResponsesTextFormatType,
    ToolType,
)
from .json_utils import loads

# ================================================================
# Content Types (Multimodal Support)
# ================================================================


class TextContent(BaseModel):
    """Text content part."""

    type: Literal[ContentType.TEXT] = ContentType.TEXT
    text: str

    model_config = ConfigDict(frozen=True)


class ImageUrlContent(BaseModel):
    """Image URL content part."""

    type: Literal[ContentType.IMAGE_URL] = ContentType.IMAGE_URL
    image_url: str | dict[str, str]

    model_config = ConfigDict(frozen=True)


class ImageDataContent(BaseModel):
    """Base64 image data content part."""

    type: Literal[ContentType.IMAGE_DATA] = ContentType.IMAGE_DATA
    image_data: str  # base64 encoded
    mime_type: str = "image/png"

    model_config = ConfigDict(frozen=True)


class InputAudioContent(BaseModel):
    """Audio input content part (OpenAI format)."""

    type: Literal[ContentType.INPUT_AUDIO] = ContentType.INPUT_AUDIO
    input_audio: dict[str, str]  # {"data": base64, "format": "wav"}

    model_config = ConfigDict(frozen=True)


class AudioUrlContent(BaseModel):
    """Audio URL content part."""

    type: Literal[ContentType.AUDIO_URL] = ContentType.AUDIO_URL
    audio_url: str | dict[str, str]  # URL or {"url": "...", "format": "wav"}

    model_config = ConfigDict(frozen=True)


class AudioDataContent(BaseModel):
    """Base64 audio data content part."""

    type: Literal[ContentType.AUDIO_DATA] = ContentType.AUDIO_DATA
    audio_data: str  # base64 encoded
    mime_type: str = "audio/wav"

    model_config = ConfigDict(frozen=True)


ContentPart = (
    TextContent
    | ImageUrlContent
    | ImageDataContent
    | InputAudioContent
    | AudioUrlContent
    | AudioDataContent
)


# ================================================================
# Tool/Function Calling
# ================================================================


class FunctionCall(BaseModel):
    """Function call information."""

    name: str
    arguments: str  # JSON string

    model_config = ConfigDict(frozen=True)

    @field_validator("arguments")
    @classmethod
    def validate_json_string(cls, v: str) -> str:
        """Ensure arguments is valid JSON string."""
        try:
            loads(v)  # Validate it's parseable with fast JSON
            return v
        except (ValueError, TypeError) as e:
            raise ValueError(f"arguments must be valid JSON string: {e}") from e


class ToolCall(BaseModel):
    """Tool call in a completion."""

    id: str
    type: Literal[ToolType.FUNCTION] = ToolType.FUNCTION
    function: FunctionCall

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> dict:
        """Convert to dict with enum values as strings."""
        # Use mode='json' to serialize enums to their values
        return self.model_dump(mode="json", exclude_none=True, by_alias=True)


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    type: str
    description: str | None = None
    enum: list[str] | None = None
    required: bool = False

    model_config = ConfigDict(extra="allow")  # Allow additional JSON schema fields


class ToolFunction(BaseModel):
    """Tool function definition."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object

    model_config = ConfigDict(frozen=True)


class Tool(BaseModel):
    """Tool definition for function calling."""

    type: Literal[ToolType.FUNCTION] = ToolType.FUNCTION
    function: ToolFunction

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> dict:
        """Convert to dict with enum values as strings."""
        # Use mode='json' to serialize enums to their values
        return self.model_dump(mode="json", exclude_none=True, by_alias=True)


# ================================================================
# Messages
# ================================================================


class Message(BaseModel):
    """Chat message with proper typing."""

    role: MessageRole
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages
    name: str | None = None  # For function/tool messages
    reasoning_content: str | None = (
        None  # For reasoning models (DeepSeek reasoner, etc.)
    )

    # Not frozen - needs to be mutable for reasoning_content updates
    # during conversation processing (e.g., DeepSeek reasoner tool calls)
    model_config = ConfigDict(frozen=False)

    @field_validator("content")
    @classmethod
    def validate_content(
        cls, v: str | list[ContentPart] | None
    ) -> str | list[ContentPart] | None:
        """Ensure content is valid."""
        if v is None:
            return v
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if not all(
                isinstance(
                    part,
                    (
                        TextContent,
                        ImageUrlContent,
                        ImageDataContent,
                        InputAudioContent,
                        AudioUrlContent,
                        AudioDataContent,
                    ),
                )
                for part in v
            ):
                raise ValueError("All content parts must be valid ContentPart types")
            return v
        raise ValueError("Content must be string, list of ContentParts, or None")

    def to_dict(self) -> dict:
        """Convert to dict with enum values as strings."""
        # Use mode='json' to serialize enums to their values
        return self.model_dump(mode="json", exclude_none=True, by_alias=True)


# ================================================================
# Token Usage
# ================================================================


class TokenUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None  # For reasoning models (o1, magistral, etc.)

    model_config = ConfigDict(frozen=True)


# ================================================================
# Response Format
# ================================================================


class ResponseFormat(BaseModel):
    """Response format specification for JSON mode."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None  # For structured output

    model_config = ConfigDict(frozen=True)


# ================================================================
# Requests
# ================================================================


class CompletionRequest(BaseModel):
    """Type-safe completion request."""

    messages: list[Message]
    model: str
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    tools: list[Tool] | None = None
    stream: bool = False
    response_format: ResponseFormat | dict[str, str] | None = None  # For JSON mode

    model_config = ConfigDict(
        frozen=True,
        extra="allow",  # Allow provider-specific params
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        """Ensure at least one message."""
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


# ================================================================
# Responses
# ================================================================


class CompletionResponse(BaseModel):
    """Type-safe completion response."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason | str
    usage: TokenUsage | None = None
    model: str | None = None

    model_config = ConfigDict(frozen=True)


class StreamChunk(BaseModel):
    """Streaming chunk with incremental data."""

    content: str | None = None  # Incremental text
    tool_calls: list[ToolCall] | None = None  # Complete tool calls when ready
    finish_reason: FinishReason | str | None = None
    usage: TokenUsage | None = None

    model_config = ConfigDict(frozen=True)


# ================================================================
# Error Handling
# ================================================================


class LLMError(BaseException):
    """Structured error information as an exception."""

    def __init__(
        self,
        error_type: str,
        error_message: str,
        retry_after: float | None = None,
    ):
        """Initialize LLM error."""
        self.error = True
        self.error_type = error_type
        self.error_message = error_message
        self.retry_after = retry_after
        super().__init__(f"{error_type}: {error_message}")

    def __str__(self) -> str:
        """String representation."""
        return f"{self.error_type}: {self.error_message}"


# ================================================================
# OpenAI Responses API Models
# ================================================================


class ResponsesInputText(BaseModel):
    """Input text content for Responses API."""

    type: Literal["input_text"] = "input_text"
    text: str

    model_config = ConfigDict(frozen=True)


class ResponsesInputImageUrl(BaseModel):
    """Input image URL for Responses API."""

    type: Literal["input_image_url"] = "input_image_url"
    image_url: str | dict[str, str]

    model_config = ConfigDict(frozen=True)


class ResponsesOutputText(BaseModel):
    """Output text content from Responses API."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class ResponsesMessage(BaseModel):
    """Message item in Responses API output."""

    type: Literal["message"] = "message"
    id: str
    status: str  # completed, in_progress, incomplete, failed
    role: str  # assistant, user
    content: list[ResponsesOutputText | dict[str, Any]]

    model_config = ConfigDict(frozen=True)


class ResponsesUsage(BaseModel):
    """Token usage for Responses API."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: dict[str, int] | None = None
    output_tokens_details: dict[str, int] | None = None

    model_config = ConfigDict(frozen=True)


class ResponsesError(BaseModel):
    """Error object from Responses API."""

    type: str
    message: str
    code: str | None = None
    param: str | None = None

    model_config = ConfigDict(frozen=True)


class ResponsesIncompleteDetails(BaseModel):
    """Details about why a response is incomplete."""

    reason: str  # max_output_tokens, max_tool_calls, etc.

    model_config = ConfigDict(frozen=True)


class ResponsesReasoningConfig(BaseModel):
    """Configuration for reasoning models (GPT-5, o-series)."""

    effort: str | None = None  # low, medium, high
    summary: str | None = None

    model_config = ConfigDict(frozen=True)


class ResponsesTextFormat(BaseModel):
    """Text format configuration."""

    type: ResponsesTextFormatType = ResponsesTextFormatType.TEXT
    name: str | None = None  # Required for json_schema type
    schema_: dict[str, Any] | None = Field(None, alias="schema")

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class ResponsesTextConfig(BaseModel):
    """Text response configuration."""

    format: ResponsesTextFormat

    model_config = ConfigDict(frozen=True)


class ResponsesRequest(BaseModel):
    """
    Type-safe Responses API request.

    Accepts same Message objects as CompletionRequest for consistency.
    The client converts them to Responses API format internally.
    """

    model: str
    # Accept either Message objects (like CompletionRequest) or raw input
    messages: list[Message] | None = None
    input: str | list[dict[str, Any]] | None = None  # Raw input (backward compat)
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool = True
    stream: bool = False
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(None, gt=0)
    max_tool_calls: int | None = None
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = True
    text: ResponsesTextConfig | None = None
    reasoning: ResponsesReasoningConfig | None = None
    metadata: dict[str, str] | None = None
    background: bool = False
    truncation: Literal["auto", "disabled"] = "disabled"

    model_config = ConfigDict(frozen=True, extra="allow")


class ResponsesResponse(BaseModel):
    """Type-safe Responses API response."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: str  # completed, failed, in_progress, cancelled, queued, incomplete
    model: str
    output: list[ResponsesMessage | dict[str, Any]]
    output_text: str | None = None  # SDK convenience property
    usage: ResponsesUsage | None = None
    error: ResponsesError | None = None
    incomplete_details: ResponsesIncompleteDetails | None = None
    previous_response_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    store: bool = True
    reasoning: ResponsesReasoningConfig | None = None
    metadata: dict[str, str] | None = None

    model_config = ConfigDict(frozen=True)
