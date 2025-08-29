# chuk_llm/llm/core/types.py
# chuk_llm/llm/core/types.py
import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string

    @field_validator("arguments", mode="before")
    @classmethod
    def validate_arguments(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        if isinstance(v, str):
            try:
                json.loads(v)  # Validate it's valid JSON
                return v
            except json.JSONDecodeError:
                return "{}"  # Fallback to empty object
        return "{}"  # Fallback for any other type


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str = "function"
    function: ToolCallFunction


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: bool = False
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_has_content(self):
        """Validate that response has content unless it's an error"""
        if not self.error:
            if not self.response and not self.tool_calls:
                raise ValueError("Response must have either text content or tool calls")
        return self


class StreamChunk(LLMResponse):
    """Streaming chunk with metadata"""

    chunk_index: int | None = None
    is_final: bool = False
    timestamp: float | None = None

    @model_validator(mode="after")
    def allow_empty_chunks(self):
        """Override parent validator - streaming chunks can be empty"""
        # StreamChunks are allowed to be empty, so we bypass the parent's validation
        return self


class ResponseValidator:
    """Validates and normalizes LLM responses"""

    @staticmethod
    def validate_response(
        raw_response: dict[str, Any], is_streaming: bool = False
    ) -> LLMResponse | StreamChunk:
        """Validate and convert raw response to typed model"""
        try:
            if is_streaming:
                return StreamChunk(**raw_response)
            else:
                return LLMResponse(**raw_response)
        except Exception as e:
            # Return error response if validation fails
            error_class = StreamChunk if is_streaming else LLMResponse
            try:
                return error_class(
                    response=None,
                    tool_calls=[],
                    error=True,
                    error_message=f"Response validation failed: {str(e)}",
                )
            except Exception:
                # If even the error response fails validation, use model_construct to bypass
                return error_class.model_construct(
                    response=None,
                    tool_calls=[],
                    error=True,
                    error_message=f"Response validation failed: {str(e)}",
                )

    @staticmethod
    def normalize_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCall]:
        """Normalize tool calls from different providers"""
        normalized = []

        for tc in raw_tool_calls:
            try:
                if isinstance(tc, dict):
                    # Handle different provider formats
                    if "function" in tc:
                        # OpenAI/Anthropic format
                        normalized.append(ToolCall(**tc))
                    elif "name" in tc:
                        # Alternative format - convert to standard
                        normalized.append(
                            ToolCall(
                                id=tc.get("id", f"call_{len(normalized)}"),
                                type="function",
                                function=ToolCallFunction(
                                    name=tc["name"], arguments=tc.get("arguments", "{}")
                                ),
                            )
                        )
            except Exception as e:
                logging.warning(f"Failed to normalize tool call: {tc}, error: {e}")
                continue

        return normalized
