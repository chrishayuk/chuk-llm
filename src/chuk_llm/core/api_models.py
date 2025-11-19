"""
API Response Models
===================

Pydantic models for raw API responses.
Eliminates dict[str, Any] goop when parsing provider responses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# ================================================================
# OpenAI API Response Models
# ================================================================


class OpenAIMessageResponse(BaseModel):
    """OpenAI API message in response."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class OpenAIChoice(BaseModel):
    """OpenAI API choice."""

    index: int = 0
    message: OpenAIMessageResponse | None = None
    delta: dict[str, Any] | None = None  # For streaming
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class OpenAIUsage(BaseModel):
    """OpenAI API usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: dict[str, Any] | None = None
    prompt_tokens_details: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class OpenAIResponse(BaseModel):
    """OpenAI API completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None

    model_config = ConfigDict(extra="allow")


class OpenAIStreamChunk(BaseModel):
    """OpenAI API streaming chunk."""

    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]
    system_fingerprint: str | None = None

    model_config = ConfigDict(extra="allow")


# ================================================================
# Performance Info Model
# ================================================================


class PerformanceInfo(BaseModel):
    """JSON library performance information."""

    library: Literal["orjson", "ujson", "stdlib"]
    orjson_available: bool
    ujson_available: bool
    speedup: str

    model_config = ConfigDict(frozen=True)


# ================================================================
# Generic API Request/Response
# ================================================================


class APIRequest(BaseModel):
    """Generic API request."""

    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    response_format: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")
