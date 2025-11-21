"""
Core Type System
================

Pydantic V2-based type-safe models and protocols for chuk-llm.

This module provides:
- Enums for type safety (no magic strings)
- Pydantic models for validation (with fast JSON via orjson/ujson)
- Protocol definitions for clients
"""

from .api_models import (
    APIRequest,
    OpenAIChoice,
    OpenAIMessageResponse,
    OpenAIResponse,
    OpenAIStreamChunk,
    OpenAIUsage,
    PerformanceInfo,
)
from .constants import (
    ConfigKey,
    ContentTypeValue,
    Default,
    EnvVar,
    ErrorType,
    HttpHeader,
    HttpMethod,
    HttpStatus,
    ModelPattern,
    OpenAIEndpoint,
    RequestParam,
    ResponseKey,
    ResponsesRequestParam,
    SSEEvent,
    SSEPrefix,
)
from .enums import (
    ContentType,
    Feature,
    FinishReason,
    MessageRole,
    Provider,
    ReasoningGeneration,
    ResponsesTextFormatType,
    ToolType,
)
from .json_utils import dumps, get_json_library, get_performance_info, loads
from .models import (
    AudioDataContent,
    AudioUrlContent,
    CompletionRequest,
    CompletionResponse,
    ContentPart,
    FunctionCall,
    ImageDataContent,
    ImageUrlContent,
    InputAudioContent,
    LLMError,
    Message,
    ResponseFormat,
    StreamChunk,
    TextContent,
    TokenUsage,
    Tool,
    ToolCall,
    ToolFunction,
    ToolParameter,
)
from .protocol import LLMClient, ModelInfo, SupportsFeatureDetection, SupportsModelInfo

__all__ = [
    # Enums
    "Provider",
    "Feature",
    "MessageRole",
    "FinishReason",
    "ContentType",
    "ToolType",
    "ReasoningGeneration",
    "ResponsesTextFormatType",
    # Constants
    "HttpMethod",
    "HttpHeader",
    "HttpStatus",
    "ContentTypeValue",
    "OpenAIEndpoint",
    "ErrorType",
    "ResponseKey",
    "RequestParam",
    "ResponsesRequestParam",
    "ConfigKey",
    "EnvVar",
    "SSEPrefix",
    "SSEEvent",
    "ModelPattern",
    "Default",
    # Models
    "Message",
    "TextContent",
    "ImageUrlContent",
    "ImageDataContent",
    "InputAudioContent",
    "AudioUrlContent",
    "AudioDataContent",
    "ContentPart",
    "ToolCall",
    "FunctionCall",
    "Tool",
    "ToolFunction",
    "ToolParameter",
    "TokenUsage",
    "ResponseFormat",
    "CompletionRequest",
    "CompletionResponse",
    "StreamChunk",
    "LLMError",
    # Protocols
    "LLMClient",
    "ModelInfo",
    "SupportsModelInfo",
    "SupportsFeatureDetection",
    # JSON utilities
    "dumps",
    "loads",
    "get_json_library",
    "get_performance_info",
    # API Models
    "OpenAIResponse",
    "OpenAIStreamChunk",
    "OpenAIChoice",
    "OpenAIMessageResponse",
    "OpenAIUsage",
    "PerformanceInfo",
    "APIRequest",
]
