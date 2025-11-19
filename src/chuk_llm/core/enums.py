"""
Core Enumerations
=================

Type-safe enums for providers, features, roles, and other constants.
No more magic strings!
"""

from enum import Enum


class Provider(str, Enum):
    """LLM Provider enumeration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    GROQ = "groq"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    WATSONX = "watsonx"
    ADVANTAGE = "advantage"
    PERPLEXITY = "perplexity"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    OPENAI_COMPATIBLE = "openai_compatible"


class Feature(str, Enum):
    """Model feature capabilities."""

    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    SYSTEM_MESSAGES = "system_messages"
    PARALLEL_CALLS = "parallel_calls"
    REASONING = "reasoning"


class MessageRole(str, Enum):
    """Chat message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Completion finish reason."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ContentType(str, Enum):
    """Content part type for multimodal messages."""

    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_DATA = "image_data"


class ToolType(str, Enum):
    """Tool/function call types."""

    FUNCTION = "function"


class ReasoningGeneration(str, Enum):
    """Reasoning model generations."""

    O1 = "o1"
    O3 = "o3"
    O4 = "o4"
    O5 = "o5"
    GPT5 = "gpt5"
    UNKNOWN = "unknown"


class ResponsesTextFormatType(str, Enum):
    """Responses API text format types."""

    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
