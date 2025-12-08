"""
LLM Client Protocol
===================

Clean, async-native protocol for LLM clients.
All implementations must follow this interface.
"""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .models import CompletionRequest, CompletionResponse, StreamChunk


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM clients.

    All provider clients must implement this interface for type safety
    and interchangeability.
    """

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Create a non-streaming completion.

        Args:
            request: Validated completion request

        Returns:
            Validated completion response

        Raises:
            LLMError: On API errors
        """
        ...

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """
        Create a streaming completion.

        Args:
            request: Validated completion request

        Yields:
            Stream chunks with incremental content

        Raises:
            LLMError: On API errors
        """
        ...

    async def close(self) -> None:
        """
        Cleanup resources (connections, pools, etc.).
        """
        ...


class ModelInfo(BaseModel):
    """Model information with capability flags."""

    provider: str
    model: str
    is_reasoning: bool = False

    # Feature support
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_system_messages: bool = True  # Most models support system messages

    # Parameter support (for reasoning models like GPT-5, O-series)
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_max_tokens: bool = True
    supports_frequency_penalty: bool = True
    supports_presence_penalty: bool = True
    supports_logit_bias: bool = True
    supports_logprobs: bool = True

    model_config = ConfigDict(frozen=True)


class SupportsModelInfo(Protocol):
    """Protocol for clients that provide model information."""

    def get_model_info(self) -> ModelInfo:
        """Get model information and capabilities."""
        ...


class SupportsFeatureDetection(Protocol):
    """Protocol for clients that support feature detection."""

    def supports_feature(self, feature: str) -> bool:
        """Check if model supports a specific feature."""
        ...
