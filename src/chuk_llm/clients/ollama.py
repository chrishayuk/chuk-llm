"""
Ollama Client
=============

Modern Ollama client using the OpenAI-compatible API endpoint.
Supports local models with reasoning capabilities (gpt-oss, qwq, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from chuk_llm.core import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    Provider,
    ReasoningGeneration,
    RequestParam,
)

from .openai_compatible import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class OllamaClient(OpenAICompatibleClient):
    """
    Ollama client using OpenAI-compatible API.

    Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions,
    so we can reuse the OpenAI-compatible client with Ollama-specific enhancements.

    Features:
    - Local model support (no API key required)
    - Reasoning model detection (gpt-oss, qwq, marco-o1, deepseek-r1)
    - Thinking stream support for reasoning models
    - All standard OpenAI-compatible features
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",  # Ollama doesn't require real API key
        **kwargs: Any,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "qwen2.5", "llama3.2", "gpt-oss")
            base_url: Ollama server URL (default: http://localhost:11434/v1)
            api_key: Not required for Ollama, use default
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.provider = Provider.OLLAMA

        # Detect model family and reasoning capabilities
        self._model_family = self._detect_model_family(model)
        self._is_reasoning = self._detect_ollama_reasoning_model(model)

        logger.info(
            f"Initialized Ollama client: model={model}, family={self._model_family}, "
            f"reasoning={self._is_reasoning}, base_url={base_url}"
        )

    @staticmethod
    def _detect_model_family(model: str) -> str:
        """Detect model family for Ollama-specific optimizations."""
        model_lower = model.lower()

        # Check for common model families
        families = {
            "llama": "llama",
            "qwen": "qwen",
            "mistral": "mistral",
            "granite": "granite",
            "gemma": "gemma",
            "phi": "phi",
            "gpt-oss": "gpt-oss",
            "qwq": "qwq",
            "marco-o1": "marco-o1",
            "deepseek-r1": "deepseek-r1",
            "codellama": "code",
            "code": "code",
        }

        for pattern, family in families.items():
            if pattern in model_lower:
                return family

        return "unknown"

    @staticmethod
    def _detect_ollama_reasoning_model(model: str) -> bool:
        """
        Check if model is a reasoning model.

        Reasoning models output their thought process and may have
        specialized behavior for chain-of-thought reasoning.
        """
        reasoning_patterns = [
            "gpt-oss",
            "qwq",
            "marco-o1",
            "deepseek-r1",
            "reasoning",
            "think",
            "r1",
        ]
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in reasoning_patterns)

    def get_model_info(self) -> ModelInfo:
        """
        Get model information with Ollama-specific metadata.

        Returns:
            ModelInfo with Ollama capabilities
        """
        return ModelInfo(
            provider=self.provider.value,
            model=self.model,
            is_reasoning=self._is_reasoning,
            supports_tools=True,
            supports_streaming=True,
            supports_vision=False,  # Model-dependent, conservative default
            supports_temperature=True,
            supports_top_p=True,
            supports_max_tokens=True,
            supports_frequency_penalty=False,  # Not supported by Ollama
            supports_presence_penalty=False,  # Not supported by Ollama
            supports_logit_bias=False,  # Not supported by Ollama
            supports_logprobs=False,  # Not supported by Ollama
        )

    def _prepare_request(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Prepare request with Ollama-specific adjustments.

        Args:
            request: Validated completion request

        Returns:
            API request parameters
        """
        # Get base request from parent
        params = super()._prepare_request(request)

        # Ollama uses num_predict instead of max_tokens
        if RequestParam.MAX_TOKENS.value in params:
            params["num_predict"] = params.pop(RequestParam.MAX_TOKENS.value)

        # Remove unsupported parameters
        unsupported = [
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "user",
            "n",
            "best_of",
            "logprobs",
        ]
        for param in unsupported:
            params.pop(param, None)

        return params
