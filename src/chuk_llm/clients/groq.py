"""
Groq Client
===========

Modern Groq client using the OpenAI-compatible API.
Ultra-fast inference with large context windows.
"""

from __future__ import annotations

import logging
from typing import Any

from chuk_llm.core import (
    ModelInfo,
    Provider,
)

from .openai_compatible import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class GroqClient(OpenAICompatibleClient):
    """
    Groq ultra-fast inference client.

    Groq is fully OpenAI-compatible with some notable features:
    - Ultra-fast inference (500+ tokens/sec)
    - Large context windows (131k tokens for most models)
    - Support for Llama, DeepSeek, Qwen, and other models
    - Reasoning model support (DeepSeek-R1, GPT-OSS)

    Notable models:
    - llama-3.3-70b-versatile: 131k context, 32k output
    - llama-3.1-8b-instant: 131k context, 131k output
    - deepseek-r1-distill-llama-70b: Reasoning model
    - openai/gpt-oss-120b: GPT-OSS reasoning model

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
        base_url: str = "https://api.groq.com/openai/v1",
        **kwargs: Any,
    ):
        """
        Initialize Groq client.

        Args:
            model: Model name (e.g., "llama-3.3-70b-versatile")
            api_key: Groq API key
            base_url: Groq API base URL
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.provider = Provider.GROQ

        # Detect model capabilities
        self._is_reasoning = self._is_reasoning_groq_model(model)
        self._model_family = self._detect_groq_family(model)

        logger.info(
            f"Initialized Groq client: model={model}, "
            f"family={self._model_family}, reasoning={self._is_reasoning}"
        )

    @staticmethod
    def _is_reasoning_groq_model(model: str) -> bool:
        """Check if model is a reasoning model."""
        reasoning_patterns = [
            "deepseek-r1",
            "gpt-oss",
            "reasoning",
        ]
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in reasoning_patterns)

    @staticmethod
    def _detect_groq_family(model: str) -> str:
        """Detect model family."""
        model_lower = model.lower()

        families = {
            "llama-4": "llama4",
            "llama-guard": "llama-guard",
            "llama": "llama",
            "deepseek": "deepseek",
            "qwen": "qwen",
            "gpt-oss": "gpt-oss",
            "kimi": "kimi",
            "whisper": "whisper",
        }

        for pattern, family in families.items():
            if pattern in model_lower:
                return family

        return "unknown"

    def get_model_info(self) -> ModelInfo:
        """
        Get model information with Groq-specific metadata.

        Returns:
            ModelInfo with Groq capabilities
        """
        return ModelInfo(
            provider=self.provider.value,
            model=self.model,
            is_reasoning=self._is_reasoning,
            supports_tools=True,
            supports_streaming=True,
            supports_vision=False,  # Groq doesn't support vision currently
            supports_temperature=True,
            supports_top_p=True,
            supports_max_tokens=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
            supports_logit_bias=False,  # Not supported by Groq
            supports_logprobs=False,  # Not supported by Groq
        )
