"""
Mistral Client
==============

Modern Mistral client using the OpenAI-compatible API.
Supports Mistral's Le Plateforme API with all features.
"""

from __future__ import annotations

import logging
from typing import Any

from chuk_llm.core import (
    CompletionRequest,
    ModelInfo,
    Provider,
    RequestParam,
)

from .openai_compatible import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class MistralClient(OpenAICompatibleClient):
    """
    Mistral Le Plateforme client.

    Mistral's API is OpenAI-compatible with some specific features:
    - Magistral reasoning models
    - Codestral for code generation
    - Ministral edge models
    - Pixtral for vision

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
        base_url: str = "https://api.mistral.ai/v1",
        **kwargs: Any,
    ):
        """
        Initialize Mistral client.

        Args:
            model: Model name (e.g., "mistral-large-latest", "ministral-8b-latest")
            api_key: Mistral API key
            base_url: Mistral API base URL
            **kwargs: Additional client options (timeout, max_connections, etc.)
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.provider = Provider.MISTRAL

        # Detect model capabilities
        self._is_code_model = self._is_codestral(model)
        self._is_edge_model = self._is_ministral(model)
        self._is_vision_model = self._is_pixtral(model)
        self._is_reasoning = self._is_magistral(model)

        logger.info(
            f"Initialized Mistral client: model={model}, "
            f"code={self._is_code_model}, edge={self._is_edge_model}, "
            f"vision={self._is_vision_model}, reasoning={self._is_reasoning}"
        )

    @staticmethod
    def _is_codestral(model: str) -> bool:
        """Check if model is Codestral (code generation)."""
        return "codestral" in model.lower() or "devstral" in model.lower()

    @staticmethod
    def _is_ministral(model: str) -> bool:
        """Check if model is Ministral (edge deployment)."""
        return "ministral" in model.lower()

    @staticmethod
    def _is_pixtral(model: str) -> bool:
        """Check if model is Pixtral (vision)."""
        return "pixtral" in model.lower()

    @staticmethod
    def _is_magistral(model: str) -> bool:
        """Check if model is Magistral (reasoning)."""
        return "magistral" in model.lower()

    def get_model_info(self) -> ModelInfo:
        """
        Get model information with Mistral-specific metadata.

        Returns:
            ModelInfo with Mistral capabilities
        """
        return ModelInfo(
            provider=self.provider.value,
            model=self.model,
            is_reasoning=self._is_reasoning,
            supports_tools=True,
            supports_streaming=True,
            supports_vision=self._is_vision_model,
            supports_temperature=True,
            supports_top_p=True,
            supports_max_tokens=True,
            supports_frequency_penalty=False,  # Not supported by Mistral
            supports_presence_penalty=False,  # Not supported by Mistral
            supports_logit_bias=False,  # Not supported by Mistral
            supports_logprobs=False,  # Not supported by Mistral
        )

    def _prepare_request(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Prepare request with Mistral-specific adjustments.

        Args:
            request: Validated completion request

        Returns:
            API request parameters
        """
        # Get base request from parent
        params = super()._prepare_request(request)

        # Remove unsupported parameters for Mistral
        unsupported = [
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "user",
            "n",
            "best_of",
            "logprobs",
            "top_k",
            "seed",
        ]
        for param in unsupported:
            params.pop(param, None)

        return params
