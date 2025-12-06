"""
Mistral model source - queries Mistral SDK for available models.
"""

from __future__ import annotations

import os

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class MistralModelSource(BaseModelSource):
    """
    Discovers Mistral models via the Mistral SDK.

    This provides dynamic discovery of all available Mistral models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
    ):
        """
        Initialize Mistral model source.

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.timeout = timeout

    async def discover(self) -> list[ModelSpec]:
        """
        Discover Mistral models via SDK.

        Returns:
            List of ModelSpec objects for available Mistral models
        """
        if not self.api_key:
            return []

        try:
            from mistralai import Mistral

            client = Mistral(api_key=self.api_key)

            # List available models
            response = client.models.list()

            specs = []
            if response.data:  # Check if data is not None
                for model in response.data:
                    model_id = model.id
                    if not model_id:
                        continue

                    # Filter out non-chat models
                    if self._is_chat_model(model_id):
                        family = self._extract_family(model_id)
                        specs.append(
                            ModelSpec(
                                provider=Provider.MISTRAL.value,
                                name=model_id,
                                family=family,
                            )
                        )

            return self._deduplicate_specs(specs)

        except Exception:
            return []

    def _is_chat_model(self, model_id: str) -> bool:
        """
        Check if model is a chat completion model.

        Filters out embedding, moderation, OCR, transcription models.

        Args:
            model_id: Model ID from API

        Returns:
            True if chat model
        """
        # Exclude patterns
        exclude_patterns = [
            "embed",
            "moderation",
            "ocr",
            "transcribe",
        ]

        model_lower = model_id.lower()
        return all(pattern not in model_lower for pattern in exclude_patterns)

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID.

        Args:
            model_id: Model ID from API

        Returns:
            Model family or None
        """
        model_lower = model_id.lower()

        # Mistral families - check more specific patterns first
        if "magistral" in model_lower:
            return "magistral"
        elif "codestral" in model_lower:
            return "codestral"
        elif "devstral" in model_lower:
            return "devstral"
        elif "voxtral" in model_lower:
            return "voxtral"
        elif "pixtral" in model_lower:
            return "pixtral"
        elif "ministral-3" in model_lower:
            return "ministral-3"
        elif "ministral" in model_lower:
            return "ministral"
        elif "mistral-large-3" in model_lower:
            return "mistral-large-3"
        elif "mistral-large" in model_lower:
            return "mistral-large"
        elif "mistral-medium" in model_lower:
            return "mistral-medium"
        elif "mistral-small" in model_lower:
            return "mistral-small"
        elif "mistral-tiny" in model_lower:
            return "mistral-tiny"
        elif "open-mistral" in model_lower:
            return "open-mistral"

        return None
