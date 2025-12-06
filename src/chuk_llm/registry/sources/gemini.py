"""
Google Gemini model source.
"""

from __future__ import annotations

import os

import httpx

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class GeminiModelSource(BaseModelSource):
    """
    Discovers Google Gemini models via the API.

    Note: Google's models API returns all models, so we filter for generative models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
    ):
        """
        Initialize Gemini model source.

        Args:
            api_key: Google API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self.timeout = timeout

    async def discover(self) -> list[ModelSpec]:
        """
        Discover Gemini models via API.

        Returns:
            List of ModelSpec objects for available Gemini models
        """
        if not self.api_key:
            return []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}",
                )
                response.raise_for_status()
                data = response.json()

                specs = []
                for model_data in data.get("models", []):
                    name = model_data.get("name", "")
                    if not name:
                        continue

                    # Extract model name from full path (models/gemini-1.5-pro -> gemini-1.5-pro)
                    model_id = name.split("/", 1)[1] if "/" in name else name

                    # Filter for generative models only
                    if self._is_generative_model(model_id):
                        family = self._extract_family(model_id)
                        specs.append(
                            ModelSpec(
                                provider=Provider.GEMINI.value,
                                name=model_id,
                                family=family,
                            )
                        )

                return self._deduplicate_specs(specs)

        except (httpx.HTTPError, httpx.ConnectError, KeyError):
            return []

    def _is_generative_model(self, model_id: str) -> bool:
        """
        Check if model is a generative (chat/completion) model.

        Filters out embedding models and other non-generative models.

        Args:
            model_id: Model ID from API

        Returns:
            True if generative model
        """
        model_lower = model_id.lower()

        # Exclude patterns
        if "embedding" in model_lower or "aqa" in model_lower:
            return False

        # Include patterns for Gemini generative models
        return "gemini" in model_lower

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID.

        Args:
            model_id: Model ID from API

        Returns:
            Model family or None
        """
        model_lower = model_id.lower()

        # Check in descending version order to match correctly
        if "gemini-3" in model_lower:
            return "gemini-3"
        elif "gemini-2.5" in model_lower:
            return "gemini-2.5"
        elif "gemini-2" in model_lower:
            return "gemini-2"
        elif "gemini-1.5" in model_lower:
            return "gemini-1.5"
        elif "gemini-1" in model_lower:
            return "gemini-1"
        elif "gemini" in model_lower:
            return "gemini"

        return None
