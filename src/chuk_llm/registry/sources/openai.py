"""
OpenAI model source - queries /v1/models API dynamically.
"""

from __future__ import annotations

import os

import httpx

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class OpenAIModelSource(BaseModelSource):
    """
    Discovers OpenAI models via the /v1/models API.

    This provides dynamic discovery of all available OpenAI models,
    including new models that haven't been added to static tables yet.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 10.0,
    ):
        """
        Initialize OpenAI model source.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            api_base: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    async def discover(self) -> list[ModelSpec]:
        """
        Discover OpenAI models via API.

        Returns:
            List of ModelSpec objects for available OpenAI models
        """
        if not self.api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.api_base}/models",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                specs = []
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")
                    if not model_id:
                        continue

                    # Filter out non-chat models
                    if self._is_chat_model(model_id):
                        family = self._extract_family(model_id)
                        specs.append(
                            ModelSpec(
                                provider=Provider.OPENAI.value,
                                name=model_id,
                                family=family,
                            )
                        )

                return self._deduplicate_specs(specs)

        except (httpx.HTTPError, httpx.ConnectError):
            return []

    def _is_chat_model(self, model_id: str) -> bool:
        """
        Check if model is a chat completion model.

        Filters out embedding, moderation, TTS, whisper, etc.

        Args:
            model_id: Model ID from API

        Returns:
            True if chat model
        """
        # Exclude patterns
        exclude_patterns = [
            "embedding",
            "moderation",
            "whisper",
            "tts",
            "dall-e",
            "babbage",
            "davinci-002",  # Legacy completion models
            "ada",
            "curie",
        ]

        model_lower = model_id.lower()
        for pattern in exclude_patterns:
            if pattern in model_lower:
                return False

        # Include patterns for chat models
        include_patterns = [
            "gpt-3.5",
            "gpt-4",
            "gpt-5",
            "o1",
            "o3",
        ]

        return any(pattern in model_lower for pattern in include_patterns)

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID.

        Args:
            model_id: Model ID from API

        Returns:
            Model family or None
        """
        model_lower = model_id.lower()

        if "gpt-5" in model_lower:
            return "gpt-5"
        elif "gpt-4o" in model_lower:
            return "gpt-4o"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower:
            return "gpt-3.5"
        elif model_lower.startswith("o1"):
            return "o1"
        elif model_lower.startswith("o3"):
            return "o3"

        return None
