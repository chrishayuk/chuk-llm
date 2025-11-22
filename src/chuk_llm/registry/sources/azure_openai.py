"""
Azure OpenAI model source - queries deployments API dynamically.
"""

from __future__ import annotations

import os

import httpx

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class AzureOpenAIModelSource(BaseModelSource):
    """
    Discovers Azure OpenAI deployments via the deployments API.

    This provides dynamic discovery of all available Azure OpenAI deployments,
    including custom deployment names that haven't been added to static tables yet.
    """

    def __init__(
        self,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-02-01",
        timeout: float = 10.0,
    ):
        """
        Initialize Azure OpenAI model source.

        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY env var)
            azure_endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            api_version: API version to use
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or ""
        self.azure_endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout

    async def discover(self) -> list[ModelSpec]:
        """
        Discover Azure OpenAI models via API.

        Note: Azure OpenAI distinguishes between base models (what's available)
        and deployments (your specific instances). The data plane API only
        provides base models. Custom deployment names can't be discovered without
        Management API access.

        Returns:
            List of ModelSpec objects for available Azure OpenAI chat models
        """
        if not self.api_key or not self.azure_endpoint:
            return []

        try:
            headers = {"api-key": self.api_key}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.azure_endpoint}/openai/models",
                    headers=headers,
                    params={"api-version": self.api_version},
                )
                response.raise_for_status()
                data = response.json()

                specs = []
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")
                    if not model_id:
                        continue

                    # Only include chat completion models
                    capabilities = model_data.get("capabilities", {})
                    if not capabilities.get("chat_completion", False):
                        continue

                    # Filter out non-chat models by ID
                    if self._is_chat_model(model_id):
                        family = self._extract_family(model_id)
                        specs.append(
                            ModelSpec(
                                provider=Provider.AZURE_OPENAI.value,
                                name=model_id,  # Base model ID
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
            "gpt-3",  # gpt-35-turbo, gpt-3.5-turbo
            "gpt-4",
            "gpt-5",
            "o1",
            "o3",
        ]

        return any(pattern in model_lower for pattern in include_patterns)

    def _extract_family(self, model_name: str) -> str | None:
        """
        Extract model family from model/deployment name.

        Args:
            model_name: Model or deployment name

        Returns:
            Model family or None
        """
        model_lower = model_name.lower()

        if "gpt-5" in model_lower or "gpt5" in model_lower:
            return "gpt-5"
        elif "gpt-4o" in model_lower or "gpt4o" in model_lower:
            return "gpt-4o"
        elif "gpt-4" in model_lower or "gpt4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower:
            return "gpt-3.5"
        elif (
            model_lower.startswith("o1")
            or "-o1-" in model_lower
            or "-o1" in model_lower
        ):
            return "o1"
        elif (
            model_lower.startswith("o3")
            or "-o3-" in model_lower
            or "-o3" in model_lower
        ):
            return "o3"

        return None
