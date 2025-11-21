"""
OpenRouter model source - queries OpenRouter /models API.
"""

from __future__ import annotations

import os

import httpx

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class OpenRouterModelSource(BaseModelSource):
    """
    Discovers OpenRouter models via the /api/v1/models API.

    OpenRouter provides access to many different LLM providers through
    a unified API, so this source discovers models from multiple providers.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = "https://openrouter.ai/api/v1",
        timeout: float = 10.0,
    ):
        """
        Initialize OpenRouter model source.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            api_base: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    async def discover(self) -> list[ModelSpec]:
        """
        Discover OpenRouter models via API.

        Returns:
            List of ModelSpec objects for available OpenRouter models
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

                    # OpenRouter model IDs are like "openai/gpt-4", "anthropic/claude-3-opus"
                    family = self._extract_family(model_id)
                    specs.append(
                        ModelSpec(
                            provider=Provider.OPENROUTER.value,
                            name=model_id,
                            family=family,
                        )
                    )

                return self._deduplicate_specs(specs)

        except (httpx.HTTPError, httpx.ConnectError):
            return []

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID.

        OpenRouter uses format: provider/model-name
        We extract the base model family from the second part.

        Args:
            model_id: Model ID from API (e.g., "openai/gpt-4-turbo")

        Returns:
            Model family or None
        """
        # Split by / to get provider and model parts
        parts = model_id.split("/")
        if len(parts) < 2:
            return None

        model_name = parts[1].lower()

        # Common families
        if "gpt-4" in model_name:
            return "gpt-4"
        elif "gpt-3.5" in model_name:
            return "gpt-3.5"
        elif "claude-3" in model_name:
            if "opus" in model_name:
                return "claude-3-opus"
            elif "sonnet" in model_name:
                return "claude-3-sonnet"
            elif "haiku" in model_name:
                return "claude-3-haiku"
            return "claude-3"
        elif "claude" in model_name:
            return "claude"
        elif "llama-3" in model_name or "llama3" in model_name:
            return "llama-3"
        elif "llama" in model_name:
            return "llama"
        elif "gemini" in model_name:
            return "gemini"
        elif "mistral" in model_name:
            return "mistral"
        elif "mixtral" in model_name:
            return "mixtral"
        elif "qwen" in model_name:
            return "qwen"
        elif "deepseek" in model_name:
            return "deepseek"
        elif "gemma" in model_name:
            return "gemma"
        elif "phi" in model_name:
            return "phi"

        return None
