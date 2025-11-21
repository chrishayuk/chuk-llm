"""
Ollama-specific model source.

Discovers local Ollama models via the Ollama API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class OllamaSource(BaseModelSource):
    """
    Discovers locally available Ollama models.

    Queries the Ollama API to get the list of installed models.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 5.0):
        """
        Initialize Ollama source.

        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.timeout = timeout

    async def discover(self) -> list[ModelSpec]:
        """
        Discover Ollama models via API.

        Returns:
            List of ModelSpec objects for installed Ollama models.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()

                data = response.json()
                models = data.get("models", [])

                specs = []
                for model in models:
                    name = model.get("name", "")
                    if not name:
                        continue

                    # Extract family from model details if available
                    family = self._extract_family(model)

                    specs.append(
                        ModelSpec(
                            provider=Provider.OLLAMA.value,
                            name=name,
                            family=family,
                        )
                    )

                return self._deduplicate_specs(specs)

        except (httpx.HTTPError, httpx.ConnectError):
            # Ollama not available - return empty list
            return []

    def _extract_family(self, model_data: dict[str, Any]) -> str | None:
        """
        Extract model family from Ollama model metadata.

        Args:
            model_data: Model data from Ollama API

        Returns:
            Model family or None
        """
        # Try to extract from model name
        name = model_data.get("name", "")

        # Common patterns
        if "llama" in name.lower():
            if "llama3" in name.lower() or "llama-3" in name.lower():
                return "llama-3"
            elif "llama2" in name.lower() or "llama-2" in name.lower():
                return "llama-2"
            return "llama"

        if "mistral" in name.lower():
            return "mistral"

        if "gemma" in name.lower():
            return "gemma"

        if "phi" in name.lower():
            return "phi"

        if "qwen" in name.lower():
            return "qwen"

        # Could also parse from model_data.get("details", {}).get("family")
        # but that's not always available

        return None
