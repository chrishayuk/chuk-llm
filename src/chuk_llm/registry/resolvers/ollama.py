"""
Ollama capability resolver.

Resolves capabilities by querying the Ollama API for model metadata.
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime

import httpx

from chuk_llm.registry.models import ModelCapabilities, ModelSpec, QualityTier
from chuk_llm.registry.resolvers.base import BaseCapabilityResolver


class OllamaCapabilityResolver(BaseCapabilityResolver):
    """
    Resolves capabilities from Ollama API metadata.

    Uses the /api/show endpoint to get model details.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 5.0):
        """
        Initialize Ollama capability resolver.

        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.timeout = timeout

    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """
        Get capabilities from Ollama API.

        Args:
            spec: Model specification

        Returns:
            Model capabilities (empty if not Ollama or API fails)
        """
        if spec.provider != "ollama":
            return self._empty_capabilities()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": spec.name},
                )
                response.raise_for_status()

                data = response.json()
                return self._parse_ollama_metadata(data, spec)

        except (httpx.HTTPError, httpx.ConnectError, KeyError):
            # Ollama not available or model not found
            return self._empty_capabilities()

    def _parse_ollama_metadata(self, data: dict, spec: ModelSpec) -> ModelCapabilities:
        """
        Parse Ollama metadata into capabilities.

        Args:
            data: Response from /api/show
            spec: Model specification

        Returns:
            Model capabilities
        """
        # Extract context length from parameters
        params = data.get("parameters", {})
        max_context = None

        if isinstance(params, dict):
            # Try num_ctx parameter
            max_context = params.get("num_ctx")
        elif isinstance(params, str):
            # Sometimes params is a string like "num_ctx 4096"
            for line in params.split("\n"):
                if "num_ctx" in line:
                    with contextlib.suppress(ValueError, IndexError):
                        max_context = int(line.split()[-1])

        # Check for vision support based on families
        families = data.get("details", {}).get("families", [])
        supports_vision = any(
            family in ["clip", "vision", "llava"]
            for family in (families if isinstance(families, list) else [])
        )

        # Ollama models generally don't support native tool calling
        # (unless using a custom adapter)
        supports_tools = False

        # Determine quality tier based on model size
        quality_tier = QualityTier.UNKNOWN
        model_size = data.get("details", {}).get("parameter_size", "")
        if isinstance(model_size, str):
            if "70b" in model_size.lower() or "65b" in model_size.lower():
                quality_tier = QualityTier.BALANCED
            elif any(size in model_size.lower() for size in ["7b", "8b", "13b"]):
                quality_tier = QualityTier.CHEAP

        return ModelCapabilities(
            max_context=max_context,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            supports_json_mode=False,  # Model-dependent
            supports_streaming=True,  # Ollama always supports streaming
            supports_system_messages=True,  # Most models support system messages
            known_params={"temperature", "top_p", "num_ctx", "num_predict"},
            quality_tier=quality_tier,
            source="ollama_api",
            last_updated=datetime.now().isoformat(),
        )
