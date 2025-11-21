"""
Ollama capability resolver.

Resolves capabilities by querying the Ollama API for model metadata.
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime

import httpx

from chuk_llm.core.enums import Provider
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
        if spec.provider != Provider.OLLAMA.value:
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
        Parse Ollama metadata into capabilities from GGUF model info.

        Args:
            data: Response from /api/show
            spec: Model specification

        Returns:
            Model capabilities
        """
        # Get model_info (GGUF metadata)
        model_info = data.get("model_info", {})
        details = data.get("details", {})

        # Extract context length from GGUF metadata (most reliable)
        max_context = None
        for key in ["llama.context_length", "context_length", "ctx_length"]:
            if key in model_info:
                max_context = model_info[key]
                break

        # Fallback to parameters if not in GGUF
        if not max_context:
            params = data.get("parameters", {})
            if isinstance(params, dict):
                max_context = params.get("num_ctx")
            elif isinstance(params, str):
                for line in params.split("\n"):
                    if "num_ctx" in line:
                        with contextlib.suppress(ValueError, IndexError):
                            max_context = int(line.split()[-1])

        # Check for vision support based on families
        families = details.get("families", [])
        supports_vision = any(
            family in ["clip", "vision", "llava", "minicpm"]
            for family in (families if isinstance(families, list) else [])
        )

        # Check for tool support from template
        # Models with tool templates support function calling
        template = data.get("template", "")
        supports_tools = (
            ".Tools" in template
            or "{{.Tools}}" in template
            or "tool" in template.lower()
        )

        # Check for JSON mode support (some models have it in template)
        supports_json_mode = "json" in template.lower()

        # Determine quality tier based on model size
        quality_tier = QualityTier.UNKNOWN
        model_size = details.get("parameter_size", "")
        if isinstance(model_size, str):
            size_lower = model_size.lower()
            if any(s in size_lower for s in ["70b", "65b", "72b"]):
                quality_tier = QualityTier.BEST
            elif any(s in size_lower for s in ["30b", "32b", "34b", "40b"]):
                quality_tier = QualityTier.BALANCED
            elif any(s in size_lower for s in ["7b", "8b", "13b", "14b"]):
                quality_tier = QualityTier.CHEAP

        return ModelCapabilities(
            max_context=max_context,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            supports_json_mode=supports_json_mode,
            supports_streaming=True,  # Ollama always supports streaming
            supports_system_messages=True,  # Most models support system messages
            known_params={"temperature", "top_p", "num_ctx", "num_predict", "top_k"},
            quality_tier=quality_tier,
            source="ollama_gguf",
            last_updated=datetime.now().isoformat(),
        )
