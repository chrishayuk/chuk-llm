"""
Gemini capability resolver - queries Google's models API.
"""

from __future__ import annotations

import os
from datetime import datetime

import httpx

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelCapabilities, ModelSpec, QualityTier
from chuk_llm.registry.resolvers.base import BaseCapabilityResolver


class GeminiCapabilityResolver(BaseCapabilityResolver):
    """
    Resolves capabilities from Gemini's models API.

    Gemini provides rich metadata including supported generation methods,
    input/output token limits, and supported features.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
    ):
        """
        Initialize Gemini capability resolver.

        Args:
            api_key: Google API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self.timeout = timeout
        self._cache: dict[str, ModelCapabilities] = {}

    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """
        Get capabilities from Gemini API.

        Args:
            spec: Model specification

        Returns:
            Model capabilities (empty if not Gemini or API fails)
        """
        if spec.provider != Provider.GEMINI.value:
            return self._empty_capabilities()

        if not self.api_key:
            return self._empty_capabilities()

        # Check cache
        if spec.name in self._cache:
            return self._cache[spec.name]

        try:
            # Query individual model metadata
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{spec.name}?key={self.api_key}",
                )
                response.raise_for_status()
                data = response.json()

                capabilities = self._parse_gemini_metadata(data, spec)
                self._cache[spec.name] = capabilities
                return capabilities

        except (httpx.HTTPError, httpx.ConnectError, KeyError):
            return self._empty_capabilities()

    def _parse_gemini_metadata(self, data: dict, spec: ModelSpec) -> ModelCapabilities:
        """
        Parse Gemini model metadata into capabilities.

        Args:
            data: Response from Gemini models API
            spec: Model specification

        Returns:
            Model capabilities
        """
        # Extract supported generation methods
        supported_methods = data.get("supportedGenerationMethods", [])
        supports_tools = "generateContent" in supported_methods

        # Extract token limits
        max_input_tokens = data.get("inputTokenLimit")
        max_output_tokens = data.get("outputTokenLimit")

        # Vision support - check if model supports images
        # This is indicated by the model name or supported methods
        supports_vision = "vision" in spec.name.lower() or "pro" in spec.name.lower()

        # Infer quality tier from model family
        quality_tier = self._infer_quality_tier(spec.name)

        return ModelCapabilities(
            max_context=max_input_tokens,
            max_output_tokens=max_output_tokens,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            supports_json_mode=True,  # Gemini supports JSON mode
            supports_streaming=True,  # Gemini supports streaming
            supports_system_messages=True,
            known_params={"temperature", "top_p", "top_k", "max_output_tokens"},
            quality_tier=quality_tier,
            source="gemini_api",
            last_updated=datetime.now().isoformat(),
        )

    def _infer_quality_tier(self, model_name: str) -> QualityTier:
        """Infer quality tier from Gemini model name."""
        name_lower = model_name.lower()

        if "pro" in name_lower:
            return QualityTier.BEST
        elif "flash" in name_lower or "lite" in name_lower:
            return QualityTier.CHEAP

        return QualityTier.BALANCED
