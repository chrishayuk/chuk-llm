"""
Gemini Model Discoverer
~~~~~~~~~~~~~~~~~~~~~~~

Discovers available Gemini models from Google's Generative AI API.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseModelDiscoverer

log = logging.getLogger(__name__)


class GeminiModelDiscoverer(BaseModelDiscoverer):
    """Discoverer for Google Gemini models"""

    def __init__(
        self,
        provider_name: str = "gemini",
        api_key: str | None = None,
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        """
        Initialize Gemini discoverer.

        Args:
            provider_name: Name of the provider (default: "gemini")
            api_key: Google API key (required)
            api_base: API base URL
        """
        super().__init__(provider_name)
        self.api_key = api_key
        self.api_base = api_base

    async def discover_models(self) -> list[dict[str, Any]]:
        """
        Discover available Gemini models.

        Returns:
            List of model information dicts
        """
        if not self.api_key:
            log.warning(
                "No API key provided for Gemini discovery, returning fallback models"
            )
            return self._get_fallback_models()

        try:
            log.info(f"Discovering Gemini models from {self.api_base}")

            # Gemini API endpoint for listing models
            url = f"{self.api_base}/models?key={self.api_key}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                models = []
                for model_data in data.get("models", []):
                    model_name = model_data.get("name", "").replace("models/", "")

                    # Only include generative models (not embeddings, etc)
                    if not any(method in model_data.get("supportedGenerationMethods", [])
                              for method in ["generateContent", "streamGenerateContent"]):
                        continue

                    # Categorize model
                    model_info = self._categorize_model(model_name, model_data)
                    models.append(model_info)

                log.info(f"Discovered {len(models)} Gemini models")
                return models

        except Exception as e:
            log.error(f"Failed to discover Gemini models: {e}")
            return self._get_fallback_models()

    def _categorize_model(
        self, model_id: str, model_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Categorize a Gemini model with metadata"""
        model_lower = model_id.lower()

        # Determine capabilities
        is_vision = "vision" in model_lower or "pro" in model_lower or "flash" in model_lower
        is_code = "code" in model_lower

        # Extract version
        version = "unknown"
        if "gemini-2" in model_lower:
            version = "2.0"
        elif "gemini-1.5" in model_lower:
            version = "1.5"
        elif "gemini-1.0" in model_lower:
            version = "1.0"

        # Determine tier
        if "flash" in model_lower:
            tier = "fast"
        elif "pro" in model_lower:
            tier = "advanced"
        else:
            tier = "standard"

        model_info = {
            "name": model_id,
            "display_name": model_data.get("displayName", model_id),
            "description": model_data.get("description", ""),
            "source": "gemini_api",
            # Capabilities
            "is_vision": is_vision,
            "is_code": is_code,
            "supports_tools": True,  # Most Gemini models support function calling
            "supports_streaming": "streamGenerateContent" in model_data.get("supportedGenerationMethods", []),
            "supports_system_messages": True,
            # Model characteristics
            "version": version,
            "tier": tier,
            "model_family": "gemini",
            # Context limits
            "input_token_limit": model_data.get("inputTokenLimit", 32768),
            "output_token_limit": model_data.get("outputTokenLimit", 8192),
        }

        return model_info

    def _get_fallback_models(self) -> list[dict[str, Any]]:
        """Fallback model list with known Gemini models"""
        return [
            {
                "name": "gemini-2.0-flash",
                "display_name": "Gemini 2.0 Flash",
                "is_vision": True,
                "is_code": False,
                "supports_tools": True,
                "supports_streaming": True,
                "supports_system_messages": True,
                "version": "2.0",
                "tier": "fast",
                "model_family": "gemini",
                "input_token_limit": 1000000,
                "output_token_limit": 8192,
            },
            {
                "name": "gemini-2.5-pro",
                "display_name": "Gemini 2.5 Pro",
                "is_vision": True,
                "is_code": False,
                "supports_tools": True,
                "supports_streaming": True,
                "supports_system_messages": True,
                "version": "2.5",
                "tier": "advanced",
                "model_family": "gemini",
                "input_token_limit": 2000000,
                "output_token_limit": 8192,
            },
            {
                "name": "gemini-2.5-flash",
                "display_name": "Gemini 2.5 Flash",
                "is_vision": True,
                "is_code": False,
                "supports_tools": True,
                "supports_streaming": True,
                "supports_system_messages": True,
                "version": "2.5",
                "tier": "fast",
                "model_family": "gemini",
                "input_token_limit": 1000000,
                "output_token_limit": 8192,
            },
            {
                "name": "gemini-1.5-pro",
                "display_name": "Gemini 1.5 Pro",
                "is_vision": True,
                "is_code": False,
                "supports_tools": True,
                "supports_streaming": True,
                "supports_system_messages": True,
                "version": "1.5",
                "tier": "advanced",
                "model_family": "gemini",
                "input_token_limit": 2000000,
                "output_token_limit": 8192,
            },
            {
                "name": "gemini-1.5-flash",
                "display_name": "Gemini 1.5 Flash",
                "is_vision": True,
                "is_code": False,
                "supports_tools": True,
                "supports_streaming": True,
                "supports_system_messages": True,
                "version": "1.5",
                "tier": "fast",
                "model_family": "gemini",
                "input_token_limit": 1000000,
                "output_token_limit": 8192,
            },
        ]
