# chuk_llm/llm/discovery/mistral_discoverer.py
"""
Mistral AI model discoverer
"""

import logging
from typing import Any

import httpx

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory

log = logging.getLogger(__name__)


class MistralModelDiscoverer(BaseModelDiscoverer):
    """Discoverer for Mistral AI models"""

    def __init__(self, provider_name: str, api_key: str, **config):
        super().__init__(provider_name, **config)
        self.api_key = api_key
        self.api_base = config.get("api_base", "https://api.mistral.ai/v1")

    async def discover_models(self) -> list[dict[str, Any]]:
        """Discover models via Mistral API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/models", headers=headers)
                response.raise_for_status()
                data = response.json()

                models = []
                for model_data in data.get("data", []):
                    model_id = model_data.get("id")
                    if not model_id:
                        continue

                    models.append(
                        {
                            "name": model_id,
                            "created_at": model_data.get("created"),
                            "owned_by": model_data.get("owned_by"),
                            "object": model_data.get("object"),
                            "source": "mistral_api",
                            "provider_specific": self._get_mistral_specifics(model_id),
                        }
                    )

                return models

        except Exception as e:
            log.error(f"Failed to discover Mistral models: {e}")
            return []

    def _get_mistral_specifics(self, model_id: str) -> dict[str, Any]:
        """Get Mistral-specific model characteristics"""
        model_lower = model_id.lower()

        characteristics = {
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False,
            "model_family": "unknown",
        }

        # Magistral models (reasoning)
        if "magistral" in model_lower:
            characteristics.update(
                {
                    "model_family": "magistral",
                    "tier": "reasoning",
                    "reasoning_capable": True,
                    "estimated_context_length": 128000,
                    "max_output_tokens": 65536,
                }
            )
        # Mistral Large
        elif "mistral-large" in model_lower:
            characteristics.update(
                {
                    "model_family": "mistral_large",
                    "tier": "flagship",
                    "estimated_context_length": 128000,
                    "max_output_tokens": 4096,
                }
            )
        # Mistral Medium
        elif "mistral-medium" in model_lower or "medium" in model_lower:
            characteristics.update(
                {
                    "model_family": "mistral_medium",
                    "tier": "balanced",
                    "estimated_context_length": 32000,
                    "max_output_tokens": 4096,
                }
            )
        # Mistral Small
        elif "mistral-small" in model_lower or "small" in model_lower:
            characteristics.update(
                {
                    "model_family": "mistral_small",
                    "tier": "fast",
                    "estimated_context_length": 32000,
                    "max_output_tokens": 4096,
                }
            )
        # Mixtral models
        elif "mixtral" in model_lower:
            characteristics.update(
                {
                    "model_family": "mixtral",
                    "tier": "moe",
                    "estimated_context_length": 32000,
                    "max_output_tokens": 4096,
                }
            )
        # Pixtral (vision models)
        elif "pixtral" in model_lower:
            characteristics.update(
                {
                    "model_family": "pixtral",
                    "tier": "vision",
                    "supports_vision": True,
                    "estimated_context_length": 128000,
                    "max_output_tokens": 4096,
                }
            )
        # Codestral (coding models)
        elif "codestral" in model_lower or "devstral" in model_lower:
            characteristics.update(
                {
                    "model_family": "codestral",
                    "tier": "code",
                    "specialization": "code_generation",
                    "estimated_context_length": 32000,
                    "max_output_tokens": 4096,
                }
            )
        # Mistral Nemo
        elif "nemo" in model_lower:
            characteristics.update(
                {
                    "model_family": "mistral_nemo",
                    "tier": "efficient",
                    "estimated_context_length": 128000,
                    "max_output_tokens": 4096,
                }
            )

        return characteristics

    def normalize_model_data(self, raw_model: dict[str, Any]) -> DiscoveredModel:
        """Convert model data to DiscoveredModel"""
        provider_specifics = raw_model.get("provider_specific", {})

        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            created_at=raw_model.get("created_at"),
            family=provider_specifics.get("model_family", "unknown"),
            context_length=provider_specifics.get("estimated_context_length"),
            max_output_tokens=provider_specifics.get("max_output_tokens"),
            metadata={
                "owned_by": raw_model.get("owned_by"),
                "object": raw_model.get("object"),
                "source": raw_model.get("source"),
                **provider_specifics,
            },
        )


# Register the discoverer
DiscovererFactory.register_discoverer("mistral", MistralModelDiscoverer)
