# chuk_llm/llm/discovery/anthropic_discoverer.py
"""
Anthropic/Claude model discoverer
"""

import logging
from typing import Any

import httpx

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory

log = logging.getLogger(__name__)


class AnthropicModelDiscoverer(BaseModelDiscoverer):
    """Discoverer for Anthropic Claude models"""

    def __init__(self, provider_name: str, api_key: str, **config):
        super().__init__(provider_name, **config)
        self.api_key = api_key
        self.api_base = "https://api.anthropic.com/v1"
        self.anthropic_version = config.get("anthropic_version", "2023-06-01")

    async def discover_models(self) -> list[dict[str, Any]]:
        """Discover models via Anthropic API"""
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base}/models", headers=headers
                )
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
                            "display_name": model_data.get("display_name"),
                            "created_at": model_data.get("created_at"),
                            "type": model_data.get("type"),
                            "source": "anthropic_api",
                            "provider_specific": self._get_anthropic_specifics(
                                model_id, model_data
                            ),
                        }
                    )

                return models

        except Exception as e:
            log.error(f"Failed to discover Anthropic models: {e}")
            return []

    def _get_anthropic_specifics(
        self, model_id: str, model_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get Anthropic-specific model characteristics"""
        model_lower = model_id.lower()

        characteristics = {
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False,
            "model_family": "unknown",
        }

        # Identify model family and capabilities
        if "opus-4" in model_lower:
            characteristics.update(
                {
                    "model_family": "opus_4",
                    "tier": "flagship",
                    "supports_vision": True,
                    "extended_thinking": True,
                    "estimated_context_length": 200000,
                    "max_output_tokens": 32000,
                }
            )
        elif "opus" in model_lower:
            characteristics.update(
                {
                    "model_family": "opus",
                    "tier": "flagship",
                    "supports_vision": "3-opus" in model_lower
                    or "3.5-opus" in model_lower,
                    "estimated_context_length": 200000,
                    "max_output_tokens": 4096,
                }
            )
        elif "sonnet-4" in model_lower:
            characteristics.update(
                {
                    "model_family": "sonnet_4",
                    "tier": "balanced",
                    "supports_vision": True,
                    "extended_thinking": True,
                    "estimated_context_length": 200000,
                    "max_output_tokens": 32000,
                }
            )
        elif "sonnet" in model_lower:
            characteristics.update(
                {
                    "model_family": "sonnet",
                    "tier": "balanced",
                    "supports_vision": "3-5" in model_lower or "3.5" in model_lower or "3.7" in model_lower,
                    "estimated_context_length": 200000,
                    "max_output_tokens": 8192 if "3.5" in model_lower or "3.7" in model_lower else 4096,
                }
            )
        elif "haiku" in model_lower:
            characteristics.update(
                {
                    "model_family": "haiku",
                    "tier": "fast",
                    "supports_vision": "3-5" in model_lower or "3.5" in model_lower,
                    "estimated_context_length": 200000,
                    "max_output_tokens": 4096,
                }
            )
        elif "claude-2" in model_lower:
            characteristics.update(
                {
                    "model_family": "claude_2",
                    "tier": "legacy",
                    "supports_vision": False,
                    "estimated_context_length": 100000,
                    "max_output_tokens": 4096,
                }
            )
        elif "instant" in model_lower:
            characteristics.update(
                {
                    "model_family": "instant",
                    "tier": "fast_legacy",
                    "supports_vision": False,
                    "estimated_context_length": 100000,
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
                "display_name": raw_model.get("display_name"),
                "type": raw_model.get("type"),
                "source": raw_model.get("source"),
                **provider_specifics,
            },
        )


# Register the discoverer
DiscovererFactory.register_discoverer("anthropic", AnthropicModelDiscoverer)
