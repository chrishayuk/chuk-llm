# chuk_llm/llm/discovery/watsonx_discoverer.py
"""
IBM Watsonx model discoverer
"""

import logging
from typing import Any

import httpx

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory

log = logging.getLogger(__name__)


class WatsonxModelDiscoverer(BaseModelDiscoverer):
    """Discoverer for IBM Watsonx foundation models"""

    def __init__(self, provider_name: str, api_key: str, watsonx_url: str, **config):
        super().__init__(provider_name, **config)
        self.api_key = api_key
        self.watsonx_url = watsonx_url.rstrip("/")
        self.api_version = config.get("api_version", "2024-03-14")

    async def discover_models(self) -> list[dict[str, Any]]:
        """Discover models via Watsonx foundation models API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            models_url = (
                f"{self.watsonx_url}/ml/v4/foundation_model_specs"
                f"?version={self.api_version}"
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(models_url, headers=headers)
                response.raise_for_status()
                data = response.json()

                models = []
                for spec in data.get("resources", []):
                    model_id = spec.get("model_id")
                    if not model_id:
                        continue

                    models.append(
                        {
                            "name": model_id,
                            "label": spec.get("label"),
                            "provider": spec.get("provider"),
                            "source": "watsonx_api",
                            "provider_specific": self._get_watsonx_specifics(
                                model_id, spec
                            ),
                        }
                    )

                return models

        except Exception as e:
            log.error(f"Failed to discover Watsonx models: {e}")
            return []

    def _get_watsonx_specifics(
        self, model_id: str, spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Get Watsonx-specific model characteristics"""
        model_lower = model_id.lower()

        characteristics = {
            "supports_streaming": True,
            "supports_tools": False,
            "model_family": "unknown",
            "provider_family": spec.get("provider", "unknown"),
        }

        # IBM Granite models
        if "granite" in model_lower:
            characteristics.update(
                {
                    "model_family": "granite",
                    "tier": "enterprise",
                    "supports_tools": "3" in model_lower,  # Granite 3.x supports tools
                    "estimated_context_length": 128000 if "3" in model_lower else 8192,
                    "max_output_tokens": 8192 if "3" in model_lower else 4096,
                }
            )
        # Meta Llama models
        elif "llama" in model_lower:
            if "3.3" in model_lower or "3-3" in model_lower:
                characteristics.update(
                    {
                        "model_family": "llama_3_3",
                        "tier": "latest",
                        "supports_tools": True,
                        "estimated_context_length": 128000,
                        "max_output_tokens": 8192,
                    }
                )
            elif "3.2" in model_lower or "3-2" in model_lower:
                characteristics.update(
                    {
                        "model_family": "llama_3_2",
                        "tier": "efficient",
                        "supports_tools": True,
                        "supports_vision": "vision" in model_lower,
                        "estimated_context_length": 128000,
                        "max_output_tokens": 8192,
                    }
                )
            elif "3.1" in model_lower or "3-1" in model_lower:
                characteristics.update(
                    {
                        "model_family": "llama_3_1",
                        "tier": "reliable",
                        "supports_tools": True,
                        "estimated_context_length": 128000,
                        "max_output_tokens": 8192,
                    }
                )
            else:
                characteristics.update(
                    {
                        "model_family": "llama",
                        "estimated_context_length": 8192,
                        "max_output_tokens": 4096,
                    }
                )
        # Mistral models on Watsonx
        elif "mistral" in model_lower or "mixtral" in model_lower:
            characteristics.update(
                {
                    "model_family": "mistral",
                    "tier": "third_party",
                    "supports_tools": True,
                    "estimated_context_length": 32000,
                    "max_output_tokens": 4096,
                }
            )
        # Code Llama
        elif "codellama" in model_lower:
            characteristics.update(
                {
                    "model_family": "codellama",
                    "tier": "specialized",
                    "specialization": "code",
                    "estimated_context_length": 16384,
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
            family=provider_specifics.get("model_family", "unknown"),
            context_length=provider_specifics.get("estimated_context_length"),
            max_output_tokens=provider_specifics.get("max_output_tokens"),
            metadata={
                "label": raw_model.get("label"),
                "provider_family": raw_model.get("provider"),
                "source": raw_model.get("source"),
                **provider_specifics,
            },
        )


# Register the discoverer
DiscovererFactory.register_discoverer("watsonx", WatsonxModelDiscoverer)
