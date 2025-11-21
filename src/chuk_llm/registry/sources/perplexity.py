"""
Perplexity model source.

Note: Perplexity API doesn't provide a /models endpoint,
so we use a static list of known models.
"""

from __future__ import annotations

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class PerplexityModelSource(BaseModelSource):
    """Perplexity model source with static model list."""

    # Perplexity doesn't have a /models endpoint, so we maintain a static list
    KNOWN_MODELS = [
        # Search Models
        "sonar",
        "sonar-pro",
        # Reasoning Models
        "sonar-reasoning",
        "sonar-reasoning-pro",
        # Research Model
        "sonar-deep-research",
        # Deprecated but may still work
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-huge-128k-online",
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize Perplexity model source.

        Args:
            api_key: Perplexity API key (not used for discovery, but kept for compatibility)
        """
        pass

    async def discover(self) -> list[ModelSpec]:
        """
        Return static list of Perplexity models.

        Returns:
            List of known Perplexity model specs
        """
        return self._get_known_models()

    def _get_known_models(self) -> list[ModelSpec]:
        """
        Get the static list of known Perplexity models.

        Returns:
            List of known Perplexity model specs
        """
        models = []
        for model_name in self.KNOWN_MODELS:
            family = self._extract_family(model_name)

            models.append(
                ModelSpec(
                    provider=Provider.PERPLEXITY.value,
                    name=model_name,
                    family=family,
                )
            )

        return models

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID.

        Args:
            model_id: Model ID

        Returns:
            Model family name or None if not recognized
        """
        model_lower = model_id.lower()

        # Sonar models
        if "sonar" in model_lower:
            if "reasoning" in model_lower:
                return "sonar-reasoning"
            elif "research" in model_lower:
                return "sonar-research"
            else:
                return "sonar"

        # Llama models
        if "llama" in model_lower:
            return "llama"

        return None
