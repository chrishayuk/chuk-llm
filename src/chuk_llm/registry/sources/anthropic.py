"""
Anthropic Claude model source.

Note: Anthropic doesn't have a /v1/models endpoint, so we return
a curated list of known models. This is still cleaner than hardcoding
in the EnvProviderSource.
"""

from __future__ import annotations

import os

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class AnthropicModelSource(BaseModelSource):
    """
    Provides known Anthropic Claude models.

    Anthropic doesn't expose a models API endpoint, so this returns
    a curated list of available models.
    """

    # Known Anthropic models (updated as new models are released)
    KNOWN_MODELS = [
        ("claude-3-5-sonnet-20241022", "claude-3"),
        ("claude-3-5-sonnet-20240620", "claude-3"),
        ("claude-3-5-haiku-20241022", "claude-3"),
        ("claude-3-opus-20240229", "claude-3"),
        ("claude-3-sonnet-20240229", "claude-3"),
        ("claude-3-haiku-20240307", "claude-3"),
        ("claude-2.1", "claude-2"),
        ("claude-2.0", "claude-2"),
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize Anthropic model source.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    async def discover(self) -> list[ModelSpec]:
        """
        Discover Anthropic models.

        Returns known models if API key is available.

        Returns:
            List of ModelSpec objects for Anthropic models
        """
        if not self.api_key:
            return []

        specs = []
        for model_name, family in self.KNOWN_MODELS:
            specs.append(
                ModelSpec(
                    provider=Provider.ANTHROPIC.value,
                    name=model_name,
                    family=family,
                )
            )

        return self._deduplicate_specs(specs)
