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

    # Note: We no longer maintain a hardcoded model list!
    # Models are discovered dynamically via the capabilities script:
    #   python scripts/update_capabilities.py --provider anthropic
    #
    # This ensures the registry stays up-to-date without code changes

    def __init__(self, api_key: str | None = None):
        """
        Initialize Anthropic model source.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    async def discover(self) -> list[ModelSpec]:
        """
        Discover Anthropic models from capabilities cache.

        Models are discovered by running:
            python scripts/update_capabilities.py --provider anthropic

        Returns:
            List of ModelSpec objects for Anthropic models
        """
        if not self.api_key:
            return []

        # Load models from capabilities YAML
        from pathlib import Path

        import yaml

        capabilities_file = (
            Path(__file__).parent.parent / "capabilities" / "anthropic.yaml"
        )

        if not capabilities_file.exists():
            # No capabilities file - return empty list
            # User needs to run: python scripts/update_capabilities.py --provider anthropic
            return []

        with open(capabilities_file) as f:
            data = yaml.safe_load(f)

        specs = []
        for model_name, model_data in data.get("models", {}).items():
            family = model_data.get("inherits_from")
            specs.append(
                ModelSpec(
                    provider=Provider.ANTHROPIC.value,
                    name=model_name,
                    family=family,
                )
            )

        return self._deduplicate_specs(specs)
