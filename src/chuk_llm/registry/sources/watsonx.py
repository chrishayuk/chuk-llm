"""
WatsonX model source - returns known WatsonX models.

WatsonX doesn't have a simple models list API, so we use a known list.
"""

from __future__ import annotations

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class WatsonxModelSource(BaseModelSource):
    """
    Provides known WatsonX models.

    WatsonX requires project-specific configuration and doesn't have
    a simple public models list API, so we maintain a known list.
    """

    def __init__(self):
        """Initialize WatsonX model source."""
        pass

    async def discover(self) -> list[ModelSpec]:
        """
        Return known WatsonX models.

        Returns:
            List of ModelSpec objects for known WatsonX models
        """
        known_models = [
            # IBM Granite models
            ("ibm/granite-13b-chat-v2", "granite"),
            ("ibm/granite-13b-instruct-v2", "granite"),
            ("ibm/granite-20b-multilingual", "granite"),
            ("ibm/granite-3b-code-instruct", "granite-code"),
            ("ibm/granite-8b-code-instruct", "granite-code"),
            ("ibm/granite-20b-code-instruct", "granite-code"),
            ("ibm/granite-34b-code-instruct", "granite-code"),
            # Meta Llama models
            ("meta-llama/llama-3-70b-instruct", "llama-3"),
            ("meta-llama/llama-3-8b-instruct", "llama-3"),
            ("meta-llama/llama-2-70b-chat", "llama-2"),
            ("meta-llama/llama-2-13b-chat", "llama-2"),
            # Mixtral
            ("mistralai/mixtral-8x7b-instruct-v01", "mixtral"),
            # Google models
            ("google/flan-t5-xxl", "flan-t5"),
            ("google/flan-ul2", "flan-ul2"),
        ]

        return [
            ModelSpec(provider=Provider.WATSONX.value, name=name, family=family)
            for name, family in known_models
        ]
