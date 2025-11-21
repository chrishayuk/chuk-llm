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
            # IBM Granite 3.x models (Latest - 2025)
            ("ibm/granite-4-h-small", "granite-4"),
            ("ibm/granite-3-3-8b-instruct", "granite-3"),
            ("ibm/granite-3-2-8b-instruct", "granite-3"),
            ("ibm/granite-vision-3-2-2b", "granite-vision"),
            ("ibm/granite-3-2b-instruct", "granite-3"),
            ("ibm/granite-3-8b-instruct", "granite-3"),
            # IBM Granite Guardian (Content Safety)
            ("ibm/granite-guardian-3-8b", "granite-guardian"),
            ("ibm/granite-guardian-3-2b", "granite-guardian"),
            # IBM Granite Code models
            ("ibm/granite-8b-code-instruct", "granite-code"),
            ("ibm/granite-3b-code-instruct", "granite-code"),
            ("ibm/granite-20b-code-instruct", "granite-code"),
            ("ibm/granite-34b-code-instruct", "granite-code"),
            # IBM Granite Time Series
            ("ibm/granite-timeseries-ttm-r2", "granite-timeseries"),
            # IBM Granite Legacy (v2)
            ("ibm/granite-13b-chat-v2", "granite-2"),
            ("ibm/granite-13b-instruct-v2", "granite-2"),
            ("ibm/granite-20b-multilingual", "granite-2"),
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
