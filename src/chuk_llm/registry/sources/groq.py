"""
Groq model source - uses OpenAI-compatible API.
"""

from __future__ import annotations

from chuk_llm.core.constants import ApiBaseUrl
from chuk_llm.core.enums import Provider
from chuk_llm.registry.sources.openai_compatible import OpenAICompatibleSource


class GroqModelSource(OpenAICompatibleSource):
    """Groq model source using OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Groq model source.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        super().__init__(
            provider=Provider.GROQ.value,
            api_base=ApiBaseUrl.GROQ.value,
            api_key=api_key,
            api_key_env="GROQ_API_KEY",
        )
