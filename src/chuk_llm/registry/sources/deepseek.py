"""
DeepSeek model source - uses OpenAI-compatible API.
"""

from __future__ import annotations

from chuk_llm.core.constants import ApiBaseUrl
from chuk_llm.core.enums import Provider
from chuk_llm.registry.sources.openai_compatible import OpenAICompatibleSource


class DeepSeekModelSource(OpenAICompatibleSource):
    """DeepSeek model source using OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize DeepSeek model source.

        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
        """
        super().__init__(
            provider=Provider.DEEPSEEK.value,
            api_base=ApiBaseUrl.DEEPSEEK.value,
            api_key=api_key,
            api_key_env="DEEPSEEK_API_KEY",
        )
