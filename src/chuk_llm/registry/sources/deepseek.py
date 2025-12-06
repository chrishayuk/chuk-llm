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

    def _is_chat_model(self, model_id: str) -> bool:
        """
        Check if model is a chat completion model.

        Args:
            model_id: Model ID from API

        Returns:
            True if chat model
        """
        model_lower = model_id.lower()

        # DeepSeek chat models typically have "chat" in the name
        if "chat" in model_lower:
            return True

        # Include reasoner models (thinking mode)
        if "reasoner" in model_lower:
            return True

        # Also include versioned models (deepseek-v3, etc.) which are chat models
        if "deepseek-v" in model_lower and "coder" not in model_lower:
            return True

        return False

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID.

        Args:
            model_id: Model ID

        Returns:
            Model family name or None if not recognized
        """
        model_lower = model_id.lower()

        # Extract version-based families (deepseek-v3.2-chat -> deepseek-v3.2)
        if "deepseek-v" in model_lower:
            # Extract version part (e.g., deepseek-v3.2, deepseek-v3, deepseek-v2.5)
            import re

            match = re.match(r"(deepseek-v[\d.]+)", model_lower)
            if match:
                return match.group(1)

        # Known model families
        if model_lower.startswith("deepseek-"):
            # Return the model as its own family if it's a known base model
            if model_lower in ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]:
                return model_lower

        return None
