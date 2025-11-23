"""
Moonshot AI (Kimi) model source - uses OpenAI-compatible API.
"""

from __future__ import annotations

from chuk_llm.core.constants import ApiBaseUrl
from chuk_llm.core.enums import Provider
from chuk_llm.registry.sources.openai_compatible import OpenAICompatibleSource


class MoonshotModelSource(OpenAICompatibleSource):
    """Moonshot AI (Kimi) model source using OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Moonshot model source.

        Args:
            api_key: Moonshot API key (defaults to MOONSHOT_API_KEY env var)
        """
        super().__init__(
            provider=Provider.MOONSHOT.value,
            api_base=ApiBaseUrl.MOONSHOT.value,
            api_key=api_key,
            api_key_env="MOONSHOT_API_KEY",
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

        # All current Moonshot models are chat models
        # Kimi K2 models: kimi-k2-*, kimi-k2-turbo-*, kimi-k2-thinking*
        # Legacy models: moonshot-v1-*, kimi-latest
        if model_lower.startswith(("kimi-", "moonshot-")):
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

        # Kimi K2 families
        if "kimi-k2-thinking" in model_lower:
            return "kimi-k2-thinking"
        elif "kimi-k2-turbo" in model_lower:
            return "kimi-k2-turbo"
        elif "kimi-k2" in model_lower:
            return "kimi-k2"

        # Legacy Kimi models
        if model_lower == "kimi-latest":
            return "kimi-latest"

        # Moonshot v1 models - extract version and size
        if model_lower.startswith("moonshot-v1-"):
            # Extract the variant (8k, 32k, 128k, auto, vision-preview, etc.)
            if "vision" in model_lower:
                # moonshot-v1-8k-vision-preview -> moonshot-v1-vision
                return "moonshot-v1-vision"
            elif "auto" in model_lower:
                return "moonshot-v1-auto"
            else:
                # moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k -> moonshot-v1
                import re

                match = re.match(r"(moonshot-v1)-\d+k", model_lower)
                if match:
                    return match.group(1)

        return None
