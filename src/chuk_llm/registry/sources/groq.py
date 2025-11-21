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

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID.

        Args:
            model_id: Model ID

        Returns:
            Model family name or None if not recognized
        """
        import re

        model_lower = model_id.lower()

        # Llama models (llama-3.3-70b-versatile -> llama-3.3, llama3.2 -> llama-3.2)
        if "llama" in model_lower:
            # Normalize llama3.2 to llama-3.2
            normalized = model_lower.replace("llama3.", "llama-3.")
            match = re.match(r"llama-(\d+(?:\.\d+)?)", normalized)
            if match:
                return f"llama-{match.group(1)}"
            # Handle llama3-70b -> llama-3
            match = re.match(r"llama(\d+)", normalized)
            if match:
                return f"llama-{match.group(1)}"

        # Gemma models (gemma-7b-it -> gemma)
        if model_lower.startswith("gemma"):
            return "gemma"

        # Mixtral models (mixtral-8x7b -> mixtral)
        if model_lower.startswith("mixtral"):
            return "mixtral"

        # Qwen models (qwen-2-7b -> qwen)
        if model_lower.startswith("qwen"):
            return "qwen"

        # DeepSeek models (deepseek-r1 -> deepseek)
        if model_lower.startswith("deepseek"):
            return "deepseek"

        return None
