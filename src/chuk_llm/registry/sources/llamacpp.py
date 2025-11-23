"""
llama.cpp model source - uses OpenAI-compatible API.
"""

from __future__ import annotations

from chuk_llm.core.enums import Provider
from chuk_llm.registry.sources.openai_compatible import OpenAICompatibleSource


class LlamaCppModelSource(OpenAICompatibleSource):
    """llama.cpp model source using OpenAI-compatible API."""

    def __init__(
        self, api_base: str = "http://localhost:8080", api_key: str | None = None
    ):
        """
        Initialize llama.cpp model source.

        Args:
            api_base: llama.cpp server base URL (default: http://localhost:8080)
            api_key: API key (not typically required for local llama.cpp server)
        """
        super().__init__(
            provider=Provider.LLAMA_CPP.value,
            api_base=api_base,
            api_key=api_key,
            api_key_env="LLAMA_CPP_API_KEY",  # Optional, for custom setups
            family_extractor=self._extract_family,  # Use our custom family extractor
        )

    def _is_chat_model(self, model_id: str) -> bool:
        """
        Check if model is a chat completion model.

        For llama.cpp, we assume all models can be used for chat if they have
        a chat template. Since we can't easily detect this from the model list,
        we'll be optimistic and return True for all models.

        Args:
            model_id: Model ID from API

        Returns:
            True (llama.cpp models are typically loaded for chat use)
        """
        return True

    def _extract_family(self, model_id: str) -> str | None:
        """
        Extract model family from model ID/filename.

        Args:
            model_id: Model ID (typically the filename)

        Returns:
            Model family name or None
        """
        model_lower = model_id.lower()

        # Common model families
        if "llama-3" in model_lower or "llama3" in model_lower:
            return "llama-3"
        elif "llama-2" in model_lower or "llama2" in model_lower:
            return "llama-2"
        elif "llama" in model_lower:
            return "llama"
        elif "mistral" in model_lower:
            return "mistral"
        elif "mixtral" in model_lower:
            return "mixtral"
        elif "qwen" in model_lower:
            return "qwen"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "gemma" in model_lower:
            return "gemma"
        elif "phi" in model_lower:
            return "phi"
        elif "command-r" in model_lower or "command_r" in model_lower:
            return "command-r"

        return None
