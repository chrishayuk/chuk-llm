"""
Environment-based model source.

Discovers providers based on environment variables (API keys).
Returns default models for each detected provider.
"""

from __future__ import annotations

import os

from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class EnvProviderSource(BaseModelSource):
    """
    Discovers providers via environment variables.

    Checks for API key environment variables and returns default models
    for each available provider.
    """

    # Map of provider name -> (env_var, default_model, family)
    PROVIDER_DEFAULTS = {
        "openai": ("OPENAI_API_KEY", "gpt-4o-mini", "gpt-4o"),
        "anthropic": ("ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022", "claude-3"),
        "groq": ("GROQ_API_KEY", "llama-3.3-70b-versatile", "llama-3"),
        "deepseek": ("DEEPSEEK_API_KEY", "deepseek-chat", "deepseek"),
        "together": ("TOGETHER_API_KEY", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "llama-3"),
        "perplexity": ("PERPLEXITY_API_KEY", "llama-3.1-sonar-small-128k-online", "llama-3"),
        "mistral": ("MISTRAL_API_KEY", "mistral-small-latest", "mistral"),
        "gemini": ("GEMINI_API_KEY", "gemini-2.0-flash-exp", "gemini"),
        "watsonx": ("WATSONX_API_KEY", "ibm/granite-3-8b-instruct", "granite"),
        "azure_openai": ("AZURE_OPENAI_API_KEY", "gpt-4o", "gpt-4o"),
        "advantage": ("ADVANTAGE_API_KEY", "meta-llama/llama-3-3-70b-instruct", "llama-3"),
    }

    def __init__(self, include_ollama: bool = True):
        """
        Initialize environment provider source.

        Args:
            include_ollama: Whether to include Ollama (always available locally).
        """
        self.include_ollama = include_ollama

    async def discover(self) -> list[ModelSpec]:
        """
        Discover providers based on environment variables.

        Returns:
            List of ModelSpec objects for default models of available providers.
        """
        specs = []

        for provider, (env_var, default_model, family) in self.PROVIDER_DEFAULTS.items():
            if os.getenv(env_var):
                specs.append(
                    ModelSpec(
                        provider=provider,
                        name=default_model,
                        family=family,
                    )
                )

        # Ollama is always available if installed (no API key needed)
        if self.include_ollama:
            specs.append(
                ModelSpec(
                    provider="ollama",
                    name="llama3.2:latest",  # Common default
                    family="llama-3",
                )
            )

        return self._deduplicate_specs(specs)
