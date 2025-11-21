"""
Generic OpenAI-compatible model source for providers using OpenAI API format.

This source can be used for any provider that implements the OpenAI /v1/models API,
including DeepSeek, Groq, Perplexity, Together, Anyscale, and others.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx

from chuk_llm.core.constants import ResponseKey
from chuk_llm.registry.models import ModelSpec
from chuk_llm.registry.sources.base import BaseModelSource


class OpenAICompatibleSource(BaseModelSource):
    """
    Generic source for OpenAI-compatible providers.

    Discovers models via the standard /v1/models endpoint that many
    providers implement for OpenAI compatibility.

    Example:
        ```python
        # DeepSeek
        class DeepSeekModelSource(OpenAICompatibleSource):
            def __init__(self, api_key: str | None = None):
                super().__init__(
                    provider="deepseek",
                    api_base="https://api.deepseek.com/v1",
                    api_key=api_key,
                    api_key_env="DEEPSEEK_API_KEY"
                )
        ```
    """

    def __init__(
        self,
        provider: str,
        api_base: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 10.0,
        model_filter: Callable[[str], bool] | None = None,
        family_extractor: Callable[[str], str | None] | None = None,
    ):
        """
        Initialize OpenAI-compatible model source.

        Args:
            provider: Provider name (e.g., "deepseek", "groq", "perplexity")
            api_base: API base URL (e.g., "https://api.deepseek.com/v1")
            api_key: API key (if None, uses api_key_env)
            api_key_env: Environment variable name for API key
            timeout: Request timeout in seconds
            model_filter: Optional function to filter models (return True to include)
            family_extractor: Optional function to extract model family from model ID
        """
        self.provider = provider
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.model_filter = model_filter
        self.family_extractor = family_extractor or self._default_family_extractor

        # Get API key - declare type as str | None
        self.api_key: str | None
        if api_key:
            self.api_key = api_key
        elif api_key_env:
            self.api_key = os.getenv(api_key_env)
        else:
            # Try common patterns
            env_patterns = [
                f"{provider.upper()}_API_KEY",
                f"{provider.replace('-', '_').upper()}_API_KEY",
            ]
            self.api_key = None
            for env_var in env_patterns:
                self.api_key = os.getenv(env_var)
                if self.api_key:
                    break

    async def discover(self) -> list[ModelSpec]:
        """
        Discover models via OpenAI-compatible /v1/models endpoint.

        Returns:
            List of ModelSpec objects for available models
        """
        if not self.api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.api_base}/models",
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                specs = []
                for model_data in data.get(ResponseKey.DATA.value, []):
                    model_id = model_data.get(ResponseKey.ID.value, "")
                    if not model_id:
                        continue

                    # Apply filter if provided
                    if self.model_filter and not self.model_filter(model_id):
                        continue

                    family = self.family_extractor(model_id)
                    specs.append(
                        ModelSpec(
                            provider=self.provider,
                            name=model_id,
                            family=family,
                        )
                    )

                return self._deduplicate_specs(specs)

        except (httpx.HTTPError, httpx.ConnectError):
            return []

    def _default_family_extractor(self, model_id: str) -> str | None:
        """
        Default family extraction - identifies common model families.

        Args:
            model_id: Model ID from API

        Returns:
            Model family or None
        """
        model_lower = model_id.lower()

        # Common families across providers
        if "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            return "gpt-3.5"
        elif "llama-3.3" in model_lower or "llama3.3" in model_lower:
            return "llama-3.3"
        elif "llama-3.2" in model_lower or "llama3.2" in model_lower:
            return "llama-3.2"
        elif "llama-3.1" in model_lower or "llama3.1" in model_lower:
            return "llama-3.1"
        elif "llama-3" in model_lower or "llama3" in model_lower:
            return "llama-3"
        elif "llama" in model_lower:
            return "llama"
        elif "mixtral" in model_lower:
            return "mixtral"
        elif "mistral" in model_lower:
            return "mistral"
        elif "gemma" in model_lower:
            return "gemma"
        elif "qwen" in model_lower:
            return "qwen"
        elif "deepseek-v3" in model_lower:
            return "deepseek-v3"
        elif "deepseek-v2" in model_lower:
            return "deepseek-v2"
        elif "deepseek-chat" in model_lower:
            return "deepseek-chat"
        elif "deepseek-reasoner" in model_lower or "deepseek-r" in model_lower:
            return "deepseek-reasoner"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "sonar" in model_lower:
            return "sonar"
        elif "claude" in model_lower:
            return "claude"
        elif "gemini" in model_lower:
            return "gemini"

        return None
