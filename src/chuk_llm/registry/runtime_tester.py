"""
Runtime capability tester for dynamically discovered models.

This module provides lightweight capability testing that can be performed
at runtime when unknown models are discovered, without requiring the full
update_capabilities.py script to be run.
"""

from __future__ import annotations

import logging
from datetime import datetime

from chuk_llm.core.constants import CapabilityKey
from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelCapabilities, QualityTier
from chuk_llm.registry.testing import (
    test_json_mode,
    test_streaming,
    test_tools,
    test_vision,
)

log = logging.getLogger(__name__)


class RuntimeCapabilityTester:
    """
    Lightweight capability tester for runtime model discovery.

    Tests essential capabilities quickly to enable immediate use of
    newly discovered models without waiting for offline capability updates.
    """

    def __init__(self, provider: str):
        """
        Initialize runtime tester.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
        """
        self.provider = provider

    async def test_model(self, model_name: str) -> ModelCapabilities:
        """
        Test model capabilities at runtime.

        Performs quick essential tests:
        - Tools
        - Vision
        - JSON mode
        - Streaming

        Args:
            model_name: Model to test

        Returns:
            ModelCapabilities with tested values
        """
        log.info(f"Runtime testing capabilities for {self.provider}/{model_name}")

        client = self._get_client(model_name)

        # Use shared testing functions
        capabilities = {
            CapabilityKey.SUPPORTS_TOOLS.value: await test_tools(client),
            CapabilityKey.SUPPORTS_VISION.value: await test_vision(client),
            CapabilityKey.SUPPORTS_JSON_MODE.value: await test_json_mode(client),
            CapabilityKey.SUPPORTS_STREAMING.value: await test_streaming(client),
            CapabilityKey.SUPPORTS_SYSTEM_MESSAGES.value: True,  # Assume true for modern models
            "quality_tier": QualityTier.UNKNOWN,
            "source": "runtime_test",
            "last_updated": datetime.now().isoformat(),
        }

        log.info(
            f"Runtime test complete for {self.provider}/{model_name}: "
            f"tools={capabilities[CapabilityKey.SUPPORTS_TOOLS.value]}, "
            f"vision={capabilities[CapabilityKey.SUPPORTS_VISION.value]}, "
            f"json={capabilities[CapabilityKey.SUPPORTS_JSON_MODE.value]}, "
            f"streaming={capabilities[CapabilityKey.SUPPORTS_STREAMING.value]}"
        )

        return ModelCapabilities(**capabilities)

    def _get_client(self, model_name: str):
        """Get provider client for testing."""
        # Import here to avoid circular dependencies
        if self.provider == Provider.OPENAI.value:
            from chuk_llm.llm.providers.openai_client import OpenAILLMClient

            return OpenAILLMClient(model=model_name)
        elif self.provider == Provider.ANTHROPIC.value:
            from chuk_llm.llm.providers.anthropic_client import AnthropicLLMClient

            return AnthropicLLMClient(model=model_name)
        elif self.provider == Provider.GEMINI.value:
            from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

            return GeminiLLMClient(model=model_name)
        elif self.provider == Provider.GROQ.value:
            from chuk_llm.llm.providers.groq_client import GroqAILLMClient

            return GroqAILLMClient(model=model_name)
        elif self.provider == Provider.MISTRAL.value:
            from chuk_llm.llm.providers.mistral_client import MistralLLMClient

            return MistralLLMClient(model=model_name)
        elif self.provider == Provider.DEEPSEEK.value:
            from chuk_llm.llm.providers.openai_client import OpenAILLMClient

            return OpenAILLMClient(model=model_name)
        elif self.provider == Provider.PERPLEXITY.value:
            from chuk_llm.llm.providers.perplexity_client import PerplexityLLMClient

            return PerplexityLLMClient(model=model_name)
        elif self.provider == Provider.OPENROUTER.value:
            from chuk_llm.llm.llm.providers.openrouter_client import OpenRouterLLMClient

            return OpenRouterLLMClient(model=model_name)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
