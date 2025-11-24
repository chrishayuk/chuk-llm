# chuk_llm/llm/providers/advantage_client.py
"""
Advantage API Client

Extends OpenAILLMClient with JSON function calling fallback enabled.
The Advantage API accepts OpenAI format but doesn't natively support function calling.
"""

import logging

from .openai_client import OpenAILLMClient

log = logging.getLogger(__name__)


class AdvantageClient(OpenAILLMClient):
    """
    Advantage API client using JSON function calling fallback.

    Advantage API characteristics (determined by debug script):
    - Accepts OpenAI tools parameter but doesn't call them natively
    - Requires JSON mode for function calling (model outputs JSON when instructed)
    - Doesn't support 'tool' role messages (must use 'user' role)
    - Doesn't support 'function' role messages
    """

    # Enable JSON function calling fallback
    ENABLE_JSON_FUNCTION_FALLBACK = True
    SUPPORTS_TOOL_ROLE = False
    SUPPORTS_FUNCTION_ROLE = False

    def __init__(self, model: str, api_key: str, api_base: str | None = None, **kwargs):
        """Initialize Advantage client."""
        if not api_base:
            raise ValueError(
                "api_base is required for Advantage client. "
                "Set ADVANTAGE_API_BASE environment variable or configure in chuk_llm.yaml"
            )

        # Filter out config-only parameters
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["api_base_env", "api_key_fallback_env"]
        }

        super().__init__(model, api_key, api_base, **filtered_kwargs)

        # Override detected provider for proper configuration lookup
        self.detected_provider = "advantage"

        # Reinitialize the config mixin with correct provider
        from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

        ConfigAwareProviderMixin.__init__(self, "advantage", model)

        log.info(
            f"Initialized Advantage client (JSON fallback enabled) for model={model}, base={api_base}"
        )


# Alias for backward compatibility
AdvantageLLMClient = AdvantageClient
