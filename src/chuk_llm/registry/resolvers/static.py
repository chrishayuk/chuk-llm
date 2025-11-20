"""
Static capability resolver.

Provides baseline capabilities for well-known models from static data.
This doesn't need to be exhaustive - just covers the major models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from chuk_llm.registry.models import ModelCapabilities, ModelSpec, QualityTier
from chuk_llm.registry.resolvers.base import BaseCapabilityResolver


class StaticCapabilityResolver(BaseCapabilityResolver):
    """
    Resolves capabilities from static data.

    This table doesn't need to be complete - it just provides baseline
    capabilities for common models. Other resolvers can override or augment.
    """

    # Static capability data for well-known models
    # Format: (provider, model_pattern) -> capability_data
    STATIC_CAPABILITIES: dict[tuple[str, str], dict[str, Any]] = {
        # OpenAI
        ("openai", "gpt-4o"): {
            "max_context": 128_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {
                "temperature",
                "top_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            },
            "input_cost_per_1m": 2.50,
            "output_cost_per_1m": 10.00,
            "quality_tier": QualityTier.BEST,
        },
        ("openai", "gpt-4o-mini"): {
            "max_context": 128_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {
                "temperature",
                "top_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            },
            "input_cost_per_1m": 0.15,
            "output_cost_per_1m": 0.60,
            "quality_tier": QualityTier.BALANCED,
        },
        ("openai", "gpt-4-turbo"): {
            "max_context": 128_000,
            "max_output_tokens": 4_096,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {
                "temperature",
                "top_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            },
            "input_cost_per_1m": 10.00,
            "output_cost_per_1m": 30.00,
            "quality_tier": QualityTier.BEST,
        },
        ("openai", "gpt-3.5-turbo"): {
            "max_context": 16_385,
            "max_output_tokens": 4_096,
            "supports_tools": True,
            "supports_vision": False,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {
                "temperature",
                "top_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            },
            "input_cost_per_1m": 0.50,
            "output_cost_per_1m": 1.50,
            "quality_tier": QualityTier.CHEAP,
        },
        ("groq", "llama-3.1-70b-versatile"): {
            "max_context": 128_000,
            "max_output_tokens": 32_768,
            "supports_tools": True,
            "supports_vision": False,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "max_tokens"},
            "input_cost_per_1m": 0.59,
            "output_cost_per_1m": 0.79,
            "quality_tier": QualityTier.BALANCED,
            "speed_hint_tps": 800.0,
        },
        # Anthropic
        ("anthropic", "claude-3-5-sonnet-20241022"): {
            "max_context": 200_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": False,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "top_k", "max_tokens", "metadata"},
            "input_cost_per_1m": 3.00,
            "output_cost_per_1m": 15.00,
            "quality_tier": QualityTier.BEST,
        },
        ("anthropic", "claude-3-5-haiku-20241022"): {
            "max_context": 200_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": False,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "top_k", "max_tokens", "metadata"},
            "input_cost_per_1m": 0.80,
            "output_cost_per_1m": 4.00,
            "quality_tier": QualityTier.BALANCED,
        },
        ("anthropic", "claude-3-opus-20240229"): {
            "max_context": 200_000,
            "max_output_tokens": 4_096,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": False,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "top_k", "max_tokens", "metadata"},
            "input_cost_per_1m": 15.00,
            "output_cost_per_1m": 75.00,
            "quality_tier": QualityTier.BEST,
        },
        # Groq
        ("groq", "llama-3.3-70b-versatile"): {
            "max_context": 128_000,
            "max_output_tokens": 32_768,
            "supports_tools": True,
            "supports_vision": False,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "max_tokens"},
            "input_cost_per_1m": 0.59,
            "output_cost_per_1m": 0.79,
            "quality_tier": QualityTier.BALANCED,
            "speed_hint_tps": 800.0,  # Groq is very fast
        },
        ("groq", "llama-3.1-8b-instant"): {
            "max_context": 128_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": False,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "max_tokens"},
            "input_cost_per_1m": 0.05,
            "output_cost_per_1m": 0.08,
            "quality_tier": QualityTier.CHEAP,
            "speed_hint_tps": 1200.0,
        },
        # DeepSeek
        ("deepseek", "deepseek-chat"): {
            "max_context": 64_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": False,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "max_tokens"},
            "input_cost_per_1m": 0.14,
            "output_cost_per_1m": 0.28,
            "quality_tier": QualityTier.BALANCED,
        },
        # Mistral
        ("mistral", "mistral-small-latest"): {
            "max_context": 32_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": False,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "max_tokens"},
            "input_cost_per_1m": 0.20,
            "output_cost_per_1m": 0.60,
            "quality_tier": QualityTier.CHEAP,
        },
        ("mistral", "mistral-large-latest"): {
            "max_context": 128_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": False,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "max_tokens"},
            "input_cost_per_1m": 2.00,
            "output_cost_per_1m": 6.00,
            "quality_tier": QualityTier.BEST,
        },
        # Gemini
        ("gemini", "gemini-2.0-flash-exp"): {
            "max_context": 1_000_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "top_k", "max_output_tokens"},
            "input_cost_per_1m": 0.00,  # Free during preview
            "output_cost_per_1m": 0.00,
            "quality_tier": QualityTier.BALANCED,
        },
        ("gemini", "gemini-1.5-pro"): {
            "max_context": 2_000_000,
            "max_output_tokens": 8_192,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "top_k", "max_output_tokens"},
            "input_cost_per_1m": 1.25,
            "output_cost_per_1m": 5.00,
            "quality_tier": QualityTier.BEST,
        },
        # Watsonx
        ("watsonx", "ibm/granite-3-8b-instruct"): {
            "max_context": 8_192,
            "max_output_tokens": 2_048,
            "supports_tools": False,
            "supports_vision": False,
            "supports_json_mode": False,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {"temperature", "top_p", "max_new_tokens"},
            "quality_tier": QualityTier.CHEAP,
        },
        # Azure OpenAI (same as OpenAI models)
        ("azure_openai", "gpt-4o"): {
            "max_context": 128_000,
            "max_output_tokens": 16_384,
            "supports_tools": True,
            "supports_vision": True,
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_system_messages": True,
            "known_params": {
                "temperature",
                "top_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            },
            "quality_tier": QualityTier.BEST,
        },
    }

    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """
        Get static capabilities for a model.

        Tries exact match first, then prefix match for versioned models.

        Args:
            spec: Model specification

        Returns:
            Model capabilities (empty if not in static table)
        """
        # Try exact match
        key = (spec.provider, spec.name)
        if key in self.STATIC_CAPABILITIES:
            return self._build_capabilities(self.STATIC_CAPABILITIES[key])

        # Try prefix match for versioned models
        # e.g., "gpt-4o-2024-08-06" matches "gpt-4o"
        for (provider, model_name), cap_data in self.STATIC_CAPABILITIES.items():
            if spec.provider == provider and spec.name.startswith(model_name):
                return self._build_capabilities(cap_data)

        # No match
        return self._empty_capabilities()

    def _build_capabilities(self, data: dict[str, Any]) -> ModelCapabilities:
        """Build ModelCapabilities from static data."""
        return ModelCapabilities(
            **data,
            source="static_resolver",
            last_updated=datetime.now().isoformat(),
        )
