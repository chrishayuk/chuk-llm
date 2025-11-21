"""
Heuristic capability resolver.

Infers basic model properties from model names using heuristics.
Only used for properties that cannot be easily tested via API calls.
"""

from __future__ import annotations

from datetime import datetime

from chuk_llm.registry.models import ModelCapabilities, ModelSpec, QualityTier
from chuk_llm.registry.resolvers.base import BaseCapabilityResolver


class HeuristicCapabilityResolver(BaseCapabilityResolver):
    """
    Heuristic-based resolver for model properties.

    This resolver uses name patterns to infer basic model properties
    that cannot be easily tested via API calls:
    - Quality tier (best/balanced/cheap) - affects model selection
    - Max context length - affects request validation

    All testable capabilities (tools, vision, JSON, streaming) default
    to False and must be tested via update_capabilities.py or runtime testing.
    """

    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """
        Return minimal capabilities for model spec.

        Args:
            spec: Model specification

        Returns:
            Minimal model capabilities (conservative defaults)
        """
        provider = spec.provider
        name_lower = spec.name.lower()

        # Only infer quality tier and context length - these can't be tested
        quality_tier = self._infer_quality_tier(provider, name_lower)
        max_context = self._infer_context_length(provider, name_lower)

        # All other capabilities default to False/None - they must be tested
        return ModelCapabilities(
            supports_tools=False,  # Must be tested
            supports_vision=False,  # Must be tested
            supports_json_mode=False,  # Must be tested
            supports_structured_outputs=False,  # Must be tested
            supports_streaming=True,  # Safe assumption for modern models
            supports_system_messages=True,  # Safe assumption for modern models
            quality_tier=quality_tier,
            max_context=max_context,
            source="heuristic_resolver",
            last_updated=datetime.now().isoformat(),
        )

    def _infer_quality_tier(self, provider: str, name_lower: str) -> QualityTier:
        """Infer quality tier from model name."""
        # Strategy: Check most specific patterns first to avoid false matches
        # e.g., "gpt-4o-mini" should be CHEAP, not BEST (from "gpt-4o")
        # e.g., "gemini-1.5-pro" should be BEST, not CHEAP (from "mini" in "gemini")

        # CHEAP tier - check these first because they're often substrings of BEST patterns
        cheap_patterns = [
            "gpt-4o-mini",  # Must check before "gpt-4o"
            "gpt-3.5",
            "gpt-4-turbo-preview",
            "nano",
            "small",
            "haiku",
            "flash",
            "instant",
            "lite",
        ]

        for pattern in cheap_patterns:
            if pattern in name_lower:
                return QualityTier.CHEAP

        # Check "mini" separately with gemini exclusion
        if "mini" in name_lower and "gemini" not in name_lower:
            return QualityTier.CHEAP

        # BEST tier - checked after CHEAP to avoid substring issues
        best_patterns = [
            "gemini-2.5-pro",
            "gemini-1.5-pro",
            "claude-3-5-sonnet",
            "claude-3-opus",
            "gpt-4o",  # Now safe - mini already ruled out
            "gpt-5",
            "o3-pro",
            "o1-pro",
            "claude-4",
            "mistral-large",
            "ultra",
            "pro",  # Generic fallback
        ]

        for pattern in best_patterns:
            if pattern in name_lower:
                return QualityTier.BEST

        # Balanced tier indicators (default for most models)
        balanced_patterns = [
            "gpt-4",
            "claude-3-sonnet",
            "claude-3.5-haiku",
            "gemini-1.5",
            "gemini-2.0",
            "llama-3",
            "70b",
            "mistral-",
            "deepseek",
        ]

        for pattern in balanced_patterns:
            if pattern in name_lower:
                return QualityTier.BALANCED

        # Default to UNKNOWN if we can't infer
        return QualityTier.UNKNOWN

    def _infer_context_length(self, provider: str, name_lower: str) -> int | None:
        """Infer context length from model name."""
        # Explicit context in name (e.g., "128k")
        if "128k" in name_lower:
            return 128_000
        elif "200k" in name_lower:
            return 200_000
        elif "1m" in name_lower or "1000k" in name_lower:
            return 1_000_000
        elif "2m" in name_lower or "2000k" in name_lower:
            return 2_000_000

        # Provider-specific defaults
        if provider == "openai":
            if "gpt-4" in name_lower or "gpt-5" in name_lower:
                return 128_000
            elif "gpt-3.5" in name_lower:
                return 16_385

        elif provider == "anthropic":
            return 200_000  # Claude 3 models have 200k context

        elif provider == "gemini":
            if "gemini-1.5" in name_lower:
                return 2_000_000
            elif "gemini-2" in name_lower:
                return 1_000_000

        elif provider == "groq":
            return 128_000  # Groq typically has 128k context

        return None
