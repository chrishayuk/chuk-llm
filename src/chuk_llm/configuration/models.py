# chuk_llm/configuration/models.py
"""
Configuration data models and feature definitions
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Feature(str, Enum):
    """Supported LLM features"""

    TEXT = "text"  # Basic text completion capability
    STREAMING = "streaming"  # Streaming response capability
    TOOLS = "tools"  # Function calling/tools
    VISION = "vision"  # Image/visual input processing
    JSON_MODE = "json_mode"  # Structured JSON output
    PARALLEL_CALLS = "parallel_calls"  # Multiple simultaneous function calls
    SYSTEM_MESSAGES = "system_messages"  # System message support
    MULTIMODAL = "multimodal"  # Multiple input modalities
    REASONING = "reasoning"  # Advanced reasoning capabilities

    @classmethod
    def from_string(cls, value: str) -> "Feature":
        """Convert string to Feature enum"""
        try:
            return cls(value.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown feature: {value}") from exc


@dataclass
class ModelCapabilities:
    """Model-specific capabilities with inheritance from provider"""

    pattern: str
    features: set[Feature] = field(default_factory=set)
    max_context_length: int | None = None
    max_output_tokens: int | None = None

    def matches(self, model_name: str) -> bool:
        """Check if this capability applies to the given model"""
        return bool(re.match(self.pattern, model_name, flags=re.IGNORECASE))

    def get_effective_features(self, provider_features: set[Feature]) -> set[Feature]:
        """Get effective features by inheriting from provider and adding model-specific"""
        return provider_features.union(self.features)


@dataclass
class ProviderConfig:
    """Complete unified provider configuration"""

    name: str

    # Client configuration
    client_class: str = ""
    api_key_env: str | None = None
    api_key_fallback_env: str | None = None
    api_base: str | None = None

    # Model configuration
    default_model: str = ""
    models: list[str] = field(default_factory=list)
    model_aliases: dict[str, str] = field(default_factory=dict)

    # Provider-level capabilities (baseline for all models)
    features: set[Feature] = field(default_factory=set)
    max_context_length: int | None = None
    max_output_tokens: int | None = None
    rate_limits: dict[str, int] = field(default_factory=dict)

    # Model-specific capability overrides
    model_capabilities: list[ModelCapabilities] = field(default_factory=list)

    # Inheritance and extras
    inherits: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def supports_feature(
        self, feature: str | Feature, model: str | None = None
    ) -> bool:
        """Check if provider/model supports a feature"""
        if isinstance(feature, str):
            feature = Feature.from_string(feature)

        if model:
            # Check model-specific capabilities
            model_caps = self.get_model_capabilities(model)
            effective_features = model_caps.get_effective_features(self.features)
            return feature in effective_features
        else:
            # Check provider baseline
            return feature in self.features

    def get_model_capabilities(self, model: str | None = None) -> ModelCapabilities:
        """Get capabilities for specific model"""
        if model and self.model_capabilities:
            for mc in self.model_capabilities:
                if mc.matches(model):
                    # Return model-specific caps with proper inheritance
                    return ModelCapabilities(
                        pattern=mc.pattern,
                        features=mc.get_effective_features(self.features),
                        max_context_length=mc.max_context_length
                        or self.max_context_length,
                        max_output_tokens=mc.max_output_tokens
                        or self.max_output_tokens,
                    )

        # Return provider defaults
        return ModelCapabilities(
            pattern=".*",
            features=self.features.copy(),
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens,
        )

    def get_rate_limit(self, tier: str = "default") -> int | None:
        """Get rate limit for tier"""
        return self.rate_limits.get(tier)


@dataclass
class DiscoveryConfig:
    """Discovery configuration parsed from provider YAML"""

    enabled: bool = False
    discoverer_type: str | None = None
    cache_timeout: int = 300
    inference_config: dict[str, Any] = field(default_factory=dict)
    discoverer_config: dict[str, Any] = field(default_factory=dict)
