# chuk_llm/configuration/models.py
"""
Configuration data models using Pydantic v2
"""

import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chuk_llm.core.enums import Feature


class ModelCapabilities(BaseModel):
    """
    Model-specific capabilities with inheritance from provider.

    This model is immutable to ensure thread safety and predictable behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: str = Field(..., description="Regex pattern matching model names")
    features: set[Feature] = Field(
        default_factory=set, description="Model-specific features"
    )
    max_context_length: int | None = Field(
        None, ge=1, description="Maximum context window size"
    )
    max_output_tokens: int | None = Field(
        None, ge=1, description="Maximum output tokens"
    )

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate that pattern is a valid regex"""
        if not v:
            raise ValueError("Pattern cannot be empty")
        try:
            re.compile(v, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e
        return v

    def matches(self, model_name: str) -> bool:
        """Check if this capability applies to the given model"""
        if not model_name:
            return False
        try:
            return bool(re.match(self.pattern, model_name, flags=re.IGNORECASE))
        except re.error:
            return False

    def get_effective_features(self, provider_features: set[Feature]) -> set[Feature]:
        """Get effective features by inheriting from provider and adding model-specific"""
        return provider_features.union(self.features)


class ProviderConfig(BaseModel):
    """
    Complete unified provider configuration.

    This model allows mutation for runtime updates while providing validation.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for provider-specific config
        validate_assignment=True,  # Validate on mutation
        arbitrary_types_allowed=True,
    )

    name: str = Field(..., min_length=1, description="Provider name")

    # Client configuration
    client_class: str = Field(
        default="",
        description="Fully qualified Python import path to client class",
    )
    api_key_env: str | None = Field(
        None, description="Primary environment variable for API key"
    )
    api_key_fallback_env: str | None = Field(
        None, description="Fallback environment variable for API key"
    )
    api_base: str | None = Field(None, description="Base URL for API endpoints")
    api_base_env: str | None = Field(
        None, description="Environment variable for API base URL"
    )

    # Model configuration
    default_model: str = Field(default="", description="Default model to use")
    models: list[str] = Field(default_factory=list, description="Available model names")
    model_aliases: dict[str, str] = Field(
        default_factory=dict, description="Model name aliases"
    )

    # Provider-level capabilities (baseline for all models)
    features: set[Feature] = Field(
        default_factory=set, description="Provider-level supported features"
    )
    max_context_length: int | None = Field(
        None, ge=1, description="Default maximum context window size"
    )
    max_output_tokens: int | None = Field(
        None, ge=1, description="Default maximum output tokens"
    )
    rate_limits: dict[str, int] = Field(
        default_factory=dict, description="Rate limits by tier"
    )

    # Model-specific capability overrides
    model_capabilities: list[ModelCapabilities] = Field(
        default_factory=list, description="Model-specific capability overrides"
    )

    # Inheritance and extras
    inherits: str | None = Field(
        None, description="Parent provider to inherit configuration from"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra configuration"
    )

    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, v: str | None) -> str | None:
        """Validate API base URL format"""
        if v is None:
            return v

        v = v.strip()
        if not v:
            return None

        # Basic URL validation
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("API base must start with http:// or https://")

        # Remove trailing slashes
        return v.rstrip("/")

    @field_validator("client_class")
    @classmethod
    def validate_client_class(cls, v: str) -> str:
        """Validate client class path format"""
        if not v:
            return v

        # Allow simple class names for testing/backwards compatibility
        # Full paths should be module.path:ClassName or module.path.ClassName
        # Simple names like "OpenAIClient" are allowed
        return v

    @model_validator(mode="after")
    def validate_model_defaults(self) -> "ProviderConfig":
        """Ensure default_model is in models list if both are specified"""
        if self.default_model and self.models:
            # Check if default_model is in models or model_aliases
            if (
                self.default_model not in self.models
                and self.default_model not in self.model_aliases
            ):
                # This is a warning, not an error - might be discovered later
                pass
        return self

    def supports_feature(
        self, feature: str | Feature, model: str | None = None
    ) -> bool:
        """Check if provider/model supports a feature"""
        if isinstance(feature, str):
            try:
                feature = Feature.from_string(feature)
            except ValueError:
                return False

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

    def get_api_key(self) -> str | None:
        """Get API key from environment variables or runtime storage"""
        # Check runtime API key first (highest priority)
        if "_runtime_api_key" in self.extra:
            return self.extra["_runtime_api_key"]

        # Then check environment variables
        if self.api_key_env:
            key = os.getenv(self.api_key_env)
            if key:
                return key

        if self.api_key_fallback_env:
            key = os.getenv(self.api_key_fallback_env)
            if key:
                return key

        return None


class DiscoveryConfig(BaseModel):
    """
    Discovery configuration parsed from provider YAML.

    This configuration controls dynamic model discovery behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=False, description="Whether discovery is enabled")
    discoverer_type: str | None = Field(None, description="Type of discoverer to use")
    cache_timeout: int = Field(
        default=300, ge=0, description="Cache timeout in seconds"
    )
    inference_config: dict[str, Any] = Field(
        default_factory=dict, description="Inference configuration"
    )
    discoverer_config: dict[str, Any] = Field(
        default_factory=dict, description="Discoverer-specific configuration"
    )


class GlobalConfig(BaseModel):
    """Global configuration settings"""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    active_provider: str | None = Field(None, description="Currently active provider")
    default_temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Default temperature for requests"
    )
    default_max_tokens: int | None = Field(
        None, ge=1, description="Default max tokens for requests"
    )
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v_upper
