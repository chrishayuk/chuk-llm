"""
Configuration Models
====================

Type-safe Pydantic models for configuration.
Replaces dictionary-based configuration with validated models.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chuk_llm.core import Feature, Provider


class ModelCapabilityConfig(BaseModel):
    """Model capability configuration."""

    model_pattern: str = Field(..., description="Regex pattern matching model names")
    features: set[Feature] = Field(
        default_factory=set, description="Supported features"
    )
    max_context_length: int = Field(
        default=4096, description="Maximum context window size"
    )
    max_output_tokens: int = Field(default=4096, description="Maximum output tokens")

    model_config = ConfigDict(frozen=True)

    def matches(self, model_name: str) -> bool:
        """Check if model name matches this capability pattern."""
        import re

        try:
            return bool(re.match(self.model_pattern, model_name, re.IGNORECASE))
        except re.error:
            return False


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    max_concurrent: int | None = None

    model_config = ConfigDict(frozen=True)


class ProviderConfigModel(BaseModel):
    """Type-safe provider configuration."""

    name: Provider = Field(..., description="Provider name")
    client_class: str = Field(..., description="Python import path to client class")
    api_base: str | None = Field(None, description="API base URL")
    api_key_env: str | None = Field(
        None, description="Environment variable for API key"
    )
    default_model: str = Field(..., description="Default model to use")
    models: list[str] = Field(default_factory=list, description="Available models")
    model_aliases: dict[str, str] = Field(
        default_factory=dict, description="Model name aliases"
    )
    features: set[Feature] = Field(
        default_factory=set, description="Provider-level features"
    )
    model_capabilities: list[ModelCapabilityConfig] = Field(
        default_factory=list, description="Model-specific capabilities"
    )
    rate_limits: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra configuration"
    )

    model_config = ConfigDict(frozen=True)

    def get_api_key(self) -> str | None:
        """Get API key from environment."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def supports_feature(self, feature: Feature, model: str | None = None) -> bool:
        """Check if provider/model supports a feature."""
        # Check model-specific capabilities first
        if model:
            for capability in self.model_capabilities:
                if capability.matches(model):
                    return feature in capability.features

        # Fall back to provider-level features
        return feature in self.features

    def get_model_capabilities(self, model: str) -> ModelCapabilityConfig | None:
        """Get capabilities for a specific model."""
        for capability in self.model_capabilities:
            if capability.matches(model):
                return capability
        return None


class GlobalConfig(BaseModel):
    """Global configuration settings."""

    default_provider: Provider = Field(
        default=Provider.OPENAI, description="Default provider to use"
    )
    default_temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Default temperature"
    )
    default_max_tokens: int | None = Field(None, gt=0, description="Default max tokens")
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = ConfigDict(frozen=True)


class ChukLLMConfig(BaseModel):
    """Complete configuration for chuk-llm."""

    version: str = Field(default="2.0", description="Config version")
    global_config: GlobalConfig = Field(
        default_factory=lambda: GlobalConfig(
            default_temperature=None, default_max_tokens=None
        ),
        description="Global settings",
    )
    providers: dict[str, ProviderConfigModel] = Field(
        default_factory=dict, description="Provider configurations"
    )
    model_aliases: dict[str, str] = Field(
        default_factory=dict, description="Global model aliases (provider/model)"
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("providers")
    @classmethod
    def validate_providers(
        cls, v: dict[str, ProviderConfigModel]
    ) -> dict[str, ProviderConfigModel]:
        """Ensure provider keys match provider names."""
        for key, provider in v.items():
            if key != provider.name.value:
                raise ValueError(
                    f"Provider key '{key}' does not match provider name '{provider.name.value}'"
                )
        return v

    def get_provider(self, provider: Provider | str) -> ProviderConfigModel:
        """Get provider configuration."""
        if isinstance(provider, Provider):
            provider = provider.value

        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not configured")

        return self.providers[provider]

    def resolve_alias(self, alias: str) -> tuple[Provider, str] | None:
        """
        Resolve global model alias to (provider, model).

        Args:
            alias: Model alias to resolve

        Returns:
            (provider, model) tuple or None if not found
        """
        if alias not in self.model_aliases:
            return None

        target = self.model_aliases[alias]
        if "/" not in target:
            return None

        provider_str, model = target.split("/", 1)
        try:
            provider = Provider(provider_str)
            return provider, model
        except ValueError:
            return None
