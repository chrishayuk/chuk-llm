# chuk_llm/llm/providers/_config_mixin.py
"""
Configuration-aware mixin with smart fallback to provider clients.
ENHANCED: Now defers to provider client expertise for unknown models.
"""

import logging
from typing import Any

from chuk_llm.core.constants import CapabilityKey

log = logging.getLogger(__name__)


class ConfigAwareProviderMixin:
    """
    Mixin that provides configuration-aware capabilities with smart fallback.
    ENHANCED: Defers to provider client for unknown models.
    """

    def __init__(self, provider_name: str, model: str):
        """Initialize with provider name and model for config lookup"""
        self.provider_name = provider_name
        self.model = model
        self._cached_config = None
        self._cached_model_caps = None

    def _get_provider_config(self):
        """Get provider configuration with caching"""
        if self._cached_config is None:
            try:
                from chuk_llm.configuration import get_config

                config = get_config()
                self._cached_config = config.get_provider(self.provider_name)
            except Exception as e:
                log.error(f"Failed to get config for {self.provider_name}: {e}")
                self._cached_config = None
        return self._cached_config

    def _get_model_capabilities(self):
        """Get model capabilities with caching"""
        if self._cached_model_caps is None:
            provider_config = self._get_provider_config()
            if provider_config:
                self._cached_model_caps = provider_config.get_model_capabilities(
                    self.model
                )
        return self._cached_model_caps

    def _has_explicit_model_config(self) -> bool:
        """Check if this model has explicit configuration"""
        provider_config = self._get_provider_config()
        if not provider_config:
            return False

        # Check if any model capability pattern matches this model
        try:
            for capability in provider_config.model_capabilities:
                if capability.matches(self.model):
                    return True
        except Exception:
            pass

        # Check if model is in the explicit models list
        try:
            return self.model in provider_config.models
        except Exception:
            return False

    def get_model_info(self) -> dict[str, Any]:
        """
        Universal get_model_info that works for all providers using configuration.
        ENHANCED: Includes fallback information when config is unavailable.
        """
        try:
            from chuk_llm.configuration import Feature

            provider_config = self._get_provider_config()
            has_explicit_config = self._has_explicit_model_config()

            if not provider_config:
                return {
                    "provider": self.provider_name,
                    "model": self.model,
                    "error": "Configuration not available",
                    "has_explicit_config": False,
                    "using_fallback": True,
                    "features": [],
                    CapabilityKey.SUPPORTS_TEXT.value: False,
                    CapabilityKey.SUPPORTS_STREAMING.value: False,
                    CapabilityKey.SUPPORTS_TOOLS.value: False,
                    CapabilityKey.SUPPORTS_VISION.value: False,
                    CapabilityKey.SUPPORTS_JSON_MODE.value: False,
                    CapabilityKey.SUPPORTS_SYSTEM_MESSAGES.value: False,
                    CapabilityKey.SUPPORTS_PARALLEL_CALLS.value: False,
                    "supports_multimodal": False,
                    CapabilityKey.SUPPORTS_REASONING.value: False,
                }

            model_caps = provider_config.get_model_capabilities(self.model)

            return {
                "provider": self.provider_name,
                "model": self.model,
                "client_class": provider_config.client_class,
                "api_base": getattr(provider_config, "api_base", None),
                # Configuration metadata
                "has_explicit_config": has_explicit_config,
                "using_fallback": not has_explicit_config,
                # All capabilities from configuration
                "features": [
                    f.value if hasattr(f, "value") else str(f)
                    for f in model_caps.features
                ],
                "max_context_length": model_caps.max_context_length,
                "max_output_tokens": model_caps.max_output_tokens,
                # Individual capability flags for backward compatibility
                CapabilityKey.SUPPORTS_TEXT.value: Feature.TEXT in model_caps.features,
                CapabilityKey.SUPPORTS_STREAMING.value: Feature.STREAMING
                in model_caps.features,
                CapabilityKey.SUPPORTS_TOOLS.value: Feature.TOOLS
                in model_caps.features,
                CapabilityKey.SUPPORTS_VISION.value: Feature.VISION
                in model_caps.features,
                CapabilityKey.SUPPORTS_JSON_MODE.value: Feature.JSON_MODE
                in model_caps.features,
                CapabilityKey.SUPPORTS_SYSTEM_MESSAGES.value: Feature.SYSTEM_MESSAGES
                in model_caps.features,
                CapabilityKey.SUPPORTS_PARALLEL_CALLS.value: Feature.PARALLEL_CALLS
                in model_caps.features,
                "supports_multimodal": Feature.MULTIMODAL in model_caps.features,
                CapabilityKey.SUPPORTS_REASONING.value: Feature.REASONING
                in model_caps.features,
                # Provider metadata
                "rate_limits": provider_config.rate_limits,
                "available_models": provider_config.models,
                "model_aliases": provider_config.model_aliases,
            }

        except Exception as e:
            log.error(f"Configuration error for {self.provider_name}: {e}")
            return {
                "provider": self.provider_name,
                "model": self.model,
                "error": f"Configuration error: {e}",
                "has_explicit_config": False,
                "using_fallback": True,
                "features": [],
                CapabilityKey.SUPPORTS_TEXT.value: False,
                CapabilityKey.SUPPORTS_STREAMING.value: False,
                CapabilityKey.SUPPORTS_TOOLS.value: False,
                CapabilityKey.SUPPORTS_VISION.value: False,
                CapabilityKey.SUPPORTS_JSON_MODE.value: False,
                CapabilityKey.SUPPORTS_SYSTEM_MESSAGES.value: False,
                CapabilityKey.SUPPORTS_PARALLEL_CALLS.value: False,
                "supports_multimodal": False,
                CapabilityKey.SUPPORTS_REASONING.value: False,
            }

    def supports_feature(self, feature_name) -> bool:
        """
        Check if this provider/model supports a specific feature.
        ENHANCED: Checks registry YAML capabilities first, then falls back to configuration.
        """
        try:
            from chuk_llm.configuration import Feature

            # First try registry YAML capabilities (tested data)
            registry_support = self._check_registry_capabilities(feature_name)
            if registry_support is not None:
                return registry_support

            # Fall back to explicit configuration
            model_caps = self._get_model_capabilities()
            if model_caps:
                if isinstance(feature_name, str):
                    feature = Feature.from_string(feature_name)
                else:
                    feature = feature_name

                config_supports = feature in model_caps.features
                log.debug(
                    f"Configuration says {self.provider_name}/{self.model} supports {feature_name}: {config_supports}"
                )
                return config_supports

            # No explicit config found - return None to let provider client decide
            log.debug(
                f"No explicit config for {self.provider_name}/{self.model} - deferring to client"
            )
            return False  # type: ignore[return-value]

        except Exception as e:
            log.warning(f"Feature support check failed for {feature_name}: {e}")
            return False  # type: ignore[return-value]

    def _check_registry_capabilities(self, feature_name: str) -> bool | None:
        """
        Check registry YAML capabilities for tested feature support.
        Returns None if no data found, True/False if capability is known.
        """
        try:
            from pathlib import Path

            import yaml

            # Get capabilities directory
            try:
                # This file is in llm/providers/, so go up 3 levels to get to src/chuk_llm
                package_dir = Path(__file__).parent.parent.parent
                capabilities_dir = package_dir / "registry" / "capabilities"
            except Exception:
                return None

            # Load provider's capability file
            yaml_file = capabilities_dir / f"{self.provider_name}.yaml"
            if not yaml_file.exists():
                return None

            with open(yaml_file) as f:
                cache = yaml.safe_load(f) or {}

            # Check for model in cache
            models = cache.get("models", {})
            model_data = models.get(self.model)
            if not model_data:
                return None

            # Map feature names to capability fields
            feature_map = {
                "text": CapabilityKey.SUPPORTS_TEXT.value,
                "streaming": CapabilityKey.SUPPORTS_STREAMING.value,
                "tools": CapabilityKey.SUPPORTS_TOOLS.value,
                "vision": CapabilityKey.SUPPORTS_VISION.value,
                "json_mode": CapabilityKey.SUPPORTS_JSON_MODE.value,
                "structured_outputs": "supports_structured_outputs",
                "system_messages": CapabilityKey.SUPPORTS_SYSTEM_MESSAGES.value,
                "audio": "supports_audio_input",
            }

            capability_field = feature_map.get(feature_name)
            if capability_field and capability_field in model_data:
                supports = model_data[capability_field]
                log.debug(
                    f"Registry says {self.provider_name}/{self.model} supports {feature_name}: {supports}"
                )
                return supports

            return None

        except Exception as e:
            log.debug(f"Registry capability check failed: {e}")
            return None

    def get_max_tokens_limit(self) -> int | None:
        """Get the max output tokens limit for this model"""
        model_caps = self._get_model_capabilities()
        return model_caps.max_output_tokens if model_caps else None

    def get_context_length_limit(self) -> int | None:
        """Get the max context length for this model"""
        model_caps = self._get_model_capabilities()
        return model_caps.max_context_length if model_caps else None

    def validate_parameters(self, **kwargs) -> dict[str, Any]:
        """Validate and adjust parameters based on model capabilities"""
        adjusted = kwargs.copy()

        # CRITICAL FIX: Don't set max_tokens if max_completion_tokens is already set
        # Some models (like GPT-5 on Azure) don't support both parameters simultaneously
        has_max_completion_tokens = (
            "max_completion_tokens" in adjusted
            and adjusted.get("max_completion_tokens") is not None
        )

        # Validate max_tokens against model limits
        if "max_tokens" in adjusted and adjusted["max_tokens"] is not None:
            limit = self.get_max_tokens_limit()
            if limit and adjusted["max_tokens"] > limit:
                log.debug(
                    f"Capping max_tokens from {adjusted['max_tokens']} to {limit} for {self.provider_name}"
                )
                adjusted["max_tokens"] = limit

        # Validate max_completion_tokens against model limits (for GPT-5/reasoning models)
        if (
            "max_completion_tokens" in adjusted
            and adjusted["max_completion_tokens"] is not None
        ):
            limit = self.get_max_tokens_limit()
            if limit and adjusted["max_completion_tokens"] > limit:
                log.debug(
                    f"Capping max_completion_tokens from {adjusted['max_completion_tokens']} to {limit} for {self.provider_name}"
                )
                adjusted["max_completion_tokens"] = limit

        # Add default max_tokens if not specified or is None
        # BUT: Skip this if max_completion_tokens is already set
        elif not has_max_completion_tokens and (
            "max_tokens" not in adjusted or adjusted.get("max_tokens") is None
        ):
            default_limit = self.get_max_tokens_limit()
            if default_limit:
                adjusted["max_tokens"] = min(4096, default_limit)

        return adjusted
