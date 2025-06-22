# chuk_llm/llm/providers/_config_mixin.py
"""
Configuration-aware mixin for all provider clients.
"""

import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)


class ConfigAwareProviderMixin:
    """
    Mixin that provides configuration-aware capabilities for any provider client.
    Use this instead of hardcoding capabilities in each provider.
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
                self._cached_model_caps = provider_config.get_model_capabilities(self.model)
        return self._cached_model_caps
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Universal get_model_info that works for all providers using configuration.
        Override this in provider clients if you need provider-specific info.
        """
        try:
            from chuk_llm.configuration import Feature
            
            provider_config = self._get_provider_config()
            if not provider_config:
                return {
                    "provider": self.provider_name,
                    "model": self.model,
                    "error": "Configuration not available",
                    "features": [],
                    "supports_text": False,
                    "supports_streaming": False,
                    "supports_tools": False,
                    "supports_vision": False,
                    "supports_json_mode": False,
                    "supports_system_messages": False,
                    "supports_parallel_calls": False,
                    "supports_multimodal": False,
                    "supports_reasoning": False,
                }
            
            model_caps = provider_config.get_model_capabilities(self.model)
            
            return {
                "provider": self.provider_name,
                "model": self.model,
                "client_class": provider_config.client_class,
                "api_base": getattr(provider_config, 'api_base', None),
                
                # All capabilities from configuration
                "features": [f.value if hasattr(f, 'value') else str(f) for f in model_caps.features],
                "max_context_length": model_caps.max_context_length,
                "max_output_tokens": model_caps.max_output_tokens,
                
                # Individual capability flags for backward compatibility
                "supports_text": Feature.TEXT in model_caps.features,
                "supports_streaming": Feature.STREAMING in model_caps.features,
                "supports_tools": Feature.TOOLS in model_caps.features,
                "supports_vision": Feature.VISION in model_caps.features,
                "supports_json_mode": Feature.JSON_MODE in model_caps.features,
                "supports_system_messages": Feature.SYSTEM_MESSAGES in model_caps.features,
                "supports_parallel_calls": Feature.PARALLEL_CALLS in model_caps.features,
                "supports_multimodal": Feature.MULTIMODAL in model_caps.features,
                "supports_reasoning": Feature.REASONING in model_caps.features,
                
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
                "error": "Configuration not available",
                "features": [],
                "supports_text": False,
                "supports_streaming": False,
                "supports_tools": False,
                "supports_vision": False,
                "supports_json_mode": False,
                "supports_system_messages": False,
                "supports_parallel_calls": False,
                "supports_multimodal": False,
                "supports_reasoning": False,
            }
    
    def supports_feature(self, feature_name) -> bool:
        """Check if this provider/model supports a specific feature"""
        try:
            from chuk_llm.configuration import Feature
            
            model_caps = self._get_model_capabilities()
            if not model_caps:
                return False
            
            if isinstance(feature_name, str):
                feature = Feature.from_string(feature_name)
            else:
                feature = feature_name
            
            return feature in model_caps.features
            
        except Exception:
            return False
    
    def get_max_tokens_limit(self) -> Optional[int]:
        """Get the max output tokens limit for this model"""
        model_caps = self._get_model_capabilities()
        return model_caps.max_output_tokens if model_caps else None
    
    def get_context_length_limit(self) -> Optional[int]:
        """Get the max context length for this model"""
        model_caps = self._get_model_capabilities()
        return model_caps.max_context_length if model_caps else None
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """Validate and adjust parameters based on model capabilities"""
        adjusted = kwargs.copy()
        
        # Validate max_tokens against model limits
        if 'max_tokens' in adjusted and adjusted['max_tokens'] is not None:
            limit = self.get_max_tokens_limit()
            if limit and adjusted['max_tokens'] > limit:
                log.debug(f"Capping max_tokens from {adjusted['max_tokens']} to {limit} for {self.provider_name}")
                adjusted['max_tokens'] = limit
        
        # Add default max_tokens if not specified or is None
        elif 'max_tokens' not in adjusted or adjusted.get('max_tokens') is None:
            default_limit = self.get_max_tokens_limit()
            if default_limit:
                adjusted['max_tokens'] = min(4096, default_limit)
        
        return adjusted