# diagnostics/capabilities/utils/provider_configs.py
"""
Dynamic provider configurations that use the unified configuration system.
No hardcoded provider lists - everything comes from your YAML config!
"""
from __future__ import annotations

import base64
from typing import Dict, Any, Optional, Set
from abc import ABC, abstractmethod

# Simple 1x1 red pixel PNG for maximum compatibility
SIMPLE_RED_PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

class DynamicProviderConfig:
    """
    Dynamic provider configuration that reads everything from unified config.
    No hardcoding - adapts to your YAML configuration automatically!
    """
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name.lower()
        self._config_manager = None
        self._provider_config = None
        
    @property
    def config_manager(self):
        """Lazy load config manager"""
        if self._config_manager is None:
            from chuk_llm.configuration.unified_config import get_config
            self._config_manager = get_config()
        return self._config_manager
    
    @property 
    def provider_config(self):
        """Lazy load provider config"""
        if self._provider_config is None:
            try:
                self._provider_config = self.config_manager.get_provider(self.provider_name)
            except ValueError:
                # Provider not found - create minimal fallback
                from chuk_llm.configuration.unified_config import ProviderConfig
                self._provider_config = ProviderConfig(
                    name=self.provider_name,
                    features=set(),
                    models=[],
                    model_aliases={}
                )
        return self._provider_config
    
    def supports_feature(self, feature: str, model: Optional[str] = None) -> bool:
        """Check if provider supports a feature using unified config"""
        try:
            from chuk_llm.configuration.unified_config import Feature
            
            # Map diagnostic feature names to unified config features
            feature_map = {
                "text": Feature.STREAMING,  # Basic text implies streaming capability
                "streaming": Feature.STREAMING,
                "tools": Feature.TOOLS,
                "streaming_tools": Feature.TOOLS,  # If tools work, streaming tools should too
                "vision": Feature.VISION,
                "json_mode": Feature.JSON_MODE,
                "system_messages": Feature.SYSTEM_MESSAGES,
                "multimodal": Feature.MULTIMODAL,
                "reasoning": Feature.REASONING
            }
            
            if feature in feature_map:
                return self.provider_config.supports_feature(feature_map[feature], model)
            
            # For unmapped features, try direct string lookup
            try:
                feature_enum = Feature.from_string(feature)
                return self.provider_config.supports_feature(feature_enum, model)
            except ValueError:
                return False
                
        except Exception:
            # Fallback: assume basic features are supported
            return feature in ["text", "streaming"]
    
    def get_vision_format(self) -> str:
        """Determine vision message format based on provider"""
        # Check if provider inherits from another (e.g., DeepSeek inherits from OpenAI)
        if hasattr(self.provider_config, 'inherits') and self.provider_config.inherits:
            parent = self.provider_config.inherits.lower()
            if parent == "openai":
                return "openai"
        
        # Determine format based on provider name patterns
        provider = self.provider_name
        
        if provider in ["anthropic", "claude"]:
            return "anthropic"
        elif provider in ["gemini", "google"]:
            return "openai"  # Gemini uses OpenAI-style format
        elif provider == "ollama":
            return "text_only"  # Ollama has format issues with complex content
        elif provider in ["groq", "perplexity"]:
            return "text_only"  # Limited vision support
        else:
            return "openai"  # Default to OpenAI format for most providers
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        """Create vision message using dynamic format detection"""
        vision_format = self.get_vision_format()
        
        if vision_format == "anthropic":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": SIMPLE_RED_PIXEL
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        elif vision_format == "text_only":
            return {
                "role": "user",
                "content": f"{prompt} [Vision test with text-only format for {self.provider_name} compatibility]"
            }
        else:  # openai format (default)
            return {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{SIMPLE_RED_PIXEL}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        """Get error patterns - can be enhanced based on provider type"""
        base_categories = {
            "vision_format": [
                "input should be a valid string",
                "validation error for message", 
                "content must be a string",
                "invalid image",
                "image format not supported",
                "does not have the 'vision' capability",
                "vision not supported",
                "multimodal not supported"
            ],
            "tools_unsupported": [
                "does not support tools",
                "function calling not available",
                "model does not support function calling",
                "tools are not supported"
            ],
            "rate_limit": [
                "rate limit",
                "too many requests",
                "quota exceeded",
                "overloaded"
            ],
            "model_error": [
                "model not found",
                "invalid model",
                "model not supported for this environment",
                "model not loaded"
            ],
            "auth_error": [
                "invalid api key",
                "authentication failed",
                "unauthorized"
            ],
            "connection_error": [
                "connection refused",
                "connection failed",
                "connection error",
                "timeout",
                "i/o operation on closed file"
            ]
        }
        
        return base_categories
    
    def get_models(self) -> list[str]:
        """Get available models for this provider"""
        return self.provider_config.models
    
    def get_default_model(self) -> str:
        """Get default model for this provider"""
        return self.provider_config.default_model
    
    def get_model_aliases(self) -> Dict[str, str]:
        """Get model aliases for this provider"""
        return self.provider_config.model_aliases
    
    def get_supported_features(self, model: Optional[str] = None) -> Set[str]:
        """Get all supported features for this provider/model"""
        try:
            from chuk_llm.configuration.unified_config import Feature
            
            # Get model capabilities
            model_caps = self.provider_config.get_model_capabilities(model)
            
            # Convert to diagnostic feature names
            diagnostic_features = set()
            for feature in model_caps.features:
                if feature == Feature.STREAMING:
                    diagnostic_features.update(["text", "streaming"])
                elif feature == Feature.TOOLS:
                    diagnostic_features.update(["tools", "streaming_tools"])
                elif feature == Feature.VISION:
                    diagnostic_features.add("vision")
                elif feature == Feature.JSON_MODE:
                    diagnostic_features.add("json_mode")
                elif feature == Feature.REASONING:
                    diagnostic_features.add("reasoning")
            
            return diagnostic_features
        except Exception:
            return {"text"}  # Minimum fallback


# Factory functions that work with any provider in your config
def get_provider_config(provider: str) -> DynamicProviderConfig:
    """Get dynamic configuration for any provider in your unified config"""
    return DynamicProviderConfig(provider)

def get_all_configured_providers() -> list[str]:
    """Get all providers from your unified configuration"""
    try:
        from chuk_llm.configuration.unified_config import get_config
        config_manager = get_config()
        return config_manager.get_all_providers()
    except Exception:
        # Fallback if config loading fails
        return ["openai", "anthropic", "groq"]

def is_vision_supported(provider: str, model: Optional[str] = None) -> bool:
    """Check if provider supports vision using unified config"""
    config = get_provider_config(provider)
    return config.supports_feature("vision", model)

def get_provider_capabilities(provider: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive provider capabilities"""
    config = get_provider_config(provider)
    
    try:
        supported_features = config.get_supported_features(model)
        
        return {
            "provider": provider,
            "model": model or config.get_default_model(),
            "supported_features": list(supported_features),
            "models": config.get_models(),
            "model_aliases": config.get_model_aliases(),
            "vision_format": config.get_vision_format(),
            "supports": {
                "text": config.supports_feature("text", model),
                "streaming": config.supports_feature("streaming", model),
                "tools": config.supports_feature("tools", model),
                "streaming_tools": config.supports_feature("streaming_tools", model),
                "vision": config.supports_feature("vision", model),
                "json_mode": config.supports_feature("json_mode", model),
                "reasoning": config.supports_feature("reasoning", model),
            }
        }
    except Exception as e:
        return {
            "provider": provider,
            "error": str(e),
            "supported_features": ["text"],
            "supports": {"text": True}
        }

def validate_provider_setup(provider: str) -> Dict[str, Any]:
    """Validate a provider's configuration"""
    try:
        from chuk_llm.configuration.unified_config import get_config, ConfigValidator
        
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider)
        
        # Use the built-in validator
        is_valid, issues = ConfigValidator.validate_provider_config(provider_config)
        
        return {
            "provider": provider,
            "valid": is_valid,
            "issues": issues,
            "has_api_key": bool(config_manager.get_api_key(provider)),
            "default_model": provider_config.default_model,
            "model_count": len(provider_config.models),
            "features": [f.value for f in provider_config.features]
        }
    except Exception as e:
        return {
            "provider": provider,
            "valid": False,
            "error": str(e)
        }

# Utility function for diagnostic scripts
def auto_detect_capabilities(provider: str, model: Optional[str] = None) -> Dict[str, bool]:
    """Auto-detect what capabilities a provider actually supports"""
    config = get_provider_config(provider)
    
    capabilities = {}
    test_features = ["text", "streaming", "tools", "streaming_tools", "vision", "json_mode", "reasoning"]
    
    for feature in test_features:
        try:
            capabilities[feature] = config.supports_feature(feature, model)
        except Exception:
            capabilities[feature] = False
    
    return capabilities

# Migration helper for existing code
def get_supported_providers() -> list[str]:
    """Get all supported providers (alias for get_all_configured_providers)"""
    return get_all_configured_providers()

def check_capability_support(provider: str, feature: str, model: Optional[str] = None) -> bool:
    """Check capability support (unified interface)"""
    config = get_provider_config(provider)
    return config.supports_feature(feature, model)