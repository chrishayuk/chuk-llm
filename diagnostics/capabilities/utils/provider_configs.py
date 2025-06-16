# diagnostics/capabilities/utils/provider_configs.py
"""
Provider-specific configurations and message formatting for LLM diagnostics.
Updated to work with the new chuk-llm configuration system.
"""
from __future__ import annotations

import base64
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Simple 1x1 red pixel PNG for maximum compatibility
SIMPLE_RED_PIXEL = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

class ProviderConfig(ABC):
    """Base class for provider-specific configurations"""
    
    @abstractmethod
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        """Create a vision message in the provider's expected format"""
        pass
    
    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """Check if provider supports a specific feature"""
        pass
    
    @abstractmethod
    def get_error_categories(self) -> Dict[str, list[str]]:
        """Get provider-specific error patterns for categorization"""
        pass

class OpenAIConfig(ProviderConfig):
    """OpenAI-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
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
    
    def supports_feature(self, feature: str) -> bool:
        # Check the new capability system first
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "openai" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["openai"]
                feature_map = {
                    "text": Feature.STREAMING,  # Text implies basic functionality
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,  # If tools work, streaming tools should too
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        # Fallback to hardcoded values
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["image_parse_error", "unsupported image", "invalid image"],
            "tools_unsupported": ["does not support tools", "function calling not available"],
            "rate_limit": ["rate limit", "too many requests"],
            "model_error": ["model not found", "invalid model"],
            "auth_error": ["invalid api key", "authentication failed"]
        }

class AnthropicConfig(ProviderConfig):
    """Anthropic-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
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
    
    def supports_feature(self, feature: str) -> bool:
        # Check the new capability system first
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "anthropic" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["anthropic"]
                feature_map = {
                    "text": Feature.STREAMING,
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["invalid base64", "could not process image"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["overloaded", "rate limit"],
            "model_error": ["model not found"],
            "auth_error": ["authentication failed", "invalid api key"]
        }

class GeminiConfig(ProviderConfig):
    """Google Gemini-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{SIMPLE_RED_PIXEL}"
                    }
                }
            ]
        }
    
    def supports_feature(self, feature: str) -> bool:
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "gemini" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["gemini"]
                feature_map = {
                    "text": Feature.STREAMING,
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["validation errors", "extra inputs are not permitted", "contents are required"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["quota exceeded", "rate limit"],
            "model_error": ["model not found"],
            "auth_error": ["authentication failed", "invalid api key"]
        }

class GroqConfig(ProviderConfig):
    """Groq-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        # Groq typically doesn't support multimodal, use text-only
        return {
            "role": "user",
            "content": f"{prompt} (Note: Groq vision testing with text-only)"
        }
    
    def supports_feature(self, feature: str) -> bool:
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "groq" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["groq"]
                feature_map = {
                    "text": Feature.STREAMING,
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        # Groq has limited vision support
        if feature == "vision":
            return False  # Mark as unsupported for now
        return feature in ["text", "streaming", "tools", "streaming_tools"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["must be a string", "invalid content format"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["rate limit exceeded"],
            "model_error": ["model not found"],
            "auth_error": ["authentication failed", "invalid api key"]
        }

class OllamaConfig(ProviderConfig):
    """Ollama-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        # Ollama may use a different format for images
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{SIMPLE_RED_PIXEL}"
                    }
                }
            ]
        }
    
    def supports_feature(self, feature: str) -> bool:
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "ollama" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["ollama"]
                feature_map = {
                    "text": Feature.STREAMING,
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["image format error", "unsupported image"],
            "tools_unsupported": ["does not support tools", "function calling not available"],
            "connection_error": ["connection refused", "connection failed"],
            "model_error": ["model not found", "model not loaded"]
        }

class MistralConfig(ProviderConfig):
    """Mistral-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        return {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{SIMPLE_RED_PIXEL}"
                    }
                }
            ]
        }
    
    def supports_feature(self, feature: str) -> bool:
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "mistral" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["mistral"]
                feature_map = {
                    "text": Feature.STREAMING,
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["invalid image", "image format not supported"],
            "tools_unsupported": ["model does not support function calling"],
            "rate_limit": ["rate limit exceeded", "quota exceeded"],
            "model_error": ["model not found", "invalid model"],
            "auth_error": ["authentication failed", "invalid api key"]
        }

class DeepSeekConfig(ProviderConfig):
    """DeepSeek-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        # DeepSeek uses OpenAI-compatible format
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
    
    def supports_feature(self, feature: str) -> bool:
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "deepseek" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["deepseek"]
                feature_map = {
                    "text": Feature.STREAMING,
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        # DeepSeek currently has limited vision support
        if feature == "vision":
            return False
        return feature in ["text", "streaming", "tools", "streaming_tools"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["image not supported", "multimodal not supported"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["rate limit exceeded", "quota exceeded"],
            "model_error": ["model not found", "invalid model"],
            "auth_error": ["authentication failed", "invalid api key"]
        }

class PerplexityConfig(ProviderConfig):
    """Perplexity-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        # Perplexity typically doesn't support vision
        return {
            "role": "user",
            "content": f"{prompt} (Note: Perplexity vision testing with text-only)"
        }
    
    def supports_feature(self, feature: str) -> bool:
        try:
            from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
            if "perplexity" in PROVIDER_CAPABILITIES:
                caps = PROVIDER_CAPABILITIES["perplexity"]
                feature_map = {
                    "text": Feature.STREAMING,
                    "streaming": Feature.STREAMING,
                    "tools": Feature.TOOLS,
                    "streaming_tools": Feature.TOOLS,
                    "vision": Feature.VISION
                }
                if feature in feature_map:
                    return feature_map[feature] in caps.features
        except ImportError:
            pass
        
        # Perplexity has limited vision support
        if feature == "vision":
            return False
        return feature in ["text", "streaming", "tools", "streaming_tools"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["vision not supported", "multimodal not supported"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["rate limit exceeded", "quota exceeded"],
            "model_error": ["model not found", "invalid model"],
            "auth_error": ["authentication failed", "invalid api key"]
        }

# Provider registry - now includes all providers from your config
PROVIDER_CONFIGS = {
    "openai": OpenAIConfig(),
    "anthropic": AnthropicConfig(),
    "claude": AnthropicConfig(),  # Alias
    "gemini": GeminiConfig(),
    "groq": GroqConfig(),
    "ollama": OllamaConfig(),
    "mistral": MistralConfig(),
    "deepseek": DeepSeekConfig(),
    "perplexity": PerplexityConfig(),
    "watsonx": OpenAIConfig(),  # IBM Watson uses similar format
    "togetherai": OpenAIConfig(),  # Together AI uses OpenAI-compatible format
}

def get_provider_config(provider: str) -> ProviderConfig:
    """Get the configuration for a specific provider"""
    return PROVIDER_CONFIGS.get(provider.lower(), OpenAIConfig())  # Default to OpenAI format

def get_supported_providers() -> list[str]:
    """Get list of all supported providers"""
    return list(PROVIDER_CONFIGS.keys())

def is_vision_supported(provider: str) -> bool:
    """Quick check if a provider supports vision"""
    config = get_provider_config(provider)
    return config.supports_feature("vision")