# diagnostics/capabilities/provider_configs.py
"""
Provider-specific configurations and message formatting for LLM diagnostics.
"""
from __future__ import annotations

import base64
from typing import Dict, Any
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
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["image_parse_error", "unsupported image", "invalid image"],
            "tools_unsupported": ["does not support tools", "function calling not available"],
            "rate_limit": ["rate limit", "too many requests"],
            "model_error": ["model not found", "invalid model"]
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
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["invalid base64", "could not process image"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["overloaded", "rate limit"],
            "model_error": ["model not found"]
        }

class GeminiConfig(ProviderConfig):
    """Google Gemini-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        # Try the standard ChatML format first
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
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["validation errors", "extra inputs are not permitted", "contents are required"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["quota exceeded", "rate limit"],
            "model_error": ["model not found"]
        }

class GroqConfig(ProviderConfig):
    """Groq-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        # Groq might not support multimodal, use text-only
        return {
            "role": "user",
            "content": f"{prompt} (Note: Groq vision testing with text-only)"
        }
    
    def supports_feature(self, feature: str) -> bool:
        # Groq has limited vision support
        if feature == "vision":
            return False  # Mark as unsupported for now
        return feature in ["text", "streaming", "tools", "streaming_tools"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["must be a string", "invalid content format"],
            "tools_unsupported": ["does not support tools"],
            "rate_limit": ["rate limit exceeded"],
            "model_error": ["model not found"]
        }

class OllamaConfig(ProviderConfig):
    """Ollama-specific configuration"""
    
    def create_vision_message(self, prompt: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": prompt,
            "images": [base64.b64decode(SIMPLE_RED_PIXEL)]
        }
    
    def supports_feature(self, feature: str) -> bool:
        return feature in ["text", "streaming", "tools", "streaming_tools", "vision"]
    
    def get_error_categories(self) -> Dict[str, list[str]]:
        return {
            "vision_format": ["image format error", "unsupported image"],
            "tools_unsupported": ["does not support tools", "function calling not available"],
            "connection_error": ["connection refused", "connection failed"],
            "model_error": ["model not found", "model not loaded"]
        }

# Provider registry
PROVIDER_CONFIGS = {
    "openai": OpenAIConfig(),
    "anthropic": AnthropicConfig(),
    "claude": AnthropicConfig(),  # Alias
    "gemini": GeminiConfig(),
    "groq": GroqConfig(),
    "ollama": OllamaConfig(),
}

def get_provider_config(provider: str) -> ProviderConfig:
    """Get the configuration for a specific provider"""
    return PROVIDER_CONFIGS.get(provider.lower(), OpenAIConfig())  # Default to OpenAI format