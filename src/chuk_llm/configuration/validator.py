# chuk_llm/configuration/validator.py
"""
Configuration validation utilities
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple

from .models import ProviderConfig, Feature


class ConfigValidator:
    """Validates configurations and requests"""
    
    @staticmethod
    def validate_provider_config(provider: ProviderConfig, strict: bool = False) -> Tuple[bool, List[str]]:
        """Validate provider configuration"""
        issues = []
        
        # Check required fields
        if not provider.client_class:
            issues.append(f"Missing 'client_class' for provider {provider.name}")
        
        # Check API key for providers that typically need them
        api_key_optional_providers = {"ollama", "local"}
        if provider.name not in api_key_optional_providers:
            if provider.api_key_env and not os.getenv(provider.api_key_env):
                if not provider.api_key_fallback_env or not os.getenv(provider.api_key_fallback_env):
                    issues.append(f"Missing API key: {provider.api_key_env} environment variable not set")
        
        # Validate API base URL
        if provider.api_base and not ConfigValidator._is_valid_url(provider.api_base):
            issues.append(f"Invalid API base URL: {provider.api_base}")
        
        # Check default model
        if not provider.default_model:
            issues.append(f"Missing 'default_model' for provider {provider.name}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_request_compatibility(
        provider_name: str,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[bool, List[str]]:
        """Validate if request is compatible with provider/model"""
        issues = []
        
        try:
            # Import here to avoid circular dependency
            from .unified_config import get_config
            
            config_manager = get_config()
            provider = config_manager.get_provider(provider_name)
            
            # Check streaming support
            if stream and not provider.supports_feature(Feature.STREAMING, model):
                issues.append(f"{provider_name}/{model or 'default'} doesn't support streaming")
            
            # Check tools support
            if tools and not provider.supports_feature(Feature.TOOLS, model):
                issues.append(f"{provider_name}/{model or 'default'} doesn't support function calling")
            
            # Check vision support
            if messages and ConfigValidator._has_vision_content(messages):
                if not provider.supports_feature(Feature.VISION, model):
                    issues.append(f"{provider_name}/{model or 'default'} doesn't support vision/image inputs")
            
            # Check JSON mode
            if kwargs.get("response_format") == "json":
                if not provider.supports_feature(Feature.JSON_MODE, model):
                    issues.append(f"{provider_name}/{model or 'default'} doesn't support JSON mode")
            
        except Exception as exc:
            issues.append(f"Configuration error: {exc}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Basic URL validation"""
        if not url:
            return False
        
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$',
            re.IGNORECASE
        )
        return url_pattern.match(url) is not None
    
    @staticmethod
    def _has_vision_content(messages: List[Dict[str, Any]]) -> bool:
        """Check if messages contain vision/image content"""
        if not messages:
            return False
        
        for message in messages:
            if not message:
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["image", "image_url"]:
                        return True
        return False