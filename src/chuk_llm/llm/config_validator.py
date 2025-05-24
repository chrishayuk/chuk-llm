# chuk_llm/llm/config/config_validator.py
from typing import Dict, Any, List, Optional, Set
import os
import re
from .configuration.capabilities import PROVIDER_CAPABILITIES, Feature

class ConfigValidator:
    """Validates provider configurations comprehensively"""
    
    @staticmethod
    def validate_provider_config(
        provider: str, 
        config: Dict[str, Any],
        strict: bool = False
    ) -> tuple[bool, List[str]]:
        """Validate provider configuration"""
        issues = []
        
        # Check if provider is supported
        if provider not in PROVIDER_CAPABILITIES:
            issues.append(f"Unsupported provider: {provider}")
            return False, issues
        
        capabilities = PROVIDER_CAPABILITIES[provider]
        
        # Check required fields
        if not config.get("client"):
            issues.append(f"Missing 'client' for provider {provider}")
        
        # Check API key if required
        api_key_env = config.get("api_key_env")
        api_key = config.get("api_key")
        
        if api_key_env and not api_key and not os.getenv(api_key_env):
            issues.append(f"Missing API key: {api_key_env} environment variable not set")
        
        # Validate model
        model = config.get("default_model")
        if model and capabilities.supported_models:
            if model not in capabilities.supported_models:
                issues.append(f"Model '{model}' not supported by {provider}")
        
        # Validate API base URL format
        api_base = config.get("api_base")
        if api_base and not ConfigValidator._is_valid_url(api_base):
            issues.append(f"Invalid API base URL: {api_base}")
        
        # Check rate limits configuration
        if strict and capabilities.rate_limits:
            # Could add rate limit validation here
            pass
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_request_compatibility(
        provider: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> tuple[bool, List[str]]:
        """Validate if a request is compatible with provider capabilities"""
        
        if provider not in PROVIDER_CAPABILITIES:
            return False, [f"Unknown provider: {provider}"]
        
        capabilities = PROVIDER_CAPABILITIES[provider]
        issues = []
        
        # Check streaming support
        if stream and not capabilities.supports(Feature.STREAMING):
            issues.append(f"{provider} doesn't support streaming")
        
        # Check tools support
        if tools and not capabilities.supports(Feature.TOOLS):
            issues.append(f"{provider} doesn't support function calling")
        
        # Check for vision content
        has_vision = ConfigValidator._has_vision_content(messages)
        if has_vision and not capabilities.supports(Feature.VISION):
            issues.append(f"{provider} doesn't support vision/image inputs")
        
        # Check context length
        total_length = ConfigValidator._estimate_token_count(messages)
        if capabilities.max_context_length and total_length > capabilities.max_context_length:
            issues.append(
                f"Estimated token count ({total_length}) exceeds {provider} "
                f"context limit ({capabilities.max_context_length})"
            )
        
        # Check for JSON mode
        if kwargs.get("response_format") == "json" and not capabilities.supports(Feature.JSON_MODE):
            issues.append(f"{provider} doesn't support JSON mode")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Basic URL validation"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)
    , re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    @staticmethod
    def _has_vision_content(messages: List[Dict[str, Any]]) -> bool:
        """Check if messages contain vision/image content"""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["image", "image_url"]:
                        return True
        return False
    
    @staticmethod
    def _estimate_token_count(messages: List[Dict[str, Any]]) -> int:
        """Rough estimation of token count"""
        total_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
        
        # Rough approximation: 4 characters per token
        return total_chars // 4

# Enhanced ProviderConfig with validation
class ValidatedProviderConfig(ProviderConfig):
    """ProviderConfig with automatic validation"""
    
    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None, strict: bool = False):
        super().__init__(config)
        self.strict = strict
        self._validate_all_configs()
    
    def _validate_all_configs(self):
        """Validate all provider configurations"""
        issues = []
        
        for provider in self.providers:
            if provider == "__global__":
                continue
                
            config = self.get_provider_config(provider)
            is_valid, provider_issues = ConfigValidator.validate_provider_config(
                provider, config, self.strict
            )
            
            if not is_valid:
                issues.extend([f"{provider}: {issue}" for issue in provider_issues])
        
        if issues and self.strict:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(issues))
        elif issues:
            import warnings
            warnings.warn(f"Configuration issues found:\n" + "\n".join(issues))
    
    def validate_request(
        self,
        provider: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> tuple[bool, List[str]]:
        """Validate a request before sending"""
        return ConfigValidator.validate_request_compatibility(
            provider, messages, tools, **kwargs
        )