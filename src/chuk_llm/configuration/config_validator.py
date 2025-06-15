# chuk_llm/configuration/config_validator.py
"""
Configuration validation for ChukLLM
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import re
from .config import get_config


class ConfigValidator:
    """Validates provider configurations"""
    
    @staticmethod
    def validate_provider_config(
        provider: str, 
        config: Dict[str, Any],
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate provider configuration
        
        Args:
            provider: Provider name
            config: Provider configuration dictionary
            strict: Whether to apply strict validation
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if config is None:
            issues.append(f"Configuration is None for provider {provider}")
            return False, issues
        
        # Check required fields
        if not config.get("client_class"):
            issues.append(f"Missing 'client_class' for provider {provider}")
        
        # Check API key configuration for non-local providers
        if provider not in ["ollama"]:
            api_key_env = config.get("api_key_env")
            api_key = config.get("api_key")
            
            if api_key_env and not api_key and not os.getenv(api_key_env):
                issues.append(f"Missing API key: {api_key_env} environment variable not set")
        
        # Validate API base URL format if provided
        api_base = config.get("api_base")
        if api_base and not ConfigValidator._is_valid_url(api_base):
            issues.append(f"Invalid API base URL: {api_base}")
        
        # Check default model
        if not config.get("default_model"):
            issues.append(f"Missing 'default_model' for provider {provider}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_request_compatibility(
        provider: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a request is compatible with provider
        
        Args:
            provider: Provider name
            messages: Chat messages
            tools: Function tools
            stream: Whether streaming is requested
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Get provider config to check features
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(provider)
            
            # Check if provider supports requested features
            features = getattr(provider_config, 'features', set())
            
            if stream and 'streaming' not in features:
                issues.append(f"{provider} doesn't support streaming")
            
            if tools and 'tools' not in features:
                issues.append(f"{provider} doesn't support function calling")
            
            # Check for vision content
            has_vision = ConfigValidator._has_vision_content(messages)
            if has_vision and 'vision' not in features:
                issues.append(f"{provider} doesn't support vision/image inputs")
            
            # Check for JSON mode
            if kwargs.get("response_format") == "json" and 'json_mode' not in features:
                issues.append(f"{provider} doesn't support JSON mode")
                
        except Exception:
            # If we can't get provider config, we can't validate features
            pass
        
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
    
    @staticmethod
    def _estimate_token_count(messages: List[Dict[str, Any]]) -> int:
        """Rough estimation of token count"""
        if not messages:
            return 0
            
        total_chars = 0
        for message in messages:
            if not message:
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
        
        # Rough approximation: 4 characters per token
        return total_chars // 4