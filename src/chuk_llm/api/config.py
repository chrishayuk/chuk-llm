# chuk_llm/api/config.py
"""
API-level configuration management with unified config
====================================================

Simple, clean configuration for the API layer using the unified configuration system.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class APIConfig:
    """API-level configuration manager using unified config"""
    
    def __init__(self):
        self.overrides: Dict[str, Any] = {}
        self._cached_client = None
        self._cache_key = None
    
    def set(self, **kwargs):
        """Set configuration overrides"""
        # Only update with non-None values
        for key, value in kwargs.items():
            if value is not None:
                self.overrides[key] = value
        self._invalidate_cache()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current effective configuration using unified config"""
        # Import inside method for better testability
        from chuk_llm.configuration import get_config
        
        try:
            config_manager = get_config()
            global_settings = config_manager.global_settings
        except Exception as e:
            logger.warning(f"Could not load unified config: {e}")
            global_settings = {}
        
        # Start with global defaults
        result = {
            "provider": global_settings.get("active_provider", "openai"),
            "model": None,
            "system_prompt": None,
            "temperature": None,
            "max_tokens": None,
            "api_key": None,
            "api_base": None,
            "stream": False,
            "tools": None,
            "json_mode": False,
        }
        
        # Apply overrides
        result.update(self.overrides)
        
        provider_name = result["provider"]
        
        # Resolve provider-specific defaults
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(provider_name)
            
            # Set model default
            if result["model"] is None:
                result["model"] = provider_config.default_model
            
            # Set API base default
            if result["api_base"] is None:
                result["api_base"] = provider_config.api_base
            
            # Resolve API key for current provider
            if result["api_key"] is None:
                result["api_key"] = config_manager.get_api_key(provider_name)
            
            # Add provider capabilities info for reference
            model_caps = provider_config.get_model_capabilities(result["model"])
            result["_capabilities"] = {
                "features": [f.value for f in model_caps.features],
                "max_context_length": model_caps.max_context_length,
                "max_output_tokens": model_caps.max_output_tokens,
            }
            
        except Exception as e:
            logger.warning(f"Could not resolve provider config for '{provider_name}': {e}")
            # Set minimal defaults for unknown providers
            if result["model"] is None:
                result["model"] = "default"
        
        return result
    
    def get_client(self):
        """Get LLM client with current configuration"""
        config = self.get_current_config()
        
        # Create cache key with all relevant parameters
        cache_key = (
            config["provider"],
            config["model"], 
            config["api_key"],
            config["api_base"]
        )
        
        if self._cached_client and self._cache_key == cache_key:
            return self._cached_client
        
        # Create new client using unified client factory
        try:
            from chuk_llm.llm.client import get_client
            client = get_client(
                provider=config["provider"],
                model=config["model"],
                api_key=config["api_key"],
                api_base=config["api_base"]
            )
            
            # Cache it
            self._cached_client = client
            self._cache_key = cache_key
            
            logger.debug(f"Created API client: {config['provider']}/{config['model']}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create client: {e}")
            raise
    
    def validate_current_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        from chuk_llm.configuration.unified_config import ConfigValidator
        
        config = self.get_current_config()
        
        try:
            is_valid, issues = ConfigValidator.validate_request_compatibility(
                provider_name=config["provider"],
                model=config["model"],
                tools=config.get("tools"),
                stream=config.get("stream", False)
            )
            
            return {
                "valid": is_valid,
                "issues": issues,
                "config": config
            }
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {e}"],
                "config": config
            }
    
    def get_provider_capabilities(self) -> Dict[str, Any]:
        """Get capabilities for current provider/model"""
        config = self.get_current_config()
        
        try:
            from chuk_llm.llm.client import get_provider_info
            return get_provider_info(config["provider"], config["model"])
        except Exception as e:
            return {"error": str(e)}
    
    def supports_feature(self, feature: str) -> bool:
        """Check if current provider/model supports a feature"""
        from chuk_llm.configuration.unified_config import Feature
        
        config = self.get_current_config()
        
        try:
            from chuk_llm.configuration.unified_config import get_config
            config_manager = get_config()
            return config_manager.supports_feature(
                config["provider"], 
                Feature.from_string(feature), 
                config["model"]
            )
        except Exception as e:
            logger.warning(f"Could not check feature support: {e}")
            return False
    
    def auto_configure_for_task(self, task_type: str = "general", **requirements):
        """Auto-configure for specific task types"""
        # Simple implementation - just use first available provider
        # This would need more sophisticated logic in a real implementation
        try:
            from chuk_llm.configuration.unified_config import get_config
            config_manager = get_config()
            providers = config_manager.get_all_providers()
            
            if providers:
                self.set(provider=providers[0])
                logger.info(f"Auto-configured for {task_type}: {providers[0]}")
                return True
            else:
                logger.warning(f"No providers available for task: {task_type}")
                return False
        except Exception as e:
            logger.error(f"Auto-configuration failed: {e}")
            return False
    
    def _invalidate_cache(self):
        """Invalidate cached client"""
        self._cached_client = None
        self._cache_key = None
    
    def reset(self):
        """Reset to defaults"""
        self.overrides.clear()
        self._invalidate_cache()


# Global API config instance
_api_config = APIConfig()


def configure(**kwargs):
    """Configure API defaults"""
    _api_config.set(**kwargs)


def get_current_config() -> Dict[str, Any]:
    """Get current configuration"""
    return _api_config.get_current_config()


def get_client():
    """Get client with current configuration"""
    return _api_config.get_client()


def validate_config() -> Dict[str, Any]:
    """Validate current configuration"""
    return _api_config.validate_current_config()


def get_capabilities() -> Dict[str, Any]:
    """Get current provider/model capabilities"""
    return _api_config.get_provider_capabilities()


def supports_feature(feature: str) -> bool:
    """Check if current configuration supports a feature"""
    return _api_config.supports_feature(feature)


def auto_configure(task_type: str = "general", **requirements) -> bool:
    """Auto-configure for specific task type"""
    return _api_config.auto_configure_for_task(task_type, **requirements)


def reset():
    """Reset configuration"""
    _api_config.reset()


def debug_config_state() -> Dict[str, Any]:
    """Debug current configuration state with enhanced info"""
    config = get_current_config()
    capabilities = get_capabilities()
    validation = validate_config()
    
    debug_info = {
        "config": {
            "provider": config["provider"],
            "model": config["model"],
            "has_api_key": bool(config["api_key"]),
            "api_base": config["api_base"],
            "overrides": _api_config.overrides
        },
        "capabilities": capabilities.get("supports", {}),
        "validation": {
            "valid": validation["valid"],
            "issues": validation["issues"]
        },
        "cache_key": _api_config._cache_key
    }
    
    print("🔍 Enhanced API Config State:")
    print(f"   Provider: {config['provider']}")
    print(f"   Model: {config['model']}")
    print(f"   API Key: {'✓' if config['api_key'] else '✗'}")
    print(f"   Valid Config: {'✓' if validation['valid'] else '✗'}")
    if not validation['valid']:
        print(f"   Issues: {', '.join(validation['issues'])}")
    print(f"   Capabilities: {list(capabilities.get('supports', {}).keys())}")
    
    return debug_info


# Enhanced convenience functions
def quick_setup(provider: str, model: Optional[str] = None, **kwargs):
    """Quickly setup a provider/model combination"""
    configure(provider=provider, model=model, **kwargs)
    
    # Validate the setup
    validation = validate_config()
    if not validation["valid"]:
        logger.warning(f"Setup issues: {', '.join(validation['issues'])}")
    
    return validation["valid"]


def list_available_setups() -> Dict[str, Any]:
    """List all available provider/model combinations"""
    from chuk_llm.llm.client import list_available_providers
    return list_available_providers()


def switch_provider(provider: str, model: Optional[str] = None):
    """Switch to a different provider/model"""
    old_config = get_current_config()
    
    try:
        configure(provider=provider, model=model)
        validation = validate_config()
        
        if validation["valid"]:
            logger.info(f"Switched from {old_config['provider']}/{old_config['model']} to {provider}/{model or 'default'}")
            return True
        else:
            # Revert on validation failure
            configure(provider=old_config["provider"], model=old_config["model"])
            logger.error(f"Failed to switch to {provider}: {', '.join(validation['issues'])}")
            return False
    except Exception as e:
        # Revert on any error
        configure(provider=old_config["provider"], model=old_config["model"])
        logger.error(f"Error switching to {provider}: {e}")
        return False