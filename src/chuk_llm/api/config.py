# chuk_llm/api/config.py
"""
API-level configuration management - FIXED
=================================

Simple, clean configuration for the API layer.
Uses the dynamic configuration system.
"""

from typing import Dict, Any, Optional


class APIConfig:
    """API-level configuration manager"""
    
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
        """Get current effective configuration"""
        # CRITICAL FIX: Import inside method so mocking works
        from chuk_llm.configuration.config import get_config
        
        config_manager = get_config()
        global_settings = config_manager.get_global_settings()
        
        # Start with global defaults
        result = {
            "provider": global_settings.get("active_provider", "openai"),
            "model": None,
            "system_prompt": None,
            "temperature": None,
            "max_tokens": None,
            "api_key": None,
            "api_base": None,
        }
        
        # Apply overrides
        result.update(self.overrides)
        
        provider_name = result["provider"]
        
        # Resolve provider-specific defaults
        try:
            provider = config_manager.get_provider(provider_name)
            if result["model"] is None:
                result["model"] = provider.default_model
            if result["api_base"] is None:
                result["api_base"] = provider.api_base
        except:
            pass
        
        # FIX: Always resolve API key for the CURRENT provider, not cached provider
        if result["api_key"] is None:
            try:
                # CRITICAL: Use current provider_name, not old cached value
                result["api_key"] = config_manager.get_api_key(provider_name)
            except ValueError:
                # Provider doesn't exist, leave api_key as None
                pass
            except Exception:
                # Any other error, leave api_key as None
                pass
        
        return result
    
    def get_client(self):
        """Get LLM client with current configuration"""
        config = self.get_current_config()
        
        # FIX: Include provider in cache key to prevent cross-contamination
        cache_key = (
            config["provider"],  # This was missing proper provider resolution
            config["model"], 
            config["api_key"],
            config["api_base"]
        )
        
        if self._cached_client and self._cache_key == cache_key:
            return self._cached_client
        
        # Create new client
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
        
        return client
    
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


def reset():
    """Reset configuration"""
    _api_config.reset()

# DEBUGGING ADDITION: Add function to check config state
def debug_config_state():
    """Debug current configuration state"""
    config = get_current_config()
    print("🔍 Current API Config State:")
    print(f"   Provider: {config['provider']}")
    print(f"   Model: {config['model']}")
    print(f"   API Key: {config['api_key'][:10] if config['api_key'] else 'None'}...")
    print(f"   API Base: {config['api_base']}")
    print(f"   Cache Key: {_api_config._cache_key}")
    return config