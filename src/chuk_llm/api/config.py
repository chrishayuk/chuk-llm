# src/chuk_llm/api/config.py
"""Configuration management for the simple API - NO circular imports."""

from typing import Dict, Any

# Global configuration
_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "system_prompt": None,
    "temperature": None,
    "max_tokens": None,
    "api_key": None,
    "api_base": None,
}

# Cached client for performance
_cached_client = None
_cached_config_hash = None

def configure(
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    api_key: str = None,
    api_base: str = None,
    **kwargs
):
    """Configure global defaults for the LLM interface."""
    global _config, _cached_client, _cached_config_hash
    
    # Update config with non-None values
    updates = {
        k: v for k, v in {
            "provider": provider,
            "model": model,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key,
            "api_base": api_base,
        }.items() if v is not None
    }
    
    _config.update(updates)
    _config.update(kwargs)
    
    # Invalidate cached client since config changed
    _cached_client = None
    _cached_config_hash = None

def get_config() -> Dict[str, Any]:
    """Get current configuration."""
    return _config.copy()

def reset_config():
    """Reset to default configuration."""
    global _config, _cached_client, _cached_config_hash
    _config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "system_prompt": None,
        "temperature": None,
        "max_tokens": None,
        "api_key": None,
        "api_base": None,
    }
    _cached_client = None
    _cached_config_hash = None

def get_client_for_config(config: Dict[str, Any]):
    """Get or create client for the given config."""
    global _cached_client, _cached_config_hash
    
    # Create a hash of the relevant config for caching
    config_key = (
        config.get("provider"),
        config.get("model"),
        config.get("api_key"),
        config.get("api_base"),
    )
    
    # Return cached client if config hasn't changed
    if _cached_client and _cached_config_hash == config_key:
        return _cached_client
    
    # Create client using your existing basic factory
    from chuk_llm.llm.llm_client import get_llm_client
    client = get_llm_client(
        provider=config["provider"],
        model=config["model"],
        api_key=config.get("api_key"),
        api_base=config.get("api_base"),
    )
    
    # Cache the client
    _cached_client = client
    _cached_config_hash = config_key
    
    return client

def get_current_config():
    """Get the current global config (internal)."""
    return _config