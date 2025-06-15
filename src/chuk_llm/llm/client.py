"""
Clean LLM client factory
========================

Simple client factory using dynamic configuration.
"""

import importlib
import inspect
import logging
from typing import Optional, Type, Any, Dict

from chuk_llm.configuration.config import get_config
from chuk_llm.llm.core.base import BaseLLMClient

logger = logging.getLogger(__name__)


def _import_string(import_string: str) -> Type:
    """Import class from string path.
    
    Supports both colon syntax (module:class) and dot syntax (module.class).
    """
    if ':' in import_string:
        module_path, class_name = import_string.split(':', 1)
    else:
        module_path, class_name = import_string.rsplit('.', 1)
    
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _import_class(class_path: str) -> Type:
    """Import class from string path (alias for _import_string)"""
    return _import_string(class_path)


def _supports_param(cls: Type, param_name: str) -> bool:
    """Check if class constructor supports a parameter"""
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    
    # Check if parameter exists directly
    if param_name in params:
        return True
    
    # Check if **kwargs parameter exists
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return has_kwargs


def _constructor_kwargs(cls: Type, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get constructor arguments for client class from config"""
    # Get constructor signature
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    
    # Map config to constructor arguments
    args = {}
    
    # Common parameter mappings
    if 'model' in params and config.get('model'):
        args['model'] = config['model']
    
    if 'api_key' in params and config.get('api_key'):
        args['api_key'] = config['api_key']
    
    if 'api_base' in params and config.get('api_base'):
        args['api_base'] = config['api_base']
    
    # Check for **kwargs parameter
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    
    if has_kwargs:
        # Add all config values if constructor accepts **kwargs
        for key, value in config.items():
            if key not in args and value is not None:
                args[key] = value
    else:
        # Only add values for known parameters
        for key, value in config.items():
            if key in params and key not in args and value is not None:
                args[key] = value
    
    return args


def _get_constructor_args(cls: Type, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get constructor arguments for client class (alias for _constructor_kwargs)"""
    return _constructor_kwargs(cls, config)


def get_client(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Get LLM client for provider.
    
    Args:
        provider: Provider name (from YAML config)
        model: Model override (uses provider default if not specified)
        api_key: API key override (uses environment if not specified)
        api_base: API base URL override (uses provider default if not specified)
        **kwargs: Additional client arguments
    
    Returns:
        Configured LLM client
    """
    # Try to get config manager, but don't fail if it's not available
    provider_config = None
    config_manager = None
    
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider)
    except Exception as e:
        # If we can't get the provider from the config system, 
        # let the error propagate so test mocks work properly
        raise e
    
    if provider_config is None:
        raise ValueError(f"Unknown provider '{provider}'")
    
    # Build client configuration
    client_config = {
        'model': model or getattr(provider_config, 'default_model', 'gpt-4o-mini'),
        'api_key': api_key or _get_api_key_with_fallback(config_manager, provider, provider_config),
        'api_base': api_base or getattr(provider_config, 'api_base', None),
    }
    
    # Add extra provider config
    extra = getattr(provider_config, 'extra', {})
    client_config.update(extra)
    
    # Add explicit kwargs (highest priority)
    client_config.update(kwargs)
    
    # Get client class - be very explicit about this
    client_class = getattr(provider_config, 'client_class', None)
    
    # Handle both None and empty string
    if not client_class or client_class.strip() == '':
        raise ValueError(f"No client class configured for provider '{provider}'")
    
    try:
        ClientClass = _import_string(client_class)
    except Exception as e:
        raise ValueError(f"Failed to import client class '{client_class}': {e}")
    
    # Get constructor arguments
    constructor_args = _constructor_kwargs(ClientClass, client_config)
    
    # Create client instance
    try:
        client = ClientClass(**constructor_args)
        logger.debug(f"Created {provider} client with model {client_config.get('model')}")
        return client
    except Exception as e:
        raise ValueError(f"Failed to create {provider} client: {e}")


def _get_api_key_with_fallback(config_manager, provider: str, provider_config=None) -> Optional[str]:
    """Get API key with fallback for testing."""
    if config_manager and hasattr(config_manager, 'get_api_key'):
        try:
            return config_manager.get_api_key(provider)
        except:
            pass
    
    # Fallback to environment variables
    import os
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GOOGLE_API_KEY"
    }
    return os.environ.get(env_vars.get(provider))


def _get_test_provider_config(provider: str):
    """Get built-in test configuration for common providers."""
    
    # Define a simple but explicit config object
    if provider == "openai":
        class OpenAITestConfig:
            name = "openai"
            client_class = "chuk_llm.llm.providers.openai_client.OpenAILLMClient"
            default_model = "gpt-4o-mini"
            api_base = None
            extra = {}
        return OpenAITestConfig()
    
    elif provider == "anthropic":
        class AnthropicTestConfig:
            name = "anthropic"
            client_class = "chuk_llm.llm.providers.anthropic_client.AnthropicLLMClient"
            default_model = "claude-3-sonnet"
            api_base = None
            extra = {}
        return AnthropicTestConfig()
    
    return None


def list_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    List all available providers and their info.
    
    Returns:
        Dictionary with provider info
    """
    config_manager = get_config()
    providers = {}
    
    for provider_name in config_manager.get_all_providers():
        try:
            provider_config = config_manager.get_provider(provider_name)
            has_api_key = bool(config_manager.get_api_key(provider_name))
            
            providers[provider_name] = {
                'default_model': provider_config.default_model,
                'models': provider_config.models,
                'model_aliases': provider_config.model_aliases,
                'features': list(provider_config.features),
                'has_api_key': has_api_key,
                'api_base': provider_config.api_base,
            }
        except Exception as e:
            providers[provider_name] = {'error': str(e)}
    
    return providers


def validate_provider_setup(provider: str) -> Dict[str, Any]:
    """
    Validate provider setup and configuration.
    
    Args:
        provider: Provider name to validate
    
    Returns:
        Validation results
    """
    config_manager = get_config()
    
    try:
        provider_config = config_manager.get_provider(provider)
    except ValueError as e:
        return {
            'valid': False,
            'error': f"Provider not found: {e}",
        }
    
    issues = []
    warnings = []
    
    # Check client class
    if not provider_config.client_class:
        issues.append("No client class configured")
    else:
        try:
            _import_string(provider_config.client_class)
        except Exception as e:
            issues.append(f"Cannot import client class: {e}")
    
    # Check API key
    api_key = config_manager.get_api_key(provider)
    if not api_key:
        if provider_config.api_key_env:
            issues.append(f"API key not found in environment variable {provider_config.api_key_env}")
        else:
            warnings.append("No API key configuration")
    
    # Check default model
    if not provider_config.default_model:
        warnings.append("No default model configured")
    
    # Check models list
    if not provider_config.models:
        warnings.append("No models configured")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'has_api_key': bool(api_key),
        'default_model': provider_config.default_model,
        'models_count': len(provider_config.models),
        'aliases_count': len(provider_config.model_aliases),
        'features': list(provider_config.features),
    }