# chuk_llm/llm/client.py
"""
Clean LLM client factory with unified configuration
==================================================

Simple client factory using the new unified configuration system.
"""

import importlib
import inspect
import logging
from typing import Optional, Type, Any, Dict

from chuk_llm.configuration.unified_config import get_config, ConfigValidator, Feature
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


def get_client(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Get LLM client for provider using unified configuration.
    
    Args:
        provider: Provider name (from YAML config)
        model: Model override (uses provider default if not specified)
        api_key: API key override (uses environment if not specified)
        api_base: API base URL override (uses provider default if not specified)
        **kwargs: Additional client arguments
    
    Returns:
        Configured LLM client
    """
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider)
    except Exception as e:
        raise ValueError(f"Failed to get provider '{provider}': {e}")
    
    # Determine the model to use
    target_model = model or provider_config.default_model
    if not target_model:
        raise ValueError(f"No model specified and no default model for provider '{provider}'")
    
    # Validate provider configuration
    is_valid, issues = ConfigValidator.validate_provider_config(provider_config)
    if not is_valid:
        raise ValueError(f"Invalid provider configuration for '{provider}': {', '.join(issues)}")
    
    # Build client configuration
    client_config = {
        'model': target_model,
        'api_key': api_key or config_manager.get_api_key(provider),
        'api_base': api_base or provider_config.api_base,
    }
    
    # Add extra provider config
    client_config.update(provider_config.extra)
    
    # Add explicit kwargs (highest priority)
    client_config.update(kwargs)
    
    # Get client class
    client_class = provider_config.client_class
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
        logger.debug(f"Created {provider} client with model {target_model}")
        return client
    except Exception as e:
        raise ValueError(f"Failed to create {provider} client: {e}")


def validate_request_compatibility(
    provider: str,
    model: Optional[str] = None,
    messages: Optional[Any] = None,
    tools: Optional[Any] = None,
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate if a request is compatible with provider/model.
    
    Args:
        provider: Provider name
        model: Model name (optional)
        messages: Chat messages (for vision detection)
        tools: Function tools
        stream: Whether streaming is requested
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with validation results
    """
    try:
        is_valid, issues = ConfigValidator.validate_request_compatibility(
            provider_name=provider,
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            **kwargs
        )
        
        return {
            'valid': is_valid,
            'issues': issues,
            'provider': provider,
            'model': model
        }
    except Exception as e:
        return {
            'valid': False,
            'issues': [f"Validation error: {e}"],
            'provider': provider,
            'model': model
        }


def list_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    List all available providers and their info.
    
    Returns:
        Dictionary with provider info including model-level capabilities
    """
    config_manager = get_config()
    providers = {}
    
    for provider_name in config_manager.get_all_providers():
        try:
            provider_config = config_manager.get_provider(provider_name)
            has_api_key = bool(config_manager.get_api_key(provider_name))
            
            # Get model capabilities info
            model_info = {}
            for model in provider_config.models:
                model_caps = provider_config.get_model_capabilities(model)
                model_info[model] = {
                    'features': [f.value for f in model_caps.features],
                    'max_context_length': model_caps.max_context_length,
                    'max_output_tokens': model_caps.max_output_tokens
                }
            
            providers[provider_name] = {
                'default_model': provider_config.default_model,
                'models': provider_config.models,
                'model_aliases': provider_config.model_aliases,
                'baseline_features': [f.value for f in provider_config.features],
                'model_capabilities': model_info,
                'has_api_key': has_api_key,
                'api_base': provider_config.api_base,
                'rate_limits': provider_config.rate_limits
            }
        except Exception as e:
            providers[provider_name] = {'error': str(e)}
    
    return providers


def get_provider_info(provider: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed information about a specific provider/model combination.
    
    Args:
        provider: Provider name
        model: Model name (optional, uses default if not specified)
    
    Returns:
        Detailed provider/model information
    """
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider)
        
        target_model = model or provider_config.default_model
        model_caps = provider_config.get_model_capabilities(target_model)
        
        return {
            'provider': provider,
            'model': target_model,
            'client_class': provider_config.client_class,
            'api_base': provider_config.api_base,
            'has_api_key': bool(config_manager.get_api_key(provider)),
            'features': [f.value for f in model_caps.features],
            'max_context_length': model_caps.max_context_length,
            'max_output_tokens': model_caps.max_output_tokens,
            'rate_limits': provider_config.rate_limits,
            'available_models': provider_config.models,
            'model_aliases': provider_config.model_aliases,
            'supports': {
                'streaming': provider_config.supports_feature(Feature.STREAMING, target_model),
                'tools': provider_config.supports_feature(Feature.TOOLS, target_model),
                'vision': provider_config.supports_feature(Feature.VISION, target_model),
                'json_mode': provider_config.supports_feature(Feature.JSON_MODE, target_model),
                'parallel_calls': provider_config.supports_feature(Feature.PARALLEL_CALLS, target_model),
            }
        }
    except Exception as e:
        return {'error': str(e)}


def validate_provider_setup(provider: str) -> Dict[str, Any]:
    """
    Validate provider setup and configuration.
    
    Args:
        provider: Provider name to validate
    
    Returns:
        Comprehensive validation results
    """
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider)
    except Exception as e:
        return {
            'valid': False,
            'error': f"Provider not found: {e}",
        }
    
    # Validate provider config
    is_valid, config_issues = ConfigValidator.validate_provider_config(provider_config)
    
    warnings = []
    
    # Check client class import
    client_import_ok = True
    try:
        _import_string(provider_config.client_class)
    except Exception as e:
        client_import_ok = False
        config_issues.append(f"Cannot import client class: {e}")
    
    # Check API key
    api_key = config_manager.get_api_key(provider)
    has_api_key = bool(api_key)
    
    # Check models
    if not provider_config.models:
        warnings.append("No models configured")
    
    # Get model capabilities summary
    model_capabilities = {}
    for model in provider_config.models[:5]:  # Limit to first 5 for brevity
        caps = provider_config.get_model_capabilities(model)
        model_capabilities[model] = [f.value for f in caps.features]
    
    return {
        'valid': is_valid and client_import_ok,
        'issues': config_issues,
        'warnings': warnings,
        'has_api_key': has_api_key,
        'client_import_ok': client_import_ok,
        'default_model': provider_config.default_model,
        'models_count': len(provider_config.models),
        'aliases_count': len(provider_config.model_aliases),
        'baseline_features': [f.value for f in provider_config.features],
        'model_capabilities_sample': model_capabilities,
        'rate_limits': provider_config.rate_limits,
    }


def find_best_provider_for_request(
    required_features: Optional[list] = None,
    model_pattern: Optional[str] = None,
    exclude_providers: Optional[list] = None
) -> Optional[Dict[str, Any]]:
    """
    Find the best provider for a request with specific requirements.
    
    Args:
        required_features: List of required features (e.g., ['tools', 'vision'])
        model_pattern: Regex pattern to match model names
        exclude_providers: List of providers to exclude
    
    Returns:
        Best provider info or None if no match found
    """
    from chuk_llm.configuration.unified_config import CapabilityChecker
    
    if required_features:
        feature_set = {Feature.from_string(f) for f in required_features}
        exclude_set = set(exclude_providers or [])
        
        best_provider = CapabilityChecker.get_best_provider_for_features(
            feature_set, 
            model_name=model_pattern,
            exclude=exclude_set
        )
        
        if best_provider:
            return get_provider_info(best_provider)
    
    return None


# Backward compatibility aliases
_get_api_key_with_fallback = lambda config_manager, provider, provider_config=None: config_manager.get_api_key(provider)
_import_class = _import_string
_get_constructor_args = _constructor_kwargs