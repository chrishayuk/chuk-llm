# chuk_llm/api/providers.py
"""Dynamically generated provider shortcuts from configuration."""

import functools
import asyncio
import os
import re
import logging
import threading
import atexit
from typing import Dict, Optional, List, AsyncIterator
from .provider_utils import get_all_providers, get_provider_config, find_providers_yaml_path

# Set up logger for this module
logger = logging.getLogger(__name__)

# Global persistent event loop for sync operations
_loop_thread = None
_persistent_loop = None
_loop_lock = threading.Lock()

def _get_persistent_loop():
    """Get or create a persistent event loop running in a background thread."""
    global _loop_thread, _persistent_loop
    
    with _loop_lock:
        if _persistent_loop is None or _persistent_loop.is_closed():
            # Create a new event loop in a background thread
            loop_ready = threading.Event()
            
            def run_loop():
                global _persistent_loop
                _persistent_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_persistent_loop)
                loop_ready.set()
                _persistent_loop.run_forever()
            
            _loop_thread = threading.Thread(target=run_loop, daemon=True)
            _loop_thread.start()
            loop_ready.wait()  # Wait for loop to be ready
        
        return _persistent_loop

def _cleanup_persistent_loop():
    """Clean up the persistent loop on exit."""
    global _persistent_loop, _loop_thread
    
    if _persistent_loop and not _persistent_loop.is_closed():
        # Schedule the loop to stop
        _persistent_loop.call_soon_threadsafe(_persistent_loop.stop)
        
        # Give it a moment to clean up
        if _loop_thread and _loop_thread.is_alive():
            _loop_thread.join(timeout=1.0)

# Register cleanup on exit
atexit.register(_cleanup_persistent_loop)

def _create_provider_function(provider_name: str, model_name: Optional[str] = None):
    """Create async provider function."""
    if model_name:
        # ask_openai_gpt4o()
        async def provider_model_func(prompt: str, **kwargs) -> str:
            from .core import ask  # Import here to avoid circular import
            return await ask(prompt, provider=provider_name, model=model_name, **kwargs)
        return provider_model_func
    else:
        # ask_openai()
        async def provider_func(prompt: str, model: Optional[str] = None, **kwargs) -> str:
            from .core import ask  # Import here to avoid circular import
            return await ask(prompt, provider=provider_name, model=model, **kwargs)
        return provider_func

def _create_stream_function(provider_name: str, model_name: Optional[str] = None):
    """Create async streaming function."""
    if model_name:
        # stream_openai_gpt4o()
        async def stream_model_func(prompt: str, **kwargs) -> AsyncIterator[str]:
            from .core import stream  # Import here to avoid circular import
            async for chunk in stream(prompt, provider=provider_name, model=model_name, **kwargs):
                yield chunk
        return stream_model_func
    else:
        # stream_openai()
        async def stream_func(prompt: str, model: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
            from .core import stream  # Import here to avoid circular import
            async for chunk in stream(prompt, provider=provider_name, model=model, **kwargs):
                yield chunk
        return stream_func

def _run_async_on_persistent_loop(coro):
    """Run async coroutine on the persistent background loop."""
    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot call sync functions from async context. "
            "Use the async version instead."
        )
    except RuntimeError as e:
        if "Cannot call sync functions" in str(e):
            raise e
    
    # Get the persistent loop
    loop = _get_persistent_loop()
    
    # Run the coroutine on the persistent loop and wait for result
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

def _create_sync_function(provider_name: str, model_name: Optional[str] = None):
    """Create sync provider function using persistent loop."""
    if model_name:
        # ask_openai_gpt4o_sync()
        def sync_model_func(prompt: str, **kwargs) -> str:
            from .core import ask  # Import here to avoid circular import
            return _run_async_on_persistent_loop(ask(prompt, provider=provider_name, model=model_name, **kwargs))
        return sync_model_func
    else:
        # ask_openai_sync()
        def sync_func(prompt: str, model: Optional[str] = None, **kwargs) -> str:
            from .core import ask  # Import here to avoid circular import
            # Handle model parameter properly
            if model is not None:
                kwargs['model'] = model
            return _run_async_on_persistent_loop(ask(prompt, provider=provider_name, **kwargs))
        return sync_func

def _sanitize_model_name(model_name: str) -> str:
    """Convert model name to valid Python function name.
    
    Examples:
        'gpt-4o-mini' -> 'gpt4o_mini'
        'claude-3-sonnet-20240229' -> 'claude3_sonnet_20240229'
        'llama-3.3-70b-versatile' -> 'llama33_70b_versatile'
    """
    if not model_name:
        return ""
    
    # Replace hyphens with underscores, but keep dots for now
    sanitized = model_name.replace('-', '_')
    
    # Remove dots only when they're between numbers (like "3.3" -> "33")
    import re
    sanitized = re.sub(r'(\d)\.(\d)', r'\1\2', sanitized)
    
    # Remove any remaining non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"model_{sanitized}"
    
    return sanitized.lower()

def _get_common_models_for_provider(provider: str) -> List[str]:
    """Get models for a provider from YAML configuration.
    
    This reads from providers.yaml - no fallbacks to hardcoded values.
    """
    # Get provider configuration from YAML
    provider_config = get_provider_config(provider)
    
    if not provider_config:
        logger.debug(f"No configuration found for provider '{provider}'")
        return []
    
    # Method 1: Direct 'models' key (simple list)
    if 'models' in provider_config:
        models_section = provider_config['models']
        if isinstance(models_section, list):
            # Handle both simple strings and complex model objects
            extracted_models = []
            for model in models_section:
                if isinstance(model, dict) and 'name' in model:
                    extracted_models.append(model['name'])
                elif isinstance(model, str):
                    extracted_models.append(model)
            
            if extracted_models:
                logger.debug(f"Found {len(extracted_models)} models for {provider}: {extracted_models}")
                return extracted_models
    
    # No models found in configuration
    logger.debug(f"No models found in configuration for provider '{provider}'")
    return []

def _get_model_aliases_for_provider(provider: str) -> Dict[str, str]:
    """Get model aliases for a provider from YAML configuration.
    
    Returns:
        Dictionary mapping alias names to actual model names
    """
    provider_config = get_provider_config(provider)
    
    if not provider_config:
        return {}
    
    # Get model aliases from YAML
    if 'model_aliases' in provider_config:
        aliases = provider_config['model_aliases']
        if isinstance(aliases, dict):
            logger.debug(f"Found {len(aliases)} model aliases for {provider}: {list(aliases.keys())}")
            return aliases
    
    return {}

def _get_global_aliases() -> Dict[str, str]:
    """Get global aliases from YAML configuration.
    
    Returns:
        Dictionary mapping alias names to provider/model combinations
    """
    try:
        import yaml
        
        # Use the existing path finding logic
        config_path = find_providers_yaml_path()
        if not config_path:
            logger.debug("No providers.yaml found for global aliases")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config and '__global_aliases__' in config:
            global_aliases = config['__global_aliases__']
            logger.debug(f"Found {len(global_aliases)} global aliases: {list(global_aliases.keys())}")
            return global_aliases
                
    except Exception as e:
        logger.debug(f"Could not read global aliases: {e}")
    
    return {}

def _create_global_alias_function(alias_name: str, provider_model: str):
    """Create a global alias function that calls a specific provider/model.
    
    Args:
        alias_name: The alias name (e.g., 'gpt4')
        provider_model: The provider/model string (e.g., 'openai/gpt-4o')
    
    Returns:
        Dictionary with the generated functions
    """
    if '/' not in provider_model:
        logger.warning(f"Invalid global alias format: {provider_model}, expected 'provider/model'")
        return {}
    
    provider, model = provider_model.split('/', 1)
    
    # Create async function
    async def global_alias_func(prompt: str, **kwargs) -> str:
        from .core import ask
        return await ask(prompt, provider=provider, model=model, **kwargs)
    
    # Create sync function
    def global_alias_sync_func(prompt: str, **kwargs) -> str:
        return _run_async_on_persistent_loop(global_alias_func(prompt, **kwargs))
    
    # Create stream function
    async def global_alias_stream_func(prompt: str, **kwargs) -> AsyncIterator[str]:
        from .core import stream
        async for chunk in stream(prompt, provider=provider, model=model, **kwargs):
            yield chunk
    
    return {
        f"ask_{alias_name}": global_alias_func,
        f"ask_{alias_name}_sync": global_alias_sync_func,
        f"stream_{alias_name}": global_alias_stream_func,
    }

def _generate_all_functions():
    """Generate all provider functions dynamically."""
    providers = get_all_providers()
    functions = {}
    
    logger.debug(f"Found {len(providers)} providers: {providers}")
    
    for provider in providers:
        logger.debug(f"Generating functions for provider: {provider}")
        
        # Generate base provider functions (using default model)
        # ask_openai(), stream_openai(), ask_openai_sync()
        functions[f"ask_{provider}"] = _create_provider_function(provider)
        functions[f"stream_{provider}"] = _create_stream_function(provider)
        functions[f"ask_{provider}_sync"] = _create_sync_function(provider)
        
        # Generate model-specific functions for models from YAML
        models = _get_common_models_for_provider(provider)
        
        for model in models:
            model_suffix = _sanitize_model_name(model)
            if model_suffix:  # Only create if we have a valid suffix
                logger.debug(f"Creating model-specific functions for {provider}_{model_suffix}")
                # ask_openai_gpt4o(), ask_anthropic_claude3_sonnet(), etc.
                functions[f"ask_{provider}_{model_suffix}"] = _create_provider_function(provider, model)
                functions[f"stream_{provider}_{model_suffix}"] = _create_stream_function(provider, model)
                functions[f"ask_{provider}_{model_suffix}_sync"] = _create_sync_function(provider, model)
        
        # Generate functions for model aliases from YAML
        model_aliases = _get_model_aliases_for_provider(provider)
        
        for alias, actual_model in model_aliases.items():
            alias_suffix = _sanitize_model_name(alias)
            if alias_suffix:  # Only create if we have a valid suffix
                logger.debug(f"Creating alias functions for {provider}_{alias_suffix} -> {actual_model}")
                # ask_openai_gpt4o() (alias) -> gpt-4o (actual model)
                functions[f"ask_{provider}_{alias_suffix}"] = _create_provider_function(provider, actual_model)
                functions[f"stream_{provider}_{alias_suffix}"] = _create_stream_function(provider, actual_model)
                functions[f"ask_{provider}_{alias_suffix}_sync"] = _create_sync_function(provider, actual_model)
    
    # Generate global alias functions
    global_aliases = _get_global_aliases()
    logger.debug(f"Processing {len(global_aliases)} global aliases")
    
    for alias_name, provider_model in global_aliases.items():
        alias_functions = _create_global_alias_function(alias_name, provider_model)
        if alias_functions:
            functions.update(alias_functions)
            logger.debug(f"Created global alias functions for {alias_name} -> {provider_model}")
    
    # Set proper names and docs for all generated functions
    for name, func in functions.items():
        func.__name__ = name
        
        # Extract provider and model info for documentation
        parts = name.replace('ask_', '').replace('stream_', '').replace('_sync', '')
        
        if name.startswith("ask_") and name.endswith("_sync"):
            if '_' in parts and len(parts.split('_')) > 1:
                provider_part = parts.split('_')[0]
                model_part = '_'.join(parts.split('_')[1:])
                func.__doc__ = f"Synchronous {provider_part} with model {model_part}."
            else:
                func.__doc__ = f"Synchronous {parts} (default model)."
        elif name.startswith("ask_"):
            if '_' in parts and len(parts.split('_')) > 1:
                provider_part = parts.split('_')[0]
                model_part = '_'.join(parts.split('_')[1:])
                func.__doc__ = f"Ask {provider_part} with model {model_part}."
            else:
                func.__doc__ = f"Ask {parts} (default model)."
        elif name.startswith("stream_"):
            if '_' in parts and len(parts.split('_')) > 1:
                provider_part = parts.split('_')[0]
                model_part = '_'.join(parts.split('_')[1:])
                func.__doc__ = f"Stream from {provider_part} with model {model_part}."
            else:
                func.__doc__ = f"Stream from {parts} (default model)."
    
    return functions

def _create_utility_functions(available_functions):
    """Create utility functions dynamically based on available providers."""
    utils = {}
    
    # Create quick_question function
    def quick_question(question: str, provider: str = "openai") -> str:
        """Ultra-simple function for one-off questions."""
        from .sync import ask_sync
        return ask_sync(question, provider=provider)
    
    # Create compare_providers function
    def compare_providers(question: str, providers: list = None) -> dict:
        """Compare responses from multiple providers."""
        if providers is None:
            providers = ["openai", "anthropic"]
        
        from .sync import ask_sync
        
        results = {}
        for provider in providers:
            try:
                results[provider] = ask_sync(question, provider=provider)
            except Exception as e:
                results[provider] = f"Error: {str(e)}"
        
        return results
    
    utils['quick_question'] = quick_question
    utils['compare_providers'] = compare_providers
    
    logger.debug(f"Created {len(utils)} utility functions")
    
    return utils

# Generate all functions at import time
logger.info("Generating provider functions from YAML configuration...")
_all_functions = _generate_all_functions()

# Debug: Log what functions we generated
base_functions = [name for name in _all_functions.keys() if '_' not in name.replace('ask_', '').replace('stream_', '').replace('_sync', '') or name.endswith('_sync') and name.count('_') <= 2]
model_functions = [name for name in _all_functions.keys() if name not in base_functions]
global_functions = [name for name in _all_functions.keys() if any(alias in name for alias in ['gpt4', 'claude', 'llama', 'gemini', 'mistral', 'deepseek', 'qwen', 'granite']) and not any(provider in name for provider in ['openai', 'anthropic', 'groq', 'ollama'])]

logger.info(f"Generated {len(base_functions)} base provider functions")
logger.info(f"Generated {len(model_functions)} model-specific functions")
logger.info(f"Generated {len(global_functions)} global alias functions")
logger.info(f"Total: {len(_all_functions)} provider functions")

# Create and add utility functions
_utility_functions = _create_utility_functions(_all_functions)

# Combine all functions
_all_functions.update(_utility_functions)

# Add to module namespace
globals().update(_all_functions)

# Export everything
__all__ = list(_all_functions.keys())

logger.info(f"Final total: {len(__all__)} functions (including {len(_utility_functions)} utilities)")

# Log some examples of what was generated
examples = [name for name in __all__ if any(x in name for x in ['gpt4o', 'claude', 'llama']) and not name.endswith('_sync')][:5]
if examples:
    logger.info(f"Example functions available: {', '.join(examples)}")
else:
    logger.info("No model-specific functions generated (check your providers.yaml configuration)")

# Log global alias examples
global_examples = [name for name in __all__ if any(alias in name for alias in ['gpt4', 'claude', 'llama']) and not any(provider in name for provider in ['openai', 'anthropic', 'groq']) and not name.endswith('_sync')][:3]
if global_examples:
    logger.info(f"Global alias functions available: {', '.join(global_examples)}")