# src/chuk_llm/api/providers.py
"""Dynamically generated provider shortcuts from configuration."""

import functools
import asyncio
import os
import re
from typing import Optional, List, AsyncIterator
from .provider_utils import get_all_providers, get_provider_config

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

def _create_sync_function(provider_name: str, model_name: Optional[str] = None):
    """Create sync provider function."""
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                f"Cannot call sync functions from async context. "
                f"Use ask_{provider_name}() instead of ask_{provider_name}_sync()"
            )
        except RuntimeError as e:
            if "Cannot call sync functions" in str(e):
                raise e
            return asyncio.run(coro)
    
    if model_name:
        # ask_openai_gpt4o_sync()
        def sync_model_func(prompt: str, **kwargs) -> str:
            from .core import ask  # Import here to avoid circular import
            return _run_async(ask(prompt, provider=provider_name, model=model_name, **kwargs))
        return sync_model_func
    else:
        # ask_openai_sync()
        def sync_func(prompt: str, model: Optional[str] = None, **kwargs) -> str:
            from .core import ask  # Import here to avoid circular import
            # Handle model parameter properly
            if model is not None:
                kwargs['model'] = model
            return _run_async(ask(prompt, provider=provider_name, **kwargs))
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
    
    # Replace hyphens and dots with underscores
    sanitized = model_name.replace('-', '_').replace('.', '')
    
    # Remove any non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"model_{sanitized}"
    
    return sanitized.lower()

def _get_common_models_for_provider(provider: str) -> List[str]:
    """Get common models for a provider to generate specific functions.
    
    This creates convenience functions for popular models.
    """
    common_models = {
        'openai': [
            'gpt-4o',
            'gpt-4o-mini', 
            'gpt-4-turbo',
            'gpt-3.5-turbo'
        ],
        'anthropic': [
            'claude-3-5-sonnet-20241022',
            'claude-3-sonnet-20240229',
            'claude-3-opus-20240229',
            'claude-3-haiku-20240307'
        ],
        'groq': [
            'llama-3.3-70b-versatile',
            'llama-3.1-8b-instant',
            'mixtral-8x7b-32768'
        ],
        'deepseek': [
            'deepseek-chat',
            'deepseek-reasoner'
        ],
        'gemini': [
            'gemini-2.0-flash',
            'gemini-1.5-pro',
            'gemini-1.5-flash'
        ],
        'mistral': [
            'mistral-large-latest',
            'mistral-medium-latest',
            'mistral-small-latest'
        ],
        'perplexity': [
            'sonar-pro',
            'sonar-reasoning'
        ]
    }
    
    return common_models.get(provider, [])

def _generate_all_functions():
    """Generate all provider functions dynamically."""
    providers = get_all_providers()
    functions = {}
    
    for provider in providers:
        # Generate base provider functions (using default model)
        # ask_openai(), stream_openai(), ask_openai_sync()
        functions[f"ask_{provider}"] = _create_provider_function(provider)
        functions[f"stream_{provider}"] = _create_stream_function(provider)
        functions[f"ask_{provider}_sync"] = _create_sync_function(provider)
        
        # Generate model-specific functions for common models
        common_models = _get_common_models_for_provider(provider)
        for model in common_models:
            model_suffix = _sanitize_model_name(model)
            if model_suffix:  # Only create if we have a valid suffix
                # ask_openai_gpt4o(), ask_anthropic_claude3_sonnet(), etc.
                functions[f"ask_{provider}_{model_suffix}"] = _create_provider_function(provider, model)
                functions[f"stream_{provider}_{model_suffix}"] = _create_stream_function(provider, model)
                functions[f"ask_{provider}_{model_suffix}_sync"] = _create_sync_function(provider, model)
    
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
    
    # DON'T create ask_sync alias here - it's defined in sync.py
    # This was causing the conflict!
    
    # Create convenient aliases for popular providers
    provider_aliases = {
        'ask_claude': 'ask_anthropic',
        'ask_claude_sync': 'ask_anthropic_sync',
        'stream_claude': 'stream_anthropic',
        'ask_google': 'ask_gemini',
        'ask_google_sync': 'ask_gemini_sync',
        'stream_google': 'stream_gemini',
        'ask_granite': 'ask_watsonx',
        'ask_granite_sync': 'ask_watsonx_sync',
    }
    
    # Add model-specific aliases for popular combinations
    model_aliases = {
        'ask_gpt4o': 'ask_openai_gpt4o',
        'ask_gpt4o_mini': 'ask_openai_gpt4o_mini',
        'ask_claude_sonnet': 'ask_anthropic_claude3_5_sonnet_20241022',
        'ask_claude_opus': 'ask_anthropic_claude3_opus_20240229',
        'ask_llama': 'ask_groq_llama33_70b_versatile',
    }
    
    # Combine aliases
    all_aliases = {**provider_aliases, **model_aliases}
    
    for alias, target in all_aliases.items():
        if target in available_functions:
            utils[alias] = available_functions[target]
    
    # Create quick_question function - but import ask_sync from sync.py
    def quick_question(question: str, provider: str = "openai") -> str:
        """Ultra-simple function for one-off questions."""
        # Import the real ask_sync function from sync.py
        from .sync import ask_sync
        return ask_sync(question, provider=provider)
    
    # Create compare_providers function - but import ask_sync from sync.py
    def compare_providers(question: str, providers: list = None) -> dict:
        """Compare responses from multiple providers."""
        if providers is None:
            providers = ["openai", "anthropic"]
        
        # Import the real ask_sync function from sync.py
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
    
    return utils

# Generate all functions at import time
print("Generating provider functions...")
_all_functions = _generate_all_functions()

# Debug: Print what functions we generated
base_functions = [name for name in _all_functions.keys() if '_' not in name.replace('ask_', '').replace('stream_', '').replace('_sync', '') or name.endswith('_sync') and name.count('_') <= 2]
model_functions = [name for name in _all_functions.keys() if name not in base_functions]

print(f"Generated {len(base_functions)} base provider functions")
print(f"Generated {len(model_functions)} model-specific functions")
print(f"Total: {len(_all_functions)} provider functions")

# Create and add utility functions
_utility_functions = _create_utility_functions(_all_functions)

# Combine all functions
_all_functions.update(_utility_functions)

# Add to module namespace
globals().update(_all_functions)

# Export everything
__all__ = list(_all_functions.keys())

print(f"Final total: {len(__all__)} functions (including {len(_utility_functions)} utilities)")

# Show some examples of what was generated
print("\n🎯 Example functions available:")
examples = [name for name in __all__ if any(x in name for x in ['gpt4o', 'claude', 'llama']) and not name.endswith('_sync')][:5]
for example in examples:
    print(f"  - {example}()")
print("  ... and many more!")