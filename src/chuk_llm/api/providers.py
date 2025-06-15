# chuk_llm/api/providers.py
"""
Dynamic provider function generation - everything from YAML
==========================================================

Generates functions like ask_openai_gpt4o(), ask_claude_sync(), etc.
All models, aliases, and providers come from YAML configuration.
"""

import asyncio
import re
import logging
import threading
import atexit
from typing import Dict, Optional, List, AsyncIterator

from chuk_llm.configuration.config import get_config

logger = logging.getLogger(__name__)

# Async loop management for sync functions
_loop_thread = None
_persistent_loop = None
_loop_lock = threading.Lock()


def _get_persistent_loop():
    """Get or create persistent event loop for sync functions"""
    global _loop_thread, _persistent_loop
    
    with _loop_lock:
        if _persistent_loop is None or _persistent_loop.is_closed():
            loop_ready = threading.Event()
            
            def run_loop():
                global _persistent_loop
                _persistent_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_persistent_loop)
                loop_ready.set()
                _persistent_loop.run_forever()
            
            _loop_thread = threading.Thread(target=run_loop, daemon=True)
            _loop_thread.start()
            loop_ready.wait()
        
        return _persistent_loop


def _run_sync(coro):
    """Run async coroutine synchronously"""
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot call sync functions from async context. "
            "Use the async version instead."
        )
    except RuntimeError as e:
        if "Cannot call sync functions" in str(e):
            raise e
    
    loop = _get_persistent_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _cleanup_loop():
    """Cleanup persistent loop on exit"""
    global _persistent_loop, _loop_thread
    
    if _persistent_loop and not _persistent_loop.is_closed():
        _persistent_loop.call_soon_threadsafe(_persistent_loop.stop)
        if _loop_thread and _loop_thread.is_alive():
            _loop_thread.join(timeout=1.0)


atexit.register(_cleanup_loop)


def _sanitize_name(name: str) -> str:
    """Convert any name to valid Python identifier
    
    Single rule: Convert to lowercase, replace separators with underscores,
    remove special chars, and merge alphabetic parts with short numeric parts.
    """
    if not name:
        return ""
    
    # Single rule approach
    sanitized = name.lower()
    
    # Replace all separators (dots, dashes, slashes, spaces) with underscores
    sanitized = re.sub(r'[-./\s]+', '_', sanitized)
    
    # Remove all non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
    
    # Consolidate multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Split and apply simple merging rule
    parts = [p for p in sanitized.split('_') if p]
    
    if not parts:
        return ""
    
    # Single merging rule: alphabetic + short alphanumeric = merge
    merged_parts = []
    i = 0
    
    while i < len(parts):
        current = parts[i]
        
        if (i + 1 < len(parts) and 
            current.isalpha() and 
            len(parts[i + 1]) <= 3 and 
            any(c.isdigit() for c in parts[i + 1])):
            # Merge current with next
            merged_parts.append(current + parts[i + 1])
            i += 2
        else:
            merged_parts.append(current)
            i += 1
    
    result = '_'.join(merged_parts)
    
    # Handle leading digits
    if result and result[0].isdigit():
        result = f"model_{result}"
    
    return result.strip('_')


def _create_provider_function(provider_name: str, model_name: Optional[str] = None):
    """Create async provider function"""
    if model_name:
        async def provider_model_func(prompt: str, **kwargs) -> str:
            from .core import ask
            return await ask(prompt, provider=provider_name, model=model_name, **kwargs)
        return provider_model_func
    else:
        async def provider_func(prompt: str, model: Optional[str] = None, **kwargs) -> str:
            from .core import ask
            return await ask(prompt, provider=provider_name, model=model, **kwargs)
        return provider_func


def _create_stream_function(provider_name: str, model_name: Optional[str] = None):
    """Create async streaming function"""
    if model_name:
        async def stream_model_func(prompt: str, **kwargs) -> AsyncIterator[str]:
            from .core import stream
            async for chunk in stream(prompt, provider=provider_name, model=model_name, **kwargs):
                yield chunk
        return stream_model_func
    else:
        async def stream_func(prompt: str, model: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
            from .core import stream
            async for chunk in stream(prompt, provider=provider_name, model=model, **kwargs):
                yield chunk
        return stream_func


def _create_sync_function(provider_name: str, model_name: Optional[str] = None):
    """Create sync provider function"""
    if model_name:
        def sync_model_func(prompt: str, **kwargs) -> str:
            from .core import ask
            return _run_sync(ask(prompt, provider=provider_name, model=model_name, **kwargs))
        return sync_model_func
    else:
        def sync_func(prompt: str, model: Optional[str] = None, **kwargs) -> str:
            from .core import ask
            if model is not None:
                kwargs['model'] = model
            return _run_sync(ask(prompt, provider=provider_name, **kwargs))
        return sync_func


def _create_global_alias_function(alias_name: str, provider_model: str):
    """Create global alias function"""
    if '/' not in provider_model:
        logger.warning(f"Invalid global alias: {provider_model} (expected 'provider/model')")
        return {}
    
    provider, model = provider_model.split('/', 1)
    
    async def alias_func(prompt: str, **kwargs) -> str:
        from .core import ask
        return await ask(prompt, provider=provider, model=model, **kwargs)
    
    def alias_sync_func(prompt: str, **kwargs) -> str:
        return _run_sync(alias_func(prompt, **kwargs))
    
    async def alias_stream_func(prompt: str, **kwargs) -> AsyncIterator[str]:
        from .core import stream
        async for chunk in stream(prompt, provider=provider, model=model, **kwargs):
            yield chunk
    
    return {
        f"ask_{alias_name}": alias_func,
        f"ask_{alias_name}_sync": alias_sync_func,
        f"stream_{alias_name}": alias_stream_func,
    }


def _generate_functions():
    """Generate all provider functions from YAML config"""
    config_manager = get_config()
    functions = {}
    
    providers = config_manager.get_all_providers()
    logger.info(f"Generating functions for {len(providers)} providers")
    
    # Generate provider functions
    for provider_name in providers:
        try:
            provider_config = config_manager.get_provider(provider_name)
        except ValueError as e:
            logger.error(f"Error loading provider {provider_name}: {e}")
            continue
        
        # Base provider functions: ask_openai(), stream_openai(), ask_openai_sync()
        functions[f"ask_{provider_name}"] = _create_provider_function(provider_name)
        functions[f"stream_{provider_name}"] = _create_stream_function(provider_name)
        functions[f"ask_{provider_name}_sync"] = _create_sync_function(provider_name)
        
        # Model-specific functions from YAML models list
        for model in provider_config.models:
            model_suffix = _sanitize_name(model)
            if model_suffix:
                functions[f"ask_{provider_name}_{model_suffix}"] = _create_provider_function(provider_name, model)
                functions[f"stream_{provider_name}_{model_suffix}"] = _create_stream_function(provider_name, model)
                functions[f"ask_{provider_name}_{model_suffix}_sync"] = _create_sync_function(provider_name, model)
        
        # Alias functions from YAML model_aliases
        for alias, actual_model in provider_config.model_aliases.items():
            alias_suffix = _sanitize_name(alias)
            if alias_suffix:
                functions[f"ask_{provider_name}_{alias_suffix}"] = _create_provider_function(provider_name, actual_model)
                functions[f"stream_{provider_name}_{alias_suffix}"] = _create_stream_function(provider_name, actual_model)
                functions[f"ask_{provider_name}_{alias_suffix}_sync"] = _create_sync_function(provider_name, actual_model)
    
    # Generate global alias functions from YAML
    global_aliases = config_manager.get_global_aliases()
    for alias_name, provider_model in global_aliases.items():
        alias_functions = _create_global_alias_function(alias_name, provider_model)
        functions.update(alias_functions)
    
    # Set function names and docstrings
    for name, func in functions.items():
        func.__name__ = name
        
        if name.endswith("_sync"):
            base = name[:-5].replace("_", " ")
            func.__doc__ = f"Synchronous {base} call."
        elif name.startswith("ask_"):
            base = name[4:].replace("_", " ")
            func.__doc__ = f"Async {base} call."
        elif name.startswith("stream_"):
            base = name[7:].replace("_", " ")
            func.__doc__ = f"Stream from {base}."
    
    logger.info(f"Generated {len(functions)} provider functions")
    return functions


def _create_utility_functions():
    """Create utility functions"""
    config_manager = get_config()
    
    def quick_question(question: str, provider: str = None) -> str:
        """Quick one-off question using sync API"""
        if not provider:
            settings = config_manager.get_global_settings()
            provider = settings.get("active_provider", "openai")
        
        from .sync import ask_sync
        return ask_sync(question, provider=provider)
    
    def compare_providers(question: str, providers: List[str] = None) -> Dict[str, str]:
        """Compare responses from multiple providers"""
        if not providers:
            all_providers = config_manager.get_all_providers()
            providers = all_providers[:3] if len(all_providers) >= 3 else all_providers
        
        from .sync import ask_sync
        results = {}
        
        for provider in providers:
            try:
                results[provider] = ask_sync(question, provider=provider)
            except Exception as e:
                results[provider] = f"Error: {str(e)}"
        
        return results
    
    def show_config():
        """Show current configuration status"""
        from chuk_llm.configuration.config import get_config
        config = get_config()
        
        print("🔧 ChukLLM Configuration")
        print("=" * 30)
        
        providers = config.get_all_providers()
        print(f"📦 Providers: {len(providers)}")
        for provider_name in providers:
            try:
                provider = config.get_provider(provider_name)
                has_key = "✅" if config.get_api_key(provider_name) else "❌"
                print(f"  {has_key} {provider_name:<12} | {len(provider.models):2d} models | {len(provider.model_aliases):2d} aliases")
            except Exception as e:
                print(f"  ❌ {provider_name:<12} | Error: {e}")
        
        aliases = config.get_global_aliases()
        if aliases:
            print(f"\n🌍 Global Aliases: {len(aliases)}")
            for alias, target in list(aliases.items())[:5]:
                print(f"  ask_{alias}() -> {target}")
            if len(aliases) > 5:
                print(f"  ... and {len(aliases) - 5} more")
    
    return {
        'quick_question': quick_question,
        'compare_providers': compare_providers,
        'show_config': show_config,
    }


# Generate all functions at module import
logger.info("Generating dynamic provider functions from YAML...")

try:
    # Generate provider functions
    _provider_functions = _generate_functions()
    
    # Generate utility functions
    _utility_functions = _create_utility_functions()
    
    # Combine all functions
    _all_functions = {}
    _all_functions.update(_provider_functions)
    _all_functions.update(_utility_functions)
    
    # Add to module namespace
    globals().update(_all_functions)
    
    # Export all function names
    __all__ = list(_all_functions.keys())
    
    logger.info(f"Generated {len(_all_functions)} total functions")
    
    # Log some examples
    examples = [name for name in __all__ 
               if any(x in name for x in ['gpt4', 'claude', 'llama']) 
               and not name.endswith('_sync')][:5]
    if examples:
        logger.info(f"Example functions: {', '.join(examples)}")

except Exception as e:
    logger.error(f"Error generating provider functions: {e}")
    # Fallback - at least provide utility functions
    __all__ = ['show_config']
    
    def show_config():
        print(f"❌ Error loading configuration: {e}")
        print("Create a providers.yaml file to use ChukLLM")
    
    globals()['show_config'] = show_config