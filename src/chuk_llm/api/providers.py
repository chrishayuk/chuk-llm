# chuk_llm/api/providers.py
"""
Dynamic provider function generation with live discovery support
==============================================================

Generates functions like ask_openai_gpt4o(), ask_claude_sync(), etc.
All models, aliases, and providers come from YAML configuration.
NOW SUPPORTS LIVE DISCOVERY - functions are generated on-demand for discovered models.
"""

import asyncio
import re
import logging
import warnings
from typing import Dict, Optional, List, AsyncIterator, Union
from pathlib import Path
import base64

from chuk_llm.configuration.unified_config import get_config, Feature

logger = logging.getLogger(__name__)

# Suppress specific asyncio cleanup warnings that don't affect functionality
warnings.filterwarnings("ignore", message=".*Event loop is closed.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*", category=RuntimeWarning)

# Cache for generated functions
_GENERATED_FUNCTIONS = {}
_FUNCTION_CACHE_DIRTY = True


def _run_sync(coro):
    """Simple sync wrapper using event loop manager"""
    try:
        # Try to import the event loop manager
        from .event_loop_manager import run_sync
        return run_sync(coro)
    except ImportError:
        # Fallback to simple asyncio.run if event loop manager not available
        import asyncio
        import warnings
        
        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*Event loop is closed.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*", category=RuntimeWarning)
        
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call sync functions from async context. "
                "Use the async version instead."
            )
        except RuntimeError as e:
            if "Cannot call sync functions" in str(e):
                raise e
        
        # Use asyncio.run - each call gets a fresh loop and fresh client connections
        return asyncio.run(coro)


def _sanitize_name(name: str) -> str:
    """Convert any name to valid Python identifier
    
    Improved version that keeps separators as underscores for better readability.
    
    Examples:
        devstral:latest -> devstral_latest
        qwen3:32b -> qwen3_32b  
        phi4-reasoning:latest -> phi4_reasoning_latest
        llama3.3:latest -> llama3_3_latest
    """
    if not name:
        return ""
    
    # Start with lowercase
    sanitized = name.lower()
    
    # Replace separators with underscores (keep them separated!)
    sanitized = sanitized.replace(':', '_')    # version separators
    sanitized = sanitized.replace('-', '_')    # hyphens  
    sanitized = sanitized.replace('.', '_')    # dots
    sanitized = sanitized.replace('/', '_')    # slashes
    sanitized = sanitized.replace(' ', '_')    # spaces
    
    # Remove any other non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
    
    # Consolidate multiple underscores into single underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    if not sanitized:
        return ""
    
    # Handle leading digits (Python identifiers can't start with digits)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"model_{sanitized}"
    
    return sanitized


def _prepare_vision_message(prompt: str, image: Union[str, Path, bytes], provider: str = None) -> Dict[str, any]:
    """Prepare a vision message with text and image, handling provider-specific formats"""
    
    # First, get the image data and determine format
    image_data = None
    image_url = None
    media_type = 'image/jpeg'  # default
    
    if isinstance(image, (str, Path)):
        image_path = Path(image) if not isinstance(image, str) or not image.startswith(('http://', 'https://')) else None
        
        if image_path and image_path.exists():
            # Local file
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                # Determine media type from extension
                suffix = image_path.suffix.lower()
                media_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }.get(suffix, 'image/jpeg')
                image_url = f"data:{media_type};base64,{image_data}"
                
        elif isinstance(image, str) and image.startswith(('http://', 'https://')):
            # URL - handle provider differences
            image_url = image
            
            # For providers that need base64 (like Anthropic), download the image
            if provider and 'anthropic' in provider.lower():
                try:
                    import urllib.request
                    import urllib.parse
                    from io import BytesIO
                    
                    # Download the image
                    with urllib.request.urlopen(image) as response:
                        image_bytes = response.read()
                        
                    # Try to determine media type from headers
                    content_type = response.headers.get('Content-Type', 'image/jpeg')
                    if 'image/' in content_type:
                        media_type = content_type
                    
                    # Convert to base64
                    image_data = base64.b64encode(image_bytes).decode('utf-8')
                    
                except Exception as e:
                    raise ValueError(f"Failed to download image from URL for Anthropic: {e}")
        else:
            raise ValueError(f"Image file not found: {image}")
            
    elif isinstance(image, bytes):
        # Raw bytes
        image_data = base64.b64encode(image).decode('utf-8')
        media_type = 'image/png'  # Default to PNG for bytes
        image_url = f"data:{media_type};base64,{image_data}"
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Now format based on provider
    if provider and 'anthropic' in provider.lower():
        # Anthropic format - always needs base64 data
        if image_data is None and image_url:
            # Extract base64 from data URL if needed
            if image_url.startswith('data:'):
                # Extract base64 part
                base64_part = image_url.split(',')[1] if ',' in image_url else image_url
                image_data = base64_part
                # Extract media type
                media_type_match = re.match(r'data:([^;]+);', image_url)
                media_type = media_type_match.group(1) if media_type_match else 'image/jpeg'
        
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }
    else:
        # OpenAI/Gemini/others format - can use URLs directly
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        }


def _create_provider_function(provider_name: str, model_name: Optional[str] = None, supports_vision: bool = False):
    """Create async provider function with optional vision support"""
    if model_name:
        if supports_vision:
            async def provider_model_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, **kwargs) -> str:
                from .core import ask
                if image is not None:
                    vision_message = _prepare_vision_message(prompt, image, provider_name)
                    kwargs['messages'] = [vision_message]
                return await ask(prompt, provider=provider_name, model=model_name, **kwargs)
        else:
            async def provider_model_func(prompt: str, **kwargs) -> str:
                from .core import ask
                return await ask(prompt, provider=provider_name, model=model_name, **kwargs)
        return provider_model_func
    else:
        if supports_vision:
            async def provider_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, model: Optional[str] = None, **kwargs) -> str:
                from .core import ask
                if image is not None:
                    vision_message = _prepare_vision_message(prompt, image, provider_name)
                    kwargs['messages'] = [vision_message]
                return await ask(prompt, provider=provider_name, model=model, **kwargs)
        else:
            async def provider_func(prompt: str, model: Optional[str] = None, **kwargs) -> str:
                from .core import ask
                return await ask(prompt, provider=provider_name, model=model, **kwargs)
        return provider_func


def _create_stream_function(provider_name: str, model_name: Optional[str] = None, supports_vision: bool = False):
    """Create async streaming function with optional vision support"""
    if model_name:
        if supports_vision:
            async def stream_model_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, **kwargs) -> AsyncIterator[str]:
                from .core import stream
                if image is not None:
                    vision_message = _prepare_vision_message(prompt, image, provider_name)
                    kwargs['messages'] = [vision_message]
                async for chunk in stream(prompt, provider=provider_name, model=model_name, **kwargs):
                    yield chunk
        else:
            async def stream_model_func(prompt: str, **kwargs) -> AsyncIterator[str]:
                from .core import stream
                async for chunk in stream(prompt, provider=provider_name, model=model_name, **kwargs):
                    yield chunk
        return stream_model_func
    else:
        if supports_vision:
            async def stream_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, model: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
                from .core import stream
                if image is not None:
                    vision_message = _prepare_vision_message(prompt, image, provider_name)
                    kwargs['messages'] = [vision_message]
                async for chunk in stream(prompt, provider=provider_name, model=model, **kwargs):
                    yield chunk
        else:
            async def stream_func(prompt: str, model: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
                from .core import stream
                async for chunk in stream(prompt, provider=provider_name, model=model, **kwargs):
                    yield chunk
        return stream_func


def _create_sync_function(provider_name: str, model_name: Optional[str] = None, supports_vision: bool = False):
    """Create sync provider function with optional vision support"""
    if model_name:
        if supports_vision:
            def sync_model_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, **kwargs) -> str:
                from .core import ask
                if image is not None:
                    vision_message = _prepare_vision_message(prompt, image, provider_name)
                    kwargs['messages'] = [vision_message]
                return _run_sync(ask(prompt, provider=provider_name, model=model_name, **kwargs))
        else:
            def sync_model_func(prompt: str, **kwargs) -> str:
                from .core import ask
                return _run_sync(ask(prompt, provider=provider_name, model=model_name, **kwargs))
        return sync_model_func
    else:
        if supports_vision:
            def sync_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, model: Optional[str] = None, **kwargs) -> str:
                from .core import ask
                if model is not None:
                    kwargs['model'] = model
                if image is not None:
                    vision_message = _prepare_vision_message(prompt, image, provider_name)
                    kwargs['messages'] = [vision_message]
                return _run_sync(ask(prompt, provider=provider_name, **kwargs))
        else:
            def sync_func(prompt: str, model: Optional[str] = None, **kwargs) -> str:
                from .core import ask
                if model is not None:
                    kwargs['model'] = model
                return _run_sync(ask(prompt, provider=provider_name, **kwargs))
        return sync_func


def _create_global_alias_function(alias_name: str, provider_model: str, supports_vision: bool = False):
    """Create global alias function with optional vision support"""
    if '/' not in provider_model:
        logger.warning(f"Invalid global alias: {provider_model} (expected 'provider/model')")
        return {}
    
    provider, model = provider_model.split('/', 1)
    
    if supports_vision:
        async def alias_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, **kwargs) -> str:
            from .core import ask
            if image is not None:
                vision_message = _prepare_vision_message(prompt, image, provider)
                kwargs['messages'] = [vision_message]
            return await ask(prompt, provider=provider, model=model, **kwargs)
        
        def alias_sync_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, **kwargs) -> str:
            return _run_sync(alias_func(prompt, image=image, **kwargs))
        
        async def alias_stream_func(prompt: str, image: Optional[Union[str, Path, bytes]] = None, **kwargs) -> AsyncIterator[str]:
            from .core import stream
            if image is not None:
                vision_message = _prepare_vision_message(prompt, image, provider)
                kwargs['messages'] = [vision_message]
            async for chunk in stream(prompt, provider=provider, model=model, **kwargs):
                yield chunk
    else:
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


def _generate_functions_for_models(provider_name: str, provider_config, models: List[str]) -> Dict[str, callable]:
    """Generate functions for a specific list of models"""
    functions = {}
    
    # Check if provider supports vision at all
    provider_supports_vision = Feature.VISION in provider_config.features
    
    for model in models:
        model_suffix = _sanitize_name(model)
        if model_suffix:
            # Check if this specific model supports vision
            model_caps = provider_config.get_model_capabilities(model)
            model_supports_vision = Feature.VISION in model_caps.features
            
            # Generate the three function types for the full model name
            ask_func = _create_provider_function(provider_name, model, model_supports_vision)
            stream_func = _create_stream_function(provider_name, model, model_supports_vision)
            sync_func = _create_sync_function(provider_name, model, model_supports_vision)
            
            # Set function names and docstrings
            func_names = [
                f"ask_{provider_name}_{model_suffix}",
                f"stream_{provider_name}_{model_suffix}",
                f"ask_{provider_name}_{model_suffix}_sync"
            ]
            
            for func, name in zip([ask_func, stream_func, sync_func], func_names):
                func.__name__ = name
                
                # Check if this is a vision-capable function
                has_image_param = 'image' in func.__code__.co_varnames
                
                if name.endswith("_sync"):
                    base = name[:-5].replace("_", " ")
                    if has_image_param:
                        func.__doc__ = f"Synchronous {base} call with optional image support."
                    else:
                        func.__doc__ = f"Synchronous {base} call."
                elif name.startswith("ask_"):
                    base = name[4:].replace("_", " ")
                    if has_image_param:
                        func.__doc__ = f"Async {base} call with optional image support."
                    else:
                        func.__doc__ = f"Async {base} call."
                elif name.startswith("stream_"):
                    base = name[7:].replace("_", " ")
                    if has_image_param:
                        func.__doc__ = f"Stream from {base} with optional image support."
                    else:
                        func.__doc__ = f"Stream from {base}."
                
                functions[name] = func
            
            # ALSO generate functions for the base name (without :latest)
            if model.endswith(':latest'):
                base_model = model[:-7]  # Remove ':latest'
                base_suffix = _sanitize_name(base_model)
                
                if base_suffix and base_suffix != model_suffix:
                    # Generate additional functions for the base name
                    base_ask_func = _create_provider_function(provider_name, model, model_supports_vision)
                    base_stream_func = _create_stream_function(provider_name, model, model_supports_vision)
                    base_sync_func = _create_sync_function(provider_name, model, model_supports_vision)
                    
                    base_func_names = [
                        f"ask_{provider_name}_{base_suffix}",
                        f"stream_{provider_name}_{base_suffix}",
                        f"ask_{provider_name}_{base_suffix}_sync"
                    ]
                    
                    for func, name in zip([base_ask_func, base_stream_func, base_sync_func], base_func_names):
                        func.__name__ = name
                        
                        # Set docstrings
                        has_image_param = 'image' in func.__code__.co_varnames
                        
                        if name.endswith("_sync"):
                            base = name[:-5].replace("_", " ")
                            if has_image_param:
                                func.__doc__ = f"Synchronous {base} call with optional image support."
                            else:
                                func.__doc__ = f"Synchronous {base} call."
                        elif name.startswith("ask_"):
                            base = name[4:].replace("_", " ")
                            if has_image_param:
                                func.__doc__ = f"Async {base} call with optional image support."
                            else:
                                func.__doc__ = f"Async {base} call."
                        elif name.startswith("stream_"):
                            base = name[7:].replace("_", " ")
                            if has_image_param:
                                func.__doc__ = f"Stream from {base} with optional image support."
                            else:
                                func.__doc__ = f"Stream from {base}."
                        
                        functions[name] = func
    
    return functions


def _ensure_provider_models_current(provider_name: str) -> List[str]:
    """Ensure provider models are current (trigger discovery if needed)"""
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider_name)
        
        # For Ollama, also query the API directly to get current models
        if provider_name == "ollama":
            try:
                import httpx
                import asyncio
                
                async def get_ollama_models():
                    try:
                        async with httpx.AsyncClient(timeout=3.0) as client:
                            response = await client.get("http://localhost:11434/api/tags")
                            if response.status_code == 200:
                                data = response.json()
                                return [model['name'] for model in data.get('models', [])]
                    except:
                        return []
                    return []
                
                # Get current models from Ollama API
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, skip the API call to avoid blocking
                        pass
                    else:
                        ollama_models = loop.run_until_complete(get_ollama_models())
                        if ollama_models:
                            # Merge with existing models
                            all_models = list(set(provider_config.models + ollama_models))
                            provider_config.models = all_models
                            logger.debug(f"Updated Ollama models: {len(all_models)} total")
                except:
                    # If async fails, continue with existing models
                    pass
            except ImportError:
                # httpx not available, continue with existing models
                pass
        
        # Check if discovery is enabled
        discovery_data = provider_config.extra.get("dynamic_discovery")
        if discovery_data and discovery_data.get("enabled", False):
            # Trigger discovery by accessing models through the ensure method
            # This will automatically discover new models if they exist
            try:
                dummy_model = config_manager._ensure_model_available(provider_name, "dummy_model_to_trigger_discovery")
            except:
                # Discovery might fail, continue with current models
                pass
        
        return provider_config.models
        
    except Exception as e:
        logger.debug(f"Could not ensure current models for {provider_name}: {e}")
        return []


def _generate_static_functions():
    """Generate functions for static (YAML-configured) models"""
    config_manager = get_config()
    functions = {}
    
    providers = config_manager.get_all_providers()
    logger.info(f"Generating static functions for {len(providers)} providers")
    
    # Generate provider functions
    for provider_name in providers:
        try:
            provider_config = config_manager.get_provider(provider_name)
        except ValueError as e:
            logger.error(f"Error loading provider {provider_name}: {e}")
            continue
        
        # Check if provider supports vision at all
        provider_supports_vision = Feature.VISION in provider_config.features
        
        # Base provider functions: ask_openai(), stream_openai(), ask_openai_sync()
        functions[f"ask_{provider_name}"] = _create_provider_function(provider_name, supports_vision=provider_supports_vision)
        functions[f"stream_{provider_name}"] = _create_stream_function(provider_name, supports_vision=provider_supports_vision)
        functions[f"ask_{provider_name}_sync"] = _create_sync_function(provider_name, supports_vision=provider_supports_vision)
        
        # Get current models (including discovered ones)
        current_models = _ensure_provider_models_current(provider_name)
        
        # Generate functions for current models
        model_functions = _generate_functions_for_models(provider_name, provider_config, current_models)
        functions.update(model_functions)
        
        # Alias functions from YAML model_aliases
        for alias, actual_model in provider_config.model_aliases.items():
            alias_suffix = _sanitize_name(alias)
            if alias_suffix:
                # Check if the actual model supports vision
                model_caps = provider_config.get_model_capabilities(actual_model)
                model_supports_vision = Feature.VISION in model_caps.features
                
                functions[f"ask_{provider_name}_{alias_suffix}"] = _create_provider_function(provider_name, actual_model, model_supports_vision)
                functions[f"stream_{provider_name}_{alias_suffix}"] = _create_stream_function(provider_name, actual_model, model_supports_vision)
                functions[f"ask_{provider_name}_{alias_suffix}_sync"] = _create_sync_function(provider_name, actual_model, model_supports_vision)
    
    # Generate global alias functions from YAML
    global_aliases = config_manager.get_global_aliases()
    for alias_name, provider_model in global_aliases.items():
        # Check if the aliased model supports vision
        if '/' in provider_model:
            provider, model = provider_model.split('/', 1)
            try:
                provider_config = config_manager.get_provider(provider)
                model_caps = provider_config.get_model_capabilities(model)
                alias_supports_vision = Feature.VISION in model_caps.features
            except:
                alias_supports_vision = False
        else:
            alias_supports_vision = False
            
        alias_functions = _create_global_alias_function(alias_name, provider_model, alias_supports_vision)
        functions.update(alias_functions)
    
    return functions


def _generate_functions():
    """Generate all provider functions from YAML config"""
    return _generate_static_functions()


def refresh_provider_functions(provider_name: str = None):
    """Refresh functions for a specific provider or all providers"""
    global _GENERATED_FUNCTIONS, _FUNCTION_CACHE_DIRTY
    
    config_manager = get_config()
    
    if provider_name:
        # Refresh specific provider
        try:
            provider_config = config_manager.get_provider(provider_name)
            
            # Get current models (this will trigger discovery and API query for Ollama)
            current_models = _ensure_provider_models_current(provider_name)
            
            logger.info(f"Generating functions for {len(current_models)} {provider_name} models")
            
            # Generate functions for new models
            new_functions = _generate_functions_for_models(provider_name, provider_config, current_models)
            
            # Update the global functions dict
            _GENERATED_FUNCTIONS.update(new_functions)
            
            # Update module namespace
            import sys
            current_module = sys.modules[__name__]
            for name, func in new_functions.items():
                setattr(current_module, name, func)
            
            # Update main chuk_llm module too
            try:
                import chuk_llm
                for name, func in new_functions.items():
                    if name.startswith(('ask_', 'stream_')) and provider_name in name:
                        setattr(chuk_llm, name, func)
            except:
                pass
            
            # Update __all__
            if hasattr(current_module, '__all__'):
                for name in new_functions.keys():
                    if name not in current_module.__all__:
                        current_module.__all__.append(name)
            
            logger.info(f"Refreshed {len(new_functions)} functions for {provider_name}")
            return new_functions
            
        except Exception as e:
            logger.error(f"Failed to refresh functions for {provider_name}: {e}")
            return {}
    else:
        # Refresh all providers
        all_new_functions = _generate_static_functions()
        _GENERATED_FUNCTIONS.update(all_new_functions)
        _FUNCTION_CACHE_DIRTY = False
        
        # Update module namespace
        import sys
        current_module = sys.modules[__name__]
        for name, func in all_new_functions.items():
            setattr(current_module, name, func)
        
        # Update main chuk_llm module too
        try:
            import chuk_llm
            for name, func in all_new_functions.items():
                if name.startswith(('ask_', 'stream_')):
                    setattr(chuk_llm, name, func)
        except:
            pass
        
        logger.info(f"Refreshed {len(all_new_functions)} total functions")
        return all_new_functions


def trigger_ollama_discovery_and_refresh():
    """Trigger Ollama model discovery and refresh functions"""
    try:
        # This will trigger discovery for Ollama specifically
        current_models = _ensure_provider_models_current("ollama")
        
        # Refresh functions for Ollama
        new_functions = refresh_provider_functions("ollama")
        
        logger.info(f"Ollama discovery triggered: {len(current_models)} models, {len(new_functions)} functions generated")
        return new_functions
        
    except Exception as e:
        logger.error(f"Failed to trigger Ollama discovery: {e}")
        return {}


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
        from chuk_llm.configuration.unified_config import get_config
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
    
    def discover_and_refresh(provider: str = "ollama"):
        """Discover models and refresh functions for a provider"""
        return refresh_provider_functions(provider)
    
    return {
        'quick_question': quick_question,
        'compare_providers': compare_providers,
        'show_config': show_config,
        'discover_and_refresh': discover_and_refresh,
        'refresh_provider_functions': refresh_provider_functions,
        'trigger_ollama_discovery_and_refresh': trigger_ollama_discovery_and_refresh,
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
    
    # Store in cache
    _GENERATED_FUNCTIONS.update(_all_functions)
    
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
    __all__ = ['show_config', 'refresh_provider_functions', 'trigger_ollama_discovery_and_refresh']
    
    def show_config():
        print(f"❌ Error loading configuration: {e}")
        print("Create a providers.yaml file to use ChukLLM")
    
    def refresh_provider_functions(provider_name: str = None):
        print(f"❌ Function refresh not available: {e}")
        return {}
    
    def trigger_ollama_discovery_and_refresh():
        print(f"❌ Discovery not available: {e}")
        return {}
    
    globals()['show_config'] = show_config
    globals()['refresh_provider_functions'] = refresh_provider_functions
    globals()['trigger_ollama_discovery_and_refresh'] = trigger_ollama_discovery_and_refresh


# Enhanced __getattr__ to support on-demand function generation
def __getattr__(name):
    """Allow access to generated functions with on-demand discovery"""
    # First check if it's in our generated functions
    if name in _GENERATED_FUNCTIONS:
        return _GENERATED_FUNCTIONS[name]
    
    # Check if it looks like a provider function we might need to generate
    if name.startswith(('ask_', 'stream_')) and ('_' in name[4:] if name.startswith('ask_') else '_' in name[7:]):
        # Parse the function name to extract provider and model
        if name.startswith('ask_'):
            base_name = name[4:]  # Remove 'ask_'
            is_sync = base_name.endswith('_sync')
            if is_sync:
                base_name = base_name[:-5]  # Remove '_sync'
        elif name.startswith('stream_'):
            base_name = name[7:]  # Remove 'stream_'
            is_sync = False
        else:
            raise AttributeError(f"module 'providers' has no attribute '{name}'")
        
        # Split provider and model
        parts = base_name.split('_', 1)
        if len(parts) == 2:
            provider_name, model_part = parts
            
            # Try to refresh functions for this provider
            try:
                new_functions = refresh_provider_functions(provider_name)
                if name in new_functions:
                    return new_functions[name]
            except Exception as e:
                logger.debug(f"Could not refresh functions for {provider_name}: {e}")
    
    # If we still don't have it, raise AttributeError
    raise AttributeError(f"module 'providers' has no attribute '{name}'")


# Export all generated functions for external access
def get_all_functions():
    """Get all generated provider functions"""
    return _GENERATED_FUNCTIONS.copy()


def list_provider_functions():
    """List all available provider functions"""
    return sorted(list(_GENERATED_FUNCTIONS.keys()))


def has_function(name):
    """Check if a provider function exists"""
    return name in _GENERATED_FUNCTIONS


def get_discovered_functions(provider: str = None):
    """Get functions that were created for discovered models"""
    discovered = {}
    config_manager = get_config()
    
    if provider:
        providers_to_check = [provider]
    else:
        providers_to_check = config_manager.get_all_providers()
    
    for provider_name in providers_to_check:
        try:
            provider_config = config_manager.get_provider(provider_name)
            discovery_data = provider_config.extra.get("dynamic_discovery")
            
            if discovery_data and discovery_data.get("enabled", False):
                # Check which functions exist for this provider
                provider_functions = {
                    name: func for name, func in _GENERATED_FUNCTIONS.items()
                    if name.startswith(f'ask_{provider_name}_') or name.startswith(f'stream_{provider_name}_')
                }
                discovered[provider_name] = provider_functions
                
        except Exception as e:
            logger.debug(f"Could not check discovered functions for {provider_name}: {e}")
    
    return discovered