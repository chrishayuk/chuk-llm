# chuk_llm/api/providers.py
"""
Dynamic provider function generation with environment-controlled discovery
========================================================================

Generates functions like ask_openai_gpt4o(), ask_claude_sync(), etc.
All models, aliases, and providers come from YAML configuration.
NOW SUPPORTS LIVE DISCOVERY with full environment variable controls.

Environment Variables for Discovery Control:
- CHUK_LLM_DISCOVERY_ENABLED=false          # Disable all discovery
- CHUK_LLM_DISCOVERY_ON_STARTUP=false       # Disable startup discovery checks
- CHUK_LLM_AUTO_DISCOVER=false              # Disable on-demand discovery
- CHUK_LLM_OLLAMA_DISCOVERY=false           # Disable Ollama discovery only
- CHUK_LLM_DISCOVERY_TIMEOUT=10             # Set discovery timeout
- CHUK_LLM_DISCOVERY_QUICK_TIMEOUT=1.0      # Set quick check timeout
"""

import asyncio
import re
import logging
import warnings
import os
import sys
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


def _env_bool(key: str, default: bool = False) -> bool:
    """Parse boolean environment variable"""
    value = os.getenv(key, '').lower()
    if value in ('true', '1', 'yes', 'on', 'enabled'):
        return True
    elif value in ('false', '0', 'no', 'off', 'disabled'):
        return False
    else:
        return default


def _is_discovery_enabled(provider_name: str = None) -> bool:
    """Check if discovery is enabled via environment variables"""
    # Global check
    if not _env_bool('CHUK_LLM_DISCOVERY_ENABLED', True):
        return False
    
    # Provider-specific check
    if provider_name:
        provider_key = f'CHUK_LLM_{provider_name.upper()}_DISCOVERY'
        return _env_bool(provider_key, True)
    
    return True


def _is_startup_discovery_enabled() -> bool:
    """Check if startup discovery is enabled"""
    return _env_bool('CHUK_LLM_DISCOVERY_ON_STARTUP', True)


def _is_auto_discover_enabled() -> bool:
    """Check if auto discovery is enabled"""
    return _env_bool('CHUK_LLM_AUTO_DISCOVER', True)


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


def _check_ollama_available_models(timeout: float = None) -> List[str]:
    """Check what Ollama models are actually available locally (non-blocking)"""
    # Check if Ollama discovery is disabled
    if not _is_discovery_enabled('ollama'):
        logger.debug("Ollama discovery disabled by environment variable")
        return []
    
    # Use environment timeout or default
    if timeout is None:
        timeout = float(os.getenv('CHUK_LLM_DISCOVERY_QUICK_TIMEOUT', '2.0'))
    
    try:
        import httpx
        
        # Quick check with timeout
        with httpx.Client(timeout=timeout) as client:
            try:
                response = client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    available = [model['name'] for model in data.get('models', [])]
                    logger.debug(f"Found {len(available)} available Ollama models")
                    return available
            except (httpx.ConnectError, httpx.TimeoutException):
                logger.debug("Ollama not available or timeout - skipping model check")
                return []
            except Exception as e:
                logger.debug(f"Error checking Ollama models: {e}")
                return []
                
    except ImportError:
        logger.debug("httpx not available - skipping Ollama model check")
        return []
    
    return []


def _ensure_provider_models_current(provider_name: str) -> List[str]:
    """FIXED: Ensure provider models are current (trigger discovery if needed)"""
    # Check if auto-discovery is disabled globally
    if not _is_auto_discover_enabled() or not _is_discovery_enabled(provider_name):
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(provider_name)
            return provider_config.models
        except Exception as e:
            logger.debug(f"Could not get models for {provider_name}: {e}")
            return []
    
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider_name)
        
        # 🎯 CRITICAL FIX: Check discovery cache and merge discovered models
        cache_data = getattr(config_manager, '_discovery_cache', {}).get(provider_name)
        if cache_data and 'models' in cache_data:
            cached_models = cache_data['models']
            original_count = len(provider_config.models)
            
            # Merge static and discovered models (avoid duplicates)
            all_models = list(set(provider_config.models + cached_models))
            
            # 🎯 CRITICAL: Update provider config with merged models
            provider_config.models = all_models
            config_manager.providers[provider_name] = provider_config  # Ensure it's saved
            
            logger.info(f"Merged discovery cache for {provider_name}: {original_count} -> {len(all_models)} models")
            return all_models
        
        # For Ollama, also query the API directly to get current models
        if provider_name == "ollama":
            try:
                import httpx
                import asyncio
                
                async def get_ollama_models():
                    try:
                        timeout = float(os.getenv('CHUK_LLM_DISCOVERY_TIMEOUT', '3'))
                        async with httpx.AsyncClient(timeout=timeout) as client:
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
            # 🎯 CRITICAL FIX: Trigger discovery and merge results
            try:
                # Force discovery to populate cache
                discovered_count = config_manager.force_discover_models(provider_name)
                if discovered_count > 0:
                    logger.info(f"Force discovery found {discovered_count} new models for {provider_name}")
                    
                    # Re-check cache after discovery
                    cache_data = getattr(config_manager, '_discovery_cache', {}).get(provider_name)
                    if cache_data and 'models' in cache_data:
                        cached_models = cache_data['models']
                        # Merge and update
                        all_models = list(set(provider_config.models + cached_models))
                        provider_config.models = all_models
                        config_manager.providers[provider_name] = provider_config
                        logger.info(f"Merged after force discovery: {len(all_models)} total models")
                        return all_models
            except Exception as e:
                logger.debug(f"Force discovery failed: {e}")
        
        return provider_config.models
        
    except Exception as e:
        logger.debug(f"Could not ensure current models for {provider_name}: {e}")
        return []


def _get_safe_models_for_provider(provider_name: str, provider_config) -> List[str]:
    """FIXED: Get models that are safe to generate static functions for"""
    
    # 🎯 FIRST: Get current models including discovered ones
    current_models = _ensure_provider_models_current(provider_name)
    
    if provider_name == "ollama":
        # Check if startup discovery is enabled
        if not _is_startup_discovery_enabled():
            logger.info("Ollama startup discovery disabled - using configured models only")
            return current_models  # Use all current models (including discovered)
        
        # For Ollama, only include models that are actually available
        available_models = _check_ollama_available_models()
        
        if available_models:
            # Filter to only include models that are actually downloaded
            safe_models = []
            for model in current_models:  # Use current_models instead of config_models
                # Check both exact match and :latest variants
                model_base = model.replace(':latest', '') if model.endswith(':latest') else model
                model_latest = f"{model_base}:latest"
                
                if (model in available_models or 
                    model_base in available_models or 
                    model_latest in available_models):
                    safe_models.append(model)
            
            logger.info(f"Ollama: Using {len(safe_models)}/{len(current_models)} available models for static generation")
            return safe_models
        else:
            # Ollama not available - use minimal set or empty
            logger.info("Ollama not available - generating minimal static functions")
            return []  # Will generate provider-level functions only
    else:
        # For non-Ollama providers, check if their discovery is enabled
        if not _is_discovery_enabled(provider_name):
            logger.debug(f"Discovery disabled for {provider_name} - using all configured models")
        
        # For non-Ollama providers, use all current models (including discovered)
        return current_models


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
            
            # FIXED: Generate functions for the base name (without ANY version tag)
            # This was the bug - it only checked for :latest, but should check for any ':'
            if ':' in model:
                base_model = model.split(':')[0]  # Take everything before the first ':'
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
                        # Only add if not already exists (avoid conflicts)
                        if name not in functions:
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


def force_discovery_integration():
    """CRITICAL FIX: Force integration of discovery results into function generation"""
    
    logger.debug("🔧 Force integrating discovery results...")
    
    config_manager = get_config()
    total_new_functions = 0
    
    # Check each provider's discovery cache
    for provider_name in config_manager.get_all_providers():
        if not _is_discovery_enabled(provider_name):
            continue
        
        try:
            provider_config = config_manager.get_provider(provider_name)
            cache_data = getattr(config_manager, '_discovery_cache', {}).get(provider_name)
            
            if cache_data and 'models' in cache_data:
                cached_models = cache_data['models']
                original_models = set(provider_config.models)
                
                # Find new models from cache
                new_models = [m for m in cached_models if m not in original_models]
                
                if new_models:
                    logger.debug(f"   📈 {provider_name}: Adding {len(new_models)} discovered models")
                    
                    # Update provider config
                    provider_config.models.extend(new_models)
                    config_manager.providers[provider_name] = provider_config
                    
                    # Generate functions for new models
                    new_functions = _generate_functions_for_models(provider_name, provider_config, new_models)
                    
                    # Update global functions
                    global _GENERATED_FUNCTIONS
                    _GENERATED_FUNCTIONS.update(new_functions)
                    
                    # Update module namespace
                    current_module = sys.modules[__name__]
                    for name, func in new_functions.items():
                        setattr(current_module, name, func)
                    
                    # Update __all__
                    if hasattr(current_module, '__all__'):
                        for name in new_functions.keys():
                            if name not in current_module.__all__:
                                current_module.__all__.append(name)
                    
                    total_new_functions += len(new_functions)
                    logger.debug(f"   ✅ Generated {len(new_functions)} functions for {provider_name}")
            
        except Exception as e:
            logger.debug(f"   ❌ Failed to integrate {provider_name}: {e}")
    
    logger.debug(f"✅ Integration complete: {total_new_functions} new functions generated")
    return total_new_functions


def _generate_static_functions():
    """Generate functions for static (YAML-configured) models with environment controls"""
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
        
        # Get SAFE models for this provider (filtered for Ollama)
        safe_models = _get_safe_models_for_provider(provider_name, provider_config)
        
        if safe_models:
            # Generate functions for safe models only
            model_functions = _generate_functions_for_models(provider_name, provider_config, safe_models)
            functions.update(model_functions)
            logger.debug(f"Generated {len(model_functions)} model functions for {provider_name}")
        else:
            logger.debug(f"No safe models for {provider_name} - provider functions only")
        
        # Alias functions from YAML model_aliases (only for safe models)
        for alias, actual_model in provider_config.model_aliases.items():
            # Skip aliases that point to models we didn't include
            if safe_models and actual_model not in safe_models:
                logger.debug(f"Skipping alias {alias} -> {actual_model} (model not safe)")
                continue
                
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
        # Check if the aliased model is safe
        if '/' in provider_model:
            provider, model = provider_model.split('/', 1)
            try:
                provider_config = config_manager.get_provider(provider)
                safe_models = _get_safe_models_for_provider(provider, provider_config)
                
                # Skip global aliases that point to unsafe models
                if provider == "ollama" and safe_models and model not in safe_models:
                    logger.debug(f"Skipping global alias {alias_name} -> {provider_model} (model not safe)")
                    continue
                    
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


def _create_utility_functions():
    """Create utility functions with discovery status"""
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
        """Show current configuration status including discovery settings"""
        config = get_config()
        
        print("🔧 ChukLLM Configuration")
        print("=" * 50)
        
        # Discovery settings
        print("\n🔍 Discovery Settings:")
        discovery_enabled = _env_bool('CHUK_LLM_DISCOVERY_ENABLED', True)
        startup_enabled = _env_bool('CHUK_LLM_DISCOVERY_ON_STARTUP', True)
        auto_enabled = _env_bool('CHUK_LLM_AUTO_DISCOVER', True)
        
        print(f"  Global Discovery: {'✅ Enabled' if discovery_enabled else '❌ Disabled'}")
        print(f"  Startup Check:    {'✅ Enabled' if startup_enabled else '❌ Disabled'}")
        print(f"  Auto Discovery:   {'✅ Enabled' if auto_enabled else '❌ Disabled'}")
        
        timeout = os.getenv('CHUK_LLM_DISCOVERY_TIMEOUT', '5')
        quick_timeout = os.getenv('CHUK_LLM_DISCOVERY_QUICK_TIMEOUT', '2.0')
        print(f"  Timeout:          {timeout}s (quick: {quick_timeout}s)")
        
        providers = config.get_all_providers()
        print(f"\n📦 Providers: {len(providers)}")
        for provider_name in providers:
            try:
                provider = config.get_provider(provider_name)
                has_key = "✅" if config.get_api_key(provider_name) else "❌"
                discovery_status = "🔍" if _is_discovery_enabled(provider_name) else "🚫"
                print(f"  {has_key} {discovery_status} {provider_name:<12} | {len(provider.models):2d} models | {len(provider.model_aliases):2d} aliases")
            except Exception as e:
                print(f"  ❌ 🚫 {provider_name:<12} | Error: {e}")
        
        aliases = config.get_global_aliases()
        if aliases:
            print(f"\n🌍 Global Aliases: {len(aliases)}")
            for alias, target in list(aliases.items())[:5]:
                print(f"  ask_{alias}() -> {target}")
            if len(aliases) > 5:
                print(f"  ... and {len(aliases) - 5} more")
        
        print(f"\n🎛️  Environment Controls:")
        print(f"  CHUK_LLM_DISCOVERY_ENABLED={_env_bool('CHUK_LLM_DISCOVERY_ENABLED', True)}")
        print(f"  CHUK_LLM_DISCOVERY_ON_STARTUP={_env_bool('CHUK_LLM_DISCOVERY_ON_STARTUP', True)}")
        print(f"  CHUK_LLM_AUTO_DISCOVER={_env_bool('CHUK_LLM_AUTO_DISCOVER', True)}")
        print(f"  CHUK_LLM_OLLAMA_DISCOVERY={_env_bool('CHUK_LLM_OLLAMA_DISCOVERY', True)}")
        
        # Show discovery methods
        print(f"\n🎮 Discovery Control Methods:")
        print(f"  # Disable all discovery")
        print(f"  export CHUK_LLM_DISCOVERY_ENABLED=false")
        print(f"  ")
        print(f"  # Disable only Ollama discovery")
        print(f"  export CHUK_LLM_OLLAMA_DISCOVERY=false")
        print(f"  ")
        print(f"  # Disable startup checks (but allow on-demand)")
        print(f"  export CHUK_LLM_DISCOVERY_ON_STARTUP=false")
    
    def discover_and_refresh(provider: str = "ollama"):
        """Discover models and refresh functions for a provider"""
        if not _is_discovery_enabled(provider):
            print(f"❌ Discovery disabled for {provider}")
            print(f"   Set CHUK_LLM_{provider.upper()}_DISCOVERY=true to enable")
            return {}
        
        return refresh_provider_functions(provider)
    
    def disable_discovery(provider: str = None):
        """Disable discovery at runtime"""
        if provider:
            env_key = f'CHUK_LLM_{provider.upper()}_DISCOVERY'
            os.environ[env_key] = 'false'
            print(f"✅ Disabled discovery for {provider}")
        else:
            os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'false'
            print("✅ Disabled discovery globally")
    
    def enable_discovery(provider: str = None):
        """Enable discovery at runtime"""
        if provider:
            env_key = f'CHUK_LLM_{provider.upper()}_DISCOVERY'
            os.environ[env_key] = 'true'
            print(f"✅ Enabled discovery for {provider}")
        else:
            os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'true'
            print("✅ Enabled discovery globally")
    
    def test_discovery_bridge(provider_name: str = "ollama", model_name: str = "llama3.1:latest"):
        """Test the discovery bridge"""
        print(f"🧪 Testing Discovery Bridge: {provider_name}/{model_name}")
        
        # Step 1: Force discovery integration
        new_functions = force_discovery_integration()
        print(f"   📈 Discovery integration: {new_functions} new functions")
        
        # Step 2: Test _ensure_model_available
        config = get_config()
        resolved = config._ensure_model_available(provider_name, model_name)
        print(f"   🔍 Model resolution: {model_name} -> {resolved}")
        
        # Step 3: Check if model is now in provider list
        provider = config.get_provider(provider_name)
        in_list = model_name in provider.models
        print(f"   📋 In provider models: {in_list}")
        
        # Step 4: Try to get the function
        func_name = f"ask_{provider_name}_{_sanitize_name(model_name)}"
        try:
            func = globals().get(func_name) or getattr(sys.modules[__name__], func_name, None)
            has_func = func is not None
            print(f"   🎯 Function {func_name}: {'✅ Available' if has_func else '❌ Missing'}")
            return has_func
        except:
            print(f"   🎯 Function {func_name}: ❌ Error accessing")
            return False
    
    return {
        'quick_question': quick_question,
        'compare_providers': compare_providers,
        'show_config': show_config,
        'discover_and_refresh': discover_and_refresh,
        'refresh_provider_functions': refresh_provider_functions,
        'trigger_ollama_discovery_and_refresh': trigger_ollama_discovery_and_refresh,
        'disable_discovery': disable_discovery,
        'enable_discovery': enable_discovery,
        'test_discovery_bridge': test_discovery_bridge,
        'force_discovery_integration': force_discovery_integration,
    }


def refresh_provider_functions(provider_name: str = None):
    """Refresh functions for a specific provider or all providers"""
    global _GENERATED_FUNCTIONS, _FUNCTION_CACHE_DIRTY
    
    # Check if discovery is allowed
    if provider_name and not _is_discovery_enabled(provider_name):
        logger.warning(f"Discovery disabled for {provider_name} by environment variable")
        return {}
    
    config_manager = get_config()
    
    if provider_name:
        # Refresh specific provider
        try:
            provider_config = config_manager.get_provider(provider_name)
            
            if provider_name == "ollama":
                # For Ollama, force a fresh check of available models
                timeout = float(os.getenv('CHUK_LLM_DISCOVERY_TIMEOUT', '5'))
                discovered_models = _check_ollama_available_models(timeout=timeout)
                if discovered_models:
                    logger.info(f"Ollama refresh: found {len(discovered_models)} available models")
                    
                    # 🎯 CRITICAL FIX: Replace the entire model list with discovered models
                    original_models = provider_config.models.copy()
                    
                    # FIXED: Use complete discovered list instead of merging
                    provider_config.models = discovered_models  # Replace, don't merge!
                    
                    # FIXED: Force update the config manager's cache
                    config_manager.providers[provider_name] = provider_config
                    
                    # FIXED: Also update any internal caches the config manager might have
                    if hasattr(config_manager, '_provider_cache'):
                        config_manager._provider_cache[provider_name] = provider_config
                    
                    # Get fresh reference to ensure we're using the updated config
                    provider_config = config_manager.get_provider(provider_name)
                    
                    logger.info(f"Updated provider config: {len(original_models)} -> {len(provider_config.models)} models")
                    logger.info(f"New model list: {provider_config.models}")
                    
                    current_models = provider_config.models  # Use the updated list
                else:
                    logger.info("Ollama refresh: no models available")
                    current_models = provider_config.models  # Fall back to static models
            else:
                # For other providers, get all configured models
                current_models = provider_config.models
            
            logger.info(f"Generating functions for {len(current_models)} {provider_name} models")
            
            # Generate functions for current models
            new_functions = _generate_functions_for_models(provider_name, provider_config, current_models)
            
            # Update the global functions dict
            _GENERATED_FUNCTIONS.update(new_functions)
            
            # Update module namespace
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
        # Refresh all providers using the smart static generation
        all_new_functions = _generate_static_functions()
        _GENERATED_FUNCTIONS.update(all_new_functions)
        _FUNCTION_CACHE_DIRTY = False
        
        # Update module namespace
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
    """
    CRITICAL FIX: Use singleton pattern to ensure we're working with the same config instance
    Trigger Ollama model discovery and refresh functions - FIXED VERSION
    """
    if not _is_discovery_enabled('ollama'):
        logger.warning("Ollama discovery disabled by environment variable CHUK_LLM_OLLAMA_DISCOVERY=false")
        print("💡 To enable: export CHUK_LLM_OLLAMA_DISCOVERY=true")
        return {}
    
    try:
        # Check what's actually available right now
        timeout = float(os.getenv('CHUK_LLM_DISCOVERY_TIMEOUT', '5'))
        current_models = _check_ollama_available_models(timeout=timeout)
        
        if not current_models:
            logger.warning("No Ollama models available - is Ollama running?")
            return {}
        
        # 🎯 CRITICAL FIX: Use singleton config manager
        config_manager = get_config()
        
        # Get original count for logging
        original_provider = config_manager.get_provider("ollama")
        original_count = len(original_provider.models)
        
        logger.info(f"Before update: {original_count} models")
        logger.info(f"Available from Ollama: {len(current_models)} models")
        logger.info(f"Available models: {current_models}")
        
        # CRITICAL FIX: Update the provider's models list directly on the singleton
        original_provider.models = current_models
        
        # CRITICAL FIX: Also update the provider in the config manager's internal cache
        config_manager.providers["ollama"] = original_provider
        
        # Force any other internal caches to update
        if hasattr(config_manager, '_provider_cache'):
            config_manager._provider_cache["ollama"] = original_provider
        
        # Verify the update worked by getting a fresh reference
        updated_provider = config_manager.get_provider("ollama")
        logger.info(f"After update verification: {len(updated_provider.models)} models")
        logger.info(f"Updated model list: {updated_provider.models}")
        logger.info(f"Contains mistral-nemo:latest: {'mistral-nemo:latest' in updated_provider.models}")
        
        # Generate functions with the updated models
        logger.info(f"Generating functions for {len(current_models)} ollama models")
        new_functions = _generate_functions_for_models("ollama", updated_provider, current_models)
        
        # Update the global functions dict
        global _GENERATED_FUNCTIONS
        _GENERATED_FUNCTIONS.update(new_functions)
        
        # Update module namespace
        current_module = sys.modules[__name__]
        for name, func in new_functions.items():
            setattr(current_module, name, func)
        
        # Update main chuk_llm module too
        try:
            import chuk_llm
            for name, func in new_functions.items():
                if name.startswith(('ask_', 'stream_')) and 'ollama' in name:
                    setattr(chuk_llm, name, func)
        except:
            pass
        
        # Update __all__
        if hasattr(current_module, '__all__'):
            for name in new_functions.keys():
                if name not in current_module.__all__:
                    current_module.__all__.append(name)
        
        logger.info(f"Ollama discovery: {len(current_models)} available models, {len(new_functions)} functions generated")
        
        return new_functions
        
    except Exception as e:
        logger.error(f"Failed to trigger Ollama discovery: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}
    

def _ensure_model_available_for_get_client(provider_name: str, model_name: str) -> bool:
    """
    Enhanced model availability check that triggers discovery if needed.
    This is called by get_client() when a model isn't found.
    """
    if not _is_auto_discover_enabled() or not _is_discovery_enabled(provider_name):
        return False
    
    try:
        config_manager = get_config()
        provider_config = config_manager.get_provider(provider_name)
        
        # If model is already available, no need to discover
        if model_name in provider_config.models:
            return True
        
        # For Ollama, check if the model is actually available
        if provider_name == "ollama":
            timeout = float(os.getenv('CHUK_LLM_DISCOVERY_QUICK_TIMEOUT', '2.0'))
            available_models = _check_ollama_available_models(timeout=timeout)
            
            if model_name in available_models:
                # Model exists in Ollama but not in our config - add it
                logger.info(f"Discovered {model_name} in Ollama - updating provider config")
                
                # Update provider configuration
                provider_config.models.append(model_name)
                config_manager.providers[provider_name] = provider_config
                
                # Generate functions for this model
                model_functions = _generate_functions_for_models(provider_name, provider_config, [model_name])
                _GENERATED_FUNCTIONS.update(model_functions)
                
                # Update module namespace
                current_module = sys.modules[__name__]
                for name, func in model_functions.items():
                    setattr(current_module, name, func)
                
                logger.info(f"Added {model_name} to {provider_name} and generated {len(model_functions)} functions")
                return True
        
        return False
        
    except Exception as e:
        logger.debug(f"Model availability check failed for {provider_name}/{model_name}: {e}")
        return False


# Enhanced __getattr__ to support on-demand function generation with environment controls
def __getattr__(name):
    """FIXED: Allow access to generated functions with on-demand discovery"""
    # First check if it's in our generated functions
    if name in _GENERATED_FUNCTIONS:
        return _GENERATED_FUNCTIONS[name]
    
    # Check if auto-discovery is disabled
    if not _is_auto_discover_enabled():
        raise AttributeError(f"module 'providers' has no attribute '{name}' (auto-discovery disabled)")
    
    # 🎯 CRITICAL FIX: Try discovery integration first
    try:
        logger.debug(f"🔍 Function {name} not found, trying discovery integration...")
        new_count = force_discovery_integration()
        
        if new_count > 0 and name in _GENERATED_FUNCTIONS:
            logger.debug(f"✅ Found {name} after discovery integration!")
            return _GENERATED_FUNCTIONS[name]
    except Exception as e:
        logger.debug(f"Discovery integration failed: {e}")
    
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
            
            # Check if discovery is enabled for this provider
            if not _is_discovery_enabled(provider_name):
                raise AttributeError(f"module 'providers' has no attribute '{name}' (discovery disabled for {provider_name})")
            
            # 🎯 CRITICAL FIX: Force a specific model check
            try:
                config_manager = get_config()
                
                # Convert model_part back to actual model name (reverse sanitization)
                possible_models = [
                    model_part.replace('_', ':'),
                    model_part.replace('_', '.'),
                    model_part.replace('_', '-'),
                    model_part,
                    f"{model_part}:latest"
                ]
                
                for possible_model in possible_models:
                    resolved = config_manager._ensure_model_available(provider_name, possible_model)
                    if resolved:
                        logger.debug(f"✅ Model {possible_model} resolved to {resolved} via discovery")
                        # Refresh functions for this provider
                        new_functions = refresh_provider_functions(provider_name)
                        if name in new_functions:
                            return new_functions[name]
                        break
                
            except Exception as e:
                logger.debug(f"Model availability check failed: {e}")
            
            # Try to refresh functions for this provider
            try:
                new_functions = refresh_provider_functions(provider_name)
                if name in new_functions:
                    return new_functions[name]
            except Exception as e:
                logger.debug(f"Could not refresh functions for {provider_name}: {e}")
    
    # If we still don't have it, raise AttributeError
    raise AttributeError(f"module 'providers' has no attribute '{name}'")


# Generate all functions at module import with environment controls
startup_enabled = _is_startup_discovery_enabled()
discovery_enabled = _env_bool('CHUK_LLM_DISCOVERY_ENABLED', True)

if startup_enabled and discovery_enabled:
    logger.info("Generating dynamic provider functions from YAML...")
else:
    logger.info("Generating static provider functions (discovery disabled)...")

try:
    # Generate provider functions using smart generation
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
    
    # Show discovery status
    if not discovery_enabled:
        logger.info("🚫 Discovery globally disabled (CHUK_LLM_DISCOVERY_ENABLED=false)")
    elif not startup_enabled:
        logger.info("🚫 Startup discovery disabled (CHUK_LLM_DISCOVERY_ON_STARTUP=false)")
    
    # Log some examples
    examples = [name for name in __all__ 
               if any(x in name for x in ['gpt4', 'claude', 'llama']) 
               and not name.endswith('_sync')][:5]
    if examples:
        logger.info(f"Example functions: {', '.join(examples)}")

except Exception as e:
    logger.error(f"Error generating provider functions: {e}")
    # Fallback - at least provide utility functions
    __all__ = ['show_config', 'refresh_provider_functions', 'trigger_ollama_discovery_and_refresh', 'disable_discovery', 'enable_discovery', 'test_discovery_bridge', 'force_discovery_integration']
    
    def show_config():
        print(f"❌ Error loading configuration: {e}")
        print("Create a providers.yaml file to use ChukLLM")
    
    def refresh_provider_functions(provider_name: str = None):
        print(f"❌ Function refresh not available: {e}")
        return {}
    
    def trigger_ollama_discovery_and_refresh():
        print(f"❌ Discovery not available: {e}")
        return {}
    
    def disable_discovery(provider: str = None):
        print("❌ Discovery control not available")
    
    def enable_discovery(provider: str = None):
        print("❌ Discovery control not available")
    
    def test_discovery_bridge(provider_name: str = "ollama", model_name: str = "llama3.1:latest"):
        print("❌ Discovery bridge test not available")
        return False
    
    def force_discovery_integration():
        print("❌ Discovery integration not available")
        return 0
    
    globals()['show_config'] = show_config
    globals()['refresh_provider_functions'] = refresh_provider_functions
    globals()['trigger_ollama_discovery_and_refresh'] = trigger_ollama_discovery_and_refresh
    globals()['disable_discovery'] = disable_discovery
    globals()['enable_discovery'] = enable_discovery
    globals()['test_discovery_bridge'] = test_discovery_bridge
    globals()['force_discovery_integration'] = force_discovery_integration


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


def apply_discovery_bridge_fix():
    """
    CRITICAL DISCOVERY BRIDGE FIX
    =============================
    
    This function fixes the gap between discovery and inference by ensuring
    that discovered models are immediately added to provider model lists and
    functions are generated for them.
    
    Call this function at startup or when discovery results aren't appearing.
    """
    
    print("🔧 Applying Critical Discovery Bridge Fix...")
    
    config_manager = get_config()
    
    # Store original _ensure_model_available method
    original_ensure = getattr(config_manager, '_ensure_model_available', None)
    if not original_ensure:
        print("❌ _ensure_model_available method not found")
        return False
    
    def fixed_ensure_model_available(provider_name: str, model_name: str):
        """
        FIXED VERSION: Actually adds discovered models to provider lists
        and generates functions immediately
        """
        if not model_name:
            return None
        
        provider = config_manager.providers[provider_name]
        
        # Step 1: Check if already available (exact match)
        if model_name in provider.models:
            return model_name
        
        # Step 2: Check aliases and variants
        resolved = provider.model_aliases.get(model_name, model_name)
        if resolved in provider.models:
            return resolved
        
        # Step 3: Try :latest variants
        variants_to_check = []
        if model_name.endswith(':latest'):
            base_name = model_name.replace(':latest', '')
            variants_to_check.append(base_name)
        else:
            latest_name = f"{model_name}:latest"
            variants_to_check.append(latest_name)
        
        for variant in variants_to_check:
            if variant in provider.models:
                return variant
        
        # Step 4: 🎯 CRITICAL FIX - Check discovery cache and ADD MODEL
        cache_data = getattr(config_manager, '_discovery_cache', {}).get(provider_name)
        if cache_data and 'models' in cache_data:
            cached_models = cache_data['models']
            
            # Check all variants
            all_variants = [model_name] + variants_to_check
            
            for variant in all_variants:
                if variant in cached_models and variant not in provider.models:
                    # 🎯 CRITICAL: Actually add the model to provider
                    provider.models.append(variant)
                    print(f"✅ Added {variant} to {provider_name} from discovery cache")
                    
                    # 🎯 CRITICAL: Generate functions immediately
                    try:
                        global _GENERATED_FUNCTIONS
                        model_functions = _generate_functions_for_models(provider_name, provider, [variant])
                        _GENERATED_FUNCTIONS.update(model_functions)
                        
                        # Add to module namespace
                        current_module = sys.modules[__name__]
                        for name, func in model_functions.items():
                            setattr(current_module, name, func)
                        
                        # Update __all__
                        if hasattr(current_module, '__all__'):
                            for name in model_functions.keys():
                                if name not in current_module.__all__:
                                    current_module.__all__.append(name)
                        
                        print(f"✅ Generated {len(model_functions)} functions for {variant}")
                        
                    except Exception as e:
                        print(f"⚠️ Function generation failed for {variant}: {e}")
                    
                    return variant
        
        # Step 5: No luck in cache - try force discovery (if enabled)
        if (hasattr(config_manager, '_is_discovery_enabled') and 
            config_manager._is_discovery_enabled(provider_name)):
            
            try:
                print(f"🔍 Force discovering models for {provider_name}...")
                discovered = config_manager.force_discover_models(provider_name)
                
                if discovered > 0:
                    print(f"✅ Discovered {discovered} new models for {provider_name}")
                    
                    # Re-check after discovery
                    updated_provider = config_manager.providers[provider_name]
                    all_variants = [model_name] + variants_to_check
                    
                    for variant in all_variants:
                        if variant in updated_provider.models:
                            print(f"✅ Found {variant} after force discovery")
                            return variant
                
            except Exception as e:
                print(f"⚠️ Force discovery failed: {e}")
        
        # Model not found anywhere
        return original_ensure(provider_name, model_name) if original_ensure else None
    
    # Replace the method
    config_manager._ensure_model_available = fixed_ensure_model_available
    
    print("✅ Discovery bridge fix applied!")
    print("   Discovered models will now be immediately available for inference")
    return True


def test_discovery_fix():
    """Test that the discovery bridge fix works"""
    
    print("\n🧪 Testing Discovery Bridge Fix")
    print("=" * 50)
    
    # Apply the fix first
    fix_applied = apply_discovery_bridge_fix()
    if not fix_applied:
        print("❌ Could not apply discovery bridge fix")
        return False
    
    # Test cases
    test_cases = [
        ("ollama", "llama3.1:latest"),
        ("ollama", "qwen3:latest"),
        ("openai", "o1-mini-2024-09-12"),
    ]
    
    success_count = 0
    
    for provider, model in test_cases:
        print(f"\n🎯 Testing: {provider}/{model}")
        
        try:
            config = get_config()
            
            # Get initial state
            provider_obj = config.get_provider(provider)
            initial_models = set(provider_obj.models)
            print(f"   📋 Initial models: {len(initial_models)}")
            
            # Test _ensure_model_available
            resolved = config._ensure_model_available(provider, model)
            print(f"   🔍 Resolution result: {resolved}")
            
            # Check if model was added
            current_models = set(provider_obj.models)
            new_models = current_models - initial_models
            
            if new_models:
                print(f"   📈 New models added: {list(new_models)}")
                
                # Check if function exists
                func_name = f"ask_{provider}_{_sanitize_name(model)}"
                try:
                    func = globals().get(func_name)
                    if func:
                        print(f"   ✅ Function {func_name} is available!")
                        success_count += 1
                    else:
                        print(f"   ❌ Function {func_name} not found")
                except:
                    print(f"   ❌ Error accessing function {func_name}")
            else:
                if resolved:
                    print(f"   ✅ Model already available (resolved to {resolved})")
                    success_count += 1
                else:
                    print(f"   ❌ Model not resolved and not added")
                    
        except Exception as e:
            print(f"   💥 Test failed: {e}")
    
    success_rate = success_count / len(test_cases)
    print(f"\n📊 Test Results: {success_count}/{len(test_cases)} successful ({success_rate:.1%})")
    
    if success_rate >= 0.5:
        print("✅ Discovery bridge is working!")
        return True
    else:
        print("❌ Discovery bridge needs more work")
        return False


# Add these functions to the utility functions
def _create_utility_functions():
    """Create utility functions with discovery status - ENHANCED VERSION"""
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
        """Show current configuration status including discovery settings"""
        config = get_config()
        
        print("🔧 ChukLLM Configuration")
        print("=" * 50)
        
        # Discovery settings
        print("\n🔍 Discovery Settings:")
        discovery_enabled = _env_bool('CHUK_LLM_DISCOVERY_ENABLED', True)
        startup_enabled = _env_bool('CHUK_LLM_DISCOVERY_ON_STARTUP', True)
        auto_enabled = _env_bool('CHUK_LLM_AUTO_DISCOVER', True)
        
        print(f"  Global Discovery: {'✅ Enabled' if discovery_enabled else '❌ Disabled'}")
        print(f"  Startup Check:    {'✅ Enabled' if startup_enabled else '❌ Disabled'}")
        print(f"  Auto Discovery:   {'✅ Enabled' if auto_enabled else '❌ Disabled'}")
        
        timeout = os.getenv('CHUK_LLM_DISCOVERY_TIMEOUT', '5')
        quick_timeout = os.getenv('CHUK_LLM_DISCOVERY_QUICK_TIMEOUT', '2.0')
        print(f"  Timeout:          {timeout}s (quick: {quick_timeout}s)")
        
        providers = config.get_all_providers()
        print(f"\n📦 Providers: {len(providers)}")
        for provider_name in providers:
            try:
                provider = config.get_provider(provider_name)
                has_key = "✅" if config.get_api_key(provider_name) else "❌"
                discovery_status = "🔍" if _is_discovery_enabled(provider_name) else "🚫"
                print(f"  {has_key} {discovery_status} {provider_name:<12} | {len(provider.models):2d} models | {len(provider.model_aliases):2d} aliases")
            except Exception as e:
                print(f"  ❌ 🚫 {provider_name:<12} | Error: {e}")
        
        aliases = config.get_global_aliases()
        if aliases:
            print(f"\n🌍 Global Aliases: {len(aliases)}")
            for alias, target in list(aliases.items())[:5]:
                print(f"  ask_{alias}() -> {target}")
            if len(aliases) > 5:
                print(f"  ... and {len(aliases) - 5} more")
        
        print(f"\n🎛️  Environment Controls:")
        print(f"  CHUK_LLM_DISCOVERY_ENABLED={_env_bool('CHUK_LLM_DISCOVERY_ENABLED', True)}")
        print(f"  CHUK_LLM_DISCOVERY_ON_STARTUP={_env_bool('CHUK_LLM_DISCOVERY_ON_STARTUP', True)}")
        print(f"  CHUK_LLM_AUTO_DISCOVER={_env_bool('CHUK_LLM_AUTO_DISCOVER', True)}")
        print(f"  CHUK_LLM_OLLAMA_DISCOVERY={_env_bool('CHUK_LLM_OLLAMA_DISCOVERY', True)}")
        
        # Show discovery methods
        print(f"\n🎮 Discovery Control Methods:")
        print(f"  # Disable all discovery")
        print(f"  export CHUK_LLM_DISCOVERY_ENABLED=false")
        print(f"  ")
        print(f"  # Disable only Ollama discovery")
        print(f"  export CHUK_LLM_OLLAMA_DISCOVERY=false")
        print(f"  ")
        print(f"  # Disable startup checks (but allow on-demand)")
        print(f"  export CHUK_LLM_DISCOVERY_ON_STARTUP=false")
    
    def discover_and_refresh(provider: str = "ollama"):
        """Discover models and refresh functions for a provider"""
        if not _is_discovery_enabled(provider):
            print(f"❌ Discovery disabled for {provider}")
            print(f"   Set CHUK_LLM_{provider.upper()}_DISCOVERY=true to enable")
            return {}
        
        return refresh_provider_functions(provider)
    
    def disable_discovery(provider: str = None):
        """Disable discovery at runtime"""
        if provider:
            env_key = f'CHUK_LLM_{provider.upper()}_DISCOVERY'
            os.environ[env_key] = 'false'
            print(f"✅ Disabled discovery for {provider}")
        else:
            os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'false'
            print("✅ Disabled discovery globally")
    
    def enable_discovery(provider: str = None):
        """Enable discovery at runtime"""
        if provider:
            env_key = f'CHUK_LLM_{provider.upper()}_DISCOVERY'
            os.environ[env_key] = 'true'
            print(f"✅ Enabled discovery for {provider}")
        else:
            os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'true'
            print("✅ Enabled discovery globally")
    
    def test_discovery_bridge(provider_name: str = "ollama", model_name: str = "llama3.1:latest"):
        """Test the discovery bridge"""
        print(f"🧪 Testing Discovery Bridge: {provider_name}/{model_name}")
        
        # Step 1: Force discovery integration
        new_functions = force_discovery_integration()
        print(f"   📈 Discovery integration: {new_functions} new functions")
        
        # Step 2: Test _ensure_model_available
        config = get_config()
        resolved = config._ensure_model_available(provider_name, model_name)
        print(f"   🔍 Model resolution: {model_name} -> {resolved}")
        
        # Step 3: Check if model is now in provider list
        provider = config.get_provider(provider_name)
        in_list = model_name in provider.models
        print(f"   📋 In provider models: {in_list}")
        
        # Step 4: Try to get the function
        func_name = f"ask_{provider_name}_{_sanitize_name(model_name)}"
        try:
            func = globals().get(func_name) or getattr(sys.modules[__name__], func_name, None)
            has_func = func is not None
            print(f"   🎯 Function {func_name}: {'✅ Available' if has_func else '❌ Missing'}")
            return has_func
        except:
            print(f"   🎯 Function {func_name}: ❌ Error accessing")
            return False
    
    return {
        'quick_question': quick_question,
        'compare_providers': compare_providers,
        'show_config': show_config,
        'discover_and_refresh': discover_and_refresh,
        'refresh_provider_functions': refresh_provider_functions,
        'trigger_ollama_discovery_and_refresh': trigger_ollama_discovery_and_refresh,
        'disable_discovery': disable_discovery,
        'enable_discovery': enable_discovery,
        'test_discovery_bridge': test_discovery_bridge,
        'force_discovery_integration': force_discovery_integration,
        'apply_discovery_bridge_fix': apply_discovery_bridge_fix,
        'test_discovery_fix': test_discovery_fix,
    }