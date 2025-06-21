# chuk_llm/api/discovery.py
"""
Universal model discovery API that works with any provider
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager, DiscoveredModel
from chuk_llm.llm.discovery.providers import DiscovererFactory
from chuk_llm.configuration import get_config

log = logging.getLogger(__name__)

# Global discovery managers cache
_discovery_managers: Dict[str, UniversalModelDiscoveryManager] = {}


def get_discovery_manager(
    provider_name: str, 
    force_recreate: bool = False,
    inference_config: Optional[Dict[str, Any]] = None,
    **discoverer_config
) -> UniversalModelDiscoveryManager:
    """
    Get or create a discovery manager for a provider.
    
    Args:
        provider_name: Name of the provider
        force_recreate: Force recreation of manager
        inference_config: Custom inference configuration
        **discoverer_config: Provider-specific discoverer configuration
    
    Returns:
        Universal discovery manager for the provider
    """
    cache_key = f"{provider_name}_{hash(str(inference_config))}"
    
    if force_recreate or cache_key not in _discovery_managers:
        # Get provider configuration for discoverer
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(provider_name)
            
            # Merge provider config with discoverer config
            merged_config = {
                "api_base": provider_config.api_base,
                **provider_config.extra,
                **discoverer_config
            }
            
            # Add API key if available
            api_key = config_manager.get_api_key(provider_name)
            if api_key:
                merged_config["api_key"] = api_key
                
        except Exception as e:
            log.debug(f"Could not load provider config for {provider_name}: {e}")
            merged_config = discoverer_config
        
        # Create discoverer
        discoverer = DiscovererFactory.create_discoverer(provider_name, **merged_config)
        
        # Create universal manager
        manager = UniversalModelDiscoveryManager(
            provider_name=provider_name,
            discoverer=discoverer,
            inference_config=inference_config
        )
        
        _discovery_managers[cache_key] = manager
    
    return _discovery_managers[cache_key]


async def discover_models(
    provider_name: str,
    force_refresh: bool = False,
    inference_config: Optional[Dict[str, Any]] = None,
    inference_config_path: Optional[str] = None,
    **discoverer_config
) -> List[Dict[str, Any]]:
    """
    Universal model discovery that works with any provider.
    
    Args:
        provider_name: Name of the provider (ollama, openai, huggingface, etc.)
        force_refresh: Force refresh of model cache
        inference_config: Custom inference configuration dict
        inference_config_path: Path to custom inference config YAML
        **discoverer_config: Provider-specific configuration
    
    Returns:
        List of discovered model dictionaries
    """
    # Load inference config from file if provided
    if inference_config_path and not inference_config:
        try:
            import yaml
            with open(inference_config_path, 'r') as f:
                inference_config = yaml.safe_load(f)
        except Exception as e:
            log.error(f"Failed to load inference config from {inference_config_path}: {e}")
            inference_config = None
    
    # Get discovery manager
    manager = get_discovery_manager(
        provider_name=provider_name,
        force_recreate=bool(inference_config or inference_config_path),
        inference_config=inference_config,
        **discoverer_config
    )
    
    # Discover models
    models = await manager.discover_models(force_refresh=force_refresh)
    
    # Convert to API format
    return [
        {
            "name": model.name,
            "provider": model.provider,
            "family": model.family,
            "size_gb": round(model.size_bytes / (1024**3), 1) if model.size_bytes else None,
            "parameters": model.parameters,
            "context_length": model.context_length,
            "max_output_tokens": model.max_output_tokens,
            "features": [f.value for f in model.capabilities],
            "created_at": model.created_at,
            "modified_at": model.modified_at,
            "metadata": model.metadata,
        }
        for model in models
    ]


def discover_models_sync(provider_name: str, **kwargs) -> List[Dict[str, Any]]:
    """Sync version of discover_models"""
    return asyncio.run(discover_models(provider_name, **kwargs))


async def update_provider_configuration(
    provider_name: str,
    inference_config: Optional[Dict[str, Any]] = None,
    inference_config_path: Optional[str] = None,
    **discoverer_config
) -> Dict[str, Any]:
    """
    Update provider configuration with discovered models.
    
    Args:
        provider_name: Name of the provider
        inference_config: Custom inference configuration dict
        inference_config_path: Path to custom inference config YAML
        **discoverer_config: Provider-specific configuration
    
    Returns:
        Update results dictionary
    """
    try:
        # Discover models
        models = await discover_models(
            provider_name=provider_name,
            force_refresh=True,
            inference_config=inference_config,
            inference_config_path=inference_config_path,
            **discoverer_config
        )
        
        if not models:
            return {
                "success": False,
                "error": f"No models discovered for {provider_name}",
                "total_models": 0,
            }
        
        # Update provider configuration
        config_manager = get_config()
        provider = config_manager.get_provider(provider_name)
        
        # Get discovery manager for capabilities
        manager = get_discovery_manager(
            provider_name=provider_name,
            inference_config=inference_config,
            **discoverer_config
        )
        
        # Update models list (exclude models without text capability)
        text_models = [m["name"] for m in models if "text" in m["features"]]
        if text_models:
            provider.models = text_models
            if not provider.default_model or provider.default_model not in text_models:
                provider.default_model = text_models[0]
        
        # Update model capabilities
        provider.model_capabilities.clear()
        for model_name in text_models:
            caps = manager.get_model_capabilities(model_name)
            if caps:
                provider.model_capabilities.append(caps)
        
        # Store discovery configuration
        if not hasattr(provider.extra, "model_discovery"):
            provider.extra["model_discovery"] = {}
        
        if inference_config:
            provider.extra["model_discovery"]["inference_config"] = inference_config
        elif inference_config_path:
            provider.extra["model_discovery"]["inference_config_path"] = inference_config_path
        
        # Collect stats
        families = set(m["family"] for m in models)
        feature_counts = {}
        for model in models:
            for feature in model["features"]:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        return {
            "success": True,
            "provider": provider_name,
            "total_models": len(models),
            "text_models": len(text_models),
            "other_models": len(models) - len(text_models),
            "default_model": text_models[0] if text_models else None,
            "families": list(families),
            "feature_summary": feature_counts,
            "config_source": "custom" if (inference_config or inference_config_path) else "provider",
        }
        
    except Exception as e:
        log.error(f"Failed to update {provider_name} configuration: {e}")
        return {
            "success": False,
            "provider": provider_name,
            "error": str(e),
        }


def update_provider_configuration_sync(provider_name: str, **kwargs) -> Dict[str, Any]:
    """Sync version of update_provider_configuration"""
    return asyncio.run(update_provider_configuration(provider_name, **kwargs))


async def show_discovered_models(
    provider_name: str,
    force_refresh: bool = False,
    inference_config: Optional[Dict[str, Any]] = None,
    inference_config_path: Optional[str] = None,
    **discoverer_config
) -> None:
    """
    Display discovered models in a nice format.
    
    Args:
        provider_name: Name of the provider
        force_refresh: Force refresh of model cache
        inference_config: Custom inference configuration dict
        inference_config_path: Path to custom inference config YAML
        **discoverer_config: Provider-specific configuration
    """
    # Get discovery manager for stats
    manager = get_discovery_manager(
        provider_name=provider_name,
        inference_config=inference_config,
        **discoverer_config
    )
    
    models = await manager.discover_models(force_refresh=force_refresh)
    stats = manager.get_discovery_stats()
    
    # Show header
    print(f"\n🔍 Discovered {stats['total']} {provider_name.title()} Models")
    print("=" * 60)
    
    # Show config source
    if inference_config or inference_config_path:
        config_source = inference_config_path or "custom dict"
        print(f"🔧 Using custom inference config: {config_source}")
    else:
        print("🔧 Using provider inference config")
    
    # Show stats
    if stats["total_size_gb"] > 0:
        print(f"📊 Total size: {stats['total_size_gb']}GB")
    if stats["cache_age_seconds"] > 0:
        print(f"⏰ Cache age: {stats['cache_age_seconds']}s")
    
    # Group by family
    families = {}
    for model in models:
        family = model.family
        if family not in families:
            families[family] = []
        families[family].append(model)
    
    for family, family_models in sorted(families.items()):
        print(f"\n📁 {family.title()} Models ({len(family_models)}):")
        
        for model in sorted(family_models, key=lambda x: x.name):
            size_gb = f"{model.size_bytes / (1024**3):.1f}GB" if model.size_bytes else "Unknown"
            features = ", ".join(f.value for f in sorted(model.capabilities))
            
            print(f"  • {model.name}")
            print(f"    Size: {size_gb} | Context: {model.context_length or 'Unknown'}")
            if model.parameters:
                print(f"    Parameters: {model.parameters}")
            print(f"    Features: {features}")
            if model.metadata:
                if "downloads" in model.metadata:
                    print(f"    Downloads: {model.metadata['downloads']:,}")
                if "likes" in model.metadata:
                    print(f"    Likes: {model.metadata['likes']:,}")
            print()


def show_discovered_models_sync(provider_name: str, **kwargs) -> None:
    """Sync version of show_discovered_models"""
    asyncio.run(show_discovered_models(provider_name, **kwargs))


async def get_model_info(
    provider_name: str,
    model_name: str,
    **discoverer_config
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific discovered model.
    
    Args:
        provider_name: Name of the provider
        model_name: Name of the model
        **discoverer_config: Provider-specific configuration
    
    Returns:
        Model information dictionary or None if not found
    """
    manager = get_discovery_manager(provider_name=provider_name, **discoverer_config)
    models = await manager.discover_models()
    
    for model in models:
        if model.name == model_name:
            return {
                "name": model.name,
                "provider": model.provider,
                "family": model.family,
                "size_gb": round(model.size_bytes / (1024**3), 1) if model.size_bytes else None,
                "size_bytes": model.size_bytes,
                "parameters": model.parameters,
                "context_length": model.context_length,
                "max_output_tokens": model.max_output_tokens,
                "features": [f.value for f in model.capabilities],
                "supports_chat": "text" in [f.value for f in model.capabilities],
                "supports_vision": "vision" in [f.value for f in model.capabilities],
                "supports_tools": "tools" in [f.value for f in model.capabilities],
                "created_at": model.created_at,
                "modified_at": model.modified_at,
                "metadata": model.metadata,
            }
    
    return None


def get_model_info_sync(provider_name: str, model_name: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Sync version of get_model_info"""
    return asyncio.run(get_model_info(provider_name, model_name, **kwargs))


async def generate_provider_config_yaml(
    provider_name: str,
    **discoverer_config
) -> str:
    """
    Generate YAML configuration for discovered models.
    
    Args:
        provider_name: Name of the provider
        **discoverer_config: Provider-specific configuration
    
    Returns:
        YAML configuration string
    """
    manager = get_discovery_manager(provider_name=provider_name, **discoverer_config)
    await manager.discover_models()
    return manager.generate_config_yaml()


def generate_provider_config_yaml_sync(provider_name: str, **kwargs) -> str:
    """Sync version of generate_provider_config_yaml"""
    return asyncio.run(generate_provider_config_yaml(provider_name, **kwargs))


def list_supported_providers() -> List[str]:
    """List providers that support dynamic discovery"""
    return DiscovererFactory.list_supported_providers()


def register_custom_discoverer(provider_name: str, discoverer_class: type):
    """Register a custom discoverer for a provider"""
    DiscovererFactory.register_discoverer(provider_name, discoverer_class)


# Export universal API
__all__ = [
    "discover_models",
    "discover_models_sync",
    "update_provider_configuration",
    "update_provider_configuration_sync", 
    "show_discovered_models",
    "show_discovered_models_sync",
    "get_model_info",
    "get_model_info_sync",
    "generate_provider_config_yaml",
    "generate_provider_config_yaml_sync",
    "list_supported_providers",
    "register_custom_discoverer",
]