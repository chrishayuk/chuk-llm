# chuk_llm/configuration/discovery.py
"""
Discovery integration for configuration manager
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any

from .models import ProviderConfig, ModelCapabilities, Feature, DiscoveryConfig

logger = logging.getLogger(__name__)


class ConfigDiscoveryMixin:
    """
    Mixin that adds discovery capabilities to configuration manager.
    Only performs discovery on-demand, no background tasks.
    """
    
    def __init__(self):
        # Discovery state (internal)
        self._discovery_managers: Dict[str, Any] = {}
        self._discovery_cache: Dict[str, Dict[str, Any]] = {}  # provider -> {models, timestamp}
    
    def _parse_discovery_config(self, provider_data: Dict[str, Any]) -> Optional[DiscoveryConfig]:
        """Parse discovery configuration from provider YAML"""
        discovery_data = provider_data.get("extra", {}).get("dynamic_discovery")
        if not discovery_data or not discovery_data.get("enabled", False):
            return None
        
        return DiscoveryConfig(
            enabled=discovery_data.get("enabled", False),
            discoverer_type=discovery_data.get("discoverer_type"),
            cache_timeout=discovery_data.get("cache_timeout", 300),
            inference_config=discovery_data.get("inference_config", {}),
            discoverer_config=discovery_data.get("discoverer_config", {})
        )
    
    async def _refresh_provider_models(self, provider_name: str, discovery_config: DiscoveryConfig) -> bool:
        """Refresh models for provider using discovery (internal method)"""
        # Check cache first
        cache_key = provider_name
        cached_data = self._discovery_cache.get(cache_key)
        if cached_data:
            cache_age = time.time() - cached_data["timestamp"]
            if cache_age < discovery_config.cache_timeout:
                logger.debug(f"Using cached discovery for {provider_name} (age: {cache_age:.1f}s)")
                return True
        
        try:
            # Get discovery manager
            manager = await self._get_discovery_manager(provider_name, discovery_config)
            if not manager:
                return False
            
            # Discover models
            discovered_models = await manager.discover_models()
            text_models = [m for m in discovered_models if hasattr(m, 'capabilities') and 
                          any(f.value == 'text' for f in m.capabilities)]
            
            if not text_models:
                logger.debug(f"No text models discovered for {provider_name}")
                return False
            
            # Update provider configuration seamlessly
            provider = self.providers[provider_name]
            static_models = set(provider.models)
            
            # Merge models (static take precedence)
            new_model_names = []
            new_capabilities = []
            
            # Keep all static models
            new_model_names.extend(provider.models)
            
            # Add new discovered models
            for model in text_models:
                if model.name not in static_models:
                    new_model_names.append(model.name)
                    
                    # Create capabilities for discovered model
                    new_capabilities.append(ModelCapabilities(
                        pattern=f"^{re.escape(model.name)}$",
                        features=model.capabilities,
                        max_context_length=model.context_length,
                        max_output_tokens=model.max_output_tokens
                    ))
            
            # Update provider (seamlessly)
            provider.models = new_model_names
            provider.model_capabilities.extend(new_capabilities)
            
            # Cache results
            self._discovery_cache[cache_key] = {
                "models": new_model_names,
                "timestamp": time.time(),
                "discovered_count": len(text_models),
                "new_count": len(new_model_names) - len(static_models)
            }
            
            logger.info(f"Discovery updated {provider_name}: {len(new_model_names)} total models "
                       f"({self._discovery_cache[cache_key]['new_count']} discovered)")
            return True
            
        except Exception as e:
            logger.debug(f"Discovery failed for {provider_name}: {e}")
            return False
    
    async def _get_discovery_manager(self, provider_name: str, discovery_config: DiscoveryConfig):
        """Get discovery manager for provider (internal method)"""
        if provider_name in self._discovery_managers:
            return self._discovery_managers[provider_name]
        
        try:
            # Import discovery components
            from chuk_llm.llm.discovery.engine import UniversalModelDiscoveryManager
            from chuk_llm.llm.discovery.providers import DiscovererFactory
            
            # Build discoverer config from provider config + discovery config
            provider = self.providers[provider_name]
            discoverer_config = {
                **discovery_config.discoverer_config,
                "api_base": provider.api_base,
            }
            
            # Add other provider config fields that might be useful
            for key, value in provider.extra.items():
                if key != "dynamic_discovery" and value is not None:
                    discoverer_config[key] = value
            
            # Add API key if available
            api_key = self.get_api_key(provider_name)
            if api_key:
                discoverer_config["api_key"] = api_key
            
            # Create discoverer using configured type
            discoverer_type = discovery_config.discoverer_type or provider_name
            discoverer = DiscovererFactory.create_discoverer(discoverer_type, **discoverer_config)
            
            # Create universal manager
            manager = UniversalModelDiscoveryManager(
                provider_name=provider_name,
                discoverer=discoverer,
                inference_config=discovery_config.inference_config
            )
            
            self._discovery_managers[provider_name] = manager
            return manager
            
        except Exception as e:
            logger.debug(f"Could not create discovery manager for {provider_name}: {e}")
            return None
    
    def _ensure_model_available(self, provider_name: str, model_name: Optional[str]) -> Optional[str]:
        """
        Ensure model is available, trigger discovery if needed.
        Returns resolved model name or None if not found.
        Completely transparent to caller.
        """
        if not model_name:
            return None
        
        provider = self.providers[provider_name]
        
        # Check if model exists (including aliases)
        resolved_model = provider.model_aliases.get(model_name, model_name)
        if resolved_model in provider.models:
            return resolved_model
        
        # Model not found - check if discovery is enabled
        discovery_config = self._parse_discovery_config({"extra": provider.extra})
        if not discovery_config or not discovery_config.enabled:
            return None  # No discovery available
        
        # Try discovery (with timeout to avoid blocking)
        try:
            # Create task for discovery
            async def _discover_and_check():
                success = await self._refresh_provider_models(provider_name, discovery_config)
                if success:
                    # Re-check if model is now available
                    updated_provider = self.providers[provider_name]
                    resolved_model = updated_provider.model_aliases.get(model_name, model_name)
                    if resolved_model in updated_provider.models:
                        return resolved_model
                return None
            
            # Run with a reasonable timeout
            try:
                loop = asyncio.get_event_loop()
                task = loop.create_task(_discover_and_check())
                
                # If we have a running loop, we can await
                result = asyncio.wait_for(task, timeout=5.0)  # 5 second timeout
                return loop.run_until_complete(result)
            except RuntimeError:
                # No event loop, skip discovery for now
                return None
            
        except (asyncio.TimeoutError, RuntimeError):
            logger.debug(f"Discovery timeout for {provider_name}/{model_name}")
            return None
        except Exception as e:
            logger.debug(f"Discovery error for {provider_name}/{model_name}: {e}")
            return None
    
    def reload(self):
        """Enhanced reload that clears discovery state"""
        # Clear discovery state
        self._discovery_managers.clear()
        self._discovery_cache.clear()
        
        # Call parent reload if it exists
        if hasattr(super(), 'reload'):
            super().reload()