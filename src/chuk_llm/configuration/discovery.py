# chuk_llm/configuration/discovery.py - Fully clean version
"""
Discovery integration for configuration manager - FULLY CLEAN VERSION
No hardcoded provider lists or configurations anywhere.
"""

import asyncio
import logging
import os
import re
import time
from typing import Dict, List, Optional, Set, Any

from .models import ProviderConfig, ModelCapabilities, Feature, DiscoveryConfig

logger = logging.getLogger(__name__)


class ConfigDiscoveryMixin:
    """
    Mixin that adds discovery capabilities to configuration manager.
    FULLY CLEAN: No hardcoded provider lists or configurations.
    """
    
    def __init__(self):
        # Discovery state (internal)
        self._discovery_managers: Dict[str, Any] = {}
        self._discovery_cache: Dict[str, Dict[str, Any]] = {}  # provider -> {models, timestamp}
        
        # CRITICAL: Track discovered models separately from static config
        self._discovered_models: Dict[str, Set[str]] = {}  # provider -> {model_names}
        self._discovered_capabilities: Dict[str, List[ModelCapabilities]] = {}  # provider -> capabilities
        
        # Cache discovery settings from environment
        self._discovery_settings = self._load_discovery_settings()
    
    def _load_discovery_settings(self) -> Dict[str, Any]:
        """Load discovery settings from environment variables"""
        settings = {
            # Global discovery controls
            'enabled': self._env_bool('CHUK_LLM_DISCOVERY_ENABLED', True),
            'startup_check': self._env_bool('CHUK_LLM_DISCOVERY_ON_STARTUP', True),
            'auto_discover': self._env_bool('CHUK_LLM_AUTO_DISCOVER', True),
            'timeout': int(os.getenv('CHUK_LLM_DISCOVERY_TIMEOUT', '5')),
            
            # Performance controls
            'cache_timeout': int(os.getenv('CHUK_LLM_DISCOVERY_CACHE_TIMEOUT', '300')),
            'max_concurrent': int(os.getenv('CHUK_LLM_DISCOVERY_MAX_CONCURRENT', '3')),
            'quick_check_timeout': float(os.getenv('CHUK_LLM_DISCOVERY_QUICK_TIMEOUT', '2.0')),
            
            # Debug and development
            'debug': self._env_bool('CHUK_LLM_DISCOVERY_DEBUG', False),
            'force_refresh': self._env_bool('CHUK_LLM_DISCOVERY_FORCE_REFRESH', False),
        }
        
        if settings['debug']:
            logger.info(f"Discovery settings loaded: {settings}")
        
        return settings
    
    def _env_bool(self, key: str, default: bool = False) -> bool:
        """Parse boolean environment variable"""
        value = os.getenv(key, '').lower()
        if value in ('true', '1', 'yes', 'on', 'enabled'):
            return True
        elif value in ('false', '0', 'no', 'off', 'disabled'):
            return False
        else:
            return default
    
    def _is_discovery_enabled(self, provider_name: str = None) -> bool:
        """Check if discovery is enabled globally and for specific provider"""
        # Global check
        if not self._discovery_settings['enabled']:
            return False
        
        # Provider-specific check using dynamic environment variable
        if provider_name:
            provider_env_key = f'CHUK_LLM_{provider_name.upper()}_DISCOVERY'
            return self._env_bool(provider_env_key, True)  # Default enabled for any provider
        
        return True
    
    def _parse_discovery_config(self, provider_data: Dict[str, Any]) -> Optional[DiscoveryConfig]:
        """Parse discovery configuration from provider YAML with environment overrides"""
        discovery_data = provider_data.get("extra", {}).get("dynamic_discovery")
        if not discovery_data:
            return None
        
        # Check if explicitly disabled in config first
        enabled = discovery_data.get("enabled", False)
        if not enabled:
            return None
        
        # Check if discovery is disabled by environment
        provider_name = provider_data.get("name", "unknown")
        if not self._is_discovery_enabled(provider_name):
            logger.debug(f"Discovery disabled for {provider_name} by environment variable")
            return None
        
        # Apply environment overrides
        if not self._discovery_settings['enabled']:
            return None
        
        cache_timeout = discovery_data.get("cache_timeout", self._discovery_settings['cache_timeout'])
        if self._discovery_settings['force_refresh']:
            cache_timeout = 0
        
        return DiscoveryConfig(
            enabled=True,  # We know it's enabled if we get here
            discoverer_type=discovery_data.get("discoverer_type"),
            cache_timeout=cache_timeout,
            inference_config=discovery_data.get("inference_config", {}),
            discoverer_config=discovery_data.get("discoverer_config", {})
        )
    
    async def _get_discovery_manager(self, provider_name: str, discovery_config: DiscoveryConfig):
        """Get or create discovery manager for provider"""
        if provider_name in self._discovery_managers:
            return self._discovery_managers[provider_name]
        
        try:
            from chuk_llm.llm.discovery import DiscovererFactory, UniversalModelDiscoveryManager
            
            # Build discoverer config from provider configuration
            discoverer_config = self._build_discoverer_config(provider_name, discovery_config)
            
            # Create discoverer
            discoverer = DiscovererFactory.create_discoverer(provider_name, **discoverer_config)
            
            # Create universal manager
            inference_config = discovery_config.inference_config
            manager = UniversalModelDiscoveryManager(provider_name, discoverer, inference_config)
            
            self._discovery_managers[provider_name] = manager
            return manager
            
        except Exception as e:
            logger.error(f"Failed to create discovery manager for {provider_name}: {e}")
            return None
    
    def _build_discoverer_config(self, provider_name: str, discovery_config: DiscoveryConfig) -> Dict[str, Any]:
        """Build discoverer configuration from provider config - NO HARDCODED VALUES"""
        # Get provider configuration
        provider_config = self.providers.get(provider_name)
        if not provider_config:
            return {}
        
        # Start with discoverer-specific config from YAML
        config = discovery_config.discoverer_config.copy()
        
        # Add provider configuration that discoverers typically need
        if provider_config.api_key_env:
            api_key = os.getenv(provider_config.api_key_env)
            if not api_key and provider_config.api_key_fallback_env:
                api_key = os.getenv(provider_config.api_key_fallback_env)
            if api_key:
                config["api_key"] = api_key
        
        if provider_config.api_base:
            config["api_base"] = provider_config.api_base
        
        # Add cache timeout
        config["cache_timeout"] = discovery_config.cache_timeout
        
        # Add any extra configuration that might be relevant for discovery
        if "discovery" in provider_config.extra:
            config.update(provider_config.extra["discovery"])
        
        return config
    
    async def _refresh_provider_models(self, provider_name: str, discovery_config: DiscoveryConfig) -> bool:
        """Refresh models for provider using discovery with environment controls"""
        # Check if discovery is allowed
        if not self._is_discovery_enabled(provider_name):
            logger.debug(f"Discovery disabled for {provider_name}")
            return False
        
        # Check cache first (unless force refresh is enabled)
        if not self._discovery_settings['force_refresh']:
            cache_key = provider_name
            cached_data = self._discovery_cache.get(cache_key)
            if cached_data:
                cache_age = time.time() - cached_data["timestamp"]
                if cache_age < discovery_config.cache_timeout:
                    logger.debug(f"Using cached discovery for {provider_name} (age: {cache_age:.1f}s)")
                    return True
        
        try:
            # Get discovery manager with timeout
            manager = await asyncio.wait_for(
                self._get_discovery_manager(provider_name, discovery_config),
                timeout=self._discovery_settings['timeout']
            )
            
            if not manager:
                return False
            
            # Discover models with timeout
            discovered_models = await asyncio.wait_for(
                manager.discover_models(),
                timeout=self._discovery_settings['timeout']
            )
            
            text_models = [m for m in discovered_models if hasattr(m, 'capabilities') and 
                          any(f.value == 'text' for f in m.capabilities)]
            
            if not text_models:
                logger.debug(f"No text models discovered for {provider_name}")
                return False
            
            # CRITICAL FIX: Store discovered models separately AND update provider
            discovered_names = {m.name for m in text_models}
            self._discovered_models[provider_name] = discovered_names
            
            # Store discovered capabilities
            discovered_capabilities = []
            for model in text_models:
                model_name = model.name
                
                # Create capabilities for discovered model (exact pattern)
                discovered_capabilities.append(ModelCapabilities(
                    pattern=f"^{re.escape(model_name)}$",
                    features=model.capabilities,
                    max_context_length=model.context_length,
                    max_output_tokens=model.max_output_tokens
                ))
                
                # Also create pattern for alternative form (:latest handling)
                if model_name.endswith(':latest'):
                    # Add pattern for base name too
                    alt_pattern = f"^{re.escape(model_name.replace(':latest', ''))}$"
                else:
                    # Add pattern for :latest version too
                    alt_pattern = f"^{re.escape(model_name)}:latest$"
                
                discovered_capabilities.append(ModelCapabilities(
                    pattern=alt_pattern,
                    features=model.capabilities,
                    max_context_length=model.context_length,
                    max_output_tokens=model.max_output_tokens
                ))
            
            self._discovered_capabilities[provider_name] = discovered_capabilities
            
            # ALSO update the provider configuration for backward compatibility
            provider = self.providers[provider_name]
            static_models = set(provider.models)
            
            # Create lookup sets for both forms to avoid duplicates
            static_models_normalized = set()
            for model in static_models:
                static_models_normalized.add(model)
                if model.endswith(':latest'):
                    static_models_normalized.add(model.replace(':latest', ''))
                else:
                    static_models_normalized.add(f"{model}:latest")
            
            # Merge models (static take precedence)
            new_model_names = list(provider.models)  # Keep static models
            new_capabilities = list(provider.model_capabilities)  # Keep static capabilities
            
            # Add new discovered models with :latest deduplication
            for model in text_models:
                model_name = model.name
                base_name = model_name.replace(':latest', '') if model_name.endswith(':latest') else model_name
                
                # Skip if we already have this model in any form
                if (model_name not in static_models_normalized and 
                    base_name not in static_models_normalized):
                    
                    new_model_names.append(model_name)
            
            # Add all discovered capabilities
            new_capabilities.extend(discovered_capabilities)
            
            # Update provider
            provider.models = new_model_names
            provider.model_capabilities = new_capabilities
            
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
            
        except asyncio.TimeoutError:
            logger.debug(f"Discovery timeout for {provider_name} after {self._discovery_settings['timeout']}s")
            return False
        except Exception as e:
            logger.debug(f"Discovery failed for {provider_name}: {e}")
            return False
    
    def _is_model_available(self, provider_name: str, model_name: str) -> bool:
        """Check if model is available (static OR discovered)"""
        if not model_name:
            return False
        
        provider = self.providers.get(provider_name)
        if not provider:
            return False
        
        # Check static models first
        resolved_model = provider.model_aliases.get(model_name, model_name)
        if resolved_model in provider.models:
            return True
        
        # Check :latest variants in static models
        if not model_name.endswith(':latest'):
            latest_variant = f"{model_name}:latest"
            resolved_latest = provider.model_aliases.get(latest_variant, latest_variant)
            if resolved_latest in provider.models:
                return True
        else:
            base_variant = model_name.replace(':latest', '')
            resolved_base = provider.model_aliases.get(base_variant, base_variant)
            if resolved_base in provider.models:
                return True
        
        # CRITICAL: Check discovered models
        discovered_models = self._discovered_models.get(provider_name, set())
        
        # Direct match
        if model_name in discovered_models:
            return True
        
        # Check :latest variants in discovered models
        if not model_name.endswith(':latest'):
            if f"{model_name}:latest" in discovered_models:
                return True
        else:
            base_name = model_name.replace(':latest', '')
            if base_name in discovered_models:
                return True
        
        return False
    
    def _ensure_model_available(self, provider_name: str, model_name: Optional[str]) -> Optional[str]:
        """
        FIXED: Ensure model is available, trigger discovery if enabled by environment.
        Returns resolved model name or None if not found.
        """
        if not model_name:
            return None
        
        # Check if auto-discovery is disabled
        if not self._discovery_settings['auto_discover'] or not self._is_discovery_enabled(provider_name):
            # Just do static lookup without discovery
            return self._static_model_lookup(provider_name, model_name)
        
        # Step 1: Check if model is already available (static or discovered)
        if self._is_model_available(provider_name, model_name):
            return model_name  # Model is available
        
        # Step 2: Model not found - check if discovery is enabled
        provider = self.providers.get(provider_name)
        if not provider:
            return None
        
        discovery_config = self._parse_discovery_config({"extra": provider.extra, "name": provider_name})
        if not discovery_config or not discovery_config.enabled:
            return None  # No discovery available
        
        # Step 3: Try discovery with environment controls
        try:
            import asyncio
            import threading
            import concurrent.futures
            
            # Check if we're already in an async context
            try:
                asyncio.get_running_loop()
                in_async_context = True
            except RuntimeError:
                in_async_context = False
            
            async def _discover_and_check():
                success = await self._refresh_provider_models(provider_name, discovery_config)
                if success:
                    # Re-check if model is now available
                    if self._is_model_available(provider_name, model_name):
                        logger.debug(f"Found {model_name} via discovery")
                        return model_name
                    
                    # Check :latest variants
                    if not model_name.endswith(':latest'):
                        latest_variant = f"{model_name}:latest"
                        if self._is_model_available(provider_name, latest_variant):
                            logger.debug(f"Found {model_name} as {latest_variant} via discovery")
                            return latest_variant
                    else:
                        base_variant = model_name.replace(':latest', '')
                        if self._is_model_available(provider_name, base_variant):
                            logger.debug(f"Found {model_name} as {base_variant} via discovery")
                            return base_variant
                
                return None
            
            discovery_timeout = self._discovery_settings['timeout']
            
            if in_async_context:
                # We're in an async context - run in thread pool to avoid blocking
                def run_discovery():
                    # Create new event loop in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(asyncio.wait_for(_discover_and_check(), timeout=discovery_timeout))
                    finally:
                        loop.close()
                
                # Use thread pool executor with timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_discovery)
                    try:
                        return future.result(timeout=discovery_timeout + 1)
                    except concurrent.futures.TimeoutError:
                        logger.debug(f"Discovery timeout for {provider_name}/{model_name}")
                        return None
            else:
                # No event loop - create one
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(asyncio.wait_for(_discover_and_check(), timeout=discovery_timeout))
                finally:
                    loop.close()
            
        except Exception as e:
            logger.debug(f"Discovery error for {provider_name}/{model_name}: {e}")
            return None
    
    def _static_model_lookup(self, provider_name: str, model_name: str) -> Optional[str]:
        """Static model lookup without discovery"""
        provider = self.providers.get(provider_name)
        if not provider:
            return None
        
        resolved_model = provider.model_aliases.get(model_name, model_name)
        if resolved_model in provider.models:
            return resolved_model
        
        # Try :latest variants
        if not model_name.endswith(':latest'):
            latest_variant = f"{model_name}:latest"
            resolved_latest = provider.model_aliases.get(latest_variant, latest_variant)
            if resolved_latest in provider.models:
                return resolved_latest
        else:
            base_variant = model_name.replace(':latest', '')
            resolved_base = provider.model_aliases.get(base_variant, base_variant)
            if resolved_base in provider.models:
                return resolved_base
        
        return None
    
    def get_discovery_settings(self) -> Dict[str, Any]:
        """Get current discovery settings (for debugging/status)"""
        return self._discovery_settings.copy()
    
    def get_discovered_models(self, provider_name: str) -> Set[str]:
        """Get discovered models for a provider"""
        return self._discovered_models.get(provider_name, set()).copy()
    
    def get_all_available_models(self, provider_name: str) -> Set[str]:
        """Get all available models for a provider (static + discovered)"""
        provider = self.providers.get(provider_name)
        if not provider:
            return set()
        
        # Start with static models
        all_models = set(provider.models)
        
        # Add discovered models
        discovered = self._discovered_models.get(provider_name, set())
        all_models.update(discovered)
        
        return all_models
    
    def reload(self):
        """Enhanced reload that clears discovery state and reloads settings"""
        # Clear discovery state
        self._discovery_managers.clear()
        self._discovery_cache.clear()
        self._discovered_models.clear()
        self._discovered_capabilities.clear()
        
        # Reload discovery settings from environment
        self._discovery_settings = self._load_discovery_settings()
        
        # Call parent reload if it exists
        if hasattr(super(), 'reload'):
            super().reload()


# Generic utility functions for discovery control
def disable_discovery_globally():
    """Disable discovery globally at runtime"""
    os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'false'


def enable_discovery_globally():
    """Enable discovery globally at runtime"""
    os.environ['CHUK_LLM_DISCOVERY_ENABLED'] = 'true'


def disable_provider_discovery(provider_name: str):
    """Disable discovery for a specific provider"""
    env_key = f'CHUK_LLM_{provider_name.upper()}_DISCOVERY'
    os.environ[env_key] = 'false'


def enable_provider_discovery(provider_name: str):
    """Enable discovery for a specific provider"""
    env_key = f'CHUK_LLM_{provider_name.upper()}_DISCOVERY'
    os.environ[env_key] = 'true'


def set_discovery_timeout(seconds: int):
    """Set discovery timeout at runtime"""
    os.environ['CHUK_LLM_DISCOVERY_TIMEOUT'] = str(seconds)