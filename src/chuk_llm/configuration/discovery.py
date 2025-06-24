# chuk_llm/configuration/discovery.py - Enhanced with Environment Controls
"""
Discovery integration for configuration manager with environment variable controls
"""

import asyncio
import logging
import os
import re
import time
from typing import Dict, List, Optional, Any

from .models import ProviderConfig, ModelCapabilities, Feature, DiscoveryConfig

logger = logging.getLogger(__name__)


class ConfigDiscoveryMixin:
    """
    Mixin that adds discovery capabilities to configuration manager.
    Now supports environment variable controls for fine-grained discovery management.
    """
    
    def __init__(self):
        # Discovery state (internal)
        self._discovery_managers: Dict[str, Any] = {}
        self._discovery_cache: Dict[str, Dict[str, Any]] = {}  # provider -> {models, timestamp}
        
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
            
            # Provider-specific controls
            'ollama_enabled': self._env_bool('CHUK_LLM_OLLAMA_DISCOVERY', True),
            'openai_enabled': self._env_bool('CHUK_LLM_OPENAI_DISCOVERY', True),
            'anthropic_enabled': self._env_bool('CHUK_LLM_ANTHROPIC_DISCOVERY', True),
            'gemini_enabled': self._env_bool('CHUK_LLM_GEMINI_DISCOVERY', True),
            'groq_enabled': self._env_bool('CHUK_LLM_GROQ_DISCOVERY', True),
            'mistral_enabled': self._env_bool('CHUK_LLM_MISTRAL_DISCOVERY', True),
            'deepseek_enabled': self._env_bool('CHUK_LLM_DEEPSEEK_DISCOVERY', True),
            'perplexity_enabled': self._env_bool('CHUK_LLM_PERPLEXITY_DISCOVERY', True),
            'watsonx_enabled': self._env_bool('CHUK_LLM_WATSONX_DISCOVERY', True),
            
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
        
        # Provider-specific check
        if provider_name:
            provider_key = f'{provider_name}_enabled'
            return self._discovery_settings.get(provider_key, True)
        
        return True
    
    def _parse_discovery_config(self, provider_data: Dict[str, Any]) -> Optional[DiscoveryConfig]:
        """Parse discovery configuration from provider YAML with environment overrides"""
        discovery_data = provider_data.get("extra", {}).get("dynamic_discovery")
        if not discovery_data:
            return None
        
        # FIXED: Check if explicitly disabled in config first
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
            
            # Update provider configuration seamlessly with :latest handling
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
            new_model_names = []
            new_capabilities = []
            
            # Keep all static models
            new_model_names.extend(provider.models)
            
            # Add new discovered models with :latest deduplication
            for model in text_models:
                model_name = model.name
                base_name = model_name.replace(':latest', '') if model_name.endswith(':latest') else model_name
                
                # Skip if we already have this model in any form
                if (model_name not in static_models_normalized and 
                    base_name not in static_models_normalized):
                    
                    new_model_names.append(model_name)
                    
                    # Create capabilities for discovered model (exact pattern)
                    new_capabilities.append(ModelCapabilities(
                        pattern=f"^{re.escape(model_name)}$",
                        features=model.capabilities,
                        max_context_length=model.context_length,
                        max_output_tokens=model.max_output_tokens
                    ))
                    
                    # Also create pattern for alternative form (:latest handling)
                    if model_name.endswith(':latest'):
                        # Add pattern for base name too
                        alt_pattern = f"^{re.escape(base_name)}$"
                    else:
                        # Add pattern for :latest version too
                        alt_pattern = f"^{re.escape(model_name)}:latest$"
                    
                    new_capabilities.append(ModelCapabilities(
                        pattern=alt_pattern,
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
            
        except asyncio.TimeoutError:
            logger.debug(f"Discovery timeout for {provider_name} after {self._discovery_settings['timeout']}s")
            return False
        except Exception as e:
            logger.debug(f"Discovery failed for {provider_name}: {e}")
            return False
    
    def _ensure_model_available(self, provider_name: str, model_name: Optional[str]) -> Optional[str]:
        """
        Ensure model is available, trigger discovery if enabled by environment.
        Returns resolved model name or None if not found.
        """
        if not model_name:
            return None
        
        # Check if auto-discovery is disabled
        if not self._discovery_settings['auto_discover'] or not self._is_discovery_enabled(provider_name):
            # Just do static lookup without discovery
            provider = self.providers[provider_name]
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
        
        # Original discovery logic continues here...
        provider = self.providers[provider_name]
        
        # Step 1: Check exact match first (including aliases)
        resolved_model = provider.model_aliases.get(model_name, model_name)
        if resolved_model in provider.models:
            return resolved_model
        
        # Step 2: Try :latest variants before discovery
        latest_variant = None
        base_variant = None
        
        if not model_name.endswith(':latest'):
            # Try adding :latest suffix
            latest_variant = f"{model_name}:latest"
            resolved_latest = provider.model_aliases.get(latest_variant, latest_variant)
            if resolved_latest in provider.models:
                logger.debug(f"Resolved {model_name} to existing model {resolved_latest}")
                return resolved_latest
        else:
            # Try removing :latest suffix
            base_variant = model_name.replace(':latest', '')
            resolved_base = provider.model_aliases.get(base_variant, base_variant)
            if resolved_base in provider.models:
                logger.debug(f"Resolved {model_name} to existing model {resolved_base}")
                return resolved_base
        
        # Step 3: Model not found in static list - check if discovery is enabled
        discovery_config = self._parse_discovery_config({"extra": provider.extra, "name": provider_name})
        if not discovery_config or not discovery_config.enabled:
            return None  # No discovery available
        
        # Step 4: Try discovery with environment controls
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
                    # Re-check with both forms after discovery
                    updated_provider = self.providers[provider_name]
                    
                    # Check exact match
                    resolved_model = updated_provider.model_aliases.get(model_name, model_name)
                    if resolved_model in updated_provider.models:
                        logger.debug(f"Found {model_name} via discovery")
                        return resolved_model
                    
                    # Check :latest variant
                    if latest_variant:
                        resolved_latest = updated_provider.model_aliases.get(latest_variant, latest_variant)
                        if resolved_latest in updated_provider.models:
                            logger.debug(f"Found {model_name} as {resolved_latest} via discovery")
                            return resolved_latest
                    
                    # Check base variant
                    if base_variant:
                        resolved_base = updated_provider.model_aliases.get(base_variant, base_variant)
                        if resolved_base in updated_provider.models:
                            logger.debug(f"Found {model_name} as {resolved_base} via discovery")
                            return resolved_base
                
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
    
    def get_discovery_settings(self) -> Dict[str, Any]:
        """Get current discovery settings (for debugging/status)"""
        return self._discovery_settings.copy()
    
    def reload(self):
        """Enhanced reload that clears discovery state and reloads settings"""
        # Clear discovery state
        self._discovery_managers.clear()
        self._discovery_cache.clear()
        
        # Reload discovery settings from environment
        self._discovery_settings = self._load_discovery_settings()
        
        # Call parent reload if it exists
        if hasattr(super(), 'reload'):
            super().reload()


# Additional utility functions for discovery control
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