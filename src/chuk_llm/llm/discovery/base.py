# chuk_llm/llm/discovery/base.py
"""
Base classes and factory for model discovery - Clean Version
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

from chuk_llm.configuration import Feature

log = logging.getLogger(__name__)


@dataclass
class DiscoveredModel:
    """Universal model information from discovery"""
    name: str
    provider: str
    
    # Basic metadata
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    version: Optional[str] = None
    
    # Provider-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Inferred properties
    family: str = "unknown"
    capabilities: Set[Feature] = field(default_factory=set)
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    parameters: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "provider": self.provider,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "version": self.version,
            "family": self.family,
            "capabilities": [f.value for f in self.capabilities],
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "parameters": self.parameters,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveredModel':
        """Create from dictionary"""
        capabilities = set()
        if data.get("capabilities"):
            capabilities = {Feature.from_string(f) for f in data["capabilities"]}
        
        return cls(
            name=data.get("name", "unknown"),
            provider=data.get("provider", "unknown"),
            size_bytes=data.get("size_bytes"),
            created_at=data.get("created_at"),
            modified_at=data.get("modified_at"),
            version=data.get("version"),
            metadata=data.get("metadata", {}),
            family=data.get("family", "unknown"),
            capabilities=capabilities,
            context_length=data.get("context_length"),
            max_output_tokens=data.get("max_output_tokens"),
            parameters=data.get("parameters")
        )


class BaseModelDiscoverer(ABC):
    """Base class for provider-specific model discoverers"""
    
    def __init__(self, provider_name: str, **config):
        self.provider_name = provider_name
        self.config = config
        self._discovery_cache = {}
        self._cache_timeout = config.get('cache_timeout', 300)
        self._last_discovery = None
    
    @abstractmethod
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover available models and return raw model data"""
        pass
    
    async def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metadata for a specific model (optional)"""
        return None
    
    def normalize_model_data(self, raw_model: Dict[str, Any]) -> DiscoveredModel:
        """Convert raw model data to DiscoveredModel"""
        return DiscoveredModel(
            name=raw_model.get("name", "unknown"),
            provider=self.provider_name,
            size_bytes=raw_model.get("size"),
            created_at=raw_model.get("created_at"),
            modified_at=raw_model.get("modified_at"),
            version=raw_model.get("version"),
            metadata=raw_model
        )
    
    async def discover_with_cache(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Discover models with caching support"""
        cache_key = f"{self.provider_name}_models"
        current_time = time.time()
        
        # Check cache
        if not force_refresh and cache_key in self._discovery_cache:
            cached_data, cache_time = self._discovery_cache[cache_key]
            if current_time - cache_time < self._cache_timeout:
                log.debug(f"Using cached discovery data for {self.provider_name}")
                return cached_data
        
        # Discover fresh data
        try:
            models = await self.discover_models()
            self._discovery_cache[cache_key] = (models, current_time)
            self._last_discovery = current_time
            return models
        except Exception as e:
            log.error(f"Discovery failed for {self.provider_name}: {e}")
            # Return cached data if available, even if stale
            if cache_key in self._discovery_cache:
                log.warning(f"Using stale cache data for {self.provider_name}")
                return self._discovery_cache[cache_key][0]
            return []
    
    def get_discovery_info(self) -> Dict[str, Any]:
        """Get information about the discoverer"""
        return {
            "provider": self.provider_name,
            "cache_timeout": self._cache_timeout,
            "last_discovery": self._last_discovery,
            "cached_models": len(self._discovery_cache.get(f"{self.provider_name}_models", ([], 0))[0]),
            "config": {k: v for k, v in self.config.items() if not k.startswith('_')}
        }


class DiscovererFactory:
    """Factory for creating provider-specific discoverers"""
    
    _discoverers = {}
    _imported = False
    
    @classmethod
    def register_discoverer(cls, provider_name: str, discoverer_class: type):
        """Register a discoverer for a provider"""
        cls._discoverers[provider_name] = discoverer_class
        log.debug(f"Registered discoverer for {provider_name}")
    
    @classmethod
    def create_discoverer(cls, provider_name: str, **config) -> BaseModelDiscoverer:
        """Create a discoverer for the given provider"""
        # Ensure discoverers are imported and registered
        cls._auto_import_discoverers()
        
        discoverer_class = cls._discoverers.get(provider_name)
        if not discoverer_class:
            available = list(cls._discoverers.keys())
            raise ValueError(f"No discoverer available for provider: {provider_name}. Available: {available}")
        
        return discoverer_class(provider_name=provider_name, **config)
    
    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """List providers that support discovery"""
        cls._auto_import_discoverers()
        return list(cls._discoverers.keys())
    
    @classmethod
    def _auto_import_discoverers(cls):
        """Auto-import discoverer modules to register them"""
        if cls._imported:
            return  # Already imported
        
        try:
            # Import all discoverer modules to trigger registration
            from . import ollama_discoverer
            from . import openai_discoverer
            from . import azure_openai_discoverer
            from . import general_discoverers
            cls._imported = True
            log.debug(f"Auto-imported discoverers: {list(cls._discoverers.keys())}")
        except ImportError as e:
            log.warning(f"Failed to import some discoverers: {e}")
            cls._imported = True  # Don't keep trying