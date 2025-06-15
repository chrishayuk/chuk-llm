# chuk_llm/configuration/config.py
"""
Clean Dynamic Configuration System
=================================

Everything comes from YAML. Zero hardcoding. Super clean.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Complete provider configuration from YAML"""
    name: str
    client_class: str = ""
    api_key_env: Optional[str] = None
    api_key_fallback_env: Optional[str] = None
    api_base: Optional[str] = None
    default_model: str = ""
    models: List[str] = field(default_factory=list)
    model_aliases: Dict[str, str] = field(default_factory=dict)
    features: Set[str] = field(default_factory=set)
    inherits: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Clean dynamic configuration manager - everything from YAML"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.providers: Dict[str, ProviderConfig] = {}
        self.global_aliases: Dict[str, str] = {}
        self.global_settings: Dict[str, Any] = {}
        self._loaded = False
    
    def _find_config_file(self) -> Optional[Path]:
        """Find YAML configuration file"""
        if self.config_path:
            path = Path(self.config_path)
            if path.exists():
                return path
        
        candidates = [
            # Environment variable
            os.getenv("CHUK_LLM_CONFIG"),
            # Current directory
            "providers.yaml",
            "chuk_llm.yaml",
            # Config directory
            "config/providers.yaml",
            # Package root
            Path(__file__).parent.parent / "providers.yaml",
            # User config
            Path.home() / ".chuk_llm" / "providers.yaml",
        ]
        
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser().resolve()
                if path.exists():
                    return path
        
        return None
    
    def _load_yaml(self) -> Dict:
        """Load YAML configuration"""
        if not yaml:
            raise RuntimeError("PyYAML required for configuration")
        
        config_file = self._find_config_file()
        if not config_file:
            raise RuntimeError("No configuration file found. Create providers.yaml")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        logger.info(f"Loaded configuration from {config_file}")
        return config
    
    def _process_config(self, config: Dict):
        """Process YAML configuration"""
        # Global settings
        self.global_settings = config.get("__global__", {})
        
        # Global aliases
        self.global_aliases = config.get("__global_aliases__", {})
        
        # Process providers
        for name, data in config.items():
            if name.startswith("__"):
                continue
            
            provider = ProviderConfig(
                name=name,
                client_class=data.get("client", ""),
                api_key_env=data.get("api_key_env"),
                api_key_fallback_env=data.get("api_key_fallback_env"),
                api_base=data.get("api_base"),
                default_model=data.get("default_model", ""),
                models=data.get("models", []),
                model_aliases=data.get("model_aliases", {}),
                features=set(data.get("features", [])),
                inherits=data.get("inherits"),
                extra={k: v for k, v in data.items() 
                      if k not in {"client", "api_key_env", "api_key_fallback_env", 
                                  "api_base", "default_model", "models", 
                                  "model_aliases", "features", "inherits"}}
            )
            
            self.providers[name] = provider
    
    def _resolve_inheritance(self):
        """Resolve provider inheritance"""
        for _ in range(10):  # Max 10 levels of inheritance
            changes = False
            
            for provider in self.providers.values():
                if provider.inherits and provider.inherits in self.providers:
                    parent = self.providers[provider.inherits]
                    
                    if not parent.inherits:  # Parent is resolved
                        # Inherit fields if not set
                        if not provider.client_class:
                            provider.client_class = parent.client_class
                        if not provider.api_key_env:
                            provider.api_key_env = parent.api_key_env
                        if not provider.api_base:
                            provider.api_base = parent.api_base
                        if not provider.default_model:
                            provider.default_model = parent.default_model
                        
                        # Merge collections
                        if not provider.models:
                            provider.models = parent.models.copy()
                        
                        parent_aliases = parent.model_aliases.copy()
                        parent_aliases.update(provider.model_aliases)
                        provider.model_aliases = parent_aliases
                        
                        provider.features.update(parent.features)
                        
                        parent_extra = parent.extra.copy()
                        parent_extra.update(provider.extra)
                        provider.extra = parent_extra
                        
                        provider.inherits = None  # Mark as resolved
                        changes = True
            
            if not changes:
                break
    
    def load(self):
        """Load configuration"""
        if self._loaded:
            return
        
        config = self._load_yaml()
        self._process_config(config)
        self._resolve_inheritance()
        self._loaded = True
    
    def get_provider(self, name: str) -> ProviderConfig:
        """Get provider configuration"""
        self.load()
        if name not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(f"Unknown provider: {name}. Available: {available}")
        return self.providers[name]
    
    def get_all_providers(self) -> List[str]:
        """Get all provider names"""
        self.load()
        return list(self.providers.keys())
    
    def get_global_aliases(self) -> Dict[str, str]:
        """Get global aliases"""
        self.load()
        return self.global_aliases.copy()
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global settings"""
        self.load()
        return self.global_settings.copy()
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for provider"""
        provider = self.get_provider(provider_name)
        
        if provider.api_key_env:
            key = os.getenv(provider.api_key_env)
            if key:
                return key
        
        if provider.api_key_fallback_env:
            return os.getenv(provider.api_key_fallback_env)
        
        return None
    
    def supports_feature(self, provider_name: str, feature: str) -> bool:
        """Check if provider supports feature"""
        provider = self.get_provider(provider_name)
        return feature in provider.features
    
    def reload(self):
        """Reload configuration"""
        self._loaded = False
        self.providers.clear()
        self.global_aliases.clear()
        self.global_settings.clear()
        self.load()


# Global instance
_config = ConfigManager()


def get_config() -> ConfigManager:
    """Get global configuration manager"""
    return _config


def reset_config():
    """Reset configuration"""
    global _config
    _config = ConfigManager()


# Export all public functions
__all__ = ["ConfigManager", "get_config", "reset_config"]