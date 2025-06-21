"""
Clean Unified Configuration System with Transparent Discovery
===========================================================

Single source of truth for all provider configuration, capabilities, and validation.
Everything comes from one YAML file with intelligent merging and inheritance.
Discovery is completely transparent - no API changes needed.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
    _dotenv_available = True
except ImportError:
    _dotenv_available = False

# Modern package resource handling
try:
    from importlib.resources import files
    _importlib_resources_available = True
except ImportError:
    try:
        from importlib_resources import files
        _importlib_resources_available = True
    except ImportError:
        _importlib_resources_available = False

# Fallback package resource handling
try:
    import pkg_resources
    _pkg_resources_available = True
except ImportError:
    _pkg_resources_available = False

from .models import Feature, ModelCapabilities, ProviderConfig
from .discovery import ConfigDiscoveryMixin
from .validator import ConfigValidator

logger = logging.getLogger(__name__)


class UnifiedConfigManager(ConfigDiscoveryMixin):
    """
    Unified configuration manager with transparent dynamic discovery.
    
    Discovery is completely seamless - existing APIs work unchanged.
    All configuration comes from YAML, nothing hardcoded.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()  # Initialize discovery mixin
        
        self.config_path = config_path
        self.providers: Dict[str, ProviderConfig] = {}
        self.global_aliases: Dict[str, str] = {}
        self.global_settings: Dict[str, Any] = {}
        self._loaded = False
        
        # Load environment variables first
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        if not _dotenv_available:
            logger.debug("python-dotenv not available, skipping .env file loading")
            return
        
        # Look for .env files in common locations
        env_candidates = [
            ".env",
            ".env.local", 
            os.path.expanduser("~/.chuk_llm/.env"),
            Path(__file__).parent.parent.parent / ".env",  # Project root
        ]
        
        for env_file in env_candidates:
            env_path = Path(env_file).expanduser().resolve()
            if env_path.exists():
                logger.info(f"Loading environment from {env_path}")
                load_dotenv(env_path, override=False)  # Don't override existing env vars
                break
        else:
            logger.debug("No .env file found in standard locations")
    
    def _find_config_files(self) -> tuple[Optional[Path], Optional[Path]]:
        """Find configuration files - returns (user_config, package_config)"""
        user_config = None
        package_config = None
        
        # 1. Look for user configuration files (these override/replace)
        if self.config_path:
            path = Path(self.config_path)
            if path.exists():
                user_config = path
        
        if not user_config:
            user_candidates = [
                # Environment variable with location
                os.getenv("CHUK_LLM_CONFIG"),
                
                # Working directory of consuming project
                "chuk_llm.yaml",      # REPLACES package config completely
                "providers.yaml",     # EXTENDS package config (inherits + overrides)
                "llm_config.yaml",    # EXTENDS package config
                "config/chuk_llm.yaml",
                Path.home() / ".chuk_llm" / "config.yaml",
            ]
            
            for candidate in user_candidates:
                if candidate:
                    path = Path(candidate).expanduser().resolve()
                    if path.exists():
                        logger.info(f"Found user config file: {path}")
                        user_config = path
                        break
        
        # 2. Always try to get package config as baseline
        package_config = self._get_package_config_path()
        if package_config:
            package_config = Path(package_config)
            
        return user_config, package_config

    def _load_yaml_files(self) -> Dict:
        """Load YAML configuration with inheritance logic"""
        if not yaml:
            logger.warning("PyYAML not available, using built-in defaults only")
            return {}
        
        user_config_file, package_config_file = self._find_config_files()
        
        config = {}
        
        # 1. Start with package config as baseline (if available)
        if package_config_file and package_config_file.exists():
            try:
                package_config = self._load_single_yaml(package_config_file)
                if package_config:
                    config.update(package_config)
                    logger.info(f"Loaded package config from {package_config_file}")
            except Exception as e:
                logger.error(f"Failed to load package config from {package_config_file}: {e}")
        
        # 2. Handle user config based on filename
        if user_config_file:
            try:
                user_config = self._load_single_yaml(user_config_file)
                if user_config:
                    filename = user_config_file.name
                    
                    if filename == "chuk_llm.yaml":
                        # COMPLETE REPLACEMENT - ignore package config
                        logger.info(f"chuk_llm.yaml found - replacing package config completely")
                        config = user_config
                    else:
                        # INHERITANCE MODE - merge with package config
                        logger.info(f"{filename} found - extending package config")
                        config = self._merge_configs(config, user_config)
                    
                    logger.info(f"Loaded user config from {user_config_file}")
            except Exception as e:
                logger.error(f"Failed to load user config from {user_config_file}: {e}")
        
        if not config:
            logger.info("No configuration files found, using built-in defaults")
        
        return config

    def _load_single_yaml(self, config_file: Path) -> Dict:
        """Load a single YAML file"""
        # Handle package resource files specially
        if self._is_package_resource_path(config_file):
            return self._load_package_yaml()
        
        # Regular file loading
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _merge_configs(self, base_config: Dict, user_config: Dict) -> Dict:
        """Merge user config into base config with intelligent merging"""
        merged = base_config.copy()
        
        for key, value in user_config.items():
            if key.startswith("__"):
                # Global sections - merge intelligently
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    # Merge global sections (like __global__, __global_aliases__)
                    merged[key].update(value)
                else:
                    # Replace if not both dicts
                    merged[key] = value
            else:
                # Provider sections
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    # Merge provider config - user overrides base
                    provider_config = merged[key].copy()
                    
                    # Handle special cases for lists that should be extended vs replaced
                    for sub_key, sub_value in value.items():
                        if sub_key == "models" and isinstance(sub_value, list) and isinstance(provider_config.get(sub_key), list):
                            # EXTEND models list (unique values only)
                            existing_models = set(provider_config[sub_key])
                            new_models = [m for m in sub_value if m not in existing_models]
                            provider_config[sub_key].extend(new_models)
                        elif sub_key == "model_aliases" and isinstance(sub_value, dict) and isinstance(provider_config.get(sub_key), dict):
                            # MERGE model aliases
                            provider_config[sub_key].update(sub_value)
                        elif sub_key == "features" and isinstance(sub_value, list) and isinstance(provider_config.get(sub_key), list):
                            # EXTEND features list (unique values only)
                            existing_features = set(provider_config[sub_key])
                            new_features = [f for f in sub_value if f not in existing_features]
                            provider_config[sub_key].extend(new_features)
                        elif sub_key == "rate_limits" and isinstance(sub_value, dict) and isinstance(provider_config.get(sub_key), dict):
                            # MERGE rate limits
                            provider_config[sub_key].update(sub_value)
                        elif sub_key == "extra" and isinstance(sub_value, dict) and isinstance(provider_config.get(sub_key), dict):
                            # DEEP MERGE extra config
                            provider_config[sub_key] = self._deep_merge_dict(provider_config[sub_key], sub_value)
                        else:
                            # REPLACE for other fields
                            provider_config[sub_key] = sub_value
                    
                    merged[key] = provider_config
                else:
                    # Replace entire provider config if not both dicts
                    merged[key] = value
        
        return merged

    def _deep_merge_dict(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result

    def _get_package_config_path(self) -> Optional[str]:
        """Get the path to the packaged chuk_llm.yaml file"""
        # Try modern importlib.resources first
        if _importlib_resources_available:
            try:
                package_files = files('chuk_llm')
                config_file = package_files / 'chuk_llm.yaml'
                if config_file.is_file():
                    # Return a temporary path for importlib.resources
                    return str(config_file)
            except Exception as e:
                logger.debug(f"importlib.resources method failed: {e}")
        
        # Fallback to pkg_resources
        if _pkg_resources_available:
            try:
                if pkg_resources.resource_exists('chuk_llm', 'chuk_llm.yaml'):
                    return pkg_resources.resource_filename('chuk_llm', 'chuk_llm.yaml')
            except Exception as e:
                logger.debug(f"pkg_resources method failed: {e}")
        
        return None

    def _is_package_resource_path(self, config_file: Path) -> bool:
        """Check if this is a package resource path that needs special handling"""
        config_str = str(config_file)
        # Check for importlib.resources paths (they often contain special markers)
        return (not config_file.exists() and 
                ('chuk_llm' in config_str or 'importlib' in config_str))

    def _load_package_yaml(self) -> Dict:
        """Load YAML from package resources"""
        # Try modern importlib.resources first
        if _importlib_resources_available:
            try:
                package_files = files('chuk_llm')
                config_file = package_files / 'chuk_llm.yaml'
                if config_file.is_file():
                    content = config_file.read_text(encoding='utf-8')
                    config = yaml.safe_load(content) or {}
                    logger.info("Loaded configuration from package resources (importlib)")
                    return config
            except Exception as e:
                logger.debug(f"importlib.resources loading failed: {e}")
        
        # Fallback to pkg_resources
        if _pkg_resources_available:
            try:
                content = pkg_resources.resource_string('chuk_llm', 'chuk_llm.yaml').decode('utf-8')
                config = yaml.safe_load(content) or {}
                logger.info("Loaded configuration from package resources (pkg_resources)")
                return config
            except Exception as e:
                logger.debug(f"pkg_resources loading failed: {e}")
        
        return {}

    def _parse_features(self, features_data: Any) -> Set[Feature]:
        """Parse features from YAML data"""
        if not features_data:
            return set()
        
        if isinstance(features_data, str):
            features_data = [features_data]
        
        result = set()
        for feature in features_data:
            if isinstance(feature, Feature):
                result.add(feature)
            else:
                result.add(Feature.from_string(str(feature)))
        
        return result
    
    def _parse_model_capabilities(self, models_data: List[Dict]) -> List[ModelCapabilities]:
        """Parse model-specific capabilities"""
        if not models_data:
            return []
        
        capabilities = []
        for model_data in models_data:
            cap = ModelCapabilities(
                pattern=model_data.get("pattern", ".*"),
                features=self._parse_features(model_data.get("features", [])),
                max_context_length=model_data.get("max_context_length"),
                max_output_tokens=model_data.get("max_output_tokens")
            )
            capabilities.append(cap)
        
        return capabilities
    
    def _load_yaml(self) -> Dict:
        """Load YAML configuration with inheritance support"""
        return self._load_yaml_files()
    
    def _process_config(self, config: Dict):
        """Process YAML configuration and merge with defaults"""
        # Global settings
        self.global_settings.update(config.get("__global__", {}))
        self.global_aliases.update(config.get("__global_aliases__", {}))
        
        # Process providers
        for name, data in config.items():
            if name.startswith("__"):
                continue
            
            # Start with existing provider or create new
            if name in self.providers:
                provider = self.providers[name]
                logger.info(f"Merging configuration for existing provider: {name}")
            else:
                provider = ProviderConfig(name=name)
                self.providers[name] = provider
            
            # Update basic fields
            if "client_class" in data:
                provider.client_class = data["client_class"]
            if "api_key_env" in data:
                provider.api_key_env = data["api_key_env"]
            if "api_key_fallback_env" in data:
                provider.api_key_fallback_env = data["api_key_fallback_env"]
            if "api_base" in data:
                provider.api_base = data["api_base"]
            if "default_model" in data:
                provider.default_model = data["default_model"]
            
            # Update collections
            if "models" in data:
                provider.models = data["models"]
            if "model_aliases" in data:
                provider.model_aliases.update(data["model_aliases"])
            
            # Update capabilities
            if "features" in data:
                provider.features = self._parse_features(data["features"])
            if "max_context_length" in data:
                provider.max_context_length = data["max_context_length"]
            if "max_output_tokens" in data:
                provider.max_output_tokens = data["max_output_tokens"]
            if "rate_limits" in data:
                provider.rate_limits.update(data["rate_limits"])
            if "model_capabilities" in data:
                provider.model_capabilities = self._parse_model_capabilities(data["model_capabilities"])
            
            # Inheritance
            if "inherits" in data:
                provider.inherits = data["inherits"]
            
            # CRITICAL FIX: Process extra fields INSIDE the provider loop
            known_fields = {"client_class", "api_key_env", "api_key_fallback_env",
                           "api_base", "default_model", "models", "model_aliases",
                           "features", "max_context_length", "max_output_tokens",
                           "rate_limits", "model_capabilities", "inherits"}

            # Extract extra fields for THIS provider
            extra_fields = {k: v for k, v in data.items() if k not in known_fields}

            # CRITICAL FIX: Deep merge extra fields for THIS provider
            for key, value in extra_fields.items():
                if isinstance(value, dict) and key in provider.extra and isinstance(provider.extra[key], dict):
                    # Deep merge dictionaries (like dynamic_discovery)
                    provider.extra[key].update(value)
                else:
                    # Replace for non-dict values
                    provider.extra[key] = value
            
            # CRITICAL FIX: Handle double nesting issue
            if "extra" in provider.extra and isinstance(provider.extra["extra"], dict):
                # Flatten double-nested extra fields
                nested_extra = provider.extra["extra"]
                del provider.extra["extra"]
                provider.extra.update(nested_extra)
                logger.debug(f"Fixed double nesting for provider {name}")
            
            # Debug logging for discovery config
            if "dynamic_discovery" in extra_fields:
                logger.debug(f"Added discovery config to {name}: {extra_fields['dynamic_discovery']}")
            elif "dynamic_discovery" in provider.extra:
                logger.debug(f"Discovery config available for {name}: enabled={provider.extra['dynamic_discovery'].get('enabled')}")
    
    def _resolve_inheritance(self):
        """Resolve provider inheritance - inherit config but NOT models/aliases"""
        for _ in range(10):  # Max 10 levels of inheritance
            changes = False
            
            for provider in self.providers.values():
                if provider.inherits and provider.inherits in self.providers:
                    parent = self.providers[provider.inherits]
                    
                    if not parent.inherits:  # Parent is resolved
                        # Inherit TECHNICAL fields if not set
                        if not provider.client_class:
                            provider.client_class = parent.client_class
                        if not provider.api_key_env:
                            provider.api_key_env = parent.api_key_env
                        if not provider.api_base:
                            provider.api_base = parent.api_base
                        
                        # Inherit baseline features (this is good)
                        provider.features.update(parent.features)
                        
                        # Inherit capabilities (this is good)
                        if not provider.max_context_length:
                            provider.max_context_length = parent.max_context_length
                        if not provider.max_output_tokens:
                            provider.max_output_tokens = parent.max_output_tokens
                        
                        # Inherit rate limits (this is good)
                        parent_limits = parent.rate_limits.copy()
                        parent_limits.update(provider.rate_limits)
                        provider.rate_limits = parent_limits
                        
                        # Inherit model capabilities (this is good)
                        parent_model_caps = parent.model_capabilities.copy()
                        parent_model_caps.extend(provider.model_capabilities)
                        provider.model_capabilities = parent_model_caps
                        
                        # Inherit extra fields (this is good)
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
        """Get provider configuration (with transparent discovery)"""
        self.load()
        
        if name not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(f"Unknown provider: {name}. Available: {available}")
        
        return self.providers[name]
    
    def get_all_providers(self) -> List[str]:
        """Get all provider names"""
        self.load()
        return list(self.providers.keys())
    
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
    
    def supports_feature(self, provider_name: str, feature: Union[str, Feature], 
                        model: Optional[str] = None) -> bool:
        """Check if provider/model supports feature"""
        provider = self.get_provider(provider_name)
        return provider.supports_feature(feature, model)
    
    def get_global_aliases(self) -> Dict[str, str]:
        """Get global aliases configuration"""
        self.load()
        return self.global_aliases.copy()

    def get_global_settings(self) -> Dict[str, Any]:
        """Get global settings configuration"""
        self.load()
        return self.global_settings.copy()

    def set_global_setting(self, key: str, value: Any):
        """Set a global setting"""
        self.load()
        self.global_settings[key] = value

    def add_global_alias(self, alias: str, target: str):
        """Add a global alias"""
        self.load()
        self.global_aliases[alias] = target
    
    def reload(self):
        """Reload configuration"""
        self._loaded = False
        self.providers.clear()
        self.global_aliases.clear()
        self.global_settings.clear()
        super().reload()  # Clear discovery state
        self.load()


# ──────────────────────────── Capability Checker ─────────────────────────────
class CapabilityChecker:
    """Query helpers for provider capabilities"""
    
    @staticmethod
    def can_handle_request(
        provider: str,
        model: Optional[str] = None,
        *,
        has_tools: bool = False,
        has_vision: bool = False,
        needs_streaming: bool = False,
        needs_json: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Check if provider/model can handle request"""
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(provider)
            
            problems = []
            if has_tools and not provider_config.supports_feature(Feature.TOOLS, model):
                problems.append("tools not supported")
            if has_vision and not provider_config.supports_feature(Feature.VISION, model):
                problems.append("vision not supported")
            if needs_streaming and not provider_config.supports_feature(Feature.STREAMING, model):
                problems.append("streaming not supported")
            if needs_json and not provider_config.supports_feature(Feature.JSON_MODE, model):
                problems.append("JSON mode not supported")
            
            return len(problems) == 0, problems
            
        except Exception as exc:
            return False, [f"Provider not found: {exc}"]
    
    @staticmethod
    def get_best_provider_for_features(
        required_features: Set[Feature],
        model_name: Optional[str] = None,
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Find best provider that supports required features"""
        exclude = exclude or set()
        config_manager = get_config()
        
        candidates = []
        for provider_name in config_manager.get_all_providers():
            if provider_name in exclude:
                continue
            
            provider = config_manager.get_provider(provider_name)
            model_caps = provider.get_model_capabilities(model_name)
            
            if required_features.issubset(model_caps.features):
                rate_limit = provider.get_rate_limit() or 0
                candidates.append((provider_name, rate_limit))
        
        return max(candidates, key=lambda x: x[1])[0] if candidates else None
    
    @staticmethod
    def get_model_info(provider: str, model: str) -> Dict[str, Any]:
        """Get comprehensive model information"""
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(provider)
            model_caps = provider_config.get_model_capabilities(model)
            
            return {
                "provider": provider,
                "model": model,
                "features": [f.value for f in model_caps.features],
                "max_context_length": model_caps.max_context_length,
                "max_output_tokens": model_caps.max_output_tokens,
                "supports_streaming": Feature.STREAMING in model_caps.features,
                "supports_tools": Feature.TOOLS in model_caps.features,
                "supports_vision": Feature.VISION in model_caps.features,
                "supports_json_mode": Feature.JSON_MODE in model_caps.features,
                "rate_limits": provider_config.rate_limits
            }
        except Exception as exc:
            return {"error": f"Failed to get model info: {exc}"}


# ──────────────────────────── Global Instance ─────────────────────────────
_unified_config = UnifiedConfigManager()


def get_config() -> UnifiedConfigManager:
    """Get global configuration manager"""
    return _unified_config


def reset_config():
    """Reset configuration"""
    global _unified_config
    _unified_config = UnifiedConfigManager()


def reset_unified_config():
    """Reset unified configuration (alias for reset_config)"""
    reset_config()


# Clean aliases
ConfigManager = UnifiedConfigManager


# Export clean API
__all__ = [
    "Feature", 
    "ModelCapabilities", 
    "ProviderConfig", 
    "UnifiedConfigManager", 
    "ConfigValidator", 
    "CapabilityChecker",
    "get_config",
    "reset_config",
    "reset_unified_config",
    "ConfigManager"
]