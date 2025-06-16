# chuk_llm/configuration/unified_config.py
"""
Unified Configuration System for ChukLLM
========================================

Single source of truth for all provider configuration, capabilities, and validation.
Everything comes from one YAML file with intelligent merging and inheritance.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
    _dotenv_available = True
except ImportError:
    _dotenv_available = False

logger = logging.getLogger(__name__)


# ──────────────────────────── Feature Definitions ─────────────────────────────
class Feature(str, Enum):
    """Supported LLM features"""
    STREAMING = "streaming"
    TOOLS = "tools" 
    VISION = "vision"
    JSON_MODE = "json_mode"
    PARALLEL_CALLS = "parallel_calls"
    SYSTEM_MESSAGES = "system_messages"
    MULTIMODAL = "multimodal"

    @classmethod
    def from_string(cls, value: str) -> "Feature":
        """Convert string to Feature enum"""
        try:
            return cls(value.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown feature: {value}") from exc


# ──────────────────────────── Data Classes ─────────────────────────────
@dataclass
class ModelCapabilities:
    """Model-specific capabilities with inheritance from provider"""
    pattern: str
    features: Set[Feature] = field(default_factory=set)
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    
    def matches(self, model_name: str) -> bool:
        """Check if this capability applies to the given model"""
        return bool(re.match(self.pattern, model_name, flags=re.IGNORECASE))
    
    def get_effective_features(self, provider_features: Set[Feature]) -> Set[Feature]:
        """Get effective features by inheriting from provider and adding model-specific"""
        return provider_features.union(self.features)


@dataclass 
class ProviderConfig:
    """Complete unified provider configuration"""
    name: str
    
    # Client configuration
    client_class: str = ""
    api_key_env: Optional[str] = None
    api_key_fallback_env: Optional[str] = None
    api_base: Optional[str] = None
    
    # Model configuration
    default_model: str = ""
    models: List[str] = field(default_factory=list)
    model_aliases: Dict[str, str] = field(default_factory=dict)
    
    # Provider-level capabilities (baseline for all models)
    features: Set[Feature] = field(default_factory=set)
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    rate_limits: Dict[str, int] = field(default_factory=dict)
    
    # Model-specific capability overrides
    model_capabilities: List[ModelCapabilities] = field(default_factory=list)
    
    # Inheritance and extras
    inherits: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def supports_feature(self, feature: Union[str, Feature], model: Optional[str] = None) -> bool:
        """Check if provider/model supports a feature"""
        if isinstance(feature, str):
            feature = Feature.from_string(feature)
        
        if model:
            # Check model-specific capabilities
            model_caps = self.get_model_capabilities(model)
            effective_features = model_caps.get_effective_features(self.features)
            return feature in effective_features
        else:
            # Check provider baseline
            return feature in self.features
    
    def get_model_capabilities(self, model: Optional[str] = None) -> ModelCapabilities:
        """Get capabilities for specific model"""
        if model and self.model_capabilities:
            for mc in self.model_capabilities:
                if mc.matches(model):
                    # Return model-specific caps with proper inheritance
                    return ModelCapabilities(
                        pattern=mc.pattern,
                        features=mc.get_effective_features(self.features),
                        max_context_length=mc.max_context_length or self.max_context_length,
                        max_output_tokens=mc.max_output_tokens or self.max_output_tokens
                    )
        
        # Return provider defaults
        return ModelCapabilities(
            pattern=".*",
            features=self.features.copy(),
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens
        )
    
    def get_rate_limit(self, tier: str = "default") -> Optional[int]:
        """Get rate limit for tier"""
        return self.rate_limits.get(tier)


# ──────────────────────────── Configuration Manager ─────────────────────────────
class UnifiedConfigManager:
    """Unified configuration manager with validation and capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.providers: Dict[str, ProviderConfig] = {}
        self.global_aliases: Dict[str, str] = {}
        self.global_settings: Dict[str, Any] = {}
        self._loaded = False
        
        # Load environment variables first
        self._load_environment()
        
        # No built-in defaults - everything comes from config files
    
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
    
    def _find_config_file(self) -> Optional[Path]:
        """Find YAML configuration file"""
        if self.config_path:
            path = Path(self.config_path)
            if path.exists():
                return path
        
        candidates = [
            # Environment variable
            os.getenv("CHUK_LLM_CONFIG"),
            # Current directory - unified names
            "chuk_llm.yaml",
            "providers.yaml", 
            "llm_config.yaml",
            # Config directory
            "config/chuk_llm.yaml",
            # Package root
            Path(__file__).parent.parent / "chuk_llm.yaml",
            Path(__file__).parent.parent / "providers.yaml",
            # User config
            Path.home() / ".chuk_llm" / "config.yaml",
        ]
        
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser().resolve()
                if path.exists():
                    return path
        
        return None
    
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
        """Load YAML configuration"""
        if not yaml:
            logger.warning("PyYAML not available, using built-in defaults only")
            return {}
        
        config_file = self._find_config_file()
        if not config_file:
            logger.info("No configuration file found, using built-in defaults")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as exc:
            logger.error(f"Failed to load config from {config_file}: {exc}")
            return {}
    
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
            
            # Extra fields
            extra_fields = {k: v for k, v in data.items() 
                          if k not in {"client_class", "api_key_env", "api_key_fallback_env", 
                                      "api_base", "default_model", "models", "model_aliases",
                                      "features", "max_context_length", "max_output_tokens", 
                                      "rate_limits", "model_capabilities", "inherits"}}
            provider.extra.update(extra_fields)
    
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
        self.load()


# ──────────────────────────── Validation ─────────────────────────────
class ConfigValidator:
    """Validates configurations and requests"""
    
    @staticmethod
    def validate_provider_config(provider: ProviderConfig, strict: bool = False) -> Tuple[bool, List[str]]:
        """Validate provider configuration"""
        issues = []
        
        # Check required fields
        if not provider.client_class:
            issues.append(f"Missing 'client_class' for provider {provider.name}")
        
        # Check API key for non-local providers
        if provider.name not in ["ollama", "local"]:
            if provider.api_key_env and not os.getenv(provider.api_key_env):
                if not provider.api_key_fallback_env or not os.getenv(provider.api_key_fallback_env):
                    issues.append(f"Missing API key: {provider.api_key_env} environment variable not set")
        
        # Validate API base URL
        if provider.api_base and not ConfigValidator._is_valid_url(provider.api_base):
            issues.append(f"Invalid API base URL: {provider.api_base}")
        
        # Check default model
        if not provider.default_model:
            issues.append(f"Missing 'default_model' for provider {provider.name}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_request_compatibility(
        provider_name: str,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[bool, List[str]]:
        """Validate if request is compatible with provider/model"""
        issues = []
        
        try:
            config_manager = get_config()
            provider = config_manager.get_provider(provider_name)
            
            # Check streaming support
            if stream and not provider.supports_feature(Feature.STREAMING, model):
                issues.append(f"{provider_name}/{model or 'default'} doesn't support streaming")
            
            # Check tools support
            if tools and not provider.supports_feature(Feature.TOOLS, model):
                issues.append(f"{provider_name}/{model or 'default'} doesn't support function calling")
            
            # Check vision support
            if messages and ConfigValidator._has_vision_content(messages):
                if not provider.supports_feature(Feature.VISION, model):
                    issues.append(f"{provider_name}/{model or 'default'} doesn't support vision/image inputs")
            
            # Check JSON mode
            if kwargs.get("response_format") == "json":
                if not provider.supports_feature(Feature.JSON_MODE, model):
                    issues.append(f"{provider_name}/{model or 'default'} doesn't support JSON mode")
            
        except Exception as exc:
            issues.append(f"Configuration error: {exc}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Basic URL validation"""
        if not url:
            return False
        
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$',
            re.IGNORECASE
        )
        return url_pattern.match(url) is not None
    
    @staticmethod
    def _has_vision_content(messages: List[Dict[str, Any]]) -> bool:
        """Check if messages contain vision/image content"""
        if not messages:
            return False
        
        for message in messages:
            if not message:
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["image", "image_url"]:
                        return True
        return False


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