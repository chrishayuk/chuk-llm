# chuk_llm/llm/discovery/engine.py
"""
Universal dynamic model discovery engine for all providers
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Protocol
from dataclasses import dataclass
from pathlib import Path

from chuk_llm.configuration import Feature, ModelCapabilities, ProviderConfig

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
    metadata: Dict[str, Any] = None
    
    # Inferred properties
    family: str = "unknown"
    capabilities: Set[Feature] = None
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    parameters: Optional[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = set()
        if self.metadata is None:
            self.metadata = {}


class ModelDiscoveryProtocol(Protocol):
    """Protocol for provider-specific model discovery"""
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover available models and return raw model data"""
        ...
    
    async def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metadata for a specific model"""
        ...


class BaseModelDiscoverer(ABC):
    """Base class for provider-specific model discoverers"""
    
    def __init__(self, provider_name: str, **config):
        self.provider_name = provider_name
        self.config = config
    
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


class ConfigDrivenInferenceEngine:
    """Universal inference engine that works with any provider"""
    
    def __init__(self, provider_name: str, inference_config: Dict[str, Any]):
        """
        Initialize inference engine for a provider.
        
        Args:
            provider_name: Name of the provider
            inference_config: Configuration dict with inference rules
        """
        self.provider_name = provider_name
        self.config = inference_config
        
        # Parse configuration sections
        self.default_features = set(Feature.from_string(f) for f in self.config.get("default_features", ["text"]))
        self.default_context = self.config.get("default_context_length", 8192)
        self.default_max_output = self.config.get("default_max_output_tokens", 4096)
        
        self.family_rules = self.config.get("family_rules", {})
        self.pattern_rules = self.config.get("pattern_rules", {})
        self.size_rules = self.config.get("size_rules", {})
        self.model_overrides = self.config.get("model_overrides", {})
        
    def infer_capabilities(self, model: DiscoveredModel) -> DiscoveredModel:
        """Infer model capabilities using configuration rules"""
        model_name = model.name.lower()
        
        # Start with defaults
        model.capabilities = self.default_features.copy()
        model.context_length = model.context_length or self.default_context
        model.max_output_tokens = model.max_output_tokens or self.default_max_output
        
        # Apply model-specific overrides first (highest priority)
        if model.name in self.model_overrides:
            self._apply_model_override(model, self.model_overrides[model.name])
            return model
        
        # Apply family rules
        model = self._apply_family_rules(model, model_name)
        
        # Apply pattern-based rules
        model = self._apply_pattern_rules(model, model_name)
        
        # Apply size-based rules
        model = self._apply_size_rules(model)
        
        # Extract parameters
        model = self._extract_parameters(model, model_name)
        
        return model
    
    def _apply_model_override(self, model: DiscoveredModel, override_config: Dict[str, Any]):
        """Apply specific model override configuration"""
        if "features" in override_config:
            model.capabilities = set(Feature.from_string(f) for f in override_config["features"])
        
        if "context_length" in override_config:
            model.context_length = override_config["context_length"]
        
        if "max_output_tokens" in override_config:
            model.max_output_tokens = override_config["max_output_tokens"]
        
        if "family" in override_config:
            model.family = override_config["family"]
    
    def _apply_family_rules(self, model: DiscoveredModel, model_name: str) -> DiscoveredModel:
        """Apply family-based inference rules"""
        for family, family_config in self.family_rules.items():
            patterns = family_config.get("patterns", [])
            
            # Check if model matches any family pattern
            for pattern in patterns:
                if re.search(pattern, model_name, re.IGNORECASE):
                    model.family = family
                    
                    # Add family features
                    if "features" in family_config:
                        family_features = set(Feature.from_string(f) for f in family_config["features"])
                        model.capabilities.update(family_features)
                    
                    # Set context length rules
                    context_rules = family_config.get("context_rules", {})
                    for ctx_pattern, ctx_length in context_rules.items():
                        if re.search(ctx_pattern, model_name, re.IGNORECASE):
                            model.context_length = ctx_length
                            break
                    
                    # Override default context if family has base context
                    if "base_context_length" in family_config:
                        model.context_length = family_config["base_context_length"]
                    
                    # Set max output tokens
                    if "base_max_output_tokens" in family_config:
                        model.max_output_tokens = family_config["base_max_output_tokens"]
                    
                    return model
        
        return model
    
    def _apply_pattern_rules(self, model: DiscoveredModel, model_name: str) -> DiscoveredModel:
        """Apply pattern-based inference rules"""
        for rule_name, rule_config in self.pattern_rules.items():
            patterns = rule_config.get("patterns", [])
            
            for pattern in patterns:
                if re.search(pattern, model_name, re.IGNORECASE):
                    # Add features
                    if "add_features" in rule_config:
                        add_features = set(Feature.from_string(f) for f in rule_config["add_features"])
                        model.capabilities.update(add_features)
                    
                    # Remove features
                    if "remove_features" in rule_config:
                        remove_features = set(Feature.from_string(f) for f in rule_config["remove_features"])
                        model.capabilities -= remove_features
                    
                    # Override context
                    if "context_length" in rule_config:
                        model.context_length = rule_config["context_length"]
                    
                    # Override max output
                    if "max_output_tokens" in rule_config:
                        model.max_output_tokens = rule_config["max_output_tokens"]
                    
                    # Set family if specified
                    if "family" in rule_config:
                        model.family = rule_config["family"]
        
        return model
    
    def _apply_size_rules(self, model: DiscoveredModel) -> DiscoveredModel:
        """Apply size-based inference rules"""
        if model.size_bytes is None:
            return model
        
        for rule_name, rule_config in self.size_rules.items():
            min_size = rule_config.get("min_size_bytes", 0)
            max_size = rule_config.get("max_size_bytes", float('inf'))
            
            if min_size <= model.size_bytes <= max_size:
                # Add features based on size
                if "add_features" in rule_config:
                    add_features = set(Feature.from_string(f) for f in rule_config["add_features"])
                    model.capabilities.update(add_features)
                
                # Set context based on size
                if "context_length" in rule_config:
                    model.context_length = rule_config["context_length"]
                
                # Set max output based on size
                if "max_output_tokens" in rule_config:
                    model.max_output_tokens = rule_config["max_output_tokens"]
        
        return model
    
    def _extract_parameters(self, model: DiscoveredModel, model_name: str) -> DiscoveredModel:
        """Extract parameter count from model name"""
        param_patterns = self.config.get("parameter_patterns", [r'(\d+(?:\.\d+)?)b'])
        
        for pattern in param_patterns:
            match = re.search(pattern, model_name, re.IGNORECASE)
            if match:
                model.parameters = f"{match.group(1)}B"
                break
        
        return model


class UniversalModelDiscoveryManager:
    """Universal model discovery manager that works with any provider"""
    
    def __init__(self, provider_name: str, discoverer: BaseModelDiscoverer, inference_config: Optional[Dict[str, Any]] = None):
        """
        Initialize universal discovery manager.
        
        Args:
            provider_name: Name of the provider
            discoverer: Provider-specific discoverer implementation
            inference_config: Configuration for capability inference
        """
        self.provider_name = provider_name
        self.discoverer = discoverer
        
        # Load inference config
        self.inference_config = inference_config or self._load_default_inference_config()
        self.inference_engine = ConfigDrivenInferenceEngine(provider_name, self.inference_config)
        
        # Caching
        self._cached_models: Optional[List[DiscoveredModel]] = None
        self._cache_timeout = 300  # 5 minutes
        self._last_update: Optional[float] = None
    
    def _load_default_inference_config(self) -> Dict[str, Any]:
        """Load default inference configuration for provider"""
        try:
            from chuk_llm.configuration import get_config
            config_manager = get_config()
            provider_config = config_manager.get_provider(self.provider_name)
            
            # Look for discovery config in provider extra
            discovery_config = provider_config.extra.get("model_discovery", {})
            inference_config = discovery_config.get("inference_config", {})
            
            # Merge with any provider-level inference config
            if "model_inference" in provider_config.extra:
                inference_config = {**provider_config.extra["model_inference"], **inference_config}
            
            return inference_config or self._get_minimal_config()
            
        except Exception as e:
            log.debug(f"Failed to load inference config for {self.provider_name}: {e}")
            return self._get_minimal_config()
    
    def _get_minimal_config(self) -> Dict[str, Any]:
        """Get minimal fallback configuration"""
        return {
            "default_features": ["text"],
            "default_context_length": 8192,
            "default_max_output_tokens": 4096,
            "family_rules": {},
            "pattern_rules": {},
            "size_rules": {},
            "model_overrides": {},
            "parameter_patterns": [r'(\d+(?:\.\d+)?)b']
        }
    
    async def discover_models(self, force_refresh: bool = False) -> List[DiscoveredModel]:
        """Discover models using provider-specific discoverer and universal inference"""
        # Check cache
        if not force_refresh and self._cached_models and self._last_update:
            if time.time() - self._last_update < self._cache_timeout:
                return self._cached_models
        
        try:
            # Get raw model data from provider
            raw_models = await self.discoverer.discover_models()
            
            # Convert to DiscoveredModel objects
            discovered_models = []
            for raw_model in raw_models:
                model = self.discoverer.normalize_model_data(raw_model)
                
                # Apply universal inference
                model = self.inference_engine.infer_capabilities(model)
                discovered_models.append(model)
            
            # Cache results
            self._cached_models = discovered_models
            self._last_update = time.time()
            
            log.info(f"Discovered {len(discovered_models)} models for {self.provider_name} using universal inference")
            return discovered_models
            
        except Exception as e:
            log.error(f"Failed to discover models for {self.provider_name}: {e}")
            return self._cached_models or []
    
    def update_inference_config(self, new_config: Dict[str, Any]):
        """Update inference configuration and clear cache"""
        self.inference_config = new_config
        self.inference_engine = ConfigDrivenInferenceEngine(self.provider_name, new_config)
        self._cached_models = None  # Force refresh
        log.info(f"Updated inference configuration for {self.provider_name}")
    
    def get_model_capabilities(self, model_name: str) -> Optional[ModelCapabilities]:
        """Get capabilities for a specific model"""
        if not self._cached_models:
            return None
        
        # Find model
        target_model = None
        for model in self._cached_models:
            if model.name == model_name:
                target_model = model
                break
            elif model_name in model.name or model.name in model_name:
                target_model = model  # Fuzzy match fallback
        
        if not target_model:
            return None
        
        return ModelCapabilities(
            pattern=f"^{re.escape(target_model.name)}$",
            features=target_model.capabilities,
            max_context_length=target_model.context_length,
            max_output_tokens=target_model.max_output_tokens
        )
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        if not self._cached_models:
            return []
        return [model.name for model in self._cached_models]
    
    def generate_config_yaml(self) -> str:
        """Generate YAML configuration for discovered models"""
        if not self._cached_models:
            return ""
        
        config_lines = []
        config_lines.append(f"# Dynamically discovered {self.provider_name} models")
        config_lines.append("models:")
        
        # Filter out models without text capability (e.g., embeddings)
        text_models = [m for m in self._cached_models if Feature.TEXT in m.capabilities]
        
        for model in text_models:
            config_lines.append(f'  - "{model.name}"')
        
        if text_models:
            config_lines.append("\nmodel_capabilities:")
            
            # Group by capabilities to reduce duplication
            capability_groups = {}
            for model in text_models:
                cap_key = (
                    tuple(sorted(f.value for f in model.capabilities)),
                    model.context_length,
                    model.max_output_tokens
                )
                
                if cap_key not in capability_groups:
                    capability_groups[cap_key] = []
                capability_groups[cap_key].append(model.name)
            
            for (features, context_length, max_output), model_names in capability_groups.items():
                # Create regex pattern for models
                if len(model_names) == 1:
                    pattern = f"^{re.escape(model_names[0])}$"
                else:
                    escaped_names = [re.escape(name) for name in model_names]
                    pattern = f"^({'|'.join(escaped_names)})$"
                
                config_lines.append(f'  - pattern: "{pattern}"')
                config_lines.append(f'    features: [{", ".join(features)}]')
                if context_length:
                    config_lines.append(f'    max_context_length: {context_length}')
                if max_output:
                    config_lines.append(f'    max_output_tokens: {max_output}')
                config_lines.append("")
        
        return "\n".join(config_lines)
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered models"""
        if not self._cached_models:
            return {"total": 0}
        
        families = {}
        feature_counts = {}
        total_size = 0
        
        for model in self._cached_models:
            # Count by family
            families[model.family] = families.get(model.family, 0) + 1
            
            # Count by features
            for feature in model.capabilities:
                feature_counts[feature.value] = feature_counts.get(feature.value, 0) + 1
            
            # Sum sizes
            if model.size_bytes:
                total_size += model.size_bytes
        
        return {
            "total": len(self._cached_models),
            "families": families,
            "features": feature_counts,
            "total_size_gb": round(total_size / (1024**3), 1),
            "cache_age_seconds": int(time.time() - self._last_update) if self._last_update else 0,
            "provider": self.provider_name
        }