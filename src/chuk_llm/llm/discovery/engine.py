# chuk_llm/llm/discovery/engine.py
"""
Universal dynamic model discovery engine for all providers
Enhanced with modular discoverer architecture
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
            "capabilities": [f.value if hasattr(f, 'value') else str(f) for f in self.capabilities],
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "parameters": self.parameters,
            "metadata": self.metadata or {}
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

        # Enhanced configuration sections
        self.deployment_rules = self.config.get("deployment_rules", {})
        self.universal_size_rules = self.config.get("universal_size_rules", {})
        self.universal_patterns = self.config.get("universal_patterns", {})

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

        # Apply universal patterns (from global defaults)
        model = self._apply_universal_patterns(model, model_name)

        # Apply family rules
        model = self._apply_family_rules(model, model_name)

        # Apply deployment rules (for Azure OpenAI, etc.)
        model = self._apply_deployment_rules(model, model_name)

        # Apply pattern-based rules
        model = self._apply_pattern_rules(model, model_name)

        # Apply size-based rules
        model = self._apply_size_rules(model)
        model = self._apply_universal_size_rules(model)

        # Extract parameters
        model = self._extract_parameters(model, model_name)

        # Final validation and cleanup
        model = self._validate_and_cleanup(model)

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

    def _apply_universal_patterns(self, model: DiscoveredModel, model_name: str) -> DiscoveredModel:
        """Apply universal patterns that work across all providers"""
        for pattern_name, pattern_config in self.universal_patterns.items():
            patterns = pattern_config.get("patterns", [])

            for pattern in patterns:
                if re.search(pattern, model_name, re.IGNORECASE):
                    # Add features
                    if "add_features" in pattern_config:
                        add_features = set(Feature.from_string(f) for f in pattern_config["add_features"])
                        model.capabilities.update(add_features)

                    # Remove features
                    if "remove_features" in pattern_config:
                        remove_features = set(Feature.from_string(f) for f in pattern_config["remove_features"])
                        model.capabilities -= remove_features

                    # Set family
                    if "family" in pattern_config:
                        model.family = pattern_config["family"]

                    # Set context length
                    if "context_length" in pattern_config:
                        model.context_length = pattern_config["context_length"]

        return model

    def _apply_family_rules(self, model: DiscoveredModel, model_name: str) -> DiscoveredModel:
        """Apply family-based inference rules"""
        for family, family_config in self.family_rules.items():
            patterns = family_config.get("patterns", [])

            # Check if model matches any family pattern
            for pattern in patterns:
                if re.search(pattern, model_name, re.IGNORECASE):
                    model.family = family

                    # Base context first (so specific rules can override)
                    if "base_context_length" in family_config:
                        model.context_length = family_config["base_context_length"]

                    # Add family features
                    if "features" in family_config:
                        family_features = set(Feature.from_string(f) for f in family_config["features"])
                        model.capabilities.update(family_features)

                    # Context rules override base
                    context_rules = family_config.get("context_rules", {})
                    for ctx_pattern, ctx_length in context_rules.items():
                        if re.search(ctx_pattern, model_name, re.IGNORECASE):
                            model.context_length = ctx_length
                            break

                    # Set max output tokens
                    if "base_max_output_tokens" in family_config:
                        model.max_output_tokens = family_config["base_max_output_tokens"]

                    # Handle restrictions (for reasoning models)
                    if "restrictions" in family_config:
                        restrictions = family_config["restrictions"]
                        if "no_streaming" in restrictions:
                            model.capabilities.discard(Feature.STREAMING)
                            model.metadata.setdefault("parameter_requirements", {})
                            model.metadata["parameter_requirements"]["no_streaming"] = True
                        if "no_system_messages" in restrictions:
                            model.capabilities.discard(Feature.SYSTEM_MESSAGES)
                            model.metadata.setdefault("parameter_requirements", {})
                            model.metadata["parameter_requirements"]["no_system_messages"] = True
                        if "org_verification_for_streaming" in restrictions:
                            model.metadata["requires_org_verification_for_streaming"] = True

                    # Handle special parameters
                    if "special_params" in family_config:
                        model.metadata.setdefault("special_parameters", {})
                        model.metadata["special_parameters"] = family_config["special_params"]

                    return model

        return model

    def _apply_deployment_rules(self, model: DiscoveredModel, model_name: str) -> DiscoveredModel:
        """Apply deployment-specific rules (for Azure OpenAI, etc.)"""
        for deployment_name, deployment_config in self.deployment_rules.items():
            patterns = deployment_config.get("patterns", [])

            for pattern in patterns:
                if re.search(pattern, model_name, re.IGNORECASE):
                    # Add deployment features
                    if "features" in deployment_config:
                        deployment_features = set(Feature.from_string(f) for f in deployment_config["features"])
                        model.capabilities.update(deployment_features)

                    # Set deployment context and output limits
                    if "base_context_length" in deployment_config:
                        model.context_length = deployment_config["base_context_length"]

                    if "base_max_output_tokens" in deployment_config:
                        model.max_output_tokens = deployment_config["base_max_output_tokens"]

                    break

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

    def _apply_universal_size_rules(self, model: DiscoveredModel) -> DiscoveredModel:
        """Apply universal size rules from global defaults"""
        if model.size_bytes is None:
            return model

        for rule_name, rule_config in self.universal_size_rules.items():
            min_size = rule_config.get("min_size_bytes", 0)
            max_size = rule_config.get("max_size_bytes", float('inf'))

            if min_size <= model.size_bytes <= max_size:
                # Add universal features based on size
                if "add_features" in rule_config:
                    add_features = set(Feature.from_string(f) for f in rule_config["add_features"])
                    model.capabilities.update(add_features)

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

    def _validate_and_cleanup(self, model: DiscoveredModel) -> DiscoveredModel:
        """Final validation and cleanup of model data"""
        model.metadata = model.metadata or {}

        # Ensure we have at least the TEXT feature
        if not model.capabilities:
            model.capabilities = {Feature.TEXT}

        # Ensure reasonable defaults for context and output
        if not model.context_length or model.context_length <= 0:
            model.context_length = self.default_context

        if not model.max_output_tokens or model.max_output_tokens <= 0:
            model.max_output_tokens = self.default_max_output

        # Respect parameter_requirements flags
        params_req = model.metadata.get("parameter_requirements", {}) or {}
        if params_req.get("no_streaming"):
            model.capabilities.discard(Feature.STREAMING)
        if params_req.get("no_system_messages"):
            model.capabilities.discard(Feature.SYSTEM_MESSAGES)

        # If org verification is required for streaming, don't advertise streaming
        if model.metadata.get("requires_org_verification_for_streaming"):
            model.capabilities.discard(Feature.STREAMING)

        # Ensure family is set
        if model.family == "unknown" and model.capabilities:
            if Feature.REASONING in model.capabilities:
                model.family = "reasoning"
            elif Feature.VISION in model.capabilities:
                model.family = "vision"
            elif Feature.MULTIMODAL in model.capabilities:
                model.family = "multimodal"

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
            discovery_config = provider_config.extra.get("dynamic_discovery", {})
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
            # Get raw model data from provider using cached discovery
            raw_models = await self.discoverer.discover_with_cache(force_refresh)

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
                    model.max_output_tokens,
                )

                if cap_key not in capability_groups:
                    capability_groups[cap_key] = []
                capability_groups[cap_key].append(model.name)

            for (features, context_length, max_output_tokens), model_names in capability_groups.items():
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
                if max_output_tokens:
                    config_lines.append(f'    max_output_tokens: {max_output_tokens}')
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
