# chuk_llm/llm/discovery/manager.py
"""
Universal model discovery manager that coordinates all discoverers
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base import BaseModelDiscoverer, DiscoveredModel, DiscovererFactory
from .engine import ConfigDrivenInferenceEngine, UniversalModelDiscoveryManager

log = logging.getLogger(__name__)


@dataclass
class DiscoveryResults:
    """Results from model discovery across all providers"""
    total_models: int
    models_by_provider: Dict[str, List[DiscoveredModel]]
    discovery_time: float
    errors: Dict[str, str]
    summary: Dict[str, Any]


class UniversalDiscoveryManager:
    """Manages model discovery across all providers"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.provider_managers = {}
        self._discovery_cache = {}
        self._cache_timeout = 300  # 5 minutes
        self._last_full_discovery = None

        # Load provider configurations
        if config_manager:
            self._initialize_from_config()

    def _initialize_from_config(self):
        """Initialize discovery managers from configuration"""
        try:
            providers = self.config_manager.get_all_providers()

            for provider_name, provider_config in providers.items():
                discovery_config = provider_config.extra.get("dynamic_discovery", {})

                if discovery_config.get("enabled", False):
                    self._setup_provider_discovery(provider_name, discovery_config)

        except Exception as e:
            log.warning(f"Failed to initialize from config: {e}")

    def _setup_provider_discovery(self, provider_name: str, discovery_config: Dict[str, Any]):
        """Setup discovery for a specific provider"""
        try:
            discoverer_config = self._build_discoverer_config(provider_name, discovery_config)
            discoverer = DiscovererFactory.create_discoverer(provider_name, **discoverer_config)

            inference_config = discovery_config.get("inference_config", {})
            manager = UniversalModelDiscoveryManager(provider_name, discoverer, inference_config)

            self.provider_managers[provider_name] = manager
            log.debug(f"Setup discovery for {provider_name}")

        except Exception as e:
            log.warning(f"Failed to setup discovery for {provider_name}: {e}")

    def _build_discoverer_config(self, provider_name: str, discovery_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build discoverer configuration for a provider"""
        config = {}

        # Add provider-specific configuration
        if provider_name == "openai":
            import os
            config.update({
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_base": "https://api.openai.com/v1"
            })
        elif provider_name == "azure_openai":
            import os
            config.update({
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_version": discovery_config.get("api_version", "2024-02-01"),
                "azure_ad_token": os.getenv("AZURE_AD_TOKEN")
            })
        elif provider_name == "ollama":
            config.update({
                "api_base": discovery_config.get("api_base", "http://localhost:11434")
            })
        elif provider_name == "groq":
            import os
            config.update({
                "api_key": os.getenv("GROQ_API_KEY"),
                "api_base": "https://api.groq.com/openai/v1"
            })
        elif provider_name == "deepseek":
            import os
            config.update({
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "api_base": "https://api.deepseek.com"
            })
        elif provider_name == "huggingface":
            import os
            config.update({
                "api_key": os.getenv("HUGGINGFACE_API_KEY"),
                "limit": discovery_config.get("limit", 50)
            })
        elif provider_name == "local":
            config.update({
                "model_paths": discovery_config.get("model_paths", [])
            })

        # Add common configuration
        config.update({
            "cache_timeout": discovery_config.get("cache_timeout", 300),
            **discovery_config.get("discoverer_config", {})
        })

        return config

    async def discover_all_models(self, force_refresh: bool = False) -> DiscoveryResults:
        """Discover models from all configured providers"""
        start_time = time.time()

        # Check cache
        if not force_refresh and self._discovery_cache and self._last_full_discovery:
            if time.time() - self._last_full_discovery < self._cache_timeout:
                log.debug("Using cached discovery results")
                return self._discovery_cache

        log.info("Starting universal model discovery...")

        # Run discovery for all providers concurrently
        discovery_tasks = {}
        for provider_name, manager in self.provider_managers.items():
            discovery_tasks[provider_name] = asyncio.create_task(
                self._discover_provider_models(provider_name, manager, force_refresh)
            )

        # Wait for all discoveries to complete
        results = await asyncio.gather(*discovery_tasks.values(), return_exceptions=True)

        # Process results
        models_by_provider = {}
        errors = {}
        total_models = 0

        for provider_name, result in zip(discovery_tasks.keys(), results):
            if isinstance(result, Exception):
                errors[provider_name] = str(result)
                models_by_provider[provider_name] = []
            else:
                models_by_provider[provider_name] = result
                total_models += len(result)

        discovery_time = time.time() - start_time

        # Generate summary
        summary = self._generate_discovery_summary(models_by_provider, errors, discovery_time)

        # Create results
        discovery_results = DiscoveryResults(
            total_models=total_models,
            models_by_provider=models_by_provider,
            discovery_time=discovery_time,
            errors=errors,
            summary=summary
        )

        # Cache results
        self._discovery_cache = discovery_results
        self._last_full_discovery = time.time()

        log.info(f"Discovery completed: {total_models} models from {len(models_by_provider)} providers in {discovery_time:.2f}s")
        return discovery_results

    async def _discover_provider_models(self, provider_name: str, manager: UniversalModelDiscoveryManager,
                                        force_refresh: bool) -> List[DiscoveredModel]:
        """Discover models for a specific provider"""
        try:
            log.debug(f"Discovering models for {provider_name}...")
            models = await manager.discover_models(force_refresh)
            log.debug(f"Found {len(models)} models for {provider_name}")
            return models
        except Exception as e:
            log.error(f"Discovery failed for {provider_name}: {e}")
            raise

    def _generate_discovery_summary(self, models_by_provider: Dict[str, List[DiscoveredModel]],
                                    errors: Dict[str, str], discovery_time: float) -> Dict[str, Any]:
        """Generate discovery summary statistics"""

        # Count models by family
        families = {}
        capabilities = {}
        reasoning_models = 0
        vision_models = 0
        code_models = 0

        for provider_name, models in models_by_provider.items():
            for model in models:
                # Count by family
                family = model.family
                families[family] = families.get(family, 0) + 1

                # Count capabilities
                for capability in model.capabilities:
                    cap_name = capability.value if hasattr(capability, 'value') else str(capability)
                    capabilities[cap_name] = capabilities.get(cap_name, 0) + 1

                # Count special model types
                metadata = model.metadata or {}
                if metadata.get('reasoning_capable', False) or metadata.get('is_reasoning', False):
                    reasoning_models += 1
                if metadata.get('supports_vision', False) or metadata.get('is_vision', False):
                    vision_models += 1
                if metadata.get('specialization') == 'code' or metadata.get('is_code', False):
                    code_models += 1

        # Provider success rate
        total_providers = len(self.provider_managers)
        successful_providers = len([p for p in models_by_provider.keys() if p not in errors])
        success_rate = (successful_providers / total_providers * 100) if total_providers > 0 else 0

        # Top families and capabilities
        top_families = sorted(families.items(), key=lambda x: x[1], reverse=True)[:10]
        top_capabilities = sorted(capabilities.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "discovery_time": round(discovery_time, 2),
            "total_providers": total_providers,
            "successful_providers": successful_providers,
            "success_rate": round(success_rate, 1),
            "total_models": sum(len(models) for models in models_by_provider.values()),
            "models_per_provider": {
                provider: len(models) for provider, models in models_by_provider.items()
            },
            "special_model_counts": {
                "reasoning_models": reasoning_models,
                "vision_models": vision_models,
                "code_models": code_models
            },
            "top_families": top_families,
            "top_capabilities": top_capabilities,
            "errors": errors
        }

    async def discover_provider_models(self, provider_name: str, force_refresh: bool = False) -> List[DiscoveredModel]:
        """Discover models for a specific provider"""
        if provider_name not in self.provider_managers:
            raise ValueError(f"No discovery manager for provider: {provider_name}")

        manager = self.provider_managers[provider_name]
        return await manager.discover_models(force_refresh)

    def get_available_providers(self) -> List[str]:
        """Get list of providers with discovery enabled"""
        return list(self.provider_managers.keys())

    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a specific provider's discovery"""
        if provider_name not in self.provider_managers:
            return {"error": f"No discovery manager for provider: {provider_name}"}

        manager = self.provider_managers[provider_name]
        return manager.get_discovery_stats()

    def register_custom_provider(self, provider_name: str, discoverer: BaseModelDiscoverer,
                                 inference_config: Optional[Dict[str, Any]] = None):
        """Register a custom provider discoverer"""
        manager = UniversalModelDiscoveryManager(provider_name, discoverer, inference_config)
        self.provider_managers[provider_name] = manager
        log.info(f"Registered custom discoverer for {provider_name}")

    def get_model_recommendations(self, use_case: str = "general") -> List[Dict[str, Any]]:
        """Get model recommendations based on use case"""
        if not self._discovery_cache:
            return []

        all_models = []
        for provider_models in self._discovery_cache.models_by_provider.values():
            all_models.extend(provider_models)

        # Filter and rank models based on use case
        recommendations = []

        for model in all_models:
            metadata = model.metadata or {}
            score = 0

            # Base scoring
            if use_case == "reasoning" and metadata.get('reasoning_capable', False):
                score += 100
            elif use_case == "vision" and metadata.get('supports_vision', False):
                score += 100
            elif use_case == "code" and metadata.get('specialization') == 'code':
                score += 100
            elif use_case == "general":
                score += 50

            # Performance tier bonus
            tier_bonuses = {
                "highest": 20, "maximum": 20, "high": 15, "very-fast": 15,
                "medium": 10, "fast": 10, "large": 8, "small": 5
            }
            tier = metadata.get('performance_tier', 'medium')
            score += tier_bonuses.get(tier, 0)

            # Capability bonuses
            if metadata.get('supports_tools', False):
                score += 10
            if metadata.get('supports_streaming', False):
                score += 5

            # Provider reliability bonus
            provider_bonuses = {
                "openai": 15, "anthropic": 15, "ollama": 10,
                "groq": 8, "deepseek": 5
            }
            score += provider_bonuses.get(model.provider, 0)

            if score > 50:  # Only recommend models above threshold
                recommendations.append({
                    "model": model.name,
                    "provider": model.provider,
                    "family": model.family,
                    "score": score,
                    "reasoning": metadata.get('reasoning_capable', False),
                    "vision": metadata.get('supports_vision', False),
                    "tools": metadata.get('supports_tools', False),
                    "streaming": metadata.get('supports_streaming', False),
                    "specialization": metadata.get('specialization', 'general'),
                    "performance_tier": metadata.get('performance_tier', 'medium'),
                    "context_length": metadata.get('estimated_context_length', 8192)
                })

        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:10]  # Top 10 recommendations

    def generate_config_updates(self) -> Dict[str, Any]:
        """Generate configuration updates based on discovered models"""
        if not self._discovery_cache:
            return {}

        config_updates = {}

        for provider_name, models in self._discovery_cache.models_by_provider.items():
            if provider_name in self.provider_managers:
                manager = self.provider_managers[provider_name]
                yaml_config = manager.generate_config_yaml()
                if yaml_config:
                    config_updates[provider_name] = yaml_config

        return config_updates

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all discovery managers"""
        health_status = {
            "overall_status": "healthy",
            "providers": {},
            "total_providers": len(self.provider_managers),
            "healthy_providers": 0,
            "check_time": time.time()
        }

        for provider_name, manager in self.provider_managers.items():
            try:
                # Quick discovery test
                start_time = time.time()
                models = await manager.discover_models(force_refresh=True)
                response_time = time.time() - start_time

                provider_status = {
                    "status": "healthy",
                    "response_time": round(response_time, 2),
                    "model_count": len(models),
                    "error": None
                }
                health_status["healthy_providers"] += 1

            except Exception as e:
                provider_status = {
                    "status": "unhealthy",
                    "response_time": None,
                    "model_count": 0,
                    "error": str(e)
                }

            health_status["providers"][provider_name] = provider_status

        # Determine overall status
        if health_status["healthy_providers"] == 0:
            health_status["overall_status"] = "critical"
        elif health_status["healthy_providers"] < health_status["total_providers"]:
            health_status["overall_status"] = "degraded"

        return health_status
