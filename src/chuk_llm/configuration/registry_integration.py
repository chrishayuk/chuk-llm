# chuk_llm/configuration/registry_integration.py
"""
Registry integration for configuration manager.

This replaces ConfigDiscoveryMixin with a registry-based approach.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class RegistryIntegrationMixin:
    """
    Mixin that adds registry-based discovery capabilities to configuration manager.

    This is a clean replacement for ConfigDiscoveryMixin that uses the new registry system.
    """

    def __init__(self):
        # Cache for discovered models per provider
        self._registry_cache: dict[str, set[str]] = {}  # provider -> {model_names}
        self._registry_cache_time: dict[str, float] = {}  # provider -> timestamp
        self._cache_ttl = 300  # 5 minutes default

    async def _get_registry_models(self, provider_name: str, force_refresh: bool = False) -> set[str]:
        """
        Get models from registry for a provider.

        Args:
            provider_name: Name of the provider
            force_refresh: Force refresh of cache

        Returns:
            Set of model names
        """
        import time

        # Check cache first
        if not force_refresh and provider_name in self._registry_cache:
            cache_age = time.time() - self._registry_cache_time.get(provider_name, 0)
            if cache_age < self._cache_ttl:
                logger.debug(f"Using cached registry models for {provider_name}")
                return self._registry_cache[provider_name]

        try:
            from chuk_llm.api.discovery import discover_models

            # Discover models using registry
            models_list = await discover_models(provider_name, force_refresh=force_refresh)

            if models_list:
                model_names = {m["name"] for m in models_list}
                self._registry_cache[provider_name] = model_names
                self._registry_cache_time[provider_name] = time.time()
                logger.debug(f"Discovered {len(model_names)} models for {provider_name}")
                return model_names

            return set()

        except Exception as e:
            logger.debug(f"Registry discovery failed for {provider_name}: {e}")
            return set()

    def get_discovered_models(self, provider_name: str) -> set[str]:
        """
        Get discovered models for a provider (sync version).

        Args:
            provider_name: Name of the provider

        Returns:
            Set of discovered model names
        """
        # Try to get from cache first
        if provider_name in self._registry_cache:
            return self._registry_cache[provider_name].copy()

        # Run async discovery in sync context
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._get_registry_models(provider_name))
            finally:
                loop.close()
        except Exception as e:
            logger.debug(f"Failed to get discovered models for {provider_name}: {e}")
            return set()

    def get_all_available_models(self, provider_name: str) -> set[str]:
        """
        Get all available models for a provider (static + discovered).

        Args:
            provider_name: Name of the provider

        Returns:
            Set of all model names
        """
        if not hasattr(self, "providers"):
            return set()

        provider = self.providers.get(provider_name)
        if not provider:
            return set()

        # Start with static models
        all_models = set(provider.models)

        # Add discovered models from registry
        discovered = self.get_discovered_models(provider_name)
        all_models.update(discovered)

        return all_models

    def _is_model_available(self, provider_name: str, model_name: str) -> bool:
        """
        Check if model is available (static OR discovered).

        Args:
            provider_name: Name of the provider
            model_name: Name of the model

        Returns:
            True if model is available
        """
        if not model_name:
            return False

        if not hasattr(self, "providers"):
            return False

        provider = self.providers.get(provider_name)
        if not provider:
            return False

        # Check static models first
        resolved_model = provider.model_aliases.get(model_name, model_name)
        if resolved_model in provider.models:
            return True

        # Check :latest variants in static models
        if not model_name.endswith(":latest"):
            latest_variant = f"{model_name}:latest"
            resolved_latest = provider.model_aliases.get(latest_variant, latest_variant)
            if resolved_latest in provider.models:
                return True
        else:
            base_variant = model_name.replace(":latest", "")
            resolved_base = provider.model_aliases.get(base_variant, base_variant)
            if resolved_base in provider.models:
                return True

        # Check discovered models from registry
        discovered_models = self.get_discovered_models(provider_name)

        # Direct match
        if model_name in discovered_models:
            return True

        # Check :latest variants in discovered models
        if not model_name.endswith(":latest"):
            if f"{model_name}:latest" in discovered_models:
                return True
        else:
            base_name = model_name.replace(":latest", "")
            if base_name in discovered_models:
                return True

        return False

    def _ensure_model_available(
        self, provider_name: str, model_name: str | None
    ) -> str | None:
        """
        Ensure model is available, trigger registry discovery if needed.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model

        Returns:
            Resolved model name or None if not found
        """
        if not model_name:
            return None

        # Check if model is already available (static or discovered)
        if self._is_model_available(provider_name, model_name):
            return model_name

        # Try registry discovery
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Force refresh to find new models
                discovered = loop.run_until_complete(
                    self._get_registry_models(provider_name, force_refresh=True)
                )

                # Re-check if model is now available
                if self._is_model_available(provider_name, model_name):
                    logger.debug(f"Found {model_name} via registry discovery")
                    return model_name

                # Check :latest variants
                if not model_name.endswith(":latest"):
                    latest_variant = f"{model_name}:latest"
                    if self._is_model_available(provider_name, latest_variant):
                        logger.debug(f"Found {model_name} as {latest_variant} via registry")
                        return latest_variant
                else:
                    base_variant = model_name.replace(":latest", "")
                    if self._is_model_available(provider_name, base_variant):
                        logger.debug(f"Found {model_name} as {base_variant} via registry")
                        return base_variant

                return None

            finally:
                loop.close()

        except Exception as e:
            logger.debug(f"Registry discovery error for {provider_name}/{model_name}: {e}")
            return None

    def reload(self):
        """Clear registry cache and reload settings"""
        self._registry_cache.clear()
        self._registry_cache_time.clear()

        # Call parent reload if it exists
        import contextlib

        with contextlib.suppress(AttributeError):
            super().reload()  # type: ignore[misc]
