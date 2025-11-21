# chuk_llm/configuration/unified_config.py
"""
Simplified Configuration Manager using Pydantic + Registry
==========================================================

Clean, focused configuration system that:
- Uses Pydantic for validation (no manual checks)
- Uses Registry for model discovery (no manual parsing)
- Simple YAML loading (no complex inheritance)
- Easy to test and maintain
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from .models import Feature, ProviderConfig

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False

try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


class UnifiedConfigManager:
    """
    Simplified configuration manager with registry integration.

    Responsibilities:
    1. Load YAML configuration files
    2. Validate with Pydantic models
    3. Provide access to provider configs
    4. Integrate with Registry for model discovery
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to config file. If not provided,
                        searches standard locations.
        """
        self.config_path = config_path
        self.providers: dict[str, ProviderConfig] = {}
        self.global_settings: dict[str, Any] = {}
        self.global_aliases: dict[str, str] = {}

        # Track which providers are builtin (from YAML) vs dynamically registered
        self._builtin_providers: set[str] = set()

        # Registry cache for discovered models
        self._registry_cache: dict[str, set[str]] = {}
        self._registry_cache_time: dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes

        # Load environment variables
        self._load_environment()

        # Load configuration
        self._load_config()

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        if not _DOTENV_AVAILABLE:
            logger.debug("python-dotenv not available, skipping .env loading")
            return

        env_candidates = [
            Path(".env"),
            Path(".env.local"),
            Path.home() / ".chuk_llm" / ".env",
        ]

        for env_path in env_candidates:
            if env_path.exists():
                logger.info(f"Loading environment from {env_path}")
                load_dotenv(env_path, override=False)
                return

        logger.debug("No .env file found")

    def _find_config_file(self) -> Path | None:
        """Find configuration file."""
        # 1. Explicit path provided
        if self.config_path:
            path = Path(self.config_path)
            if path.exists():
                return path
            logger.warning(f"Config path specified but not found: {self.config_path}")

        # 2. Environment variable
        env_path = os.getenv("CHUK_LLM_CONFIG")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path

        # 3. Standard locations (in order of precedence)
        candidates = [
            Path("chuk_llm.yaml"),
            Path("providers.yaml"),
            Path("config/chuk_llm.yaml"),
            Path.home() / ".chuk_llm" / "config.yaml",
        ]

        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Using config file: {candidate}")
                return candidate

        # 4. Try package default
        package_config = self._get_package_config()
        if package_config:
            logger.info("Using package default configuration")
            return package_config

        logger.debug("No configuration file found, using defaults")
        return None

    def _get_package_config(self) -> Path | None:
        """Get package default config file."""
        # Try to find chuk_llm.yaml in the package
        try:
            package_dir = Path(__file__).parent.parent
            config_file = package_dir / "chuk_llm.yaml"
            if config_file.exists():
                return config_file
        except Exception as e:
            logger.debug(f"Could not locate package config: {e}")

        return None

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_file = self._find_config_file()

        if not config_file:
            logger.info("No config file found, providers will be empty")
            return

        if not _YAML_AVAILABLE:
            logger.warning("PyYAML not available, cannot load config file")
            return

        try:
            with open(config_file) as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"Empty config file: {config_file}")
                return

            self._process_config_data(data)
            logger.info(f"Loaded configuration with {len(self.providers)} providers")

        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            raise

    def _process_config_data(self, data: dict[str, Any]) -> None:
        """
        Process configuration data using Pydantic validation.

        This is where Pydantic does all the heavy lifting!
        """
        # Global settings (if present)
        if "__global__" in data or "global" in data:
            global_data = data.get("__global__") or data.get("global", {})
            self.global_settings.update(global_data)

        # Global aliases
        if "__global_aliases__" in data or "aliases" in data:
            aliases_data = data.get("__global_aliases__") or data.get("aliases", {})
            self.global_aliases.update(aliases_data)

        # Process providers
        for key, provider_data in data.items():
            if key.startswith("__") or key in ("global", "aliases"):
                continue

            try:
                # Ensure provider has a name
                if "name" not in provider_data:
                    provider_data["name"] = key

                # Parse features from strings to Feature enum
                if "features" in provider_data:
                    provider_data["features"] = self._parse_features(
                        provider_data["features"]
                    )

                # Parse model capabilities
                if "model_capabilities" in provider_data:
                    provider_data["model_capabilities"] = (
                        self._parse_model_capabilities(
                            provider_data["model_capabilities"]
                        )
                    )

                # Let Pydantic validate and create the model
                provider = ProviderConfig.model_validate(provider_data)
                self.providers[key] = provider
                # Mark as builtin (loaded from YAML)
                self._builtin_providers.add(key)

                logger.debug(f"Loaded provider: {key}")

            except Exception as e:
                logger.error(f"Failed to load provider {key}: {e}")
                raise

    def _parse_features(self, features_data: Any) -> set[Feature]:
        """Parse features from various formats to Feature enum set."""
        if not features_data:
            return set()

        features = set()

        # Handle string (single feature)
        if isinstance(features_data, str):
            try:
                features.add(Feature.from_string(features_data))
            except ValueError:
                logger.warning(f"Unknown feature: {features_data}")
            return features

        # Handle list of features
        if isinstance(features_data, list):
            for item in features_data:
                try:
                    features.add(Feature.from_string(str(item)))
                except ValueError:
                    logger.warning(f"Unknown feature: {item}")
            return features

        # Handle set
        if isinstance(features_data, set):
            for item in features_data:
                try:
                    features.add(Feature.from_string(str(item)))
                except ValueError:
                    logger.warning(f"Unknown feature: {item}")
            return features

        return features

    def _parse_model_capabilities(self, caps_data: list[dict[str, Any]]) -> list:
        """Parse model capabilities using Pydantic."""
        from .models import ModelCapabilities

        capabilities = []
        for cap_data in caps_data:
            # Parse features in capability
            if "features" in cap_data:
                cap_data["features"] = self._parse_features(cap_data["features"])

            # Let Pydantic validate
            try:
                cap = ModelCapabilities.model_validate(cap_data)
                capabilities.append(cap)
            except Exception as e:
                logger.warning(f"Failed to parse model capability: {e}")

        return capabilities

    # =============================================================================
    # Public API
    # =============================================================================

    def get_provider(self, name: str) -> ProviderConfig:
        """
        Get provider configuration.

        Args:
            name: Provider name

        Returns:
            Provider configuration

        Raises:
            ValueError: If provider not found
        """
        if name not in self.providers:
            available = ", ".join(sorted(self.providers.keys()))
            raise ValueError(f"Unknown provider: {name}. Available: {available}")
        return self.providers[name]

    def get_all_providers(self) -> list[str]:
        """Get list of all configured provider names."""
        return sorted(self.providers.keys())

    def get_global_aliases(self) -> dict[str, str]:
        """Get global model aliases."""
        return self.global_aliases.copy()

    def supports_feature(
        self, provider_name: str, feature: Feature, model: str | None = None
    ) -> bool:
        """
        Check if provider/model supports a feature.

        Args:
            provider_name: Provider name
            feature: Feature to check
            model: Optional model name

        Returns:
            True if feature is supported
        """
        try:
            provider = self.get_provider(provider_name)
            return provider.supports_feature(feature, model)
        except ValueError:
            return False

    def get_api_key(self, provider_name: str) -> str | None:
        """
        Get API key for provider from environment.

        Args:
            provider_name: Provider name

        Returns:
            API key or None if not found
        """
        try:
            provider = self.get_provider(provider_name)
            return provider.get_api_key()
        except ValueError:
            return None

    def get_api_base(self, provider_name: str) -> str | None:
        """
        Get API base URL for provider with proper priority order.

        Priority (highest to lowest):
        1. Runtime api_base (_runtime_api_base in extra)
        2. Custom api_base_env environment variable
        3. Standard environment patterns ({PROVIDER}_API_BASE, etc.)
        4. Configured api_base value

        Args:
            provider_name: Provider name

        Returns:
            API base URL or None
        """
        try:
            provider = self.get_provider(provider_name)

            # Highest priority: Runtime API base
            if "_runtime_api_base" in provider.extra:
                return provider.extra["_runtime_api_base"]

            # High priority: Custom api_base_env
            if provider.api_base_env:
                env_value = os.getenv(provider.api_base_env)
                if env_value:
                    return env_value

            # Medium priority: Standard environment variable patterns
            provider_upper = provider_name.upper()
            standard_patterns = [
                f"{provider_upper}_API_BASE",
                f"{provider_upper}_BASE_URL",
                f"{provider_upper}_API_URL",
                f"{provider_upper}_ENDPOINT",
            ]
            for env_var in standard_patterns:
                env_value = os.getenv(env_var)
                if env_value:
                    return env_value

            # Lowest priority: Configured api_base
            if provider.api_base:
                return provider.api_base

            return None
        except ValueError:
            return None

    def reload(self) -> None:
        """Reload configuration from disk."""
        # Clear current state
        self.providers.clear()
        self.global_settings.clear()
        self.global_aliases.clear()

        # Clear registry cache
        self._registry_cache.clear()
        self._registry_cache_time.clear()

        # Reload config
        self._load_config()

    # =============================================================================
    # Dynamic Provider Registration (for runtime provider addition)
    # =============================================================================

    def register_provider(
        self,
        name: str,
        client_class: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base: str | None = None,
        default_model: str | None = None,
        models: list[str] | None = None,
        features: list[str] | set[Feature] | None = None,
        **extra_kwargs,
    ) -> ProviderConfig:
        """
        Register a provider dynamically at runtime.

        Args:
            name: Provider name
            client_class: Client class path (defaults to OpenAI-compatible)
            api_key: Direct API key (not recommended, use api_key_env)
            api_key_env: Environment variable name for API key
            api_base: Base URL for API
            default_model: Default model name
            models: List of supported models
            features: List of supported features
            **extra_kwargs: Additional provider-specific configuration

        Returns:
            The created ProviderConfig object
        """
        # Default client class to OpenAI-compatible
        if not client_class:
            client_class = "chuk_llm.llm.providers.openai_client:OpenAILLMClient"

        # Parse features if provided
        if features:
            if isinstance(features, list):
                features = self._parse_features(features)
            elif not isinstance(features, set):
                features = {features}
        else:
            features = set()

        # Separate known fields from extra kwargs
        known_fields = {
            "name",
            "client_class",
            "api_key_env",
            "api_key_fallback_env",
            "api_base",
            "api_base_env",
            "default_model",
            "models",
            "model_aliases",
            "features",
            "max_context_length",
            "max_output_tokens",
            "rate_limits",
            "model_capabilities",
            "inherits",
        }

        # Build provider data with known fields
        provider_data = {
            "name": name,
            "client_class": client_class,
            "api_key_env": api_key_env,
            "api_base": api_base,
            "default_model": default_model or "",
            "models": models or ["*"],
            "features": features,
        }

        # Separate known kwargs from extra
        extra_data = {}
        for key, value in extra_kwargs.items():
            if key in known_fields:
                provider_data[key] = value
            else:
                extra_data[key] = value

        # Store extra kwargs in the extra field
        if extra_data:
            provider_data["extra"] = extra_data  # type: ignore[assignment]

        # Handle api_key specially - store in extra as _runtime_api_key
        if api_key:
            if "extra" not in provider_data:
                provider_data["extra"] = {}  # type: ignore[assignment]
            provider_data["extra"]["_runtime_api_key"] = api_key  # type: ignore[index,call-overload]

        # Validate and create provider
        provider = ProviderConfig.model_validate(provider_data)
        self.providers[name] = provider

        logger.info(f"Registered dynamic provider: {name}")

        return provider

    def update_provider(self, name: str, **kwargs) -> ProviderConfig:
        """
        Update an existing provider's configuration.

        Args:
            name: Provider name to update
            **kwargs: Fields to update

        Returns:
            Updated ProviderConfig

        Raises:
            ValueError: If provider doesn't exist
        """
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found")

        # Get existing provider
        existing = self.providers[name]

        # Convert to dict and update
        provider_dict = existing.model_dump()

        # Handle features specially
        if "features" in kwargs:
            features = kwargs["features"]
            if isinstance(features, list):
                kwargs["features"] = self._parse_features(features)
            elif not isinstance(features, set):
                kwargs["features"] = {features}

        # Separate known fields from those that should go in extra
        known_fields = {
            "name",
            "client_class",
            "api_key_env",
            "api_key_fallback_env",
            "api_base",
            "api_base_env",
            "default_model",
            "models",
            "model_aliases",
            "features",
            "max_context_length",
            "max_output_tokens",
            "rate_limits",
            "model_capabilities",
            "inherits",
        }

        # Fields that should go in extra dict (for legacy/special cases)
        extra_fields = {"inherits_from"}

        # Update known fields directly
        for key, value in kwargs.items():
            if key in known_fields:
                provider_dict[key] = value
            elif key in extra_fields:
                # Put in extra dict
                if "extra" not in provider_dict:
                    provider_dict["extra"] = {}
                provider_dict["extra"][key] = value
            else:
                # Other unknown fields - let Pydantic handle them
                provider_dict[key] = value

        # Validate and recreate
        updated_provider = ProviderConfig.model_validate(provider_dict)
        self.providers[name] = updated_provider

        logger.info(f"Updated provider: {name}")

        return updated_provider

    def unregister_provider(self, name: str) -> bool:
        """
        Unregister a dynamically registered provider.

        Args:
            name: Provider name to remove

        Returns:
            True if successfully unregistered

        Raises:
            ValueError: If provider doesn't exist or is a builtin provider
        """
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found")

        # Cannot unregister builtin providers
        if name in self._builtin_providers:
            raise ValueError(f"Cannot unregister builtin provider '{name}'")

        del self.providers[name]

        # Clear from registry cache if present
        if name in self._registry_cache:
            del self._registry_cache[name]
        if name in self._registry_cache_time:
            del self._registry_cache_time[name]

        logger.info(f"Unregistered provider: {name}")

        return True

    def list_dynamic_providers(self) -> list[str]:
        """
        List dynamically registered providers (excludes builtin providers from YAML).

        Returns:
            List of dynamically registered provider names (not from YAML config)
        """
        return [name for name in self.providers if name not in self._builtin_providers]

    def provider_exists(self, name: str) -> bool:
        """
        Check if a provider exists.

        Args:
            name: Provider name

        Returns:
            True if provider exists
        """
        return name in self.providers

    def _ensure_model_available(self, provider_name: str, model: str) -> bool:
        """
        Check if a model is available for a provider.

        Args:
            provider_name: Provider name
            model: Model name

        Returns:
            True if model is available
        """
        try:
            provider = self.get_provider(provider_name)

            # Check static models
            if model in provider.models or "*" in provider.models:
                return True

            # Check discovered models
            discovered = self.get_discovered_models(provider_name)
            if model in discovered:
                return True

            return False
        except ValueError:
            return False

    # =============================================================================
    # Registry Integration for Model Discovery
    # =============================================================================

    async def _get_registry_models(
        self, provider_name: str, force_refresh: bool = False
    ) -> set[str]:
        """
        Get models from registry for a provider.

        Args:
            provider_name: Name of the provider
            force_refresh: Force refresh of cache

        Returns:
            Set of model names
        """
        # Check cache first
        if not force_refresh and provider_name in self._registry_cache:
            cache_age = time.time() - self._registry_cache_time.get(provider_name, 0)
            if cache_age < self._cache_ttl:
                logger.debug(f"Using cached registry models for {provider_name}")
                return self._registry_cache[provider_name]

        try:
            from chuk_llm.api.discovery import discover_models

            # Discover models using registry
            models_list = await discover_models(
                provider_name, force_refresh=force_refresh
            )

            if models_list:
                model_names = {m["name"] for m in models_list}
                self._registry_cache[provider_name] = model_names
                self._registry_cache_time[provider_name] = time.time()
                logger.debug(
                    f"Discovered {len(model_names)} models for {provider_name}"
                )
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
        provider = self.providers.get(provider_name)
        if not provider:
            return set()

        # Start with static models
        all_models = set(provider.models)

        # Add discovered models from registry
        discovered = self.get_discovered_models(provider_name)
        all_models.update(discovered)

        return all_models


# =============================================================================
# Capability Checker Helper
# =============================================================================


class CapabilityChecker:
    """Helper class for checking provider/model capabilities."""

    @staticmethod
    def can_handle_request(
        provider_name: str,
        model: str | None = None,
        features: list[Feature] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if provider/model can handle a request.

        Args:
            provider_name: Provider name
            model: Optional model name
            features: Optional list of required features

        Returns:
            (can_handle, issues) tuple
        """
        config = get_config()
        issues = []

        try:
            provider = config.get_provider(provider_name)

            if features:
                for feature in features:
                    if not provider.supports_feature(feature, model):
                        issues.append(
                            f"{provider_name}/{model or 'default'} "
                            f"doesn't support {feature.value}"
                        )

        except ValueError as e:
            issues.append(str(e))

        return len(issues) == 0, issues

    @staticmethod
    def get_best_provider_for_features(
        features: list[Feature], model: str | None = None
    ) -> str | None:
        """
        Find best provider that supports all required features.

        Args:
            features: Required features
            model: Optional model constraint

        Returns:
            Provider name or None if no match
        """
        config = get_config()

        for provider_name in config.get_all_providers():
            can_handle, _ = CapabilityChecker.can_handle_request(
                provider_name, model, features
            )
            if can_handle:
                return provider_name

        return None

    @staticmethod
    def get_model_info(provider_name: str, model: str) -> dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            provider_name: Provider name
            model: Model name

        Returns:
            Dictionary with model information
        """
        config = get_config()

        try:
            provider = config.get_provider(provider_name)
            capabilities = provider.get_model_capabilities(model)

            return {
                "provider": provider_name,
                "model": model,
                "features": [f.value for f in capabilities.features],
                "max_context_length": capabilities.max_context_length,
                "max_output_tokens": capabilities.max_output_tokens,
            }

        except ValueError as e:
            return {"error": str(e)}


# =============================================================================
# Global Singleton
# =============================================================================

_global_config: UnifiedConfigManager | None = None


def get_config(config_path: str | None = None) -> UnifiedConfigManager:
    """
    Get global configuration manager instance.

    Args:
        config_path: Optional path to config file (only used on first call)

    Returns:
        Global configuration manager
    """
    global _global_config

    if _global_config is None or config_path:
        _global_config = UnifiedConfigManager(config_path)

    return _global_config


def reset_config() -> None:
    """Reset global configuration (mainly for testing)."""
    global _global_config
    _global_config = None


# Alias for backward compatibility
reset_unified_config = reset_config
ConfigManager = UnifiedConfigManager
