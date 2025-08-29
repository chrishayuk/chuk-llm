# chuk_llm/configuration/unified_config.py
"""
Clean Unified Configuration System with FIXED Discovery Integration
================================================================

Key changes:
1. Fixed _ensure_model_available to work with discovered models
2. Added proper model resolution logic
3. Enhanced provider model lookup
"""

import logging
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml: ModuleType | None = None  # type: ignore[no-redef]

try:
    from dotenv import load_dotenv

    _dotenv_available = True
except ImportError:
    _dotenv_available = False

# Modern package resource handling
try:
    from importlib.resources import files  # type: ignore[attr-defined]

    _importlib_resources_available = True
except ImportError:
    try:
        from importlib_resources import files  # type: ignore[import-untyped,no-redef]

        _importlib_resources_available = True
    except ImportError:
        _importlib_resources_available = False

# Fallback package resource handling
try:
    import pkg_resources

    _pkg_resources_available = True
except ImportError:
    _pkg_resources_available = False

from .discovery import ConfigDiscoveryMixin
from .models import Feature, ModelCapabilities, ProviderConfig
from .validator import ConfigValidator

logger = logging.getLogger(__name__)


class UnifiedConfigManager(ConfigDiscoveryMixin):
    """
    Unified configuration manager with transparent dynamic discovery.

    Discovery is completely seamless - existing APIs work unchanged.
    All configuration comes from YAML, nothing hardcoded.
    """

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()  # Initialize discovery mixin

        self.config_path = config_path
        self.providers: dict[str, ProviderConfig] = {}
        self.global_aliases: dict[str, str] = {}
        self.global_settings: dict[str, Any] = {}
        self._loaded = False

        # Load environment variables first
        self._load_environment()

    def _load_environment(self) -> None:
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
            env_path = Path(str(env_file)).expanduser().resolve()  # type: ignore[arg-type]
            if env_path.exists():
                logger.info(f"Loading environment from {env_path}")
                load_dotenv(
                    env_path, override=False
                )  # Don't override existing env vars
                break
        else:
            logger.debug("No .env file found in standard locations")

    def _find_config_files(self) -> tuple[Path | None, Path | None]:
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
                "chuk_llm.yaml",  # REPLACES package config completely
                "providers.yaml",  # EXTENDS package config (inherits + overrides)
                "llm_config.yaml",  # EXTENDS package config
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
            package_config_path: Path | None = (
                Path(str(package_config)) if package_config else None
            )
            package_config = package_config_path  # type: ignore[assignment]

        return user_config, package_config  # type: ignore[return-value]

    def _load_yaml_files(self) -> dict[str, Any]:
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
                logger.error(
                    f"Failed to load package config from {package_config_file}: {e}"
                )

        # 2. Handle user config based on filename
        if user_config_file:
            try:
                user_config = self._load_single_yaml(user_config_file)
                if user_config:
                    filename = user_config_file.name

                    if filename == "chuk_llm.yaml":
                        # COMPLETE REPLACEMENT - ignore package config
                        logger.info(
                            "chuk_llm.yaml found - replacing package config completely"
                        )
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

    def _load_single_yaml(self, config_file: Path) -> dict[str, Any]:
        """Load a single YAML file"""
        # Handle package resource files specially
        if self._is_package_resource_path(config_file):
            return self._load_package_yaml()

        # Regular file loading
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _merge_configs(
        self, base_config: dict[str, Any], user_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user config into base config with intelligent merging"""
        merged = base_config.copy()

        for key, value in user_config.items():
            if key.startswith("__"):
                # Global sections - merge intelligently
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    # Merge global sections (like __global__, __global_aliases__)
                    merged[key].update(value)
                else:
                    # Replace if not both dicts
                    merged[key] = value
            else:
                # Provider sections
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    # Merge provider config - user overrides base
                    provider_config = merged[key].copy()

                    # Handle special cases for lists that should be extended vs replaced
                    for sub_key, sub_value in value.items():
                        if (
                            sub_key == "models"
                            and isinstance(sub_value, list)
                            and isinstance(provider_config.get(sub_key), list)
                        ):
                            # EXTEND models list (unique values only)
                            existing_models = set(provider_config[sub_key])
                            new_models = [
                                m for m in sub_value if m not in existing_models
                            ]
                            provider_config[sub_key].extend(new_models)
                        elif (
                            sub_key == "model_aliases"
                            and isinstance(sub_value, dict)
                            and isinstance(provider_config.get(sub_key), dict)
                        ):
                            # MERGE model aliases
                            provider_config[sub_key].update(sub_value)
                        elif (
                            sub_key == "features"
                            and isinstance(sub_value, list)
                            and isinstance(provider_config.get(sub_key), list)
                        ):
                            # EXTEND features list (unique values only)
                            existing_features = set(provider_config[sub_key])
                            new_features = [
                                f for f in sub_value if f not in existing_features
                            ]
                            provider_config[sub_key].extend(new_features)
                        elif (
                            sub_key == "rate_limits"
                            and isinstance(sub_value, dict)
                            and isinstance(provider_config.get(sub_key), dict)
                        ):
                            # MERGE rate limits
                            provider_config[sub_key].update(sub_value)
                        elif (
                            sub_key == "extra"
                            and isinstance(sub_value, dict)
                            and isinstance(provider_config.get(sub_key), dict)
                        ):
                            # DEEP MERGE extra config
                            provider_config[sub_key] = self._deep_merge_dict(
                                provider_config[sub_key], sub_value
                            )
                        else:
                            # REPLACE for other fields
                            provider_config[sub_key] = sub_value

                    merged[key] = provider_config
                else:
                    # Replace entire provider config if not both dicts
                    merged[key] = value

        return merged

    def _deep_merge_dict(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value

        return result

    def _get_package_config_path(self) -> str | None:
        """Get the path to the packaged chuk_llm.yaml file"""
        # Try modern importlib.resources first
        if _importlib_resources_available:
            try:
                package_files = files("chuk_llm")
                config_file = package_files / "chuk_llm.yaml"
                if config_file.is_file():
                    # Return a temporary path for importlib.resources
                    return str(config_file)
            except Exception as e:
                logger.debug(f"importlib.resources method failed: {e}")

        # Fallback to pkg_resources
        if _pkg_resources_available:
            try:
                if pkg_resources.resource_exists("chuk_llm", "chuk_llm.yaml"):
                    return pkg_resources.resource_filename("chuk_llm", "chuk_llm.yaml")
            except Exception as e:
                logger.debug(f"pkg_resources method failed: {e}")

        return None

    def _is_package_resource_path(self, config_file: Path) -> bool:
        """Check if this is a package resource path that needs special handling"""
        config_str = str(config_file)
        # Check for importlib.resources paths (they often contain special markers)
        return not config_file.exists() and (
            "chuk_llm" in config_str or "importlib" in config_str
        )

    def _load_package_yaml(self) -> dict[str, Any]:
        """Load YAML from package resources"""
        # Try modern importlib.resources first
        if _importlib_resources_available:
            try:
                package_files = files("chuk_llm")
                config_file = package_files / "chuk_llm.yaml"
                if config_file.is_file():
                    content = config_file.read_text(encoding="utf-8")
                    config = yaml.safe_load(content) or {}
                    logger.info(
                        "Loaded configuration from package resources (importlib)"
                    )
                    return config
            except Exception as e:
                logger.debug(f"importlib.resources loading failed: {e}")

        # Fallback to pkg_resources
        if _pkg_resources_available:
            try:
                content = pkg_resources.resource_string(  # type: ignore[attr-defined]
                    "chuk_llm", "chuk_llm.yaml"
                ).decode("utf-8")
                config = yaml.safe_load(content) or {}
                logger.info(
                    "Loaded configuration from package resources (pkg_resources)"
                )
                return config
            except Exception as e:
                logger.debug(f"pkg_resources loading failed: {e}")

        return {}

    def _parse_features(self, features_data: Any) -> set[Feature]:
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

    def _parse_model_capabilities(
        self, models_data: list[dict]
    ) -> list[ModelCapabilities]:
        """Parse model-specific capabilities"""
        if not models_data:
            return []

        capabilities = []
        for model_data in models_data:
            cap = ModelCapabilities(
                pattern=model_data.get("pattern", ".*"),
                features=self._parse_features(model_data.get("features", [])),
                max_context_length=model_data.get("max_context_length"),
                max_output_tokens=model_data.get("max_output_tokens"),
            )
            capabilities.append(cap)

        return capabilities

    def _load_yaml(self) -> dict[str, Any]:
        """Load YAML configuration with inheritance support"""
        return self._load_yaml_files()

    def _process_config(self, config: dict):
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
                provider.model_capabilities = self._parse_model_capabilities(
                    data["model_capabilities"]
                )

            # Inheritance
            if "inherits" in data:
                provider.inherits = data["inherits"]

            # Process extra fields INSIDE the provider loop
            known_fields = {
                "client_class",
                "api_key_env",
                "api_key_fallback_env",
                "api_base",
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

            # Extract extra fields for THIS provider
            extra_fields = {k: v for k, v in data.items() if k not in known_fields}

            # Deep merge extra fields for THIS provider
            for key, value in extra_fields.items():
                if (
                    isinstance(value, dict)
                    and key in provider.extra
                    and isinstance(provider.extra[key], dict)
                ):
                    # Deep merge dictionaries (like dynamic_discovery)
                    provider.extra[key].update(value)
                else:
                    # Replace for non-dict values
                    provider.extra[key] = value

            # Handle double nesting issue
            if "extra" in provider.extra and isinstance(provider.extra["extra"], dict):
                # Flatten double-nested extra fields
                nested_extra = provider.extra["extra"]
                del provider.extra["extra"]
                provider.extra.update(nested_extra)
                logger.debug(f"Fixed double nesting for provider {name}")

            # Debug logging for discovery config
            if "dynamic_discovery" in extra_fields:
                logger.debug(
                    f"Added discovery config to {name}: {extra_fields['dynamic_discovery']}"
                )
            elif "dynamic_discovery" in provider.extra:
                logger.debug(
                    f"Discovery config available for {name}: enabled={provider.extra['dynamic_discovery'].get('enabled')}"
                )

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

    def get_all_providers(self) -> list[str]:
        """Get all provider names"""
        self.load()
        return list(self.providers.keys())

    def get_api_key(self, provider_name: str) -> str | None:
        """Get API key for provider"""
        provider = self.get_provider(provider_name)

        # Check for runtime API key first (from dynamic registration)
        if provider.extra and "_runtime_api_key" in provider.extra:
            return provider.extra["_runtime_api_key"]

        if provider.api_key_env:
            key = os.getenv(provider.api_key_env)
            if key:
                return key

        if provider.api_key_fallback_env:
            return os.getenv(provider.api_key_fallback_env)

        return None

    def get_api_base(self, provider_name: str) -> str | None:
        """Get API base URL for provider, checking environment variables first"""
        provider = self.get_provider(provider_name)

        # Check for runtime API base first (from dynamic registration)
        if provider.extra and "_runtime_api_base" in provider.extra:
            return provider.extra["_runtime_api_base"]

        # Check for environment variable override
        if provider.extra and "api_base_env" in provider.extra:
            env_base = os.getenv(provider.extra["api_base_env"])
            if env_base:
                logger.debug(
                    f"Using API base from environment variable {provider.extra['api_base_env']}: {env_base}"
                )
                return env_base

        # Standard environment variable patterns
        # For provider "openai", check OPENAI_API_BASE, OPENAI_BASE_URL, etc.
        provider_upper = provider_name.upper()
        env_names = [
            f"{provider_upper}_API_BASE",
            f"{provider_upper}_BASE_URL",
            f"{provider_upper}_API_URL",
            f"{provider_upper}_ENDPOINT",
        ]

        for env_name in env_names:
            env_value = os.getenv(env_name)
            if env_value:
                logger.debug(
                    f"Using API base from environment variable {env_name}: {env_value}"
                )
                return env_value

        # Return configured base URL
        return provider.api_base

    def supports_feature(
        self,
        provider_name: str,
        feature: str | Feature,
        model: str | None = None,
    ) -> bool:
        """Check if provider/model supports feature"""
        provider = self.get_provider(provider_name)
        return provider.supports_feature(feature, model)

    def get_global_aliases(self) -> dict[str, str]:
        """Get global aliases configuration"""
        self.load()
        return self.global_aliases.copy()

    def get_global_settings(self) -> dict[str, Any]:
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

    # CRITICAL: Override the model resolution to use discovery
    def is_model_available(self, provider_name: str, model_name: str) -> bool:
        """Check if a model is available (static or discovered)"""
        self.load()
        return self._is_model_available(provider_name, model_name)

    def resolve_model(self, provider_name: str, model_name: str | None) -> str | None:
        """Resolve a model name, using discovery if needed"""
        self.load()
        return self._ensure_model_available(provider_name, model_name)

    def get_available_models(self, provider_name: str) -> set[str]:
        """Get all available models for provider (static + discovered)"""
        self.load()
        return self.get_all_available_models(provider_name)

    def reload(self):
        """Reload configuration"""
        self._loaded = False
        self.providers.clear()
        self.global_aliases.clear()
        self.global_settings.clear()
        super().reload()  # Clear discovery state
        self.load()

    # ─────────────── DYNAMIC PROVIDER REGISTRATION ──────────────

    def register_provider(
        self,
        name: str,
        api_base: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_base_env: str | None = None,
        models: list[str] | None = None,
        default_model: str | None = None,
        client_class: str | None = None,
        features: list[str] | None = None,
        max_context_length: int | None = None,
        max_output_tokens: int | None = None,
        inherits_from: str | None = None,
        **extra_kwargs,
    ) -> ProviderConfig:
        """
        Register a new provider dynamically at runtime.

        Args:
            name: Unique provider name
            api_base: Base URL for the API
            api_key: API key (stored in memory)
            api_key_env: Environment variable name for API key
            api_base_env: Environment variable name for base URL
            models: List of supported models
            default_model: Default model to use
            client_class: Client class path
            features: List of features
            max_context_length: Maximum context length
            max_output_tokens: Maximum output tokens
            inherits_from: Inherit from existing provider
            **extra_kwargs: Additional configuration

        Returns:
            The created ProviderConfig
        """
        self.load()  # Ensure config is loaded

        # Check if provider already exists
        if name in self.providers and not hasattr(self, "_dynamic_providers"):
            self._dynamic_providers: set[str] = set()  # type: ignore[var-annotated]

        # Start with base config or inherit from existing
        if inherits_from and inherits_from in self.providers:
            # Deep copy the inherited provider config
            base_provider = self.providers[inherits_from]
            provider = ProviderConfig(name=name)
            provider.client_class = base_provider.client_class
            provider.api_key_env = base_provider.api_key_env
            provider.api_base = base_provider.api_base
            provider.default_model = base_provider.default_model
            provider.models = (
                base_provider.models.copy() if base_provider.models else []
            )
            if hasattr(base_provider, "aliases"):  # type: ignore[attr-defined]
                provider.aliases = (  # type: ignore[attr-defined]
                    getattr(base_provider, "aliases", {}).copy()
                    if hasattr(base_provider, "aliases")
                    else {}  # type: ignore[attr-defined]
                )
            provider.rate_limits = (
                base_provider.rate_limits.copy() if base_provider.rate_limits else {}
            )
            provider.extra = base_provider.extra.copy() if base_provider.extra else {}
        else:
            provider = ProviderConfig(name=name)

        # Apply provided configuration
        if api_base:
            provider.api_base = api_base
        if api_base_env:
            # Store in extra config for environment variable checking
            if not provider.extra:
                provider.extra = {}
            provider.extra["api_base_env"] = api_base_env
        if api_key:
            # Store API key in extra config (not persisted)
            if not provider.extra:
                provider.extra = {}
            provider.extra["_runtime_api_key"] = api_key
        if api_key_env:
            provider.api_key_env = api_key_env
        if models:
            provider.models = models
        if default_model:
            provider.default_model = default_model
        if client_class:
            provider.client_class = client_class

        # Handle features
        if features:
            if not provider.extra:
                provider.extra = {}
            provider.extra["features"] = features

        # Handle context limits
        if max_context_length:
            if not provider.extra:
                provider.extra = {}
            provider.extra["max_context_length"] = max_context_length
        if max_output_tokens:
            if not provider.extra:
                provider.extra = {}
            provider.extra["max_output_tokens"] = max_output_tokens

        # Add any extra kwargs
        if extra_kwargs:
            if not provider.extra:
                provider.extra = {}
            provider.extra.update(extra_kwargs)

        # Register the provider
        self.providers[name] = provider

        # Track as dynamic provider
        if not hasattr(self, "_dynamic_providers"):
            self._dynamic_providers = set()
        self._dynamic_providers.add(name)

        logger.info(f"Registered dynamic provider: {name}")
        return provider

    def update_provider(self, name: str, **kwargs) -> ProviderConfig:
        """
        Update an existing provider's configuration.

        Args:
            name: Provider name to update
            **kwargs: Fields to update

        Returns:
            The updated ProviderConfig
        """
        self.load()

        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found")

        provider = self.providers[name]

        # Update fields
        for key, value in kwargs.items():
            if key == "api_key":
                # Special handling for API key
                if not provider.extra:
                    provider.extra = {}
                provider.extra["_runtime_api_key"] = value
            elif key == "api_base_env":
                # Special handling for API base environment variable
                if not provider.extra:
                    provider.extra = {}
                provider.extra["api_base_env"] = value
            elif hasattr(provider, key):
                setattr(provider, key, value)
            else:
                # Store in extra
                if not provider.extra:
                    provider.extra = {}
                provider.extra[key] = value

        logger.info(f"Updated provider: {name}")
        return provider

    def unregister_provider(self, name: str) -> bool:
        """
        Remove a dynamically registered provider.

        Args:
            name: Provider name to remove

        Returns:
            True if removed, False otherwise
        """
        self.load()

        # Only allow removing dynamic providers
        if not hasattr(self, "_dynamic_providers"):
            return False

        if name not in self._dynamic_providers:
            logger.warning(f"Cannot remove non-dynamic provider '{name}'")
            return False

        if name in self.providers:
            del self.providers[name]
            self._dynamic_providers.remove(name)
            logger.info(f"Unregistered dynamic provider: {name}")
            return True

        return False

    def list_dynamic_providers(self) -> list[str]:
        """
        List all dynamically registered providers.

        Returns:
            List of dynamic provider names
        """
        if not hasattr(self, "_dynamic_providers"):
            return []
        return list(self._dynamic_providers)


# ──────────────────────────── Capability Checker ─────────────────────────────
class CapabilityChecker:
    """Query helpers for provider capabilities"""

    @staticmethod
    def can_handle_request(
        provider: str,
        model: str | None = None,
        *,
        has_tools: bool = False,
        has_vision: bool = False,
        needs_streaming: bool = False,
        needs_json: bool = False,
    ) -> tuple[bool, list[str]]:
        """Check if provider/model can handle request"""
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(provider)

            problems = []
            if has_tools and not provider_config.supports_feature(Feature.TOOLS, model):
                problems.append("tools not supported")
            if has_vision and not provider_config.supports_feature(
                Feature.VISION, model
            ):
                problems.append("vision not supported")
            if needs_streaming and not provider_config.supports_feature(
                Feature.STREAMING, model
            ):
                problems.append("streaming not supported")
            if needs_json and not provider_config.supports_feature(
                Feature.JSON_MODE, model
            ):
                problems.append("JSON mode not supported")

            return len(problems) == 0, problems

        except Exception as exc:
            return False, [f"Provider not found: {exc}"]

    @staticmethod
    def get_best_provider_for_features(
        required_features: set[Feature],
        model_name: str | None = None,
        exclude: set[str] | None = None,
    ) -> str | None:
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
    def get_model_info(provider: str, model: str) -> dict[str, Any]:
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
                "rate_limits": provider_config.rate_limits,
            }
        except Exception as exc:
            return {"error": f"Failed to get model info: {exc}"}


class SingletonConfigManager(UnifiedConfigManager):
    """
    Singleton version of UnifiedConfigManager that persists runtime changes
    """

    _instance: Optional["SingletonConfigManager"] = None
    _initialized = False

    def __new__(cls, config_path: str | None = None) -> "SingletonConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str | None = None) -> None:
        # Only initialize once
        if not self.__class__._initialized:
            super().__init__(config_path)
            self.__class__._initialized = True

    def update_provider_models(
        self, provider_name: str, models: list[str], persist: bool = True
    ):
        """Update provider models and persist the change"""
        self.load()  # Ensure config is loaded

        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")

        # Update the models
        old_count = len(self.providers[provider_name].models)
        self.providers[provider_name].models = models
        new_count = len(models)

        logger.info(f"Updated {provider_name} models: {old_count} → {new_count}")

        if persist:
            # Mark as runtime-modified
            if not hasattr(self, "_runtime_modifications"):
                self._runtime_modifications = {}
            self._runtime_modifications[provider_name] = {
                "models": models,
                "timestamp": time.time(),
            }

    def get_provider(self, name: str) -> ProviderConfig:
        """Get provider configuration (preserves runtime changes)"""
        self.load()  # Ensure loaded

        if name not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(f"Unknown provider: {name}. Available: {available}")

        return self.providers[name]

    def reload(self):
        """Reload configuration but preserve runtime modifications"""
        # Save runtime modifications
        runtime_mods = getattr(self, "_runtime_modifications", {})

        # Call parent reload
        super().reload()

        # Restore runtime modifications
        for provider_name, mod_data in runtime_mods.items():
            if provider_name in self.providers:
                self.providers[provider_name].models = mod_data["models"]
                logger.info(f"Restored runtime modifications for {provider_name}")

        # Restore the modifications tracker
        self._runtime_modifications = runtime_mods

    @classmethod
    def reset_singleton(cls):
        """Reset singleton instance (for testing)"""
        cls._instance = None
        cls._initialized = False


# ──────────────────────────── REPLACE GLOBAL INSTANCE ─────────────────────────────
# Use singleton instead of regular UnifiedConfigManager
_unified_config = SingletonConfigManager()


def get_config() -> SingletonConfigManager:
    """Get global configuration manager (singleton)"""
    return _unified_config


def reset_config():
    """Reset configuration"""
    global _unified_config
    SingletonConfigManager.reset_singleton()
    _unified_config = SingletonConfigManager()


def reset_unified_config():
    """Reset unified configuration (alias for reset_config)"""
    reset_config()


# Clean aliases
ConfigManager = SingletonConfigManager


# Export clean API
__all__ = [
    "Feature",
    "ModelCapabilities",
    "ProviderConfig",
    "UnifiedConfigManager",
    "SingletonConfigManager",
    "ConfigValidator",
    "CapabilityChecker",
    "get_config",
    "reset_config",
    "reset_unified_config",
    "ConfigManager",
]
