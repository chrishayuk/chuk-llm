"""
Dynamic Provider Registration API
==================================

Public API for registering and managing LLM providers at runtime.
"""

import logging
import os

from chuk_llm.configuration.models import ProviderConfig
from chuk_llm.configuration.unified_config import get_config

logger = logging.getLogger(__name__)


def register_provider(
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
    Register a new LLM provider dynamically at runtime.

    This allows you to add providers without modifying configuration files.
    The provider will be immediately available for use with ask(), stream(), etc.

    Args:
        name: Unique provider name
        api_base: Base URL for the API (e.g., "https://api.example.com/v1")
        api_key: API key (stored in memory only, not persisted)
        api_key_env: Environment variable name for API key (e.g., "MY_API_KEY")
        api_base_env: Environment variable name for base URL
        models: List of supported models
        default_model: Default model to use
        client_class: Client class path (defaults to OpenAI-compatible)
        features: List of features ["streaming", "tools", "vision", etc.]
        max_context_length: Maximum context length
        max_output_tokens: Maximum output tokens
        inherits_from: Inherit configuration from existing provider
        **extra_kwargs: Additional provider-specific configuration

    Returns:
        The created ProviderConfig object

    Examples:
        # Register a simple OpenAI-compatible provider
        register_provider(
            "my_service",
            api_base="https://api.myservice.com/v1",
            api_key="sk-abc123",
            models=["model-a", "model-b"],
            default_model="model-a"
        )

        # Register with environment variables
        register_provider(
            "custom_llm",
            api_key_env="CUSTOM_API_KEY",
            api_base_env="CUSTOM_API_BASE",
            models=["gpt-3.5-turbo", "gpt-4"]
        )

        # Inherit from existing provider
        register_provider(
            "my_openai",
            inherits_from="openai",
            api_base="https://proxy.company.com/v1",
            api_key="company-key"
        )

        # Use the provider
        from chuk_llm import ask_sync
        response = ask_sync("Hello!", provider="my_service")
    """
    config = get_config()
    return config.register_provider(
        name=name,
        api_base=api_base,
        api_key=api_key,
        api_key_env=api_key_env,
        api_base_env=api_base_env,
        models=models,
        default_model=default_model,
        client_class=client_class,
        features=features,
        max_context_length=max_context_length,
        max_output_tokens=max_output_tokens,
        inherits_from=inherits_from,
        **extra_kwargs,
    )


def update_provider(name: str, **kwargs) -> ProviderConfig:
    """
    Update an existing provider's configuration at runtime.

    Args:
        name: Provider name to update
        **kwargs: Fields to update (api_base, models, features, etc.)

    Returns:
        The updated ProviderConfig

    Example:
        update_provider(
            "my_service",
            api_base="https://new-endpoint.com/v1",
            models=["model-a", "model-b", "model-c"]
        )
    """
    config = get_config()
    return config.update_provider(name, **kwargs)


def unregister_provider(name: str) -> bool:
    """
    Remove a dynamically registered provider.

    Note: Only providers registered at runtime can be removed.
    Providers from configuration files cannot be unregistered.

    Args:
        name: Provider name to remove

    Returns:
        True if removed, False if not found or not removable

    Example:
        success = unregister_provider("my_service")
    """
    config = get_config()
    return config.unregister_provider(name)


def list_dynamic_providers() -> list[str]:
    """
    List all providers that were registered dynamically at runtime.

    Returns:
        List of dynamically registered provider names

    Example:
        providers = list_dynamic_providers()
        # ['my_service', 'custom_llm']
    """
    config = get_config()
    return config.list_dynamic_providers()


def get_provider_config(name: str) -> ProviderConfig:
    """
    Get the configuration for a specific provider.

    Args:
        name: Provider name

    Returns:
        ProviderConfig object

    Example:
        config = get_provider_config("my_service")
        print(config.api_base)
        print(config.models)
    """
    config = get_config()
    return config.get_provider(name)


def provider_exists(name: str) -> bool:
    """
    Check if a provider exists (either from config or dynamically registered).

    Args:
        name: Provider name to check

    Returns:
        True if provider exists

    Example:
        if provider_exists("my_service"):
            response = ask_sync("Hello", provider="my_service")
    """
    config = get_config()
    try:
        config.get_provider(name)
        return True
    except Exception:
        return False


def register_openai_compatible(
    name: str,
    api_base: str | None = None,
    api_key: str | None = None,
    api_base_env: str | None = None,
    api_key_env: str | None = None,
    models: list[str] | None = None,
    **kwargs,
) -> ProviderConfig:
    """
    Convenience function to register an OpenAI-compatible provider.

    This is a shortcut for the common case of registering services that
    implement the OpenAI API (LocalAI, FastChat, vLLM, etc.).

    Args:
        name: Provider name
        api_base: API endpoint URL (optional if api_base_env is provided)
        api_key: API key (optional)
        api_base_env: Environment variable name for API base URL
        api_key_env: Environment variable name for API key
        models: List of available models
        **kwargs: Additional configuration

    Returns:
        The created ProviderConfig

    Examples:
        # Register LocalAI
        register_openai_compatible(
            "localai",
            api_base="http://localhost:8080/v1",
            models=["llama", "mistral", "phi"]
        )

        # Register vLLM server
        register_openai_compatible(
            "my_vllm",
            api_base="http://gpu-server:8000/v1",
            models=["meta-llama/Llama-3-70b-hf"]
        )

        # Register with environment variables
        register_openai_compatible(
            "proxy_openai",
            api_base_env="PROXY_ENDPOINT",
            api_key_env="PROXY_KEY",
            models=["gpt-3.5-turbo", "gpt-4"]
        )
    """
    # If neither api_base nor api_base_env is provided, check if it's in kwargs
    if not api_base and not api_base_env and "api_base_env" not in kwargs:
        # Try to get from environment using standard patterns
        provider_upper = name.upper()
        for env_name in [f"{provider_upper}_API_BASE", f"{provider_upper}_BASE_URL"]:
            if os.getenv(env_name):
                api_base_env = env_name
                break

    # Set default model if not in kwargs
    if "default_model" not in kwargs:
        if models and len(models) > 0:
            kwargs["default_model"] = models[0]
        else:
            kwargs["default_model"] = "gpt-3.5-turbo"

    return register_provider(
        name=name,
        api_base=api_base,
        api_base_env=api_base_env,
        api_key=api_key,
        api_key_env=api_key_env,
        models=models or ["gpt-3.5-turbo"],
        client_class="chuk_llm.llm.providers.openai_client.OpenAILLMClient",
        features=["text", "streaming", "system_messages", "tools", "json_mode"],
        **kwargs,
    )
