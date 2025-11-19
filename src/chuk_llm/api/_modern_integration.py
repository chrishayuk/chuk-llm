"""
Modern Client Integration
==========================

Internal module for integrating modern Pydantic clients into the API.

This module provides clean integration of modern Pydantic clients.
No fallbacks - if a modern client is selected, it must work.
Fix forward by improving modern clients, not by falling back to legacy.

Architecture:
- Modern providers: Use Pydantic clients (type-safe, fast)
- Legacy providers: Use old clients (will be migrated)
- Clean separation: No fallbacks, no hybrid paths
"""

from __future__ import annotations

import logging
import os
from typing import Any

from chuk_llm.clients.anthropic import AnthropicClient
from chuk_llm.clients.azure_openai import AzureOpenAIClient
from chuk_llm.clients.gemini import GeminiClient
from chuk_llm.clients.openai import OpenAIClient
from chuk_llm.clients.openai_compatible import OpenAICompatibleClient
from chuk_llm.clients.watsonx import WatsonxClient
from chuk_llm.compat import completion_response_to_dict, dict_to_completion_request
from chuk_llm.core import CompletionResponse, EnvVar

logger = logging.getLogger(__name__)


def _can_use_modern_client(provider: str) -> bool:
    """
    Check if we can use a modern Pydantic client for this provider.

    Args:
        provider: Provider name

    Returns:
        True if modern client is available
    """
    provider_lower = provider.lower() if provider else ""

    # Providers with modern clients
    modern_providers = {
        "openai",
        "anthropic",
        "groq",
        "deepseek",
        "together",
        "perplexity",
        "openai_compatible",
        "mistral",  # OpenAI-compatible endpoint
        "ollama",  # OpenAI-compatible endpoint
        "azure_openai",  # Azure OpenAI with special auth
        "azure",  # Alias for azure_openai
        "advantage",  # IBM Advantage (OpenAI-compatible)
        "gemini",  # Google Gemini with custom client
        "watsonx",  # IBM Watsonx with custom client
    }

    return provider_lower in modern_providers


def _get_modern_client_for_provider(
    provider: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    **kwargs: Any,
) -> (
    OpenAIClient
    | OpenAICompatibleClient
    | AnthropicClient
    | AzureOpenAIClient
    | GeminiClient
    | WatsonxClient
):
    """
    Get a modern Pydantic client for the given provider.

    Args:
        provider: Provider name
        model: Model name
        api_key: API key
        api_base: API base URL
        **kwargs: Additional client options

    Returns:
        Modern AsyncLLMClient instance

    Raises:
        ValueError: If provider not supported or missing credentials
    """
    provider_lower = provider.lower() if provider else "openai"

    # OpenAI (actual OpenAI API)
    if provider_lower == "openai":
        # Get API key
        if not api_key:
            api_key = os.getenv(EnvVar.OPENAI_API_KEY.value)

        if not api_key:
            raise ValueError(
                "No API key provided for OpenAI. "
                "Set OPENAI_API_KEY environment variable."
            )

        # Use actual OpenAIClient (will migrate to new Responses API later)
        return OpenAIClient(
            model=model,
            api_key=api_key,
            base_url=api_base or "https://api.openai.com/v1",
            **kwargs,
        )

    # OpenAI-compatible providers (NOT actual OpenAI)
    elif provider_lower in [
        "groq",
        "deepseek",
        "together",
        "perplexity",
        "openai_compatible",
        "mistral",
        "ollama",
        "advantage",
    ]:
        # Get API key (optional for Ollama which runs locally)
        if not api_key:
            # Try provider-specific env var first
            api_key = os.getenv(f"{provider_lower.upper()}_API_KEY")
            if not api_key:
                # Fallback to OPENAI_API_KEY for compatible providers
                api_key = os.getenv(EnvVar.OPENAI_API_KEY.value)

        # Ollama doesn't require API key for local usage
        if not api_key and provider_lower not in ["ollama"]:
            raise ValueError(
                f"No API key provided for {provider}. "
                f"Set {provider_lower.upper()}_API_KEY or OPENAI_API_KEY environment variable."
            )

        # Use dummy key for Ollama if none provided (local usage)
        if not api_key and provider_lower == "ollama":
            api_key = "ollama-local"

        # Get base URL
        if not api_base:
            base_urls = {
                "groq": "https://api.groq.com/openai/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "together": "https://api.together.xyz/v1",
                "perplexity": "https://api.perplexity.ai",
                "openai": "https://api.openai.com/v1",
                "mistral": "https://api.mistral.ai/v1",
                "ollama": "http://localhost:11434/v1",
                "advantage": os.getenv(
                    "ADVANTAGE_API_BASE", "https://servicesessentials.ibm.com/apis/v3"
                ),
            }
            api_base = base_urls.get(provider_lower, "https://api.openai.com/v1")

        # Ensure api_key is not None (we validated above for non-Ollama)
        final_api_key = api_key if api_key else "not-set"

        # Use OpenAICompatibleClient (follows OpenAI v1 API format)
        return OpenAICompatibleClient(
            model=model,
            api_key=final_api_key,
            base_url=api_base,
            **kwargs,
        )

    # Azure OpenAI
    elif provider_lower in ["azure_openai", "azure"]:
        # Get Azure endpoint
        azure_endpoint = api_base or os.getenv(EnvVar.AZURE_OPENAI_ENDPOINT.value)
        if not azure_endpoint:
            raise ValueError(
                "azure_endpoint required for Azure OpenAI. "
                "Set AZURE_OPENAI_ENDPOINT environment variable or pass api_base."
            )

        # Get API key
        if not api_key:
            api_key = os.getenv(EnvVar.AZURE_OPENAI_API_KEY.value)

        if not api_key:
            raise ValueError(
                "No API key provided for Azure OpenAI. "
                "Set AZURE_OPENAI_API_KEY environment variable."
            )

        # Extract Azure-specific parameters from kwargs
        api_version = kwargs.pop("api_version", "2024-02-01")
        azure_deployment = kwargs.pop("azure_deployment", model)

        return AzureOpenAIClient(
            model=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=azure_deployment,
            **kwargs,
        )

    # Watsonx
    elif provider_lower == "watsonx":
        if not api_key:
            api_key = os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")

        if not api_key:
            raise ValueError(
                "No API key provided for Watsonx. "
                "Set WATSONX_API_KEY or IBM_CLOUD_API_KEY environment variable."
            )

        # Get Watsonx-specific parameters
        project_id = kwargs.pop("project_id", None) or os.getenv("WATSONX_PROJECT_ID")
        space_id = kwargs.pop("space_id", None) or os.getenv("WATSONX_SPACE_ID")
        watsonx_url = (
            api_base
            if api_base
            else (os.getenv("WATSONX_AI_URL") or "https://us-south.ml.cloud.ibm.com")
        )

        if not project_id and not space_id:
            raise ValueError(
                "Watsonx requires project_id or space_id. "
                "Set WATSONX_PROJECT_ID or WATSONX_SPACE_ID environment variable."
            )

        return WatsonxClient(
            model=model,
            api_key=api_key,
            project_id=project_id,
            space_id=space_id,
            watsonx_ai_url=watsonx_url,
            **kwargs,
        )

    # Gemini
    elif provider_lower == "gemini":
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "No API key provided for Gemini. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        return GeminiClient(
            model=model,
            api_key=api_key,
            **kwargs,
        )

    # Anthropic
    elif provider_lower == "anthropic":
        if not api_key:
            api_key = os.getenv(EnvVar.ANTHROPIC_API_KEY.value)

        if not api_key:
            raise ValueError(
                "No API key provided for Anthropic. "
                "Set ANTHROPIC_API_KEY environment variable."
            )

        return AnthropicClient(
            model=model,
            api_key=api_key,
            base_url=api_base or "https://api.anthropic.com/v1",
            **kwargs,
        )

    else:
        raise ValueError(
            f"Modern client not available for provider '{provider}'. "
            f"Supported: openai, anthropic, groq, deepseek, together, perplexity, mistral, ollama, azure_openai, advantage, gemini, watsonx"
        )


async def modern_client_complete(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Complete request using modern Pydantic client, return dict for compatibility.

    This function:
    1. Converts dict inputs to Pydantic models
    2. Uses modern type-safe client
    3. Converts Pydantic response back to dict

    Args:
        provider: Provider name
        model: Model name
        messages: List of message dicts (legacy format)
        tools: Optional list of tool dicts (legacy format)
        temperature: Temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        api_key: API key
        api_base: API base URL
        **kwargs: Additional parameters

    Returns:
        Dict response for backward compatibility (with 'response' and 'tool_calls' keys)
    """
    # Convert dict request to Pydantic model
    request_dict = {
        "messages": messages,
        "model": model,
        "tools": tools,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }

    request = dict_to_completion_request(request_dict, default_model=model)

    # Get modern client
    client = _get_modern_client_for_provider(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
    )

    try:
        # Use modern Pydantic client
        response: CompletionResponse = await client.complete(request)

        # Convert Pydantic response to dict
        result = completion_response_to_dict(response)

        # Add legacy "response" field for backward compatibility
        if "content" in result:
            result["response"] = result["content"]
        elif not result.get("tool_calls"):
            # No content and no tool calls - set empty response
            result["response"] = ""

        # Ensure tool_calls field exists
        if "tool_calls" not in result:
            result["tool_calls"] = []

        return result

    finally:
        # Close client
        await client.close()
