"""
Modern API Layer
================

Type-safe API functions using Pydantic models internally.

This module provides the modern, type-safe implementation of the chuk-llm API.
External interfaces remain dict-based for backward compatibility, but all
internal processing uses Pydantic models.

Architecture:
- External: dict-based (backward compatible)
- Internal: Pydantic models (type-safe)
- Conversion: Compatibility layer at boundaries
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from chuk_llm.clients.anthropic import AnthropicClient
from chuk_llm.clients.openai import OpenAIClient
from chuk_llm.compat import completion_response_to_dict, dict_to_completion_request
from chuk_llm.core import (
    CompletionRequest,
    CompletionResponse,
    EnvVar,
    Message,
    MessageRole,
)

logger = logging.getLogger(__name__)


def get_modern_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> OpenAIClient | AnthropicClient:
    """
    Get a modern Pydantic-based client for the specified provider.

    Args:
        provider: Provider name ("openai", "anthropic", etc.)
        model: Model name (uses provider default if not specified)
        api_key: API key override
        api_base: API base URL override
        **kwargs: Additional client options

    Returns:
        Modern AsyncLLMClient instance

    Raises:
        ValueError: If provider is not supported or missing credentials
    """
    provider_lower = provider.lower()

    # OpenAI and compatible providers
    if provider_lower in ["openai", "groq", "deepseek", "together", "perplexity"]:
        # Get API key from kwargs, env, or raise
        if not api_key:
            api_key = os.getenv(EnvVar.OPENAI_API_KEY.value)
            if not api_key and provider_lower != "openai":
                # Try provider-specific env var
                api_key = os.getenv(f"{provider_lower.upper()}_API_KEY")

        if not api_key:
            raise ValueError(
                f"No API key provided for {provider}. "
                f"Set {provider_lower.upper()}_API_KEY environment variable "
                f"or pass api_key parameter."
            )

        # Get base URL
        if not api_base:
            base_urls = {
                "groq": "https://api.groq.com/openai/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "together": "https://api.together.xyz/v1",
                "perplexity": "https://api.perplexity.ai",
            }
            api_base = base_urls.get(provider_lower, "https://api.openai.com/v1")

        # Default models
        if not model:
            default_models = {
                "openai": "gpt-4o-mini",
                "groq": "llama-3.3-70b-versatile",
                "deepseek": "deepseek-chat",
                "together": "meta-llama/Llama-3-70b-chat-hf",
                "perplexity": "llama-3.1-sonar-small-128k-online",
            }
            model = default_models.get(provider_lower, "gpt-4o-mini")

        return OpenAIClient(
            model=model,
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    # Anthropic
    elif provider_lower == "anthropic":
        if not api_key:
            api_key = os.getenv(EnvVar.ANTHROPIC_API_KEY.value)

        if not api_key:
            raise ValueError(
                "No API key provided for Anthropic. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        if not model:
            model = "claude-3-5-sonnet-20241022"

        return AnthropicClient(
            model=model,
            api_key=api_key,
            base_url=api_base or "https://api.anthropic.com/v1",
            **kwargs,
        )

    else:
        raise ValueError(
            f"Provider '{provider}' not yet migrated to modern client. "
            f"Supported: openai, anthropic, groq, deepseek, together, perplexity"
        )


async def modern_ask(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> CompletionResponse:
    """
    Type-safe ask function using Pydantic models internally.

    Args:
        prompt: User prompt
        provider: Provider name
        model: Model name (optional)
        system: System message (optional)
        temperature: Temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters

    Returns:
        CompletionResponse Pydantic model
    """
    # Build messages
    messages = []
    if system:
        messages.append(Message(role=MessageRole.SYSTEM, content=system))
    messages.append(Message(role=MessageRole.USER, content=prompt))

    # Build request
    request = CompletionRequest(
        messages=messages,
        model=model or "",
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    # Get modern client
    client = get_modern_client(provider, model=model)

    try:
        # Type-safe call
        response = await client.complete(request)
        return response
    finally:
        await client.close()


async def modern_stream(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Type-safe streaming function using Pydantic models internally.

    Args:
        prompt: User prompt
        provider: Provider name
        model: Model name (optional)
        system: System message (optional)
        temperature: Temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters

    Yields:
        Content chunks as strings
    """
    # Build messages
    messages = []
    if system:
        messages.append(Message(role=MessageRole.SYSTEM, content=system))
    messages.append(Message(role=MessageRole.USER, content=prompt))

    # Build request
    request = CompletionRequest(
        messages=messages,
        model=model or "",
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        **kwargs,
    )

    # Get modern client
    client = get_modern_client(provider, model=model)

    try:
        # Type-safe streaming
        async for chunk in client.stream(request):
            if chunk.content:
                yield chunk.content
    finally:
        await client.close()


# Backward-compatible wrappers that return dicts
async def ask_dict(
    prompt: str,
    provider: str = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Backward-compatible ask that returns dict.

    Uses Pydantic internally but returns dict for compatibility.
    """
    response = await modern_ask(prompt, provider=provider, model=model, **kwargs)
    return completion_response_to_dict(response)


async def ask_with_tools_dict(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    provider: str = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Backward-compatible ask_with_tools that accepts/returns dicts.

    Uses Pydantic internally for type safety.
    """
    # Convert to Pydantic
    request = dict_to_completion_request(
        {"messages": messages, "tools": tools, "model": model or "", **kwargs},
        default_model=model or "gpt-4o-mini",
    )

    # Get modern client
    client = get_modern_client(provider, model=model)

    try:
        # Type-safe call
        response = await client.complete(request)
        # Convert back to dict
        return completion_response_to_dict(response)
    finally:
        await client.close()
