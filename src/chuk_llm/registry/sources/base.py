"""
Base abstraction for model discovery sources.

A ModelSource discovers available models from various locations:
- Environment variables (API keys)
- Provider APIs (Ollama, OpenAI /v1/models, etc.)
- Configuration files
- User overrides
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from chuk_llm.registry.models import ModelSpec


class ModelSource(Protocol):
    """
    Protocol for model discovery sources.

    Sources discover ModelSpec objects (raw identities) from various locations.
    They don't resolve capabilities - that's the job of CapabilityResolvers.
    """

    async def discover(self) -> list[ModelSpec]:
        """
        Discover available models.

        Returns:
            List of discovered model specifications.
        """
        ...


class BaseModelSource(ABC):
    """Base implementation with common utilities."""

    @abstractmethod
    async def discover(self) -> list[ModelSpec]:
        """Discover available models."""
        ...

    def _deduplicate_specs(self, specs: list[ModelSpec]) -> list[ModelSpec]:
        """
        Remove duplicate model specs.

        Uses (provider, name) as the unique key.
        """
        seen = set()
        unique = []

        for spec in specs:
            key = (spec.provider, spec.name)
            if key not in seen:
                seen.add(key)
                unique.append(spec)

        return unique
