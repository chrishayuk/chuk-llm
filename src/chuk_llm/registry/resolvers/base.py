"""
Base abstraction for capability resolvers.

A CapabilityResolver enriches ModelSpec objects with capability metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from chuk_llm.registry.models import ModelCapabilities, ModelSpec


class CapabilityResolver(Protocol):
    """
    Protocol for capability resolvers.

    Resolvers take a ModelSpec (raw identity) and return ModelCapabilities
    (enriched metadata). Resolvers are layered - later resolvers override
    earlier ones.
    """

    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """
        Get capabilities for a model.

        Args:
            spec: Model specification

        Returns:
            Model capabilities (may be partial/empty if resolver has no info)
        """
        ...


class BaseCapabilityResolver(ABC):
    """Base implementation with common utilities."""

    @abstractmethod
    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """Get capabilities for a model."""
        ...

    def _empty_capabilities(self) -> ModelCapabilities:
        """Return an empty capabilities object."""
        return ModelCapabilities()

    def _partial_capabilities(self, **kwargs) -> ModelCapabilities:
        """Create a partial capabilities object."""
        return ModelCapabilities(**kwargs)
