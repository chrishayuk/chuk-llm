"""Model discovery sources."""

from chuk_llm.registry.sources.base import BaseModelSource, ModelSource
from chuk_llm.registry.sources.env import EnvProviderSource
from chuk_llm.registry.sources.ollama import OllamaSource

__all__ = [
    "ModelSource",
    "BaseModelSource",
    "EnvProviderSource",
    "OllamaSource",
]
