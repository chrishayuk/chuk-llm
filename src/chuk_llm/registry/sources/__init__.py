"""Model discovery sources."""

from chuk_llm.registry.sources.anthropic import AnthropicModelSource
from chuk_llm.registry.sources.azure_openai import AzureOpenAIModelSource
from chuk_llm.registry.sources.base import BaseModelSource, ModelSource
from chuk_llm.registry.sources.deepseek import DeepSeekModelSource
from chuk_llm.registry.sources.env import EnvProviderSource
from chuk_llm.registry.sources.gemini import GeminiModelSource
from chuk_llm.registry.sources.groq import GroqModelSource
from chuk_llm.registry.sources.mistral import MistralModelSource
from chuk_llm.registry.sources.moonshot import MoonshotModelSource
from chuk_llm.registry.sources.ollama import OllamaSource
from chuk_llm.registry.sources.openai import OpenAIModelSource
from chuk_llm.registry.sources.openai_compatible import OpenAICompatibleSource
from chuk_llm.registry.sources.openrouter import OpenRouterModelSource
from chuk_llm.registry.sources.perplexity import PerplexityModelSource
from chuk_llm.registry.sources.watsonx import WatsonxModelSource

__all__ = [
    # Base classes
    "ModelSource",
    "BaseModelSource",
    # Generic sources
    "EnvProviderSource",
    "OpenAICompatibleSource",
    # Provider-specific sources
    "OpenAIModelSource",
    "AzureOpenAIModelSource",
    "AnthropicModelSource",
    "GeminiModelSource",
    "DeepSeekModelSource",
    "MistralModelSource",
    "MoonshotModelSource",
    "GroqModelSource",
    "PerplexityModelSource",
    "OpenRouterModelSource",
    "WatsonxModelSource",
    "OllamaSource",
]
