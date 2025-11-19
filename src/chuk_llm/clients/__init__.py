"""
Modern Async-Native Clients
============================

New client implementations using Pydantic models and httpx.

Features:
- Type-safe with Pydantic V2
- Fast JSON with orjson/ujson
- Connection pooling with httpx
- Zero-copy streaming
- Proper async/await
"""

from .base import AsyncLLMClient
from .openai import OpenAIClient
from .anthropic import AnthropicClient
from .azure_openai import AzureOpenAIClient
from .gemini import GeminiClient
from .watsonx import WatsonxClient
from .openai_compatible import OpenAICompatibleClient
from .openai_responses import OpenAIResponsesClient
from .ollama import OllamaClient
from .mistral import MistralClient
from .groq import GroqClient
from .advantage import AdvantageClient

__all__ = [
    "AsyncLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "AzureOpenAIClient",
    "GeminiClient",
    "WatsonxClient",
    "OpenAICompatibleClient",
    "OpenAIResponsesClient",
    "OllamaClient",
    "MistralClient",
    "GroqClient",
    "AdvantageClient",
]
