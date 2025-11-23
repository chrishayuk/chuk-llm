"""
llama.cpp capability resolver.

Resolves capabilities by querying the llama.cpp server's /props endpoint.
"""

from __future__ import annotations

import httpx

from chuk_llm.core.enums import Provider
from chuk_llm.registry.models import ModelCapabilities, ModelSpec, QualityTier
from chuk_llm.registry.resolvers.base import BaseCapabilityResolver


class LlamaCppCapabilityResolver(BaseCapabilityResolver):
    """
    Resolves capabilities from llama.cpp server metadata.

    Uses the /props endpoint to get server and model information.
    """

    def __init__(self, api_base: str = "http://localhost:8080", timeout: float = 5.0):
        """
        Initialize llama.cpp capability resolver.

        Args:
            api_base: llama.cpp server base URL (default: http://localhost:8080)
            timeout: Request timeout in seconds
        """
        self.api_base = api_base
        self.timeout = timeout

    async def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities:
        """
        Get capabilities from llama.cpp server.

        Args:
            spec: Model specification

        Returns:
            Model capabilities (empty if not llama.cpp or API fails)
        """
        if spec.provider != Provider.LLAMA_CPP.value:
            return self._empty_capabilities()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Get server properties
                response = await client.get(f"{self.api_base}/props")
                response.raise_for_status()

                props = response.json()
                return self._parse_server_props(props, spec)

        except (httpx.HTTPError, httpx.ConnectError, KeyError):
            # Server not available or endpoint not supported
            return self._empty_capabilities()

    def _parse_server_props(self, props: dict, spec: ModelSpec) -> ModelCapabilities:
        """
        Parse llama.cpp server properties into capabilities.

        Args:
            props: Response from /props endpoint
            spec: Model specification

        Returns:
            Model capabilities
        """
        # Extract context length from server props
        max_context = None
        if "default_generation_settings" in props:
            settings = props["default_generation_settings"]
            max_context = settings.get("n_ctx")

        # Check for tool/function calling support
        # llama.cpp supports tools via chat templates
        supports_tools = False
        if "chat_template" in props:
            template = props.get("chat_template", "")
            supports_tools = bool(
                template
                and ("tool" in template.lower() or "function" in template.lower())
            )

        # Streaming is always supported by llama.cpp
        supports_streaming = True

        # System messages typically supported if chat template exists
        supports_system_messages = "chat_template" in props

        # Determine quality tier (if we can infer from model name)
        quality_tier = self._infer_quality_tier(spec.name)

        return ModelCapabilities(
            max_context=max_context,
            supports_tools=supports_tools,
            supports_vision=False,  # Would need to check model-specific capabilities
            supports_json_mode=True,  # llama.cpp supports JSON mode via grammar
            supports_streaming=supports_streaming,
            supports_system_messages=supports_system_messages,
            known_params={
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            },
            quality_tier=quality_tier,
            source="llamacpp_props",
        )

    def _infer_quality_tier(self, model_name: str) -> QualityTier:
        """
        Infer quality tier from model name/size indicators.

        Args:
            model_name: Model name (may contain size info)

        Returns:
            Quality tier
        """
        model_lower = model_name.lower()

        # Look for size indicators in model name
        if any(s in model_lower for s in ["70b", "72b", "65b", "80b"]):
            return QualityTier.BEST
        elif any(s in model_lower for s in ["30b", "32b", "34b", "40b"]):
            return QualityTier.BALANCED
        elif any(s in model_lower for s in ["7b", "8b", "13b", "14b"]):
            return QualityTier.CHEAP

        return QualityTier.UNKNOWN
