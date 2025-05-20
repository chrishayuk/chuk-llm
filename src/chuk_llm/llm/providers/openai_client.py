# chuk_llm/providers/openai_client.py
"""
OpenAI chat-completion adapter.
"""
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional
from openai import OpenAI

# mixins
from chuk_llm.llm.openai_style_mixin import OpenAIStyleMixin

# base
from .base import BaseLLMClient


class OpenAILLMClient(OpenAIStyleMixin, BaseLLMClient):
    """
    Thin wrapper around the official `openai` SDK.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        self.client = (
            OpenAI(api_key=api_key, base_url=api_base)
            if api_base else
            OpenAI(api_key=api_key)
        )

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
    ) -> Dict[str, Any] | AsyncIterator[Dict[str, Any]]:
        """
        • stream=False → returns a single normalised dict
        • stream=True  → returns an async iterator yielding MCP-delta dicts
        """
        tools = self._sanitize_tool_names(tools)

        # 1️⃣ streaming
        if stream:
            return self._stream_from_blocking(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                tools=tools or [],
            )

        # 2️⃣ one-shot
        resp = await self._call_blocking(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            tools=tools or [],
        )
        return self._normalise_message(resp.choices[0].message)
