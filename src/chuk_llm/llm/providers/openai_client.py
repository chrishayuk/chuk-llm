# chuk_llm/llm/providers/openai_client.py
"""
OpenAI chat-completion adapter.
"""
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import openai

# mixins
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin

# base
from ..core.base import BaseLLMClient

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
        
        # Use AsyncOpenAI for real streaming support
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        # Keep sync client for backwards compatibility if needed
        self.client = openai.OpenAI(
            api_key=api_key, 
            base_url=api_base
        ) if api_base else openai.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #
    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Use native async streaming for real-time response.
        
        • stream=False → returns awaitable that resolves to a single normalised dict
        • stream=True  → returns async iterator directly (no buffering!)
        """
        tools = self._sanitize_tool_names(tools)

        # 1️⃣ streaming - use native async streaming (NO BUFFERING)
        if stream:
            return self._stream_completion_async(messages, tools, **kwargs)

        # 2️⃣ one-shot - use existing method
        return self._regular_completion(messages, tools, **kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Native async streaming using AsyncOpenAI.
        """
        try:
            # Make direct async call for real streaming
            response_stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
                stream=True,
                **kwargs
            )
            
            # Use the new real-time streaming method from mixin
            async for result in self._stream_from_async(response_stream):
                yield result
                
        except Exception as e:
            yield {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming completion using async client."""
        try:
            resp = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
                stream=False,
                **kwargs
            )
            return self._normalise_message(resp.choices[0].message)
            
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }