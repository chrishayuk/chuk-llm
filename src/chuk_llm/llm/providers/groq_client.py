# chuk_llm/llm/providers/groq_client.py
"""
Groq chat-completion adapter for MCP-CLI.

Features
--------
* Shares sanitising / normalising helpers via OpenAIStyleMixin.
* `create_completion(..., stream=False)`  → same dict as before.
* `create_completion(..., stream=True)`   → **async iterator** yielding
  incremental deltas with REAL streaming (no buffering).

      async for chunk in llm.create_completion(msgs, tools, stream=True):
          # chunk = {"response": "...", "tool_calls":[...]}
          ...

  Works in chat UIs for live-token updates.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from groq import AsyncGroq

# providers
from chuk_llm.llm.core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin

log = logging.getLogger(__name__)


class GroqAILLMClient(OpenAIStyleMixin, BaseLLMClient):
    """
    Adapter around `groq` SDK compatible with MCP-CLI's BaseLLMClient.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        
        # Use AsyncGroq for real streaming support
        self.async_client = AsyncGroq(
            api_key=api_key,
            base_url=api_base
        )
        
        # Keep sync client for backwards compatibility if needed
        from groq import Groq
        self.client = (
            Groq(api_key=api_key, base_url=api_base)
            if api_base else
            Groq(api_key=api_key)
        )

    # ──────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────
    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Real streaming support without buffering.
        
        • stream=False → returns awaitable that resolves to single normalised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """
        tools = self._sanitize_tool_names(tools)

        if stream:
            # Return async generator directly for real streaming
            return self._stream_completion_async(messages, tools or [], **kwargs)

        # non-streaming path
        return self._regular_completion(messages, tools or [], **kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        NEW: Real streaming using AsyncGroq without buffering.
        This provides true real-time streaming from Groq's API.
        """
        try:
            log.debug("Starting Groq streaming...")
            
            # Use async client for real streaming
            response_stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=True,
                **kwargs
            )
            
            chunk_count = 0
            # Yield chunks immediately as they arrive from Groq
            async for chunk in response_stream:
                chunk_count += 1
                
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Extract content and tool calls
                    content = delta.content or ""
                    tool_calls = getattr(delta, "tool_calls", [])
                    
                    # Only yield if we have actual content or tool calls
                    if content or tool_calls:
                        yield {
                            "response": content,
                            "tool_calls": tool_calls,
                        }
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            log.debug(f"Groq streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            log.error(f"Error in Groq streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming completion using async client."""
        try:
            resp = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=False,
                **kwargs
            )
            return self._normalise_message(resp.choices[0].message)
            
        except Exception as e:
            log.error(f"Error in Groq completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }