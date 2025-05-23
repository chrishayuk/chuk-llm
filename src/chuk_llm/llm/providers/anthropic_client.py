# chuk_llm/llm/providers/anthropic_client.py
"""
Anthropic chat-completion adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps the official `anthropic` SDK and exposes an **OpenAI-style** interface
compatible with the rest of *chuk-llm*.

Key points
----------
*   Converts ChatML → Claude Messages format (tools / multimodal, …)
*   Maps Claude replies back to the common `{response, tool_calls}` schema
*   **Real Streaming** - uses Anthropic's native async streaming API
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from anthropic import AsyncAnthropic

from chuk_llm.llm.openai_style_mixin import OpenAIStyleMixin
from chuk_llm.llm.providers.base import BaseLLMClient

log = logging.getLogger(__name__)
if os.getenv("LOGLEVEL"):
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())

# ────────────────────────── helpers ──────────────────────────


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:  # noqa: D401 – util
    """Get *key* from dict **or** attribute-style object; fallback to *default*."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _parse_claude_response(resp) -> Dict[str, Any]:  # noqa: D401 – small helper
    """Convert Claude response → standard `{response, tool_calls}` dict."""
    tool_calls: List[Dict[str, Any]] = []

    for blk in getattr(resp, "content", []):
        if _safe_get(blk, "type") != "tool_use":
            continue
        tool_calls.append(
            {
                "id": _safe_get(blk, "id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": _safe_get(blk, "name"),
                    "arguments": json.dumps(_safe_get(blk, "input", {})),
                },
            }
        )

    if tool_calls:
        return {"response": None, "tool_calls": tool_calls}

    text = resp.content[0].text if getattr(resp, "content", None) else ""
    return {"response": text, "tool_calls": []}


# ─────────────────────────── client ───────────────────────────


class AnthropicLLMClient(OpenAIStyleMixin, BaseLLMClient):
    """Adapter around the *anthropic* SDK with OpenAI-style semantics."""

    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250219",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        
        # Use AsyncAnthropic for real streaming support
        kwargs: Dict[str, Any] = {"base_url": api_base} if api_base else {}
        if api_key:
            kwargs["api_key"] = api_key
        
        self.async_client = AsyncAnthropic(**kwargs)
        
        # Keep sync client for backwards compatibility if needed
        from anthropic import Anthropic
        self.client = Anthropic(**kwargs)

    # ── tool schema helpers ─────────────────────────────────

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not tools:
            return []

        converted: List[Dict[str, Any]] = []
        for entry in tools:
            fn = entry.get("function", entry)
            try:
                converted.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters") or fn.get("input_schema") or {},
                    }
                )
            except Exception as exc:  # pragma: no cover – permissive fallback
                log.debug("Tool schema error (%s) – using permissive schema", exc)
                converted.append(
                    {
                        "name": fn.get("name", f"tool_{uuid.uuid4().hex[:6]}"),
                        "description": fn.get("description", ""),
                        "input_schema": {"type": "object", "additionalProperties": True},
                    }
                )
        return converted

    @staticmethod
    def _split_for_anthropic(
        messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Separate system text & convert ChatML list to Anthropic format."""
        sys_txt: List[str] = []
        out: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                sys_txt.append(msg.get("content", ""))
                continue

            # assistant function calls → tool_use blocks
            if role == "assistant" and msg.get("tool_calls"):
                blocks = [
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"].get("arguments", "{}")),
                    }
                    for tc in msg["tool_calls"]
                ]
                out.append({"role": "assistant", "content": blocks})
                continue

            # tool response
            if role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id")
                                or msg.get("id", f"tr_{uuid.uuid4().hex[:8]}"),
                                "content": msg.get("content") or "",
                            }
                        ],
                    }
                )
                continue

            # normal / multimodal messages
            if role in {"user", "assistant"}:
                cont = msg.get("content")
                if cont is None:
                    continue
                if isinstance(cont, str):
                    msg = dict(msg)
                    msg["content"] = [{"type": "text", "text": cont}]
                out.append(msg)

        return "\n".join(sys_txt).strip(), out

    # ── main entrypoint ─────────────────────────────────────

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        FIXED: Generate a completion with real streaming support.
        
        • stream=False → returns awaitable that resolves to standardised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """

        tools = self._sanitize_tool_names(tools)
        anth_tools = self._convert_tools(tools)
        system_txt, msg_no_system = self._split_for_anthropic(messages)

        base_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msg_no_system,
            "tools": anth_tools,
            "max_tokens": max_tokens or 1024,
            **extra,
        }
        if system_txt:
            base_payload["system"] = system_txt
        if anth_tools:
            base_payload["tool_choice"] = {"type": "auto"}

        log.debug("Claude payload: %s", base_payload)

        # ––– streaming: use real async streaming -------------------------
        if stream:
            return self._stream_completion_async(base_payload)

        # ––– non-streaming: use async client ------------------------------
        return self._regular_completion(base_payload)

    async def _stream_completion_async(
        self, 
        payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        NEW: Real streaming using AsyncAnthropic.
        This provides true real-time streaming from Anthropic's API.
        """
        try:
            # Use async client for real streaming
            async with self.async_client.messages.stream(
                **payload
            ) as stream:
                
                # Handle different event types from Anthropic's stream
                async for event in stream:
                    # Text content events
                    if hasattr(event, 'type') and event.type == 'content_block_delta':
                        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                            yield {
                                "response": event.delta.text,
                                "tool_calls": []
                            }
                    
                    # Tool use events
                    elif hasattr(event, 'type') and event.type == 'content_block_start':
                        if hasattr(event, 'content_block') and event.content_block.type == 'tool_use':
                            tool_call = {
                                "id": event.content_block.id,
                                "type": "function",
                                "function": {
                                    "name": event.content_block.name,
                                    "arguments": json.dumps(getattr(event.content_block, 'input', {}))
                                }
                            }
                            yield {
                                "response": "",
                                "tool_calls": [tool_call]
                            }
        
        except Exception as e:
            log.error(f"Error in Anthropic streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _regular_completion(
        self, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming completion using async client."""
        try:
            resp = await self.async_client.messages.create(**payload)
            return _parse_claude_response(resp)
            
        except Exception as e:
            log.error(f"Error in Anthropic completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    # ── DEPRECATED: Keep old method for backwards compatibility ──────────
    async def _fake_streaming_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        ⚠️  DEPRECATED: Old fake streaming method.
        This returns the entire response as a single chunk.
        Use _stream_completion_async for real streaming.
        """
        tools = self._sanitize_tool_names(tools)
        anth_tools = self._convert_tools(tools)
        system_txt, msg_no_system = self._split_for_anthropic(messages)

        base_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msg_no_system,
            "tools": anth_tools,
            "max_tokens": max_tokens or 1024,
            **extra,
        }
        if system_txt:
            base_payload["system"] = system_txt
        if anth_tools:
            base_payload["tool_choice"] = {"type": "auto"}

        resp = await self._call_blocking(
            self.client.messages.create, stream=False, **base_payload
        )
        parsed = _parse_claude_response(resp)

        async def _single_chunk():
            yield parsed

        return _single_chunk()