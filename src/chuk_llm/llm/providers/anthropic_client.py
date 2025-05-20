# chuk_llm/providers/anthropic_client.py
"""
Anthropic chat-completion adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps the official `anthropic` SDK and exposes an **OpenAI-style** interface
compatible with the rest of *chuk-llm*.

Key points
----------
*   Converts ChatML → Claude Messages format (tools / multimodal, …)
*   Maps Claude replies back to the common `{response, tool_calls}` schema
*   **Streaming** – for now we return a *single* async chunk (good enough for
    diagnostics) because Claude’s event stream doesn’t match the OpenAI one.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import Anthropic

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
        kwargs: Dict[str, Any] = {"base_url": api_base} if api_base else {}
        if api_key:
            kwargs["api_key"] = api_key
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

    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Dict[str, Any] | AsyncIterator[Dict[str, Any]]:
        """Generate a completion or (fake) async stream."""

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

        # ––– streaming ----------------------------------------------------
        # Claude streaming events are *not* OpenAI-compatible; instead of
        # trying to shoe-horn them into the OpenAI parser we return a simple
        # one-shot async iterator – good enough for diagnostics / basic usage.
        if stream:
            resp = await self._call_blocking(
                self.client.messages.create, stream=False, **base_payload
            )
            parsed = _parse_claude_response(resp)

            async def _single_chunk():
                yield parsed

            return _single_chunk()

        # ––– non-streaming ------------------------------------------------
        resp = await self._call_blocking(
            self.client.messages.create, stream=False, **base_payload
        )
        return _parse_claude_response(resp)
