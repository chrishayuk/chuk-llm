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
*   **Proper Parameter Filtering** - removes unsupported parameters
"""
from __future__ import annotations
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# llm
from anthropic import AsyncAnthropic

# providers
from .base import BaseLLMClient
from ._mixins import OpenAIStyleMixin

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
    """Adapter around the *anthropic* SDK with OpenAI-style semantics and proper parameter filtering."""

    # Parameters that Anthropic does NOT support
    UNSUPPORTED_PARAMS = {
        "frequency_penalty",
        "presence_penalty", 
        "stop",
        "logit_bias",
        "user",
        "n",
        "best_of",
        "top_k",  # Anthropic has top_k but it's not in the standard create API
        "seed",
        "response_format"
    }
    
    # Parameters that Anthropic DOES support
    SUPPORTED_PARAMS = {
        "temperature",
        "max_tokens", 
        "top_p",
        "stream"
    }

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

    def _filter_anthropic_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters to only include those supported by Anthropic"""
        filtered = {}
        
        for key, value in params.items():
            if key in self.SUPPORTED_PARAMS:
                # Anthropic has specific constraints
                if key == "temperature" and value > 1.0:
                    filtered[key] = 1.0  # Cap at 1.0 for Anthropic
                    log.debug(f"Capped temperature from {value} to 1.0 for Anthropic")
                else:
                    filtered[key] = value
            elif key in self.UNSUPPORTED_PARAMS:
                log.debug(f"Filtered out unsupported parameter for Anthropic: {key}={value}")
            else:
                log.warning(f"Unknown parameter for Anthropic: {key}={value}")
        
        # Anthropic requires max_tokens
        if "max_tokens" not in filtered:
            filtered["max_tokens"] = 1024
            log.debug("Added required max_tokens=1024 for Anthropic")
        
        return filtered

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
        Generate a completion with real streaming support and proper parameter filtering.
        
        • stream=False → returns awaitable that resolves to standardised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """

        tools = self._sanitize_tool_names(tools)
        anth_tools = self._convert_tools(tools)
        system_txt, msg_no_system = self._split_for_anthropic(messages)

        # Filter parameters for Anthropic compatibility
        if max_tokens:
            extra["max_tokens"] = max_tokens
        filtered_params = self._filter_anthropic_params(extra)

        base_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msg_no_system,
            "tools": anth_tools,
            **filtered_params,  # Use filtered parameters
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
        Real streaming using AsyncAnthropic.
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
        """Non-streaming completion using async client with proper error handling."""
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "supported_parameters": list(self.SUPPORTED_PARAMS),
            "unsupported_parameters": list(self.UNSUPPORTED_PARAMS),
        }