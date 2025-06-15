# chuk_llm/llm/providers/watsonx_client.py
"""
Watson X chat-completion adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps the official `ibm-watsonx-ai` SDK and exposes an **OpenAI-style** interface
compatible with the rest of *chuk-llm*.

Key points
----------
*   Converts ChatML → Watson X Messages format (tools / multimodal, …)
*   Maps Watson X replies back to the common `{response, tool_calls}` schema
*   **Real Streaming** - uses Watson X's native streaming API
"""
from __future__ import annotations
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# llm
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

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


def _parse_watsonx_response(resp) -> Dict[str, Any]:  # noqa: D401 – small helper
    """Convert Watson X response → standard `{response, tool_calls}` dict."""
    tool_calls: List[Dict[str, Any]] = []
    
    # Handle Watson X response format - check choices first
    if hasattr(resp, 'choices') and resp.choices:
        choice = resp.choices[0]
        message = _safe_get(choice, 'message', {})
        
        # Check for tool calls in Watson X format
        if _safe_get(message, 'tool_calls'):
            for tc in message['tool_calls']:
                tool_calls.append({
                    "id": _safe_get(tc, "id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": _safe_get(tc, "function", {}).get("name"),
                        "arguments": _safe_get(tc, "function", {}).get("arguments", "{}"),
                    },
                })
        
        if tool_calls:
            return {"response": None, "tool_calls": tool_calls}
        
        # Extract text content
        content = _safe_get(message, "content", "")
        if isinstance(content, list) and content:
            content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        
        return {"response": content, "tool_calls": []}
    
    # Fallback: try direct dictionary access
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            message = choice.get("message", {})
            
            # Check for tool calls
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        },
                    })
                
                if tool_calls:
                    return {"response": None, "tool_calls": tool_calls}
            
            # Extract text content
            content = message.get("content", "")
            return {"response": content, "tool_calls": []}
    
    # Fallback for other response formats
    if hasattr(resp, 'results') and resp.results:
        result = resp.results[0]
        text = _safe_get(result, 'generated_text', '') or _safe_get(result, 'text', '')
        return {"response": text, "tool_calls": []}
    
    return {"response": str(resp), "tool_calls": []}


# ─────────────────────────── client ───────────────────────────


class WatsonXLLMClient(OpenAIStyleMixin, BaseLLMClient):
    """Adapter around the *ibm-watsonx-ai* SDK with OpenAI-style semantics."""

    def __init__(
        self,
        model: str = "meta-llama/llama-3-8b-instruct",
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        watsonx_ai_url: Optional[str] = None,
        space_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.space_id = space_id or os.getenv("WATSONX_SPACE_ID")
        
        # Set up credentials
        credentials = Credentials(
            url=watsonx_ai_url or os.getenv("WATSONX_AI_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=api_key or os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")
        )
        
        self.client = APIClient(credentials)
        
        # Default parameters
        self.default_params = {
            "time_limit": 10000,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 1.0,
        }

    def _get_model_inference(self, params: Optional[Dict[str, Any]] = None) -> ModelInference:
        """Create a ModelInference instance with the given parameters."""
        merged_params = {**self.default_params}
        if params:
            merged_params.update(params)
        
        return ModelInference(
            model_id=self.model,
            api_client=self.client,
            params=merged_params,
            project_id=self.project_id,
            space_id=self.space_id,
            verify=False
        )

    # ── tool schema helpers ─────────────────────────────────

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Watson X format."""
        if not tools:
            return []

        converted: List[Dict[str, Any]] = []
        for entry in tools:
            fn = entry.get("function", entry)
            try:
                converted.append({
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or fn.get("input_schema") or {},
                    }
                })
            except Exception as exc:  # pragma: no cover – permissive fallback
                log.debug("Tool schema error (%s) – using permissive schema", exc)
                converted.append({
                    "type": "function",
                    "function": {
                        "name": fn.get("name", f"tool_{uuid.uuid4().hex[:6]}"),
                        "description": fn.get("description", ""),
                        "parameters": {"type": "object", "additionalProperties": True},
                    }
                })
        return converted

    @staticmethod
    def _format_messages_for_watsonx(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages for Watson X API."""
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                formatted.append({
                    "role": "system",
                    "content": content
                })
            elif role == "user":
                if isinstance(content, str):
                    formatted.append({
                        "role": "user",
                        "content": [{"type": "text", "text": content}]
                    })
                elif isinstance(content, list):
                    # Handle multimodal content for Watson X
                    watsonx_content = []
                    for item in content:
                        if item.get("type") == "text":
                            watsonx_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") == "image":
                            # Convert image format for Watson X
                            source = item.get("source", {})
                            if source.get("type") == "base64":
                                # Watson X expects image_url format
                                data_url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                watsonx_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                })
                    formatted.append({
                        "role": "user",
                        "content": watsonx_content
                    })
                else:
                    formatted.append({
                        "role": "user",
                        "content": content
                    })
            elif role == "assistant":
                if msg.get("tool_calls"):
                    formatted.append({
                        "role": "assistant",
                        "tool_calls": msg["tool_calls"]
                    })
                else:
                    formatted.append({
                        "role": "assistant",
                        "content": content
                    })
            elif role == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id"),
                    "content": content
                })

        return formatted

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
        Generate a completion with streaming support.
        
        • stream=False → returns awaitable that resolves to standardised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """
        tools = self._sanitize_tool_names(tools)
        watsonx_tools = self._convert_tools(tools)
        formatted_messages = self._format_messages_for_watsonx(messages)

        # Update parameters
        params = dict(self.default_params)
        if max_tokens:
            params["max_tokens"] = max_tokens
        params.update(extra)

        log.debug("Watson X payload: messages=%s, tools=%s", formatted_messages, watsonx_tools)

        # ––– streaming: use Watson X streaming -------------------------
        if stream:
            return self._stream_completion_async(formatted_messages, watsonx_tools, params)

        # ––– non-streaming: use regular completion ----------------------
        return self._regular_completion(formatted_messages, watsonx_tools, params)

    async def _stream_completion_async(
        self, 
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming using Watson X.
        """
        try:
            model = self._get_model_inference(params)
            
            # Use Watson X streaming
            if tools:
                # For tool calling, we need to use chat_stream with tools
                stream_response = model.chat_stream(messages=messages, tools=tools)
            else:
                # For regular chat, use chat_stream
                stream_response = model.chat_stream(messages=messages)
            
            for chunk in stream_response:
                if isinstance(chunk, str):
                    yield {
                        "response": chunk,
                        "tool_calls": []
                    }
                elif isinstance(chunk, dict):
                    # Handle structured chunk responses
                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        
                        content = delta.get("content", "")
                        tool_calls = delta.get("tool_calls", [])
                        
                        yield {
                            "response": content,
                            "tool_calls": tool_calls
                        }
                    else:
                        yield {
                            "response": str(chunk),
                            "tool_calls": []
                        }
        
        except Exception as e:
            log.error(f"Error in Watson X streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _regular_completion(
        self, 
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming completion using Watson X."""
        try:
            model = self._get_model_inference(params)
            
            if tools:
                # Use chat with tools
                resp = model.chat(messages=messages, tools=tools)
            else:
                # Use regular chat
                resp = model.chat(messages=messages)
            
            return _parse_watsonx_response(resp)
            
        except Exception as e:
            log.error(f"Error in Watson X completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    # ── utility methods ──────────────────────────────────────

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "watsonx",
            "model": self.model,
            "project_id": self.project_id,
            "space_id": self.space_id,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": "vision" in self.model.lower(),
        }