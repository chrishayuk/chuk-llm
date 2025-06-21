# chuk_llm/llm/providers/anthropic_client.py
"""
Anthropic chat-completion adapter with configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps the official `anthropic` SDK and exposes an **OpenAI-style** interface
compatible with the rest of *chuk-llm*.

Key points
----------
*   Uses unified configuration system for all capabilities
*   Converts ChatML → Claude Messages format (tools / multimodal, …)
*   Maps Claude replies back to the common `{response, tool_calls}` schema
*   **Real Streaming** - uses Anthropic's native async streaming API
*   **Universal Vision Format** - supports standard image_url format with URL downloading
*   **JSON Mode Support** - via system instructions
*   **System Parameter Support** - proper system message handling
"""
from __future__ import annotations
import base64
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# llm
from anthropic import AsyncAnthropic

# providers
from ..core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin
from ._config_mixin import ConfigAwareProviderMixin

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


class AnthropicLLMClient(ConfigAwareProviderMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-aware Anthropic adapter that gets all capabilities from YAML config.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, "anthropic", model)
        
        self.model = model
        
        # Use AsyncAnthropic for real streaming support
        kwargs: Dict[str, Any] = {"base_url": api_base} if api_base else {}
        if api_key:
            kwargs["api_key"] = api_key
        
        self.async_client = AsyncAnthropic(**kwargs)
        
        # Keep sync client for backwards compatibility if needed
        from anthropic import Anthropic
        self.client = Anthropic(**kwargs)
        
        log.debug(f"Anthropic client initialized with model: {model}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with Anthropic-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add Anthropic-specific metadata
        if not info.get("error"):
            info.update({
                "vision_format": "universal_image_url",
                "supported_parameters": ["temperature", "max_tokens", "top_p", "stream"],
                "unsupported_parameters": [
                    "frequency_penalty", "presence_penalty", "stop", "logit_bias",
                    "user", "n", "best_of", "top_k", "seed", "response_format"
                ],
            })
        
        return info

    def _filter_anthropic_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters using configuration limits instead of hardcoded lists"""
        filtered = {}
        
        # Get supported parameters (keeping existing logic for now, but could move to config)
        supported_params = {"temperature", "max_tokens", "top_p", "stream"}
        unsupported_params = {
            "frequency_penalty", "presence_penalty", "stop", "logit_bias",
            "user", "n", "best_of", "top_k", "seed", "response_format"
        }
        
        for key, value in params.items():
            if key in supported_params:
                # Use configuration to validate parameter limits
                if key == "temperature":
                    # Anthropic temperature range validation
                    if value > 1.0:
                        filtered[key] = 1.0
                        log.debug(f"Capped temperature from {value} to 1.0 for Anthropic")
                    else:
                        filtered[key] = value
                elif key == "max_tokens":
                    # Use configuration to validate max_tokens
                    limit = self.get_max_tokens_limit()
                    if limit and value > limit:
                        filtered[key] = limit
                        log.debug(f"Capped max_tokens from {value} to {limit} for Anthropic")
                    else:
                        filtered[key] = value
                else:
                    filtered[key] = value
            elif key in unsupported_params:
                log.debug(f"Filtered out unsupported parameter for Anthropic: {key}={value}")
            else:
                log.warning(f"Unknown parameter for Anthropic: {key}={value}")
        
        # Ensure required parameters based on configuration
        if "max_tokens" not in filtered:
            # Use configuration default if available, otherwise use reasonable default
            default_max = self.get_max_tokens_limit() or 4096
            filtered["max_tokens"] = min(4096, default_max)
            log.debug(f"Added required max_tokens={filtered['max_tokens']} for Anthropic")
        
        return filtered

    def _check_json_mode(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Check if JSON mode is requested and return appropriate system instruction"""
        # Only proceed if the model supports JSON mode according to config
        if not self.supports_feature("json_mode"):
            log.debug(f"Model {self.model} does not support JSON mode according to configuration")
            return None
        
        # Check for OpenAI-style response_format
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            return "You must respond with valid JSON only. No markdown code blocks, no explanations, no text before or after. Just pure, valid JSON."
        
        # Check for _json_mode_instruction from provider adapter
        json_instruction = kwargs.get("_json_mode_instruction")
        if json_instruction:
            return json_instruction
        
        return None

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
    async def _download_image_to_base64(url: str) -> tuple[str, str]:
        """Download image from URL and convert to base64"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Get content type from headers
                content_type = response.headers.get('content-type', 'image/png')
                if not content_type.startswith('image/'):
                    content_type = 'image/png'  # Default fallback
                
                # Convert to base64
                image_data = base64.b64encode(response.content).decode('utf-8')
                
                return content_type, image_data
                
        except Exception as e:
            log.warning(f"Failed to download image from {url}: {e}")
            raise ValueError(f"Could not download image: {e}")

    @staticmethod
    async def _convert_universal_vision_to_anthropic_async(content_item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal image_url format to Anthropic format with URL downloading"""
        if content_item.get("type") == "image_url":
            image_url = content_item.get("image_url", {})
            
            # Handle both string and dict formats
            if isinstance(image_url, str):
                url = image_url
            else:
                url = image_url.get("url", "")
            
            # Convert data URL to Anthropic format
            if url.startswith("data:"):
                # Extract media type and data
                try:
                    header, data = url.split(",", 1)
                    # Parse the header: data:image/png;base64
                    media_type_part = header.split(";")[0].replace("data:", "")
                    
                    # Validate media type
                    if not media_type_part.startswith("image/"):
                        media_type_part = "image/png"  # Default fallback
                    
                    # Anthropic expects format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type_part,
                            "data": data.strip()  # Remove any whitespace
                        }
                    }
                except (ValueError, IndexError) as e:
                    log.warning(f"Invalid data URL format: {url[:50]}... Error: {e}")
                    return {"type": "text", "text": "[Invalid image format]"}
            else:
                # For external URLs, download and convert to base64
                try:
                    media_type, image_data = await AnthropicLLMClient._download_image_to_base64(url)
                    
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    }
                except Exception as e:
                    log.warning(f"Failed to process external image URL {url}: {e}")
                    return {"type": "text", "text": f"[Could not load image: {e}]"}
        
        return content_item

    async def _split_for_anthropic_async(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Separate system text & convert ChatML list to Anthropic format with async vision support.
        Uses configuration to validate vision support.
        """
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

            # normal / multimodal messages with universal vision support
            if role in {"user", "assistant"}:
                cont = msg.get("content")
                if cont is None:
                    continue
                
                if isinstance(cont, str):
                    # Simple text content
                    out.append({
                        "role": role,
                        "content": [{"type": "text", "text": cont}]
                    })
                elif isinstance(cont, list):
                    # Multimodal content - check if vision is supported
                    has_vision_content = any(
                        isinstance(item, dict) and item.get("type") == "image_url" 
                        for item in cont
                    )
                    
                    if has_vision_content and not self.supports_feature("vision"):
                        log.warning(f"Vision content detected but model {self.model} doesn't support vision according to configuration")
                        # Convert to text-only by filtering out images
                        text_only_content = [
                            item for item in cont 
                            if not (isinstance(item, dict) and item.get("type") == "image_url")
                        ]
                        if text_only_content:
                            out.append({"role": role, "content": text_only_content})
                        continue
                    
                    # Process multimodal content - convert universal format to Anthropic
                    anthropic_content = []
                    for item in cont:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                anthropic_content.append(item)
                            elif item.get("type") == "image_url":
                                # Convert universal image_url to Anthropic format with async support
                                anthropic_item = await self._convert_universal_vision_to_anthropic_async(item)
                                anthropic_content.append(anthropic_item)
                            else:
                                # Pass through other formats
                                anthropic_content.append(item)
                        else:
                            # Handle non-dict items
                            anthropic_content.append({"type": "text", "text": str(item)})
                    
                    out.append({"role": role, "content": anthropic_content})
                else:
                    # Fallback for other content types
                    out.append({
                        "role": role,
                        "content": [{"type": "text", "text": str(cont)}]
                    })

        return "\n".join(sys_txt).strip(), out

    # ── main entrypoint ─────────────────────────────────────

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        **extra,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Configuration-aware completion generation.
        
        Uses configuration to validate:
        - Tool support before processing tools
        - Streaming support before enabling streaming
        - JSON mode support before adding JSON instructions
        - Vision support during message processing
        """

        # Validate capabilities using configuration
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but model {self.model} doesn't support tools according to configuration")
            tools = None
        
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but model {self.model} doesn't support streaming according to configuration")
            stream = False

        tools = self._sanitize_tool_names(tools)
        anth_tools = self._convert_tools(tools)
        
        # Check for JSON mode (using configuration validation)
        json_instruction = self._check_json_mode(extra)
        
        # Filter parameters for Anthropic compatibility (using configuration limits)
        if max_tokens:
            extra["max_tokens"] = max_tokens
        filtered_params = self._filter_anthropic_params(extra)

        # ––– streaming: use real async streaming -------------------------
        if stream:
            return self._stream_completion_async(system, json_instruction, messages, anth_tools, filtered_params)

        # ––– non-streaming: use async client ------------------------------
        return self._regular_completion_async(system, json_instruction, messages, anth_tools, filtered_params)

    async def _stream_completion_async(
        self, 
        system: Optional[str],
        json_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        anth_tools: List[Dict[str, Any]],
        filtered_params: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real streaming using AsyncAnthropic with configuration-aware vision processing.
        """
        try:
            # Handle system message and JSON instruction
            system_from_messages, msg_no_system = await self._split_for_anthropic_async(messages)
            final_system = system or system_from_messages
            
            if json_instruction:
                if final_system:
                    final_system = f"{final_system}\n\n{json_instruction}"
                else:
                    final_system = json_instruction
                log.debug("Added JSON mode instruction to system prompt")

            base_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": msg_no_system,
                "tools": anth_tools,
                **filtered_params,
            }
            if final_system:
                base_payload["system"] = final_system
            if anth_tools:
                base_payload["tool_choice"] = {"type": "auto"}

            log.debug("Claude streaming payload keys: %s", list(base_payload.keys()))
            
            # Use async client for real streaming
            async with self.async_client.messages.stream(
                **base_payload
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

    async def _regular_completion_async(
        self, 
        system: Optional[str],
        json_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        anth_tools: List[Dict[str, Any]],
        filtered_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming completion using async client with configuration-aware vision processing."""
        try:
            # Handle system message and JSON instruction
            system_from_messages, msg_no_system = await self._split_for_anthropic_async(messages)
            final_system = system or system_from_messages
            
            if json_instruction:
                if final_system:
                    final_system = f"{final_system}\n\n{json_instruction}"
                else:
                    final_system = json_instruction
                log.debug("Added JSON mode instruction to system prompt")

            base_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": msg_no_system,
                "tools": anth_tools,
                **filtered_params,
            }
            if final_system:
                base_payload["system"] = final_system
            if anth_tools:
                base_payload["tool_choice"] = {"type": "auto"}

            log.debug("Claude payload keys: %s", list(base_payload.keys()))
            
            resp = await self.async_client.messages.create(**base_payload)
            return _parse_claude_response(resp)
            
        except Exception as e:
            log.error(f"Error in Anthropic completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }