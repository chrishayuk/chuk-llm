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
*   **Universal Tool Name Compatibility** - handles any naming convention with bidirectional mapping
"""
from __future__ import annotations
import base64
import json
import logging
import os
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

# llm
from anthropic import AsyncAnthropic

# providers
from ..core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin
from ._config_mixin import ConfigAwareProviderMixin
from ._tool_compatibility import ToolCompatibilityMixin

log = logging.getLogger(__name__)
if os.getenv("LOGLEVEL"):
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())

# ────────────────────────── helpers ──────────────────────────


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:  # noqa: D401 - util
    """Get *key* from dict **or** attribute-style object; fallback to *default*."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _parse_claude_response(resp) -> Dict[str, Any]:  # noqa: D401 - small helper
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


class AnthropicLLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-aware Anthropic adapter that gets all capabilities from YAML config.
    
    Uses universal tool name compatibility system to handle any naming convention:
    - stdio.read_query -> stdio_read_query
    - web.api:search -> web_api_search  
    - database.sql.execute -> database_sql_execute
    - service:method -> service_method
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Initialize mixins
        ConfigAwareProviderMixin.__init__(self, "anthropic", model)
        ToolCompatibilityMixin.__init__(self, "anthropic")
        
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
        
        # Add tool compatibility info
        tool_compatibility = self.get_tool_compatibility_info()
        
        # Add Anthropic-specific metadata
        if not info.get("error"):
            info.update({
                "vision_format": "universal_image_url",
                # Universal tool compatibility info
                **tool_compatibility,
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
        """
        Convert OpenAI-style tools to Anthropic format.
        
        Note: Tool names should already be sanitized by ToolCompatibilityMixin
        before reaching this method.
        """
        if not tools:
            return []

        converted: List[Dict[str, Any]] = []
        for entry in tools:
            fn = entry.get("function", entry)
            try:
                tool_name = fn["name"]
                
                converted.append(
                    {
                        "name": tool_name,  # Should already be sanitized
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters") or fn.get("input_schema") or {},
                    }
                )
            except Exception as exc:  # pragma: no cover - permissive fallback
                log.debug("Tool schema error (%s) - using permissive schema", exc)
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
                        "name": tc["function"]["name"],  # Should contain sanitized names
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

    def _parse_claude_response_with_restoration(self, resp, name_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Convert Claude response to standard format and restore tool names"""
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
            result = {"response": None, "tool_calls": tool_calls}
            # Restore original tool names using universal restoration
            if name_mapping:
                result = self._restore_tool_names_in_response(result, name_mapping)
            return result

        text = resp.content[0].text if getattr(resp, "content", None) else ""
        return {"response": text, "tool_calls": []}

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
        Configuration-aware completion generation with universal tool name compatibility.
        
        Uses configuration to validate:
        - Tool support before processing tools
        - Streaming support before enabling streaming
        - JSON mode support before adding JSON instructions
        - Vision support during message processing
        
        Universal tool name compatibility handles any naming convention:
        - stdio.read_query -> stdio_read_query
        - web.api:search -> web_api_search  
        - database.sql.execute -> database_sql_execute
        """

        # Validate capabilities using configuration
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but model {self.model} doesn't support tools according to configuration")
            tools = None
        
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but model {self.model} doesn't support streaming according to configuration")
            stream = False

        # Apply universal tool name sanitization (stores mapping for restoration)
        name_mapping = {}
        if tools:
            tools = self._sanitize_tool_names(tools)
            name_mapping = self._current_name_mapping
            log.debug(f"Tool sanitization: {len(name_mapping)} tools processed for Anthropic compatibility")
        
        anth_tools = self._convert_tools(tools)
        
        # Check for JSON mode (using configuration validation)
        json_instruction = self._check_json_mode(extra)
        
        # Filter parameters for Anthropic compatibility (using configuration limits)
        if max_tokens:
            extra["max_tokens"] = max_tokens
        filtered_params = self._filter_anthropic_params(extra)

        # --- streaming: use real async streaming -------------------------
        if stream:
            return self._stream_completion_async(system, json_instruction, messages, anth_tools, filtered_params, name_mapping)

        # --- non-streaming: use async client ------------------------------
        return self._regular_completion_async(system, json_instruction, messages, anth_tools, filtered_params, name_mapping)

    async def _stream_completion_async(
        self, 
        system: Optional[str],
        json_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        anth_tools: List[Dict[str, Any]],
        filtered_params: Dict[str, Any],
        name_mapping: Dict[str, str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real streaming using AsyncAnthropic with FIXED tool call accumulation.
        Restores original tool names using universal restoration.
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
            
            # CRITICAL FIX: Track tool calls across streaming events
            accumulated_tool_calls = {}  # {tool_id: {name, input_json_parts, complete}}
            
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
                        
                        # CRITICAL FIX: Handle streaming tool input (input_json_delta)
                        elif hasattr(event, 'delta') and hasattr(event.delta, 'type'):
                            if event.delta.type == 'input_json_delta':
                                # Get the content block index to identify which tool call this belongs to
                                content_index = getattr(event, 'index', 0)
                                
                                # Find the tool call by index (assuming tools are processed in order)
                                tool_id = None
                                for tid, tool_data in accumulated_tool_calls.items():
                                    if tool_data.get('content_index') == content_index:
                                        tool_id = tid
                                        break
                                
                                if tool_id and hasattr(event.delta, 'partial_json'):
                                    # Accumulate JSON input parts
                                    if 'input_json_parts' not in accumulated_tool_calls[tool_id]:
                                        accumulated_tool_calls[tool_id]['input_json_parts'] = []
                                    accumulated_tool_calls[tool_id]['input_json_parts'].append(event.delta.partial_json)
                    
                    # CRITICAL FIX: Enhanced tool use event handling
                    elif hasattr(event, 'type') and event.type == 'content_block_start':
                        if hasattr(event, 'content_block') and event.content_block.type == 'tool_use':
                            tool_id = event.content_block.id
                            content_index = getattr(event, 'index', 0)
                            
                            # Initialize tool call tracking
                            accumulated_tool_calls[tool_id] = {
                                "name": event.content_block.name,
                                "input": getattr(event.content_block, 'input', {}),
                                "input_json_parts": [],
                                "content_index": content_index,
                                "complete": False
                            }
                            
                            # If tool already has complete input, yield immediately
                            if accumulated_tool_calls[tool_id]["input"]:
                                tool_call = {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": event.content_block.name,
                                        "arguments": json.dumps(accumulated_tool_calls[tool_id]["input"])
                                    }
                                }
                                
                                # Create response with tool call
                                chunk_response = {
                                    "response": "",
                                    "tool_calls": [tool_call]
                                }
                                
                                # Restore original tool names using universal restoration
                                if name_mapping:
                                    chunk_response = self._restore_tool_names_in_response(
                                        chunk_response, 
                                        name_mapping
                                    )
                                
                                accumulated_tool_calls[tool_id]["complete"] = True
                                yield chunk_response
                    
                    # CRITICAL FIX: Handle tool call completion
                    elif hasattr(event, 'type') and event.type == 'content_block_stop':
                        content_index = getattr(event, 'index', 0)
                        
                        # Find and finalize tool call by content index
                        for tool_id, tool_data in accumulated_tool_calls.items():
                            if (tool_data.get('content_index') == content_index and 
                                not tool_data.get('complete')):
                                
                                # Reconstruct complete JSON from parts
                                final_input = tool_data.get("input", {})
                                if tool_data.get("input_json_parts"):
                                    try:
                                        # Combine all JSON parts
                                        complete_json = "".join(tool_data["input_json_parts"])
                                        final_input = json.loads(complete_json)
                                    except json.JSONDecodeError:
                                        # Fallback to original input if JSON parsing fails
                                        log.warning(f"Failed to parse streaming JSON for tool {tool_id}")
                                        final_input = tool_data.get("input", {})
                                
                                tool_call = {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_data["name"],
                                        "arguments": json.dumps(final_input)
                                    }
                                }
                                
                                # Create response with tool call
                                chunk_response = {
                                    "response": "",
                                    "tool_calls": [tool_call]
                                }
                                
                                # Restore original tool names using universal restoration
                                if name_mapping:
                                    chunk_response = self._restore_tool_names_in_response(
                                        chunk_response, 
                                        name_mapping
                                    )
                                
                                tool_data["complete"] = True
                                yield chunk_response
                                break
            
            # FINAL FIX: Ensure any incomplete tool calls are yielded at the end
            incomplete_tools = []
            for tool_id, tool_data in accumulated_tool_calls.items():
                if not tool_data.get("complete"):
                    # Reconstruct final input
                    final_input = tool_data.get("input", {})
                    if tool_data.get("input_json_parts"):
                        try:
                            complete_json = "".join(tool_data["input_json_parts"])
                            final_input = json.loads(complete_json)
                        except json.JSONDecodeError:
                            final_input = tool_data.get("input", {})
                    
                    incomplete_tools.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_data["name"],
                            "arguments": json.dumps(final_input)
                        }
                    })
            
            if incomplete_tools:
                final_response = {
                    "response": "",
                    "tool_calls": incomplete_tools
                }
                
                if name_mapping:
                    final_response = self._restore_tool_names_in_response(
                        final_response, 
                        name_mapping
                    )
                
                yield final_response
        
        except Exception as e:
            log.error(f"Error in Anthropic streaming: {e}")
            
            # Check if it's a tool name validation error
            if "tools.0.custom.name" in str(e) and "should match pattern" in str(e):
                log.error(f"Tool name validation error (this should not happen with universal compatibility): {e}")
                log.error(f"Request tools: {[t.get('name') for t in anth_tools]}")
            
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
        filtered_params: Dict[str, Any],
        name_mapping: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Non-streaming completion using async client with configuration-aware vision processing.
        Restores original tool names using universal restoration.
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

            log.debug("Claude payload keys: %s", list(base_payload.keys()))
            
            resp = await self.async_client.messages.create(**base_payload)
            
            # Parse response and restore tool names using universal restoration
            result = self._parse_claude_response_with_restoration(resp, name_mapping)
            
            return result
            
        except Exception as e:
            log.error(f"Error in Anthropic completion: {e}")
            
            # Check if it's a tool name validation error
            if "tools.0.custom.name" in str(e) and "should match pattern" in str(e):
                log.error(f"Tool name validation error (this should not happen with universal compatibility): {e}")
                log.error(f"Request tools: {[t.get('name') for t in anth_tools]}")
            
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping
        self._current_name_mapping = {}
        # AsyncAnthropic handles cleanup automatically
        pass