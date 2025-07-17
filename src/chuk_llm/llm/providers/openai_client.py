# chuk_llm/llm/providers/openai_client.py - FIXED VERSION FOR STREAMING
"""
OpenAI chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around the official `openai` SDK that uses the unified
configuration system and universal tool name compatibility.

CRITICAL FIXES:
1. Added ToolCompatibilityMixin inheritance for universal tool names
2. Fixed conversation flow tool name handling 
3. Enhanced content extraction to eliminate warnings
4. Added bidirectional mapping throughout conversation
5. FIXED streaming tool call duplication bug
"""
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple
import openai
import logging
import asyncio
import time
import re
import uuid
import json

# mixins
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin
from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

# base
from ..core.base import BaseLLMClient

log = logging.getLogger(__name__)

class OpenAILLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-driven wrapper around the official `openai` SDK that gets
    all capabilities from the unified YAML configuration.
    
    CRITICAL FIX: Now includes ToolCompatibilityMixin for universal tool name compatibility.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Detect provider from api_base for configuration lookup
        detected_provider = self._detect_provider_name(api_base)
        
        # CRITICAL FIX: Initialize ALL mixins including ToolCompatibilityMixin
        ConfigAwareProviderMixin.__init__(self, detected_provider, model)
        ToolCompatibilityMixin.__init__(self, detected_provider)
        
        self.model = model
        self.api_base = api_base
        self.detected_provider = detected_provider
        
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
        
        log.debug(f"OpenAI client initialized: provider={self.detected_provider}, model={self.model}")

    def _detect_provider_name(self, api_base: Optional[str]) -> str:
        """Detect provider name from API base URL for configuration lookup"""
        if not api_base:
            return "openai"
        
        api_base_lower = api_base.lower()
        if "deepseek" in api_base_lower:
            return "deepseek"
        elif "groq" in api_base_lower:
            return "groq"
        elif "together" in api_base_lower:
            return "together"
        elif "perplexity" in api_base_lower:
            return "perplexity"
        elif "anyscale" in api_base_lower:
            return "anyscale"
        else:
            return "openai_compatible"

    def detect_provider_name(self) -> str:
        """Public method to detect provider name"""
        return self.detected_provider

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with OpenAI-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add tool compatibility info from universal system
        tool_compatibility = self.get_tool_compatibility_info()
        
        # Add OpenAI-specific metadata only if no error
        if not info.get("error"):
            info.update({
                "api_base": self.api_base,
                "detected_provider": self.detected_provider,
                "openai_compatible": True,
                # Universal tool compatibility info
                **tool_compatibility,
                "parameter_mapping": {
                    "temperature": "temperature",
                    "max_tokens": "max_tokens",
                    "top_p": "top_p",
                    "frequency_penalty": "frequency_penalty",
                    "presence_penalty": "presence_penalty",
                    "stop": "stop",
                    "stream": "stream"
                }
            })
        
        return info

    def _normalize_message(self, msg) -> Dict[str, Any]:
        """
        ENHANCED: Improved content extraction to eliminate warnings.
        """
        content = None
        tool_calls = []
        
        # Try multiple methods to extract content
        try:
            if hasattr(msg, 'content'):
                content = msg.content
        except Exception as e:
            log.debug(f"Direct content access failed: {e}")
        
        # Try message wrapper
        if content is None:
            try:
                if hasattr(msg, 'message') and hasattr(msg.message, 'content'):
                    content = msg.message.content
            except Exception as e:
                log.debug(f"Message wrapper access failed: {e}")
        
        # Try dict access
        if content is None:
            try:
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
            except Exception as e:
                log.debug(f"Dict content access failed: {e}")
        
        # Extract tool calls with enhanced error handling
        try:
            raw_tool_calls = None
            
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                raw_tool_calls = msg.tool_calls
            elif hasattr(msg, 'message') and hasattr(msg.message, 'tool_calls') and msg.message.tool_calls:
                raw_tool_calls = msg.message.tool_calls
            elif isinstance(msg, dict) and msg.get('tool_calls'):
                raw_tool_calls = msg['tool_calls']
            
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    try:
                        tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                        
                        if hasattr(tc, 'function'):
                            func = tc.function
                            func_name = getattr(func, 'name', 'unknown_function')
                            
                            # Handle arguments with robust JSON processing
                            args = getattr(func, 'arguments', '{}')
                            try:
                                if isinstance(args, str):
                                    parsed_args = json.loads(args)
                                    args_j = json.dumps(parsed_args)
                                elif isinstance(args, dict):
                                    args_j = json.dumps(args)
                                else:
                                    args_j = "{}"
                            except json.JSONDecodeError:
                                args_j = "{}"
                            
                            tool_calls.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": args_j,
                                },
                            })
                        
                    except Exception as e:
                        log.warning(f"Failed to process tool call {tc}: {e}")
                        continue
        except Exception as e:
            log.warning(f"Failed to extract tool calls: {e}")
        
        # Set default content if None
        if content is None:
            content = ""
        
        # Determine response format
        if tool_calls:
            response_value = content if content and content.strip() else None
        else:
            response_value = content
        
        result = {
            "response": response_value,
            "tool_calls": tool_calls
        }
        
        return result

    def _prepare_messages_for_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CRITICAL FIX: Prepare messages for conversation by sanitizing tool names in message history.
        
        This is the key fix for conversation flows - tool names in assistant messages
        must be sanitized to match what the API expects.
        """
        if not hasattr(self, '_current_name_mapping') or not self._current_name_mapping:
            return messages
        
        prepared_messages = []
        
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Sanitize tool names in assistant message tool calls
                prepared_msg = msg.copy()
                sanitized_tool_calls = []
                
                for tc in msg["tool_calls"]:
                    tc_copy = tc.copy()
                    original_name = tc["function"]["name"]
                    
                    # Find sanitized name from current mapping
                    sanitized_name = None
                    for sanitized, original in self._current_name_mapping.items():
                        if original == original_name:
                            sanitized_name = sanitized
                            break
                    
                    if sanitized_name:
                        tc_copy["function"] = tc["function"].copy()
                        tc_copy["function"]["name"] = sanitized_name
                        log.debug(f"Sanitized tool name in conversation: {original_name} -> {sanitized_name}")
                    
                    sanitized_tool_calls.append(tc_copy)
                
                prepared_msg["tool_calls"] = sanitized_tool_calls
                prepared_messages.append(prepared_msg)
            else:
                prepared_messages.append(msg)
        
        return prepared_messages

    async def _stream_from_async(
        self,
        async_stream,
        name_mapping: Dict[str, str] = None,
        normalize_chunk: Optional[callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        CRITICAL FIX: Enhanced streaming with proper tool call accumulation.
        
        The key insight from diagnostics: OpenAI sends incremental deltas that need 
        concatenation, but chuk-llm was receiving pre-processed complete chunks 
        and duplicating them by extending instead of replacing.
        """
        try:
            chunk_count = 0
            total_content = ""
            
            # Track tool calls across chunks
            accumulated_tool_calls = {}  # {index: {id, name, arguments}}
            last_complete_tool_calls = None  # Store last complete set
            
            async for chunk in async_stream:
                chunk_count += 1
                
                content = ""
                current_tool_calls = []
                
                try:
                    if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            
                            # Handle content
                            if hasattr(delta, 'content') and delta.content is not None:
                                content = str(delta.content)
                                total_content += content
                            
                            # Handle tool calls - check if this is delta or complete
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                # Determine if we're getting deltas or complete chunks
                                first_tc = delta.tool_calls[0]
                                is_delta_chunk = (
                                    # Delta chunks have partial function arguments
                                    (hasattr(first_tc, 'function') and 
                                     hasattr(first_tc.function, 'arguments') and 
                                     first_tc.function.arguments is not None and
                                     not first_tc.function.arguments.startswith('{"')) or
                                    # Or they have no function name (continuation chunks)
                                    (hasattr(first_tc, 'function') and 
                                     hasattr(first_tc.function, 'name') and 
                                     first_tc.function.name is None)
                                )
                                
                                if is_delta_chunk:
                                    # DELTA MODE: Accumulate pieces (for raw OpenAI streaming)
                                    for tc in delta.tool_calls:
                                        try:
                                            tc_index = getattr(tc, 'index', 0)
                                            
                                            if tc_index not in accumulated_tool_calls:
                                                accumulated_tool_calls[tc_index] = {
                                                    "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                                    "name": "",
                                                    "arguments": ""
                                                }
                                            
                                            # Update ID if provided
                                            if hasattr(tc, 'id') and tc.id:
                                                accumulated_tool_calls[tc_index]["id"] = tc.id
                                            
                                            # Accumulate function data
                                            if hasattr(tc, 'function') and tc.function:
                                                if hasattr(tc.function, 'name') and tc.function.name:
                                                    accumulated_tool_calls[tc_index]["name"] += tc.function.name
                                                
                                                if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                                    accumulated_tool_calls[tc_index]["arguments"] += tc.function.arguments
                                        
                                        except Exception as e:
                                            log.debug(f"Error processing delta tool call: {e}")
                                            continue
                                    
                                    # Build current complete tool calls from accumulated data
                                    for tc_index, tc_data in accumulated_tool_calls.items():
                                        if tc_data["name"] and tc_data["arguments"]:
                                            # Validate JSON completeness
                                            try:
                                                json.loads(tc_data["arguments"])
                                                current_tool_calls.append({
                                                    "id": tc_data["id"],
                                                    "type": "function",
                                                    "function": {
                                                        "name": tc_data["name"],
                                                        "arguments": tc_data["arguments"]
                                                    }
                                                })
                                            except json.JSONDecodeError:
                                                # Incomplete JSON, skip for now
                                                pass
                                
                                else:
                                    # COMPLETE MODE: Use tool calls as-is (for pre-processed streams)
                                    for tc in delta.tool_calls:
                                        try:
                                            tc_id = getattr(tc, "id", f"call_{uuid.uuid4().hex[:8]}")
                                            
                                            if hasattr(tc, 'function'):
                                                func = tc.function
                                                func_name = getattr(func, 'name', 'unknown_function')
                                                
                                                # Handle arguments
                                                args = getattr(func, 'arguments', '{}')
                                                try:
                                                    if isinstance(args, str):
                                                        # Validate and normalize JSON
                                                        parsed_args = json.loads(args)
                                                        args_j = json.dumps(parsed_args)
                                                    elif isinstance(args, dict):
                                                        args_j = json.dumps(args)
                                                    else:
                                                        args_j = "{}"
                                                except json.JSONDecodeError:
                                                    args_j = args if isinstance(args, str) else "{}"
                                                
                                                current_tool_calls.append({
                                                    "id": tc_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": func_name,
                                                        "arguments": args_j,
                                                    },
                                                })
                                        
                                        except Exception as e:
                                            log.warning(f"Failed to process complete tool call: {e}")
                                            continue
                                    
                                    # CRITICAL FIX: Replace, don't extend for complete chunks
                                    last_complete_tool_calls = current_tool_calls
                
                except Exception as chunk_error:
                    log.warning(f"Error processing chunk {chunk_count}: {chunk_error}")
                    content = ""
                
                # Prepare result
                result = {
                    "response": content,
                    "tool_calls": current_tool_calls if current_tool_calls else (last_complete_tool_calls or []),
                }
                
                # Restore tool names using universal restoration
                if name_mapping and result.get("tool_calls"):
                    result = self._restore_tool_names_in_response(result, name_mapping)
                
                # Yield if we have content or tool calls
                if content or result.get("tool_calls"):
                    yield result
            
            log.debug(f"[{self.detected_provider}] Streaming completed: {chunk_count} chunks, "
                    f"{len(total_content)} total characters, {len(accumulated_tool_calls)} accumulated tool calls")
                        
        except Exception as e:
            log.error(f"Error in {self.detected_provider} streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }
            
    def _validate_request_with_config(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], bool, Dict[str, Any]]:
        """
        Validate request against configuration before processing.
        """
        validated_messages = messages
        validated_tools = tools
        validated_stream = stream
        validated_kwargs = kwargs
        
        # Check streaming support
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but {self.detected_provider}/{self.model} doesn't support streaming according to configuration")
            validated_stream = False
        
        # Check tool support
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but {self.detected_provider}/{self.model} doesn't support tools according to configuration")
            validated_tools = None
        
        # Check vision support
        has_vision = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.warning(f"Vision content detected but {self.detected_provider}/{self.model} doesn't support vision according to configuration")
        
        # Check JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            if not self.supports_feature("json_mode"):
                log.warning(f"JSON mode requested but {self.detected_provider}/{self.model} doesn't support JSON mode according to configuration")
                validated_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    # ------------------------------------------------------------------ #
    # Enhanced public API using universal tool compatibility             #
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
        CRITICAL FIX: Now includes universal tool name compatibility with conversation flow handling.
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # CRITICAL FIX: Apply universal tool name sanitization 
        name_mapping = {}
        if validated_tools:
            validated_tools = self._sanitize_tool_names(validated_tools)
            name_mapping = self._current_name_mapping
            log.debug(f"Tool sanitization: {len(name_mapping)} tools processed for {self.detected_provider} compatibility")
        
        # CRITICAL FIX: Prepare messages for conversation (sanitize tool names in history)
        if name_mapping:
            validated_messages = self._prepare_messages_for_conversation(validated_messages)
        
        # Use configuration-aware parameter adjustment
        validated_kwargs = self.validate_parameters(**validated_kwargs)

        if validated_stream:
            return self._stream_completion_async(validated_messages, validated_tools, name_mapping, **validated_kwargs)
        else:
            return self._regular_completion(validated_messages, validated_tools, name_mapping, **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced async streaming with universal tool name restoration.
        """
        max_retries = 1
        
        for attempt in range(max_retries + 1):
            try:
                log.debug(f"[{self.detected_provider}] Starting streaming (attempt {attempt + 1}): "
                         f"model={self.model}, messages={len(messages)}, tools={len(tools) if tools else 0}")
                
                response_stream = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **({"tools": tools} if tools else {}),
                    stream=True,
                    **kwargs
                )
                
                chunk_count = 0
                async for result in self._stream_from_async(response_stream, name_mapping):
                    chunk_count += 1
                    yield result
                
                log.debug(f"[{self.detected_provider}] Streaming completed successfully with {chunk_count} chunks")
                return
                
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(pattern in error_str for pattern in [
                    "timeout", "connection", "network", "temporary", "rate limit"
                ])
                
                if attempt < max_retries and is_retryable:
                    wait_time = (attempt + 1) * 1.0
                    log.warning(f"[{self.detected_provider}] Streaming attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    log.error(f"[{self.detected_provider}] Streaming failed after {attempt + 1} attempts: {e}")
                    yield {
                        "response": f"Error: {str(e)}",
                        "tool_calls": [],
                        "error": True
                    }
                    return

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enhanced non-streaming completion with universal tool name restoration."""
        try:
            log.debug(f"[{self.detected_provider}] Starting completion: "
                     f"model={self.model}, messages={len(messages)}, tools={len(tools) if tools else 0}")
            
            resp = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **({"tools": tools} if tools else {}),
                stream=False,
                **kwargs
            )
            
            result = self._normalize_message(resp.choices[0].message)
            
            # CRITICAL: Restore original tool names using universal restoration
            if name_mapping and result.get("tool_calls"):
                result = self._restore_tool_names_in_response(result, name_mapping)
            
            log.debug(f"[{self.detected_provider}] Completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            log.error(f"[{self.detected_provider}] Error in completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping from universal system
        if hasattr(self, '_current_name_mapping'):
            self._current_name_mapping = {}
        
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        if hasattr(self.client, 'close'):
            self.client.close()