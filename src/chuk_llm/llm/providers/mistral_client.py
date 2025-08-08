# chuk_llm/llm/providers/mistral_client.py - FIXED VERSION WITH DUPLICATION PREVENTION

"""
Mistral Le Plateforme chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Features
--------
* Configuration-driven capabilities from YAML instead of hardcoded patterns
* Full support for Mistral's API including vision, function calling, and streaming
* Real async streaming without buffering
* Vision capabilities for supported models
* Function calling support for compatible models
* Universal tool name compatibility with bidirectional mapping
* CRITICAL FIX: Tool call duplication prevention in streaming
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

# Import Mistral SDK
try:
    from mistralai import Mistral
except ImportError:
    raise ImportError(
        "mistralai package is required for Mistral provider. "
        "Install with: pip install mistralai"
    )

# Base imports
from chuk_llm.llm.core.base import BaseLLMClient
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin
from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

log = logging.getLogger(__name__)

class MistralLLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, BaseLLMClient):
    """
    Configuration-aware adapter for Mistral Le Plateforme API.
    
    Gets all capabilities from unified YAML configuration instead of
    hardcoded model patterns for better maintainability.
    
    CRITICAL FIX: Now includes tool call duplication prevention using the same
    pattern that was successfully implemented for Groq.
    
    Uses universal tool name compatibility system to handle any naming convention:
    - stdio.read_query -> stdio_read_query
    - web.api:search -> web_api_search  
    - database.sql.execute -> database_sql_execute
    - service:method -> service_method
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Initialize mixins
        ConfigAwareProviderMixin.__init__(self, "mistral", model)
        ToolCompatibilityMixin.__init__(self, "mistral")
        
        self.model = model
        self.provider_name = "mistral"
        
        # Initialize Mistral client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["server_url"] = api_base
            
        self.client = Mistral(**client_kwargs)
        
        log.info(f"MistralLLMClient initialized with model: {model}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with Mistral-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add tool compatibility info
        tool_compatibility = self.get_tool_compatibility_info()
        
        # Add Mistral-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "mistral_specific": {
                    "supports_magistral_reasoning": "magistral" in self.model.lower(),
                    "supports_code_generation": any(pattern in self.model.lower() 
                                                   for pattern in ["codestral", "devstral"]),
                    "is_multilingual": "saba" in self.model.lower(),
                    "is_edge_model": "ministral" in self.model.lower(),
                    "duplication_fix": "enabled",  # NEW: Indicates duplication fix is active
                },
                # Universal tool compatibility info
                **tool_compatibility,
                "parameter_mapping": {
                    "temperature": "temperature",
                    "max_tokens": "max_tokens", 
                    "top_p": "top_p",
                    "stream": "stream",
                    "tool_choice": "tool_choice"
                },
                "unsupported_parameters": [
                    "frequency_penalty", "presence_penalty", "stop", 
                    "logit_bias", "user", "n", "best_of", "top_k", "seed"
                ]
            })
        
        return info

    def _convert_messages_to_mistral_format(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert ChatML messages to Mistral format with configuration-aware vision handling"""
        mistral_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # Handle different message types
            if role == "system":
                # Check if system messages are supported
                if self.supports_feature("system_messages"):
                    mistral_messages.append({
                        "role": "system",
                        "content": content
                    })
                else:
                    # Fallback: convert to user message
                    log.warning(f"System messages not supported by {self.model}, converting to user message")
                    mistral_messages.append({
                        "role": "user",
                        "content": f"System: {content}"
                    })
            
            elif role == "user":
                if isinstance(content, str):
                    # Simple text message
                    mistral_messages.append({
                        "role": "user", 
                        "content": content
                    })
                elif isinstance(content, list):
                    # Multimodal message (text + images)
                    # Check if vision is supported before processing
                    has_images = any(item.get("type") == "image_url" for item in content)
                    
                    if has_images and not self.supports_feature("vision"):
                        log.warning(f"Vision content detected but {self.model} doesn't support vision according to configuration")
                        # Extract only text content
                        text_content = " ".join([
                            item.get("text", "") for item in content 
                            if item.get("type") == "text"
                        ])
                        mistral_messages.append({
                            "role": "user",
                            "content": text_content or "[Image content removed - not supported by model]"
                        })
                    else:
                        # Process multimodal content normally
                        mistral_content = []
                        for item in content:
                            if item.get("type") == "text":
                                mistral_content.append({
                                    "type": "text",
                                    "text": item.get("text", "")
                                })
                            elif item.get("type") == "image_url":
                                # Handle both URL and base64 formats
                                image_url = item.get("image_url", {})
                                if isinstance(image_url, dict):
                                    url = image_url.get("url", "")
                                else:
                                    url = str(image_url)
                                
                                mistral_content.append({
                                    "type": "image_url",
                                    "image_url": url
                                })
                        
                        mistral_messages.append({
                            "role": "user",
                            "content": mistral_content
                        })
            
            elif role == "assistant":
                # Handle assistant messages with potential tool calls
                if msg.get("tool_calls"):
                    # Check if tools are supported
                    if self.supports_feature("tools"):
                        # Convert tool calls to Mistral format
                        # IMPORTANT: Sanitize tool names in conversation history
                        tool_calls = []
                        for tc in msg["tool_calls"]:
                            original_name = tc["function"]["name"]
                            
                            # Apply sanitization to tool names in conversation history
                            from chuk_llm.llm.providers._tool_compatibility import ToolNameSanitizer
                            sanitizer = ToolNameSanitizer()
                            sanitized_name = sanitizer.sanitize_for_provider(original_name, "mistral")
                            
                            tool_calls.append({
                                "id": tc.get("id"),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": sanitized_name,  # Use sanitized name for API
                                    "arguments": tc["function"]["arguments"]
                                }
                            })
                        
                        mistral_messages.append({
                            "role": "assistant",
                            "content": content or "",
                            "tool_calls": tool_calls
                        })
                    else:
                        log.warning(f"Tool calls detected but {self.model} doesn't support tools according to configuration")
                        # Convert to text response
                        tool_text = f"{content or ''}\n\nNote: Tool calls were requested but not supported by this model."
                        mistral_messages.append({
                            "role": "assistant",
                            "content": tool_text
                        })
                else:
                    mistral_messages.append({
                        "role": "assistant",
                        "content": content or ""
                    })
            
            elif role == "tool":
                # Tool response messages - only include if tools are supported
                if self.supports_feature("tools"):
                    mistral_messages.append({
                        "role": "tool",
                        "name": msg.get("name", ""),
                        "content": content or "",
                        "tool_call_id": msg.get("tool_call_id", "")
                    })
                else:
                    # Convert tool response to user message
                    mistral_messages.append({
                        "role": "user",
                        "content": f"Tool result: {content or ''}"
                    })
        
        return mistral_messages

    def _normalize_mistral_response(self, response: Any, name_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Convert Mistral response to standard format and restore tool names"""
        # Handle both response types
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message
            
            content = getattr(message, 'content', '') or ''
            tool_calls = []
            
            # Extract tool calls if present and supported
            if hasattr(message, 'tool_calls') and message.tool_calls:
                if self.supports_feature("tools"):
                    for tc in message.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
                else:
                    # If tools aren't supported but we got tool calls, log warning
                    log.warning(f"Received tool calls from {self.model} but tools not supported according to configuration")
            
            # Create response
            result = {"response": content if content else None, "tool_calls": tool_calls}
            
            # Restore original tool names using universal restoration
            if name_mapping and tool_calls:
                result = self._restore_tool_names_in_response(result, name_mapping)
            
            return result
        
        # Fallback for unexpected response format
        return {"response": str(response), "tool_calls": []}

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
        validated_kwargs = kwargs.copy()
        
        # Check streaming support
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but {self.model} doesn't support streaming according to configuration")
            validated_stream = False
        
        # Check tool support
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but {self.model} doesn't support tools according to configuration")
            validated_tools = None
            # Remove tool-related parameters
            validated_kwargs.pop("tool_choice", None)
        
        # Check vision support (will be handled in message conversion)
        has_vision = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.info(f"Vision content will be filtered - {self.model} doesn't support vision according to configuration")
        
        # Validate parameters using configuration
        validated_kwargs = self.validate_parameters(**validated_kwargs)
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Configuration-aware completion with Mistral API and universal tool name compatibility.
        
        CRITICAL FIX: Now includes tool call duplication prevention using the same
        successful pattern from Groq.
        
        Args:
            messages: ChatML-style messages
            tools: Tool definitions (any naming convention supported)
            stream: Whether to stream response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            AsyncIterator for streaming, awaitable for non-streaming
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Apply universal tool name sanitization (stores mapping for restoration)
        name_mapping = {}
        if validated_tools:
            validated_tools = self._sanitize_tool_names(validated_tools)
            name_mapping = self._current_name_mapping
            log.debug(f"Tool sanitization: {len(name_mapping)} tools processed for Mistral compatibility")
        
        # Convert messages to Mistral format (with configuration-aware processing)
        mistral_messages = self._convert_messages_to_mistral_format(validated_messages)
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": mistral_messages,
            **validated_kwargs
        }
        
        # Add tools if provided and supported
        if validated_tools:
            request_params["tools"] = validated_tools
            # Set tool_choice to "auto" by default if not specified
            if "tool_choice" not in validated_kwargs:
                request_params["tool_choice"] = "auto"
        
        if validated_stream:
            return self._stream_completion_async(request_params, name_mapping)
        else:
            return self._regular_completion(request_params, name_mapping)

    async def _stream_completion_async(
        self, 
        request_params: Dict[str, Any],
        name_mapping: Dict[str, str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        COMPLETELY FIXED: Mistral streaming with proper JSON completion testing.
        
        Uses the same successful completion-based approach from OpenAI/Azure,
        not the problematic signature-based approach.
        
        Key fixes:
        - Only yield tool calls when JSON arguments are complete and parseable
        - Added completion status tracking (like OpenAI/Azure fix)
        - Removed signature tracking system entirely
        - Prevents both JSON parsing errors and tool call duplication
        """
        try:
            log.debug(f"Starting Mistral streaming for model: {self.model}")
            
            # Use Mistral's streaming endpoint
            stream = self.client.chat.stream(**request_params)
            
            # FIXED: Simple completion-based tracking (like OpenAI/Azure)
            accumulated_tool_calls = {}  # {index: {id, name, arguments, complete}}
            chunk_count = 0
            total_content = ""
            
            # Process streaming response
            for chunk in stream:
                chunk_count += 1
                
                content = ""
                completed_tool_calls = []  # Only completed tool calls this chunk
                
                try:
                    if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                        choices = chunk.data.choices
                        if choices:
                            choice = choices[0]
                            
                            if hasattr(choice, 'delta'):
                                delta = choice.delta
                                
                                # Handle content - this works fine
                                if hasattr(delta, 'content') and delta.content:
                                    content = delta.content
                                    total_content += content
                                
                                # FIXED: Handle tool calls with proper completion testing
                                if (hasattr(delta, 'tool_calls') and delta.tool_calls and 
                                    self.supports_feature("tools")):
                                    
                                    for tc in delta.tool_calls:
                                        try:
                                            tc_index = getattr(tc, 'index', 0)
                                            
                                            # Initialize accumulator with completion tracking
                                            if tc_index not in accumulated_tool_calls:
                                                accumulated_tool_calls[tc_index] = {
                                                    "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                                    "name": "",
                                                    "arguments": "",
                                                    "complete": False  # ADDED: Track completion status
                                                }
                                            
                                            tool_call_data = accumulated_tool_calls[tc_index]
                                            
                                            # Update data
                                            if hasattr(tc, 'id') and tc.id:
                                                tool_call_data["id"] = tc.id
                                            
                                            if hasattr(tc, 'function') and tc.function:
                                                if hasattr(tc.function, 'name') and tc.function.name:
                                                    tool_call_data["name"] += tc.function.name
                                                
                                                if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                                    tool_call_data["arguments"] += tc.function.arguments
                                            
                                            # CRITICAL FIX: Only yield when JSON is complete and valid
                                            if (tool_call_data["name"] and 
                                                tool_call_data["arguments"] and 
                                                not tool_call_data["complete"]):
                                                
                                                try:
                                                    # Test if JSON is complete and valid
                                                    parsed_args = json.loads(tool_call_data["arguments"])
                                                    
                                                    # Mark as complete and add to current chunk
                                                    tool_call_data["complete"] = True
                                                    
                                                    tool_call = {
                                                        "id": tool_call_data["id"],
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_call_data["name"],
                                                            "arguments": json.dumps(parsed_args)
                                                        }
                                                    }
                                                    
                                                    completed_tool_calls.append(tool_call)
                                                    log.debug(f"Mistral tool call {tc_index} completed: {tool_call_data['name']}")
                                                
                                                except json.JSONDecodeError:
                                                    # JSON incomplete - keep accumulating
                                                    log.debug(f"Mistral tool call {tc_index} JSON incomplete, continuing accumulation")
                                                    pass
                                        
                                        except Exception as e:
                                            log.debug(f"Error processing Mistral streaming tool call chunk: {e}")
                                            continue
                
                except Exception as chunk_error:
                    log.warning(f"Error processing Mistral chunk {chunk_count}: {chunk_error}")
                    content = ""
                
                # Create chunk response
                chunk_response = {
                    "response": content,
                    "tool_calls": completed_tool_calls if completed_tool_calls else None
                }
                
                # Restore tool names using universal restoration
                if name_mapping and completed_tool_calls:
                    chunk_response = self._restore_tool_names_in_response(chunk_response, name_mapping)
                
                # Only yield if we have content or completed tool calls
                if content or completed_tool_calls:
                    yield chunk_response
                
                # Allow other async tasks to run
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            log.debug(f"Mistral streaming completed with {chunk_count} chunks, "
                    f"{len(total_content)} total characters, {len(accumulated_tool_calls)} tool calls")
        
        except Exception as e:
            log.error(f"Error in Mistral streaming: {e}")
            
            # Check if it's a tool name validation error
            if "Function name" in str(e) and "must be a-z, A-Z, 0-9" in str(e):
                log.error(f"Tool name validation error (this should not happen with universal compatibility): {e}")
                log.error(f"Request tools: {[t.get('function', {}).get('name') for t in request_params.get('tools', []) if t.get('type') == 'function']}")
            
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }
            
    async def _regular_completion(
        self, 
        request_params: Dict[str, Any],
        name_mapping: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Non-streaming completion using async execution with tool name restoration."""
        try:
            log.debug(f"Starting Mistral completion for model: {self.model}")
            
            def _sync_completion():
                return self.client.chat.complete(**request_params)
            
            # Run sync call in thread to avoid blocking
            response = await asyncio.to_thread(_sync_completion)
            
            # Normalize response and restore tool names
            result = self._normalize_mistral_response(response, name_mapping)
            
            log.debug(f"Mistral completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            log.error(f"Error in Mistral completion: {e}")
            
            # Check if it's a tool name validation error
            if "Function name" in str(e) and "must be a-z, A-Z, 0-9" in str(e):
                log.error(f"Tool name validation error (this should not happen with universal compatibility): {e}")
                log.error(f"Request tools: {[t.get('function', {}).get('name') for t in request_params.get('tools', []) if t.get('type') == 'function']}")
            
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping
        self._current_name_mapping = {}
        # Mistral client doesn't require explicit cleanup
        pass