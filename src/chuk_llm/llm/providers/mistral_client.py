# chuk_llm/llm/providers/mistral_client.py

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
* MCP tool name sanitization for compatibility with Mistral's naming requirements
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

log = logging.getLogger(__name__)

class MistralLLMClient(ConfigAwareProviderMixin, BaseLLMClient):
    """
    Configuration-aware adapter for Mistral Le Plateforme API.
    
    Gets all capabilities from unified YAML configuration instead of
    hardcoded model patterns for better maintainability.
    
    Includes MCP tool name sanitization to handle tool names like
    'stdio.read_query' that need to be converted to 'stdio_read_query'
    for Mistral's API requirements.
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, "mistral", model)
        
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

    def _sanitize_tool_name_for_mistral(self, name: str) -> str:
        """
        Convert MCP/OpenAI tool names to Mistral-compatible format.
        
        Mistral requires function names to be:
        - Letters (a-z, A-Z)
        - Numbers (0-9) 
        - Underscores (_)
        - Dashes (-)
        - Maximum length of 64 characters
        
        Examples:
            stdio.read_query -> stdio_read_query
            filesystem.read_file -> filesystem_read_file  
            mcp.server:get_data -> mcp_server_get_data
        """
        if not name:
            return name
            
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it doesn't exceed 64 characters
        if len(sanitized) > 64:
            sanitized = sanitized[:64].rstrip('_')
            
        return sanitized

    def _sanitize_tools_for_mistral(self, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Sanitize tool names and create a mapping for response processing.
        
        Returns:
            tuple: (sanitized_tools, name_mapping)
                - sanitized_tools: Tools with Mistral-compatible names
                - name_mapping: Dict mapping sanitized_name -> original_name
        """
        if not tools:
            return tools, {}
            
        sanitized_tools = []
        name_mapping = {}
        
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                original_name = tool["function"]["name"]
                sanitized_name = self._sanitize_tool_name_for_mistral(original_name)
                
                # Store the mapping
                name_mapping[sanitized_name] = original_name
                
                # Create sanitized tool
                sanitized_tool = tool.copy()
                sanitized_tool["function"] = tool["function"].copy()
                sanitized_tool["function"]["name"] = sanitized_name
                
                sanitized_tools.append(sanitized_tool)
                
                # Log the sanitization if it changed
                if sanitized_name != original_name:
                    log.debug(f"Sanitized tool name: {original_name} -> {sanitized_name}")
            else:
                # Non-function tools pass through unchanged
                sanitized_tools.append(tool)
                
        return sanitized_tools, name_mapping

    def _restore_tool_names_in_response(self, response: Dict[str, Any], name_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Restore original tool names in the response using the mapping.
        """
        if not name_mapping or not response.get("tool_calls"):
            return response
            
        # Create a copy to avoid modifying the original
        restored_response = response.copy()
        restored_tool_calls = []
        
        for tool_call in response["tool_calls"]:
            if "function" in tool_call and "name" in tool_call["function"]:
                sanitized_name = tool_call["function"]["name"]
                original_name = name_mapping.get(sanitized_name, sanitized_name)
                
                # Restore the original name
                restored_tool_call = tool_call.copy()
                restored_tool_call["function"] = tool_call["function"].copy()
                restored_tool_call["function"]["name"] = original_name
                
                restored_tool_calls.append(restored_tool_call)
                
                # Log the restoration if it changed
                if original_name != sanitized_name:
                    log.debug(f"Restored tool name: {sanitized_name} -> {original_name}")
            else:
                restored_tool_calls.append(tool_call)
                
        restored_response["tool_calls"] = restored_tool_calls
        return restored_response

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with Mistral-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add Mistral-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "mistral_specific": {
                    "supports_magistral_reasoning": "magistral" in self.model.lower(),
                    "supports_code_generation": any(pattern in self.model.lower() 
                                                   for pattern in ["codestral", "devstral"]),
                    "is_multilingual": "saba" in self.model.lower(),
                    "is_edge_model": "ministral" in self.model.lower(),
                },
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
                        tool_calls = []
                        for tc in msg["tool_calls"]:
                            tool_calls.append({
                                "id": tc.get("id"),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": tc["function"]["name"],
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
            
            # Restore original tool names if we have a mapping
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
        Configuration-aware completion with Mistral API and MCP tool name sanitization.
        
        Args:
            messages: ChatML-style messages
            tools: OpenAI-style tool definitions (will be sanitized for Mistral)
            stream: Whether to stream response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            AsyncIterator for streaming, awaitable for non-streaming
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Sanitize tools for Mistral API compatibility
        name_mapping = {}
        if validated_tools:
            validated_tools, name_mapping = self._sanitize_tools_for_mistral(validated_tools)
            log.debug(f"Tool sanitization: {len(name_mapping)} tools renamed for Mistral compatibility")
        
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
        Real streaming using Mistral's async streaming API with tool name restoration.
        """
        try:
            log.debug(f"Starting Mistral streaming for model: {self.model}")
            
            # Use Mistral's streaming endpoint
            stream = self.client.chat.stream(**request_params)
            
            chunk_count = 0
            
            # Process streaming response
            for chunk in stream:
                chunk_count += 1
                
                if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                    choices = chunk.data.choices
                    if choices:
                        choice = choices[0]
                        
                        # Extract content from delta
                        content = ""
                        tool_calls = []
                        
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                            
                            # Get content
                            if hasattr(delta, 'content') and delta.content:
                                content = delta.content
                            
                            # Get tool calls (only if tools are supported)
                            if (hasattr(delta, 'tool_calls') and delta.tool_calls and 
                                self.supports_feature("tools")):
                                for tc in delta.tool_calls:
                                    tool_calls.append({
                                        "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                        "type": getattr(tc, 'type', 'function'),
                                        "function": {
                                            "name": getattr(tc.function, 'name', ''),
                                            "arguments": getattr(tc.function, 'arguments', '')
                                        }
                                    })
                        
                        # Create chunk response
                        chunk_response = {
                            "response": content,
                            "tool_calls": tool_calls
                        }
                        
                        # Restore tool names if needed
                        if name_mapping and tool_calls:
                            chunk_response = self._restore_tool_names_in_response(chunk_response, name_mapping)
                        
                        # Yield chunk if it has content
                        if content or tool_calls:
                            yield chunk_response
                
                # Allow other async tasks to run
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            log.debug(f"Mistral streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            log.error(f"Error in Mistral streaming: {e}")
            
            # Check if it's a tool name validation error
            if "Function name" in str(e) and "must be a-z, A-Z, 0-9" in str(e):
                log.error(f"Tool name sanitization may have failed: {e}")
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
                log.error(f"Tool name sanitization may have failed: {e}")
                log.error(f"Request tools: {[t.get('function', {}).get('name') for t in request_params.get('tools', []) if t.get('type') == 'function']}")
            
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def close(self):
        """Cleanup resources"""
        # Mistral client doesn't require explicit cleanup
        pass