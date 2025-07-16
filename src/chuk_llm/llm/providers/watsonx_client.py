# chuk_llm/llm/providers/watsonx_client.py
"""
Watson X chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps the official `ibm-watsonx-ai` SDK and exposes an **OpenAI-style** interface
compatible with the rest of *chuk-llm*.

Key points
----------
*   Configuration-driven capabilities from YAML instead of hardcoded assumptions
*   Converts ChatML → Watson X Messages format (tools / multimodal, …)
*   Maps Watson X replies back to the common `{response, tool_calls}` schema
*   **Real Streaming** - uses Watson X's native streaming API
*   **MCP Tool Compatibility** - handles MCP-style tool names with sanitization
"""
from __future__ import annotations
import json
import logging
import os
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple

# llm
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# providers
from ..core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin
from ._config_mixin import ConfigAwareProviderMixin

log = logging.getLogger(__name__)
if os.getenv("LOGLEVEL"):
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())

# ────────────────────────── helpers ──────────────────────────


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:  # noqa: D401 - util
    """Get *key* from dict **or** attribute-style object; fallback to *default*."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _parse_watsonx_response(resp) -> Dict[str, Any]:  # noqa: D401 - small helper
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


class WatsonXLLMClient(ConfigAwareProviderMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-aware adapter around the *ibm-watsonx-ai* SDK that gets
    all capabilities from unified YAML configuration.
    
    Includes MCP tool name sanitization for compatibility with IBM's
    tool naming requirements (typically similar to other enterprise providers).
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-3-8b-instruct",
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        watsonx_ai_url: Optional[str] = None,
        space_id: Optional[str] = None,
    ) -> None:
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, "watsonx", model)
        
        self.model = model
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.space_id = space_id or os.getenv("WATSONX_SPACE_ID")
        self.watsonx_ai_url = watsonx_ai_url or os.getenv("WATSONX_AI_URL", "https://us-south.ml.cloud.ibm.com")
        
        # Set up credentials
        credentials = Credentials(
            url=self.watsonx_ai_url,
            api_key=api_key or os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")
        )
        
        self.client = APIClient(credentials)
        
        # Store current tool name mapping for response restoration
        self._current_name_mapping: Dict[str, str] = {}
        
        # Default parameters - can be overridden by configuration
        self.default_params = {
            "time_limit": 10000,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 1.0,
        }

    def _sanitize_tool_name_for_watsonx(self, name: str) -> str:
        """
        Convert MCP/OpenAI tool names to WatsonX-compatible format.
        
        IBM WatsonX typically has enterprise-grade restrictions similar to
        other enterprise providers. This sanitizes tool names to be safe.
        
        Examples:
            stdio.read_query -> stdio_read_query
            filesystem.read_file -> filesystem_read_file  
            mcp.server:get_data -> mcp_server_get_data
        """
        if not name:
            return name
            
        # Replace invalid characters with underscores (conservative approach)
        # WatsonX likely prefers alphanumeric + underscores/dashes
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure reasonable length (conservative 64 char limit)
        if len(sanitized) > 64:
            sanitized = sanitized[:64].rstrip('_')
            
        return sanitized

    def _sanitize_tools_for_watsonx(self, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Sanitize tool names and create a mapping for response processing.
        
        Returns:
            tuple: (sanitized_tools, name_mapping)
                - sanitized_tools: Tools with WatsonX-compatible names
                - name_mapping: Dict mapping sanitized_name -> original_name
        """
        if not tools:
            return tools, {}
            
        sanitized_tools = []
        name_mapping = {}
        
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                original_name = tool["function"]["name"]
                sanitized_name = self._sanitize_tool_name_for_watsonx(original_name)
                
                # Store the mapping
                name_mapping[sanitized_name] = original_name
                
                # Create sanitized tool
                sanitized_tool = tool.copy()
                sanitized_tool["function"] = tool["function"].copy()
                sanitized_tool["function"]["name"] = sanitized_name
                
                sanitized_tools.append(sanitized_tool)
                
                # Log the sanitization if it changed
                if sanitized_name != original_name:
                    log.debug(f"Sanitized WatsonX tool name: {original_name} -> {sanitized_name}")
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
                    log.debug(f"Restored WatsonX tool name: {sanitized_name} -> {original_name}")
            else:
                restored_tool_calls.append(tool_call)
                
        restored_response["tool_calls"] = restored_tool_calls
        return restored_response

    def _sanitize_tool_names(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """
        Tool name sanitization for WatsonX.
        
        IBM WatsonX is an enterprise platform that may have tool naming
        restrictions. This provides sanitization similar to Mistral/Anthropic
        to ensure compatibility.
        
        Args:
            tools: List of tool definitions in OpenAI format
            
        Returns:
            Sanitized tools list with name mapping stored for restoration
        """
        if not tools:
            return tools
            
        log.debug(f"Sanitizing {len(tools)} tools for WatsonX compatibility")
        
        # Store the mapping for later restoration
        sanitized_tools, self._current_name_mapping = self._sanitize_tools_for_watsonx(tools)
        
        return sanitized_tools

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with WatsonX-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add WatsonX-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "watsonx_specific": {
                    "project_id": self.project_id,
                    "space_id": self.space_id,
                    "watsonx_ai_url": self.watsonx_ai_url,
                    "model_family": self._detect_model_family(),
                    "enterprise_features": True,
                },
                "tool_name_requirements": "enterprise_safe",  # Conservative approach
                "mcp_compatibility": "requires_sanitization",  # Similar to other enterprise providers
                "parameter_mapping": {
                    "temperature": "temperature",
                    "max_tokens": "max_tokens",
                    "top_p": "top_p",
                    "stream": "stream",
                    "time_limit": "time_limit"
                },
                "watsonx_parameters": [
                    "time_limit", "include_stop_sequence", "return_options"
                ]
            })
        
        return info

    def _detect_model_family(self) -> str:
        """Detect model family for WatsonX-specific optimizations"""
        model_lower = self.model.lower()
        if "llama" in model_lower:
            return "llama"
        elif "granite" in model_lower:
            return "granite"
        elif "mistral" in model_lower:
            return "mistral"
        elif "codellama" in model_lower:
            return "codellama"
        else:
            return "unknown"

    def _get_model_inference(self, params: Optional[Dict[str, Any]] = None) -> ModelInference:
        """Create a ModelInference instance with configuration-aware parameters."""
        # Start with defaults and apply configuration limits
        merged_params = {**self.default_params}
        if params:
            merged_params.update(params)
        
        # Apply configuration-based parameter validation
        validated_params = self.validate_parameters(**merged_params)
        
        return ModelInference(
            model_id=self.model,
            api_client=self.client,
            params=validated_params,
            project_id=self.project_id,
            space_id=self.space_id,
            verify=False
        )

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
        
        # Check vision support
        has_vision = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(item, dict) and item.get("type") in ["image", "image_url"] for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.warning(f"Vision content detected but {self.model} doesn't support vision according to configuration")
        
        # Check multimodal support
        has_multimodal = any(
            isinstance(msg.get("content"), list)
            for msg in messages
        )
        if has_multimodal and not self.supports_feature("multimodal"):
            log.info(f"Multimodal content will be simplified - {self.model} has limited multimodal support")
        
        # Validate parameters using configuration
        validated_kwargs = self.validate_parameters(**validated_kwargs)
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    # ── tool schema helpers ─────────────────────────────────

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tools to Watson X format.
        
        Note: Tool names should already be sanitized by _sanitize_tool_names
        before reaching this method.
        """
        if not tools:
            return []

        converted: List[Dict[str, Any]] = []
        for entry in tools:
            fn = entry.get("function", entry)
            try:
                converted.append({
                    "type": "function",
                    "function": {
                        "name": fn["name"],  # Should already be sanitized
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or fn.get("input_schema") or {},
                    }
                })
            except Exception as exc:  # pragma: no cover - permissive fallback
                log.debug("Tool schema error (%s) - using permissive schema", exc)
                converted.append({
                    "type": "function",
                    "function": {
                        "name": fn.get("name", f"tool_{uuid.uuid4().hex[:6]}"),
                        "description": fn.get("description", ""),
                        "parameters": {"type": "object", "additionalProperties": True},
                    }
                })
        return converted

    def _format_messages_for_watsonx(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages for Watson X API with configuration-aware processing."""
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                # Check if system messages are supported
                if self.supports_feature("system_messages"):
                    formatted.append({
                        "role": "system",
                        "content": content
                    })
                else:
                    # Fallback: convert to user message
                    log.warning(f"System messages not supported by {self.model}, converting to user message")
                    formatted.append({
                        "role": "user",
                        "content": f"System: {content}"
                    })
            elif role == "user":
                if isinstance(content, str):
                    formatted.append({
                        "role": "user",
                        "content": [{"type": "text", "text": content}]
                    })
                elif isinstance(content, list):
                    # Handle multimodal content for Watson X with configuration awareness
                    has_images = any(item.get("type") in ["image", "image_url"] for item in content)
                    
                    if has_images and not self.supports_feature("vision"):
                        # Extract only text content
                        text_content = " ".join([
                            item.get("text", "") for item in content 
                            if item.get("type") == "text"
                        ])
                        formatted.append({
                            "role": "user",
                            "content": [{"type": "text", "text": text_content or "[Image content removed - not supported by model]"}]
                        })
                    else:
                        # Process multimodal content normally
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
                            elif item.get("type") == "image_url":
                                # Pass through image_url format
                                watsonx_content.append(item)
                        
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
                    # Check if tools are supported
                    if self.supports_feature("tools"):
                        formatted.append({
                            "role": "assistant",
                            "tool_calls": msg["tool_calls"]
                        })
                    else:
                        # Convert tool calls to text response
                        log.warning(f"Tool calls detected but {self.model} doesn't support tools according to configuration")
                        tool_text = f"{content or ''}\n\nNote: Tool calls were requested but not supported by this model."
                        formatted.append({
                            "role": "assistant",
                            "content": tool_text
                        })
                else:
                    formatted.append({
                        "role": "assistant",
                        "content": content
                    })
            elif role == "tool":
                # Tool response messages - only include if tools are supported
                if self.supports_feature("tools"):
                    formatted.append({
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id"),
                        "content": content
                    })
                else:
                    # Convert tool response to user message
                    formatted.append({
                        "role": "user",
                        "content": f"Tool result: {content or ''}"
                    })

        return formatted

    def _parse_watsonx_response_with_restoration(self, resp, name_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Convert Watson X response to standard format and restore tool names"""
        # Use the standard parser first
        result = _parse_watsonx_response(resp)
        
        # Restore original tool names if we have a mapping
        if name_mapping and result.get("tool_calls"):
            result = self._restore_tool_names_in_response(result, name_mapping)
        
        return result

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
        Configuration-aware completion generation with streaming support and MCP tool compatibility.
        
        • stream=False → returns awaitable that resolves to standardised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        
        Includes MCP tool name sanitization for enterprise compatibility.
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **extra
        )
        
        # Apply max_tokens if provided
        if max_tokens:
            validated_kwargs["max_tokens"] = max_tokens
        
        # Sanitize tool names and convert to WatsonX format (stores mapping internally)
        validated_tools = self._sanitize_tool_names(validated_tools)
        watsonx_tools = self._convert_tools(validated_tools)
        
        # Format messages with configuration-aware processing
        formatted_messages = self._format_messages_for_watsonx(validated_messages)

        log.debug(f"Watson X payload: model={self.model}, messages={len(formatted_messages)}, tools={len(watsonx_tools)}")

        # --- streaming: use Watson X streaming -------------------------
        if validated_stream:
            return self._stream_completion_async(formatted_messages, watsonx_tools, validated_kwargs)

        # --- non-streaming: use regular completion ----------------------
        return self._regular_completion(formatted_messages, watsonx_tools, validated_kwargs)

    async def _stream_completion_async(
        self, 
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming using Watson X with configuration awareness and tool name restoration.
        """
        try:
            log.debug(f"Starting Watson X streaming for model: {self.model}")
            
            model = self._get_model_inference(params)
            
            # Use Watson X streaming (only use tools if supported)
            if tools and self.supports_feature("tools"):
                # For tool calling, we need to use chat_stream with tools
                stream_response = model.chat_stream(messages=messages, tools=tools)
            else:
                # For regular chat, use chat_stream
                stream_response = model.chat_stream(messages=messages)
            
            chunk_count = 0
            for chunk in stream_response:
                chunk_count += 1
                
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
                        tool_calls = delta.get("tool_calls", []) if self.supports_feature("tools") else []
                        
                        # Create chunk response
                        chunk_response = {
                            "response": content,
                            "tool_calls": tool_calls
                        }
                        
                        # Restore tool names if needed
                        if self._current_name_mapping and tool_calls:
                            chunk_response = self._restore_tool_names_in_response(
                                chunk_response, 
                                self._current_name_mapping
                            )
                        
                        yield chunk_response
                    else:
                        yield {
                            "response": str(chunk),
                            "tool_calls": []
                        }
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    import asyncio
                    await asyncio.sleep(0)
            
            log.debug(f"Watson X streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            log.error(f"Error in Watson X streaming: {e}")
            
            # Check if it's a tool name validation error
            error_str = str(e).lower()
            if "function" in error_str and ("name" in error_str or "invalid" in error_str):
                log.error(f"Tool name sanitization may have failed: {e}")
                log.error(f"Current mapping: {self._current_name_mapping}")
            
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
        """
        Non-streaming completion using Watson X with configuration awareness and tool name restoration.
        """
        try:
            log.debug(f"Starting Watson X completion for model: {self.model}")
            
            model = self._get_model_inference(params)
            
            # Use tools only if supported
            if tools and self.supports_feature("tools"):
                # Use chat with tools
                resp = model.chat(messages=messages, tools=tools)
            else:
                # Use regular chat
                resp = model.chat(messages=messages)
            
            # Parse response and restore tool names
            result = self._parse_watsonx_response_with_restoration(resp, self._current_name_mapping)
            
            log.debug(f"Watson X completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            log.error(f"Error in Watson X completion: {e}")
            
            # Check if it's a tool name validation error
            error_str = str(e).lower()
            if "function" in error_str and ("name" in error_str or "invalid" in error_str):
                log.error(f"Tool name sanitization may have failed: {e}")
                log.error(f"Current mapping: {self._current_name_mapping}")
            
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping
        self._current_name_mapping = {}
        # Watson X client cleanup if needed
        pass