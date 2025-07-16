# chuk_llm/llm/providers/openai_client.py
"""
OpenAI chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around the official `openai` SDK that uses the unified
configuration system for all capabilities instead of hardcoding.

Includes MCP tool name sanitization with bidirectional mapping to ensure
original tool names are preserved in responses for seamless MCP compatibility.
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

# base
from ..core.base import BaseLLMClient

log = logging.getLogger(__name__)

class OpenAILLMClient(ConfigAwareProviderMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-driven wrapper around the official `openai` SDK that gets
    all capabilities from the unified YAML configuration.
    
    Includes MCP tool name sanitization with bidirectional mapping:
    - Sanitizes MCP names (stdio.read_query -> stdio_read_query) for API compatibility
    - Restores original names in responses for seamless MCP experience
    - Works with all OpenAI-compatible providers (OpenAI, DeepSeek, Groq, etc.)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Detect provider from api_base for configuration lookup
        detected_provider = self._detect_provider_name(api_base)
        
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, detected_provider, model)
        
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
        
        # Store current tool name mapping for response restoration
        self._current_name_mapping: Dict[str, str] = {}
        
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
        
        # Add OpenAI-specific metadata only if no error
        if not info.get("error"):
            info.update({
                "api_base": self.api_base,
                "detected_provider": self.detected_provider,
                "openai_compatible": True,
                "tool_name_requirements": "aggressive_sanitization",
                "mcp_compatibility": "requires_sanitization_with_restoration",
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

    def _sanitize_tools_for_openai(self, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Sanitize tool names and create a mapping for response processing.
        
        Uses aggressive sanitization for compatibility with all OpenAI-compatible providers:
        - Replaces dots, colons, and special chars with underscores
        - Ensures names start with letter/underscore
        - Truncates to 64 characters
        - Works with OpenAI, DeepSeek, Groq, Together AI, etc.
        
        Returns:
            tuple: (sanitized_tools, name_mapping)
                - sanitized_tools: Tools with provider-compatible names
                - name_mapping: Dict mapping sanitized_name -> original_name
        """
        if not tools:
            return tools, {}
            
        sanitized_tools = []
        name_mapping = {}
        
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                tool_copy = tool.copy()
                func = tool_copy["function"].copy()
                
                original_name = func.get("name", "")
                if original_name:
                    # Aggressive sanitization for all OpenAI-compatible providers
                    # Replace any non-alphanumeric (except underscore/dash) with underscore
                    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)
                    
                    # Remove multiple consecutive underscores
                    sanitized_name = re.sub(r'_+', '_', sanitized_name)
                    
                    # Ensure it starts with a letter or underscore
                    if sanitized_name and not sanitized_name[0].isalpha() and sanitized_name[0] != '_':
                        sanitized_name = '_' + sanitized_name
                    
                    # Truncate to 64 characters and clean up trailing underscores
                    sanitized_name = sanitized_name[:64].rstrip('_')
                    
                    # Ensure we have a valid name
                    if not sanitized_name:
                        sanitized_name = "unnamed_function"
                    
                    # Store the mapping
                    name_mapping[sanitized_name] = original_name
                    
                    # Update the tool
                    func["name"] = sanitized_name
                    
                    # Log the sanitization if it changed
                    if sanitized_name != original_name:
                        log.debug(f"Sanitized {self.detected_provider} tool name: {original_name} -> {sanitized_name}")
                
                tool_copy["function"] = func
                sanitized_tools.append(tool_copy)
            else:
                # Non-function tools pass through unchanged
                sanitized_tools.append(tool)
                
        return sanitized_tools, name_mapping

    def _restore_tool_names_in_response(self, response: Dict[str, Any], name_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Restore original tool names in the response using the mapping.
        
        This makes the sanitization transparent to users - they send MCP-style names
        and receive MCP-style names back, even though internally we use sanitized names.
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
                    log.debug(f"Restored {self.detected_provider} tool name: {sanitized_name} -> {original_name}")
            else:
                restored_tool_calls.append(tool_call)
                
        restored_response["tool_calls"] = restored_tool_calls
        return restored_response

    def _sanitize_tool_names(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """
        Sanitize tool names with mapping storage for later restoration.
        
        This replaces the previous implementation to add bidirectional mapping support
        for seamless MCP tool name handling across all OpenAI-compatible providers.
        """
        if not tools:
            return tools
        
        log.debug(f"Sanitizing {len(tools)} tools for {self.detected_provider} compatibility")
        
        # Store the mapping for later restoration
        sanitized_tools, self._current_name_mapping = self._sanitize_tools_for_openai(tools)
        
        return sanitized_tools

    def _normalize_message_with_restoration(self, msg, name_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Enhanced message normalization with tool name restoration.
        """
        # Use the existing normalization logic
        result = self._normalize_message(msg)
        
        # Restore original tool names if we have a mapping
        if name_mapping and result.get("tool_calls"):
            result = self._restore_tool_names_in_response(result, name_mapping)
        
        return result

    def _normalize_message(self, msg) -> Dict[str, Any]:
        """
        Enhanced message normalization with configuration-aware debugging.
        """
        content = None
        tool_calls = []
        
        # Enhanced debug logging with provider context
        log.debug(f"[{self.detected_provider}] Normalizing message - type: {type(msg)}")
        if hasattr(msg, '__dict__'):
            log.debug(f"[{self.detected_provider}] Message attributes: {list(msg.__dict__.keys())}")
        
        # Method 1: Standard content extraction
        try:
            if hasattr(msg, 'content'):
                content = msg.content
                if content is not None:
                    log.debug(f"Content extracted via direct attribute: {len(str(content))} chars")
        except Exception as e:
            log.debug(f"Direct content access failed: {e}")
        
        # Method 2: Nested message structure
        if content is None:
            try:
                if hasattr(msg, 'message') and hasattr(msg.message, 'content'):
                    content = msg.message.content
                    if content is not None:
                        log.debug(f"Content extracted via message wrapper: {len(str(content))} chars")
            except Exception as e:
                log.debug(f"Message wrapper access failed: {e}")
        
        # Method 3: Dict-style access
        if content is None:
            try:
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
                    if content is not None:
                        log.debug(f"Content extracted via dict access: {len(str(content))} chars")
            except Exception as e:
                log.debug(f"Dict content access failed: {e}")
        
        # Method 4: Provider-specific alternative fields (only if we couldn't get content)
        if content is None or (isinstance(content, str) and content.strip() == ""):
            alternative_fields = ['text', 'response', 'output', 'generated_text']
            for field in alternative_fields:
                try:
                    if hasattr(msg, field):
                        alt_content = getattr(msg, field)
                        if alt_content and isinstance(alt_content, str) and alt_content.strip():
                            content = alt_content
                            log.debug(f"Content found in alternative field '{field}': {len(content)} chars")
                            break
                except Exception as e:
                    log.debug(f"Alternative field '{field}' access failed: {e}")
                    continue
        
        # Handle tool calls with enhanced error handling
        try:
            raw_tool_calls = None
            
            # Try multiple ways to extract tool calls
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
                        
                        # Extract function details with error handling
                        if hasattr(tc, 'function'):
                            func = tc.function
                            func_name = getattr(func, 'name', 'unknown_function')
                            
                            # Handle arguments with robust JSON processing
                            args = getattr(func, 'arguments', '{}')
                            try:
                                if isinstance(args, str):
                                    # Validate and reformat JSON
                                    parsed_args = json.loads(args)
                                    args_j = json.dumps(parsed_args)
                                elif isinstance(args, dict):
                                    args_j = json.dumps(args)
                                else:
                                    log.warning(f"Unexpected argument type in tool call: {type(args)}")
                                    args_j = "{}"
                            except json.JSONDecodeError:
                                log.warning(f"Invalid JSON in tool call arguments: {args}")
                                args_j = "{}"
                            
                            tool_calls.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": args_j,
                                },
                            })
                        else:
                            log.warning(f"Tool call missing function attribute: {tc}")
                            
                    except Exception as e:
                        log.warning(f"Failed to process tool call {tc}: {e}")
                        continue
        except Exception as e:
            log.warning(f"Failed to extract tool calls: {e}")
        
        # Content validation with provider-aware handling
        if content is None:
            content = ""
            if not tool_calls:  # Only warn if no tool calls either
                log.warning(f"No content found in {self.detected_provider} response - message type: {type(msg)}")
                if hasattr(msg, '__dict__'):
                    log.debug(f"Available attributes: {list(msg.__dict__.keys())}")
        elif isinstance(content, str) and content.strip() == "" and not tool_calls:
            log.warning(f"Empty content with no tool calls from {self.detected_provider}")
        
        # Determine response format based on content and tool calls
        if tool_calls:
            # If we have tool calls, response should be None unless there's meaningful content
            response_value = content if content and content.strip() else None
        else:
            # No tool calls, return content (even if empty)
            response_value = content
        
        result = {
            "response": response_value,
            "tool_calls": tool_calls
        }
        
        # Enhanced logging for troubleshooting
        log.debug(f"Normalized {self.detected_provider} message: "
                 f"response={len(str(response_value)) if response_value else 0} chars, "
                 f"tool_calls={len(tool_calls)}")
        
        return result

    async def _stream_from_async_with_restoration(
        self,
        async_stream,
        name_mapping: Dict[str, str] = None,
        normalize_chunk: Optional[callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced streaming with tool name restoration.
        """
        try:
            chunk_count = 0
            total_content = ""
            last_chunk_time = time.time()
            
            async for chunk in async_stream:
                chunk_count += 1
                current_time = time.time()
                
                # Handle different chunk formats
                content = ""
                tool_calls = []
                
                try:
                    # Standard OpenAI streaming format
                    if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        
                        # Handle delta format (most common for streaming)
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            
                            # Extract content with null safety
                            if hasattr(delta, 'content') and delta.content is not None:
                                content = str(delta.content)
                                total_content += content
                            
                            # Handle tool calls in delta with name restoration
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                for tc in delta.tool_calls:
                                    try:
                                        if hasattr(tc, 'function') and tc.function:
                                            sanitized_name = getattr(tc.function, 'name', '')
                                            original_name = name_mapping.get(sanitized_name, sanitized_name) if name_mapping else sanitized_name
                                            
                                            tool_calls.append({
                                                "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                                "type": "function",
                                                "function": {
                                                    "name": original_name,  # Restored name
                                                    "arguments": getattr(tc.function, 'arguments', '') or ""
                                                }
                                            })
                                    except Exception as e:
                                        log.debug(f"Error processing streaming tool call: {e}")
                                        continue
                        
                        # Handle full message format (less common for streaming)
                        elif hasattr(choice, 'message') and choice.message:
                            message = choice.message
                            if hasattr(message, 'content') and message.content:
                                content = str(message.content)
                                total_content += content
                            
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                normalized = self._normalize_message_with_restoration(message, name_mapping)
                                tool_calls = normalized.get('tool_calls', [])
                    
                    # Provider-specific chunk formats
                    elif hasattr(chunk, 'content') and chunk.content:
                        content = str(chunk.content)
                        total_content += content
                    elif isinstance(chunk, dict):
                        if 'content' in chunk:
                            content = str(chunk['content'])
                            total_content += content
                        if 'tool_calls' in chunk:
                            # Restore names in chunk tool calls
                            chunk_tool_calls = chunk['tool_calls']
                            if name_mapping:
                                restored_chunk = {"tool_calls": chunk_tool_calls}
                                restored_chunk = self._restore_tool_names_in_response(restored_chunk, name_mapping)
                                tool_calls = restored_chunk['tool_calls']
                            else:
                                tool_calls = chunk_tool_calls
                
                except Exception as chunk_error:
                    log.warning(f"Error processing chunk {chunk_count}: {chunk_error}")
                    content = ""
                    tool_calls = []
                
                # Create result chunk
                result = {
                    "response": content,
                    "tool_calls": tool_calls,
                }
                
                # Apply custom normalization if provided
                if normalize_chunk:
                    try:
                        result = normalize_chunk(result, chunk)
                    except Exception as e:
                        log.debug(f"Custom normalization failed: {e}")
                
                # Enhanced debug logging
                if chunk_count <= 5 or (chunk_count % 50 == 0) or (current_time - last_chunk_time > 2.0):
                    log.debug(f"[{self.detected_provider}] Chunk {chunk_count}: "
                             f"content_len={len(content)}, "
                             f"total_chars={len(total_content)}, "
                             f"tool_calls={len(tool_calls)}, "
                             f"chunk_interval={current_time - last_chunk_time:.2f}s")
                    last_chunk_time = current_time
                
                yield result
            
            # Final statistics
            log.debug(f"[{self.detected_provider}] Streaming completed: "
                     f"{chunk_count} chunks, {len(total_content)} total characters")
                        
        except Exception as e:
            log.error(f"Error in {self.detected_provider} streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    def _adjust_parameters_for_provider(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters using configuration instead of hardcoded rules.
        """
        adjusted = kwargs.copy()
        
        try:
            # Use the configuration-aware parameter validation
            adjusted = self.validate_parameters(**adjusted)
            
            # Additional OpenAI-specific parameter handling
            model_caps = self._get_model_capabilities()
            if model_caps:
                # Adjust max_tokens based on config if not already handled
                if 'max_tokens' in adjusted and model_caps.max_output_tokens:
                    if adjusted['max_tokens'] > model_caps.max_output_tokens:
                        log.debug(f"Adjusting max_tokens from {adjusted['max_tokens']} to {model_caps.max_output_tokens} for {self.detected_provider}")
                        adjusted['max_tokens'] = model_caps.max_output_tokens
        
        except Exception as e:
            log.debug(f"Could not adjust parameters using config: {e}")
            # Fallback: ensure max_tokens is set
            if 'max_tokens' not in adjusted:
                adjusted['max_tokens'] = 4096
        
        return adjusted

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
            # Could filter out vision content here if needed
        
        # Check JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            if not self.supports_feature("json_mode"):
                log.warning(f"JSON mode requested but {self.detected_provider}/{self.model} doesn't support JSON mode according to configuration")
                # Remove JSON mode request
                validated_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    # ------------------------------------------------------------------ #
    # Enhanced public API using configuration                            #
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
        Configuration-aware completion that validates capabilities before processing.
        Includes transparent MCP tool name sanitization and restoration.
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Sanitize tool names (stores mapping for restoration)
        validated_tools = self._sanitize_tool_names(validated_tools)
        
        # Use configuration-aware parameter adjustment
        validated_kwargs = self._adjust_parameters_for_provider(validated_kwargs)

        if validated_stream:
            return self._stream_completion_async(validated_messages, validated_tools, **validated_kwargs)
        else:
            return self._regular_completion(validated_messages, validated_tools, **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced async streaming with configuration-aware retry logic and tool name restoration.
        """
        # Get retry configuration from provider config if available
        max_retries = 1  # Default
        try:
            provider_config = self._get_provider_config()
            if provider_config:
                # Could add retry_count to provider config in future
                max_retries = 2 if self.detected_provider in ["deepseek", "perplexity"] else 1
        except:
            pass
        
        for attempt in range(max_retries + 1):
            try:
                log.debug(f"[{self.detected_provider}] Starting streaming (attempt {attempt + 1}): "
                         f"model={self.model}, messages={len(messages)}, tools={len(tools) if tools else 0}")
                
                # Create streaming request
                response_stream = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **({"tools": tools} if tools else {}),
                    stream=True,
                    **kwargs
                )
                
                # Stream results with name restoration
                chunk_count = 0
                async for result in self._stream_from_async_with_restoration(response_stream, self._current_name_mapping):
                    chunk_count += 1
                    yield result
                
                # Success - exit retry loop
                log.debug(f"[{self.detected_provider}] Streaming completed successfully with {chunk_count} chunks")
                return
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a retryable error
                is_retryable = any(pattern in error_str for pattern in [
                    "timeout", "connection", "network", "temporary", "rate limit"
                ])
                
                if attempt < max_retries and is_retryable:
                    wait_time = (attempt + 1) * 1.0  # 1s, 2s, 3s...
                    log.warning(f"[{self.detected_provider}] Streaming attempt {attempt + 1} failed: {e}. "
                               f"Retrying in {wait_time}s...")
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enhanced non-streaming completion with tool name restoration."""
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
            
            # Enhanced response debugging
            if hasattr(resp, 'choices') and resp.choices:
                choice = resp.choices[0]
                log.debug(f"[{self.detected_provider}] Response choice type: {type(choice)}")
                if hasattr(choice, 'message'):
                    message = choice.message
                    log.debug(f"[{self.detected_provider}] Message type: {type(message)}")
                    content_preview = getattr(message, 'content', 'NO CONTENT')
                    if content_preview:
                        log.debug(f"[{self.detected_provider}] Content preview: {str(content_preview)[:100]}...")
                    else:
                        log.debug(f"[{self.detected_provider}] No content in message")
            
            # Use enhanced normalization with name restoration
            result = self._normalize_message_with_restoration(resp.choices[0].message, self._current_name_mapping)
            
            # Log result
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
        # Reset name mapping
        self._current_name_mapping = {}
        
        # Cleanup OpenAI clients if needed
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        if hasattr(self.client, 'close'):
            self.client.close()