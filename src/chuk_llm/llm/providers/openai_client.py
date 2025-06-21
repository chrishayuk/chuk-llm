# chuk_llm/llm/providers/openai_client.py
"""
OpenAI chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around the official `openai` SDK that uses the unified
configuration system for all capabilities instead of hardcoding.
"""
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import openai
import logging
import asyncio
import time

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
                import json
                import uuid
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

    async def _stream_from_async(
        self,
        async_stream,
        normalize_chunk: Optional[callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced streaming with configuration-aware error handling.
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
                                content = str(delta.content)  # Ensure string type
                                total_content += content
                            
                            # Handle tool calls in delta
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                import json
                                import uuid
                                for tc in delta.tool_calls:
                                    try:
                                        if hasattr(tc, 'function') and tc.function:
                                            tool_calls.append({
                                                "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                                "type": "function",
                                                "function": {
                                                    "name": getattr(tc.function, 'name', ''),
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
                                normalized = self._normalize_message(message)
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
                            tool_calls = chunk['tool_calls']
                
                except Exception as chunk_error:
                    log.warning(f"Error processing chunk {chunk_count}: {chunk_error}")
                    # Continue processing other chunks
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
                
                # Always yield chunks (even empty ones for timing)
                yield result
            
            # Final statistics and validation
            log.debug(f"[{self.detected_provider}] Streaming completed: "
                     f"{chunk_count} chunks, {len(total_content)} total characters")
            
            # Log completion without provider-specific assumptions
            if len(total_content) == 0 and chunk_count > 0:
                log.info(f"{self.detected_provider} streaming completed with no content - this may be normal for some requests")
                        
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
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Sanitize tool names (from mixin)
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
        Enhanced async streaming with configuration-aware retry logic.
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
                
                # Stream results
                chunk_count = 0
                async for result in self._stream_from_async(response_stream):
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
        """Enhanced non-streaming completion using configuration."""
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
            
            # Use enhanced normalization
            result = self._normalize_message(resp.choices[0].message)
            
            # Log result without making assumptions about what's "normal"
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