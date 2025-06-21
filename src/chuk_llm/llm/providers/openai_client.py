# chuk_llm/llm/providers/openai_client.py
"""
OpenAI chat-completion adapter
"""
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import openai
import logging

# mixins
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin

# base
from ..core.base import BaseLLMClient

log = logging.getLogger(__name__)

class OpenAILLMClient(OpenAIStyleMixin, BaseLLMClient):
    """
    Thin wrapper around the official `openai` SDK with response parsing.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_base = api_base
        
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and provider capabilities"""
        # Determine provider based on api_base
        provider = "openai"
        if self.api_base:
            if "deepseek" in self.api_base.lower():
                provider = "deepseek"
            elif "groq" in self.api_base.lower():
                provider = "groq"
            elif "together" in self.api_base.lower():
                provider = "together"
            elif "perplexity" in self.api_base.lower():
                provider = "perplexity"
            # Add more provider detection as needed
        
        # Basic capabilities - can be enhanced based on model/provider
        supports_tools = True
        supports_vision = False
        supports_streaming = True
        supports_json_mode = True
        supports_system_messages = True
        
        # Adjust based on provider
        if provider == "deepseek":
            supports_tools = False  # DeepSeek doesn't support function calling yet
            supports_vision = False
        
        # Adjust based on model name
        if "vision" in self.model.lower() or "gpt-4o" in self.model.lower():
            supports_vision = True
        
        return {
            "provider": provider,
            "model": self.model,
            "api_base": self.api_base,
            "supports_streaming": supports_streaming,
            "supports_tools": supports_tools,
            "supports_vision": supports_vision,
            "supports_json_mode": supports_json_mode,
            "supports_system_messages": supports_system_messages,
        }

    def _normalise_message(self, msg) -> Dict[str, Any]:
        """
        Convert OpenAI response message to standard format.
        Handles different response formats from different models.
        """
        # Handle both streaming and non-streaming message formats
        content = None
        tool_calls = []
        
        # Extract content - handle multiple possible formats
        if hasattr(msg, 'content'):
            content = msg.content
        elif hasattr(msg, 'message') and hasattr(msg.message, 'content'):
            content = msg.message.content
        elif isinstance(msg, dict):
            content = msg.get('content')
        
        # Handle empty/None content - this might be the issue with DeepSeek
        if content is None:
            content = ""
        
        # Extract tool calls
        raw_tool_calls = None
        if hasattr(msg, 'tool_calls'):
            raw_tool_calls = msg.tool_calls
        elif hasattr(msg, 'message') and hasattr(msg.message, 'tool_calls'):
            raw_tool_calls = msg.message.tool_calls
        elif isinstance(msg, dict):
            raw_tool_calls = msg.get('tool_calls')
        
        if raw_tool_calls:
            import json
            import uuid
            for tc in raw_tool_calls:
                tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                try:
                    args = tc.function.arguments
                    args_j = (
                        json.dumps(json.loads(args))
                        if isinstance(args, str)
                        else json.dumps(args)
                    )
                except (TypeError, json.JSONDecodeError):
                    args_j = "{}"

                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": args_j,
                    },
                })
        
        # Return normalized format
        result = {
            "response": content if not tool_calls else None,
            "tool_calls": tool_calls
        }
        
        # Debug logging for DeepSeek issues
        if content == "" or content is None:
            log.warning(f"Empty content detected - original message type: {type(msg)}")
            if hasattr(msg, '__dict__'):
                log.debug(f"Message attributes: {list(msg.__dict__.keys())}")
        
        log.debug(f"Normalized message: content={'None' if content is None else f'{len(str(content))} chars'}, tool_calls={len(tool_calls)}")
        return result

    async def _stream_from_async(
        self,
        async_stream,
        normalize_chunk: Optional[callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream from an async iterator with proper chunk handling for all models.
        """
        try:
            chunk_count = 0
            total_content = ""  # Track total content for debugging
            
            async for chunk in async_stream:
                chunk_count += 1
                
                # Handle different chunk formats
                content = ""
                tool_calls = []
                
                # Extract content from chunk - handle multiple formats
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    
                    # Handle delta format (most common for streaming)
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            total_content += content
                        
                        # Handle tool calls in delta
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            import json
                            import uuid
                            for tc in delta.tool_calls:
                                if hasattr(tc, 'function') and tc.function:
                                    tool_calls.append({
                                        "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                        "type": "function",
                                        "function": {
                                            "name": getattr(tc.function, 'name', ''),
                                            "arguments": getattr(tc.function, 'arguments', '') or ""
                                        }
                                    })
                    
                    # Handle message format (less common for streaming)
                    elif hasattr(choice, 'message'):
                        message = choice.message
                        if hasattr(message, 'content') and message.content:
                            content = message.content
                            total_content += content
                        
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            # Use the existing normalization logic
                            normalized = self._normalise_message(message)
                            tool_calls = normalized.get('tool_calls', [])
                
                # Create result chunk
                result = {
                    "response": content,
                    "tool_calls": tool_calls,
                }
                
                # Apply custom normalization if provided
                if normalize_chunk:
                    result = normalize_chunk(result, chunk)
                
                # Debug logging for first few chunks and issues
                if chunk_count <= 5 or (chunk_count % 50 == 0):
                    log.debug(f"Chunk {chunk_count}: content_len={len(content)}, total_content_len={len(total_content)}, tool_calls={len(tool_calls)}")
                
                # Only yield if there's actual content or tool calls
                if content or tool_calls:
                    yield result
                elif chunk_count <= 3:
                    # For first few chunks, yield even if empty (helps with timing)
                    yield result
            
            # Log final statistics
            log.debug(f"Streaming completed: {chunk_count} chunks, {len(total_content)} total characters")
            if len(total_content) == 0:
                log.warning("Streaming completed but no content was received - possible API issue")
                        
        except Exception as e:
            log.error(f"Error in streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    # ------------------------------------------------------------------ #
    # public API                                                          #
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
        Use native async streaming for real-time response with parsing.
        """
        tools = self._sanitize_tool_names(tools)

        # 1️⃣ streaming
        if stream:
            return self._stream_completion_async(messages, tools, **kwargs)

        # 2️⃣ one-shot
        return self._regular_completion(messages, tools, **kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Native async streaming using AsyncOpenAI.
        """
        try:
            log.debug(f"Starting streaming for model: {self.model}")
            log.debug(f"Messages: {len(messages)} messages")
            log.debug(f"Tools: {len(tools) if tools else 0} tools")
            log.debug(f"Kwargs: {list(kwargs.keys())}")
            
            # Make direct async call for real streaming
            response_stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **({"tools": tools} if tools else {}),
                stream=True,
                **kwargs
            )
            
            # Use the streaming method
            async for result in self._stream_from_async(response_stream):
                yield result
                
        except Exception as e:
            log.error(f"Error in streaming completion: {e}")
            yield {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming completion using async client."""
        try:
            log.debug(f"Starting non-streaming completion for model: {self.model}")
            log.debug(f"Messages: {len(messages)} messages")
            log.debug(f"Tools: {len(tools) if tools else 0} tools")
            log.debug(f"Kwargs: {list(kwargs.keys())}")
            
            resp = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **({"tools": tools} if tools else {}),
                stream=False,
                **kwargs
            )
            
            # Debug the raw response
            if hasattr(resp, 'choices') and resp.choices:
                choice = resp.choices[0]
                log.debug(f"Response choice type: {type(choice)}")
                if hasattr(choice, 'message'):
                    log.debug(f"Message type: {type(choice.message)}")
                    log.debug(f"Message content: {getattr(choice.message, 'content', 'NO CONTENT')}")
            
            # Use the normalization method
            result = self._normalise_message(resp.choices[0].message)
            log.debug(f"Non-streaming result: {result}")
            
            # Additional check for empty responses
            if not result.get("response") and not result.get("tool_calls"):
                log.warning("Empty response received from API")
                result["response"] = "[No response received from API]"
            
            return result
            
        except Exception as e:
            log.error(f"Error in regular completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }