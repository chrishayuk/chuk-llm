# chuk_llm/llm/providers/groq_client.py
"""
Groq chat-completion adapter with enhanced function calling error handling
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import json

from groq import AsyncGroq

# providers
from chuk_llm.llm.core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin

log = logging.getLogger(__name__)


class GroqAILLMClient(OpenAIStyleMixin, BaseLLMClient):
    """
    Adapter around `groq` SDK compatible with MCP-CLI's BaseLLMClient.
    Enhanced with robust function calling error handling.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        
        # ✅ FIX: Provide correct default base URL for Groq
        groq_base_url = api_base or "https://api.groq.com/openai/v1"
        
        log.debug(f"Initializing Groq client with base_url: {groq_base_url}")
        
        # Use AsyncGroq for real streaming support
        self.async_client = AsyncGroq(
            api_key=api_key,
            base_url=groq_base_url
        )
        
        # Keep sync client for backwards compatibility if needed
        from groq import Groq
        self.client = Groq(
            api_key=api_key,
            base_url=groq_base_url
        )

    # ──────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────
    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Real streaming support without buffering with enhanced error handling.
        
        • stream=False → returns awaitable that resolves to single normalised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """
        tools = self._sanitize_tool_names(tools)

        if stream:
            # Return async generator directly for real streaming
            return self._stream_completion_async(messages, tools or [], **kwargs)

        # non-streaming path
        return self._regular_completion(messages, tools or [], **kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real streaming using AsyncGroq with enhanced error handling.
        """
        try:
            log.debug("Starting Groq streaming...")
            
            # Enhanced messages for better function calling with Groq
            enhanced_messages = self._enhance_messages_for_groq(messages, tools)
            
            # Use async client for real streaming
            response_stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=enhanced_messages,
                tools=tools if tools else None,
                stream=True,
                **kwargs
            )
            
            chunk_count = 0
            # Yield chunks immediately as they arrive from Groq
            async for chunk in response_stream:
                chunk_count += 1
                
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Extract content and tool calls
                    content = delta.content or ""
                    tool_calls = getattr(delta, "tool_calls", [])
                    
                    # Only yield if we have actual content or tool calls
                    if content or tool_calls:
                        yield {
                            "response": content,
                            "tool_calls": tool_calls,
                        }
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            log.debug(f"Groq streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            error_str = str(e)
            
            # Handle Groq function calling errors in streaming
            if "Failed to call a function" in error_str and tools:
                log.warning(f"Groq streaming function calling failed, retrying without tools")
                
                # Retry without tools as fallback
                try:
                    response_stream = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **kwargs
                    )
                    
                    chunk_count = 0
                    async for chunk in response_stream:
                        chunk_count += 1
                        
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            content = delta.content or ""
                            
                            if content:
                                yield {
                                    "response": content,
                                    "tool_calls": [],
                                }
                        
                        if chunk_count % 10 == 0:
                            await asyncio.sleep(0)
                    
                    # Add final note about tools being disabled
                    yield {
                        "response": "\n\n[Note: Function calling disabled due to provider limitation]",
                        "tool_calls": [],
                    }
                    
                except Exception as retry_error:
                    log.error(f"Groq streaming retry failed: {retry_error}")
                    yield {
                        "response": f"Streaming error: {str(retry_error)}",
                        "tool_calls": [],
                        "error": True
                    }
            else:
                log.error(f"Error in Groq streaming: {e}")
                yield {
                    "response": f"Streaming error: {str(e)}",
                    "tool_calls": [],
                    "error": True
                }

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming completion with enhanced Groq function calling error handling."""
        try:
            log.debug(f"Groq regular completion - model: {self.model}, tools: {len(tools) if tools else 0}")
            
            # Enhanced messages for better function calling with Groq
            if tools:
                enhanced_messages = self._enhance_messages_for_groq(messages, tools)
                
                resp = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=enhanced_messages,
                    tools=tools,
                    stream=False,
                    **kwargs
                )
            else:
                resp = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    **kwargs
                )
                
            return self._normalise_message(resp.choices[0].message)
            
        except Exception as e:
            error_str = str(e)
            
            # Handle Groq function calling errors specifically
            if "Failed to call a function" in error_str and tools:
                log.warning(f"Groq function calling failed, retrying without tools: {error_str}")
                
                # Retry without tools as fallback
                try:
                    resp = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=False,
                        **kwargs
                    )
                    result = self._normalise_message(resp.choices[0].message)
                    
                    # Add a note that tools were disabled due to Groq limitation
                    original_response = result.get("response", "")
                    result["response"] = (original_response + 
                                       "\n\n[Note: Function calling disabled due to provider limitation]")
                    return result
                    
                except Exception as retry_error:
                    log.error(f"Groq retry also failed: {retry_error}")
                    return {
                        "response": f"Error: {str(retry_error)}",
                        "tool_calls": [],
                        "error": True
                    }
            else:
                log.error(f"Error in Groq completion: {e}")
                return {
                    "response": f"Error: {str(e)}",
                    "tool_calls": [],
                    "error": True
                }

    def _enhance_messages_for_groq(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance messages with better instructions for Groq function calling.
        Groq models need explicit guidance for proper function calling.
        """
        if not tools:
            return messages
        
        enhanced_messages = messages.copy()
        
        # Create function calling guidance
        function_names = [tool.get("function", {}).get("name", "unknown") for tool in tools]
        guidance = (
            f"You have access to the following functions: {', '.join(function_names)}. "
            "When calling functions:\n"
            "1. Use proper JSON format for arguments\n"
            "2. Ensure all required parameters are provided\n"
            "3. Use exact parameter names as specified\n"
            "4. Call functions when appropriate to help answer the user's question"
        )
        
        # Add or enhance system message
        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            enhanced_messages[0]["content"] = enhanced_messages[0]["content"] + "\n\n" + guidance
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": guidance
            })
        
        return enhanced_messages

    def _validate_tool_call_arguments(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validate tool call arguments to prevent Groq function calling errors.
        """
        try:
            if "function" not in tool_call:
                return False
            
            function = tool_call["function"]
            if "arguments" not in function:
                return False
            
            # Try to parse arguments as JSON
            args = function["arguments"]
            if isinstance(args, str):
                json.loads(args)  # This will raise if invalid JSON
            elif not isinstance(args, dict):
                return False
            
            return True
            
        except (json.JSONDecodeError, TypeError, KeyError):
            return False