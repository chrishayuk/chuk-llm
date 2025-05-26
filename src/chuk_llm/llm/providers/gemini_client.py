# chuk_llm/llm/providers/gemini_client.py
"""
Google Gemini chat-completion adapter with proper async streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

# providers
from chuk_llm.llm.core.base import BaseLLMClient

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

# ───────────────────────────────────────────────────────── helpers ──────────

class GeminiAsyncStreamWrapper:
    """Wrapper to make Gemini's sync stream work with async for"""
    
    def __init__(self, sync_stream):
        self.sync_stream = sync_stream
        self._iterator = None
        self._exhausted = False
    
    def __aiter__(self):
        self._iterator = iter(self.sync_stream)
        self._exhausted = False
        return self
    
    async def __anext__(self):
        if self._exhausted:
            raise StopAsyncIteration
            
        def _get_next():
            try:
                return next(self._iterator)
            except StopIteration:
                return None  # Return None instead of raising
        
        try:
            chunk = await asyncio.to_thread(_get_next)
            if chunk is None:
                self._exhausted = True
                raise StopAsyncIteration
            return chunk
        except Exception as e:
            self._exhausted = True
            # Don't log normal end-of-stream as error
            log.debug(f"Stream ended: {e}")
            raise StopAsyncIteration


def _convert_messages_for_chat(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """Convert ChatML messages to Gemini chat format.

    Returns
    -------
    system_instruction : Optional[str]
    user_message      : str
    """
    system_txt: Optional[str] = None
    user_message: str = "Hello"  # Default fallback

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        log.debug("↻ msg[%d] role=%s keys=%s", i, role, list(msg.keys()))

        # ---------------- system -----------------------------------------
        if role == "system":
            if system_txt is None:
                system_txt = content if isinstance(content, str) else str(content)
            continue

        # ---------------- tool response ----------------------------------
        if role == "tool":
            # Convert tool response to user message for chat API
            fn_name = msg.get("name") or "tool"
            user_message = f"Tool {fn_name} result: {content}"
            continue

        # ---------------- assistant function-calls ------------------------
        if role == "assistant" and msg.get("tool_calls"):
            # Convert tool calls to assistant message
            tool_text = "I need to use tools: "
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name")
                args_raw = fn.get("arguments", "{}")
                tool_text += f"{name}({args_raw}) "
            # This becomes part of the conversation context
            continue

        # ---------------- normal messages --------------------------------
        if role in {"user", "assistant"}:
            if isinstance(content, str):
                if role == "user":
                    user_message = content
            elif isinstance(content, list):
                # Handle multimodal content - convert to text for chat API
                text_content = ""
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "image_url":
                            text_content += "[Image content]"
                
                if text_content and role == "user":
                    user_message = text_content
            elif content is not None and role == "user":
                user_message = str(content)
        else:
            log.debug("Skipping unsupported message: %s", msg)

    log.debug("System-instruction: %s", system_txt)
    log.debug("User message: %s", user_message[:100] + "..." if len(user_message) > 100 else user_message)
    return system_txt, user_message


def _convert_tools_to_gemini_format(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[gtypes.Tool]]:
    """Convert OpenAI-style tools to Gemini format"""
    if not tools:
        return None
    
    gemini_tools = []
    function_declarations = []
    
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            
            # Convert to Gemini function declaration format
            function_decl = {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {})
            }
            function_declarations.append(function_decl)
    
    if function_declarations:
        # Create Gemini Tool object
        gemini_tool = gtypes.Tool(function_declarations=function_declarations)
        gemini_tools.append(gemini_tool)
    
    return gemini_tools if gemini_tools else None


# ─────────────────────────────────────────────────── main adapter ───────────

class GeminiLLMClient(BaseLLMClient):
    """`google-genai` wrapper with proper async streaming following OpenAI pattern."""

    def __init__(self, model: str = "gemini-2.0-flash", *, api_key: Optional[str] = None) -> None:
        load_dotenv()
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY env var not set")

        self.model = model
        self.client = genai.Client(api_key=api_key)
        log.info("GeminiLLMClient initialised with model '%s'", model)

    # ---------------------------------------------------------------- OpenAI-style streaming
    
    def create_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None, 
        *, 
        stream: bool = False,
        **kwargs: Any
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Generate completion with proper streaming support following OpenAI pattern.
        
        • stream=False → returns awaitable that resolves to standardised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """
        log.debug("create_completion called – stream=%s", stream)
        
        if stream:
            return self._stream_completion_async(messages, tools, **kwargs)
        else:
            return self._regular_completion(messages, tools, **kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        OpenAI-style async streaming using proper async for pattern.
        """
        try:
            log.debug(f"Starting OpenAI-style streaming for model: {self.model}")
            
            # Extract content
            system_instruction, user_message = _convert_messages_for_chat(messages)
            gemini_tools = _convert_tools_to_gemini_format(tools)
            
            # Create config
            chat_config = {"model": self.model}
            if system_instruction:
                chat_config["system_instruction"] = system_instruction
            
            if gemini_tools:
                chat_config["config"] = gtypes.GenerateContentConfig(
                    tools=gemini_tools,
                    automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True)
                )
            
            # Create stream (like OpenAI's async call)
            def _create_gemini_stream():
                chat = self.client.chats.create(**chat_config)
                return chat.send_message_stream(user_message)
            
            gemini_stream = await asyncio.to_thread(_create_gemini_stream)
            
            # Use the streaming method (like OpenAI does)
            async for result in self._stream_from_gemini_async(gemini_stream):
                yield result
                
        except Exception as e:
            log.error(f"Error in OpenAI-style streaming: {e}")
            yield {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _stream_from_gemini_async(
        self,
        gemini_stream,
        normalize_chunk: Optional[callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream from Gemini async iterator with chunk handling (following OpenAI pattern).
        """
        try:
            chunk_count = 0
            
            # Wrap sync stream to make it async iterable
            async_stream = GeminiAsyncStreamWrapper(gemini_stream)
            
            # Use async for exactly like OpenAI's _stream_from_async_fixed
            async for chunk in async_stream:
                chunk_count += 1
                
                # Parse chunk content
                chunk_text, tool_calls = self._parse_gemini_chunk(chunk)
                
                # Create result
                result = {
                    "response": chunk_text,
                    "tool_calls": tool_calls,
                }
                
                # Apply custom normalization if provided (like OpenAI)
                if normalize_chunk:
                    result = normalize_chunk(result, chunk)
                
                # Debug logging for first few chunks (like OpenAI)
                if chunk_count <= 5:
                    log.debug(f"Gemini chunk {chunk_count}: content_len={len(chunk_text)}, tool_calls={len(tool_calls)}")
                
                # Only yield if there's actual content or tool calls (like OpenAI)
                if chunk_text or tool_calls:
                    yield result
                elif chunk_count <= 3:
                    # For first few chunks, yield even if empty (like OpenAI)
                    yield result
                            
        except Exception as e:
            log.error(f"Error in Gemini async streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }
            
    async def _regular_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Non-streaming completion using async execution."""
        try:
            def _sync_completion():
                system_instruction, user_message = _convert_messages_for_chat(messages)
                gemini_tools = _convert_tools_to_gemini_format(tools)
                
                # Create chat configuration
                chat_config = {"model": self.model}
                if system_instruction:
                    chat_config["system_instruction"] = system_instruction
                
                if gemini_tools:
                    chat_config["config"] = gtypes.GenerateContentConfig(
                        tools=gemini_tools,
                        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True)
                    )
                
                chat = self.client.chats.create(**chat_config)
                response = chat.send_message(user_message)
                
                # Parse response
                response_text, tool_calls = self._parse_gemini_response(response)
                
                return {
                    "response": response_text,
                    "tool_calls": tool_calls
                }
            
            return await asyncio.to_thread(_sync_completion)
            
        except Exception as e:
            log.error(f"Error in Gemini completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

# ───────────────────────────────────────── parse helpers ──────────

    def _parse_gemini_chunk(self, chunk) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse Gemini streaming chunk for text and function calls"""
        chunk_text = ""
        tool_calls = []
        
        try:
            # Method 1: Direct text attribute (most common)
            if hasattr(chunk, 'text') and chunk.text:
                chunk_text = chunk.text
            
            # Method 2: Check for function calls in chunk
            if hasattr(chunk, 'function_calls') and chunk.function_calls:
                for fc in chunk.function_calls:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": getattr(fc, "name", "unknown"),
                            "arguments": json.dumps(dict(getattr(fc, "args", {})))
                        }
                    })
            
            # Method 3: Check candidates structure
            elif hasattr(chunk, 'candidates') and chunk.candidates:
                try:
                    cand = chunk.candidates[0]
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                chunk_text += part.text
                            elif hasattr(part, 'function_call'):
                                fc = part.function_call
                                tool_calls.append({
                                    "id": f"call_{uuid.uuid4().hex[:8]}", 
                                    "type": "function", 
                                    "function": {
                                        "name": getattr(fc, "name", "unknown"),
                                        "arguments": json.dumps(dict(getattr(fc, "args", {})))
                                    }
                                })
                except (AttributeError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing candidates: {e}")
            
        except Exception as e:
            log.error(f"Error parsing Gemini chunk: {e}")
        
        return chunk_text, tool_calls

    def _parse_gemini_response(self, response) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse Gemini non-streaming response for text and function calls"""
        response_text = ""
        tool_calls = []
        
        try:
            # Extract text content
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            
            # Check for function calls
            if hasattr(response, 'function_calls') and response.function_calls:
                for fc in response.function_calls:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": getattr(fc, "name", "unknown"),
                            "arguments": json.dumps(dict(getattr(fc, "args", {})))
                        }
                    })
            
            # Alternative: check candidates structure for function calls
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    cand = response.candidates[0]
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                            elif hasattr(part, 'function_call'):
                                fc = part.function_call
                                tool_calls.append({
                                    "id": f"call_{uuid.uuid4().hex[:8]}", 
                                    "type": "function", 
                                    "function": {
                                        "name": getattr(fc, "name", "unknown"),
                                        "arguments": json.dumps(dict(getattr(fc, "args", {})))
                                    }
                                })
                except (AttributeError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing response candidates: {e}")
            
        except Exception as e:
            log.error(f"Error parsing Gemini response: {e}")
        
        return response_text, tool_calls