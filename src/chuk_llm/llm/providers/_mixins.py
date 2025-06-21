# chuk_llm/llm/providers/_mixins.py
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

Tool      = Dict[str, Any]
LLMResult = Dict[str, Any]          # {"response": str|None, "tool_calls":[...]}

logger = logging.getLogger(__name__)


class OpenAIStyleMixin:
    """
    Helper mix-in for providers that emit OpenAI-style messages
    (OpenAI, Groq, Anthropic, Azure OpenAI, etc.).
    Enhanced with better content extraction and error handling.
    """

    # ------------------------------------------------------------------ sanitise
    _NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

    @classmethod
    def _sanitize_tool_names(cls, tools: Optional[List[Tool]]) -> Optional[List[Tool]]:
        if not tools:
            return tools
        fixed: List[Tool] = []
        for t in tools:
            copy = dict(t)
            fn = copy.get("function", {})
            name = fn.get("name")
            if name and cls._NAME_RE.search(name):
                clean = cls._NAME_RE.sub("_", name)
                logging.debug("Sanitising tool name '%s' → '%s'", name, clean)
                fn["name"] = clean
                copy["function"] = fn
            fixed.append(copy)
        return fixed

    # ------------------------------------------------------------------ blocking
    @staticmethod
    async def _call_blocking(fn: Callable, *args, **kwargs):
        """Run a blocking SDK call in a background thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ------------------------------------------------------------------ normalise
    @staticmethod
    def _normalise_message(msg) -> LLMResult:
        """
        Convert `response.choices[0].message` (full) → MCP dict.
        ENHANCED: Robust content extraction with multiple fallback methods.
        """
        # Initialize result
        content = None
        tool_calls = []
        
        # Method 1: Direct content attribute access
        try:
            if hasattr(msg, "content"):
                content = msg.content
                logger.debug(f"Content extracted via direct attribute: {type(content)} - '{str(content)[:50]}...'")
        except Exception as e:
            logger.debug(f"Direct content access failed: {e}")
        
        # Method 2: Dict-style access
        if content is None:
            try:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    logger.debug(f"Content extracted via dict access: {type(content)} - '{str(content)[:50]}...'")
            except Exception as e:
                logger.debug(f"Dict content access failed: {e}")
        
        # Method 3: Message wrapper access
        if content is None:
            try:
                if hasattr(msg, "message") and hasattr(msg.message, "content"):
                    content = msg.message.content
                    logger.debug(f"Content extracted via message wrapper: {type(content)} - '{str(content)[:50]}...'")
            except Exception as e:
                logger.debug(f"Message wrapper access failed: {e}")
        
        # Method 4: Check for alternative content fields
        if content is None:
            try:
                for attr in ["text", "message_content", "response_text"]:
                    if hasattr(msg, attr):
                        alt_content = getattr(msg, attr)
                        if alt_content:
                            content = alt_content
                            logger.debug(f"Content extracted via alternative field '{attr}': {type(content)}")
                            break
            except Exception as e:
                logger.debug(f"Alternative field access failed: {e}")
        
        # Handle None or empty content appropriately
        if content is None:
            content = ""
            logger.warning(f"No content found in message. Message type: {type(msg)}, attributes: {dir(msg) if hasattr(msg, '__dict__') else 'no __dict__'}")
        elif content == "":
            logger.debug("Empty string content detected - this may be normal for tool-only responses")
        
        # Extract tool calls with enhanced error handling
        try:
            raw_tool_calls = None
            
            # Try multiple ways to get tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                raw_tool_calls = msg.tool_calls
            elif hasattr(msg, "message") and hasattr(msg.message, "tool_calls") and msg.message.tool_calls:
                raw_tool_calls = msg.message.tool_calls
            elif isinstance(msg, dict) and msg.get("tool_calls"):
                raw_tool_calls = msg["tool_calls"]
            
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    try:
                        # Extract tool call ID
                        tc_id = getattr(tc, "id", None)
                        if not tc_id:
                            tc_id = f"call_{uuid.uuid4().hex[:8]}"
                        
                        # Extract function details
                        if hasattr(tc, "function"):
                            func = tc.function
                            func_name = getattr(func, "name", "unknown_function")
                            
                            # Handle arguments - multiple formats possible
                            args = getattr(func, "arguments", "{}")
                            if isinstance(args, str):
                                try:
                                    # Validate JSON and reformat
                                    parsed_args = json.loads(args)
                                    args_j = json.dumps(parsed_args)
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid JSON in tool call arguments: {args}")
                                    args_j = "{}"
                            elif isinstance(args, dict):
                                args_j = json.dumps(args)
                            else:
                                logger.warning(f"Unexpected argument type: {type(args)}")
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
                            logger.warning(f"Tool call missing function attribute: {tc}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process tool call {tc}: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
        
        # Determine response value based on content and tool calls
        if tool_calls:
            # If we have tool calls, response should be None (unless there's also content)
            response_value = content if content and content.strip() else None
        else:
            # No tool calls, use content (even if empty)
            response_value = content
        
        result = {
            "response": response_value,
            "tool_calls": tool_calls
        }
        
        # Debug logging
        logger.debug(f"Normalized message result: response='{str(response_value)[:50] if response_value else None}...', tool_calls={len(tool_calls)}")
        
        return result

    # ------------------------------------------------------------------ streaming
    @classmethod
    def _stream_from_blocking(
        cls,
        sdk_call: Callable[..., Any],
        /,
        **kwargs,
    ) -> AsyncIterator[LLMResult]:
        """
        Wrap a *blocking* SDK streaming generator (``stream=True``) and yield
        MCP-style *delta dictionaries* asynchronously.

        ⚠️  WARNING: This method has buffering issues and should be avoided.
        Use _stream_from_async for better real-time streaming.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _aiter() -> AsyncIterator[LLMResult]:
            while True:
                chunk = await queue.get()
                if chunk is None:               # sentinel from worker
                    break
                
                try:
                    delta = chunk.choices[0].delta
                    yield {
                        "response": delta.content or "",
                        "tool_calls": getattr(delta, "tool_calls", []),
                    }
                except Exception as e:
                    logger.error(f"Error processing blocking stream chunk: {e}")
                    yield {
                        "response": "",
                        "tool_calls": [],
                        "error": True
                    }

        # run the blocking generator in a thread
        def _worker():
            try:
                for ch in sdk_call(stream=True, **kwargs):
                    queue.put_nowait(ch)
            except Exception as e:
                logger.error(f"Error in blocking stream worker: {e}")
                queue.put_nowait({"error": str(e)})
            finally:
                queue.put_nowait(None)

        asyncio.get_running_loop().run_in_executor(None, _worker)
        return _aiter()

    # ------------------------------------------------------------------ ENHANCED async streaming
    @staticmethod
    async def _stream_from_async(
        async_stream,
        normalize_chunk: Optional[Callable] = None
    ) -> AsyncIterator[LLMResult]:
        """
        ENHANCED: Stream from an async iterator with robust chunk handling for all models.
        
        ✅ This provides true streaming without buffering and handles model differences.
        ✅ Enhanced error handling and content extraction for problematic providers.
        """
        try:
            chunk_count = 0
            total_content_chars = 0
            
            async for chunk in async_stream:
                chunk_count += 1
                
                # Initialize result with defaults
                result = {
                    "response": "",
                    "tool_calls": [],
                }
                
                try:
                    # Method 1: Standard choices[0].delta format (most common)
                    if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        
                        # Handle delta format (most common for streaming)
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            
                            # Extract content with enhanced handling
                            content = ""
                            if hasattr(delta, 'content') and delta.content is not None:
                                content = str(delta.content)  # Ensure string
                                total_content_chars += len(content)
                            
                            result["response"] = content
                            
                            # Handle tool calls in delta
                            if hasattr(delta, "tool_calls") and delta.tool_calls:
                                delta_tool_calls = []
                                for tc in delta.tool_calls:
                                    try:
                                        if hasattr(tc, 'function') and tc.function:
                                            tool_call = {
                                                "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                                "type": "function", 
                                                "function": {
                                                    "name": getattr(tc.function, 'name', ''),
                                                    "arguments": getattr(tc.function, 'arguments', '') or ""
                                                }
                                            }
                                            delta_tool_calls.append(tool_call)
                                    except Exception as e:
                                        logger.debug(f"Error processing delta tool call: {e}")
                                        continue
                                
                                result["tool_calls"] = delta_tool_calls
                        
                        # Method 2: Full message format (less common for streaming but possible)
                        elif hasattr(choice, 'message') and choice.message:
                            message = choice.message
                            if hasattr(message, 'content') and message.content:
                                content = str(message.content)
                                result["response"] = content
                                total_content_chars += len(content)
                            
                            # Handle tool calls in full message
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                normalized = OpenAIStyleMixin._normalise_message(message)
                                result["tool_calls"] = normalized.get('tool_calls', [])
                        
                        # Method 3: Choice with direct content (some providers)
                        elif hasattr(choice, 'text'):
                            content = str(choice.text)
                            result["response"] = content
                            total_content_chars += len(content)
                    
                    # Method 4: Direct chunk content (fallback for non-standard formats)
                    elif hasattr(chunk, 'content') and chunk.content:
                        content = str(chunk.content)
                        result["response"] = content
                        total_content_chars += len(content)
                    
                    # Method 5: Dict-style chunk (fallback)
                    elif isinstance(chunk, dict):
                        if 'content' in chunk:
                            result["response"] = str(chunk['content'])
                            total_content_chars += len(result["response"])
                        if 'tool_calls' in chunk:
                            result["tool_calls"] = chunk['tool_calls']
                    
                    # Apply custom normalization if provided
                    if normalize_chunk:
                        try:
                            result = normalize_chunk(result, chunk)
                        except Exception as e:
                            logger.debug(f"Custom normalization failed: {e}")
                    
                    # Debug logging for first few chunks and periodic updates
                    if chunk_count <= 3 or chunk_count % 50 == 0:
                        logger.debug(f"Stream chunk {chunk_count}: response_len={len(result['response'])}, tool_calls={len(result['tool_calls'])}, total_chars={total_content_chars}")
                    
                    # Always yield the result (even if empty for timing)
                    yield result
                
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {chunk_count}: {chunk_error}")
                    # Yield error chunk but continue streaming
                    yield {
                        "response": "",
                        "tool_calls": [],
                        "error": True,
                        "error_message": f"Chunk processing error: {str(chunk_error)}"
                    }
                    continue
            
            # Final statistics
            logger.debug(f"Streaming completed: {chunk_count} chunks processed, {total_content_chars} total characters")
            
            # Warn if no content was received
            if chunk_count > 0 and total_content_chars == 0:
                logger.warning(f"Streaming completed with {chunk_count} chunks but no content received - possible API or model issue")
                        
        except Exception as stream_error:
            logger.error(f"Fatal error in _stream_from_async: {stream_error}")
            # Yield final error
            yield {
                "response": f"Streaming error: {str(stream_error)}",
                "tool_calls": [],
                "error": True,
                "error_message": str(stream_error)
            }

    # ------------------------------------------------------------------ Enhanced debugging
    @staticmethod
    def debug_message_structure(msg, context: str = "unknown"):
        """Debug helper to understand message structure from different providers"""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        
        logger.debug(f"=== DEBUG MESSAGE STRUCTURE ({context}) ===")
        logger.debug(f"Type: {type(msg)}")
        
        if hasattr(msg, '__dict__'):
            logger.debug(f"Attributes: {list(msg.__dict__.keys())}")
        else:
            logger.debug(f"Dir: {[attr for attr in dir(msg) if not attr.startswith('_')]}")
        
        # Try to access common attributes
        for attr in ['content', 'tool_calls', 'message', 'choices', 'delta']:
            try:
                value = getattr(msg, attr, None)
                if value is not None:
                    logger.debug(f"{attr}: {type(value)} = {str(value)[:100]}...")
            except Exception as e:
                logger.debug(f"{attr}: Error accessing - {e}")
        
        logger.debug("=== END DEBUG ===")