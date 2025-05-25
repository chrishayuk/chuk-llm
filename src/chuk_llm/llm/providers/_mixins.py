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


class OpenAIStyleMixin:
    """
    Helper mix-in for providers that emit OpenAI-style messages
    (OpenAI, Groq, Anthropic, Azure OpenAI, etc.).
    Includes:

      • _sanitize_tool_names
      • _call_blocking          - run blocking SDK in thread
      • _normalise_message      - convert full message → MCP dict
      • _stream_from_blocking   - wrap *stream=True* SDK generators
      • _stream_from_async      - FIXED: for native async streaming
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
        FIXED: More robust content extraction for all models.
        """
        # Extract content with multiple fallback methods
        content = None
        
        # Method 1: Direct content attribute
        if hasattr(msg, "content"):
            content = msg.content
        
        # Method 2: Dict-style access
        elif isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
        
        # Method 3: Message wrapper
        elif hasattr(msg, "message") and hasattr(msg.message, "content"):
            content = msg.message.content
        
        # Handle tool calls
        raw = getattr(msg, "tool_calls", None)
        calls: List[Dict[str, Any]] = []

        if raw:
            for c in raw:
                cid = getattr(c, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                try:
                    args   = c.function.arguments
                    args_j = (
                        json.dumps(json.loads(args))
                        if isinstance(args, str)
                        else json.dumps(args)
                    )
                except (TypeError, json.JSONDecodeError):
                    args_j = "{}"

                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": args_j,
                        },
                    }
                )

        return {"response": content if not calls else None, "tool_calls": calls}

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
                delta = chunk.choices[0].delta
                yield {
                    "response": delta.content or "",
                    "tool_calls": getattr(delta, "tool_calls", []),
                }

        # run the blocking generator in a thread
        def _worker():
            try:
                for ch in sdk_call(stream=True, **kwargs):
                    queue.put_nowait(ch)
            finally:
                queue.put_nowait(None)

        asyncio.get_running_loop().run_in_executor(None, _worker)
        return _aiter()

    # ------------------------------------------------------------------ FIXED: async streaming
    @staticmethod
    async def _stream_from_async(
        async_stream,
        normalize_chunk: Optional[Callable] = None
    ) -> AsyncIterator[LLMResult]:
        """
        FIXED: Stream from an async iterator with robust chunk handling for all models.
        
        ✅ This provides true streaming without buffering and handles model differences.
        """
        try:
            chunk_count = 0
            async for chunk in async_stream:
                chunk_count += 1
                
                # Initialize result
                result = {
                    "response": "",
                    "tool_calls": [],
                }
                
                # Handle different chunk structures
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    
                    # Method 1: Delta format (most common for streaming)
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                        
                        # Extract content from delta - ROBUST extraction
                        content = ""
                        if hasattr(delta, 'content') and delta.content is not None:
                            content = delta.content
                        
                        result["response"] = content
                        
                        # Handle tool calls in delta
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            tool_calls = []
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
                            result["tool_calls"] = tool_calls
                    
                    # Method 2: Message format (backup for some models)
                    elif hasattr(choice, 'message'):
                        message = choice.message
                        if hasattr(message, 'content') and message.content:
                            result["response"] = message.content
                        
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            # Use existing normalization
                            normalized = OpenAIStyleMixin._normalise_message(message)
                            result["tool_calls"] = normalized.get('tool_calls', [])
                
                # Method 3: Direct chunk content (fallback)
                elif hasattr(chunk, 'content'):
                    result["response"] = chunk.content or ""
                
                # Method 4: Dict-style chunk (fallback)
                elif isinstance(chunk, dict):
                    result["response"] = chunk.get('content', '')
                    result["tool_calls"] = chunk.get('tool_calls', [])
                
                # Apply custom normalization if provided
                if normalize_chunk:
                    result = normalize_chunk(result, chunk)
                
                # Debug logging for troubleshooting
                if chunk_count <= 5:
                    logging.debug(f"Stream chunk {chunk_count}: response='{result['response'][:50]}...', tool_calls={len(result['tool_calls'])}")
                
                # Always yield the result (even if empty for timing purposes)
                yield result
                        
        except Exception as e:
            logging.error(f"Error in _stream_from_async: {e}")
            # Yield error as final chunk
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }