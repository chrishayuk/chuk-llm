# chuk_llm/llm/middleware.py
from typing import Protocol, List, Any, Dict, Optional, AsyncIterator
import time
import logging
from abc import ABC, abstractmethod
from .core.types import LLMResponse, StreamChunk

class Middleware(ABC):
    """Base middleware class"""
    
    @abstractmethod
    async def process_request(
        self, 
        messages: List[Dict], 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> tuple[List[Dict], Optional[List[Dict]], Dict]:
        """Process before sending to provider"""
        return messages, tools, kwargs
    
    @abstractmethod
    async def process_response(
        self, 
        response: Any, 
        duration: float,
        is_streaming: bool = False
    ) -> Any:
        """Process after receiving from provider"""
        return response
    
    async def process_stream_chunk(
        self,
        chunk: Dict[str, Any],
        chunk_index: int,
        total_duration: float
    ) -> Dict[str, Any]:
        """Process individual streaming chunks"""
        return chunk
    
    async def process_error(
        self,
        error: Exception,
        duration: float
    ) -> Exception:
        """Process errors"""
        return error

class LoggingMiddleware(Middleware):
    """Middleware for request/response logging"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, log_level: int = logging.INFO):
        self.logger = logger or logging.getLogger(__name__)
        self.log_level = log_level
    
    async def process_request(self, messages, tools=None, **kwargs):
        self.logger.log(
            self.log_level,
            f"LLM Request: {len(messages)} messages, "
            f"{len(tools) if tools else 0} tools, "
            f"stream={kwargs.get('stream', False)}"
        )
        return messages, tools, kwargs
    
    async def process_response(self, response, duration, is_streaming=False):
        if is_streaming:
            self.logger.log(self.log_level, f"LLM Stream completed in {duration:.2f}s")
        else:
            response_len = len(response.get('response', '')) if isinstance(response, dict) else 0
            tool_count = len(response.get('tool_calls', [])) if isinstance(response, dict) else 0
            self.logger.log(
                self.log_level,
                f"LLM Response in {duration:.2f}s: {response_len} chars, {tool_count} tools"
            )
        return response
    
    async def process_error(self, error, duration):
        self.logger.error(f"LLM Error after {duration:.2f}s: {error}")
        return error

class MetricsMiddleware(Middleware):
    """Middleware for collecting metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "total_duration": 0.0,
            "streaming_requests": 0,
            "tool_requests": 0,
        }
    
    async def process_request(self, messages, tools=None, **kwargs):
        self.metrics["total_requests"] += 1
        if kwargs.get("stream", False):
            self.metrics["streaming_requests"] += 1
        if tools:
            self.metrics["tool_requests"] += 1
        return messages, tools, kwargs
    
    async def process_response(self, response, duration, is_streaming=False):
        self.metrics["total_duration"] += duration
        return response
    
    async def process_error(self, error, duration):
        self.metrics["total_errors"] += 1
        self.metrics["total_duration"] += duration
        return error
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics = self.metrics.copy()
        if metrics["total_requests"] > 0:
            metrics["average_duration"] = metrics["total_duration"] / metrics["total_requests"]
            metrics["error_rate"] = metrics["total_errors"] / metrics["total_requests"]
        return metrics

class CachingMiddleware(Middleware):
    """Simple in-memory caching middleware"""
    
    def __init__(self, ttl: int = 300, max_size: int = 100):
        self.cache = {}
        self.ttl = ttl
        self.max_size = max_size
    
    def _get_cache_key(self, messages: List[Dict], tools: Optional[List[Dict]]) -> str:
        import hashlib
        import json
        content = json.dumps({"messages": messages, "tools": tools or []}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def process_request(self, messages, tools=None, **kwargs):
        # Only cache non-streaming requests
        if not kwargs.get("stream", False):
            cache_key = self._get_cache_key(messages, tools)
            cached = self.cache.get(cache_key)
            
            if cached and time.time() - cached["timestamp"] < self.ttl:
                # Return cached response as a special marker
                kwargs["_cached_response"] = cached["response"]
        
        return messages, tools, kwargs
    
    async def process_response(self, response, duration, is_streaming=False):
        # Cache successful non-streaming responses
        if not is_streaming and isinstance(response, dict) and not response.get("error"):
            # This would need integration with the actual request context
            pass
        return response

class MiddlewareStack:
    """Manages a stack of middleware"""
    
    def __init__(self, middlewares: List[Middleware]):
        self.middlewares = middlewares
    
    async def process_request(self, messages, tools=None, **kwargs):
        for middleware in self.middlewares:
            messages, tools, kwargs = await middleware.process_request(messages, tools, **kwargs)
        return messages, tools, kwargs
    
    async def process_response(self, response, duration, is_streaming=False):
        # Process in reverse order for response
        for middleware in reversed(self.middlewares):
            response = await middleware.process_response(response, duration, is_streaming)
        return response
    
    async def process_stream_chunk(self, chunk, chunk_index, total_duration):
        for middleware in reversed(self.middlewares):
            chunk = await middleware.process_stream_chunk(chunk, chunk_index, total_duration)
        return chunk
    
    async def process_error(self, error, duration):
        for middleware in reversed(self.middlewares):
            error = await middleware.process_error(error, duration)
        return error
    
class PaymentGuardMiddleware(Middleware):
    """
    Marks a response as `error=True` when the provider reports
    an out-of-credit / payment-required situation.

    That lets the demo harness (or any caller) treat it as a failure
    without crashing the whole run.
    """

    _BILLING_MESSAGES = {
        "insufficient balance",
        "payment required",
        "quota exceeded",
        "credit depleted",
    }

    async def process_response(
        self,
        response: Dict[str, Any],
        duration: float,
        is_streaming: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(response, dict) and not response.get("error"):
            msg = (response.get("response") or "").lower()
            if any(phrase in msg for phrase in self._BILLING_MESSAGES):
                response["error"] = True
                response["error_message"] = "Billing / quota error detected"
        return response
