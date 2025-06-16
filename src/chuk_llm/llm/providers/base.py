# chuk_llm/llm/providers/base.py
"""
Base provider implementation with enhanced client functionality.
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Union
import time
import asyncio
from abc import ABC, abstractmethod

from chuk_llm.configuration.unified_config import get_config
from chuk_llm.llm.core.base import BaseLLMClient
from chuk_llm.llm.middleware import CachingMiddleware, LoggingMiddleware, MetricsMiddleware, MiddlewareStack, Middleware
from chuk_llm.llm.core.errors import with_retry, ProviderErrorMapper
from chuk_llm.llm.core.types import ResponseValidator

class EnhancedBaseLLMClient(BaseLLMClient):
    """Enhanced base client with middleware, error handling, and validation"""
    
    def __init__(self, middleware: Optional[List[Middleware]] = None):
        self.middleware_stack = MiddlewareStack(middleware or [])
        self.provider_name = getattr(self, 'provider_name', 'unknown')
        self.model_name = getattr(self, 'model', 'unknown')
    
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """Enhanced completion with middleware and error handling"""
        
        start_time = time.time()
        
        try:
            # Process request through middleware
            processed_messages, processed_tools, processed_kwargs = await self.middleware_stack.process_request(
                messages, tools, **kwargs
            )
            
            # Check for cached response
            if "_cached_response" in processed_kwargs:
                cached = processed_kwargs.pop("_cached_response")
                duration = time.time() - start_time
                return await self.middleware_stack.process_response(cached, duration, stream)
            
            # Call the actual provider implementation
            if stream:
                return self._create_enhanced_stream(
                    processed_messages, processed_tools, start_time, **processed_kwargs
                )
            else:
                return await self._create_enhanced_completion(
                    processed_messages, processed_tools, start_time, **processed_kwargs
                )
                
        except Exception as e:
            duration = time.time() - start_time
            # Map provider-specific errors
            mapped_error = self._map_error(e)
            processed_error = await self.middleware_stack.process_error(mapped_error, duration)
            raise processed_error
    
    async def _create_enhanced_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]],
        start_time: float,
        **kwargs
    ):
        """Enhanced non-streaming completion"""
        
        @with_retry(max_retries=3)
        async def _do_completion():
            response = await self._create_completion_impl(messages, tools, **kwargs)
            
            # Validate response
            validated_response = ResponseValidator.validate_response(response, is_streaming=False)
            
            duration = time.time() - start_time
            return await self.middleware_stack.process_response(
                validated_response.dict(), duration, is_streaming=False
            )
        
        return await _do_completion()
    
    async def _create_enhanced_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        start_time: float,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Enhanced streaming with middleware processing"""
        
        chunk_index = 0
        
        try:
            # Get the raw stream from implementation
            stream = await self._create_stream_impl(messages, tools, **kwargs)
            
            async for chunk in stream:
                current_time = time.time()
                duration = current_time - start_time
                
                # Validate chunk
                validated_chunk = ResponseValidator.validate_response(chunk, is_streaming=True)
                
                # Process through middleware
                processed_chunk = await self.middleware_stack.process_stream_chunk(
                    validated_chunk.dict(), chunk_index, duration
                )
                
                yield processed_chunk
                chunk_index += 1
                
        except Exception as e:
            duration = time.time() - start_time
            mapped_error = self._map_error(e)
            processed_error = await self.middleware_stack.process_error(mapped_error, duration)
            
            # Yield error as final chunk
            yield {
                "response": f"Streaming error: {str(processed_error)}",
                "tool_calls": [],
                "error": True,
                "error_type": type(processed_error).__name__
            }
    
    def _map_error(self, error: Exception):
        """Map provider-specific errors to unified LLM errors"""
        return ProviderErrorMapper.map_openai_error(error, self.provider_name, self.model_name)
    
    # Abstract methods for subclasses to implement
    @abstractmethod
    async def _create_completion_impl(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Implement the actual completion logic"""
        pass
    
    @abstractmethod
    async def _create_stream_impl(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Implement the actual streaming logic"""
        pass
    
    async def close(self):
        """Cleanup resources"""
        if hasattr(self, 'async_client') and hasattr(self.async_client, 'close'):
            await self.async_client.close()

# Enhanced factory with middleware support
def get_enhanced_client(
    provider: str = "openai",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    middleware: Optional[List[Middleware]] = None,
    enable_logging: bool = True,
    enable_metrics: bool = True,
    enable_caching: bool = False,
) -> BaseLLMClient:
    """Enhanced factory with automatic middleware setup"""
    
    # Build default middleware stack
    default_middleware = []
    
    if enable_logging:
        default_middleware.append(LoggingMiddleware())
    
    if enable_metrics:
        default_middleware.append(MetricsMiddleware())
    
    if enable_caching:
        default_middleware.append(CachingMiddleware())
    
    # Add custom middleware
    if middleware:
        default_middleware.extend(middleware)
    
    # Get base client
    from chuk_llm.llm.client import get_client
    base_client = get_client(provider, model=model, api_key=api_key, api_base=api_base)
    
    # Wrap with enhanced functionality
    if hasattr(base_client, '_enable_middleware'):
        base_client._enable_middleware(default_middleware)
    
    return base_client