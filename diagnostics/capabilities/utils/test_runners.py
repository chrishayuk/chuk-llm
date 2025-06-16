# diagnostics/capabilities/utils/test_runners.py
"""
Test runners for specific LLM capabilities.
FIXED: Streaming test issues and error handling.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Callable
from .provider_configs import get_provider_config

class CapabilityTester:
    """Runs capability tests for LLM providers"""
    
    def __init__(self):
        self.client_cache: Dict[tuple[str, str], Any] = {}
    
    async def get_client(self, provider: str, model: str):
        """Get or create a cached client"""
        key = (provider, model)
        if key not in self.client_cache:
            # Use the new chuk-llm client factory
            from chuk_llm.llm.client import get_client
            try:
                self.client_cache[key] = get_client(provider=provider, model=model)
            except Exception as e:
                print(f"Failed to create client for {provider}:{model} - {e}")
                raise
        return self.client_cache[key]
    
    async def timed_execution(self, result_obj, key: str, coroutine_or_generator):
        """Execute with timing measurement - handles both coroutines and async generators"""
        start = time.perf_counter()
        try:
            # Check if it's an async generator (streaming)
            if hasattr(coroutine_or_generator, '__aiter__'):
                # It's an async generator - iterate through it
                chunks = []
                async for chunk in coroutine_or_generator:
                    chunks.append(chunk)
                return chunks
            else:
                # It's a regular coroutine - await it
                return await coroutine_or_generator
        finally:
            result_obj.timings[key] = time.perf_counter() - start
    
    async def test_text_completion(self, provider: str, model: str, result_obj, tick_fn: Callable):
        """Test basic text completion"""
        try:
            client = await self.get_client(provider, model)
            messages = [{"role": "user", "content": "Why is testing LLM providers important? (3–4 sentences)"}]
            
            response = await self.timed_execution(
                result_obj, "text", 
                client.create_completion(messages)
            )
            
            # Handle the standardized response format from chuk-llm
            response_text = response.get("response", "")
            success = bool(response_text.strip())
            
            result_obj.record("text_completion", success)
            tick_fn("text", success)
            
        except Exception as exc:
            result_obj.record("text_completion", False)
            result_obj.errors["text"] = str(exc)
            tick_fn("text", False)
    
    async def test_streaming(self, provider: str, model: str, result_obj, tick_fn: Callable):
        """Test streaming capability"""
        try:
            client = await self.get_client(provider, model)
            messages = [{"role": "user", "content": "Count from 1 to 3, one number per line."}]
            
            start_time = time.perf_counter()
            
            # Get the stream and iterate through it properly
            stream = await client.create_completion(messages, stream=True)
            
            chunk_count = 0
            found_content = False
            content_chunks = []
            
            # Properly iterate through the async generator
            async for chunk in stream:
                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    found_content = True
                    content_chunks.append(chunk["response"])
                    # Don't break immediately - let at least a few chunks come through
                    if chunk_count >= 5:  # Get a few chunks to verify streaming
                        break
            
            result_obj.timings["stream"] = time.perf_counter() - start_time
            
            # Success if we got content and multiple chunks
            success = found_content and chunk_count > 1
            result_obj.record("streaming_text", success)
            tick_fn("stream", success)
            
        except Exception as exc:
            result_obj.record("streaming_text", False)
            result_obj.errors["stream"] = str(exc)
            tick_fn("stream", False)
    
    async def test_tools(self, provider: str, model: str, result_obj, tick_fn: Callable):
        """Test function calling capability"""
        weather_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
        
        try:
            client = await self.get_client(provider, model)
            messages = [{"role": "user", "content": "What is the weather in London? Use get_weather."}]
            
            response = await self.timed_execution(
                result_obj, "tools",
                client.create_completion(messages, tools=[weather_tool])
            )
            
            # Check for tool calls in the standardized format
            tool_calls = response.get("tool_calls", [])
            success = len(tool_calls) > 0 and any(
                tc.get("function", {}).get("name") == "get_weather" 
                for tc in tool_calls
            )
            
            result_obj.record("function_call", success)
            tick_fn("tools", success)
            
        except Exception as exc:
            error_msg = str(exc).lower()
            # Check for known "unsupported" error patterns
            if any(phrase in error_msg for phrase in [
                "does not support tools", 
                "tool", 
                "function calling not available",
                "tools are not supported"
            ]):
                result_obj.record("function_call", None)
                tick_fn("tools", None)
            else:
                result_obj.record("function_call", False)
                result_obj.errors["tools"] = str(exc)
                tick_fn("tools", False)
    
    async def test_streaming_tools(self, provider: str, model: str, result_obj, tick_fn: Callable):
        """Test streaming with function calling"""
        weather_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
        
        try:
            client = await self.get_client(provider, model)
            messages = [{"role": "user", "content": "What is the weather in London? Use get_weather."}]
            
            start_time = time.perf_counter()
            
            # Get the stream and iterate through it properly
            stream = await client.create_completion(messages, tools=[weather_tool], stream=True)

            found_tool_call = False
            chunk_count = 0
            
            # Properly iterate through the async generator
            async for chunk in stream:
                chunk_count += 1
                # Check for tool calls in the standardized chunk format
                if isinstance(chunk, dict) and chunk.get("tool_calls"):
                    found_tool_call = True
                    break
                # Also check if we've gone through enough chunks without finding tool calls
                if chunk_count > 10:
                    break

            result_obj.timings["stream_tools"] = time.perf_counter() - start_time
            result_obj.record("streaming_function_call", found_tool_call)
            tick_fn("stream_tools", found_tool_call)

        except Exception as exc:
            error_msg = str(exc).lower()
            # Check for known "unsupported" error patterns
            if any(phrase in error_msg for phrase in [
                "does not support tools", 
                "tool", 
                "function calling not available",
                "tools are not supported"
            ]):
                result_obj.record("streaming_function_call", None)
                tick_fn("stream_tools", None)
            else:
                result_obj.record("streaming_function_call", False)
                result_obj.errors["stream_tools"] = str(exc)
                tick_fn("stream_tools", False)
    
    async def test_vision(self, provider: str, model: str, result_obj, tick_fn: Callable):
        """Test vision/multimodal capability"""
        try:
            client = await self.get_client(provider, model)
            provider_config = get_provider_config(provider)
            
            # Check if provider supports vision using the new capability system
            try:
                from chuk_llm.configuration.capabilities import PROVIDER_CAPABILITIES, Feature
                if provider in PROVIDER_CAPABILITIES:
                    caps = PROVIDER_CAPABILITIES[provider].get_model_capabilities(model)
                    if Feature.VISION not in caps.features:
                        result_obj.record("vision", None)
                        tick_fn("vision", None)
                        return
            except ImportError:
                # Fallback to provider config
                if not provider_config.supports_feature("vision"):
                    result_obj.record("vision", None)
                    tick_fn("vision", None)
                    return
            
            vision_msg = provider_config.create_vision_message("Describe what you see in this image.")
            
            response = await self.timed_execution(
                result_obj, "vision",
                client.create_completion([vision_msg])
            )
            
            # Check the standardized response format
            response_text = response.get("response", "")
            success = bool(response_text.strip())
            
            result_obj.record("vision", success)
            tick_fn("vision", success)
            
        except Exception as exc:
            # Use provider-specific error categorization
            provider_config = get_provider_config(provider)
            error_categories = provider_config.get_error_categories()
            error_msg = str(exc).lower()
            
            # Check if this is a known vision format issue
            is_format_error = False
            for category, patterns in error_categories.items():
                if "vision" in category or "format" in category:
                    if any(pattern.lower() in error_msg for pattern in patterns):
                        is_format_error = True
                        break
            
            # Also check for common unsupported patterns
            unsupported_patterns = [
                "vision not supported",
                "multimodal not supported", 
                "image not supported",
                "does not support vision",
                "content must be a string",
                "does not have the 'vision' capability",
                "validation error for message"
            ]
            
            if is_format_error or any(pattern in error_msg for pattern in unsupported_patterns):
                # Vision capability doesn't exist or format is wrong - mark as unsupported
                result_obj.record("vision", None)
                tick_fn("vision", None)
            else:
                result_obj.record("vision", False)
                result_obj.errors["vision"] = str(exc)
                tick_fn("vision", False)