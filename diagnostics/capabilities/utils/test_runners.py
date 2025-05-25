# diagnostics/capabilities/utils/test_runners.py
"""
Test runners for specific LLM capabilities.
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
            from chuk_llm.llm.llm_client import get_llm_client
            try:
                self.client_cache[key] = get_llm_client(provider=provider, model=model)
            except Exception as e:
                print(f"Failed to create client for {provider}:{model} - {e}")
                raise
        return self.client_cache[key]
    
    async def timed_execution(self, result_obj, key: str, awaitable):
        """Execute with timing measurement"""
        start = time.perf_counter()
        try:
            return await awaitable
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
            
            # Handle both async and sync responses
            if hasattr(response, '__aiter__'):
                response_text = ""
                async for chunk in response:
                    if isinstance(chunk, dict) and chunk.get("response"):
                        response_text += chunk["response"]
                success = bool(response_text.strip())
            else:
                success = bool(response.get("response", "").strip())
            
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
            messages = [{"role": "user", "content": "Why is testing LLM providers important? (3–4 sentences)"}]
            
            start_time = time.perf_counter()
            stream = client.create_completion(messages, stream=True)
            
            if not hasattr(stream, "__aiter__"):
                raise TypeError("stream returned non-async iterator")
            
            found_content = False
            async for chunk in stream:
                if isinstance(chunk, dict) and chunk.get("response"):
                    found_content = True
                    break
            
            result_obj.timings["stream"] = time.perf_counter() - start_time
            result_obj.record("streaming_text", found_content)
            tick_fn("stream", found_content)
            
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
            
            # Handle both async and sync responses
            if hasattr(response, '__aiter__'):
                tool_calls = []
                async for chunk in response:
                    if isinstance(chunk, dict) and chunk.get("tool_calls"):
                        tool_calls.extend(chunk["tool_calls"])
                success = bool(tool_calls)
            else:
                success = bool(response.get("tool_calls"))
            
            result_obj.record("function_call", success)
            tick_fn("tools", success)
            
        except Exception as exc:
            error_msg = str(exc).lower()
            if any(phrase in error_msg for phrase in ["does not support tools", "tool", "function"]):
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
        
        def chunk_has_tool_call(chunk: Any) -> bool:
            if not isinstance(chunk, dict):
                return False
            
            # Direct tool_calls field
            if chunk.get("tool_calls"):
                return True
            
            # OpenAI/Groq style nested structure
            try:
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and choice["delta"].get("tool_calls"):
                        return True
            except (KeyError, IndexError, TypeError):
                pass
            
            return False
        
        try:
            client = await self.get_client(provider, model)
            messages = [{"role": "user", "content": "What is the weather in London? Use get_weather."}]
            
            start_time = time.perf_counter()
            stream = client.create_completion(messages, tools=[weather_tool], stream=True)

            if not hasattr(stream, "__aiter__"):
                raise TypeError("stream_tools returned non-async iterator")

            found_tool_call = False
            async for chunk in stream:
                if chunk_has_tool_call(chunk):
                    found_tool_call = True
                    break

            result_obj.timings["stream_tools"] = time.perf_counter() - start_time
            result_obj.record("streaming_function_call", found_tool_call)
            tick_fn("stream_tools", found_tool_call)

        except Exception as exc:
            error_msg = str(exc).lower()
            if any(phrase in error_msg for phrase in ["does not support tools", "tool", "function"]):
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
            
            # Skip vision test if provider doesn't support it
            if not provider_config.supports_feature("vision"):
                result_obj.record("vision", None)
                tick_fn("vision", None)
                return
            
            vision_msg = provider_config.create_vision_message("Describe what you see in this image.")
            
            response = await self.timed_execution(
                result_obj, "vision",
                client.create_completion([vision_msg])
            )
            
            # Handle both async and sync responses
            if hasattr(response, '__aiter__'):
                response_text = ""
                async for chunk in response:
                    if isinstance(chunk, dict) and chunk.get("response"):
                        response_text += chunk["response"]
                success = bool(response_text.strip())
            else:
                success = bool(response.get("response", "").strip())
            
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
            
            if is_format_error:
                # Vision capability exists but format is wrong - mark as unsupported for this test
                result_obj.record("vision", None)
                tick_fn("vision", None)
            else:
                result_obj.record("vision", False)
                result_obj.errors["vision"] = str(exc)
                tick_fn("vision", False)