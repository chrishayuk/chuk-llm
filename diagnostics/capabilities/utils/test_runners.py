# diagnostics/capabilities/utils/test_runners.py - COMPLETE ENHANCED VERSION
"""
Test runners for specific LLM capabilities.
COMPLETE: Enhanced model selection, connection retry logic, and better error handling
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import time
from collections.abc import Callable
from typing import Any

from .provider_configs import get_provider_config


class CapabilityTester:
    """Runs capability tests for LLM providers with intelligent model selection and connection management"""

    def __init__(self):
        self.client_cache: dict[tuple[str, str], Any] = {}
        self.connection_retries = 2
        self.problematic_providers = {"mistral", "perplexity"}

    async def get_client(self, provider: str, model: str, force_new: bool = False):
        """Get or create a cached client with connection retry logic"""
        key = (provider, model)

        # Always use fresh clients for problematic providers
        if provider.lower() in self.problematic_providers:
            force_new = True

        if force_new or key not in self.client_cache:
            from chuk_llm.llm.client import get_client

            try:
                # Clear old client if force_new
                if force_new and key in self.client_cache:
                    old_client = self.client_cache[key]
                    # Try to close old client if it has a close method
                    if hasattr(old_client, "close"):
                        with contextlib.suppress(builtins.BaseException):
                            await old_client.close()
                    del self.client_cache[key]

                self.client_cache[key] = get_client(provider=provider, model=model)
            except Exception as e:
                print(f"Failed to create client for {provider}:{model} - {e}")
                raise
        return self.client_cache[key]

    async def safe_client_call(
        self, provider: str, model: str, operation_name: str, operation_func
    ):
        """Execute client operation with retry logic for I/O errors"""
        provider_lower = provider.lower()

        for attempt in range(self.connection_retries + 1):
            try:
                # Get fresh client on retry attempts or for problematic providers
                force_new = attempt > 0 or provider_lower in self.problematic_providers
                client = await self.get_client(provider, model, force_new=force_new)

                # Add timeout for problematic providers
                if provider_lower in self.problematic_providers:
                    try:
                        result = await asyncio.wait_for(
                            operation_func(client), timeout=30.0
                        )
                        return result
                    except TimeoutError:
                        if attempt < self.connection_retries:
                            print(
                                f"🔄 {provider}: Timeout on {operation_name} (attempt {attempt + 2})"
                            )
                            await asyncio.sleep(1.0)
                            continue
                        else:
                            raise Exception(
                                f"Timeout after {self.connection_retries + 1} attempts"
                            ) from None
                else:
                    # Normal execution for stable providers
                    return await operation_func(client)

            except Exception as e:
                error_msg = str(e).lower()

                # Check if it's an I/O connection error
                connection_errors = [
                    "i/o operation on closed file",
                    "connection",
                    "closed",
                    "timeout",
                    "broken pipe",
                    "reset by peer",
                ]

                is_connection_error = any(err in error_msg for err in connection_errors)

                if is_connection_error and attempt < self.connection_retries:
                    print(
                        f"🔄 {provider}: Retrying {operation_name} (attempt {attempt + 2}) due to connection issue"
                    )
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    # Re-raise the exception if it's not a connection issue or we've exhausted retries
                    raise e

    async def timed_execution(self, result_obj, key: str, coroutine_or_generator):
        """Execute with timing measurement"""
        start = time.perf_counter()
        try:
            if hasattr(coroutine_or_generator, "__aiter__"):
                chunks = []
                async for chunk in coroutine_or_generator:
                    chunks.append(chunk)
                return chunks
            else:
                return await coroutine_or_generator
        finally:
            result_obj.timings[key] = time.perf_counter() - start

    def _get_best_model_for_capability(
        self, provider: str, model: str, capability: str
    ) -> tuple[bool, str | None]:
        """Get the best model for a specific capability with enhanced logic"""
        try:
            from chuk_llm.configuration.unified_config import Feature, get_config

            config = get_config()
            provider_config = config.get_provider(provider)

            capability_map = {
                "text": Feature.TEXT,  # Added TEXT mapping
                "streaming": Feature.STREAMING,
                "tools": Feature.TOOLS,
                "vision": Feature.VISION,
                "multimodal": Feature.MULTIMODAL,
            }

            if capability not in capability_map:
                return False, None

            feature = capability_map[capability]

            # Special handling for vision - check for vision-specific models first
            if capability == "vision":
                return self._find_vision_model(provider_config, model)

            # For other capabilities, check if specified model supports it
            if provider_config.supports_feature(feature, model):
                return True, model

            # Find alternative model that supports the capability
            for available_model in provider_config.models:
                if provider_config.supports_feature(feature, available_model):
                    print(
                        f"🔄 {provider}: Using {available_model} for {capability} (default {model} doesn't support it)"
                    )
                    return True, available_model

            # Check aliases
            for alias, actual_model in provider_config.model_aliases.items():
                if provider_config.supports_feature(feature, actual_model):
                    print(
                        f"🔄 {provider}: Using {actual_model} (alias: {alias}) for {capability}"
                    )
                    return True, actual_model

            return False, None

        except Exception as e:
            print(f"Capability check failed for {provider}/{capability}: {e}")
            return False, None

    def _find_vision_model(
        self, provider_config, default_model: str
    ) -> tuple[bool, str | None]:
        """Find the best vision model for a provider"""
        from chuk_llm.configuration.unified_config import Feature

        # Vision-capable model patterns by provider
        vision_patterns = {
            "mistral": ["pixtral", "mistral-medium-2505", "mistral-small-2503"],
            "anthropic": ["claude-"],  # Most Claude models have vision
            "openai": ["gpt-4o", "gpt-4-turbo", "gpt-4"],  # GPT-4 family has vision
            "gemini": ["gemini-"],  # Most Gemini models have vision
            "watsonx": ["vision", "llama-3-2-.*vision", "granite-vision"],
            "ollama": ["llama3.2.*vision", "llava"],
        }

        provider_name = provider_config.name.lower()

        # Check if default model supports vision
        if provider_config.supports_feature(Feature.VISION, default_model):
            return True, default_model

        # Look for vision-specific models
        patterns = vision_patterns.get(provider_name, [])
        for model in provider_config.models:
            # Check if model matches vision patterns
            for pattern in patterns:
                if pattern.lower() in model.lower() and provider_config.supports_feature(Feature.VISION, model):
                    print(f"🔄 {provider_name}: Using vision model {model}")
                    return True, model

        # Check aliases for vision models
        for alias, actual_model in provider_config.model_aliases.items():
            if ("vision" in alias.lower() or "pixtral" in alias.lower()) and provider_config.supports_feature(Feature.VISION, actual_model):
                print(
                    f"🔄 {provider_name}: Using vision model {actual_model} (alias: {alias})"
                )
                return True, actual_model

        # Fallback: check all models for vision capability
        for model in provider_config.models:
            if provider_config.supports_feature(Feature.VISION, model):
                print(f"🔄 {provider_name}: Using {model} for vision capability")
                return True, model

        return False, None

    def _should_mark_as_unsupported(
        self, provider: str, capability: str, error_msg: str
    ) -> bool:
        """Determine if persistent errors should be marked as unsupported rather than errors"""
        provider_lower = provider.lower()
        error_lower = error_msg.lower()

        # For known problematic combinations, mark as unsupported after retries fail
        if (
            provider_lower == "mistral"
            and capability in ["tools", "stream_tools"]
            and "i/o operation" in error_lower
        ):
            return True

        return bool(provider_lower == "perplexity" and "i/o operation" in error_lower)

    async def test_text_completion(
        self, provider: str, model: str, result_obj, tick_fn: Callable
    ):
        """Test basic text completion with connection retry"""

        async def do_text_test(client):
            messages = [
                {
                    "role": "user",
                    "content": "Why is testing LLM providers important? (3-4 sentences)",
                }
            ]
            return await self.timed_execution(
                result_obj, "text", client.create_completion(messages)
            )

        try:
            response = await self.safe_client_call(
                provider, model, "text completion", do_text_test
            )

            if isinstance(response, dict):
                response_text = response.get("response", "")
                success = bool(response_text and response_text.strip())
            else:
                success = False

            result_obj.record("text_completion", success)
            tick_fn("text", success)

        except Exception as exc:
            error_msg = str(exc)

            # Check if we should mark as unsupported instead of error
            if self._should_mark_as_unsupported(provider, "text", error_msg):
                print(
                    f"💤 {provider}: Marking text as unsupported due to persistent issues"
                )
                result_obj.record("text_completion", None)
                tick_fn("text", None)
            else:
                result_obj.record("text_completion", False)
                result_obj.errors["text"] = error_msg
                tick_fn("text", False)

    async def test_streaming(
        self, provider: str, model: str, result_obj, tick_fn: Callable
    ):
        """Test streaming capability with smart model selection and connection retry"""
        supports, best_model = self._get_best_model_for_capability(
            provider, model, "streaming"
        )
        if not supports:
            result_obj.record("streaming_text", None)
            tick_fn("stream", None)
            return

        async def do_streaming_test(client):
            messages = [
                {"role": "user", "content": "Count from 1 to 3, one number per line."}
            ]

            start_time = time.perf_counter()
            stream = client.create_completion(messages, stream=True)

            chunk_count = 0
            found_content = False

            async for chunk in stream:
                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("response"):
                    found_content = True
                    if chunk_count >= 5:
                        break

            result_obj.timings["stream"] = time.perf_counter() - start_time
            return found_content and chunk_count > 1

        try:
            success = await self.safe_client_call(
                provider, best_model, "streaming", do_streaming_test
            )
            result_obj.record("streaming_text", success)
            tick_fn("stream", success)

        except Exception as exc:
            error_msg = str(exc)

            if self._should_mark_as_unsupported(provider, "stream", error_msg):
                print(
                    f"💤 {provider}: Marking streaming as unsupported due to persistent issues"
                )
                result_obj.record("streaming_text", None)
                tick_fn("stream", None)
            else:
                result_obj.record("streaming_text", False)
                result_obj.errors["stream"] = error_msg
                tick_fn("stream", False)

    async def test_tools(
        self, provider: str, model: str, result_obj, tick_fn: Callable
    ):
        """Test function calling with smart model selection and connection retry"""
        supports, best_model = self._get_best_model_for_capability(
            provider, model, "tools"
        )
        if not supports:
            result_obj.record("function_call", None)
            tick_fn("tools", None)
            return

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

        async def do_tools_test(client):
            messages = [
                {
                    "role": "user",
                    "content": "What is the weather in London? Use get_weather.",
                }
            ]
            return await self.timed_execution(
                result_obj,
                "tools",
                client.create_completion(messages, tools=[weather_tool]),
            )

        try:
            response = await self.safe_client_call(
                provider, best_model, "tools", do_tools_test
            )

            if isinstance(response, dict):
                tool_calls = response.get("tool_calls", [])
                success = len(tool_calls) > 0 and any(
                    tc.get("function", {}).get("name") == "get_weather"
                    for tc in tool_calls
                )
            else:
                success = False

            result_obj.record("function_call", success)
            tick_fn("tools", success)

        except Exception as exc:
            error_msg = str(exc).lower()

            # Check if it's a capability limitation rather than an error
            unsupported_patterns = [
                "does not support tools",
                "function calling not available",
                "tools are not supported",
                "does not support function calling",
            ]

            if any(phrase in error_msg for phrase in unsupported_patterns):
                result_obj.record("function_call", None)
                tick_fn("tools", None)
            elif self._should_mark_as_unsupported(provider, "tools", str(exc)):
                print(
                    f"💤 {provider}: Marking tools as unsupported due to persistent I/O issues"
                )
                result_obj.record("function_call", None)
                tick_fn("tools", None)
            else:
                result_obj.record("function_call", False)
                result_obj.errors["tools"] = str(exc)
                tick_fn("tools", False)

    async def test_streaming_tools(
        self, provider: str, model: str, result_obj, tick_fn: Callable
    ):
        """Test streaming with function calling and connection retry"""
        streaming_supports, streaming_model = self._get_best_model_for_capability(
            provider, model, "streaming"
        )
        tools_supports, tools_model = self._get_best_model_for_capability(
            provider, model, "tools"
        )

        if not (streaming_supports and tools_supports):
            result_obj.record("streaming_function_call", None)
            tick_fn("stream_tools", None)
            return

        # Use the tools model since it's more restrictive
        best_model = tools_model

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

        async def do_streaming_tools_test(client):
            messages = [
                {
                    "role": "user",
                    "content": "What is the weather in London? Use get_weather.",
                }
            ]

            start_time = time.perf_counter()
            stream = client.create_completion(
                messages, tools=[weather_tool], stream=True
            )

            found_tool_call = False
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1
                if isinstance(chunk, dict) and chunk.get("tool_calls"):
                    found_tool_call = True
                    break
                if chunk_count > 10:
                    break

            result_obj.timings["stream_tools"] = time.perf_counter() - start_time
            return found_tool_call

        try:
            success = await self.safe_client_call(
                provider, best_model, "streaming tools", do_streaming_tools_test
            )
            result_obj.record("streaming_function_call", success)
            tick_fn("stream_tools", success)

        except Exception as exc:
            error_msg = str(exc).lower()

            unsupported_patterns = [
                "does not support tools",
                "function calling not available",
                "tools are not supported",
            ]

            if any(phrase in error_msg for phrase in unsupported_patterns):
                result_obj.record("streaming_function_call", None)
                tick_fn("stream_tools", None)
            elif self._should_mark_as_unsupported(provider, "stream_tools", str(exc)):
                print(
                    f"💤 {provider}: Marking streaming tools as unsupported due to persistent I/O issues"
                )
                result_obj.record("streaming_function_call", None)
                tick_fn("stream_tools", None)
            else:
                result_obj.record("streaming_function_call", False)
                result_obj.errors["stream_tools"] = str(exc)
                tick_fn("stream_tools", False)

    async def test_vision(
        self, provider: str, model: str, result_obj, tick_fn: Callable
    ):
        """Test vision capability with enhanced model selection and connection retry"""
        supports, best_model = self._get_best_model_for_capability(
            provider, model, "vision"
        )
        if not supports:
            result_obj.record("vision", None)
            tick_fn("vision", None)
            return

        async def do_vision_test(client):
            provider_config = get_provider_config(provider)
            vision_msg = provider_config.create_vision_message(
                "Describe what you see in this image."
            )

            return await self.timed_execution(
                result_obj, "vision", client.create_completion([vision_msg])
            )

        try:
            response = await self.safe_client_call(
                provider, best_model, "vision", do_vision_test
            )

            if isinstance(response, dict):
                response_text = response.get("response", "")
                success = bool(response_text and response_text.strip())
            else:
                success = False

            result_obj.record("vision", success)
            tick_fn("vision", success)

        except Exception as exc:
            error_msg = str(exc).lower()

            unsupported_patterns = [
                "vision not supported",
                "multimodal not supported",
                "does not have the 'vision' capability",
                "model which does not have the 'vision' capability",
                "containing images has been given to a model which does not have",
                "images has been given to a model",
            ]

            if any(pattern in error_msg for pattern in unsupported_patterns):
                result_obj.record("vision", None)
                tick_fn("vision", None)
            elif self._should_mark_as_unsupported(provider, "vision", str(exc)):
                print(
                    f"💤 {provider}: Marking vision as unsupported due to persistent I/O issues"
                )
                result_obj.record("vision", None)
                tick_fn("vision", None)
            else:
                result_obj.record("vision", False)
                result_obj.errors["vision"] = str(exc)
                tick_fn("vision", False)
