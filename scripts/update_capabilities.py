#!/usr/bin/env python3
"""
Update Model Capabilities Cache
================================

This script discovers models from provider APIs and tests their capabilities,
saving the results to YAML cache files for use by the registry system.
It provides comprehensive cache management for both capability files and runtime cache.

Usage:
------
Update capabilities:
    python scripts/update_capabilities.py                    # All providers (discovery only)
    python scripts/update_capabilities.py --test             # Test new models (costs $!)
    python scripts/update_capabilities.py --provider openai  # Specific provider

View information:
    python scripts/update_capabilities.py --show-capabilities  # Show YAML files info
    python scripts/update_capabilities.py --show-cache         # Show runtime cache info

Clear capability files (YAML):
    python scripts/update_capabilities.py --clear-capabilities                  # All
    python scripts/update_capabilities.py --clear-capabilities --provider openai # Specific
    python scripts/update_capabilities.py --clear-capabilities --older-than 7    # Old files

Clear runtime cache (disk):
    python scripts/update_capabilities.py --clear-cache                  # All
    python scripts/update_capabilities.py --clear-cache --provider openai # Specific
    python scripts/update_capabilities.py --clear-cache --older-than 7    # Old entries

Clear both:
    python scripts/update_capabilities.py --clear-capabilities --clear-cache

CI/CD:
    python scripts/update_capabilities.py --ci --auto-commit  # Auto-discover & commit

What it does:
-------------
1. Discovers all models from provider APIs (OpenAI, Anthropic, Gemini, Ollama)
2. For new models (if --test flag), tests:
   - Tool calling support
   - Vision/multimodal support
   - JSON mode support
   - Streaming support
   - Maximum context length
   - Supported parameters
   - Speed (tokens/second)
3. Saves results to src/chuk_llm/registry/capabilities/{provider}.yaml
4. Optionally commits changes to git (--auto-commit)
5. Manages both capability YAML files and runtime disk cache

Cache Locations:
----------------
- Capability files: src/chuk_llm/registry/capabilities/*.yaml (committed to git)
- Runtime cache:    ~/.cache/chuk-llm/registry_cache.json (local only)

Note: Testing costs API credits! Use --test only when needed.
      API keys are loaded from .env file via python-dotenv.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars only

from chuk_llm.registry.sources import (
    AnthropicModelSource,
    DeepSeekModelSource,
    GeminiModelSource,
    GroqModelSource,
    MistralModelSource,
    OllamaSource,
    OpenAIModelSource,
    OpenRouterModelSource,
    PerplexityModelSource,
    WatsonxModelSource,
)


class CapabilityTester:
    """Tests model capabilities by making actual API calls."""

    def __init__(self, provider: str):
        self.provider = provider

    def _get_provider_client(self, model_name: str):
        """Get the appropriate provider client for testing."""
        if self.provider == "openai":
            from chuk_llm.llm.providers.openai_client import OpenAILLMClient

            return OpenAILLMClient(model=model_name)
        elif self.provider == "anthropic":
            from chuk_llm.llm.providers.anthropic_client import AnthropicLLMClient

            return AnthropicLLMClient(model=model_name)
        elif self.provider == "gemini":
            from chuk_llm.llm.providers.gemini_client import GeminiLLMClient

            return GeminiLLMClient(model=model_name)
        elif self.provider == "ollama":
            from chuk_llm.llm.providers.ollama_client import OllamaLLMClient

            return OllamaLLMClient(model=model_name)
        elif self.provider == "mistral":
            from chuk_llm.llm.providers.mistral_client import MistralLLMClient

            return MistralLLMClient(model=model_name)
        elif self.provider == "groq":
            from chuk_llm.llm.providers.groq_client import GroqAILLMClient

            return GroqAILLMClient(model=model_name)
        elif self.provider == "deepseek":
            from chuk_llm.llm.providers.openai_client import OpenAILLMClient

            return OpenAILLMClient(model=model_name)
        elif self.provider == "perplexity":
            from chuk_llm.llm.providers.perplexity_client import PerplexityLLMClient

            return PerplexityLLMClient(model=model_name)
        elif self.provider == "watsonx":
            from chuk_llm.llm.providers.watsonx_client import WatsonxLLMClient

            return WatsonxLLMClient(model=model_name)
        elif self.provider == "openrouter":
            from chuk_llm.llm.providers.openrouter_client import OpenRouterLLMClient

            return OpenRouterLLMClient(model=model_name)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def test_model(self, model_name: str) -> dict[str, Any]:
        """
        Test a model's capabilities by making real API calls.

        Tests:
        - Tool calling (send function definition)
        - Vision (send image)
        - JSON mode (request JSON output)
        - Streaming (check if supported)
        - Context length (binary search for max tokens)
        - Parameters (test which params are supported)
        - Speed (tokens per second benchmark)

        Args:
            model_name: Model to test

        Returns:
            Dict of test results
        """
        results = {
            "model": model_name,
            "tested_at": datetime.utcnow().isoformat(),
            "capabilities": {},
        }

        print(f"  Testing {model_name}...")

        # Test 0: Chat vs Completion model
        try:
            supports_chat = await self._test_chat_model(model_name)
            results["capabilities"]["supports_chat"] = supports_chat
            print(f"    ✓ Chat model: {supports_chat}")
        except Exception as e:
            results["capabilities"]["supports_chat"] = False
            results["errors"] = results.get("errors", [])
            error_msg = str(e)
            results["errors"].append(f"chat: {error_msg}")
            print(f"    ✗ Chat model: Failed - {error_msg}")
            supports_chat = False

        # If not a chat model, skip chat-based tests
        if not supports_chat:
            print("    ⚠ Skipping chat-based capability tests (not a chat model)")
            results["capabilities"]["supports_tools"] = False
            results["capabilities"]["supports_text"] = False
            results["capabilities"]["supports_vision"] = False
            results["capabilities"]["supports_audio_input"] = False
            results["capabilities"]["supports_json_mode"] = False
            results["capabilities"]["supports_structured_outputs"] = False
            results["capabilities"]["supports_streaming"] = False
            results["capabilities"]["max_context"] = None
            results["capabilities"]["known_params"] = []
            results["capabilities"]["tokens_per_second"] = None
            return results

        # Test 1: Tool calling
        try:
            supports_tools = await self._test_tools(model_name)
            results["capabilities"]["supports_tools"] = supports_tools
            print(f"    ✓ Tools: {supports_tools}")
        except Exception as e:
            results["capabilities"]["supports_tools"] = False
            results["errors"] = results.get("errors", [])
            error_msg = str(e)
            results["errors"].append(f"tools: {error_msg}")
            print(f"    ✗ Tools: Failed - {error_msg}")

        # Test 2: Text input/output (baseline capability)
        try:
            supports_text = await self._test_text(model_name)
            results["capabilities"]["supports_text"] = supports_text
            print(f"    ✓ Text I/O: {supports_text}")
        except Exception:
            results["capabilities"]["supports_text"] = False
            print("    ✗ Text I/O: False")

        # Test 3: Vision (image input)
        try:
            supports_vision = await self._test_vision(model_name)
            results["capabilities"]["supports_vision"] = supports_vision
            print(f"    ✓ Vision: {supports_vision}")
        except Exception:
            results["capabilities"]["supports_vision"] = False
            print("    ✗ Vision: False")

        # Test 4: Audio input
        try:
            supports_audio_input = await self._test_audio_input(model_name)
            results["capabilities"]["supports_audio_input"] = supports_audio_input
            print(f"    ✓ Audio input: {supports_audio_input}")
        except Exception:
            results["capabilities"]["supports_audio_input"] = False
            print("    ✗ Audio input: False")

        # Test 5: JSON mode
        try:
            supports_json = await self._test_json_mode(model_name)
            results["capabilities"]["supports_json_mode"] = supports_json
            print(f"    ✓ JSON mode: {supports_json}")
        except Exception:
            results["capabilities"]["supports_json_mode"] = False
            print("    ✗ JSON mode: False")

        # Test 5b: Structured outputs (JSON Schema)
        try:
            supports_structured = await self._test_structured_outputs(model_name)
            results["capabilities"]["supports_structured_outputs"] = supports_structured
            print(f"    ✓ Structured outputs: {supports_structured}")
        except Exception:
            results["capabilities"]["supports_structured_outputs"] = False
            print("    ✗ Structured outputs: False")

        # Test 6: Streaming
        try:
            supports_streaming = await self._test_streaming(model_name)
            results["capabilities"]["supports_streaming"] = supports_streaming
            print(f"    ✓ Streaming: {supports_streaming}")
        except Exception:
            results["capabilities"]["supports_streaming"] = False
            print("    ✗ Streaming: False")

        # Test 7: Context length
        try:
            max_context = await self._test_context_length(model_name)
            results["capabilities"]["max_context"] = max_context
            if max_context:
                print(f"    ✓ Max context: {max_context:,} tokens")
            else:
                print("    ✗ Context length: Not available")
        except Exception as e:
            print(f"    ✗ Context length: Failed ({e})")

        # Test 8: Parameters
        try:
            known_params = await self._test_parameters(model_name)
            results["capabilities"]["known_params"] = list(known_params)
            print(f"    ✓ Parameters: {', '.join(sorted(known_params))}")
        except Exception as e:
            print(f"    ✗ Parameters: Failed ({e})")

        # Test 9: Speed benchmark
        try:
            tokens_per_sec = await self._benchmark_speed(model_name)
            results["capabilities"]["tokens_per_second"] = tokens_per_sec
            if tokens_per_sec:
                print(f"    ✓ Speed: {tokens_per_sec:.1f} tokens/sec")
            else:
                print("    ✗ Speed: Not available")
        except Exception as e:
            print(f"    ✗ Speed: Failed ({e})")

        return results

    async def _test_chat_model(self, model_name: str) -> bool:
        """Test if model supports chat completions (vs legacy completions)."""
        from chuk_llm.core.enums import MessageRole
        from chuk_llm.core.models import Message

        # For OpenAI, we can infer from model name for known completion models
        if self.provider == "openai":
            model_lower = model_name.lower()
            # Known completion-only models (not chat models)
            if (
                "instruct" in model_lower
                and "gpt" in model_lower
                and "turbo" in model_lower
            ):
                return False
            if model_lower.startswith("text-") or model_lower.startswith("code-"):
                return False

        try:
            client = self._get_provider_client(model_name)

            messages = [Message(role=MessageRole.USER, content="Say hello")]

            # Try to make a chat completion
            response = await client.create_completion(messages, max_tokens=10)

            # Check if response is valid (not None or empty)
            if response and (response.get("response") or response.get("choices")):
                return True

            # Got a response but it's empty - probably failed
            return False

        except Exception as e:
            error_msg = str(e).lower()
            # If error mentions "not a chat model" or similar, it's a completion model
            if (
                "not a chat model" in error_msg
                or "chat/completions" in error_msg
                or "chat model" in error_msg
            ):
                return False
            # Other errors - could be rate limiting, network, etc - return False to be safe
            return False

    async def _test_tools(self, model_name: str) -> bool:
        """Test if model supports tool calling."""
        from chuk_llm.core.enums import MessageRole, ToolType
        from chuk_llm.core.models import Message, Tool, ToolFunction

        try:
            client = self._get_provider_client(model_name)

            # Define a simple tool
            tools = [
                Tool(
                    type=ToolType.FUNCTION,
                    function=ToolFunction(
                        name="get_weather",
                        description="Get the current weather for a location",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["location"],
                        },
                    ),
                )
            ]

            messages = [
                Message(
                    role=MessageRole.USER,
                    content="What's the weather in San Francisco?",
                )
            ]

            response = await client.create_completion(
                messages, tools=tools, max_tokens=50
            )

            # Check if response contains an error
            if isinstance(response, dict) and response.get("error"):
                return False

            # If we get tool_calls in response, tools are supported
            if response.get("tool_calls"):
                return True

            # No tool calls but no error - tools might be supported
            return True

        except Exception as e:
            error_msg = str(e).lower()
            # If error mentions tools/functions not being supported, return False
            if (
                "tool" in error_msg
                or "function" in error_msg
                or "not supported" in error_msg
            ):
                return False
            # Other errors - re-raise so we can see what's happening
            raise

    async def _test_vision(self, model_name: str) -> bool:
        """
        Test if model supports vision/image inputs.

        Tests both image_url (with base64 data URL) and image_data formats
        to comprehensively detect vision support.
        """
        from chuk_llm.core.enums import ContentType, MessageRole
        from chuk_llm.core.models import (
            ImageDataContent,
            ImageUrlContent,
            Message,
            TextContent,
        )

        # 16x16 red square (larger than 1x1 pixel for better model recognition)
        red_square = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAIElEQVR4nGP8z0AaYCJRPcOoBmIAE1GqkMCoBmIAyRoAQC4BH1m1rqAAAAAASUVORK5CYII="

        # Test 1: Try image_url with data URL (most common format)
        try:
            client = self._get_provider_client(model_name)

            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        TextContent(
                            type=ContentType.TEXT, text="What color is this image?"
                        ),
                        ImageUrlContent(
                            type=ContentType.IMAGE_URL,
                            image_url={"url": f"data:image/png;base64,{red_square}"},
                        ),
                    ],
                )
            ]

            response = await client.create_completion(messages, max_tokens=50)

            # Check if response contains an error
            if isinstance(response, dict) and response.get("error"):
                # Try alternative format before giving up
                pass
            elif (
                response
                and isinstance(response, dict)
                and (response.get("response") or response.get("choices"))
            ):
                # Success with image_url format - got actual content
                return True
            else:
                # No error but also no valid response - try alternative
                pass

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "image" in error_msg
                or "vision" in error_msg
                or "multimodal" in error_msg
                or "not supported" in error_msg
                or "invalid content type" in error_msg
            ):
                # Expected error for non-vision models
                pass
            else:
                # Unexpected error, but assume no vision support
                return False

        # Test 2: Try image_data format (alternative)
        try:
            client = self._get_provider_client(model_name)

            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        TextContent(
                            type=ContentType.TEXT, text="What color is this image?"
                        ),
                        ImageDataContent(
                            type=ContentType.IMAGE_DATA,
                            image_data=red_square,
                            mime_type="image/png",
                        ),
                    ],
                )
            ]

            response = await client.create_completion(messages, max_tokens=50)

            # Check if response contains an error
            if isinstance(response, dict) and response.get("error"):
                return False

            # Success with image_data format - check for actual content
            if (
                response
                and isinstance(response, dict)
                and (response.get("response") or response.get("choices"))
            ):
                return True

            return False

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "image" in error_msg
                or "vision" in error_msg
                or "multimodal" in error_msg
                or "not supported" in error_msg
                or "invalid content type" in error_msg
            ):
                return False
            return False

    async def _test_text(self, model_name: str) -> bool:
        """Test if model supports basic text input/output."""
        from chuk_llm.core.enums import MessageRole
        from chuk_llm.core.models import Message

        try:
            client = self._get_provider_client(model_name)

            messages = [Message(role=MessageRole.USER, content="Say hello in one word")]

            await client.create_completion(messages, max_tokens=10)
            # If we get a response, text is supported
            return True

        except Exception:
            # Most models should support text - if they don't, return False
            return False

    async def _test_audio_input(self, model_name: str) -> bool:
        """
        Test if model supports audio input.

        Uses a hybrid approach:
        1. Try actual API test with audio content
        2. Fall back to name-based inference for known models
        """
        model_lower = model_name.lower()

        # First, check known audio-capable models by name
        # This is more reliable than API testing which requires exact format knowledge
        audio_keywords = ["audio", "whisper", "speech"]
        if any(keyword in model_lower for keyword in audio_keywords):
            # Try to verify with API test, but default to True if inconclusive
            try:
                if self.provider == "openai":
                    result = await self._test_audio_input_openai(model_name)
                    # If test returned True, definitely supports audio
                    if result:
                        return True
                    # If False, could be test format issue - trust the name
                    return True
            except Exception:
                # API test failed, but name suggests audio support
                return True

        # For models without audio keywords, try API test
        try:
            if self.provider == "openai":
                return await self._test_audio_input_openai(model_name)

            # Generic test for other providers
            from chuk_llm.core.enums import ContentType, MessageRole
            from chuk_llm.core.models import AudioDataContent, Message, TextContent

            client = self._get_provider_client(model_name)
            tiny_wav_base64 = (
                "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQAAAAA="
            )

            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        TextContent(type=ContentType.TEXT, text="What do you hear?"),
                        AudioDataContent(
                            type=ContentType.AUDIO_DATA,
                            audio_data=tiny_wav_base64,
                            mime_type="audio/wav",
                        ),
                    ],
                )
            ]

            response = await client.create_completion(messages, max_tokens=50)
            return bool(
                response and (response.get("response") or response.get("choices"))
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "audio" in error_msg and (
                "not supported" in error_msg or "invalid" in error_msg
            ):
                return False
            # Inconclusive - default to False
            return False

    async def _test_audio_input_openai(self, model_name: str) -> bool:
        """Test audio input for OpenAI models using their specific format."""
        from chuk_llm.core.enums import ContentType, MessageRole
        from chuk_llm.core.models import InputAudioContent, Message

        try:
            client = self._get_provider_client(model_name)

            # Minimal WAV file base64
            tiny_wav_base64 = (
                "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQAAAAA="
            )

            # OpenAI's audio models use InputAudioContent with input_audio dict
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        InputAudioContent(
                            type=ContentType.INPUT_AUDIO,
                            input_audio={"data": tiny_wav_base64, "format": "wav"},
                        )
                    ],
                )
            ]

            response = await client.create_completion(
                messages, max_tokens=50, modalities=["text", "audio"]
            )

            # Check if response contains an error
            if isinstance(response, dict) and response.get("error"):
                return False

            return bool(
                response and (response.get("response") or response.get("choices"))
            )

        except Exception as e:
            error_msg = str(e).lower()
            # Check for specific audio errors
            if "modalities" in error_msg or "audio" in error_msg:
                if "not supported" in error_msg or "invalid" in error_msg:
                    return False
            return False

    async def _test_json_mode(self, model_name: str) -> bool:
        """Test if model supports JSON mode."""
        from chuk_llm.core.enums import MessageRole
        from chuk_llm.core.models import Message

        try:
            client = self._get_provider_client(model_name)

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful assistant that outputs JSON.",
                ),
                Message(
                    role=MessageRole.USER,
                    content="Return JSON with a greeting field.",
                ),
            ]

            response = await client.create_completion(
                messages,
                response_format={"type": "json_object"},
                max_tokens=50,
            )

            # Check if response contains an error
            if isinstance(response, dict) and response.get("error"):
                return False

            return True

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "json" in error_msg
                or "response_format" in error_msg
                or "not supported" in error_msg
                or "invalid parameter" in error_msg
            ):
                return False
            return False

    async def _test_structured_outputs(self, model_name: str) -> bool:
        """Test if model supports structured outputs with JSON Schema."""
        from chuk_llm.core.enums import MessageRole
        from chuk_llm.core.models import Message

        try:
            client = self._get_provider_client(model_name)

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful assistant that outputs structured data.",
                ),
                Message(
                    role=MessageRole.USER,
                    content="Generate a person with name and age.",
                ),
            ]

            # Define a JSON schema for structured output
            json_schema = {
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            }

            response = await client.create_completion(
                messages,
                response_format={"type": "json_schema", "json_schema": json_schema},
                max_tokens=50,
            )

            # Check if response contains an error
            if isinstance(response, dict) and response.get("error"):
                return False

            return True

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "json_schema" in error_msg
                or "structured" in error_msg
                or "response_format" in error_msg
                or "not supported" in error_msg
                or "invalid parameter" in error_msg
                or "strict" in error_msg
            ):
                return False
            return False

    async def _test_streaming(self, model_name: str) -> bool:
        """Test if model supports streaming."""
        from chuk_llm.core.enums import MessageRole
        from chuk_llm.core.models import Message

        try:
            client = self._get_provider_client(model_name)

            messages = [Message(role=MessageRole.USER, content="Say hello")]

            chunks = 0
            async for chunk in client.create_completion(
                messages, stream=True, max_tokens=10
            ):
                if chunk.get("response"):
                    chunks += 1
                    if chunks > 0:
                        return True

            return chunks > 0

        except Exception:
            return False

    async def _test_context_length(self, model_name: str) -> int | None:
        """
        Test model's maximum context length.

        Uses model_info if available, otherwise tests empirically.
        """
        try:
            client = self._get_provider_client(model_name)
            info = client.get_model_info()

            # Try to get from model info first
            max_context = info.get("max_context_length")
            if max_context:
                return max_context

            # Fallback: empirical testing (but this is expensive, skip for now)
            return None

        except Exception:
            return None

    async def _test_parameters(self, model_name: str) -> set[str]:
        """
        Test which parameters the model supports.

        Gets from model_info, then empirically tests if needed.
        """
        from chuk_llm.core.enums import MessageRole
        from chuk_llm.core.models import Message

        try:
            client = self._get_provider_client(model_name)
            info = client.get_model_info()

            # Try to get from model info first
            known_params = info.get("known_params", set())
            if known_params:
                return known_params

            # Fallback: test common parameters
            supported = set()
            test_params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 20,
            }

            messages = [Message(role=MessageRole.USER, content="Hi")]

            for param, value in test_params.items():
                try:
                    kwargs = {param: value}
                    await client.create_completion(messages, **kwargs)
                    supported.add(param)
                except Exception:
                    continue

            return supported

        except Exception:
            return set()

    async def _benchmark_speed(self, model_name: str) -> float | None:
        """
        Benchmark model speed in tokens per second.

        Makes a streaming request and measures output tokens/time.
        """
        import time

        from chuk_llm.core.enums import MessageRole
        from chuk_llm.core.models import Message

        try:
            client = self._get_provider_client(model_name)

            messages = [
                Message(
                    role=MessageRole.USER,
                    content="Write a short paragraph about Python programming.",
                )
            ]

            start_time = time.time()
            token_count = 0

            async for chunk in client.create_completion(
                messages, stream=True, max_tokens=200
            ):
                if chunk.get("response"):
                    # Rough estimate: count words as tokens
                    token_count += len(chunk["response"].split())

            elapsed = time.time() - start_time

            if elapsed > 0 and token_count > 0:
                return token_count / elapsed

            return None

        except Exception:
            return None


class CapabilityCacheManager:
    """Manages YAML capability cache files."""

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is None:
            cache_dir = (
                Path(__file__).parent.parent
                / "src"
                / "chuk_llm"
                / "registry"
                / "capabilities"
            )

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_cache(self, provider: str) -> dict:
        """Load existing cache for provider."""
        cache_file = self.cache_dir / f"{provider}.yaml"

        if not cache_file.exists():
            return {
                "provider": provider,
                "last_updated": None,
                "families": {},
                "models": {},
            }

        with open(cache_file) as f:
            return yaml.safe_load(f) or {}

    def save_cache(self, provider: str, cache_data: dict):
        """Save cache for provider."""
        cache_file = self.cache_dir / f"{provider}.yaml"

        # Add header comment
        cache_data["last_updated"] = datetime.utcnow().isoformat()

        with open(cache_file, "w") as f:
            f.write(f"# Auto-generated capability cache for {provider}\n")
            f.write(f"# Last updated: {cache_data['last_updated']}\n")
            f.write(
                "# DO NOT EDIT MANUALLY - Run: python scripts/update_capabilities.py\n\n"
            )
            yaml.dump(cache_data, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Saved cache to {cache_file}")

    def detect_family(self, model_name: str, provider: str = None) -> str | None:
        """Detect model family from name and provider."""
        name_lower = model_name.lower()

        # OpenAI families
        if "gpt-5" in name_lower:
            return "gpt-5"
        elif "gpt-4o" in name_lower:
            return "gpt-4o"
        elif "gpt-4" in name_lower:
            return "gpt-4"
        elif "gpt-3.5" in name_lower:
            return "gpt-3.5"
        elif name_lower.startswith("o1"):
            return "o1"
        elif name_lower.startswith("o3"):
            return "o3"

        # Anthropic families
        elif "claude-4" in name_lower:
            return "claude-4"
        elif "claude-3-5" in name_lower or "claude-3.5" in name_lower:
            return "claude-3.5"
        elif "claude-3" in name_lower:
            return "claude-3"

        # Gemini families
        elif "gemini-2.5" in name_lower:
            return "gemini-2.5"
        elif "gemini-2.0" in name_lower or "gemini-2" in name_lower:
            return "gemini-2.0"
        elif "gemini-1.5" in name_lower:
            return "gemini-1.5"

        # Ollama - detect base model families
        elif provider == "ollama":
            # Extract base model name (before version)
            base = name_lower.split(":")[0]
            if "llama" in base:
                return base  # llama2, llama3, llama3.1, etc.
            elif (
                "qwen" in base
                or "phi" in base
                or "mistral" in base
                or "granite" in base
                or "gemma" in base
            ):
                return base

        return None


async def update_provider_capabilities(provider_name: str, test_models: bool = False):
    """
    Update capabilities cache for a provider.

    Args:
        provider_name: Provider to update (openai, anthropic, gemini)
        test_models: Whether to test new models (costs API credits)
    """
    print(f"\n{'=' * 70}")
    print(f"Updating {provider_name} capabilities")
    print("=" * 70)

    # Initialize
    cache_mgr = CapabilityCacheManager()
    tester = CapabilityTester(provider_name)

    # Load existing cache
    cache = cache_mgr.load_cache(provider_name)
    print(f"Loaded cache: {len(cache.get('models', {}))} models")

    # Discover current models
    try:
        if provider_name == "openai":
            source = OpenAIModelSource()
        elif provider_name == "anthropic":
            source = AnthropicModelSource()
        elif provider_name == "gemini":
            source = GeminiModelSource()
        elif provider_name == "ollama":
            source = OllamaSource()
        elif provider_name == "mistral":
            source = MistralModelSource()
        elif provider_name == "groq":
            source = GroqModelSource()
        elif provider_name == "deepseek":
            source = DeepSeekModelSource()
        elif provider_name == "perplexity":
            source = PerplexityModelSource()
        elif provider_name == "watsonx":
            source = WatsonxModelSource()
        elif provider_name == "openrouter":
            source = OpenRouterModelSource()
        else:
            print(f"❌ Unknown provider: {provider_name}")
            print(
                "   Supported: openai, anthropic, gemini, ollama, mistral, groq, deepseek, perplexity, watsonx, openrouter"
            )
            return

        specs = await source.discover()
        print(f"Discovered: {len(specs)} models")
    except Exception as e:
        print(f"❌ Failed to discover {provider_name} models: {e}")
        print("   Check that:")
        if provider_name == "openai":
            print("   - OPENAI_API_KEY is set in environment")
        elif provider_name == "anthropic":
            print("   - ANTHROPIC_API_KEY is set in environment")
        elif provider_name == "gemini":
            print("   - GEMINI_API_KEY or GOOGLE_API_KEY is set in environment")
        elif provider_name == "ollama":
            print("   - Ollama is running locally (http://localhost:11434)")
        return

    # Check for new models
    existing_models = set(cache.get("models", {}).keys())
    discovered_models = {spec.name for spec in specs}
    new_models = discovered_models - existing_models
    removed_models = existing_models - discovered_models

    print("\nChanges:")
    print(f"  New models: {len(new_models)}")
    print(f"  Removed: {len(removed_models)}")

    if new_models:
        print("\n  New models to process:")
        for model in sorted(new_models):
            print(f"    • {model}")

    # Process new models
    if new_models:
        if test_models:
            print(f"\n🧪 Testing {len(new_models)} new models...")
        else:
            print(
                f"\n📝 Adding {len(new_models)} new models (discovery only, no testing)"
            )

        for model_name in sorted(new_models):
            # Detect family
            family = cache_mgr.detect_family(model_name, provider_name)

            # Initialize models dict if needed
            if "models" not in cache:
                cache["models"] = {}

            # If testing, run full capability tests
            if test_models:
                test_results = await tester.test_model(model_name)

                # Check if model should be added to registry
                # Skip models that:
                # 1. Have no capabilities at all (all False/None)
                # 2. Failed basic tests with errors
                # 3. Don't support chat AND don't support text (not usable)
                capabilities = test_results.get("capabilities", {})

                # Must support at least one of: chat or text
                supports_basic_io = capabilities.get(
                    "supports_chat"
                ) or capabilities.get("supports_text")

                # Check for any additional capabilities
                has_any_capability = any(
                    [
                        capabilities.get("supports_tools"),
                        capabilities.get("supports_vision"),
                        capabilities.get("supports_audio_input"),
                        capabilities.get("supports_json_mode"),
                        capabilities.get("supports_streaming"),
                        capabilities.get("max_context"),
                        capabilities.get("tokens_per_second"),
                        capabilities.get("known_params"),
                    ]
                )

                has_critical_errors = "errors" in test_results

                # Skip if: no basic I/O support OR (no capabilities AND has errors)
                if not supports_basic_io or (
                    not has_any_capability and has_critical_errors
                ):
                    reason = (
                        "no basic I/O"
                        if not supports_basic_io
                        else "failed tests and no capabilities"
                    )
                    print(f"    ⚠ Skipping {model_name} - {reason}")
                    continue

                # Model passed - add to cache
                cache["models"][model_name] = {
                    "inherits_from": family,
                    "tested_at": test_results["tested_at"],
                    **test_results["capabilities"],
                }

                # Save cache immediately after each model test
                # This allows resuming if script fails partway through
                cache_mgr.save_cache(provider_name, cache)
            else:
                # Discovery only - get capabilities from existing resolvers
                from chuk_llm.registry.models import ModelSpec
                from chuk_llm.registry.resolvers import (
                    GeminiCapabilityResolver,
                    HeuristicCapabilityResolver,
                    OllamaCapabilityResolver,
                )

                spec = ModelSpec(provider=provider_name, name=model_name, family=family)

                # Try resolvers in order: Provider-specific -> Heuristic
                resolvers = [
                    GeminiCapabilityResolver() if provider_name == "gemini" else None,
                    OllamaCapabilityResolver() if provider_name == "ollama" else None,
                    HeuristicCapabilityResolver(),
                ]

                capabilities = None
                for resolver in resolvers:
                    if resolver is None:
                        continue
                    try:
                        caps = await resolver.get_capabilities(spec)
                        if caps and (
                            caps.max_context or caps.supports_tools is not None
                        ):
                            capabilities = caps
                            break
                    except Exception:
                        continue

                if capabilities:
                    # Convert capabilities to dict for YAML
                    cap_dict = capabilities.model_dump(
                        exclude_none=True, exclude={"source", "last_updated"}
                    )
                    # Convert sets to lists for YAML
                    if "known_params" in cap_dict:
                        cap_dict["known_params"] = list(cap_dict["known_params"])
                    # Convert enum to string
                    if "quality_tier" in cap_dict:
                        cap_dict["quality_tier"] = (
                            cap_dict["quality_tier"].value
                            if hasattr(cap_dict["quality_tier"], "value")
                            else cap_dict["quality_tier"]
                        )

                    cache["models"][model_name] = {
                        "inherits_from": family,
                        "discovered_at": datetime.utcnow().isoformat(),
                        **cap_dict,
                    }
                else:
                    # No capabilities found - just add minimal entry
                    cache["models"][model_name] = {
                        "inherits_from": family,
                        "discovered_at": datetime.utcnow().isoformat(),
                    }

    # Save updated cache
    cache_mgr.save_cache(provider_name, cache)

    print(f"\n✅ {provider_name} capabilities updated")


def clear_capabilities(provider: str | None = None, older_than_days: int | None = None):
    """
    Clear the YAML capability cache files.

    Args:
        provider: Specific provider to clear (None = all)
        older_than_days: Only clear capability files older than N days (None = all)
    """
    import time
    from pathlib import Path

    # Get capabilities directory
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent / "src" / "chuk_llm"
    capabilities_dir = package_dir / "registry" / "capabilities"

    if not capabilities_dir.exists():
        print("✓ No capability files to clear")
        return

    print(f"\n{'=' * 70}")
    print("Clearing capability cache files")
    print("=" * 70)
    print(f"Location: {capabilities_dir}")

    if provider:
        print(f"Provider filter: {provider}")
    if older_than_days:
        print(f"Age filter: older than {older_than_days} days")

    # Calculate cutoff time if needed
    cutoff_time = None
    if older_than_days:
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

    cleared_count = 0
    skipped_count = 0

    # Find capability YAML files
    for yaml_file in capabilities_dir.glob("*.yaml"):
        file_provider = yaml_file.stem  # filename without extension

        # Check provider filter
        if provider and file_provider != provider:
            skipped_count += 1
            continue

        # Check age filter
        if cutoff_time:
            file_mtime = yaml_file.stat().st_mtime
            if file_mtime > cutoff_time:
                skipped_count += 1
                continue

        # Remove the file
        yaml_file.unlink()
        cleared_count += 1
        print(f"  ✓ Removed: {yaml_file.name}")

    print(f"\n✅ Cleared {cleared_count} capability files")
    if skipped_count:
        print(f"   Skipped {skipped_count} files (didn't match filters)")


def clear_registry_cache(
    provider: str | None = None, older_than_days: int | None = None
):
    """
    Clear the registry disk cache.

    Args:
        provider: Specific provider to clear (None = all)
        older_than_days: Only clear caches older than N days (None = all)
    """
    import json
    import time
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "chuk-llm"
    cache_file = cache_dir / "registry_cache.json"

    if not cache_file.exists():
        print("✓ No registry cache to clear")
        return

    print(f"\n{'=' * 70}")
    print("Clearing registry disk cache")
    print("=" * 70)
    print(f"Cache location: {cache_file}")

    if provider:
        print(f"Provider filter: {provider}")
    if older_than_days:
        print(f"Age filter: older than {older_than_days} days")

    # Calculate cutoff time if needed
    cutoff_time = None
    if older_than_days:
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

    # If clearing everything without filters, just remove the file
    if not provider and not older_than_days:
        cache_file.unlink()
        print("  ✓ Removed entire cache")
        print("\n✅ Registry cache cleared")
        return

    # Otherwise, need to selectively remove entries
    try:
        with open(cache_file) as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read cache: {e}")
        return

    cleared_count = 0
    kept_count = 0

    # Filter cache entries
    new_cache = {}
    for key, value in cache_data.items():
        # Parse key format: "provider:model_name"
        if ":" in key:
            entry_provider = key.split(":")[0]
        else:
            # Keep entries without provider info
            new_cache[key] = value
            kept_count += 1
            continue

        # Check provider filter
        if provider and entry_provider != provider:
            new_cache[key] = value
            kept_count += 1
            continue

        # Check age filter if specified
        if cutoff_time and isinstance(value, dict):
            # Try to get timestamp from cache entry
            entry_time = value.get("cached_at", time.time())
            if entry_time > cutoff_time:
                new_cache[key] = value
                kept_count += 1
                continue

        # This entry should be cleared
        cleared_count += 1
        print(f"  ✓ Removed: {key}")

    # Save the filtered cache
    if new_cache:
        with open(cache_file, "w") as f:
            json.dump(new_cache, f, indent=2)
        print(f"\n✅ Cleared {cleared_count} entries, kept {kept_count}")
    else:
        # No entries left, remove the file
        cache_file.unlink()
        print(f"\n✅ Cleared {cleared_count} entries (removed cache file)")


def show_capabilities_info():
    """Show information about capability YAML files."""
    import time
    from pathlib import Path

    # Get capabilities directory
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent / "src" / "chuk_llm"
    capabilities_dir = package_dir / "registry" / "capabilities"

    print(f"\n{'=' * 70}")
    print("Capability Files Information")
    print("=" * 70)
    print(f"Location: {capabilities_dir}")

    if not capabilities_dir.exists():
        print("Status: No capability files exist")
        return

    # Find YAML files
    yaml_files = list(capabilities_dir.glob("*.yaml"))

    if not yaml_files:
        print("Status: Directory exists but no YAML files found")
        return

    print(f"Total files: {len(yaml_files)}")

    # Load and show info for each provider
    print("\nProviders:")
    for yaml_file in sorted(yaml_files):
        provider = yaml_file.stem
        file_mtime = yaml_file.stat().st_mtime
        file_age_days = (time.time() - file_mtime) / (24 * 60 * 60)
        file_size_kb = yaml_file.stat().st_size / 1024

        # Try to count models
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                model_count = len(data.get("models", {}))
        except Exception:
            model_count = "?"

        print(f"  {provider}:")
        print(f"    Models: {model_count}")
        print(f"    Size: {file_size_kb:.1f} KB")
        print(f"    Age: {file_age_days:.1f} days")


def show_cache_info():
    """Show information about the registry disk cache."""
    import json
    import time
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "chuk-llm"
    cache_file = cache_dir / "registry_cache.json"

    print(f"\n{'=' * 70}")
    print("Registry Disk Cache Information")
    print("=" * 70)
    print(f"Location: {cache_file}")

    if not cache_file.exists():
        print("Status: No cache exists")
        return

    # Load cache data
    try:
        with open(cache_file) as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"Status: Cache file exists but cannot be read ({e})")
        return

    if not cache_data:
        print("Status: Cache file is empty")
        return

    print(f"Total entries: {len(cache_data)}")

    # Group by provider
    by_provider = {}
    for key in cache_data:
        # Parse key format: "provider:model_name"
        if ":" in key:
            provider = key.split(":")[0]
            by_provider[provider] = by_provider.get(provider, 0) + 1

    if by_provider:
        print("\nBy Provider:")
        for provider, count in sorted(by_provider.items()):
            print(f"  {provider}: {count} models")

    # Show file age
    file_mtime = cache_file.stat().st_mtime
    file_age_days = (time.time() - file_mtime) / (24 * 60 * 60)

    print(f"\nCache file age: {file_age_days:.1f} days old")

    # Show file size
    file_size_kb = cache_file.stat().st_size / 1024
    print(f"Cache file size: {file_size_kb:.1f} KB")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update model capability caches and manage registry cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all providers (discovery only)
  python scripts/update_capabilities.py

  # Update specific provider with testing
  python scripts/update_capabilities.py --provider openai --test

  # Show capability files info
  python scripts/update_capabilities.py --show-capabilities

  # Show registry disk cache info
  python scripts/update_capabilities.py --show-cache

  # Clear capability YAML files
  python scripts/update_capabilities.py --clear-capabilities

  # Clear capability file for specific provider
  python scripts/update_capabilities.py --clear-capabilities --provider openai

  # Clear old capability files (7+ days)
  python scripts/update_capabilities.py --clear-capabilities --older-than 7

  # Clear registry disk cache
  python scripts/update_capabilities.py --clear-cache

  # Clear both capabilities and cache
  python scripts/update_capabilities.py --clear-capabilities --clear-cache

  # CI mode (discovery only, auto-commit)
  python scripts/update_capabilities.py --ci --auto-commit
        """,
    )

    # Action commands
    parser.add_argument(
        "--show-capabilities",
        action="store_true",
        help="Show capability files information and exit",
    )
    parser.add_argument(
        "--show-cache",
        action="store_true",
        help="Show registry disk cache information and exit",
    )
    parser.add_argument(
        "--clear-capabilities",
        action="store_true",
        help="Clear the capability YAML files",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear the registry disk cache"
    )

    # Provider selection
    parser.add_argument(
        "--provider",
        choices=[
            "openai",
            "anthropic",
            "gemini",
            "ollama",
            "mistral",
            "groq",
            "deepseek",
            "perplexity",
            "watsonx",
            "openrouter",
            "all",
        ],
        default="all",
        help="Provider to update/clear (default: all)",
    )

    # Update options
    parser.add_argument(
        "--test", action="store_true", help="Test new models (costs API credits!)"
    )
    parser.add_argument(
        "--ci", action="store_true", help="Running in CI - don't test, just discover"
    )
    parser.add_argument(
        "--auto-commit", action="store_true", help="Auto-commit changes to git"
    )

    # Cache clearing options
    parser.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="Only clear cache entries older than N days",
    )

    args = parser.parse_args()

    # Handle show commands
    if args.show_capabilities:
        show_capabilities_info()
        return

    if args.show_cache:
        show_cache_info()
        return

    # Handle clearing commands
    provider_filter = None if args.provider == "all" else args.provider

    if args.clear_capabilities:
        clear_capabilities(provider=provider_filter, older_than_days=args.older_than)

    if args.clear_cache:
        clear_registry_cache(provider=provider_filter, older_than_days=args.older_than)

    # If we cleared anything, exit (don't also update)
    if args.clear_capabilities or args.clear_cache:
        return

    # Determine which providers to update
    providers = (
        [
            "openai",
            "anthropic",
            "gemini",
            "ollama",
            "mistral",
            "groq",
            "deepseek",
            "perplexity",
            "watsonx",
            "openrouter",
        ]
        if args.provider == "all"
        else [args.provider]
    )

    # Update each provider
    for provider in providers:
        try:
            await update_provider_capabilities(
                provider, test_models=args.test and not args.ci
            )
        except Exception as e:
            print(f"❌ Error updating {provider}: {e}")
            continue

    # Auto-commit if requested
    if args.auto_commit:
        print("\n📝 Committing changes...")
        import subprocess

        subprocess.run(["git", "add", "src/chuk_llm/registry/capabilities/"])
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "chore: update model capability caches\n\nAuto-updated by update_capabilities.py",
            ]
        )
        print("✅ Changes committed")

    print("\n" + "=" * 70)
    print("✅ ALL PROVIDERS UPDATED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
