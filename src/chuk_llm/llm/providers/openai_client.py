# chuk_llm/llm/providers/openai_client.py - COMPLETE VERSION WITH GPT-5 SUPPORT
"""
OpenAI chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around the official `openai` SDK that uses the unified
configuration system and universal tool name compatibility.

COMPLETE FIXES INCLUDING GPT-5 SUPPORT:
1. Added ToolCompatibilityMixin inheritance for universal tool names
2. Fixed conversation flow tool name handling
3. Enhanced content extraction to eliminate warnings
4. Added bidirectional mapping throughout conversation
5. FIXED streaming tool call duplication bug - MAJOR FIX
6. ADDED comprehensive reasoning model support (o1, o3, o4, o5)
7. ADDED GPT-5 family support (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat)
8. ADDED automatic parameter mapping (max_tokens -> max_completion_tokens)
9. ADDED system message conversion for o1 models
10. FIXED streaming chunk yielding to be properly incremental
11. REMOVED o1-preview references (no longer available)
12. ADDED smart defaults for newly discovered OpenAI models
13. FIXED GPT-5 parameter restrictions (no temperature control)
14. ADDED GPT-5 generation handling and proper defaults
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import openai

from chuk_llm.configuration import get_config
from chuk_llm.core.enums import MessageRole

# base
from chuk_llm.llm.core.base import BaseLLMClient
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

# mixins
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

log = logging.getLogger(__name__)


class OpenAILLMClient(
    ConfigAwareProviderMixin, ToolCompatibilityMixin, OpenAIStyleMixin, BaseLLMClient
):
    """
    Configuration-driven wrapper around the official `openai` SDK that gets
    all capabilities from the unified YAML configuration.

    COMPLETE VERSION: Now includes GPT-5 family support, reasoning model support,
    FIXED streaming, and smart defaults.

    JSON Function Calling Fallback:
    For APIs that don't natively support function calling, enable fallback mode:
        ENABLE_JSON_FUNCTION_FALLBACK = True
        SUPPORTS_TOOL_ROLE = False  # Convert 'tool' to 'user' role
        SUPPORTS_FUNCTION_ROLE = False
    """

    # JSON Function Calling Fallback Configuration
    # Subclasses can override these to enable fallback behavior
    ENABLE_JSON_FUNCTION_FALLBACK = False  # Enable JSON-based function calling
    SUPPORTS_TOOL_ROLE = True  # Does API support 'tool' role messages?
    SUPPORTS_FUNCTION_ROLE = True  # Does API support 'function' role messages?

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        # Detect provider from api_base for configuration lookup
        detected_provider = self._detect_provider_name(api_base)

        # Initialize ALL mixins including ToolCompatibilityMixin
        ConfigAwareProviderMixin.__init__(self, detected_provider, model)
        ToolCompatibilityMixin.__init__(self, detected_provider)

        self.model = model
        self.api_base = api_base
        self.detected_provider = detected_provider

        # Use AsyncOpenAI - we are async-native
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)

        log.debug(
            f"OpenAI client initialized: provider={self.detected_provider}, model={self.model}"
        )

    def _detect_provider_name(self, api_base: str | None) -> str:
        """Detect provider name from API base URL for configuration lookup"""
        if not api_base:
            return "openai"

        api_base_lower = api_base.lower()
        # Check for OpenAI first (most common)
        if "api.openai.com" in api_base_lower:
            return "openai"
        elif "deepseek" in api_base_lower:
            return "deepseek"
        elif "groq" in api_base_lower:
            return "groq"
        elif "together" in api_base_lower:
            return "together"
        elif "perplexity" in api_base_lower:
            return "perplexity"
        elif "anyscale" in api_base_lower:
            return "anyscale"
        else:
            return "openai_compatible"

    def detect_provider_name(self) -> str:
        """Public method to detect provider name"""
        return self.detected_provider

    # ================================================================
    # JSON FUNCTION CALLING FALLBACK METHODS
    # ================================================================

    def _create_json_function_calling_prompt(self, tools: list) -> str:
        """Create system prompt for JSON-based function calling (when fallback enabled)."""

        tool_descriptions = []
        for tool in tools:
            tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else tool
            func = tool_dict.get("function", {})
            tool_descriptions.append(
                f"- {func.get('name')}: {func.get('description', '')}"
            )

        tools_text = "\n".join(tool_descriptions)

        return (
            f"You are a helpful assistant with access to functions. "
            f"Available functions:\n{tools_text}\n\n"
            f"When you need to call a function, respond with ONLY a JSON object: "
            f'{{"name": "function_name", "arguments": {{...}}}}. '
            f"After receiving tool results, answer naturally using that information."
        )

    def _parse_function_call_from_json(self, content: str) -> dict[str, Any] | None:
        """Parse function call JSON from response content (when fallback enabled)."""
        import re

        if not content:
            return None

        # Try direct JSON parse
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                log.debug(f"[json_fallback] Parsed function call: {parsed['name']}")
                return parsed
        except json.JSONDecodeError:
            pass

        # Try JSON in code blocks
        match = re.search(r"```(?:json)?\s*(\{[^`]+\})\s*```", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if (
                    isinstance(parsed, dict)
                    and "name" in parsed
                    and "arguments" in parsed
                ):
                    log.debug(
                        f"[json_fallback] Parsed from code block: {parsed['name']}"
                    )
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _convert_to_tool_calls_from_json(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert JSON function call in content to tool_calls format (when fallback enabled)."""
        content = response.get("response", "")
        if not content:
            return response

        function_call = self._parse_function_call_from_json(content)
        if not function_call:
            return response

        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": function_call["name"],
                "arguments": json.dumps(function_call["arguments"]),
            },
        }

        response["tool_calls"] = [tool_call]
        response["response"] = None

        log.info(f"[json_fallback] Converted to tool_call: {function_call['name']}")
        return response

    def _convert_incompatible_message_roles(
        self, messages_dicts: list[dict]
    ) -> tuple[list[dict], bool]:
        """
        Convert message roles that the API doesn't support (when fallback enabled).

        Returns:
            tuple: (modified_messages, has_tool_results)
        """
        has_tool_results = False

        for msg_dict in messages_dicts:
            role = msg_dict.get("role")

            # Convert 'tool' role if not supported
            if role == MessageRole.TOOL.value and not self.SUPPORTS_TOOL_ROLE:
                has_tool_results = True
                func_name = msg_dict.get("name", "unknown")
                result_content = msg_dict.get("content", "")
                msg_dict["role"] = MessageRole.USER.value
                msg_dict["content"] = f"Tool result from {func_name}: {result_content}"
                msg_dict.pop("tool_call_id", None)
                msg_dict.pop("name", None)
                log.debug(
                    f"[json_fallback] Converted 'tool' role to 'user' role for {func_name}"
                )

            # Convert 'function' role if not supported
            elif role == "function" and not self.SUPPORTS_FUNCTION_ROLE:
                has_tool_results = True
                func_name = msg_dict.get("name", "unknown")
                result_content = msg_dict.get("content", "")
                msg_dict["role"] = MessageRole.USER.value
                msg_dict["content"] = (
                    f"Function result from {func_name}: {result_content}"
                )
                msg_dict.pop("name", None)
                log.debug(
                    f"[json_fallback] Converted 'function' role to 'user' role for {func_name}"
                )

            # Check for tool results in other formats
            elif role == MessageRole.TOOL.value:
                has_tool_results = True

        return messages_dicts, has_tool_results

    # ================================================================
    # END JSON FUNCTION CALLING FALLBACK METHODS
    # ================================================================

    def _add_strict_parameter_to_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Add strict parameter to all function tools for OpenAI-compatible APIs.

        Some OpenAI-compatible APIs require tools.function.strict to be present
        as a boolean value for all function definitions.
        """
        from chuk_llm.core.enums import ToolType

        modified_tools = []
        for tool in tools:
            # Handle both dict and Pydantic models
            if isinstance(tool, dict):
                tool_copy = tool.copy()
            else:
                # Pydantic model - convert to dict
                tool_copy = (
                    tool.model_dump() if hasattr(tool, "model_dump") else tool.dict()
                )

            if (
                tool_copy.get("type") == ToolType.FUNCTION.value
                and "function" in tool_copy
            ):
                # Make a copy of the function dict to avoid modifying the original
                func_copy = (
                    tool_copy["function"].copy()
                    if isinstance(tool_copy["function"], dict)
                    else tool_copy["function"]
                )
                if "strict" not in func_copy:
                    func_copy["strict"] = False
                    log.debug(
                        f"[{self.detected_provider}] Added strict=False to tool: {func_copy.get('name', 'unknown')}"
                    )
                tool_copy["function"] = func_copy
            modified_tools.append(tool_copy)
        return modified_tools

    # ================================================================
    # SMART DEFAULTS FOR NEW OPENAI MODELS
    # ================================================================

    @staticmethod
    def _get_smart_default_features(model_name: str) -> set[str]:
        """Get smart default features for an OpenAI model based on naming patterns"""
        model_lower = str(model_name).lower()

        # Base features that ALL modern OpenAI models should have
        base_features = {"text", "streaming", "system_messages"}

        # Pattern-based feature detection
        if any(pattern in model_lower for pattern in ["o1", "o3", "o4", "o5", "o6"]):
            # Reasoning models
            if "o1" in model_lower:
                # O1 models are legacy - no tools, limited capabilities
                return {"text", "reasoning"}
            else:
                # O3+ models are modern reasoning - assume full tool support
                return {"text", "streaming", "tools", "reasoning", "system_messages"}

        elif any(pattern in model_lower for pattern in ["gpt-4", "gpt-3.5", "gpt-5"]):
            # Standard GPT models - assume modern capabilities
            features = base_features | {"tools", "json_mode"}

            # Vision support for GPT-4+ models (not 3.5)
            if any(v in model_lower for v in ["gpt-4", "gpt-5"]):
                features.add("vision")

            # GPT-5 models use reasoning architecture
            if "gpt-5" in model_lower:
                features.add("reasoning")

            return features

        else:
            # Unknown OpenAI model patterns - be optimistic about tool support
            # KEY PRINCIPLE: Assume new OpenAI models support tools by default
            log.info(
                f"Unknown OpenAI model pattern '{model_name}' - assuming modern capabilities including tools"
            )
            return base_features | {"tools", "json_mode"}

    @staticmethod
    def _get_smart_default_parameters(model_name: str) -> dict[str, Any]:
        """Get smart default parameters for an OpenAI model"""
        model_lower = str(model_name).lower()

        # Reasoning model parameter handling
        if any(pattern in model_lower for pattern in ["o1", "o3", "o4", "o5", "gpt-5"]):
            return {
                "max_context_length": 272000 if "gpt-5" in model_lower else 200000,
                "max_output_tokens": 128000 if "gpt-5" in model_lower else 65536,
                "requires_max_completion_tokens": True,
                "parameter_mapping": {"max_tokens": "max_completion_tokens"},
                "unsupported_params": [
                    "temperature",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                ],
                "supports_tools": "o1"
                not in model_lower,  # Only O1 doesn't support tools
            }

        # Standard model defaults - generous assumptions for new models
        return {
            "max_context_length": 128000,
            "max_output_tokens": 8192,
            "supports_tools": True,  # Assume new models support tools
        }

    def _has_explicit_model_config(self, model: str = None) -> bool:
        """Check if model has explicit configuration"""
        if model is None:
            model = self.model
        try:
            config_manager = get_config()
            provider_config = config_manager.get_provider(self.detected_provider)

            # Check if any model capability pattern matches this model
            for capability in provider_config.model_capabilities:
                if capability.matches(model):
                    return True

            # Check if model is in the explicit models list
            return model in provider_config.models

        except Exception:
            return False

    def supports_feature(self, feature_name: str) -> bool:
        """
        Enhanced feature support with smart defaults for unknown OpenAI models.
        ENHANCED: Now properly handles configuration fallback.
        """
        try:
            # First try the configuration system
            config_supports = super().supports_feature(feature_name)

            # If configuration gives a definitive answer, trust it
            if config_supports is not None:
                return config_supports

            # Configuration returned None (unknown model) - use our smart defaults
            if self.detected_provider == "openai":
                smart_features = self._get_smart_default_features(self.model)
                supports_smart = feature_name in smart_features

                if supports_smart:
                    log.info(
                        f"[{self.detected_provider}] No config for {self.model} - using smart default: supports {feature_name}"
                    )
                else:
                    log.debug(
                        f"[{self.detected_provider}] No config for {self.model} - smart default: doesn't support {feature_name}"
                    )

                return supports_smart

            # For non-OpenAI providers without config, be conservative
            log.warning(
                f"[{self.detected_provider}] No config for {self.model} - assuming doesn't support {feature_name}"
            )
            return False

        except Exception as e:
            log.warning(f"Feature support check failed for {feature_name}: {e}")

            # For OpenAI, be optimistic about unknown features
            if self.detected_provider == "openai":
                log.info(
                    f"[{self.detected_provider}] Error checking config - assuming {self.model} supports {feature_name} (optimistic fallback)"
                )
                return True

            return False

    # ================================================================
    # REASONING MODEL SUPPORT METHODS - ENHANCED WITH GPT-5
    # ================================================================

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model that needs special parameter handling"""
        model_lower = str(model_name).lower()

        # Check for O-series reasoning models (o1, o3, o4, o5)
        if any(pattern in model_lower for pattern in ["o1-", "o3-", "o4-", "o5-"]):
            return True

        # Check for official OpenAI GPT-5 models (not compatible models that just have gpt-5 in name)
        # Only match models that START with gpt-5 (like gpt-5, gpt-5-mini, gpt-5-nano)
        # NOT models that contain gpt-5 elsewhere (like global/gpt-5-chat)
        if model_lower.startswith("gpt-5"):
            return True

        return False

    def _get_reasoning_model_generation(self, model_name: str) -> str:
        """Get reasoning model generation (o1, o3, o4, o5, gpt5)"""
        model_lower = str(model_name).lower()
        if "o1" in model_lower:
            return "o1"
        elif "o3" in model_lower:
            return "o3"
        elif "o4" in model_lower:
            return "o4"
        elif "o5" in model_lower:
            return "o5"
        elif "gpt-5" in model_lower:
            return "gpt5"
        return "unknown"

    def _prepare_reasoning_model_parameters(self, **kwargs) -> dict[str, Any]:
        """
        Prepare parameters specifically for reasoning models.

        Key differences for reasoning models:
        - Use max_completion_tokens instead of max_tokens
        - Remove unsupported parameters like temperature, top_p
        - Handle streaming restrictions for o1
        - Handle GPT-5 specific restrictions
        """
        if not self._is_reasoning_model(self.model):
            return kwargs

        adjusted_kwargs = kwargs.copy()
        generation = self._get_reasoning_model_generation(self.model)

        # CRITICAL FIX: Replace max_tokens with max_completion_tokens
        if "max_tokens" in adjusted_kwargs:
            max_tokens_value = adjusted_kwargs.pop("max_tokens")
            adjusted_kwargs["max_completion_tokens"] = max_tokens_value
            log.debug(
                f"[{self.detected_provider}] Reasoning model parameter fix: "
                f"max_tokens -> max_completion_tokens ({max_tokens_value})"
            )

        # Add default max_completion_tokens if not specified
        if "max_completion_tokens" not in adjusted_kwargs:
            # Use reasonable defaults based on generation
            if generation == "gpt5":
                default_tokens = 128000  # GPT-5 has higher output limits
            elif generation in ["o3", "o4", "o5"]:
                default_tokens = 32768
            else:
                default_tokens = 16384
            adjusted_kwargs["max_completion_tokens"] = default_tokens
            log.debug(
                f"[{self.detected_provider}] Added default max_completion_tokens: {default_tokens}"
            )

        # Remove parameters not supported by reasoning models
        if generation == "o1":
            # O1 models have the most restrictions
            unsupported_params = [
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
            ]
        elif generation == "gpt5":
            # GPT-5 models have limited parameter restrictions (discovered: no temperature control)
            unsupported_params = [
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ]
        else:
            # O3/O4/O5 models have fewer restrictions
            unsupported_params = [
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ]

        removed_params = []
        for param in unsupported_params:
            if param in adjusted_kwargs:
                adjusted_kwargs.pop(param)
                removed_params.append(param)

        if removed_params:
            log.debug(
                f"[{self.detected_provider}] Removed unsupported reasoning model parameters: {removed_params}"
            )

        return adjusted_kwargs

    def _prepare_reasoning_model_messages(self, messages: list) -> list[dict[str, Any]]:
        """
        Prepare messages for reasoning models that may have restrictions.

        Args:
            messages: List of Pydantic Message objects OR dicts (already converted)

        Returns:
            List of dicts (for API boundary)

        O1 models don't support system messages - need to convert them.
        GPT-5 models support system messages.
        DeepSeek reasoner models require reasoning_content to be REMOVED from input messages
        (the API returns 400 if reasoning_content is present in input).
        """
        # Check if messages are already dicts (from _prepare_messages_for_conversation)
        if messages and isinstance(messages[0], dict):
            # Already dicts - apply provider-specific transformations
            result = messages

            # DeepSeek: MUST strip reasoning_content from input messages
            # API docs: "if reasoning_content is included in input messages, API returns 400 error"
            if self.detected_provider == "deepseek":
                result = self._strip_reasoning_content_from_messages(result)

            if not self._is_reasoning_model(self.model):
                return result

            generation = self._get_reasoning_model_generation(self.model)
            if generation == "o1":
                # Need to convert system messages, but we have dicts not Pydantic
                # Convert back to Pydantic temporarily
                from chuk_llm.llm.core.base import _ensure_pydantic_messages

                pydantic_messages = _ensure_pydantic_messages(result)
                return self._convert_system_messages_for_o1(pydantic_messages)
            return result

        # Messages are Pydantic objects - convert to dicts first
        dict_messages = [msg.to_dict() for msg in messages]

        # DeepSeek: MUST strip reasoning_content from input messages
        if self.detected_provider == "deepseek":
            dict_messages = self._strip_reasoning_content_from_messages(dict_messages)

        if not self._is_reasoning_model(self.model):
            return dict_messages

        generation = self._get_reasoning_model_generation(self.model)

        # Only O1 models don't support system messages
        if generation == "o1":
            return self._convert_system_messages_for_o1(messages)

        # GPT-5, O3, O4, O5 models support system messages
        return dict_messages

    def _strip_reasoning_content_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Strip reasoning_content from all messages for DeepSeek API compatibility.

        DeepSeek API docs explicitly state:
        "if the reasoning_content field is included in the sequence of input messages,
        the API will return a 400 error"

        Args:
            messages: List of message dicts

        Returns:
            List of message dicts with reasoning_content removed
        """
        result = []
        for msg in messages:
            if "reasoning_content" in msg:
                # Create a copy without reasoning_content
                cleaned_msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
                log.debug(
                    f"[{self.detected_provider}] Stripped reasoning_content from {msg.get('role', 'unknown')} message"
                )
                result.append(cleaned_msg)
            else:
                result.append(msg)
        return result

    def _convert_system_messages_for_o1(self, messages: list) -> list[dict[str, Any]]:
        """
        Convert system messages for O1 models that don't support them.

        Args:
            messages: List of Pydantic Message objects OR dicts

        Returns:
            List of dicts (for API boundary)
        """
        adjusted_messages = []
        system_instructions = []

        # Check if messages are dicts or Pydantic
        messages_are_dicts = messages and isinstance(messages[0], dict)

        for msg in messages:
            if messages_are_dicts:
                # Dict message
                if (
                    msg.get("role") == MessageRole.SYSTEM.value
                    or msg.get("role") == "system"
                ):
                    system_instructions.append(msg.get("content", ""))
                    log.debug(
                        f"[{self.detected_provider}] Converting system message for o1 model"
                    )
                else:
                    adjusted_messages.append(msg.copy())
            else:
                # Pydantic message
                if msg.role == MessageRole.SYSTEM:
                    system_instructions.append(msg.content or "")
                    log.debug(
                        f"[{self.detected_provider}] Converting system message for o1 model"
                    )
                else:
                    # Convert to dict for API
                    adjusted_messages.append(msg.to_dict())

        # If we have system instructions, prepend to first user message
        if system_instructions and adjusted_messages:
            first_user_idx = None
            for i, msg_dict in enumerate(adjusted_messages):
                # Already dict here
                role = msg_dict.get("role")
                if role == MessageRole.USER.value or role == "user":
                    first_user_idx = i
                    break

            if first_user_idx is not None:
                combined_instructions = "\n".join(system_instructions)
                original_content = adjusted_messages[first_user_idx]["content"]

                adjusted_messages[first_user_idx]["content"] = (
                    f"System Instructions: {combined_instructions}\n\n"
                    f"User Request: {original_content}"
                )

                log.debug(
                    f"[{self.detected_provider}] Merged system instructions into first user message"
                )

        return adjusted_messages

    # ================================================================
    # MODEL INFO AND CAPABILITIES
    # ================================================================

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model info using configuration, with OpenAI-specific additions and smart defaults.
        """
        # Get base info from configuration
        info = super().get_model_info()

        # Add tool compatibility info from universal system
        tool_compatibility = self.get_tool_compatibility_info()

        # Add reasoning model detection
        is_reasoning = self._is_reasoning_model(self.model)
        reasoning_generation = (
            self._get_reasoning_model_generation(self.model) if is_reasoning else None
        )

        # Check if using smart defaults
        using_smart_defaults = (
            self.detected_provider == "openai"
            and not self._has_explicit_model_config(self.model)
        )

        # Add OpenAI-specific metadata only if no error
        if not info.get("error"):
            info.update(
                {
                    "api_base": self.api_base,
                    "detected_provider": self.detected_provider,
                    "openai_compatible": True,
                    # Smart defaults info
                    "using_smart_defaults": using_smart_defaults,
                    "smart_default_features": (
                        list(self._get_smart_default_features(self.model))
                        if using_smart_defaults
                        else []
                    ),
                    # Reasoning model info
                    "is_reasoning_model": is_reasoning,
                    "reasoning_generation": reasoning_generation,
                    "requires_max_completion_tokens": is_reasoning,
                    "supports_streaming": True,  # All current models support streaming
                    "supports_system_messages": (
                        reasoning_generation != "o1" if is_reasoning else True
                    ),
                    # GPT-5 specific info
                    "is_gpt5_family": (
                        reasoning_generation == "gpt5" if is_reasoning else False
                    ),
                    "unified_reasoning": (
                        reasoning_generation == "gpt5" if is_reasoning else False
                    ),
                    # Universal tool compatibility info
                    **tool_compatibility,
                    "parameter_mapping": {
                        "temperature": (
                            "temperature" if reasoning_generation != "gpt5" else None
                        ),  # GPT-5 doesn't support temperature
                        "max_tokens": (
                            "max_completion_tokens" if is_reasoning else "max_tokens"
                        ),
                        "top_p": (
                            "top_p"
                            if reasoning_generation not in ["o1", "gpt5"]
                            else None
                        ),
                        "frequency_penalty": (
                            "frequency_penalty"
                            if reasoning_generation not in ["o1", "gpt5"]
                            else None
                        ),
                        "presence_penalty": (
                            "presence_penalty"
                            if reasoning_generation not in ["o1", "gpt5"]
                            else None
                        ),
                        "stop": "stop",
                        "stream": "stream",
                    },
                    "reasoning_model_restrictions": (
                        {
                            "unsupported_params": (
                                self._get_unsupported_params_for_generation(
                                    reasoning_generation or "o1"  # type: ignore[arg-type]
                                )
                                if is_reasoning
                                else []
                            ),
                            "requires_parameter_mapping": is_reasoning,
                            "system_message_conversion": (
                                reasoning_generation == "o1" if is_reasoning else False
                            ),
                            "temperature_fixed": (
                                reasoning_generation == "gpt5"
                                if is_reasoning
                                else False
                            ),
                        }
                        if is_reasoning
                        else {}
                    ),
                }
            )

        return info

    def _get_unsupported_params_for_generation(self, generation: str) -> list[str]:
        """Get unsupported parameters for a specific reasoning model generation"""
        if (
            generation == "o1"
            or generation == "gpt5"
            or generation in ["o3", "o4", "o5"]
        ):
            return ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
        else:
            return []

    def _normalize_message(self, msg) -> dict[str, Any]:
        """
        ENHANCED: Improved content extraction to eliminate warnings.
        Also extracts reasoning_content for DeepSeek reasoner models.
        """
        content = None
        reasoning_content = None
        tool_calls = []

        # Try multiple methods to extract content
        try:
            if hasattr(msg, "content"):
                content = msg.content
        except Exception as e:
            log.debug(f"Direct content access failed: {e}")

        # Try message wrapper
        if content is None:
            try:
                if hasattr(msg, "message") and hasattr(msg.message, "content"):
                    content = msg.message.content
            except Exception as e:
                log.debug(f"Message wrapper access failed: {e}")

        # Try dict access with Pydantic validation
        if content is None:
            try:
                if isinstance(msg, dict):
                    from chuk_llm.core import Message

                    # Try to validate as Pydantic Message
                    try:
                        validated_msg = Message.model_validate(msg)
                        content = validated_msg.content
                    except Exception:
                        # Fallback to direct dict access if validation fails
                        if "content" in msg:
                            content = msg["content"]
            except Exception as e:
                log.debug(f"Dict content access failed: {e}")

        # Extract reasoning_content for reasoning models (DeepSeek, etc.)
        try:
            if hasattr(msg, "reasoning_content"):
                reasoning_content = msg.reasoning_content
            elif hasattr(msg, "message") and hasattr(msg.message, "reasoning_content"):
                reasoning_content = msg.message.reasoning_content
            elif isinstance(msg, dict) and "reasoning_content" in msg:
                reasoning_content = msg["reasoning_content"]
        except Exception as e:
            log.debug(f"Reasoning content extraction failed: {e}")

        # Extract tool calls with enhanced error handling
        try:
            raw_tool_calls = None

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                raw_tool_calls = msg.tool_calls
            elif (
                hasattr(msg, "message")
                and hasattr(msg.message, "tool_calls")
                and msg.message.tool_calls
            ):
                raw_tool_calls = msg.message.tool_calls
            elif isinstance(msg, dict) and msg.get("tool_calls"):
                # Try Pydantic validation first
                from chuk_llm.core import Message

                try:
                    validated_msg = Message.model_validate(msg)
                    raw_tool_calls = validated_msg.tool_calls
                except Exception:
                    # Fallback to dict access
                    raw_tool_calls = msg["tool_calls"]

            if raw_tool_calls:
                for tc in raw_tool_calls:
                    try:
                        tc_id = (
                            getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                        )

                        if hasattr(tc, "function"):
                            func = tc.function
                            func_name = getattr(func, "name", "unknown_function")

                            # Handle arguments with robust JSON processing
                            args = getattr(func, "arguments", "{}")
                            try:
                                if isinstance(args, str):
                                    parsed_args = json.loads(args)
                                    args_j = json.dumps(parsed_args)
                                elif isinstance(args, dict):
                                    args_j = json.dumps(args)
                                else:
                                    args_j = "{}"
                            except json.JSONDecodeError:
                                args_j = "{}"

                            from chuk_llm.core.enums import ToolType

                            tool_calls.append(
                                {
                                    "id": tc_id,
                                    "type": ToolType.FUNCTION.value,
                                    "function": {
                                        "name": func_name,
                                        "arguments": args_j,
                                    },
                                }
                            )

                    except Exception as e:
                        log.warning(f"Failed to process tool call {tc}: {e}")
                        continue
        except Exception as e:
            log.warning(f"Failed to extract tool calls: {e}")

        # Set default content if None
        if content is None:
            content = ""

        # Determine response format
        if tool_calls:
            response_value = (
                content
                if content and isinstance(content, str) and content.strip()
                else None
            )
        else:
            response_value = content

        result = {"response": response_value, "tool_calls": tool_calls}

        # Include reasoning_content if present (for DeepSeek reasoner and similar models)
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
            log.debug(
                f"[{self.detected_provider}] Extracted reasoning_content: {len(str(reasoning_content))} chars"
            )

        return result

    def _prepare_messages_for_conversation(
        self, messages: list
    ) -> list[dict[str, Any]]:
        """
        CRITICAL FIX: Prepare messages for conversation by sanitizing tool names in message history.

        Args:
            messages: List of Pydantic Message objects

        Returns:
            List of dicts (for API boundary)

        This is the key fix for conversation flows - tool names in assistant messages
        must be sanitized to match what the API expects.
        """
        if not hasattr(self, "_current_name_mapping") or not self._current_name_mapping:
            # No name mapping needed - convert to dicts if Pydantic, otherwise return as-is
            if messages and not isinstance(messages[0], dict):
                return [msg.to_dict() for msg in messages]
            else:
                return messages

        prepared_messages = []

        # Check if messages are dicts or Pydantic
        messages_are_dicts = messages and isinstance(messages[0], dict)

        for msg in messages:
            if messages_are_dicts:
                # Dict message - check for tool_calls
                if (
                    msg.get("role") == MessageRole.ASSISTANT.value
                    or msg.get("role") == "assistant"
                ):
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        # Sanitize tool names in dict tool calls
                        prepared_msg = msg.copy()
                        sanitized_tool_calls = []

                        for tc in tool_calls:
                            tc_copy = tc.copy()
                            function = tc.get("function", {})
                            original_name = function.get("name", "")

                            # Find sanitized name from current mapping
                            sanitized_name = None
                            for (
                                sanitized,
                                original,
                            ) in self._current_name_mapping.items():
                                if original == original_name:
                                    sanitized_name = sanitized
                                    break

                            if sanitized_name:
                                tc_copy["function"] = function.copy()
                                tc_copy["function"]["name"] = sanitized_name
                                log.debug(
                                    f"Sanitized tool name in conversation: {original_name} -> {sanitized_name}"
                                )

                            sanitized_tool_calls.append(tc_copy)

                        prepared_msg["tool_calls"] = sanitized_tool_calls
                        prepared_messages.append(prepared_msg)
                    else:
                        prepared_messages.append(msg)
                else:
                    prepared_messages.append(msg)
            else:
                # Pydantic message
                if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                    # Sanitize tool names in assistant message tool calls
                    # Convert to dict first
                    prepared_msg = msg.to_dict()
                    sanitized_tool_calls = []

                    for tc in msg.tool_calls:
                        # tc is a Pydantic ToolCall object
                        original_name = tc.function.name

                        # Find sanitized name from current mapping
                        sanitized_name = None
                        for sanitized, original in self._current_name_mapping.items():
                            if original == original_name:
                                sanitized_name = sanitized
                                break

                        if sanitized_name:
                            # Create dict version with sanitized name
                            tc_dict = tc.to_dict()
                            tc_dict["function"]["name"] = sanitized_name
                            log.debug(
                                f"Sanitized tool name in conversation: {original_name} -> {sanitized_name}"
                            )
                            sanitized_tool_calls.append(tc_dict)
                        else:
                            sanitized_tool_calls.append(tc.to_dict())

                    prepared_msg["tool_calls"] = sanitized_tool_calls
                    prepared_messages.append(prepared_msg)
                else:
                    prepared_messages.append(msg.to_dict())

        return prepared_messages

    # ================================================================
    # STREAMING SUPPORT - FIXED: Proper JSON accumulation
    # ================================================================
    async def _stream_from_async(  # type: ignore[override]
        self,
        async_stream,
        name_mapping: dict[str, str] = None,
        normalize_chunk: callable = None,  # type: ignore[valid-type]
    ) -> AsyncIterator[dict[str, Any]]:
        """
        FIXED: Proper incremental tool call streaming with complete JSON handling.

        The key fix: Only yield tool calls when JSON arguments are complete and parseable.
        Stream content immediately, but accumulate tool call JSON until it's valid.
        """
        try:
            chunk_count = 0
            total_content_chars = 0
            total_reasoning_chars = 0

            # Track tool calls for incremental streaming - FIXED structure
            accumulated_tool_calls = {}  # {index: {id, name, arguments, complete}}
            accumulated_reasoning_content = (
                ""  # Accumulate reasoning_content for DeepSeek reasoner
            )

            async for chunk in async_stream:
                chunk_count += 1

                content_delta = ""  # Only new content
                reasoning_delta = ""  # Only new reasoning content
                completed_tool_calls = []  # Only complete, parseable tool calls

                try:
                    if (
                        hasattr(chunk, "choices")
                        and chunk.choices
                        and len(chunk.choices) > 0
                    ):
                        choice = chunk.choices[0]

                        if hasattr(choice, "delta") and choice.delta:
                            delta = choice.delta

                            # Handle regular content
                            if hasattr(delta, "content") and delta.content is not None:
                                content_delta = str(delta.content)
                                total_content_chars += len(content_delta)

                            # Handle reasoning_content separately (for DeepSeek reasoner, etc.)
                            if (
                                hasattr(delta, "reasoning_content")
                                and delta.reasoning_content is not None
                            ):
                                reasoning_delta = str(delta.reasoning_content)
                                accumulated_reasoning_content += reasoning_delta
                                total_reasoning_chars += len(reasoning_delta)

                            # Handle tool calls - FIXED: accumulate until complete
                            if hasattr(delta, "tool_calls") and delta.tool_calls:
                                for tc in delta.tool_calls:
                                    try:
                                        tc_index = getattr(tc, "index", 0)

                                        # Initialize or update accumulator
                                        if tc_index not in accumulated_tool_calls:
                                            accumulated_tool_calls[tc_index] = {
                                                "id": getattr(
                                                    tc,
                                                    "id",
                                                    f"call_{uuid.uuid4().hex[:8]}",
                                                ),
                                                "name": "",
                                                "arguments": "",
                                                "complete": False,
                                            }

                                        tool_call_data = accumulated_tool_calls[
                                            tc_index
                                        ]

                                        # Update ID if provided
                                        if hasattr(tc, "id") and tc.id:
                                            tool_call_data["id"] = tc.id

                                        # Update function data
                                        if hasattr(tc, "function") and tc.function:
                                            if (
                                                hasattr(tc.function, "name")
                                                and tc.function.name
                                            ):
                                                tool_call_data["name"] = (
                                                    tc.function.name
                                                )

                                            if (
                                                hasattr(tc.function, "arguments")
                                                and tc.function.arguments
                                            ):
                                                tool_call_data["arguments"] += (
                                                    tc.function.arguments
                                                )

                                        # CRITICAL FIX: Test if JSON is complete and valid
                                        if (
                                            tool_call_data["name"]
                                            and tool_call_data["arguments"]
                                        ):
                                            try:
                                                # Try to parse the accumulated JSON
                                                json.loads(
                                                    str(tool_call_data["arguments"])
                                                )  # type: ignore[arg-type]

                                                # If parsing succeeds, this tool call is complete
                                                if not tool_call_data["complete"]:
                                                    tool_call_data["complete"] = True

                                                    # Add to completed tool calls for this chunk
                                                    from chuk_llm.core.enums import (
                                                        ToolType,
                                                    )

                                                    completed_tool_calls.append(
                                                        {
                                                            "id": tool_call_data["id"],
                                                            "type": ToolType.FUNCTION.value,
                                                            "function": {
                                                                "name": tool_call_data[
                                                                    "name"
                                                                ],
                                                                "arguments": tool_call_data[
                                                                    "arguments"
                                                                ],
                                                            },
                                                        }
                                                    )

                                                    log.debug(
                                                        f"[{self.detected_provider}] Tool call {tc_index} complete: "
                                                        f"{tool_call_data['name']} with {len(str(tool_call_data['arguments']))} chars"  # type: ignore[arg-type]
                                                    )

                                            except json.JSONDecodeError:
                                                # JSON not complete yet, keep accumulating
                                                log.debug(
                                                    f"[{self.detected_provider}] Tool call {tc_index} JSON incomplete, "
                                                    f"args so far: {len(tool_call_data['arguments'])} chars"  # type: ignore[arg-type]
                                                )
                                                pass

                                    except Exception as e:
                                        log.debug(
                                            f"Error processing tool call delta: {e}"
                                        )
                                        continue

                except Exception as chunk_error:
                    log.warning(f"Error processing chunk {chunk_count}: {chunk_error}")
                    continue

                # FIXED: Yield if we have content OR reasoning OR completed tool calls
                # ALSO yield heartbeat chunks to prevent timeout (every 10 chunks)
                should_yield = (
                    content_delta
                    or reasoning_delta
                    or completed_tool_calls
                    or (chunk_count % 10 == 0)  # Heartbeat every 10 chunks
                )

                if should_yield:
                    result = {
                        "response": content_delta,
                        "tool_calls": (
                            completed_tool_calls if completed_tool_calls else None
                        ),
                    }

                    # Include reasoning_content when available (for DeepSeek reasoner, etc.)
                    # Send accumulated reasoning_content so the UI can show progress
                    if accumulated_reasoning_content:
                        result["reasoning_content"] = accumulated_reasoning_content
                        # Log reasoning progress periodically
                        if total_reasoning_chars % 1000 < len(reasoning_delta):
                            log.debug(
                                f"[{self.detected_provider}] Reasoning progress: "
                                f"{len(accumulated_reasoning_content)} total chars"
                            )

                    # For tool calls, reasoning_content is critical - must be sent back in next API call
                    if completed_tool_calls and accumulated_reasoning_content:
                        log.debug(
                            f"[{self.detected_provider}] Including reasoning_content "
                            f"({len(accumulated_reasoning_content)} chars) with tool calls"
                        )

                    # Restore tool names using universal restoration
                    if name_mapping and result.get("tool_calls"):
                        result = self._restore_tool_names_in_response(
                            result, name_mapping
                        )

                    yield result

            log.debug(
                f"[{self.detected_provider}] Streaming completed: {chunk_count} chunks, "
                f"{total_content_chars} content chars, {total_reasoning_chars} reasoning chars, "
                f"{len(accumulated_tool_calls)} tool calls"
            )

        except Exception as e:
            log.error(f"Error in {self.detected_provider} streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": None,
                "error": True,
            }

    # ================================================================
    # REQUEST VALIDATION AND PREPARATION WITH SMART DEFAULTS
    # ================================================================

    def _validate_request_with_config(
        self,
        messages: list,
        tools: list | None = None,
        stream: bool = False,
        **kwargs,
    ) -> tuple[list, list | None, bool, dict[str, Any]]:
        """
        Validate request against configuration before processing.
        ENHANCED: Uses smart defaults for newly discovered OpenAI models.

        Args:
            messages: List of Pydantic Message objects
            tools: List of Pydantic Tool objects or None

        Returns:
            Tuple of (messages, tools, stream, kwargs) - all validated
        """
        validated_messages = messages
        validated_tools = tools
        validated_stream = stream
        validated_kwargs = kwargs

        # Check streaming support (use smart defaults if needed)
        if stream and not self.supports_feature("streaming"):
            log.debug(
                f"Streaming requested but {self.detected_provider}/{self.model} doesn't support streaming - trying anyway"
            )
            # Don't disable streaming - let the API handle it

        # Check tool support (use smart defaults for unknown models)
        if tools:
            # CRITICAL: DeepSeek reasoner does NOT support function calling/tools
            # API docs: "Not Supported Features: Function Calling, FIM (Beta)"
            if (
                self.detected_provider == "deepseek"
                and "reasoner" in str(self.model).lower()
            ):
                log.warning(
                    f"[{self.detected_provider}] WARNING: {self.model} does NOT support function calling/tools. "
                    f"Tool calls may be hallucinated or fail. Consider using deepseek-chat instead."
                )
                # Still pass tools through - let the API handle it (may work partially or fail gracefully)

            if not self.supports_feature("tools"):
                log.debug(
                    f"Tools provided but {self.detected_provider}/{self.model} doesn't support tools according to config - trying anyway"
                )
                # Don't set validated_tools = None, let the API handle it
            elif (
                not self._has_explicit_model_config(self.model)
                and self.detected_provider == "openai"
            ):
                # Log when using smart defaults for tool support
                log.info(f"Using smart default: assuming {self.model} supports tools")

        # Permissive approach: Pass all content to API (vision, audio, etc.)
        # Let OpenAI API handle unsupported cases

        # Check JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            if not self.supports_feature("json_mode"):
                log.debug(
                    f"JSON mode requested but {self.detected_provider}/{self.model} doesn't support JSON mode - trying anyway"
                )
                validated_kwargs = {
                    k: v for k, v in kwargs.items() if k != "response_format"
                }

        return validated_messages, validated_tools, validated_stream, validated_kwargs

    # ================================================================
    # MAIN API METHODS
    # ================================================================

    def create_completion(
        self,
        messages: list,  # Pydantic Message objects or dicts (backward compat)
        tools: list | None = None,  # Pydantic Tool objects or dicts (backward compat)
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]] | Any:
        """
        ENHANCED: Now includes universal tool name compatibility with conversation flow handling,
        complete reasoning model support (including GPT-5) with FIXED streaming, and smart defaults for new models.

        Args:
            messages: List of Pydantic Message objects (or dicts for backward compatibility)
            tools: List of Pydantic Tool objects (or dicts for backward compatibility)
        """
        # Handle backward compatibility - convert dicts to Pydantic
        from chuk_llm.llm.core.base import (
            _ensure_pydantic_messages,
            _ensure_pydantic_tools,
        )

        pydantic_messages = _ensure_pydantic_messages(messages)
        pydantic_tools = _ensure_pydantic_tools(tools)

        # Validate request against configuration (with smart defaults)
        # Pass Pydantic objects - they'll be converted to dicts at API boundary
        validated_messages, validated_tools, validated_stream, validated_kwargs = (
            self._validate_request_with_config(
                pydantic_messages, pydantic_tools, stream, **kwargs
            )
        )

        # Convert tools to dicts for sanitization (tool name sanitization operates on dicts)
        dict_tools = (
            [tool.to_dict() for tool in validated_tools] if validated_tools else None
        )

        # Apply universal tool name sanitization
        name_mapping = {}
        if dict_tools:
            dict_tools = self._sanitize_tool_names(dict_tools)
            name_mapping = self._current_name_mapping
            log.debug(
                f"Tool sanitization: {len(name_mapping)} tools processed for {self.detected_provider} compatibility"
            )

            # Add strict parameter for OpenAI-compatible APIs that may require it
            if (
                self.detected_provider == "openai_compatible" and dict_tools
            ):  # Legacy code - TODO: migrate to Provider enum
                dict_tools = self._add_strict_parameter_to_tools(dict_tools)

        # JSON Function Calling Fallback (if enabled)
        fallback_context = {}
        if self.ENABLE_JSON_FUNCTION_FALLBACK:
            # Convert messages to dicts for processing
            messages_dicts = [msg.model_dump(mode="json") for msg in validated_messages]

            # Convert incompatible message roles
            messages_dicts, has_tool_results = self._convert_incompatible_message_roles(
                messages_dicts
            )

            # Inject function calling prompt if tools provided AND no tool results
            if dict_tools and not has_tool_results:
                prompt = self._create_json_function_calling_prompt(dict_tools)

                # Prepend to first system message or add new one
                if (
                    messages_dicts
                    and messages_dicts[0].get("role") == MessageRole.SYSTEM.value
                ):
                    messages_dicts[0]["content"] = (
                        f"{prompt}\n\n{messages_dicts[0]['content']}"
                    )
                else:
                    messages_dicts.insert(
                        0, {"role": MessageRole.SYSTEM.value, "content": prompt}
                    )

                # Don't pass tools to API (it doesn't support them natively)
                dict_tools = None
                fallback_context["should_convert"] = True
                log.debug(
                    "[json_fallback] Enabled: injected prompt, will parse JSON from response"
                )
            else:
                fallback_context["should_convert"] = False

            # Convert back to Pydantic
            from chuk_llm.llm.core.base import _ensure_pydantic_messages

            validated_messages = _ensure_pydantic_messages(messages_dicts)

        # Prepare messages for conversation (sanitize tool names in history)
        # Pass Pydantic messages, this method converts to dicts at API boundary
        if name_mapping:
            validated_messages = self._prepare_messages_for_conversation(
                validated_messages
            )

        # Use configuration-aware parameter adjustment
        validated_kwargs = self.validate_parameters(**validated_kwargs)

        if validated_stream:
            return self._stream_completion_async(
                validated_messages, dict_tools, name_mapping, **validated_kwargs
            )
        else:
            # Store fallback context for post-processing
            self._fallback_context = fallback_context
            return self._regular_completion(
                validated_messages, dict_tools, name_mapping, **validated_kwargs
            )

    async def _stream_completion_async(
        self,
        messages: list,
        tools: list | None = None,
        name_mapping: dict[str, str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Enhanced async streaming with reasoning model support (including GPT-5) and FIXED streaming logic.

        Args:
            messages: List of Pydantic Message objects OR dicts (if already converted by _prepare_messages_for_conversation)
            tools: List of dicts (tool definitions for API)
            name_mapping: Tool name mapping for restoration
            **kwargs: Additional parameters

        Returns:
            Async iterator of response dicts
        """
        max_retries = 1

        for attempt in range(max_retries + 1):
            try:
                # Prepare messages and parameters for reasoning models (including GPT-5)
                # _prepare_reasoning_model_messages converts Pydantic to dicts for API
                prepared_messages = self._prepare_reasoning_model_messages(messages)
                prepared_kwargs = self._prepare_reasoning_model_parameters(**kwargs)

                # Tools are already dicts (converted in create_completion)
                prepared_tools = tools

                # Download image URLs for non-OpenAI providers (like llama.cpp)
                # OpenAI supports direct URLs, but local servers may not
                supports_urls = self.detected_provider in [
                    "openai",
                    "anthropic",
                    "google",
                ]
                prepared_messages = await self._process_image_urls_in_messages(
                    prepared_messages, supports_direct_urls=supports_urls
                )

                log.debug(
                    f"[{self.detected_provider}] Starting streaming (attempt {attempt + 1}): "
                    f"model={self.model}, messages={len(prepared_messages)}, "
                    f"tools={len(prepared_tools) if prepared_tools else 0}, "
                    f"reasoning_model={self._is_reasoning_model(self.model)}, "
                    f"generation={self._get_reasoning_model_generation(self.model)}"
                )

                # Log reasoning model adjustments
                if self._is_reasoning_model(self.model):
                    param_changes = []
                    if "max_completion_tokens" in prepared_kwargs:
                        param_changes.append(
                            f"max_completion_tokens={prepared_kwargs['max_completion_tokens']}"
                        )

                    generation = self._get_reasoning_model_generation(self.model)
                    if generation == "gpt5":
                        param_changes.append("GPT-5 family (unified reasoning)")

                    if param_changes:
                        log.debug(
                            f"[{self.detected_provider}] Reasoning model adjustments: {', '.join(param_changes)}"
                        )

                response_stream = await self.client.chat.completions.create(  # type: ignore[call-overload]
                    model=self.model,
                    messages=prepared_messages,
                    **({"tools": prepared_tools} if prepared_tools else {}),
                    stream=True,
                    **prepared_kwargs,
                )

                chunk_count = 0
                async for result in self._stream_from_async(
                    response_stream, name_mapping
                ):
                    chunk_count += 1
                    yield result

                log.debug(
                    f"[{self.detected_provider}] Streaming completed successfully with {chunk_count} chunks"
                )
                return

            except Exception as e:
                error_str = str(e).lower()

                # Check for reasoning model parameter errors (including GPT-5)
                if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                    log.error(
                        f"[{self.detected_provider}] CRITICAL: Reasoning model parameter error not handled: {e}"
                    )
                elif "temperature" in error_str and "gpt-5" in self.model.lower():
                    log.error(
                        f"[{self.detected_provider}] GPT-5 temperature restriction not handled: {e}"
                    )

                is_retryable = any(
                    pattern in error_str
                    for pattern in [
                        "timeout",
                        "connection",
                        "network",
                        "temporary",
                        "rate limit",
                    ]
                )

                if attempt < max_retries and is_retryable:
                    wait_time = (attempt + 1) * 1.0
                    log.warning(
                        f"[{self.detected_provider}] Streaming attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    log.error(
                        f"[{self.detected_provider}] Streaming failed after {attempt + 1} attempts: {e}"
                    )
                    yield {
                        "response": f"Error: {str(e)}",
                        "tool_calls": None,
                        "error": True,
                    }
                    return

    async def _regular_completion(
        self,
        messages: list,
        tools: list | None = None,
        name_mapping: dict[str, str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Enhanced non-streaming completion with reasoning model support (including GPT-5) and universal tool name restoration.

        Args:
            messages: List of Pydantic Message objects OR dicts (if already converted by _prepare_messages_for_conversation)
            tools: List of dicts (tool definitions for API)
            name_mapping: Tool name mapping for restoration
            **kwargs: Additional parameters

        Returns:
            Response dict with response, tool_calls, usage, etc.
        """
        try:
            # Prepare messages and parameters for reasoning models (including GPT-5)
            # _prepare_reasoning_model_messages converts Pydantic to dicts for API
            prepared_messages = self._prepare_reasoning_model_messages(messages)
            prepared_kwargs = self._prepare_reasoning_model_parameters(**kwargs)

            # Tools are already dicts (converted in create_completion)
            prepared_tools = tools

            # Download image URLs for non-OpenAI providers (like llama.cpp)
            # OpenAI supports direct URLs, but local servers may not
            supports_urls = self.detected_provider in ["openai", "anthropic", "google"]
            prepared_messages = await self._process_image_urls_in_messages(
                prepared_messages, supports_direct_urls=supports_urls
            )

            log.debug(
                f"[{self.detected_provider}] Starting completion: "
                f"model={self.model}, messages={len(prepared_messages)}, "
                f"tools={len(prepared_tools) if prepared_tools else 0}, "
                f"reasoning_model={self._is_reasoning_model(self.model)}, "
                f"generation={self._get_reasoning_model_generation(self.model)}"
            )

            # Log reasoning model adjustments for debugging
            if self._is_reasoning_model(self.model):
                param_changes = []
                if "max_completion_tokens" in prepared_kwargs:
                    param_changes.append(
                        f"max_completion_tokens={prepared_kwargs['max_completion_tokens']}"
                    )

                generation = self._get_reasoning_model_generation(self.model)
                if generation == "gpt5":
                    param_changes.append("GPT-5 family (unified reasoning)")

                if param_changes:
                    log.debug(
                        f"[{self.detected_provider}] Reasoning model adjustments: {', '.join(param_changes)}"
                    )

            resp = await self.client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=prepared_messages,
                **({"tools": prepared_tools} if prepared_tools else {}),
                stream=False,
                **prepared_kwargs,
            )

            result = self._normalize_message(resp.choices[0].message)

            # Extract usage information (including reasoning_tokens for reasoning models)
            if hasattr(resp, "usage") and resp.usage:
                usage_info = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                    "total_tokens": getattr(resp.usage, "total_tokens", 0),
                }

                # Add reasoning tokens if present (for o1/gpt-5 models)
                if hasattr(resp.usage, "completion_tokens_details"):
                    details = resp.usage.completion_tokens_details
                    if (
                        hasattr(details, "reasoning_tokens")
                        and details.reasoning_tokens
                    ):
                        usage_info["reasoning_tokens"] = details.reasoning_tokens
                        # Add reasoning metadata to response
                        result["reasoning"] = {
                            "thinking_tokens": details.reasoning_tokens,
                            "model_type": "reasoning",
                        }

                result["usage"] = usage_info

            # Restore original tool names using universal restoration
            if name_mapping and result.get("tool_calls"):
                result = self._restore_tool_names_in_response(result, name_mapping)

            # JSON Function Calling Fallback: Convert JSON in content to tool_calls
            if (
                hasattr(self, "_fallback_context")
                and self._fallback_context
                and self._fallback_context.get("should_convert")
            ):
                result = self._convert_to_tool_calls_from_json(result)
                # Clean up
                delattr(self, "_fallback_context")

            log.debug(
                f"[{self.detected_provider}] Completion successful: "
                f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                f"tool_calls={len(result.get('tool_calls', []))}, "
                f"reasoning_tokens={result.get('reasoning', {}).get('thinking_tokens', 0) if result.get('reasoning') else 0}"
            )

            return result

        except Exception as e:
            error_msg = str(e)
            log.error(f"[{self.detected_provider}] Error in completion: {e}")

            # Provide helpful error messages for common reasoning model issues
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                log.error(
                    f"[{self.detected_provider}] REASONING MODEL PARAMETER ERROR: "
                    f"This appears to be a reasoning model that requires max_completion_tokens. "
                    f"The parameter conversion should have handled this automatically."
                )
            elif "temperature" in error_msg and "gpt-5" in self.model.lower():
                log.error(
                    f"[{self.detected_provider}] GPT-5 TEMPERATURE RESTRICTION: "
                    f"GPT-5 models only support default temperature (1.0). "
                    f"The parameter filtering should have handled this automatically."
                )

            return {"response": f"Error: {str(e)}", "tool_calls": [], "error": True}

    async def close(self):
        """Cleanup resources and invalidate client cache."""
        # Invalidate cache (handled by base class)
        await super().close()

        # Reset name mapping from universal system
        if hasattr(self, "_current_name_mapping"):
            self._current_name_mapping = {}

        # Close the underlying OpenAI client
        if hasattr(self.client, "close"):
            await self.client.close()
