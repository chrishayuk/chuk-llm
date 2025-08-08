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
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple, Set
import openai
import logging
import asyncio
import time
import re
import uuid
import json

# mixins
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin
from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

# base
from chuk_llm.llm.core.base import BaseLLMClient

log = logging.getLogger(__name__)

class OpenAILLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-driven wrapper around the official `openai` SDK that gets
    all capabilities from the unified YAML configuration.
    
    COMPLETE VERSION: Now includes GPT-5 family support, reasoning model support, 
    FIXED streaming, and smart defaults.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Detect provider from api_base for configuration lookup
        detected_provider = self._detect_provider_name(api_base)
        
        # Initialize ALL mixins including ToolCompatibilityMixin
        ConfigAwareProviderMixin.__init__(self, detected_provider, model)
        ToolCompatibilityMixin.__init__(self, detected_provider)
        
        self.model = model
        self.api_base = api_base
        self.detected_provider = detected_provider
        
        # Use AsyncOpenAI for real streaming support
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        # Keep sync client for backwards compatibility if needed
        self.client = openai.OpenAI(
            api_key=api_key, 
            base_url=api_base
        ) if api_base else openai.OpenAI(api_key=api_key)
        
        log.debug(f"OpenAI client initialized: provider={self.detected_provider}, model={self.model}")

    def _detect_provider_name(self, api_base: Optional[str]) -> str:
        """Detect provider name from API base URL for configuration lookup"""
        if not api_base:
            return "openai"
        
        api_base_lower = api_base.lower()
        if "deepseek" in api_base_lower:
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
    # SMART DEFAULTS FOR NEW OPENAI MODELS
    # ================================================================
    
    @staticmethod
    def _get_smart_default_features(model_name: str) -> Set[str]:
        """Get smart default features for an OpenAI model based on naming patterns"""
        model_lower = model_name.lower()
        
        # Base features that ALL modern OpenAI models should have
        base_features = {"text", "streaming", "system_messages"}
        
        # Pattern-based feature detection
        if any(pattern in model_lower for pattern in ['o1', 'o3', 'o4', 'o5', 'o6']):
            # Reasoning models
            if 'o1' in model_lower:
                # O1 models are legacy - no tools, limited capabilities
                return {"text", "reasoning"}
            else:
                # O3+ models are modern reasoning - assume full tool support
                return {"text", "streaming", "tools", "reasoning", "system_messages"}
        
        elif any(pattern in model_lower for pattern in ['gpt-4', 'gpt-3.5', 'gpt-5']):
            # Standard GPT models - assume modern capabilities
            features = base_features | {"tools", "json_mode"}
            
            # Vision support for GPT-4+ models (not 3.5)
            if any(v in model_lower for v in ['gpt-4', 'gpt-5']):
                features.add("vision")
                
            # GPT-5 models use reasoning architecture
            if 'gpt-5' in model_lower:
                features.add("reasoning")
                
            return features
        
        else:
            # Unknown OpenAI model patterns - be optimistic about tool support
            # KEY PRINCIPLE: Assume new OpenAI models support tools by default
            log.info(f"Unknown OpenAI model pattern '{model_name}' - assuming modern capabilities including tools")
            return base_features | {"tools", "json_mode"}
    
    @staticmethod
    def _get_smart_default_parameters(model_name: str) -> Dict[str, Any]:
        """Get smart default parameters for an OpenAI model"""
        model_lower = model_name.lower()
        
        # Reasoning model parameter handling
        if any(pattern in model_lower for pattern in ['o1', 'o3', 'o4', 'o5', 'gpt-5']):
            return {
                "max_context_length": 272000 if 'gpt-5' in model_lower else 200000,
                "max_output_tokens": 128000 if 'gpt-5' in model_lower else 65536,
                "requires_max_completion_tokens": True,
                "parameter_mapping": {
                    "max_tokens": "max_completion_tokens"
                },
                "unsupported_params": ["temperature", "top_p", "frequency_penalty", "presence_penalty"],
                "supports_tools": "o1" not in model_lower  # Only O1 doesn't support tools
            }
        
        # Standard model defaults - generous assumptions for new models
        return {
            "max_context_length": 128000,
            "max_output_tokens": 8192,
            "supports_tools": True  # Assume new models support tools
        }
    
    def _has_explicit_model_config(self, model: str) -> bool:
        """Check if model has explicit configuration"""
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
                    log.info(f"[{self.detected_provider}] No config for {self.model} - using smart default: supports {feature_name}")
                else:
                    log.debug(f"[{self.detected_provider}] No config for {self.model} - smart default: doesn't support {feature_name}")
                
                return supports_smart
            
            # For non-OpenAI providers without config, be conservative
            log.warning(f"[{self.detected_provider}] No config for {self.model} - assuming doesn't support {feature_name}")
            return False
            
        except Exception as e:
            log.warning(f"Feature support check failed for {feature_name}: {e}")
            
            # For OpenAI, be optimistic about unknown features
            if self.detected_provider == "openai":
                log.info(f"[{self.detected_provider}] Error checking config - assuming {self.model} supports {feature_name} (optimistic fallback)")
                return True
            
            return False
        
    # ================================================================
    # REASONING MODEL SUPPORT METHODS - ENHANCED WITH GPT-5
    # ================================================================
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model that needs special parameter handling"""
        return any(pattern in model_name.lower() for pattern in [
            'o1', 'o3', 'o4', 'o5',  # O-series reasoning models
            'gpt-5'  # GPT-5 family also uses reasoning architecture
        ])

    def _get_reasoning_model_generation(self, model_name: str) -> str:
        """Get reasoning model generation (o1, o3, o4, o5, gpt5)"""
        model_lower = model_name.lower()
        if 'o1' in model_lower:
            return 'o1'
        elif 'o3' in model_lower:
            return 'o3'
        elif 'o4' in model_lower:
            return 'o4'
        elif 'o5' in model_lower:
            return 'o5'
        elif 'gpt-5' in model_lower:
            return 'gpt5'
        return 'unknown'

    def _prepare_reasoning_model_parameters(self, **kwargs) -> Dict[str, Any]:
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
        if 'max_tokens' in adjusted_kwargs:
            max_tokens_value = adjusted_kwargs.pop('max_tokens')
            adjusted_kwargs['max_completion_tokens'] = max_tokens_value
            log.debug(f"[{self.detected_provider}] Reasoning model parameter fix: "
                     f"max_tokens -> max_completion_tokens ({max_tokens_value})")
        
        # Add default max_completion_tokens if not specified
        if 'max_completion_tokens' not in adjusted_kwargs:
            # Use reasonable defaults based on generation
            if generation == 'gpt5':
                default_tokens = 128000  # GPT-5 has higher output limits
            elif generation in ['o3', 'o4', 'o5']:
                default_tokens = 32768
            else:
                default_tokens = 16384
            adjusted_kwargs['max_completion_tokens'] = default_tokens
            log.debug(f"[{self.detected_provider}] Added default max_completion_tokens: {default_tokens}")
        
        # Remove parameters not supported by reasoning models
        if generation == 'o1':
            # O1 models have the most restrictions
            unsupported_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'logit_bias']
        elif generation == 'gpt5':
            # GPT-5 models have limited parameter restrictions (discovered: no temperature control)
            unsupported_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
        else:
            # O3/O4/O5 models have fewer restrictions
            unsupported_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
        
        removed_params = []
        for param in unsupported_params:
            if param in adjusted_kwargs:
                adjusted_kwargs.pop(param)
                removed_params.append(param)
        
        if removed_params:
            log.debug(f"[{self.detected_provider}] Removed unsupported reasoning model parameters: {removed_params}")
        
        return adjusted_kwargs

    def _prepare_reasoning_model_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare messages for reasoning models that may have restrictions.
        
        O1 models don't support system messages - need to convert them.
        GPT-5 models support system messages.
        """
        if not self._is_reasoning_model(self.model):
            return messages
        
        generation = self._get_reasoning_model_generation(self.model)
        
        # Only O1 models don't support system messages
        if generation == 'o1':
            return self._convert_system_messages_for_o1(messages)
        
        # GPT-5, O3, O4, O5 models support system messages
        return messages

    def _convert_system_messages_for_o1(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert system messages for O1 models that don't support them"""
        
        adjusted_messages = []
        system_instructions = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_instructions.append(msg['content'])
                log.debug(f"[{self.detected_provider}] Converting system message for o1 model")
            else:
                adjusted_messages.append(msg.copy())
        
        # If we have system instructions, prepend to first user message
        if system_instructions and adjusted_messages:
            first_user_idx = None
            for i, msg in enumerate(adjusted_messages):
                if msg.get('role') == 'user':
                    first_user_idx = i
                    break
            
            if first_user_idx is not None:
                combined_instructions = '\n'.join(system_instructions)
                original_content = adjusted_messages[first_user_idx]['content']
                
                adjusted_messages[first_user_idx]['content'] = (
                    f"System Instructions: {combined_instructions}\n\n"
                    f"User Request: {original_content}"
                )
                
                log.debug(f"[{self.detected_provider}] Merged system instructions into first user message")
        
        return adjusted_messages

    # ================================================================
    # MODEL INFO AND CAPABILITIES
    # ================================================================

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with OpenAI-specific additions and smart defaults.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add tool compatibility info from universal system
        tool_compatibility = self.get_tool_compatibility_info()
        
        # Add reasoning model detection
        is_reasoning = self._is_reasoning_model(self.model)
        reasoning_generation = self._get_reasoning_model_generation(self.model) if is_reasoning else None
        
        # Check if using smart defaults
        using_smart_defaults = (self.detected_provider == "openai" and 
                               not self._has_explicit_model_config(self.model))
        
        # Add OpenAI-specific metadata only if no error
        if not info.get("error"):
            info.update({
                "api_base": self.api_base,
                "detected_provider": self.detected_provider,
                "openai_compatible": True,
                
                # Smart defaults info
                "using_smart_defaults": using_smart_defaults,
                "smart_default_features": list(self._get_smart_default_features(self.model)) if using_smart_defaults else [],
                
                # Reasoning model info
                "is_reasoning_model": is_reasoning,
                "reasoning_generation": reasoning_generation,
                "requires_max_completion_tokens": is_reasoning,
                "supports_streaming": True,  # All current models support streaming
                "supports_system_messages": not (reasoning_generation == 'o1') if is_reasoning else True,
                
                # GPT-5 specific info
                "is_gpt5_family": reasoning_generation == 'gpt5' if is_reasoning else False,
                "unified_reasoning": reasoning_generation == 'gpt5' if is_reasoning else False,
                
                # Universal tool compatibility info
                **tool_compatibility,
                
                "parameter_mapping": {
                    "temperature": "temperature" if reasoning_generation != 'gpt5' else None,  # GPT-5 doesn't support temperature
                    "max_tokens": "max_completion_tokens" if is_reasoning else "max_tokens",
                    "top_p": "top_p" if reasoning_generation not in ['o1', 'gpt5'] else None,
                    "frequency_penalty": "frequency_penalty" if reasoning_generation not in ['o1', 'gpt5'] else None,
                    "presence_penalty": "presence_penalty" if reasoning_generation not in ['o1', 'gpt5'] else None,
                    "stop": "stop",
                    "stream": "stream"
                },
                
                "reasoning_model_restrictions": {
                    "unsupported_params": self._get_unsupported_params_for_generation(reasoning_generation) if is_reasoning else [],
                    "requires_parameter_mapping": is_reasoning,
                    "system_message_conversion": reasoning_generation == 'o1' if is_reasoning else False,
                    "temperature_fixed": reasoning_generation == 'gpt5' if is_reasoning else False,
                } if is_reasoning else {}
            })
        
        return info

    def _get_unsupported_params_for_generation(self, generation: str) -> List[str]:
        """Get unsupported parameters for a specific reasoning model generation"""
        if generation == 'o1':
            return ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
        elif generation == 'gpt5':
            return ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
        elif generation in ['o3', 'o4', 'o5']:
            return ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
        else:
            return []

    def _normalize_message(self, msg) -> Dict[str, Any]:
        """
        ENHANCED: Improved content extraction to eliminate warnings.
        """
        content = None
        tool_calls = []
        
        # Try multiple methods to extract content
        try:
            if hasattr(msg, 'content'):
                content = msg.content
        except Exception as e:
            log.debug(f"Direct content access failed: {e}")
        
        # Try message wrapper
        if content is None:
            try:
                if hasattr(msg, 'message') and hasattr(msg.message, 'content'):
                    content = msg.message.content
            except Exception as e:
                log.debug(f"Message wrapper access failed: {e}")
        
        # Try dict access
        if content is None:
            try:
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
            except Exception as e:
                log.debug(f"Dict content access failed: {e}")
        
        # Extract tool calls with enhanced error handling
        try:
            raw_tool_calls = None
            
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                raw_tool_calls = msg.tool_calls
            elif hasattr(msg, 'message') and hasattr(msg.message, 'tool_calls') and msg.message.tool_calls:
                raw_tool_calls = msg.message.tool_calls
            elif isinstance(msg, dict) and msg.get('tool_calls'):
                raw_tool_calls = msg['tool_calls']
            
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    try:
                        tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                        
                        if hasattr(tc, 'function'):
                            func = tc.function
                            func_name = getattr(func, 'name', 'unknown_function')
                            
                            # Handle arguments with robust JSON processing
                            args = getattr(func, 'arguments', '{}')
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
                            
                            tool_calls.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": args_j,
                                },
                            })
                        
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
            response_value = content if content and content.strip() else None
        else:
            response_value = content
        
        result = {
            "response": response_value,
            "tool_calls": tool_calls
        }
        
        return result

    def _prepare_messages_for_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CRITICAL FIX: Prepare messages for conversation by sanitizing tool names in message history.
        
        This is the key fix for conversation flows - tool names in assistant messages
        must be sanitized to match what the API expects.
        """
        if not hasattr(self, '_current_name_mapping') or not self._current_name_mapping:
            return messages
        
        prepared_messages = []
        
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Sanitize tool names in assistant message tool calls
                prepared_msg = msg.copy()
                sanitized_tool_calls = []
                
                for tc in msg["tool_calls"]:
                    tc_copy = tc.copy()
                    original_name = tc["function"]["name"]
                    
                    # Find sanitized name from current mapping
                    sanitized_name = None
                    for sanitized, original in self._current_name_mapping.items():
                        if original == original_name:
                            sanitized_name = sanitized
                            break
                    
                    if sanitized_name:
                        tc_copy["function"] = tc["function"].copy()
                        tc_copy["function"]["name"] = sanitized_name
                        log.debug(f"Sanitized tool name in conversation: {original_name} -> {sanitized_name}")
                    
                    sanitized_tool_calls.append(tc_copy)
                
                prepared_msg["tool_calls"] = sanitized_tool_calls
                prepared_messages.append(prepared_msg)
            else:
                prepared_messages.append(msg)
        
        return prepared_messages

    # ================================================================
    # STREAMING SUPPORT - FIXED: Proper JSON accumulation
    # ================================================================
    async def _stream_from_async(
        self,
        async_stream,
        name_mapping: Dict[str, str] = None,
        normalize_chunk: Optional[callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        FIXED: Proper incremental tool call streaming with complete JSON handling.
        
        The key fix: Only yield tool calls when JSON arguments are complete and parseable.
        Stream content immediately, but accumulate tool call JSON until it's valid.
        """
        try:
            chunk_count = 0
            total_content_chars = 0
            
            # Track tool calls for incremental streaming - FIXED structure
            accumulated_tool_calls = {}  # {index: {id, name, arguments, complete}}
            
            async for chunk in async_stream:
                chunk_count += 1
                
                content_delta = ""  # Only new content
                completed_tool_calls = []  # Only complete, parseable tool calls
                
                try:
                    if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            
                            # Handle content - yield immediately (this works fine)
                            if hasattr(delta, 'content') and delta.content is not None:
                                content_delta = str(delta.content)
                                total_content_chars += len(content_delta)
                            
                            # Handle tool calls - FIXED: accumulate until complete
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                for tc in delta.tool_calls:
                                    try:
                                        tc_index = getattr(tc, 'index', 0)
                                        
                                        # Initialize or update accumulator
                                        if tc_index not in accumulated_tool_calls:
                                            accumulated_tool_calls[tc_index] = {
                                                "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                                "name": "",
                                                "arguments": "",
                                                "complete": False
                                            }
                                        
                                        tool_call_data = accumulated_tool_calls[tc_index]
                                        
                                        # Update ID if provided
                                        if hasattr(tc, 'id') and tc.id:
                                            tool_call_data["id"] = tc.id
                                        
                                        # Update function data
                                        if hasattr(tc, 'function') and tc.function:
                                            if hasattr(tc.function, 'name') and tc.function.name:
                                                tool_call_data["name"] = tc.function.name
                                            
                                            if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                                tool_call_data["arguments"] += tc.function.arguments
                                        
                                        # CRITICAL FIX: Test if JSON is complete and valid
                                        if tool_call_data["name"] and tool_call_data["arguments"]:
                                            try:
                                                # Try to parse the accumulated JSON
                                                json.loads(tool_call_data["arguments"])
                                                
                                                # If parsing succeeds, this tool call is complete
                                                if not tool_call_data["complete"]:
                                                    tool_call_data["complete"] = True
                                                    
                                                    # Add to completed tool calls for this chunk
                                                    completed_tool_calls.append({
                                                        "id": tool_call_data["id"],
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_call_data["name"],
                                                            "arguments": tool_call_data["arguments"]
                                                        }
                                                    })
                                                    
                                                    log.debug(f"[{self.detected_provider}] Tool call {tc_index} complete: "
                                                            f"{tool_call_data['name']} with {len(tool_call_data['arguments'])} chars")
                                            
                                            except json.JSONDecodeError:
                                                # JSON not complete yet, keep accumulating
                                                log.debug(f"[{self.detected_provider}] Tool call {tc_index} JSON incomplete, "
                                                        f"args so far: {len(tool_call_data['arguments'])} chars")
                                                pass
                                    
                                    except Exception as e:
                                        log.debug(f"Error processing tool call delta: {e}")
                                        continue
                
                except Exception as chunk_error:
                    log.warning(f"Error processing chunk {chunk_count}: {chunk_error}")
                    continue
                
                # FIXED: Yield if we have content OR completed tool calls
                if content_delta or completed_tool_calls:
                    result = {
                        "response": content_delta,
                        "tool_calls": completed_tool_calls if completed_tool_calls else None,
                    }
                    
                    # Restore tool names using universal restoration
                    if name_mapping and result.get("tool_calls"):
                        result = self._restore_tool_names_in_response(result, name_mapping)
                    
                    yield result
            
            log.debug(f"[{self.detected_provider}] Streaming completed: {chunk_count} chunks, "
                    f"{total_content_chars} total characters, {len(accumulated_tool_calls)} tool calls")
                        
        except Exception as e:
            log.error(f"Error in {self.detected_provider} streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": None,
                "error": True
            }

    # ================================================================
    # REQUEST VALIDATION AND PREPARATION WITH SMART DEFAULTS
    # ================================================================
            
    def _validate_request_with_config(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], bool, Dict[str, Any]]:
        """
        Validate request against configuration before processing.
        ENHANCED: Uses smart defaults for newly discovered OpenAI models.
        """
        validated_messages = messages
        validated_tools = tools
        validated_stream = stream
        validated_kwargs = kwargs
        
        # Check streaming support (use smart defaults if needed)
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but {self.detected_provider}/{self.model} doesn't support streaming")
            # Don't disable streaming - let the API handle it
        
        # Check tool support (use smart defaults for unknown models)
        if tools:
            if not self.supports_feature("tools"):
                log.warning(f"Tools provided but {self.detected_provider}/{self.model} doesn't support tools")
                validated_tools = None
            elif not self._has_explicit_model_config(self.model) and self.detected_provider == "openai":
                # Log when using smart defaults for tool support
                log.info(f"Using smart default: assuming {self.model} supports tools")
        
        # Check vision support
        has_vision = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.warning(f"Vision content detected but {self.detected_provider}/{self.model} doesn't support vision")
        
        # Check JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            if not self.supports_feature("json_mode"):
                log.warning(f"JSON mode requested but {self.detected_provider}/{self.model} doesn't support JSON mode")
                validated_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    # ================================================================
    # MAIN API METHODS
    # ================================================================

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        ENHANCED: Now includes universal tool name compatibility with conversation flow handling,
        complete reasoning model support (including GPT-5) with FIXED streaming, and smart defaults for new models.
        """
        # Validate request against configuration (with smart defaults)
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Apply universal tool name sanitization 
        name_mapping = {}
        if validated_tools:
            validated_tools = self._sanitize_tool_names(validated_tools)
            name_mapping = self._current_name_mapping
            log.debug(f"Tool sanitization: {len(name_mapping)} tools processed for {self.detected_provider} compatibility")
        
        # Prepare messages for conversation (sanitize tool names in history)
        if name_mapping:
            validated_messages = self._prepare_messages_for_conversation(validated_messages)
        
        # Use configuration-aware parameter adjustment
        validated_kwargs = self.validate_parameters(**validated_kwargs)

        if validated_stream:
            return self._stream_completion_async(validated_messages, validated_tools, name_mapping, **validated_kwargs)
        else:
            return self._regular_completion(validated_messages, validated_tools, name_mapping, **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced async streaming with reasoning model support (including GPT-5) and FIXED streaming logic.
        """
        max_retries = 1
        
        for attempt in range(max_retries + 1):
            try:
                # Prepare messages and parameters for reasoning models (including GPT-5)
                prepared_messages = self._prepare_reasoning_model_messages(messages)
                prepared_kwargs = self._prepare_reasoning_model_parameters(**kwargs)
                
                log.debug(f"[{self.detected_provider}] Starting streaming (attempt {attempt + 1}): "
                         f"model={self.model}, messages={len(prepared_messages)}, "
                         f"tools={len(tools) if tools else 0}, "
                         f"reasoning_model={self._is_reasoning_model(self.model)}, "
                         f"generation={self._get_reasoning_model_generation(self.model)}")
                
                # Log reasoning model adjustments
                if self._is_reasoning_model(self.model):
                    param_changes = []
                    if 'max_completion_tokens' in prepared_kwargs:
                        param_changes.append(f"max_completion_tokens={prepared_kwargs['max_completion_tokens']}")
                    
                    generation = self._get_reasoning_model_generation(self.model)
                    if generation == 'gpt5':
                        param_changes.append("GPT-5 family (unified reasoning)")
                    
                    if param_changes:
                        log.debug(f"[{self.detected_provider}] Reasoning model adjustments: {', '.join(param_changes)}")
                
                response_stream = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=prepared_messages,
                    **({"tools": tools} if tools else {}),
                    stream=True,
                    **prepared_kwargs
                )
                
                chunk_count = 0
                async for result in self._stream_from_async(response_stream, name_mapping):
                    chunk_count += 1
                    yield result
                
                log.debug(f"[{self.detected_provider}] Streaming completed successfully with {chunk_count} chunks")
                return
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for reasoning model parameter errors (including GPT-5)
                if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                    log.error(f"[{self.detected_provider}] CRITICAL: Reasoning model parameter error not handled: {e}")
                elif "temperature" in error_str and "gpt-5" in self.model.lower():
                    log.error(f"[{self.detected_provider}] GPT-5 temperature restriction not handled: {e}")
                
                is_retryable = any(pattern in error_str for pattern in [
                    "timeout", "connection", "network", "temporary", "rate limit"
                ])
                
                if attempt < max_retries and is_retryable:
                    wait_time = (attempt + 1) * 1.0
                    log.warning(f"[{self.detected_provider}] Streaming attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    log.error(f"[{self.detected_provider}] Streaming failed after {attempt + 1} attempts: {e}")
                    yield {
                        "response": f"Error: {str(e)}",
                        "tool_calls": None,
                        "error": True
                    }
                    return

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enhanced non-streaming completion with reasoning model support (including GPT-5) and universal tool name restoration."""
        try:
            # Prepare messages and parameters for reasoning models (including GPT-5)
            prepared_messages = self._prepare_reasoning_model_messages(messages)
            prepared_kwargs = self._prepare_reasoning_model_parameters(**kwargs)
            
            log.debug(f"[{self.detected_provider}] Starting completion: "
                     f"model={self.model}, messages={len(prepared_messages)}, "
                     f"tools={len(tools) if tools else 0}, "
                     f"reasoning_model={self._is_reasoning_model(self.model)}, "
                     f"generation={self._get_reasoning_model_generation(self.model)}")
            
            # Log reasoning model adjustments for debugging
            if self._is_reasoning_model(self.model):
                param_changes = []
                if 'max_completion_tokens' in prepared_kwargs:
                    param_changes.append(f"max_completion_tokens={prepared_kwargs['max_completion_tokens']}")
                
                generation = self._get_reasoning_model_generation(self.model)
                if generation == 'gpt5':
                    param_changes.append("GPT-5 family (unified reasoning)")
                
                if param_changes:
                    log.debug(f"[{self.detected_provider}] Reasoning model adjustments: {', '.join(param_changes)}")
            
            resp = await self.async_client.chat.completions.create(
                model=self.model,
                messages=prepared_messages,
                **({"tools": tools} if tools else {}),
                stream=False,
                **prepared_kwargs
            )
            
            result = self._normalize_message(resp.choices[0].message)
            
            # Restore original tool names using universal restoration
            if name_mapping and result.get("tool_calls"):
                result = self._restore_tool_names_in_response(result, name_mapping)
            
            log.debug(f"[{self.detected_provider}] Completion successful: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            log.error(f"[{self.detected_provider}] Error in completion: {e}")
            
            # Provide helpful error messages for common reasoning model issues
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                log.error(f"[{self.detected_provider}] REASONING MODEL PARAMETER ERROR: "
                         f"This appears to be a reasoning model that requires max_completion_tokens. "
                         f"The parameter conversion should have handled this automatically.")
            elif "temperature" in error_msg and "gpt-5" in self.model.lower():
                log.error(f"[{self.detected_provider}] GPT-5 TEMPERATURE RESTRICTION: "
                         f"GPT-5 models only support default temperature (1.0). "
                         f"The parameter filtering should have handled this automatically.")
            
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping from universal system
        if hasattr(self, '_current_name_mapping'):
            self._current_name_mapping = {}
        
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        if hasattr(self.client, 'close'):
            self.client.close()