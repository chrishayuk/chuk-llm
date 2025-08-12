# chuk_llm/llm/providers/azure_openai_client.py
"""
Azure OpenAI chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around the official `openai` SDK configured for Azure OpenAI
that uses the unified configuration system for all capabilities.

COMPLETE VERSION WITH SMART DISCOVERY SUPPORT:
- Supports ANY Azure deployment name (no validation against static list)
- Smart defaults for discovered deployments
- Pattern-based capability detection
- Full compatibility with custom deployment names like "scribeflowgpt4o"
- Universal tool name compatibility with bidirectional mapping
"""
from __future__ import annotations
import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple, Set
import uuid
import openai
import logging
import os
import re

# mixins
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin
from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

# base
from ..core.base import BaseLLMClient

log = logging.getLogger(__name__)

class AzureOpenAILLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-driven wrapper around the official `openai` SDK for Azure OpenAI
    that gets all capabilities from the unified YAML configuration.
    
    ENHANCED: Now supports discovered deployments with smart defaults!
    - ANY deployment name is valid (no static list validation)
    - Smart capability detection based on deployment name patterns
    - Automatic feature inference for unknown deployments
    
    Uses universal tool name compatibility system to handle any naming convention:
    - stdio.read_query -> stdio_read_query (if needed)
    - web.api:search -> web_api_search (if needed)  
    - database.sql.execute -> database_sql_execute (if needed)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        azure_ad_token_provider: Optional[Any] = None,
    ) -> None:
        # Initialize mixins
        ConfigAwareProviderMixin.__init__(self, "azure_openai", model)
        ToolCompatibilityMixin.__init__(self, "azure_openai")
        
        self.model = model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version or "2024-02-01"
        self.azure_deployment = azure_deployment or model  # Default deployment name to model name
        
        # Azure OpenAI client configuration
        client_kwargs = {
            "api_version": self.api_version,
            "azure_endpoint": azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
        }
        
        # Authentication - priority order: token provider > token > api key
        if azure_ad_token_provider:
            client_kwargs["azure_ad_token_provider"] = azure_ad_token_provider
        elif azure_ad_token:
            client_kwargs["azure_ad_token"] = azure_ad_token
        else:
            client_kwargs["api_key"] = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        
        # Validate required parameters
        if not client_kwargs.get("azure_endpoint"):
            raise ValueError("azure_endpoint is required for Azure OpenAI. Set AZURE_OPENAI_ENDPOINT or pass azure_endpoint parameter.")
        
        if not any([azure_ad_token_provider, azure_ad_token, client_kwargs.get("api_key")]):
            raise ValueError("Authentication required: provide api_key, azure_ad_token, or azure_ad_token_provider")
        
        # Use AzureOpenAI for real streaming support
        self.async_client = openai.AsyncAzureOpenAI(**client_kwargs)
        
        # Keep sync client for backwards compatibility if needed
        self.client = openai.AzureOpenAI(**client_kwargs)
        
        log.debug(f"Azure OpenAI client initialized: endpoint={azure_endpoint}, deployment={self.azure_deployment}, model={self.model}")

    # ================================================================
    # SMART DEFAULTS FOR DISCOVERED AZURE DEPLOYMENTS
    # ================================================================
    
    @staticmethod
    def _get_smart_default_features(deployment_name: str) -> Set[str]:
        """
        Get smart default features for an Azure deployment based on naming patterns.
        
        This allows discovered deployments to work even if not in static config.
        For example, "scribeflowgpt4o" will be detected as a GPT-4 variant with full capabilities.
        """
        deployment_lower = deployment_name.lower()
        
        # Base features that ALL modern Azure OpenAI deployments should have
        base_features = {"text", "streaming", "system_messages"}
        
        # Pattern-based feature detection for Azure deployments
        if any(pattern in deployment_lower for pattern in ['o1', 'o3', 'o4', 'o5']):
            # Reasoning model deployments
            if 'o1' in deployment_lower:
                # O1 models are legacy - no tools, limited capabilities
                return {"text", "reasoning"}
            else:
                # O3+ models are modern reasoning - assume full tool support
                return {"text", "streaming", "tools", "reasoning", "system_messages"}
        
        elif any(pattern in deployment_lower for pattern in ['gpt-4', 'gpt4', 'gpt-5', 'gpt5', 'gpt-3', 'gpt3']):
            # Standard GPT model deployments - assume modern capabilities
            features = base_features | {"tools", "json_mode"}
            
            # Vision support for GPT-4+ models (not 3.5)
            if any(v in deployment_lower for v in ['gpt-4', 'gpt4', 'gpt-5', 'gpt5']):
                features.add("vision")
                features.add("parallel_calls")
                
            # GPT-5 models use reasoning architecture
            if any(v in deployment_lower for v in ['gpt-5', 'gpt5']):
                features.add("reasoning")
                
            return features
        
        elif 'embedding' in deployment_lower or 'ada' in deployment_lower:
            # Embedding models don't support chat features
            return {"text"}
        
        elif 'whisper' in deployment_lower:
            # Audio models
            return {"audio", "transcription"}
        
        elif 'dall-e' in deployment_lower or 'dalle' in deployment_lower:
            # Image generation models
            return {"image_generation"}
        
        else:
            # Unknown Azure deployment patterns - be optimistic about capabilities
            # Azure deployments of unknown models likely support standard features
            log.info(f"Unknown Azure deployment pattern '{deployment_name}' - assuming modern chat capabilities including tools")
            return base_features | {"tools", "json_mode"}
    
    @staticmethod
    def _get_smart_default_parameters(deployment_name: str) -> Dict[str, Any]:
        """Get smart default parameters for an Azure deployment"""
        deployment_lower = deployment_name.lower()
        
        # Reasoning model parameter handling
        if any(pattern in deployment_lower for pattern in ['o1', 'o3', 'o4', 'o5', 'gpt-5', 'gpt5']):
            return {
                "max_context_length": 272000 if 'gpt-5' in deployment_lower or 'gpt5' in deployment_lower else 200000,
                "max_output_tokens": 128000 if 'gpt-5' in deployment_lower or 'gpt5' in deployment_lower else 65536,
                "requires_max_completion_tokens": True,
                "parameter_mapping": {
                    "max_tokens": "max_completion_tokens"
                },
                "unsupported_params": ["temperature", "top_p", "frequency_penalty", "presence_penalty"],
                "supports_tools": "o1" not in deployment_lower
            }
        
        # Standard model defaults - generous assumptions for new deployments
        return {
            "max_context_length": 128000,
            "max_output_tokens": 8192,
            "supports_tools": True  # Assume new deployments support tools
        }
    
    def _has_explicit_deployment_config(self, deployment: str) -> bool:
        """Check if deployment has explicit configuration"""
        try:
            from chuk_llm.configuration import get_config
            config_manager = get_config()
            provider_config = config_manager.get_provider("azure_openai")
            
            # Check if deployment is in the models list
            if deployment in provider_config.models:
                return True
            
            # Check if any model capability pattern matches this deployment
            for capability in provider_config.model_capabilities:
                if hasattr(capability, 'matches') and capability.matches(deployment):
                    return True
                elif hasattr(capability, 'pattern'):
                    import re
                    if re.match(capability.pattern, deployment):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def validate_model(self, model_name: str) -> bool:
        """
        Override model validation for Azure.
        
        KEY FIX: Azure deployments can have ANY name, so we should NOT validate
        against a static list. This allows "scribeflowgpt4o" and any other custom
        deployment name to work!
        """
        # For Azure, ANY deployment name is potentially valid
        # The actual validation happens when we try to use it
        log.debug(f"[azure_openai] Deployment '{model_name}' - skipping static validation (Azure allows any deployment name)")
        return True
    
    def supports_feature(self, feature_name: str) -> bool:
        """
        Enhanced feature support with smart defaults for unknown Azure deployments.
        
        This is the KEY METHOD that enables discovered deployments to work!
        """
        try:
            # First try the configuration system
            config_supports = super().supports_feature(feature_name)
            
            # If configuration gives a definitive answer, trust it
            if config_supports is not None:
                return config_supports
            
            # Configuration returned None (unknown deployment) - use smart defaults
            # This is where we handle "scribeflowgpt4o" and other custom deployments!
            smart_features = self._get_smart_default_features(self.azure_deployment)
            supports_smart = feature_name in smart_features
            
            if supports_smart:
                log.info(f"[azure_openai] No config for deployment '{self.azure_deployment}' - "
                        f"using smart default: supports {feature_name}")
            else:
                log.debug(f"[azure_openai] No config for deployment '{self.azure_deployment}' - "
                         f"smart default: doesn't support {feature_name}")
            
            return supports_smart
            
        except Exception as e:
            log.warning(f"Feature support check failed for {feature_name}: {e}")
            
            # For Azure, be optimistic about unknown features for chat models
            deployment_lower = self.azure_deployment.lower()
            if any(pattern in deployment_lower for pattern in ['gpt', 'chat', 'turbo']):
                log.info(f"[azure_openai] Error checking config - assuming deployment '{self.azure_deployment}' "
                        f"supports {feature_name} (optimistic fallback)")
                return True
            
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Enhanced model info with smart defaults for discovered deployments.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Check if using smart defaults
        using_smart_defaults = not self._has_explicit_deployment_config(self.azure_deployment)
        
        # Get tool compatibility info
        tool_compatibility = self.get_tool_compatibility_info()
        
        # Add smart defaults info if applicable
        if using_smart_defaults:
            smart_features = self._get_smart_default_features(self.azure_deployment)
            smart_params = self._get_smart_default_parameters(self.azure_deployment)
            
            # Override with smart defaults
            info.update({
                "using_smart_defaults": True,
                "smart_default_features": list(smart_features),
                "smart_default_parameters": smart_params,
                "discovery_note": f"Deployment '{self.azure_deployment}' not in static config - using smart defaults",
                "features": list(smart_features),
                "max_context_length": smart_params.get("max_context_length", 128000),
                "max_output_tokens": smart_params.get("max_output_tokens", 8192),
            })
        
        # Add Azure-specific metadata
        if not info.get("error"):
            info.update({
                "azure_specific": {
                    "endpoint": self.azure_endpoint,
                    "deployment": self.azure_deployment,
                    "api_version": self.api_version,
                    "authentication_type": self._get_auth_type(),
                    "deployment_to_model_mapping": True,
                    "supports_custom_deployments": True,  # Key feature!
                },
                "openai_compatible": True,
                **tool_compatibility,
                "parameter_mapping": {
                    "temperature": "temperature",
                    "max_tokens": "max_tokens", 
                    "top_p": "top_p",
                    "frequency_penalty": "frequency_penalty",
                    "presence_penalty": "presence_penalty",
                    "stop": "stop",
                    "stream": "stream",
                    "tools": "tools",
                    "tool_choice": "tool_choice"
                },
                "azure_parameters": [
                    "azure_endpoint", "api_version", "azure_deployment", 
                    "azure_ad_token", "azure_ad_token_provider"
                ]
            })
        
        return info

    def _get_auth_type(self) -> str:
        """Determine the authentication type being used"""
        if hasattr(self.async_client, '_azure_ad_token_provider') and self.async_client._azure_ad_token_provider:
            return "azure_ad_token_provider"
        elif hasattr(self.async_client, '_azure_ad_token') and self.async_client._azure_ad_token:
            return "azure_ad_token"
        else:
            return "api_key"

    def _validate_request_with_config(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], bool, Dict[str, Any]]:
        """
        Enhanced validation that uses smart defaults for unknown deployments.
        FIXED: Actually disable streaming when not supported.
        """
        validated_messages = messages
        validated_tools = tools
        validated_stream = stream
        validated_kwargs = kwargs.copy()
        
        # Check streaming support (use smart defaults if needed)
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but deployment '{self.azure_deployment}' doesn't support streaming")
            # FIXED: Actually disable streaming when not supported
            validated_stream = False
        
        # Check tool support (use smart defaults for unknown deployments)
        if tools:
            if not self.supports_feature("tools"):
                log.warning(f"Tools provided but deployment '{self.azure_deployment}' doesn't support tools")
                validated_tools = None
            elif not self._has_explicit_deployment_config(self.azure_deployment):
                # Log when using smart defaults for tool support
                log.info(f"Using smart default: assuming deployment '{self.azure_deployment}' supports tools")
        
        # Check vision support
        has_vision = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.warning(f"Vision content detected but deployment '{self.azure_deployment}' doesn't support vision")
        
        # Check JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            if not self.supports_feature("json_mode"):
                log.warning(f"JSON mode requested but deployment '{self.azure_deployment}' doesn't support JSON mode")
                validated_kwargs.pop("response_format", None)
        
        # Use smart parameter defaults if no config
        if not self._has_explicit_deployment_config(self.azure_deployment):
            smart_params = self._get_smart_default_parameters(self.azure_deployment)
            
            # Apply smart defaults for max_tokens if not set
            if 'max_tokens' not in validated_kwargs and 'max_completion_tokens' not in validated_kwargs:
                if smart_params.get("requires_max_completion_tokens"):
                    validated_kwargs['max_completion_tokens'] = smart_params.get("max_output_tokens", 8192)
                else:
                    validated_kwargs['max_tokens'] = smart_params.get("max_output_tokens", 8192)
            
            log.debug(f"[azure_openai] Applied smart parameter defaults for '{self.azure_deployment}'")
        
        # Validate parameters using configuration or smart defaults
        validated_kwargs = self.validate_parameters(**validated_kwargs)
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    def _prepare_azure_request_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters for Azure OpenAI API"""
        # Use deployment name instead of model for Azure
        params = kwargs.copy()
        
        # Azure-specific parameter handling
        if "deployment_name" in params:
            params["model"] = params.pop("deployment_name")
        
        # Don't override if model is already set correctly
        if "model" not in params:
            params["model"] = self.azure_deployment
        
        return params

    # ------------------------------------------------------------------ #
    # Enhanced public API using configuration and smart defaults         #
    # ------------------------------------------------------------------ #
    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Configuration-aware completion that validates capabilities before processing.
        Uses universal tool name compatibility with bidirectional mapping.
        Supports discovered deployments with smart defaults!
        """
        # Validate request against configuration (with smart defaults for unknown deployments)
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Apply universal tool name sanitization (stores mapping for restoration)
        name_mapping = {}
        if validated_tools:
            validated_tools = self._sanitize_tool_names(validated_tools)
            name_mapping = self._current_name_mapping
            log.debug(f"Tool sanitization: {len(name_mapping)} tools processed for Azure OpenAI compatibility")
        
        # Use configuration-aware parameter adjustment
        validated_kwargs = self._adjust_parameters_for_provider(validated_kwargs)

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
        FIXED: Azure OpenAI streaming with proper JSON completion testing.
        
        Key fixes:
        - Only yield tool calls when JSON arguments are complete and parseable
        - Removed complex signature tracking system
        - Added completion status tracking
        - Prevents JSON parsing errors in downstream code
        """
        max_retries = 1
        
        for attempt in range(max_retries + 1):
            try:
                log.debug(f"[azure_openai] Starting streaming (attempt {attempt + 1}): "
                        f"deployment={self.azure_deployment}, messages={len(messages)}, tools={len(tools) if tools else 0}")
                
                # Prepare request parameters
                request_params = kwargs.copy()
                request_params["model"] = self.azure_deployment
                request_params["messages"] = messages
                if tools:
                    request_params["tools"] = tools
                request_params["stream"] = True
                
                response_stream = await self.async_client.chat.completions.create(**request_params)
                
                chunk_count = 0
                total_content = ""
                
                # FIXED: Simple completion-based tracking instead of signature system
                accumulated_tool_calls = {}  # {index: {id, name, arguments, complete}}
                
                async for chunk in response_stream:
                    chunk_count += 1
                    
                    content = ""
                    completed_tool_calls = []  # Only completed tool calls this chunk
                    
                    try:
                        if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                            choice = chunk.choices[0]
                            
                            if hasattr(choice, 'delta') and choice.delta:
                                delta = choice.delta
                                
                                # Handle content - this works fine
                                if hasattr(delta, 'content') and delta.content is not None:
                                    content = str(delta.content)
                                    total_content += content
                                
                                # FIXED: Handle tool calls with proper completion testing
                                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                    for tc in delta.tool_calls:
                                        try:
                                            tc_index = getattr(tc, 'index', 0)
                                            
                                            # Initialize accumulator with completion tracking
                                            if tc_index not in accumulated_tool_calls:
                                                accumulated_tool_calls[tc_index] = {
                                                    "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                                    "name": "",
                                                    "arguments": "",
                                                    "complete": False  # ADDED: Track completion status
                                                }
                                            
                                            tool_call_data = accumulated_tool_calls[tc_index]
                                            
                                            # Update data
                                            if hasattr(tc, 'id') and tc.id:
                                                tool_call_data["id"] = tc.id
                                            
                                            if hasattr(tc, 'function') and tc.function:
                                                if hasattr(tc.function, 'name') and tc.function.name:
                                                    tool_call_data["name"] += tc.function.name
                                                
                                                if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                                    tool_call_data["arguments"] += tc.function.arguments
                                            
                                            # CRITICAL FIX: Only yield when JSON is complete and valid
                                            if (tool_call_data["name"] and 
                                                tool_call_data["arguments"] and 
                                                not tool_call_data["complete"]):
                                                
                                                try:
                                                    # Handle Azure-specific argument formatting
                                                    args_str = tool_call_data["arguments"]
                                                    if args_str.startswith('""') and args_str.endswith('""'):
                                                        args_str = args_str[2:-2]
                                                    
                                                    # Test if JSON is complete and valid
                                                    parsed_args = json.loads(args_str)
                                                    
                                                    # Mark as complete and add to current chunk
                                                    tool_call_data["complete"] = True
                                                    
                                                    tool_call = {
                                                        "id": tool_call_data["id"],
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_call_data["name"],
                                                            "arguments": json.dumps(parsed_args)
                                                        }
                                                    }
                                                    
                                                    completed_tool_calls.append(tool_call)
                                                    log.debug(f"Azure tool call {tc_index} completed: {tool_call_data['name']}")
                                                
                                                except json.JSONDecodeError:
                                                    # JSON incomplete - keep accumulating
                                                    log.debug(f"Azure tool call {tc_index} JSON incomplete, continuing accumulation")
                                                    pass
                                        
                                        except Exception as e:
                                            log.debug(f"Error processing Azure streaming tool call chunk: {e}")
                                            continue
                    
                    except Exception as chunk_error:
                        log.warning(f"Error processing Azure chunk {chunk_count}: {chunk_error}")
                        content = ""
                    
                    # Prepare result
                    result = {
                        "response": content,
                        "tool_calls": completed_tool_calls if completed_tool_calls else None,
                    }
                    
                    # Restore tool names using universal restoration
                    if name_mapping and completed_tool_calls:
                        result = self._restore_tool_names_in_response(result, name_mapping)
                    
                    # Only yield if we have content or completed tool calls
                    if content or completed_tool_calls:
                        yield result
                
                log.debug(f"[azure_openai] Streaming completed: {chunk_count} chunks, "
                        f"{len(total_content)} total characters, {len(accumulated_tool_calls)} tool calls")
                
                # If we reach here, streaming was successful - exit the retry loop
                return
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle deployment errors immediately - these are not retryable
                if "deployment" in error_str and "not found" in error_str:
                    log.error(f"[azure_openai] Deployment error - deployment '{self.azure_deployment}' not found: {e}")
                    yield {
                        "response": f"Azure deployment error: Deployment '{self.azure_deployment}' not found",
                        "tool_calls": [],
                        "error": True
                    }
                    return  # Don't retry deployment errors
                
                # Check if this is a retryable error
                is_retryable = any(pattern in error_str for pattern in [
                    "timeout", "connection", "network", "temporary", "rate limit"
                ])
                
                if attempt < max_retries and is_retryable:
                    wait_time = (attempt + 1) * 1.0
                    log.warning(f"[azure_openai] Streaming attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue  # Retry the request
                else:
                    # Final failure - yield exactly one error chunk
                    log.error(f"[azure_openai] Streaming failed after {attempt + 1} attempts: {e}")
                    yield {
                        "response": f"Error: {str(e)}",
                        "tool_calls": [],
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
        """Enhanced non-streaming completion using Azure OpenAI configuration with tool name restoration."""
        try:
            log.debug(f"[azure_openai] Starting completion: "
                     f"deployment={self.azure_deployment}, messages={len(messages)}, tools={len(tools) if tools else 0}")
            
            # Prepare request parameters - ensure model is set to deployment name
            request_params = kwargs.copy()
            request_params["model"] = self.azure_deployment
            request_params["messages"] = messages
            if tools:
                request_params["tools"] = tools
            request_params["stream"] = False
            
            resp = await self.async_client.chat.completions.create(**request_params)
            
            # Enhanced response debugging
            if hasattr(resp, 'choices') and resp.choices:
                choice = resp.choices[0]
                log.debug(f"[azure_openai] Response choice type: {type(choice)}")
                if hasattr(choice, 'message'):
                    message = choice.message
                    log.debug(f"[azure_openai] Message type: {type(message)}")
                    content_preview = getattr(message, 'content', 'NO CONTENT')
                    if content_preview:
                        log.debug(f"[azure_openai] Content preview: {str(content_preview)[:100]}...")
                    else:
                        log.debug(f"[azure_openai] No content in message")
            
            # Use enhanced normalization from OpenAIStyleMixin
            result = self._normalize_message(resp.choices[0].message)
            
            # Restore original tool names using universal restoration
            if name_mapping and result.get("tool_calls"):
                result = self._restore_tool_names_in_response(result, name_mapping)
            
            # Log result
            log.debug(f"[azure_openai] Completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for deployment not found errors
            if "deployment" in error_str and "not found" in error_str:
                log.error(f"[azure_openai] Deployment '{self.azure_deployment}' not found. "
                         f"Please check your Azure OpenAI resource for available deployments.")
                return {
                    "response": f"Deployment error: '{self.azure_deployment}' not found in your Azure OpenAI resource",
                    "tool_calls": [],
                    "error": True
                }
            
            # Check for tool naming errors
            if "function" in error_str and ("name" in error_str or "invalid" in error_str):
                log.error(f"[azure_openai] Tool naming error (this should not happen with universal compatibility): {e}")
                return {
                    "response": f"Tool naming error: {str(e)}",
                    "tool_calls": [],
                    "error": True
                }
            
            log.error(f"[azure_openai] Error in completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    def _normalize_message(self, msg) -> Dict[str, Any]:
        """
        Azure-specific message normalization with FIXED argument parsing.
        
        CRITICAL FIX: Azure OpenAI returns tool arguments as JSON strings, not dicts.
        This properly handles the string format and ensures arguments are always
        properly formatted JSON strings for downstream processing.
        """
        try:
            # Use the inherited OpenAI normalization method as base
            result = super()._normalize_message(msg)
            
            # AZURE FIX: Ensure tool arguments are properly formatted JSON strings
            if result.get("tool_calls"):
                fixed_tool_calls = []
                
                for tool_call in result["tool_calls"]:
                    if "function" in tool_call and "arguments" in tool_call["function"]:
                        args = tool_call["function"]["arguments"]
                        
                        # Azure often returns arguments as strings, sometimes double-quoted
                        if isinstance(args, str):
                            try:
                                # Try to parse the JSON to validate it
                                parsed_args = json.loads(args)
                                # Re-serialize to ensure consistent format
                                tool_call["function"]["arguments"] = json.dumps(parsed_args)
                            except json.JSONDecodeError:
                                # If parsing fails, try to handle nested quoting
                                try:
                                    # Handle cases like ""{\"key\":\"value\"}"" (double quotes)
                                    if args.startswith('""') and args.endswith('""'):
                                        inner_json = args[2:-2]  # Remove outer quotes
                                        parsed_args = json.loads(inner_json)
                                        tool_call["function"]["arguments"] = json.dumps(parsed_args)
                                    else:
                                        # Last resort: wrap in empty object if invalid
                                        log.warning(f"Invalid tool arguments from Azure: {args}")
                                        tool_call["function"]["arguments"] = "{}"
                                except:
                                    tool_call["function"]["arguments"] = "{}"
                        elif isinstance(args, dict):
                            # Already a dict, convert to JSON string
                            tool_call["function"]["arguments"] = json.dumps(args)
                        else:
                            # Other types, default to empty object
                            tool_call["function"]["arguments"] = "{}"
                    
                    fixed_tool_calls.append(tool_call)
                
                result["tool_calls"] = fixed_tool_calls
            
            return result
            
        except AttributeError:
            # Fallback implementation if mixin method not available
            content = None
            tool_calls = []
            
            # Extract content
            if hasattr(msg, 'content'):
                content = msg.content
            
            # Extract tool calls with FIXED argument handling
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        # Get function arguments - Azure specific handling
                        raw_args = getattr(tc.function, "arguments", "{}")
                        
                        # AZURE FIX: Properly handle argument formats
                        if isinstance(raw_args, str):
                            try:
                                # Validate JSON and reformat
                                parsed_args = json.loads(raw_args)
                                formatted_args = json.dumps(parsed_args)
                            except json.JSONDecodeError:
                                log.warning(f"Invalid JSON in Azure tool call arguments: {raw_args}")
                                formatted_args = "{}"
                        elif isinstance(raw_args, dict):
                            formatted_args = json.dumps(raw_args)
                        else:
                            log.warning(f"Unexpected Azure argument type: {type(raw_args)}")
                            formatted_args = "{}"
                        
                        tool_calls.append({
                            "id": getattr(tc, "id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": getattr(tc.function, "name", "unknown"),
                                "arguments": formatted_args  # Always a properly formatted JSON string
                            }
                        })
                    except Exception as e:
                        log.warning(f"Failed to process Azure tool call: {e}")
                        continue
            
            # Return standard format
            if tool_calls:
                return {"response": content if content else None, "tool_calls": tool_calls}
            else:
                return {"response": content or "", "tool_calls": []}
            
    def _adjust_parameters_for_provider(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters using configuration instead of hardcoded rules.
        Now with smart defaults for unknown deployments!
        """
        adjusted = kwargs.copy()
        
        try:
            # Use the configuration-aware parameter validation
            adjusted = self.validate_parameters(**adjusted)
            
            # If no explicit config, use smart defaults
            if not self._has_explicit_deployment_config(self.azure_deployment):
                smart_params = self._get_smart_default_parameters(self.azure_deployment)
                
                # Apply smart defaults if not already set
                if 'max_tokens' not in adjusted and 'max_completion_tokens' not in adjusted:
                    max_output = smart_params.get("max_output_tokens", 8192)
                    if smart_params.get("requires_max_completion_tokens"):
                        adjusted['max_completion_tokens'] = max_output
                    else:
                        adjusted['max_tokens'] = max_output
                    log.debug(f"[azure_openai] Applied smart default max_tokens={max_output} for '{self.azure_deployment}'")
            
            # Additional Azure OpenAI-specific parameter handling
            model_caps = self._get_model_capabilities()
            if model_caps:
                # Adjust max_tokens based on config if not already handled
                if 'max_tokens' in adjusted and model_caps.max_output_tokens:
                    if adjusted['max_tokens'] > model_caps.max_output_tokens:
                        log.debug(f"Adjusting max_tokens from {adjusted['max_tokens']} to {model_caps.max_output_tokens} for azure_openai")
                        adjusted['max_tokens'] = model_caps.max_output_tokens
        
        except Exception as e:
            log.debug(f"Could not adjust parameters using config: {e}")
            # Fallback: ensure max_tokens is set
            if 'max_tokens' not in adjusted:
                adjusted['max_tokens'] = 4096
        
        return adjusted

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping
        self._current_name_mapping = {}
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        if hasattr(self.client, 'close'):
            self.client.close()

    def __repr__(self) -> str:
        return f"AzureOpenAILLMClient(deployment={self.azure_deployment}, model={self.model}, endpoint={self.azure_endpoint})"