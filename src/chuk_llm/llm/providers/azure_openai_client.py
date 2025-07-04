# chuk_llm/llm/providers/azure_openai_client.py
"""
Azure OpenAI chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around the official `openai` SDK configured for Azure OpenAI
that uses the unified configuration system for all capabilities.

Key Features:
- Azure-specific authentication and endpoint handling
- Deployment name to model mapping
- Azure API versioning support
- Full compatibility with existing OpenAI provider features
- Configuration-driven capabilities
"""
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple
import openai
import logging
import os
import re

# mixins
from chuk_llm.llm.providers._mixins import OpenAIStyleMixin
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

# base
from ..core.base import BaseLLMClient

log = logging.getLogger(__name__)

class AzureOpenAILLMClient(ConfigAwareProviderMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-driven wrapper around the official `openai` SDK for Azure OpenAI
    that gets all capabilities from the unified YAML configuration.
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
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, "azure_openai", model)
        
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

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with Azure OpenAI-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add Azure OpenAI-specific metadata only if no error
        if not info.get("error"):
            info.update({
                "azure_specific": {
                    "endpoint": self.azure_endpoint,
                    "deployment": self.azure_deployment,
                    "api_version": self.api_version,
                    "authentication_type": self._get_auth_type(),
                    "deployment_to_model_mapping": True,
                },
                "openai_compatible": True,
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
        Validate request against configuration before processing.
        """
        validated_messages = messages
        validated_tools = tools
        validated_stream = stream
        validated_kwargs = kwargs.copy()
        
        # Check streaming support
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but {self.model} doesn't support streaming according to configuration")
            validated_stream = False
        
        # Check tool support
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but {self.model} doesn't support tools according to configuration")
            validated_tools = None
        
        # Check vision support
        has_vision = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.warning(f"Vision content detected but {self.model} doesn't support vision according to configuration")
        
        # Check JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            if not self.supports_feature("json_mode"):
                log.warning(f"JSON mode requested but {self.model} doesn't support JSON mode according to configuration")
                validated_kwargs.pop("response_format", None)
        
        # Validate parameters using configuration
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
    # Enhanced public API using configuration                            #
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
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Sanitize tool names (from mixin)
        validated_tools = self._sanitize_tool_names(validated_tools)
        
        # Use configuration-aware parameter adjustment
        validated_kwargs = self._adjust_parameters_for_provider(validated_kwargs)

        if validated_stream:
            return self._stream_completion_async(validated_messages, validated_tools, **validated_kwargs)
        else:
            return self._regular_completion(validated_messages, validated_tools, **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced async streaming with Azure OpenAI configuration-aware retry logic.
        """
        # Get retry configuration from provider config if available
        max_retries = 1  # Default
        try:
            provider_config = self._get_provider_config()
            if provider_config:
                max_retries = 2  # Azure typically needs fewer retries than other providers
        except:
            pass
        
        for attempt in range(max_retries + 1):
            try:
                log.debug(f"[azure_openai] Starting streaming (attempt {attempt + 1}): "
                         f"deployment={self.azure_deployment}, messages={len(messages)}, tools={len(tools) if tools else 0}")
                
                # Prepare request parameters - ensure model is set to deployment name
                request_params = kwargs.copy()
                request_params["model"] = self.azure_deployment
                request_params["messages"] = messages
                if tools:
                    request_params["tools"] = tools
                request_params["stream"] = True
                
                # Create streaming request
                response_stream = await self.async_client.chat.completions.create(**request_params)
                
                # Stream results using inherited method from OpenAIStyleMixin
                chunk_count = 0
                async for result in self._stream_from_async(response_stream):
                    chunk_count += 1
                    yield result
                
                # Success - exit retry loop
                log.debug(f"[azure_openai] Streaming completed successfully with {chunk_count} chunks")
                return
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a retryable error
                is_retryable = any(pattern in error_str for pattern in [
                    "timeout", "connection", "network", "temporary", "rate limit", "throttled"
                ])
                
                if attempt < max_retries and is_retryable:
                    wait_time = (attempt + 1) * 1.0  # 1s, 2s, 3s...
                    log.warning(f"[azure_openai] Streaming attempt {attempt + 1} failed: {e}. "
                               f"Retrying in {wait_time}s...")
                    import asyncio
                    await asyncio.sleep(wait_time)
                    continue
                else:
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enhanced non-streaming completion using Azure OpenAI configuration."""
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
            
            # Log result
            log.debug(f"[azure_openai] Completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            log.error(f"[azure_openai] Error in completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    def _normalize_message(self, msg) -> Dict[str, Any]:
        """
        Azure-specific message normalization that delegates to OpenAI mixin.
        """
        try:
            # Use the inherited OpenAI normalization method
            return super()._normalize_message(msg)
        except AttributeError:
            # Fallback implementation if mixin method not available
            content = None
            tool_calls = []
            
            # Extract content
            if hasattr(msg, 'content'):
                content = msg.content
            
            # Extract tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                import json
                import uuid
                for tc in msg.tool_calls:
                    try:
                        tool_calls.append({
                            "id": getattr(tc, "id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": getattr(tc.function, "name", "unknown"),
                                "arguments": json.dumps(getattr(tc.function, "arguments", {}))
                            }
                        })
                    except Exception as e:
                        log.warning(f"Failed to process tool call: {e}")
                        continue
            
            # Return standard format
            if tool_calls:
                return {"response": content if content else None, "tool_calls": tool_calls}
            else:
                return {"response": content or "", "tool_calls": []}

    def _adjust_parameters_for_provider(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters using configuration instead of hardcoded rules.
        """
        adjusted = kwargs.copy()
        
        try:
            # Use the configuration-aware parameter validation
            adjusted = self.validate_parameters(**adjusted)
            
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
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        if hasattr(self.client, 'close'):
            self.client.close()

    def __repr__(self) -> str:
        return f"AzureOpenAILLMClient(deployment={self.azure_deployment}, model={self.model}, endpoint={self.azure_endpoint})"