# chuk_llm/llm/providers/groq_client.py
"""
Groq chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around `groq` SDK that gets all capabilities from
unified YAML configuration and includes robust function calling error handling.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple
import json

from groq import AsyncGroq

# providers
from chuk_llm.llm.core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin
from ._config_mixin import ConfigAwareProviderMixin

log = logging.getLogger(__name__)


class GroqAILLMClient(ConfigAwareProviderMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-aware adapter around `groq` SDK that gets all capabilities
    from YAML configuration and includes enhanced function calling error handling.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, "groq", model)
        
        self.model = model
        
        # ✅ FIX: Provide correct default base URL for Groq
        groq_base_url = api_base or "https://api.groq.com/openai/v1"
        
        log.debug(f"Initializing Groq client with base_url: {groq_base_url}")
        
        # Use AsyncGroq for real streaming support
        self.async_client = AsyncGroq(
            api_key=api_key,
            base_url=groq_base_url
        )
        
        # Keep sync client for backwards compatibility if needed
        from groq import Groq
        self.client = Groq(
            api_key=api_key,
            base_url=groq_base_url
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with Groq-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add Groq-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "groq_specific": {
                    "ultra_fast_inference": True,
                    "openai_compatible": True,
                    "function_calling_notes": "May require retry fallbacks for complex tool schemas",
                    "model_family": self._detect_model_family(),
                },
                "api_base": groq_base_url if hasattr(self, 'groq_base_url') else "https://api.groq.com/openai/v1",
                "parameter_mapping": {
                    "temperature": "temperature",
                    "max_tokens": "max_tokens",
                    "top_p": "top_p",
                    "stop": "stop",
                    "stream": "stream"
                },
                "unsupported_parameters": [
                    "frequency_penalty", "presence_penalty", "logit_bias",
                    "user", "n", "best_of", "top_k", "seed", "response_format"
                ]
            })
        
        return info

    def _detect_model_family(self) -> str:
        """Detect model family for Groq-specific optimizations"""
        model_lower = self.model.lower()
        if "llama" in model_lower:
            return "llama"
        elif "mixtral" in model_lower:
            return "mixtral"
        elif "gemma" in model_lower:
            return "gemma"
        else:
            return "unknown"

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
        
        # Check tool support with Groq-specific validation
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but {self.model} doesn't support tools according to configuration")
            validated_tools = None
        elif tools:
            # Validate tool schemas for Groq compatibility
            validated_tools = self._validate_tools_for_groq(tools)
        
        # Check vision support
        has_vision = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.warning(f"Vision content detected but {self.model} doesn't support vision according to configuration")
        
        # Validate parameters using configuration
        validated_kwargs = self.validate_parameters(**validated_kwargs)
        
        # Remove unsupported parameters for Groq
        unsupported = ["frequency_penalty", "presence_penalty", "logit_bias", 
                      "user", "n", "best_of", "top_k", "seed", "response_format"]
        for param in unsupported:
            if param in validated_kwargs:
                log.debug(f"Removing unsupported parameter for Groq: {param}")
                validated_kwargs.pop(param)
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    def _validate_tools_for_groq(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and potentially simplify tool schemas for better Groq compatibility.
        """
        validated_tools = []
        
        for tool in tools:
            try:
                # Validate basic tool structure
                if not tool.get("function", {}).get("name"):
                    log.warning("Skipping tool without name")
                    continue
                
                # Simplify complex schemas that might cause Groq issues
                simplified_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": self._simplify_schema_for_groq(
                            tool["function"].get("parameters", {})
                        )
                    }
                }
                
                validated_tools.append(simplified_tool)
                
            except Exception as e:
                log.warning(f"Failed to validate tool for Groq: {tool.get('function', {}).get('name', 'unknown')}, error: {e}")
                continue
        
        return validated_tools

    def _simplify_schema_for_groq(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplify complex JSON schemas that might cause Groq function calling issues.
        """
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        
        # Start with a clean schema
        simplified = {
            "type": schema.get("type", "object"),
        }
        
        # Add properties if they exist
        if "properties" in schema:
            simplified["properties"] = {}
            for prop_name, prop_def in schema["properties"].items():
                # Simplify property definitions
                if isinstance(prop_def, dict):
                    simple_prop = {
                        "type": prop_def.get("type", "string"),
                    }
                    if "description" in prop_def:
                        simple_prop["description"] = prop_def["description"]
                    simplified["properties"][prop_name] = simple_prop
                else:
                    simplified["properties"][prop_name] = {"type": "string"}
        
        # Add required fields if they exist
        if "required" in schema and isinstance(schema["required"], list):
            simplified["required"] = schema["required"]
        
        return simplified

    # ──────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────
    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Configuration-aware completion with enhanced error handling for Groq.
        
        • stream=False → returns awaitable that resolves to single normalised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # Sanitize tool names (from mixin)
        validated_tools = self._sanitize_tool_names(validated_tools)

        if validated_stream:
            # Return async generator directly for real streaming
            return self._stream_completion_async(validated_messages, validated_tools or [], **validated_kwargs)

        # non-streaming path
        return self._regular_completion(validated_messages, validated_tools or [], **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real streaming using AsyncGroq with configuration-aware error handling.
        """
        try:
            log.debug(f"Starting Groq streaming for model: {self.model}")
            
            # Enhanced messages for better function calling with Groq (only if tools supported)
            if tools and self.supports_feature("tools"):
                enhanced_messages = self._enhance_messages_for_groq(messages, tools)
            else:
                enhanced_messages = messages
                tools = None  # Don't pass tools if not supported
            
            # Use async client for real streaming
            response_stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=enhanced_messages,
                tools=tools if tools else None,
                stream=True,
                **kwargs
            )
            
            chunk_count = 0
            # Yield chunks immediately as they arrive from Groq
            async for chunk in response_stream:
                chunk_count += 1
                
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Extract content and tool calls
                    content = delta.content or ""
                    tool_calls = getattr(delta, "tool_calls", [])
                    
                    # Only yield if we have actual content or tool calls
                    if content or tool_calls:
                        yield {
                            "response": content,
                            "tool_calls": tool_calls,
                        }
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            log.debug(f"Groq streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            error_str = str(e)
            
            # Handle Groq function calling errors in streaming
            if "Failed to call a function" in error_str and tools:
                log.warning(f"Groq streaming function calling failed, retrying without tools")
                
                # Retry without tools as fallback
                try:
                    response_stream = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **kwargs
                    )
                    
                    chunk_count = 0
                    async for chunk in response_stream:
                        chunk_count += 1
                        
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            content = delta.content or ""
                            
                            if content:
                                yield {
                                    "response": content,
                                    "tool_calls": [],
                                }
                        
                        if chunk_count % 10 == 0:
                            await asyncio.sleep(0)
                    
                    # Add final note about tools being disabled
                    yield {
                        "response": "\n\n[Note: Function calling disabled due to provider limitation]",
                        "tool_calls": [],
                    }
                    
                except Exception as retry_error:
                    log.error(f"Groq streaming retry failed: {retry_error}")
                    yield {
                        "response": f"Streaming error: {str(retry_error)}",
                        "tool_calls": [],
                        "error": True
                    }
            else:
                log.error(f"Error in Groq streaming: {e}")
                yield {
                    "response": f"Streaming error: {str(e)}",
                    "tool_calls": [],
                    "error": True
                }

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming completion with enhanced Groq function calling error handling."""
        try:
            log.debug(f"Groq regular completion - model: {self.model}, tools: {len(tools) if tools else 0}")
            
            # Enhanced messages for better function calling with Groq (only if tools supported)
            if tools and self.supports_feature("tools"):
                enhanced_messages = self._enhance_messages_for_groq(messages, tools)
                
                resp = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=enhanced_messages,
                    tools=tools,
                    stream=False,
                    **kwargs
                )
            else:
                resp = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    **kwargs
                )
                
            result = self._normalise_message(resp.choices[0].message)
            
            log.debug(f"Groq completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            error_str = str(e)
            
            # Handle Groq function calling errors specifically
            if "Failed to call a function" in error_str and tools:
                log.warning(f"Groq function calling failed, retrying without tools: {error_str}")
                
                # Retry without tools as fallback
                try:
                    resp = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=False,
                        **kwargs
                    )
                    result = self._normalise_message(resp.choices[0].message)
                    
                    # Add a note that tools were disabled due to Groq limitation
                    original_response = result.get("response", "")
                    result["response"] = (original_response + 
                                       "\n\n[Note: Function calling disabled due to provider limitation]")
                    return result
                    
                except Exception as retry_error:
                    log.error(f"Groq retry also failed: {retry_error}")
                    return {
                        "response": f"Error: {str(retry_error)}",
                        "tool_calls": [],
                        "error": True
                    }
            else:
                log.error(f"Error in Groq completion: {e}")
                return {
                    "response": f"Error: {str(e)}",
                    "tool_calls": [],
                    "error": True
                }

    def _enhance_messages_for_groq(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance messages with better instructions for Groq function calling.
        Groq models need explicit guidance for proper function calling.
        """
        if not tools or not self.supports_feature("system_messages"):
            return messages
        
        enhanced_messages = messages.copy()
        
        # Create function calling guidance
        function_names = [tool.get("function", {}).get("name", "unknown") for tool in tools]
        guidance = (
            f"You have access to the following functions: {', '.join(function_names)}. "
            "When calling functions:\n"
            "1. Use proper JSON format for arguments\n"
            "2. Ensure all required parameters are provided\n"
            "3. Use exact parameter names as specified\n"
            "4. Call functions when appropriate to help answer the user's question"
        )
        
        # Add or enhance system message (only if system messages are supported)
        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            enhanced_messages[0]["content"] = enhanced_messages[0]["content"] + "\n\n" + guidance
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": guidance
            })
        
        return enhanced_messages

    def _validate_tool_call_arguments(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validate tool call arguments to prevent Groq function calling errors.
        """
        try:
            if "function" not in tool_call:
                return False
            
            function = tool_call["function"]
            if "arguments" not in function:
                return False
            
            # Try to parse arguments as JSON
            args = function["arguments"]
            if isinstance(args, str):
                json.loads(args)  # This will raise if invalid JSON
            elif not isinstance(args, dict):
                return False
            
            return True
            
        except (json.JSONDecodeError, TypeError, KeyError):
            return False