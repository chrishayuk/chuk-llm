# chuk_llm/llm/providers/groq_client.py - FIXED VERSION FOR STREAMING DUPLICATION
"""
Groq chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced wrapper around `groq` SDK that gets all capabilities from
unified YAML configuration and includes universal tool name compatibility.

CRITICAL FIXES:
1. FIXED tool call duplication in streaming by tracking yielded signatures
2. Enhanced content extraction that properly handles tool-only responses
3. Reduced noisy warnings for normal tool-calling behavior
4. Better logging for successful tool extraction
5. Improved error handling and debugging information
"""
from __future__ import annotations

import asyncio
import logging
import uuid
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple

from groq import AsyncGroq

# providers
from chuk_llm.llm.core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin
from ._config_mixin import ConfigAwareProviderMixin
from ._tool_compatibility import ToolCompatibilityMixin

log = logging.getLogger(__name__)


class GroqAILLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-aware adapter around `groq` SDK that gets all capabilities
    from YAML configuration and includes universal tool name compatibility.
    
    CRITICAL FIX: Eliminates tool call duplication in streaming by tracking
    what has already been yielded and avoiding duplicate tool calls.
    
    Uses universal tool name compatibility system to handle any naming convention:
    - stdio.read_query -> stdio_read_query (if needed)
    - web.api:search -> web_api_search (if needed)
    - database.sql.execute -> database_sql_execute (if needed)
    - service:method -> service_method (if needed)
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        # Initialize ALL mixins including ToolCompatibilityMixin
        ConfigAwareProviderMixin.__init__(self, "groq", model)
        ToolCompatibilityMixin.__init__(self, "groq")
        
        self.model = model
        
        # Provide correct default base URL for Groq
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
        
        # Add tool compatibility info from universal system
        tool_compatibility = self.get_tool_compatibility_info()
        
        # Add Groq-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "groq_specific": {
                    "ultra_fast_inference": True,
                    "openai_compatible": True,
                    "function_calling_notes": "May require retry fallbacks for complex tool schemas",
                    "model_family": self._detect_model_family(),
                    "duplication_fix": "enabled",  # NEW: Indicates duplication fix is active
                },
                # Universal tool compatibility info
                **tool_compatibility,
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

    def _normalise_message(self, msg) -> Dict[str, Any]:
        """
        ENHANCED: Properly handle tool-only responses without noisy warnings.
        
        When LLMs make tool calls, they often don't include text content - this is 
        normal and expected behavior. The previous version logged confusing warnings
        about "no content found" when this is actually successful tool calling.
        """
        content = None
        tool_calls = []
        
        # Enhanced content extraction with proper error handling
        try:
            # Method 1: Direct attribute access
            if hasattr(msg, 'content') and msg.content is not None:
                content = str(msg.content)
                
            # Method 2: Message wrapper access  
            elif hasattr(msg, 'message') and hasattr(msg.message, 'content') and msg.message.content is not None:
                content = str(msg.message.content)
                
            # Method 3: Dict access
            elif isinstance(msg, dict) and 'content' in msg and msg['content'] is not None:
                content = str(msg['content'])
                
        except Exception as e:
            log.debug(f"Content extraction attempt failed (normal for tool-only responses): {e}")
        
        # Extract tool calls with enhanced error handling
        try:
            raw_tool_calls = None
            
            # Try multiple access patterns for tool calls
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
                                    if args.strip():
                                        parsed_args = json.loads(args)
                                        args_j = json.dumps(parsed_args)
                                    else:
                                        args_j = "{}"
                                elif isinstance(args, dict):
                                    args_j = json.dumps(args)
                                else:
                                    args_j = "{}"
                            except json.JSONDecodeError as je:
                                log.debug(f"Invalid JSON in tool arguments, using empty dict: {args} - {je}")
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
                        log.debug(f"Failed to process individual tool call: {e}")
                        continue
                        
        except Exception as e:
            log.debug(f"Tool call extraction failed: {e}")
        
        # Determine response format and log appropriately
        if tool_calls and not content:
            # Tool-only response (normal for function calling)
            log.debug(f"[Groq] Tool-only response with {len(tool_calls)} tool calls (no text content - this is normal)")
            response_value = None
        elif tool_calls and content:
            # Mixed response with both content and tools
            log.debug(f"[Groq] Mixed response: {len(content)} chars + {len(tool_calls)} tool calls")
            response_value = content
        elif content:
            # Text-only response
            log.debug(f"[Groq] Text-only response: {len(content)} characters")
            response_value = content
        else:
            # Empty response
            log.debug("[Groq] Empty response (no content or tool calls)")
            response_value = ""
        
        result = {
            "response": response_value,
            "tool_calls": tool_calls
        }
        
        return result

    def _enhance_tool_call_logging(self, response: Dict[str, Any], name_mapping: Dict[str, str] = None):
        """
        Enhanced logging for tool call analysis and debugging.
        """
        if not response.get("tool_calls"):
            if response.get("response"):
                log.info(f"[Groq] Text response: {len(response['response'])} characters")
            else:
                log.debug(f"[Groq] Empty response")
            return
        
        tool_calls = response["tool_calls"]
        log.info(f"[Groq] Successfully extracted {len(tool_calls)} tool calls")
        
        for i, tc in enumerate(tool_calls):
            func_name = tc.get("function", {}).get("name", "unknown")
            func_args = tc.get("function", {}).get("arguments", "{}")
            
            # Show name restoration if applicable
            if name_mapping:
                original_name = None
                for sanitized, orig in name_mapping.items():
                    if orig == func_name:
                        original_name = orig
                        break
                
                if original_name and original_name != func_name:
                    log.info(f"   {i+1}. {func_name} (restored from sanitized name)")
                else:
                    log.info(f"   {i+1}. {func_name}")
            else:
                log.info(f"   {i+1}. {func_name}")
            
            # Log parameter summary
            try:
                args_dict = json.loads(func_args) if isinstance(func_args, str) else func_args
                if args_dict:
                    param_summary = ", ".join([f"{k}={repr(v)[:50]}" for k, v in args_dict.items()])
                    log.debug(f"      Parameters: {param_summary}")
            except:
                log.debug(f"      Raw arguments: {func_args[:100]}")

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

    def _prepare_messages_for_conversation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CRITICAL FIX: Prepare messages for conversation by sanitizing tool names in message history.
        
        This ensures tool names in assistant messages match the sanitized names sent to the API.
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
                        log.debug(f"Sanitized tool name in Groq conversation: {original_name} -> {sanitized_name}")
                    
                    sanitized_tool_calls.append(tc_copy)
                
                prepared_msg["tool_calls"] = sanitized_tool_calls
                prepared_messages.append(prepared_msg)
            else:
                prepared_messages.append(msg)
        
        return prepared_messages

    # ──────────────────────────────────────────────────────────────────
    # Enhanced public API using universal tool compatibility
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
        Configuration-aware completion with universal tool name compatibility.
        
        CRITICAL FIX: Now properly handles tool call duplication in streaming
        by tracking what has been yielded and preventing duplicate emissions.
        
        Uses universal tool name compatibility system to handle any naming convention:
        - stdio.read_query -> stdio_read_query (sanitized and restored)
        - web.api:search -> web_api_search (sanitized and restored)
        - database.sql.execute -> database_sql_execute (sanitized and restored)
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        # CRITICAL FIX: Apply universal tool name sanitization (stores mapping for restoration)
        name_mapping = {}
        if validated_tools:
            validated_tools = self._sanitize_tool_names(validated_tools)
            name_mapping = self._current_name_mapping
            log.debug(f"Tool sanitization: {len(name_mapping)} tools processed for Groq compatibility")
        
        # CRITICAL FIX: Prepare messages for conversation (sanitize tool names in history)
        if name_mapping:
            validated_messages = self._prepare_messages_for_conversation(validated_messages)

        if validated_stream:
            # Return async generator directly for real streaming
            return self._stream_completion_async(validated_messages, validated_tools or [], name_mapping, **validated_kwargs)

        # non-streaming path
        return self._regular_completion(validated_messages, validated_tools or [], name_mapping, **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        CRITICAL FIX: Real streaming using AsyncGroq with FIXED tool call duplication prevention.
        
        The key fix: Track tool call signatures that have been yielded to prevent
        emitting the same tool call multiple times across different chunks.
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
            
            # CRITICAL FIX: Tool call accumulation WITH duplication prevention
            accumulated_tool_calls = {}  # {index: {id, name, arguments}}
            yielded_tool_signatures = set()  # Track what we've already yielded
            chunk_count = 0
            total_content = ""
            
            # Stream processing with accumulation and duplication prevention
            async for chunk in response_stream:
                chunk_count += 1
                
                content = ""
                new_complete_tools = []  # Only NEW complete tools for this chunk
                
                try:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        
                        # Handle content
                        if delta.content:
                            content = str(delta.content)
                            total_content += content
                        
                        # CRITICAL FIX: Handle tool calls with proper accumulation
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            for tc in delta.tool_calls:
                                try:
                                    # Get tool call index (Groq uses same pattern as OpenAI)
                                    tc_index = getattr(tc, 'index', 0)
                                    
                                    # Initialize accumulator for this tool call if needed
                                    if tc_index not in accumulated_tool_calls:
                                        accumulated_tool_calls[tc_index] = {
                                            "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                            "name": "",
                                            "arguments": ""
                                        }
                                    
                                    # Accumulate function data
                                    if hasattr(tc, 'function') and tc.function:
                                        if hasattr(tc.function, 'name') and tc.function.name:
                                            accumulated_tool_calls[tc_index]["name"] += tc.function.name
                                        
                                        if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                            accumulated_tool_calls[tc_index]["arguments"] += tc.function.arguments
                                        
                                        # Update ID if provided in this chunk
                                        if hasattr(tc, 'id') and tc.id:
                                            accumulated_tool_calls[tc_index]["id"] = tc.id
                                
                                except Exception as e:
                                    log.debug(f"Error processing Groq streaming tool call chunk: {e}")
                                    continue
                
                except Exception as chunk_error:
                    log.warning(f"Error processing Groq chunk {chunk_count}: {chunk_error}")
                    content = ""
                
                # CRITICAL FIX: Only yield NEW complete tool calls that haven't been yielded before
                for tc_index, tc_data in accumulated_tool_calls.items():
                    if tc_data["name"] and tc_data["arguments"]:
                        # Create unique signature for this tool call
                        tool_signature = f"{tc_data['name']}:{tc_data['arguments']}"
                        
                        # CRITICAL FIX: Only yield if we haven't yielded this exact tool call before
                        if tool_signature not in yielded_tool_signatures:
                            # Try to parse arguments to ensure they're complete JSON
                            try:
                                parsed_args = json.loads(tc_data["arguments"])
                                new_complete_tools.append({
                                    "id": tc_data["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc_data["name"],
                                        "arguments": json.dumps(parsed_args)  # Re-serialize for consistency
                                    }
                                })
                                
                                # Mark this tool call as yielded
                                yielded_tool_signatures.add(tool_signature)
                                log.debug(f"[Groq] Yielding NEW tool call: {tc_data['name']}")
                                
                            except json.JSONDecodeError:
                                # Arguments incomplete, wait for more chunks
                                pass
                        else:
                            log.debug(f"[Groq] Skipping duplicate tool call: {tc_data['name']}")

                result = {
                    "response": content,
                    "tool_calls": new_complete_tools,  # Only NEW tools, no duplicates
                }
                
                # CRITICAL: Restore tool names using universal restoration
                if name_mapping and new_complete_tools:
                    result = self._restore_tool_names_in_response(result, name_mapping)
                
                # IMPROVED: Yield content chunks immediately, but only yield NEW tool calls
                if content or new_complete_tools:
                    yield result
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            # FINAL CHECK: Ensure no incomplete tool calls are left (but avoid duplicates)
            final_new_tools = []
            for tc_index, tc_data in accumulated_tool_calls.items():
                if tc_data["name"] and tc_data["arguments"]:
                    tool_signature = f"{tc_data['name']}:{tc_data['arguments']}"
                    
                    # Only include if not already yielded
                    if tool_signature not in yielded_tool_signatures:
                        # Try to parse arguments one final time
                        try:
                            parsed_args = json.loads(tc_data["arguments"])
                            final_args = json.dumps(parsed_args)
                        except json.JSONDecodeError:
                            final_args = tc_data["arguments"] or "{}"
                        
                        final_new_tools.append({
                            "id": tc_data["id"],
                            "type": "function", 
                            "function": {
                                "name": tc_data["name"],
                                "arguments": final_args
                            }
                        })
                        
                        yielded_tool_signatures.add(tool_signature)
            
            if final_new_tools:
                final_result = {
                    "response": "",
                    "tool_calls": final_new_tools,
                }
                
                if name_mapping:
                    final_result = self._restore_tool_names_in_response(final_result, name_mapping)
                
                yield final_result
            
            log.debug(f"Groq streaming completed with {chunk_count} chunks, "
                    f"{len(total_content)} total characters, {len(yielded_tool_signatures)} unique tool calls yielded")
        
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
        name_mapping: Dict[str, str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming completion with enhanced tool extraction and logging."""
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
            
            # CRITICAL: Restore original tool names using universal restoration
            if name_mapping and result.get("tool_calls"):
                result = self._restore_tool_names_in_response(result, name_mapping)
            
            # Enhanced logging
            self._enhance_tool_call_logging(result, name_mapping)
            
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

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping from universal system
        self._current_name_mapping = {}
        # Groq client cleanup if needed
        pass