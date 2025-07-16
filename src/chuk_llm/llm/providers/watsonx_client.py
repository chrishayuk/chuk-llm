# chuk_llm/llm/providers/watsonx_client.py - COMPLETE WITH ENHANCED GRANITE PARSING
"""
Watson X chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps the official `ibm-watsonx-ai` SDK and exposes an **OpenAI-style** interface
compatible with the rest of *chuk-llm*.

CRITICAL UPDATE: Now includes comprehensive IBM Granite model tool format parsing
and universal ToolCompatibilityMixin for consistent tool name handling.

Key points
----------
*   Configuration-driven capabilities from YAML instead of hardcoded assumptions
*   Converts ChatML → Watson X Messages format (tools / multimodal, …)
*   Maps Watson X replies back to the common `{response, tool_calls}` schema
*   **Real Streaming** - uses Watson X's native streaming API
*   **Universal Tool Compatibility** - uses standardized ToolCompatibilityMixin
*   **Enhanced Granite Parsing** - handles 7+ IBM Granite tool calling formats
*   **Enterprise-grade tool sanitization** - handles any naming convention
"""
from __future__ import annotations
import ast
import json
import logging
import os
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple

# llm
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# providers
from ..core.base import BaseLLMClient
from ._mixins import OpenAIStyleMixin
from ._config_mixin import ConfigAwareProviderMixin
from ._tool_compatibility import ToolCompatibilityMixin

log = logging.getLogger(__name__)
if os.getenv("LOGLEVEL"):
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())

# ────────────────────────── helpers ──────────────────────────


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:  # noqa: D401 - util
    """Get *key* from dict **or** attribute-style object; fallback to *default*."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _parse_watsonx_tool_formats(text: str) -> List[Dict[str, Any]]:
    """
    Parse WatsonX/Granite-specific tool calling formats from text content.
    
    Handles multiple enterprise tool formats including IBM Granite model variations:
    1. Granite direct format: {'name': 'func', 'arguments': {...}}
    2. <|tool|>function_name</|tool|> <|param:name|>value</param>
    3. <function_call>{"name": "func", "arguments": {...}}</function_call>
    4. ```json {"function_call": {"name": "func", "arguments": {...}}} ```
    5. ```python function_name(param="value") ```
    6. {"api": "namespace", "function": "name", "params": {...}}
    7. <|tool|>function_name<|file_sep|> <|param|> {...}
    8. Direct JSON function calls
    """
    if not text or not isinstance(text, str):
        return []
    
    tool_calls = []
    
    try:
        # Format 1: Granite direct format: {'name': 'func', 'arguments': {...}} (PRIORITY)
        # This is the expected Granite response format from the tutorial
        granite_pattern = r"'name':\s*'([^']+)',\s*'arguments':\s*(\{[^}]*\})"
        granite_matches = re.findall(granite_pattern, text)
        for func_name, args_str in granite_matches:
            try:
                # Convert single quotes to double quotes for JSON parsing
                args_json = args_str.replace("'", '"')
                args = json.loads(args_json)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                })
            except json.JSONDecodeError:
                # Try alternative parsing
                try:
                    args = ast.literal_eval(args_str)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args)
                        }
                    })
                except:
                    continue
        
        # Format 1b: Alternative Granite format with dict structure
        granite_dict_pattern = r'\{\s*["\']name["\']\s*:\s*["\']([^"\']+)["\'].*?["\']arguments["\']\s*:\s*(\{[^}]*\})\s*\}'
        granite_dict_matches = re.findall(granite_dict_pattern, text, re.DOTALL)
        for func_name, args_str in granite_dict_matches:
            try:
                args_json = args_str.replace("'", '"')
                args = json.loads(args_json)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                })
            except:
                continue
        
        # Format 2: <|tool|>function_name</|tool|> <|param:name|>value</param>
        tool_pattern = r'<\|tool\|>([^<]+)</\|tool\|>'
        param_pattern = r'<\|param:([^|]+)\|>([^<]*)</param>'
        
        tool_matches = re.findall(tool_pattern, text)
        if tool_matches:
            for tool_name in tool_matches:
                # Extract parameters for this tool
                params = {}
                param_matches = re.findall(param_pattern, text)
                for param_name, param_value in param_matches:
                    params[param_name] = param_value
                
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_name.strip(),
                        "arguments": json.dumps(params)
                    }
                })
        
        # Format 3: <function_call>{"name": "func", "arguments": {...}}</function_call>
        function_call_pattern = r'<function_call>(\{[^}]*\})</function_call>'
        function_matches = re.findall(function_call_pattern, text, re.DOTALL)
        for match in function_matches:
            try:
                call_data = json.loads(match)
                if "name" in call_data:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": json.dumps(call_data.get("arguments", {}))
                        }
                    })
            except json.JSONDecodeError:
                continue
        
        # Format 4: ```json {"function_call": {"name": "func", "arguments": {...}}} ```
        json_block_pattern = r'```json\s*(\{[^`]*\})\s*```'
        json_matches = re.findall(json_block_pattern, text, re.DOTALL)
        for match in json_matches:
            try:
                json_data = json.loads(match)
                if "function_call" in json_data:
                    call_data = json_data["function_call"]
                    if "name" in call_data:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": call_data["name"],
                                "arguments": json.dumps(call_data.get("arguments", {}))
                            }
                        })
            except json.JSONDecodeError:
                continue
        
        # Format 5: ```python function_name(param="value") ``` (NEW GRANITE FORMAT)
        python_block_pattern = r'```python\s*([^`]*)\s*```'
        python_matches = re.findall(python_block_pattern, text, re.DOTALL)
        for match in python_matches:
            # Parse Python function calls like: stdio.describe_table(table_name="users")
            python_call_pattern = r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(([^)]*)\)'
            call_matches = re.findall(python_call_pattern, match)
            for func_name, params_str in call_matches:
                try:
                    # Parse parameters from Python function call
                    params = {}
                    if params_str.strip():
                        # Handle simple parameter parsing: param="value", param2=123
                        param_pattern = r'(\w+)\s*=\s*(["\']?)([^,]*?)\2(?:,|$)'
                        param_matches = re.findall(param_pattern, params_str)
                        for param_name, quote, param_value in param_matches:
                            # Convert to appropriate type
                            if quote:  # String parameter
                                params[param_name] = param_value
                            elif param_value.isdigit():  # Integer
                                params[param_name] = int(param_value)
                            elif param_value.lower() in ['true', 'false']:  # Boolean
                                params[param_name] = param_value.lower() == 'true'
                            else:  # Default to string
                                params[param_name] = param_value
                    
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(params)
                        }
                    })
                except Exception as e:
                    log.debug(f"Error parsing Python function call: {e}")
                    continue
        
        # Format 6: {"api": "namespace", "function": "name", "params": {...}} (NEW GRANITE FORMAT)
        api_function_pattern = r'\{\s*"api"\s*:\s*"([^"]+)"\s*,\s*"function"\s*:\s*"([^"]+)"\s*,\s*"params"\s*:\s*(\{[^}]*\})\s*\}'
        api_matches = re.findall(api_function_pattern, text, re.DOTALL)
        for api_name, function_name, params_str in api_matches:
            try:
                params = json.loads(params_str)
                # Reconstruct function name from api.function format
                full_name = f"{api_name}.{function_name}" if api_name != function_name else function_name
                
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": full_name,
                        "arguments": json.dumps(params)
                    }
                })
            except json.JSONDecodeError:
                continue
        
        # Format 7: <|tool|>function_name<|file_sep|> <|param|> {...} (NEW GRANITE FORMAT)
        file_sep_pattern = r'<\|tool\|>([^<]+)<\|file_sep\|>\s*<\|param\|>\s*(\{[^}]*\})'
        file_sep_matches = re.findall(file_sep_pattern, text, re.DOTALL)
        for tool_name, params_str in file_sep_matches:
            try:
                params = json.loads(params_str)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_name.strip(),
                        "arguments": json.dumps(params)
                    }
                })
            except json.JSONDecodeError:
                continue
        
        # Format 8: Direct JSON function calls (look for JSON-like structures)
        json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
        direct_json_matches = re.findall(json_pattern, text)
        for match in direct_json_matches:
            try:
                call_data = json.loads(match)
                if "name" in call_data:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": json.dumps(call_data.get("arguments", {}))
                        }
                    })
            except json.JSONDecodeError:
                continue
        
        # Log successful parsing
        if tool_calls:
            log.debug(f"Parsed {len(tool_calls)} Granite/WatsonX tool calls from text format")
            for tc in tool_calls:
                log.debug(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
        
    except Exception as e:
        log.debug(f"Error parsing Granite/WatsonX tool formats: {e}")
    
    return tool_calls


def _parse_watsonx_response(resp) -> Dict[str, Any]:  # noqa: D401 - small helper
    """
    Convert Watson X response → standard `{response, tool_calls}` dict.
    
    ENHANCED: Now handles WatsonX-specific tool calling formats:
    - <|tool|>function_name</|tool|> <|param:name|>value</param>
    - <function_call>{"name": "func", "arguments": {...}}</function_call>
    - JSON-style function calls in text
    - Python code blocks with function calls
    - API function JSON structures
    - Standard OpenAI-style tool calls
    """
    tool_calls: List[Dict[str, Any]] = []
    
    # Handle Watson X response format - check choices first
    if hasattr(resp, 'choices') and resp.choices:
        choice = resp.choices[0]
        message = _safe_get(choice, 'message', {})
        
        # Check for standard tool calls in Watson X format
        if _safe_get(message, 'tool_calls'):
            for tc in message['tool_calls']:
                tool_calls.append({
                    "id": _safe_get(tc, "id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": _safe_get(tc, "function", {}).get("name"),
                        "arguments": _safe_get(tc, "function", {}).get("arguments", "{}"),
                    },
                })
        
        if tool_calls:
            return {"response": None, "tool_calls": tool_calls}
        
        # Extract text content for further parsing
        content = _safe_get(message, "content", "")
        if isinstance(content, list) and content:
            content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        
        # ENHANCED: Parse WatsonX-specific tool calling formats from text content
        if content and isinstance(content, str):
            parsed_tool_calls = _parse_watsonx_tool_formats(content)
            if parsed_tool_calls:
                return {"response": None, "tool_calls": parsed_tool_calls}
        
        return {"response": content, "tool_calls": []}
    
    # Fallback: try direct dictionary access
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            message = choice.get("message", {})
            
            # Check for standard tool calls
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        },
                    })
                
                if tool_calls:
                    return {"response": None, "tool_calls": tool_calls}
            
            # Extract text content and parse WatsonX formats
            content = message.get("content", "")
            if content:
                parsed_tool_calls = _parse_watsonx_tool_formats(content)
                if parsed_tool_calls:
                    return {"response": None, "tool_calls": parsed_tool_calls}
            
            return {"response": content, "tool_calls": []}
    
    # Fallback for other response formats
    if hasattr(resp, 'results') and resp.results:
        result = resp.results[0]
        text = _safe_get(result, 'generated_text', '') or _safe_get(result, 'text', '')
        
        # Try to parse WatsonX tool formats from generated text
        if text:
            parsed_tool_calls = _parse_watsonx_tool_formats(text)
            if parsed_tool_calls:
                return {"response": None, "tool_calls": parsed_tool_calls}
        
        return {"response": text, "tool_calls": []}
    
    # Final fallback - try to parse as string
    text_content = str(resp)
    parsed_tool_calls = _parse_watsonx_tool_formats(text_content)
    if parsed_tool_calls:
        return {"response": None, "tool_calls": parsed_tool_calls}
    
    return {"response": text_content, "tool_calls": []}


# ─────────────────────────── client ───────────────────────────


class WatsonXLLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, OpenAIStyleMixin, BaseLLMClient):
    """
    Configuration-aware adapter around the *ibm-watsonx-ai* SDK that gets
    all capabilities from unified YAML configuration.
    
    CRITICAL UPDATE: Now uses universal ToolCompatibilityMixin for consistent
    tool name handling across all providers with enterprise-grade sanitization
    and comprehensive IBM Granite model tool format parsing.
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-3-8b-instruct",
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        watsonx_ai_url: Optional[str] = None,
        space_id: Optional[str] = None,
    ) -> None:
        # CRITICAL UPDATE: Initialize ALL mixins including ToolCompatibilityMixin
        ConfigAwareProviderMixin.__init__(self, "watsonx", model)
        ToolCompatibilityMixin.__init__(self, "watsonx")
        
        self.model = model
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.space_id = space_id or os.getenv("WATSONX_SPACE_ID")
        self.watsonx_ai_url = watsonx_ai_url or os.getenv("WATSONX_AI_URL", "https://us-south.ml.cloud.ibm.com")
        
        # Set up credentials
        credentials = Credentials(
            url=self.watsonx_ai_url,
            api_key=api_key or os.getenv("WATSONX_API_KEY") or os.getenv("IBM_CLOUD_API_KEY")
        )
        
        self.client = APIClient(credentials)
        
        # Default parameters - can be overridden by configuration
        self.default_params = {
            "time_limit": 10000,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 1.0,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with WatsonX-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add tool compatibility info from universal system
        tool_compatibility = self.get_tool_compatibility_info()
        
        # Add WatsonX-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "watsonx_specific": {
                    "project_id": self.project_id,
                    "space_id": self.space_id,
                    "watsonx_ai_url": self.watsonx_ai_url,
                    "model_family": self._detect_model_family(),
                    "enterprise_features": True,
                    "granite_parsing": True,  # NEW: Indicates enhanced Granite parsing
                },
                # Universal tool compatibility info
                **tool_compatibility,
                "parameter_mapping": {
                    "temperature": "temperature",
                    "max_tokens": "max_tokens",
                    "top_p": "top_p",
                    "stream": "stream",
                    "time_limit": "time_limit"
                },
                "watsonx_parameters": [
                    "time_limit", "include_stop_sequence", "return_options"
                ],
                "granite_tool_formats": [
                    "pipe_format", "function_call_xml", "json_blocks", 
                    "python_code", "api_json", "file_separator", "direct_json"
                ]
            })
        
        return info

    def _detect_model_family(self) -> str:
        """Detect model family for WatsonX-specific optimizations"""
        model_lower = self.model.lower()
        if "llama" in model_lower:
            return "llama"
        elif "granite" in model_lower:
            return "granite"
        elif "mistral" in model_lower:
            return "mistral"
        elif "codellama" in model_lower:
            return "codellama"
        else:
            return "unknown"

    def _get_model_inference(self, params: Optional[Dict[str, Any]] = None) -> ModelInference:
        """Create a ModelInference instance with configuration-aware parameters."""
        # Start with defaults and apply configuration limits
        merged_params = {**self.default_params}
        if params:
            merged_params.update(params)
        
        # Apply configuration-based parameter validation
        validated_params = self.validate_parameters(**merged_params)
        
        return ModelInference(
            model_id=self.model,
            api_client=self.client,
            params=validated_params,
            project_id=self.project_id,
            space_id=self.space_id,
            verify=False
        )

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
            any(isinstance(item, dict) and item.get("type") in ["image", "image_url"] for item in msg.get("content", []))
            for msg in messages
        )
        if has_vision and not self.supports_feature("vision"):
            log.warning(f"Vision content detected but {self.model} doesn't support vision according to configuration")
        
        # Check multimodal support
        has_multimodal = any(
            isinstance(msg.get("content"), list)
            for msg in messages
        )
        if has_multimodal and not self.supports_feature("multimodal"):
            log.info(f"Multimodal content will be simplified - {self.model} has limited multimodal support")
        
        # Validate parameters using configuration
        validated_kwargs = self.validate_parameters(**validated_kwargs)
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

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
                        log.debug(f"Sanitized tool name in WatsonX conversation: {original_name} -> {sanitized_name}")
                    
                    sanitized_tool_calls.append(tc_copy)
                
                prepared_msg["tool_calls"] = sanitized_tool_calls
                prepared_messages.append(prepared_msg)
            else:
                prepared_messages.append(msg)
        
        return prepared_messages

    # ── tool schema helpers ─────────────────────────────────

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tools to Watson X format.
        
        Note: Tool names should already be sanitized by universal ToolCompatibilityMixin
        before reaching this method.
        """
        if not tools:
            return []

        converted: List[Dict[str, Any]] = []
        for entry in tools:
            fn = entry.get("function", entry)
            try:
                converted.append({
                    "type": "function",
                    "function": {
                        "name": fn["name"],  # Should already be sanitized by ToolCompatibilityMixin
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or fn.get("input_schema") or {},
                    }
                })
            except Exception as exc:  # pragma: no cover - permissive fallback
                log.debug("Tool schema error (%s) - using permissive schema", exc)
                converted.append({
                    "type": "function",
                    "function": {
                        "name": fn.get("name", f"tool_{uuid.uuid4().hex[:6]}"),
                        "description": fn.get("description", ""),
                        "parameters": {"type": "object", "additionalProperties": True},
                    }
                })
        return converted

    def _format_messages_for_watsonx(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Format messages for Watson X API with Granite chat template support.
        
        ENHANCED: Now uses proper Granite chat template format when tools are provided.
        """
        # If tools are provided and model supports tools, use Granite template format
        if tools and self.supports_feature("tools") and self._detect_model_family() == "granite":
            return self._format_granite_chat_template(messages, tools)
        
        # Fallback to standard WatsonX format
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                # Check if system messages are supported
                if self.supports_feature("system_messages"):
                    formatted.append({
                        "role": "system",
                        "content": content
                    })
                else:
                    # Fallback: convert to user message
                    log.warning(f"System messages not supported by {self.model}, converting to user message")
                    formatted.append({
                        "role": "user",
                        "content": f"System: {content}"
                    })
            elif role == "user":
                if isinstance(content, str):
                    formatted.append({
                        "role": "user",
                        "content": [{"type": "text", "text": content}]
                    })
                elif isinstance(content, list):
                    # Handle multimodal content for Watson X with configuration awareness
                    has_images = any(item.get("type") in ["image", "image_url"] for item in content)
                    
                    if has_images and not self.supports_feature("vision"):
                        # Extract only text content
                        text_content = " ".join([
                            item.get("text", "") for item in content 
                            if item.get("type") == "text"
                        ])
                        formatted.append({
                            "role": "user",
                            "content": [{"type": "text", "text": text_content or "[Image content removed - not supported by model]"}]
                        })
                    else:
                        # Process multimodal content normally
                        watsonx_content = []
                        for item in content:
                            if item.get("type") == "text":
                                watsonx_content.append({
                                    "type": "text",
                                    "text": item.get("text", "")
                                })
                            elif item.get("type") == "image":
                                # Convert image format for Watson X
                                source = item.get("source", {})
                                if source.get("type") == "base64":
                                    # Watson X expects image_url format
                                    data_url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                    watsonx_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": data_url
                                        }
                                    })
                            elif item.get("type") == "image_url":
                                # Pass through image_url format
                                watsonx_content.append(item)
                        
                        formatted.append({
                            "role": "user",
                            "content": watsonx_content
                        })
                else:
                    formatted.append({
                        "role": "user",
                        "content": content
                    })
            elif role == "assistant":
                if msg.get("tool_calls"):
                    # Check if tools are supported
                    if self.supports_feature("tools"):
                        formatted.append({
                            "role": "assistant",
                            "tool_calls": msg["tool_calls"]
                        })
                    else:
                        # Convert tool calls to text response
                        log.warning(f"Tool calls detected but {self.model} doesn't support tools according to configuration")
                        tool_text = f"{content or ''}\n\nNote: Tool calls were requested but not supported by this model."
                        formatted.append({
                            "role": "assistant",
                            "content": tool_text
                        })
                else:
                    formatted.append({
                        "role": "assistant",
                        "content": content
                    })
            elif role == "tool":
                # Tool response messages - only include if tools are supported
                if self.supports_feature("tools"):
                    formatted.append({
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id"),
                        "content": content
                    })
                else:
                    # Convert tool response to user message
                    formatted.append({
                        "role": "user",
                        "content": f"Tool result: {content or ''}"
                    })

        return formatted

    def _format_granite_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format messages using Granite's specific chat template format.
        
        Granite expects:
        1. available_tools role with tool definitions
        2. Standard system/user/assistant roles
        3. Specific response format for tool calls
        """
        formatted = []
        
        # Add available_tools role first (Granite requirement)
        if tools:
            tools_content = ""
            for tool in tools:
                tools_content += json.dumps(tool, indent=2) + "\n\n"
            
            formatted.append({
                "role": "available_tools",
                "content": tools_content.strip()
            })
        
        # Add the conversation messages
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                # Granite system message with tool instruction
                system_content = content
                if tools:
                    system_content += " Use the available functions when needed to provide accurate information."
                
                formatted.append({
                    "role": "system",
                    "content": system_content
                })
            elif role in ["user", "assistant"]:
                formatted.append({
                    "role": role,
                    "content": content
                })
            elif role == "tool":
                # Granite expects tool_response role
                formatted.append({
                    "role": "tool_response", 
                    "content": content
                })
        
        return formatted

    # ── main entrypoint with universal tool compatibility ─────────────────────────────────────

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Configuration-aware completion generation with streaming support and universal tool compatibility.
        
        • stream=False → returns awaitable that resolves to standardised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        
        CRITICAL UPDATE: Now uses universal ToolCompatibilityMixin for enterprise-grade
        tool name sanitization with bidirectional mapping and enhanced Granite parsing.
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **extra
        )
        
        # Apply max_tokens if provided
        if max_tokens:
            validated_kwargs["max_tokens"] = max_tokens
        
        # CRITICAL UPDATE: Use universal tool name sanitization (stores mapping for restoration)
        name_mapping = {}
        if validated_tools:
            validated_tools = self._sanitize_tool_names(validated_tools)
            name_mapping = self._current_name_mapping
            log.debug(f"Tool sanitization: {len(name_mapping)} tools processed for WatsonX enterprise compatibility")
        
        # CRITICAL UPDATE: Prepare messages for conversation (sanitize tool names in history)
        if name_mapping:
            validated_messages = self._prepare_messages_for_conversation(validated_messages)
        
        # Convert to WatsonX format
        watsonx_tools = self._convert_tools(validated_tools)
        
        # Format messages with configuration-aware processing and Granite template support
        formatted_messages = self._format_messages_for_watsonx(validated_messages, watsonx_tools)

        log.debug(f"Watson X payload: model={self.model}, messages={len(formatted_messages)}, tools={len(watsonx_tools)}")

        # --- streaming: use Watson X streaming -------------------------
        if validated_stream:
            return self._stream_completion_async(formatted_messages, watsonx_tools, name_mapping, validated_kwargs)

        # --- non-streaming: use regular completion ----------------------
        return self._regular_completion(formatted_messages, watsonx_tools, name_mapping, validated_kwargs)

    async def _stream_completion_async(
        self, 
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        name_mapping: Dict[str, str] = None,
        params: Dict[str, Any] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Streaming using Watson X with configuration awareness and universal tool name restoration.
        """
        try:
            log.debug(f"Starting Watson X streaming for model: {self.model}")
            
            model = self._get_model_inference(params)
            
            # Use Watson X streaming (only use tools if supported)
            if tools and self.supports_feature("tools"):
                # For tool calling, we need to use chat_stream with tools
                stream_response = model.chat_stream(messages=messages, tools=tools)
            else:
                # For regular chat, use chat_stream
                stream_response = model.chat_stream(messages=messages)
            
            chunk_count = 0
            for chunk in stream_response:
                chunk_count += 1
                
                if isinstance(chunk, str):
                    # ENHANCED: Parse WatsonX tool formats from string chunks
                    parsed_tool_calls = _parse_watsonx_tool_formats(chunk)
                    if parsed_tool_calls:
                        # Restore tool names if needed
                        chunk_response = {
                            "response": "",
                            "tool_calls": parsed_tool_calls
                        }
                        if name_mapping:
                            chunk_response = self._restore_tool_names_in_response(
                                chunk_response, 
                                name_mapping
                            )
                        yield chunk_response
                    else:
                        yield {
                            "response": chunk,
                            "tool_calls": []
                        }
                elif isinstance(chunk, dict):
                    # Handle structured chunk responses
                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        
                        content = delta.get("content", "")
                        tool_calls = delta.get("tool_calls", []) if self.supports_feature("tools") else []
                        
                        # Create chunk response
                        chunk_response = {
                            "response": content,
                            "tool_calls": tool_calls
                        }
                        
                        # CRITICAL UPDATE: Restore tool names using universal restoration
                        if name_mapping and tool_calls:
                            chunk_response = self._restore_tool_names_in_response(
                                chunk_response, 
                                name_mapping
                            )
                        
                        yield chunk_response
                    else:
                        # ENHANCED: Parse WatsonX tool formats from streaming text
                        parsed_tool_calls = _parse_watsonx_tool_formats(str(chunk))
                        if parsed_tool_calls:
                            # Restore tool names if needed
                            chunk_response = {
                                "response": "",
                                "tool_calls": parsed_tool_calls
                            }
                            if name_mapping:
                                chunk_response = self._restore_tool_names_in_response(
                                    chunk_response, 
                                    name_mapping
                                )
                            yield chunk_response
                        else:
                            yield {
                                "response": str(chunk),
                                "tool_calls": []
                            }
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    import asyncio
                    await asyncio.sleep(0)
            
            log.debug(f"Watson X streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            log.error(f"Error in Watson X streaming: {e}")
            
            # Check if it's a tool name validation error
            error_str = str(e).lower()
            if "function" in error_str and ("name" in error_str or "invalid" in error_str):
                log.error(f"Tool name sanitization may have failed: {e}")
                log.error(f"Current mapping: {name_mapping}")
            
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
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Non-streaming completion using Watson X with configuration awareness and universal tool name restoration.
        """
        try:
            log.debug(f"Starting Watson X completion for model: {self.model}")
            
            model = self._get_model_inference(params)
            
            # Use tools only if supported
            if tools and self.supports_feature("tools"):
                # Use chat with tools
                resp = model.chat(messages=messages, tools=tools)
            else:
                # Use regular chat
                resp = model.chat(messages=messages)
            
            # Parse response with enhanced Granite format support
            result = _parse_watsonx_response(resp)
            
            # CRITICAL UPDATE: Restore original tool names using universal restoration
            if name_mapping and result.get("tool_calls"):
                result = self._restore_tool_names_in_response(result, name_mapping)
            
            log.debug(f"Watson X completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
            
        except Exception as e:
            log.error(f"Error in Watson X completion: {e}")
            
            # Check if it's a tool name validation error
            error_str = str(e).lower()
            if "function" in error_str and ("name" in error_str or "invalid" in error_str):
                log.error(f"Tool name sanitization may have failed: {e}")
                log.error(f"Current mapping: {name_mapping}")
            
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
        # Watson X client cleanup if needed
        pass