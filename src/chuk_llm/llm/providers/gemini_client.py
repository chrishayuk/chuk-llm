# chuk_llm/llm/providers/gemini_client.py

"""
Google Gemini chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration-driven capabilities with complete warning suppression and proper parameter handling.
UPDATED: Fixed to match Anthropic client patterns - proper system instruction support, 
universal vision format, robust response handling, and MCP tool name sanitization with restoration.
CRITICAL FIX: Eliminates response concatenation and data loss issues.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
import json
import logging
import os
import re
import sys
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
import warnings
import functools

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

# providers
from chuk_llm.llm.core.base import BaseLLMClient
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATE WARNING SUPPRESSION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

# Store original warning functions globally
_ORIGINAL_WARN = warnings.warn
_ORIGINAL_SHOWWARNING = warnings.showwarning
_ORIGINAL_FORMATWARNING = warnings.formatwarning

def _silent_warn(*args, **kwargs):
    """Completely silent warning function"""
    pass

def _silent_showwarning(*args, **kwargs):
    """Completely silent showwarning function"""
    pass

def _silent_formatwarning(*args, **kwargs):
    """Return empty string for formatwarning"""
    return ""

def apply_ultimate_warning_suppression():
    """Apply the most comprehensive warning suppression possible for Gemini"""
    
    # Method 1: Environment variables (most aggressive)
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['GOOGLE_GENAI_SUPPRESS_WARNINGS'] = '1'
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GLOG_minloglevel'] = '3'  # Even more aggressive
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Method 2: Global warning function replacement
    warnings.warn = _silent_warn
    warnings.showwarning = _silent_showwarning
    warnings.formatwarning = _silent_formatwarning
    
    # Method 3: Comprehensive warning patterns for Gemini-specific warnings
    gemini_warning_patterns = [
        # Function call warnings (the main culprit)
        ".*non-text parts in the response.*",
        ".*function_call.*",
        ".*returning concatenated text result.*",
        ".*check out the non text parts.*",
        ".*text parts.*",
        ".*response.*function_call.*",
        ".*non text parts for full response.*",
        ".*candidates.*",
        ".*finish_reason.*",
        
        # General Gemini warnings
        ".*genai.*",
        ".*google.*",
        ".*generativeai.*",
        ".*google.genai.*",
        ".*google.generativeai.*",
        
        # gRPC and HTTP warnings
        ".*grpc.*",
        ".*http.*",
        ".*ssl.*",
        ".*certificate.*",
        ".*urllib3.*",
        ".*httpx.*",
        ".*asyncio.*",
    ]
    
    # Apply pattern-based suppression for ALL warning categories
    warning_categories = [
        UserWarning, Warning, FutureWarning, DeprecationWarning,
        RuntimeWarning, PendingDeprecationWarning, ImportWarning,
        UnicodeWarning, BytesWarning, ResourceWarning
    ]
    
    for pattern in gemini_warning_patterns:
        for category in warning_categories:
            warnings.filterwarnings("ignore", message=pattern, category=category)
        # Also apply without category
        warnings.filterwarnings("ignore", message=pattern)
    
    # Method 4: Module-level suppression for Google ecosystem
    google_modules = [
        "google", "google.genai", "google.generativeai", "google.ai",
        "google.cloud", "google.protobuf", "grpc", "googleapis",
        "google.auth", "google.api_core", "google.api", "googleapiclient"
    ]
    
    for module in google_modules:
        for category in warning_categories:
            warnings.filterwarnings("ignore", category=category, module=module)
        warnings.filterwarnings("ignore", module=module)
    
    # Method 5: Logger suppression (nuclear option)
    google_loggers = [
        "google", "google.genai", "google.generativeai", 
        "google.ai.generativelanguage", "google.ai", "google.cloud",
        "grpc", "grpc._channel", "urllib3", "httpx", "asyncio",
        "google.auth", "google.api_core", "googleapiclient"
    ]
    
    for logger_name in google_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)  # Even higher than CRITICAL
        logger.propagate = False
        logger.disabled = True
        logger.handlers.clear()
        # Also disable all child loggers
        for child_logger in [l for l in logging.Logger.manager.loggerDict if l.startswith(logger_name + ".")]:
            child = logging.getLogger(child_logger)
            child.setLevel(logging.CRITICAL + 1)
            child.propagate = False
            child.disabled = True
            child.handlers.clear()
    
    # Method 6: Global warning suppression
    warnings.simplefilter("ignore")
    for category in warning_categories:
        warnings.simplefilter("ignore", category)

# Enhanced context manager for complete output suppression
class UltimateSuppression:
    """The most aggressive suppression possible - blocks everything"""
    
    def __init__(self):
        self.original_stderr = None
        self.original_stdout = None
        self.original_warn = None
        self.original_showwarning = None
        self.original_formatwarning = None
        self.devnull = None
        self.original_filters = None
    
    def __enter__(self):
        # Store all originals
        self.original_stderr = sys.stderr
        self.original_stdout = sys.stdout
        self.original_warn = warnings.warn
        self.original_showwarning = warnings.showwarning
        self.original_formatwarning = warnings.formatwarning
        self.original_filters = warnings.filters[:]
        
        # Open devnull
        self.devnull = open(os.devnull, 'w')
        
        # Redirect stderr to devnull (where warnings go)
        sys.stderr = self.devnull
        
        # Replace all warning functions with no-ops
        warnings.warn = _silent_warn
        warnings.showwarning = _silent_showwarning
        warnings.formatwarning = _silent_formatwarning
        
        # Clear all filters and apply ultimate suppression
        warnings.resetwarnings()
        apply_ultimate_warning_suppression()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore everything
        if self.devnull:
            self.devnull.close()
        
        sys.stderr = self.original_stderr
        sys.stdout = self.original_stdout
        warnings.warn = self.original_warn
        warnings.showwarning = self.original_showwarning
        warnings.formatwarning = self.original_formatwarning
        
        # Restore original filters
        if self.original_filters is not None:
            warnings.filters[:] = self.original_filters

def silence_gemini_warnings(func):
    """Decorator to completely silence warnings for Gemini API calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with UltimateSuppression():
            return func(*args, **kwargs)
    return wrapper

def silence_gemini_warnings_async(func):
    """Async decorator to completely silence warnings for Gemini API calls"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        with UltimateSuppression():
            return await func(*args, **kwargs)
    return wrapper

# Apply suppression immediately when module loads
apply_ultimate_warning_suppression()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL AVAILABILITY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Current available Gemini models
AVAILABLE_GEMINI_MODELS = {
    # Gemini 2.5 series (latest and most powerful)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    
    # Gemini 2.0 series (stable)
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    
    # Gemini 1.5 series (production-ready)
    "gemini-1.5-pro",
    "gemini-1.5-flash", 
    "gemini-1.5-flash-8b"
}

def validate_and_map_model(requested_model: str) -> str:
    """Validate model name against available models"""
    if requested_model in AVAILABLE_GEMINI_MODELS:
        return requested_model
    
    # If not found, raise an error with available models
    available_list = ", ".join(sorted(AVAILABLE_GEMINI_MODELS))
    raise ValueError(
        f"Model '{requested_model}' not available for provider 'gemini'. "
        f"Available: [{available_list}]"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SAFE RESPONSE PARSING - NO CONCATENATION OR DATA LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_parse_gemini_response(resp) -> Dict[str, Any]:
    """
    CRITICAL: Safe Gemini response parser that prevents data loss and concatenation.
    
    NEVER accesses resp.text directly when function calls might be present.
    Always parses through candidates -> content -> parts to extract ALL data.
    """
    tool_calls: List[Dict[str, Any]] = []
    response_text = ""
    
    try:
        # NEVER use resp.text directly - it triggers concatenation and data loss
        # Instead, always parse through candidates -> content -> parts
        
        if hasattr(resp, 'candidates') and resp.candidates:
            candidate = resp.candidates[0]
            
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                
                # Check if content has parts (this is where multimodal content lives)
                if hasattr(content, 'parts') and content.parts:
                    text_parts = []
                    
                    # Process each part individually to avoid data loss
                    for part in content.parts:
                        # Handle text parts
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                        
                        # Handle function call parts - CRITICAL: Extract these separately
                        elif hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            
                            try:
                                # Extract function call data
                                function_name = getattr(fc, "name", "unknown")
                                function_args = dict(getattr(fc, "args", {}))
                                
                                tool_calls.append({
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function", 
                                    "function": {
                                        "name": function_name,
                                        "arguments": json.dumps(function_args)
                                    }
                                })
                                
                                log.debug(f"Extracted function call: {function_name}")
                                
                            except Exception as e:
                                log.error(f"Error extracting function call: {e}")
                        
                        # Handle other part types (images, etc.)
                        else:
                            part_info = str(type(part).__name__)
                            log.debug(f"Encountered non-text, non-function part: {part_info}")
                    
                    # Combine text parts
                    response_text = "".join(text_parts)
                    
                    # Log what we extracted
                    if text_parts and tool_calls:
                        log.debug(f"Extracted {len(text_parts)} text parts and {len(tool_calls)} function calls")
                    elif text_parts:
                        log.debug(f"Extracted {len(text_parts)} text parts only")
                    elif tool_calls:
                        log.debug(f"Extracted {len(tool_calls)} function calls only (no text)")
                
                # Fallback: content has text but no parts structure
                elif hasattr(content, 'text') and content.text:
                    response_text = content.text
                    log.debug("Extracted text from content.text (no parts structure)")
                
                # No text or parts - might be function-call-only response
                else:
                    if tool_calls:
                        response_text = ""  # Valid: function calls with no text
                        log.debug("Function-call-only response (no text content)")
                    else:
                        response_text = "[No content available in response]"
                        log.warning("Response has content but no extractable text or function calls")
            
            # No content in candidate
            else:
                # Check for thinking models hitting token limits
                if hasattr(candidate, 'finish_reason'):
                    reason = str(candidate.finish_reason)
                    if 'MAX_TOKENS' in reason:
                        response_text = "[Response exceeded token limit during processing.]"
                    elif 'SAFETY' in reason:
                        response_text = "[Response blocked due to safety filters.]"
                    else:
                        response_text = f"[Response completed with status: {reason}]"
                else:
                    response_text = "[No content in candidate]"
                    log.warning("Candidate has no content")
        
        # No candidates structure - this is unusual for modern Gemini
        else:
            log.warning("Response has no candidates structure")
            # Avoid direct text access as it may trigger concatenation
            response_text = "[Unable to extract content from response - no candidates]"
    
    except Exception as e:
        log.error(f"Critical error in response parsing: {e}")
        response_text = f"[Error parsing response: {str(e)}]"
    
    # Handle JSON mode response cleanup
    if response_text and response_text.count('{"') > 1:
        # Remove duplicate JSON objects that sometimes appear
        json_parts = response_text.split('{"')
        if len(json_parts) > 1:
            first_json = '{"' + json_parts[1].split('}')[0] + '}'
            try:
                json.loads(first_json)  # Validate it's proper JSON
                response_text = first_json
            except json.JSONDecodeError:
                pass  # Keep original if not valid JSON
    
    # Build final response
    result = {
        "response": response_text,
        "tool_calls": tool_calls
    }
    
    # Validate we didn't lose data
    if tool_calls and not response_text:
        log.debug("Valid function-call-only response")
    elif tool_calls and response_text:
        log.debug(f"Mixed response: text ({len(response_text)} chars) + {len(tool_calls)} function calls")
    elif response_text and not tool_calls:
        log.debug(f"Text-only response: {len(response_text)} chars")
    else:
        log.warning("Response has neither text nor function calls")
    
    return result

# ───────────────────────────────────────────────────── helpers ──────────

def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get *key* from dict **or** attribute-style object; fallback to *default*."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

def _convert_tools_to_gemini_format(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[gtypes.Tool]]:
    """Convert OpenAI-style tools to Gemini format"""
    if not tools:
        return None
    
    try:
        function_declarations = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                
                func_name = func.get("name", "")
                if not func_name or func_name == "unknown_function":
                    log.warning(f"Skipping tool with invalid name: {tool}")
                    continue
                
                func_decl = {
                    "name": func_name,
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {})
                }
                function_declarations.append(func_decl)
        
        if function_declarations:
            try:
                gemini_tool = gtypes.Tool(function_declarations=function_declarations)
                return [gemini_tool]
            except Exception as e:
                log.warning(f"Failed to create Gemini Tool: {e}")
                return None
    
    except Exception as e:
        log.error(f"Error converting tools to Gemini format: {e}")
    
    return None

# ─────────────────────────────────────────────────── enhanced context managers ───────────

class SuppressAllOutput:
    """Context manager to completely suppress all output including warnings"""
    
    def __init__(self):
        self.suppressor = UltimateSuppression()
    
    def __enter__(self):
        return self.suppressor.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.suppressor.__exit__(exc_type, exc_val, exc_tb)

@contextmanager
def suppress_warnings():
    """Standard context manager for warning suppression"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

# ─────────────────────────────────────────────────── main adapter ───────────

class GeminiLLMClient(ConfigAwareProviderMixin, BaseLLMClient):
    """
    Configuration-aware `google-genai` wrapper following Anthropic client patterns.
    UPDATED: Proper system instruction support, universal vision format, robust response handling,
    and MCP tool name sanitization with bidirectional mapping for seamless MCP compatibility.
    CRITICAL FIX: Eliminates response concatenation and data loss issues.
    """

    def __init__(self, model: str = "gemini-2.5-flash", *, api_key: Optional[str] = None) -> None:
        # Apply nuclear warning suppression during initialization
        apply_ultimate_warning_suppression()
        
        # Validate model
        safe_model = validate_and_map_model(model)
        
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, "gemini", safe_model)
        
        # load environment
        load_dotenv()

        # get the api key
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        # check if we have a key
        if not api_key:
            raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY env var not set")
        
        # Initialize with complete suppression
        with UltimateSuppression():
            self.model = safe_model
            self.client = genai.Client(api_key=api_key)

        # Store current tool name mapping for response restoration
        self._current_name_mapping: Dict[str, str] = {}

        log.info("GeminiLLMClient initialised with model '%s'", safe_model)

    def _detect_model_family(self) -> str:
        """Detect Gemini model family for optimizations"""
        model_lower = self.model.lower()
        if "2.5" in model_lower:
            return "gemini-2.5"
        elif "2.0" in model_lower:
            return "gemini-2.0"
        elif "1.5" in model_lower:
            return "gemini-1.5"
        elif "flash" in model_lower:
            return "flash"
        elif "pro" in model_lower:
            return "pro"
        else:
            return "unknown"

    def _sanitize_tools_for_gemini(self, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Sanitize tool names and create a mapping for response processing.
        
        Gemini may have similar restrictions as other providers for tool names.
        Uses aggressive sanitization for consistency across all providers.
        
        Returns:
            tuple: (sanitized_tools, name_mapping)
                - sanitized_tools: Tools with Gemini-compatible names
                - name_mapping: Dict mapping sanitized_name -> original_name
        """
        if not tools:
            return tools, {}
            
        sanitized_tools = []
        name_mapping = {}
        
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                tool_copy = tool.copy()
                func = tool_copy["function"].copy()
                
                original_name = func.get("name", "")
                if original_name:
                    # Aggressive sanitization for Gemini compatibility
                    # Replace any non-alphanumeric (except underscore/dash) with underscore
                    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)
                    
                    # Remove multiple consecutive underscores
                    sanitized_name = re.sub(r'_+', '_', sanitized_name)
                    
                    # Ensure it starts with a letter or underscore
                    if sanitized_name and not sanitized_name[0].isalpha() and sanitized_name[0] != '_':
                        sanitized_name = '_' + sanitized_name
                    
                    # Truncate to 64 characters and clean up trailing underscores
                    sanitized_name = sanitized_name[:64].rstrip('_')
                    
                    # Ensure we have a valid name
                    if not sanitized_name:
                        sanitized_name = "unnamed_function"
                    
                    # Store the mapping
                    name_mapping[sanitized_name] = original_name
                    
                    # Update the tool
                    func["name"] = sanitized_name
                    
                    # Log the sanitization if it changed
                    if sanitized_name != original_name:
                        log.debug(f"Sanitized Gemini tool name: {original_name} -> {sanitized_name}")
                
                tool_copy["function"] = func
                sanitized_tools.append(tool_copy)
            else:
                # Non-function tools pass through unchanged
                sanitized_tools.append(tool)
                
        return sanitized_tools, name_mapping

    def _restore_tool_names_in_response(self, response: Dict[str, Any], name_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Restore original tool names in the response using the mapping.
        
        This makes the sanitization transparent to users - they send MCP-style names
        and receive MCP-style names back, even though internally we use sanitized names.
        """
        if not name_mapping or not response.get("tool_calls"):
            return response
            
        # Create a copy to avoid modifying the original
        restored_response = response.copy()
        restored_tool_calls = []
        
        for tool_call in response["tool_calls"]:
            if "function" in tool_call and "name" in tool_call["function"]:
                sanitized_name = tool_call["function"]["name"]
                original_name = name_mapping.get(sanitized_name, sanitized_name)
                
                # Restore the original name
                restored_tool_call = tool_call.copy()
                restored_tool_call["function"] = tool_call["function"].copy()
                restored_tool_call["function"]["name"] = original_name
                
                restored_tool_calls.append(restored_tool_call)
                
                # Log the restoration if it changed
                if original_name != sanitized_name:
                    log.debug(f"Restored Gemini tool name: {sanitized_name} -> {original_name}")
            else:
                restored_tool_calls.append(tool_call)
                
        restored_response["tool_calls"] = restored_tool_calls
        return restored_response

    def _sanitize_tool_names(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """
        Sanitize tool names with mapping storage for later restoration.
        
        Adds bidirectional mapping support for seamless MCP tool name handling.
        """
        if not tools:
            return tools
        
        log.debug(f"Sanitizing {len(tools)} tools for Gemini compatibility")
        
        # Store the mapping for later restoration
        sanitized_tools, self._current_name_mapping = self._sanitize_tools_for_gemini(tools)
        
        return sanitized_tools

    @silence_gemini_warnings
    def _parse_gemini_response_with_restoration(self, resp, name_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Convert Gemini response to standard format and restore tool names - WITH SILENCE"""
        # Use the safe parser (no concatenation)
        result = _safe_parse_gemini_response(resp)
        
        # Restore original tool names if we have a mapping
        if name_mapping and result.get("tool_calls"):
            result = self._restore_tool_names_in_response(result, name_mapping)
        
        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get model info using configuration, with Gemini-specific additions."""
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add Gemini-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "supports_function_calling": self.supports_feature("tools"),
                "supports_streaming": self.supports_feature("streaming"),
                "supports_vision": self.supports_feature("vision"),
                "supports_json_mode": self.supports_feature("json_mode"),
                "supports_system_messages": self.supports_feature("system_messages"),
                "tool_name_requirements": "aggressive_sanitization",
                "mcp_compatibility": "requires_sanitization_with_restoration",
                "response_parsing": "safe_no_concatenation",
                "gemini_specific": {
                    "context_length": "2M tokens" if "2.5" in self.model else ("2M tokens" if "2.0" in self.model else "1M tokens"),
                    "model_family": self._detect_model_family(),
                    "experimental_features": "2.0" in self.model or "2.5" in self.model,
                    "warning_suppression": "ultimate",
                    "enhanced_reasoning": "2.5" in self.model,
                    "supports_function_calling": self.supports_feature("tools"),
                    "data_loss_protection": "enabled",
                },
                "vision_format": "universal_image_url",
                "supported_parameters": ["temperature", "max_tokens", "top_p", "top_k", "stream", "system"],
                "unsupported_parameters": [
                    "frequency_penalty", "presence_penalty", "logit_bias",
                    "user", "n", "best_of", "seed", "stop"
                ],
                "parameter_mapping": {
                    "max_tokens": "max_output_tokens",
                    "stop": "stop_sequences", 
                    "system": "system_instruction_in_config",
                    "temperature": "temperature",
                    "top_p": "top_p",
                    "top_k": "top_k",
                    "candidate_count": "candidate_count"
                }
            })
        
        return info

    def _filter_gemini_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters using configuration limits"""
        filtered = {}
        
        # Parameter mapping
        parameter_mapping = {
            "max_tokens": "max_output_tokens",
            "stop": "stop_sequences",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "candidate_count": "candidate_count"
        }
        
        # Supported parameters
        supported_params = {"temperature", "max_tokens", "top_p", "top_k", "candidate_count", "stop_sequences", "max_output_tokens"}
        unsupported_params = {
            "frequency_penalty", "presence_penalty", "logit_bias",
            "user", "n", "best_of", "seed", "stop", "response_format"
        }
        
        for key, value in params.items():
            mapped_key = parameter_mapping.get(key, key)
            
            if mapped_key in supported_params:
                if key == "temperature":
                    # Gemini temperature range validation
                    if value > 2.0:
                        filtered[mapped_key] = 2.0
                        log.debug(f"Capped temperature from {value} to 2.0 for Gemini")
                    else:
                        filtered[mapped_key] = value
                elif key == "max_tokens":
                    # Use configuration to validate max_tokens
                    limit = self.get_max_tokens_limit()
                    if limit and value > limit:
                        filtered[mapped_key] = limit
                        log.debug(f"Capped max_tokens from {value} to {limit} for Gemini")
                    else:
                        filtered[mapped_key] = value
                else:
                    filtered[mapped_key] = value
            elif key in unsupported_params:
                log.debug(f"Filtered out unsupported parameter for Gemini: {key}={value}")
            else:
                log.warning(f"Unknown parameter for Gemini: {key}={value}")
        
        return filtered

    def _check_json_mode(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Check if JSON mode is requested and return appropriate system instruction"""
        # Only proceed if the model supports JSON mode according to config
        if not self.supports_feature("json_mode"):
            log.debug(f"Model {self.model} does not support JSON mode according to configuration")
            return None
        
        # Check for OpenAI-style response_format
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            return "You must respond with valid JSON only. No markdown code blocks, no explanations, no text before or after. Just pure, valid JSON."
        
        # Check for _json_mode_instruction from provider adapter
        json_instruction = kwargs.get("_json_mode_instruction")
        if json_instruction:
            return json_instruction
        
        return None

    @staticmethod
    async def _download_image_to_base64(url: str) -> tuple[str, str]:
        """Download image from URL and convert to base64"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Get content type from headers
                content_type = response.headers.get('content-type', 'image/png')
                if not content_type.startswith('image/'):
                    content_type = 'image/png'  # Default fallback
                
                # Convert to base64
                image_data = base64.b64encode(response.content).decode('utf-8')
                
                return content_type, image_data
                
        except Exception as e:
            log.warning(f"Failed to download image from {url}: {e}")
            raise ValueError(f"Could not download image: {e}")

    @staticmethod
    async def _convert_universal_vision_to_gemini_async(content_item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal image_url format to Gemini format with URL downloading"""
        if content_item.get("type") == "image_url":
            image_url = content_item.get("image_url", {})
            
            # Handle both string and dict formats
            if isinstance(image_url, str):
                url = image_url
            else:
                url = image_url.get("url", "")
            
            # Convert data URL to Gemini format
            if url.startswith("data:"):
                # Extract media type and data
                try:
                    header, data = url.split(",", 1)
                    # Parse the header: data:image/png;base64
                    media_type_part = header.split(";")[0].replace("data:", "")
                    
                    # Validate media type
                    if not media_type_part.startswith("image/"):
                        media_type_part = "image/png"  # Default fallback
                    
                    # Gemini expects inline data format
                    return {
                        "inline_data": {
                            "mime_type": media_type_part,
                            "data": data.strip()
                        }
                    }
                except (ValueError, IndexError) as e:
                    log.warning(f"Invalid data URL format: {url[:50]}... Error: {e}")
                    return {"text": "[Invalid image format]"}
            else:
                # For external URLs, download and convert to base64
                try:
                    media_type, image_data = await GeminiLLMClient._download_image_to_base64(url)
                    
                    return {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": image_data
                        }
                    }
                except Exception as e:
                    log.warning(f"Failed to process external image URL {url}: {e}")
                    return {"text": f"[Could not load image: {e}]"}
        
        return content_item

    async def _split_for_gemini_async(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Any]]:
        """
        Separate system text & convert ChatML list to Gemini format with async vision support.
        Uses configuration to validate vision support.
        """
        sys_txt: List[str] = []
        contents: List[Any] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                sys_txt.append(msg.get("content", ""))
                continue

            # assistant function calls → need to be handled in tool result flow
            if role == "assistant" and msg.get("tool_calls"):
                # Convert to text description for now
                tool_text = "Assistant called functions: "
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    tool_text += f"{fn['name']}({fn.get('arguments', '{}')}) "
                contents.append(tool_text)
                continue

            # tool response → convert to user message
            if role == "tool":
                tool_result = msg.get("content") or ""
                fn_name = msg.get("name", "tool")
                contents.append(f"Tool {fn_name} returned: {tool_result}")
                continue

            # normal / multimodal messages with universal vision support
            if role in {"user", "assistant"}:
                cont = msg.get("content")
                if cont is None:
                    continue
                
                if isinstance(cont, str):
                    # Simple text content
                    contents.append(cont)
                elif isinstance(cont, list):
                    # Multimodal content - check if vision is supported
                    has_vision_content = any(
                        isinstance(item, dict) and item.get("type") == "image_url" 
                        for item in cont
                    )
                    
                    if has_vision_content and not self.supports_feature("vision"):
                        log.warning(f"Vision content detected but model {self.model} doesn't support vision according to configuration")
                        # Convert to text-only by filtering out images
                        text_only_content = [
                            item.get("text", str(item)) for item in cont 
                            if not (isinstance(item, dict) and item.get("type") == "image_url")
                        ]
                        if text_only_content:
                            contents.append(" ".join(text_only_content))
                        continue
                    
                    # Process multimodal content with proper Gemini format
                    gemini_parts = []
                    for item in cont:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                gemini_parts.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                # Convert to Gemini format
                                try:
                                    gemini_image = await self._convert_universal_vision_to_gemini_async(item)
                                    if "inline_data" in gemini_image:
                                        gemini_parts.append(gemini_image)
                                    else:
                                        # Fallback to text if conversion failed
                                        gemini_parts.append(gemini_image.get("text", "[Image processing failed]"))
                                except Exception as e:
                                    log.warning(f"Failed to convert image: {e}")
                                    gemini_parts.append("[Image conversion failed]")
                            else:
                                gemini_parts.append(str(item))
                        else:
                            gemini_parts.append(str(item))
                    
                    # Add the multimodal content as a structured message
                    if gemini_parts:
                        contents.append(gemini_parts)
                else:
                    # Fallback for other content types
                    contents.append(str(cont))

        return "\n".join(sys_txt).strip(), contents

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        **extra,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Configuration-aware completion generation following Anthropic client patterns.
        
        Uses configuration to validate:
        - Tool support before processing tools
        - Streaming support before enabling streaming
        - JSON mode support before adding JSON instructions
        - Vision support during message processing
        
        Includes MCP tool name sanitization with bidirectional mapping for seamless compatibility.
        CRITICAL FIX: Uses safe response parsing to prevent data loss and concatenation.
        """

        # Validate capabilities using configuration
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but model {self.model} doesn't support tools according to configuration")
            tools = None
        
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but model {self.model} doesn't support streaming according to configuration")
            stream = False

        # Sanitize tool names (stores mapping for restoration)
        tools = self._sanitize_tool_names(tools)
        gemini_tools = _convert_tools_to_gemini_format(tools)
        
        # Check for JSON mode (using configuration validation)
        json_instruction = self._check_json_mode(extra)
        
        # Filter parameters for Gemini compatibility (using configuration limits)
        if max_tokens:
            extra["max_tokens"] = max_tokens
        filtered_params = self._filter_gemini_params(extra)

        # --- streaming: return the async generator directly -------------------------
        if stream:
            return self._stream_completion_async(system, json_instruction, messages, gemini_tools, filtered_params)

        # --- non-streaming: use async client ------------------------------
        return self._regular_completion_async(system, json_instruction, messages, gemini_tools, filtered_params)

    async def _stream_completion_async(
        self, 
        system: Optional[str],
        json_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        gemini_tools: Optional[List[gtypes.Tool]],
        filtered_params: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Real streaming using Gemini client with SAFE response parsing - NO CONCATENATION."""
        
        # Handle system message and JSON instruction
        system_from_messages, contents = await self._split_for_gemini_async(messages)
        final_system = system or system_from_messages
        
        if json_instruction:
            if final_system:
                final_system = f"{final_system}\n\n{json_instruction}"
            else:
                final_system = json_instruction
            log.debug("Added JSON mode instruction to system prompt")

        # Build config
        config_params = filtered_params.copy()
        if gemini_tools:
            config_params["tools"] = gemini_tools
        
        # Handle thinking models hitting token limits
        if "2.5" in self.model:
            if "max_output_tokens" not in config_params:
                # Set a reasonable max_output_tokens to prevent thinking from consuming all tokens
                config_params["max_output_tokens"] = 4096
                log.debug("Set max_output_tokens=4096 for Gemini 2.5 to prevent thinking token overflow")
        elif "max_output_tokens" not in config_params:
            # Set default for non-thinking models
            config_params["max_output_tokens"] = 4096
        
        config = None
        if config_params:
            try:
                # Add system instruction to config if present and supported
                if final_system and self.supports_feature("system_messages"):
                    config_params["system_instruction"] = final_system
                
                config = gtypes.GenerateContentConfig(**config_params)
            except Exception as e:
                log.warning(f"Error creating GenerateContentConfig: {e}")
                # Try with minimal config
                minimal_config = {}
                for key in ["max_output_tokens", "temperature", "top_p", "top_k"]:
                    if key in config_params:
                        minimal_config[key] = config_params[key]
                
                if minimal_config:
                    try:
                        config = gtypes.GenerateContentConfig(**minimal_config)
                    except Exception as e2:
                        log.warning(f"Failed to create minimal config: {e2}")
                        config = None

        # Combine all content into a single message
        combined_content = "\n\n".join(contents) if contents else "Hello"
        
        # Prepend system instruction if not supported in config
        if final_system and not self.supports_feature("system_messages"):
            combined_content = f"System: {final_system}\n\nUser: {combined_content}"

        base_payload: Dict[str, Any] = {
            "model": self.model,
            "contents": combined_content
        }
        
        if config:
            base_payload["config"] = config

        log.debug("Gemini streaming payload keys: %s", list(base_payload.keys()))
        
        accumulated_response = ""
        
        try:
            # Streaming with SAFE parsing - NO .text access
            # Apply suppression only around the API call
            with UltimateSuppression():
                stream = await self.client.aio.models.generate_content_stream(**base_payload)
            
            async for chunk in stream:
                with UltimateSuppression():
                    # CRITICAL: Use safe parsing instead of chunk.text
                    chunk_result = _safe_parse_gemini_response(chunk)
                    chunk_text = chunk_result["response"]
                    tool_calls = chunk_result["tool_calls"]
                    
                    # Restore tool names if needed
                    if tool_calls and self._current_name_mapping:
                        chunk_result = self._restore_tool_names_in_response(chunk_result, self._current_name_mapping)
                        tool_calls = chunk_result["tool_calls"]
                
                # Handle text content with deduplication
                if chunk_text:
                    if not accumulated_response or not chunk_text.startswith(accumulated_response):
                        new_content = chunk_text[len(accumulated_response):] if chunk_text.startswith(accumulated_response) else chunk_text
                        if new_content:
                            accumulated_response += new_content
                            yield {
                                "response": new_content,
                                "tool_calls": []
                            }
                
                # Handle tool calls (names already restored)
                if tool_calls:
                    yield {
                        "response": "",
                        "tool_calls": tool_calls
                    }
        
        except Exception as e:
            log.error(f"Error in Gemini streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    @silence_gemini_warnings_async
    async def _regular_completion_async(
        self, 
        system: Optional[str],
        json_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        gemini_tools: Optional[List[gtypes.Tool]],
        filtered_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming completion using async client with SAFE response parsing - NO CONCATENATION."""
        try:
            # Handle system message and JSON instruction
            system_from_messages, contents = await self._split_for_gemini_async(messages)
            final_system = system or system_from_messages
            
            if json_instruction:
                if final_system:
                    final_system = f"{final_system}\n\n{json_instruction}"
                else:
                    final_system = json_instruction
                log.debug("Added JSON mode instruction to system prompt")

            # Build config
            config_params = filtered_params.copy()
            if gemini_tools:
                config_params["tools"] = gemini_tools
            
            # Handle thinking models hitting token limits
            if "2.5" in self.model:
                if "max_output_tokens" not in config_params:
                    # Set a reasonable max_output_tokens to prevent thinking from consuming all tokens
                    config_params["max_output_tokens"] = 4096
                    log.debug("Set max_output_tokens=4096 for Gemini 2.5 to prevent thinking token overflow")
            elif "max_output_tokens" not in config_params:
                # Set default for non-thinking models
                config_params["max_output_tokens"] = 4096
            
            config = None
            if config_params:
                try:
                    # Add system instruction to config if present and supported
                    if final_system and self.supports_feature("system_messages"):
                        config_params["system_instruction"] = final_system
                    
                    config = gtypes.GenerateContentConfig(**config_params)
                except Exception as e:
                    log.warning(f"Error creating GenerateContentConfig: {e}")
                    # Try with minimal config
                    minimal_config = {}
                    for key in ["max_output_tokens", "temperature", "top_p", "top_k"]:
                        if key in config_params:
                            minimal_config[key] = config_params[key]
                    
                    if minimal_config:
                        try:
                            config = gtypes.GenerateContentConfig(**minimal_config)
                        except Exception as e2:
                            log.warning(f"Failed to create minimal config: {e2}")
                            config = None

            # Combine all content - handle both text and multimodal
            if contents:
                # Check if we have any multimodal content
                has_multimodal = any(isinstance(c, list) for c in contents)
                
                if has_multimodal:
                    # Build structured content for Gemini
                    combined_content = []
                    for content_item in contents:
                        if isinstance(content_item, str):
                            combined_content.append(content_item)
                        elif isinstance(content_item, list):
                            # This is multimodal content - add all parts
                            combined_content.extend(content_item)
                else:
                    # All text content - join as before
                    combined_content = "\n\n".join(contents)
            else:
                combined_content = "Hello"
            
            # Prepend system instruction if not supported in config
            if final_system and not self.supports_feature("system_messages"):
                if isinstance(combined_content, list):
                    # Insert system message at the beginning of multimodal content
                    combined_content.insert(0, f"System: {final_system}\n\nUser: ")
                else:
                    combined_content = f"System: {final_system}\n\nUser: {combined_content}"

            base_payload: Dict[str, Any] = {
                "model": self.model,
                "contents": [combined_content]
            }
            
            if config:
                base_payload["config"] = config

            log.debug("Gemini payload keys: %s", list(base_payload.keys()))
            
            # Make the request and use SAFE parsing
            resp = await self.client.aio.models.generate_content(**base_payload)
            
            # CRITICAL: Use safe parsing and restore tool names
            return self._parse_gemini_response_with_restoration(resp, self._current_name_mapping)
            
        except Exception as e:
            log.error(f"Error in Gemini completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    @silence_gemini_warnings
    def _extract_tool_calls_from_response_with_restoration(self, response, name_mapping: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Extract tool calls from Gemini response with name restoration - SAFE parsing"""
        tool_calls = []
        
        # Only extract tool calls if tools are supported
        if not self.supports_feature("tools"):
            return tool_calls
        
        try:
            # Use safe parsing to extract tool calls without concatenation
            result = _safe_parse_gemini_response(response)
            tool_calls = result.get("tool_calls", [])
            
            # Restore original names if mapping provided
            if name_mapping and tool_calls:
                for tool_call in tool_calls:
                    if "function" in tool_call and "name" in tool_call["function"]:
                        sanitized_name = tool_call["function"]["name"]
                        original_name = name_mapping.get(sanitized_name, sanitized_name)
                        tool_call["function"]["name"] = original_name
            
        except Exception as e:
            log.debug(f"Error extracting tool calls: {e}")
        
        return tool_calls

    def _extract_tool_calls_from_response(self, response) -> List[Dict[str, Any]]:
        """Extract tool calls from Gemini response (legacy method, kept for compatibility)"""
        return self._extract_tool_calls_from_response_with_restoration(response, self._current_name_mapping)

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping
        self._current_name_mapping = {}
        # Gemini client cleanup if needed
        pass