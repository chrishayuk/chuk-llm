# chuk_llm/llm/providers/gemini_client.py

"""
Google Gemini chat-completion adapter with complete warning suppression and proper parameter handling.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import logging
import os
import sys
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
import warnings

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

# providers
from chuk_llm.llm.core.base import BaseLLMClient

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE WARNING SUPPRESSION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def apply_complete_warning_suppression():
    """Apply nuclear-level warning suppression for Gemini"""
    
    # Method 1: Environment variables
    os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
    os.environ.setdefault('GOOGLE_GENAI_SUPPRESS_WARNINGS', '1')
    
    # Method 2: Comprehensive warnings patterns
    warning_patterns = [
        ".*non-text parts in the response.*",
        ".*function_call.*",
        ".*returning concatenated text result.*",
        ".*check out the non text parts.*",
        ".*text parts.*",
        ".*response.*function_call.*"
    ]
    
    for pattern in warning_patterns:
        warnings.filterwarnings("ignore", message=pattern, category=UserWarning)
        warnings.filterwarnings("ignore", message=pattern, category=Warning)
    
    # Method 3: Module-level suppression for all Google modules
    google_modules = [
        "google",
        "google.*", 
        "google.genai",
        "google.genai.*",
        "google.generativeai",
        "google.generativeai.*",
        "google.ai",
        "google.ai.*"
    ]
    
    for module in google_modules:
        warnings.filterwarnings("ignore", category=UserWarning, module=module)
        warnings.filterwarnings("ignore", category=Warning, module=module)
        warnings.filterwarnings("ignore", module=module)
    
    # Method 4: Logger suppression
    google_loggers = [
        "google",
        "google.genai",
        "google.generativeai", 
        "google.ai.generativelanguage",
        "google.ai",
        "google.cloud"
    ]
    
    for logger_name in google_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # Only critical errors
        logger.propagate = False
        logger.disabled = True
        # Clear all handlers
        logger.handlers.clear()

# Apply suppression immediately when module loads
apply_complete_warning_suppression()

# ───────────────────────────────────────────────────────── helpers ──────────

def _convert_messages_for_chat(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """Convert ChatML messages to Gemini chat format.

    Returns
    -------
    system_instruction : Optional[str]
    user_message      : str
    """
    system_txt: Optional[str] = None
    user_message: str = "Hello"  # Default fallback

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        log.debug("↻ msg[%d] role=%s keys=%s", i, role, list(msg.keys()))

        # ---------------- system -----------------------------------------
        if role == "system":
            if system_txt is None:
                system_txt = content if isinstance(content, str) else str(content)
            continue

        # ---------------- tool response ----------------------------------
        if role == "tool":
            # Convert tool response to user message for chat API
            fn_name = msg.get("name") or "tool"
            user_message = f"Tool {fn_name} result: {content}"
            continue

        # ---------------- assistant function-calls ------------------------
        if role == "assistant" and msg.get("tool_calls"):
            # Convert tool calls to assistant message
            tool_text = "I need to use tools: "
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name")
                args_raw = fn.get("arguments", "{}")
                tool_text += f"{name}({args_raw}) "
            # This becomes part of the conversation context
            continue

        # ---------------- normal messages --------------------------------
        if role in {"user", "assistant"}:
            if isinstance(content, str):
                if role == "user":
                    user_message = content
            elif isinstance(content, list):
                # Handle multimodal content - convert to text for chat API
                text_content = ""
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "image_url":
                            text_content += "[Image content]"
                
                if text_content and role == "user":
                    user_message = text_content
            elif content is not None and role == "user":
                user_message = str(content)
        else:
            log.debug("Skipping unsupported message: %s", msg)

    log.debug("System-instruction: %s", system_txt)
    log.debug("User message: %s", user_message[:100] + "..." if len(user_message) > 100 else user_message)
    return system_txt, user_message


def _convert_tools_to_gemini_format(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[gtypes.Tool]]:
    """Convert OpenAI-style tools to Gemini format"""
    if not tools:
        return None
    
    gemini_tools = []
    function_declarations = []
    
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            
            # Convert to Gemini function declaration format
            function_decl = {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {})
            }
            function_declarations.append(function_decl)
    
    if function_declarations:
        # Create Gemini Tool object
        gemini_tool = gtypes.Tool(function_declarations=function_declarations)
        gemini_tools.append(gemini_tool)
    
    return gemini_tools if gemini_tools else None


# ─────────────────────────────────────────────────── enhanced context managers ───────────

class SuppressAllOutput:
    """Context manager to completely suppress all output including warnings"""
    
    def __init__(self):
        self.original_stderr = None
        self.original_stdout = None
        self.original_warn = None
        self.original_showwarning = None
        self.devnull = None
    
    def __enter__(self):
        # Store originals
        self.original_stderr = sys.stderr
        self.original_stdout = sys.stdout
        self.original_warn = warnings.warn
        self.original_showwarning = warnings.showwarning
        
        # Open devnull
        self.devnull = open(os.devnull, 'w')
        
        # Redirect output
        sys.stderr = self.devnull
        
        # Replace warning functions
        warnings.warn = lambda *args, **kwargs: None
        warnings.showwarning = lambda *args, **kwargs: None
        
        # Suppress all warnings
        warnings.simplefilter("ignore")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore everything
        if self.devnull:
            self.devnull.close()
        
        sys.stderr = self.original_stderr
        sys.stdout = self.original_stdout
        warnings.warn = self.original_warn
        warnings.showwarning = self.original_showwarning

@contextmanager
def suppress_warnings():
    """Standard context manager for warning suppression"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

# ─────────────────────────────────────────────────── main adapter ───────────

class GeminiLLMClient(BaseLLMClient):
    """`google-genai` wrapper with complete warning suppression and proper parameter handling."""

    def __init__(self, model: str = "gemini-2.0-flash", *, api_key: Optional[str] = None) -> None:
        # Apply nuclear warning suppression during initialization
        apply_complete_warning_suppression()
        
        # load environment
        load_dotenv()

        # get the api key
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # check if we have a key
        if not api_key:
            raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY env var not set")
        
        # Initialize with complete suppression
        with SuppressAllOutput():
            self.model = model
            self.client = genai.Client(api_key=api_key)

        log.info("GeminiLLMClient initialised with model '%s'", model)

    def _prepare_gemini_config(self, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Tuple[Optional[gtypes.GenerateContentConfig], Dict[str, Any]]:
        """
        Prepare proper GenerateContentConfig for Gemini with parameter mapping.
        
        Returns:
            Tuple of (config_object, remaining_direct_params)
        """
        config_params = {}
        direct_params = {}
        
        # Handle parameter mapping from OpenAI-style to Gemini-style
        parameter_mapping = {
            "max_tokens": "max_output_tokens",
            "stop": "stop_sequences",
            # Direct mappings (no change needed)
            "temperature": "temperature",
            "top_p": "top_p", 
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "top_k": "top_k",
            "candidate_count": "candidate_count"
        }
        
        # Handle system instruction separately (direct parameter)
        if "system" in kwargs:
            direct_params["system_instruction"] = kwargs.pop("system")
        
        # Process parameters
        for param, value in kwargs.items():
            if param in parameter_mapping:
                # Map to Gemini parameter name
                gemini_param = parameter_mapping[param]
                config_params[gemini_param] = value
                log.debug(f"Mapped parameter {param} -> {gemini_param} = {value}")
            elif param in ["temperature", "top_p", "frequency_penalty", "presence_penalty", 
                          "top_k", "candidate_count", "max_output_tokens", "stop_sequences"]:
                # Already in Gemini format
                config_params[param] = value
            else:
                # Unknown parameter - log and skip
                log.warning(f"Unknown parameter for Gemini: {param} = {value}")
        
        # Add tools if provided
        gemini_tools = _convert_tools_to_gemini_format(tools) if tools else None
        if gemini_tools:
            config_params["tools"] = gemini_tools
            # Disable automatic function calling to maintain control
            config_params["automatic_function_calling"] = gtypes.AutomaticFunctionCallingConfig(disable=True)
        
        # Create config object if we have parameters
        config = None
        if config_params:
            try:
                config = gtypes.GenerateContentConfig(**config_params)
                log.debug(f"Created GenerateContentConfig with: {list(config_params.keys())}")
            except Exception as e:
                log.error(f"Error creating GenerateContentConfig: {e}")
                log.debug(f"Failed config_params: {config_params}")
                # Fallback to basic config with just tools if provided
                if gemini_tools:
                    config = gtypes.GenerateContentConfig(
                        tools=gemini_tools,
                        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True)
                    )
        
        return config, direct_params

    def create_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None, 
        *, 
        stream: bool = False,
        **kwargs: Any
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Generate completion with complete warning suppression and proper parameter handling.
        
        • stream=False → returns awaitable that resolves to standardised dict
        • stream=True  → returns async iterator that yields chunks in real-time
        """
        log.debug("create_completion called – stream=%s, kwargs=%s", stream, kwargs)
        
        if stream:
            return self._stream_completion_async(messages, tools, **kwargs)
        else:
            return self._regular_completion(messages, tools, **kwargs)
        
    def _parse_gemini_chunk(self, chunk) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse Gemini streaming chunk for text and function calls"""
        chunk_text = ""
        tool_calls = []
        
        try:
            # Extract text content
            if hasattr(chunk, 'text') and chunk.text:
                chunk_text = chunk.text
            
            # Check for function calls
            if hasattr(chunk, 'function_calls') and chunk.function_calls:
                for fc in chunk.function_calls:
                    try:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": getattr(fc, "name", "unknown"),
                                "arguments": json.dumps(dict(getattr(fc, "args", {})))
                            }
                        })
                    except Exception as e:
                        log.debug(f"Error parsing function call in chunk: {e}")
                        continue
            
            # Alternative: check candidates structure for function calls
            elif hasattr(chunk, 'candidates') and chunk.candidates:
                try:
                    cand = chunk.candidates[0]
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        for part in cand.content.parts:
                            try:
                                if hasattr(part, 'text') and part.text:
                                    chunk_text += part.text
                                elif hasattr(part, 'function_call'):
                                    fc = part.function_call
                                    tool_calls.append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}", 
                                        "type": "function", 
                                        "function": {
                                            "name": getattr(fc, "name", "unknown"),
                                            "arguments": json.dumps(dict(getattr(fc, "args", {})))
                                        }
                                    })
                            except Exception as e:
                                log.debug(f"Error parsing part in chunk: {e}")
                                continue
                except (AttributeError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing chunk candidates: {e}")
            
        except Exception as e:
            log.debug(f"Error parsing Gemini chunk: {e}")
        
        return chunk_text, tool_calls

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real async streaming with complete warning suppression and proper parameter handling.
        """
        try:
            log.debug(f"Starting Gemini streaming for model: {self.model}")
            
            # Extract content and prepare config
            system_instruction, user_message = _convert_messages_for_chat(messages)
            config, direct_params = self._prepare_gemini_config(tools, **kwargs)
            
            # Build request parameters
            request_params = {
                "model": self.model,
                "contents": user_message
            }
            
            # Add system instruction if present (direct or from kwargs)
            if system_instruction:
                request_params["system_instruction"] = system_instruction
            elif "system_instruction" in direct_params:
                request_params["system_instruction"] = direct_params["system_instruction"]
            
            # Add config if present
            if config:
                request_params["config"] = config
            
            log.debug(f"Streaming request params keys: {list(request_params.keys())}")
            
            chunk_count = 0
            # Use nuclear suppression for the entire streaming operation
            with SuppressAllOutput():
                async for chunk in await self.client.aio.models.generate_content_stream(**request_params):
                    chunk_count += 1
                    
                    # Direct access to chunk.text
                    chunk_text = ""
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = chunk.text
                        log.debug(f"Chunk {chunk_count}: '{chunk_text[:50]}...'")
                    
                    # Immediate yield
                    if chunk_text:
                        yield {
                            "response": chunk_text,
                            "tool_calls": [],
                        }
                        await asyncio.sleep(0.001)
                    
                    # Handle tool calls separately if present
                    tool_calls = self._extract_tool_calls_from_chunk(chunk)
                    if tool_calls:
                        yield {
                            "response": "",
                            "tool_calls": tool_calls,
                        }
                        await asyncio.sleep(0.001)
            
            log.debug(f"Gemini streaming completed with {chunk_count} chunks")
                
        except Exception as e:
            log.error(f"Error in Gemini streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }
            
    async def _regular_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Non-streaming completion with complete warning suppression and proper parameter handling."""
        try:
            # Extract content and prepare config
            system_instruction, user_message = _convert_messages_for_chat(messages)
            config, direct_params = self._prepare_gemini_config(tools, **kwargs)
            
            # Build request parameters
            request_params = {
                "model": self.model,
                "contents": [user_message]
            }
            
            # Add system instruction if present (direct or from kwargs)
            if system_instruction:
                request_params["system_instruction"] = system_instruction
            elif "system_instruction" in direct_params:
                request_params["system_instruction"] = direct_params["system_instruction"]
            
            # Add config if present
            if config:
                request_params["config"] = config
            
            log.debug(f"Regular completion request params keys: {list(request_params.keys())}")
            
            # Make the async request with complete suppression
            with SuppressAllOutput():
                response = await self.client.aio.models.generate_content(**request_params)
            
            # Parse response
            response_text, tool_calls = self._parse_gemini_response(response)
            
            return {
                "response": response_text,
                "tool_calls": tool_calls
            }
            
        except Exception as e:
            log.error(f"Error in Gemini completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

# ───────────────────────────────────────── parse helpers ──────────

    def _extract_tool_calls_from_chunk(self, chunk) -> List[Dict[str, Any]]:
        """Fast tool call extraction - simplified for streaming performance"""
        tool_calls = []
        
        try:
            # Quick check for function calls
            if hasattr(chunk, 'function_calls') and chunk.function_calls:
                for fc in chunk.function_calls:
                    try:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": getattr(fc, "name", "unknown"),
                                "arguments": json.dumps(dict(getattr(fc, "args", {})))
                            }
                        })
                    except Exception:
                        continue
        except Exception:
            pass
        
        return tool_calls

    def _parse_gemini_response(self, response) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse Gemini non-streaming response for text and function calls"""
        response_text = ""
        tool_calls = []
        
        try:
            # Extract text content
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            
            # Check for function calls
            if hasattr(response, 'function_calls') and response.function_calls:
                for fc in response.function_calls:
                    try:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": getattr(fc, "name", "unknown"),
                                "arguments": json.dumps(dict(getattr(fc, "args", {})))
                            }
                        })
                    except Exception as e:
                        log.debug(f"Error parsing function call: {e}")
                        continue
            
            # Alternative: check candidates structure for function calls
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    cand = response.candidates[0]
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        for part in cand.content.parts:
                            try:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                                elif hasattr(part, 'function_call'):
                                    fc = part.function_call
                                    tool_calls.append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}", 
                                        "type": "function", 
                                        "function": {
                                            "name": getattr(fc, "name", "unknown"),
                                            "arguments": json.dumps(dict(getattr(fc, "args", {})))
                                        }
                                    })
                            except Exception as e:
                                log.debug(f"Error parsing part: {e}")
                                continue
                except (AttributeError, IndexError, TypeError) as e:
                    log.debug(f"Error parsing response candidates: {e}")
            
        except Exception as e:
            log.debug(f"Error parsing Gemini response: {e}")
        
        return response_text, tool_calls