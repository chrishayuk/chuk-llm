# chuk_llm/llm/providers/gemini_client.py

"""
Google Gemini chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration-driven capabilities with complete warning suppression and proper parameter handling.
UPDATED: Fixed to match Anthropic client patterns - proper system instruction support, 
universal vision format, and robust response handling.
"""

from __future__ import annotations

import asyncio
import base64
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
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

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

# ───────────────────────────────────────────────────── helpers ──────────

def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get *key* from dict **or** attribute-style object; fallback to *default*."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

def _parse_gemini_response(resp) -> Dict[str, Any]:
    """Convert Gemini response → standard `{response, tool_calls}` dict."""
    tool_calls: List[Dict[str, Any]] = []
    response_text = ""
    
    try:
        # Method 1: Direct text access
        if hasattr(resp, 'text') and resp.text:
            response_text = resp.text
        
        # Method 2: Check candidates structure  
        elif hasattr(resp, 'candidates') and resp.candidates:
            cand = resp.candidates[0]
            if hasattr(cand, 'content') and cand.content:
                # Check if content has parts
                if hasattr(cand.content, 'parts') and cand.content.parts:
                    text_parts = []
                    for part in cand.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                        elif hasattr(part, 'function_call'):
                            # Extract function call
                            fc = part.function_call
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": getattr(fc, "name", "unknown"),
                                    "arguments": json.dumps(dict(getattr(fc, "args", {})))
                                }
                            })
                    
                    if text_parts:
                        response_text = "".join(text_parts)
                
                # Fallback: try to get text directly from content
                elif hasattr(cand.content, 'text') and cand.content.text:
                    response_text = cand.content.text
        
        # Method 3: If no text found but response has candidates, it might be a thinking response
        if not response_text and hasattr(resp, 'candidates') and resp.candidates:
            # Check if this is a thinking response with no output (MAX_TOKENS reached)
            cand = resp.candidates[0]
            if hasattr(cand, 'finish_reason') and str(cand.finish_reason) == 'FinishReason.MAX_TOKENS':
                # For thinking models that hit token limits, provide a helpful message
                response_text = "[Response was cut off due to thinking token limit. The model was processing but ran out of output tokens.]"
            elif hasattr(cand, 'content') and cand.content is None:
                # Content is None, likely thinking response
                response_text = "[No text response generated - this may be a thinking response that exceeded token limits.]"
    
    except Exception as e:
        log.debug(f"Error parsing Gemini response: {e}")
        # Fallback: try to extract any meaningful text from the raw response
        if hasattr(resp, 'candidates') and resp.candidates:
            try:
                # Try to get some meaningful info from the response
                cand = resp.candidates[0]
                if hasattr(cand, 'finish_reason'):
                    reason = str(cand.finish_reason)
                    if 'MAX_TOKENS' in reason:
                        response_text = "[Response exceeded token limit during processing.]"
                    else:
                        response_text = f"[Response processing completed with status: {reason}]"
                else:
                    response_text = "[Unable to parse response - no text content available.]"
            except:
                response_text = "[Error: Could not parse response.]"
        else:
            response_text = str(resp) if resp else "[No response received.]"
    
    # Handle JSON mode duplication
    if response_text and response_text.count('{"') > 1:
        json_parts = response_text.split('{"')
        if len(json_parts) > 1:
            first_json = '{"' + json_parts[1].split('}')[0] + '}'
            try:
                json.loads(first_json)
                response_text = first_json
            except json.JSONDecodeError:
                pass
    
    if tool_calls:
        return {"response": response_text if response_text else "", "tool_calls": tool_calls}
    
    return {"response": response_text, "tool_calls": []}

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

class GeminiLLMClient(ConfigAwareProviderMixin, BaseLLMClient):
    """
    Configuration-aware `google-genai` wrapper following Anthropic client patterns.
    UPDATED: Proper system instruction support, universal vision format, robust response handling.
    """

    def __init__(self, model: str = "gemini-2.5-flash", *, api_key: Optional[str] = None) -> None:
        # Apply nuclear warning suppression during initialization
        apply_complete_warning_suppression()
        
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
        with SuppressAllOutput():
            self.model = safe_model
            self.client = genai.Client(api_key=api_key)

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
                "gemini_specific": {
                    "context_length": "2M tokens" if "2.5" in self.model else ("2M tokens" if "2.0" in self.model else "1M tokens"),
                    "model_family": self._detect_model_family(),
                    "experimental_features": "2.0" in self.model or "2.5" in self.model,
                    "warning_suppression": "complete",
                    "enhanced_reasoning": "2.5" in self.model,
                    "supports_function_calling": self.supports_feature("tools"),
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
    ) -> tuple[str, List[str]]:
        """
        Separate system text & convert ChatML list to Gemini format with async vision support.
        Uses configuration to validate vision support.
        """
        sys_txt: List[str] = []
        contents: List[str] = []

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
                    
                    # Process multimodal content - for now, convert to text description
                    # TODO: Implement proper multimodal content handling for Gemini
                    text_parts = []
                    for item in cont:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                text_parts.append("[Image content provided]")
                            else:
                                text_parts.append(str(item))
                        else:
                            text_parts.append(str(item))
                    
                    if text_parts:
                        contents.append(" ".join(text_parts))
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
        """

        # Validate capabilities using configuration
        if tools and not self.supports_feature("tools"):
            log.warning(f"Tools provided but model {self.model} doesn't support tools according to configuration")
            tools = None
        
        if stream and not self.supports_feature("streaming"):
            log.warning(f"Streaming requested but model {self.model} doesn't support streaming according to configuration")
            stream = False

        gemini_tools = _convert_tools_to_gemini_format(tools)
        
        # Check for JSON mode (using configuration validation)
        json_instruction = self._check_json_mode(extra)
        
        # Filter parameters for Gemini compatibility (using configuration limits)
        if max_tokens:
            extra["max_tokens"] = max_tokens
        filtered_params = self._filter_gemini_params(extra)

        # ––– streaming: use async streaming -------------------------
        if stream:
            return self._stream_completion_async(system, json_instruction, messages, gemini_tools, filtered_params)

        # ––– non-streaming: use async client ------------------------------
        return self._regular_completion_async(system, json_instruction, messages, gemini_tools, filtered_params)

    async def _stream_completion_async(
        self, 
        system: Optional[str],
        json_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        gemini_tools: Optional[List[gtypes.Tool]],
        filtered_params: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Real streaming using Gemini client with configuration-aware processing."""
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
            
            # Use complete suppression for streaming
            with SuppressAllOutput():
                async for chunk in await self.client.aio.models.generate_content_stream(**base_payload):
                    chunk_text = ""
                    tool_calls = []
                    
                    # Parse chunk
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = chunk.text
                    
                    # Check for function calls if tools are supported
                    if gemini_tools and self.supports_feature("tools"):
                        tool_calls = self._extract_tool_calls_from_response(chunk)
                    
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
                    
                    # Handle tool calls
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

    async def _regular_completion_async(
        self, 
        system: Optional[str],
        json_instruction: Optional[str],
        messages: List[Dict[str, Any]],
        gemini_tools: Optional[List[gtypes.Tool]],
        filtered_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming completion using async client with configuration-aware processing."""
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

            # Combine all content into a single message
            combined_content = "\n\n".join(contents) if contents else "Hello"
            
            # Prepend system instruction if not supported in config
            if final_system and not self.supports_feature("system_messages"):
                combined_content = f"System: {final_system}\n\nUser: {combined_content}"

            base_payload: Dict[str, Any] = {
                "model": self.model,
                "contents": [combined_content]
            }
            
            if config:
                base_payload["config"] = config

            log.debug("Gemini payload keys: %s", list(base_payload.keys()))
            
            # Make the request with complete suppression
            with SuppressAllOutput():
                resp = await self.client.aio.models.generate_content(**base_payload)
            
            return _parse_gemini_response(resp)
            
        except Exception as e:
            log.error(f"Error in Gemini completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    def _extract_tool_calls_from_response(self, response) -> List[Dict[str, Any]]:
        """Extract tool calls from Gemini response"""
        tool_calls = []
        
        # Only extract tool calls if tools are supported
        if not self.supports_feature("tools"):
            return tool_calls
        
        try:
            # Check response structure for function calls
            if hasattr(response, 'candidates') and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, 'content') and cand.content:
                    # Check if content has parts
                    if hasattr(cand.content, 'parts') and cand.content.parts:
                        for part in cand.content.parts:
                            if hasattr(part, 'function_call'):
                                fc = part.function_call
                                tool_calls.append({
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": getattr(fc, "name", "unknown"),
                                        "arguments": json.dumps(dict(getattr(fc, "args", {})))
                                    }
                                })
            
            # Alternative: check for function calls at response level
            elif hasattr(response, 'function_calls') and response.function_calls:
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
                    except Exception:
                        continue
            
        except Exception as e:
            log.debug(f"Error extracting tool calls: {e}")
        
        return tool_calls