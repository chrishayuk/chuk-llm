# chuk_llm/llm/providers/ollama_client.py
"""
Ollama chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration-driven capabilities with local model support.
"""
import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Tuple

# provider
import ollama

# providers
from chuk_llm.llm.core.base import BaseLLMClient
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin

log = logging.getLogger(__name__)

class OllamaLLMClient(ConfigAwareProviderMixin, BaseLLMClient):
    """
    Configuration-aware wrapper around `ollama` SDK that gets all capabilities
    from unified YAML configuration for local model support.
    """

    def __init__(self, model: str = "qwen3", api_base: Optional[str] = None) -> None:
        """
        Initialize Ollama client.
        
        Args:
            model: Name of the model to use
            api_base: Optional API base URL
        """
        # Initialize the configuration mixin FIRST
        ConfigAwareProviderMixin.__init__(self, "ollama", model)
        
        self.model = model
        self.api_base = api_base or "http://localhost:11434"
        
        # Verify that the installed ollama package supports chat
        if not hasattr(ollama, 'chat'):
            raise ValueError(
                "The installed ollama package does not expose 'chat'; "
                "check your ollama-python version."
            )
        
        # Create clients with proper host configuration
        # Modern ollama-python uses host parameter in Client constructor
        try:
            self.async_client = ollama.AsyncClient(host=self.api_base)
            self.sync_client = ollama.Client(host=self.api_base)
            log.debug(f"Ollama clients initialized with host: {self.api_base}")
        except TypeError:
            # Fallback for older versions that don't support host parameter
            self.async_client = ollama.AsyncClient()
            self.sync_client = ollama.Client()
            
            # Try the old set_host method as fallback
            if hasattr(ollama, 'set_host'):
                ollama.set_host(self.api_base)
                log.debug(f"Using ollama.set_host() with: {self.api_base}")
            else:
                log.debug(f"Ollama using default host (localhost:11434)")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model info using configuration, with Ollama-specific additions.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add Ollama-specific metadata only if no error occurred
        if not info.get("error"):
            info.update({
                "ollama_specific": {
                    "host": self.api_base,
                    "local_deployment": True,
                    "model_family": self._detect_model_family(),
                    "supports_custom_models": True,
                    "no_api_key_required": True,
                },
                "parameter_mapping": {
                    "temperature": "temperature",
                    "top_p": "top_p",
                    "max_tokens": "num_predict",  # Ollama-specific mapping
                    "stop": "stop",
                    "top_k": "top_k",
                    "seed": "seed"
                },
                "unsupported_parameters": [
                    "logit_bias", "user", "n", "best_of", "response_format"
                ]
            })
        
        return info

    def _detect_model_family(self) -> str:
        """Detect model family for Ollama-specific optimizations"""
        model_lower = self.model.lower()
        if "llama" in model_lower:
            return "llama"
        elif "qwen" in model_lower:
            return "qwen"
        elif "mistral" in model_lower:
            return "mistral"
        elif "granite" in model_lower:
            return "granite"
        elif "gemma" in model_lower:
            return "gemma"
        elif "phi" in model_lower:
            return "phi"
        elif "codellama" in model_lower or "code" in model_lower:
            return "code"
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
        
        # Check system message support
        has_system = any(msg.get("role") == "system" for msg in messages)
        if has_system and not self.supports_feature("system_messages"):
            log.info(f"System messages will be converted - {self.model} has limited system message support")
        
        # Validate parameters using configuration
        validated_kwargs = self.validate_parameters(**validated_kwargs)
        
        # Remove unsupported parameters for Ollama
        unsupported = ["logit_bias", "user", "n", "best_of", "response_format"]
        for param in unsupported:
            if param in validated_kwargs:
                log.debug(f"Removing unsupported parameter for Ollama: {param}")
                validated_kwargs.pop(param)
        
        return validated_messages, validated_tools, validated_stream, validated_kwargs

    def _prepare_ollama_messages(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for Ollama with configuration-aware processing.
        """
        ollama_messages = []
        
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            
            # Handle system messages based on configuration
            if role == "system":
                if self.supports_feature("system_messages"):
                    message = {"role": "system", "content": content}
                else:
                    # Convert to user message as fallback
                    log.debug(f"Converting system message to user message - {self.model} doesn't support system messages")
                    message = {"role": "user", "content": f"System: {content}"}
            else:
                message = {"role": role, "content": content}
            
            # Handle images if present in the message content and vision is supported
            if isinstance(content, list):
                has_images = any(item.get("type") in ["image", "image_url"] for item in content)
                
                if has_images and not self.supports_feature("vision"):
                    # Extract only text content
                    text_content = " ".join([
                        item.get("text", "") for item in content 
                        if item.get("type") == "text"
                    ])
                    message["content"] = text_content or "[Image content removed - not supported by model]"
                    log.warning(f"Removed vision content - {self.model} doesn't support vision according to configuration")
                else:
                    # Process images for Ollama format
                    for item in content:
                        if item.get("type") == "image" or item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image"):
                                # Extract base64 data and convert to proper format
                                import base64
                                _, encoded = image_url.split(",", 1)
                                message["images"] = [base64.b64decode(encoded)]
                            else:
                                message["images"] = [image_url]
            
            ollama_messages.append(message)
        
        return ollama_messages

    def _create_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous internal completion call with configuration awareness.
        """
        # Prepare messages for Ollama with configuration-aware processing
        ollama_messages = self._prepare_ollama_messages(messages)
        
        # Convert tools to Ollama format if supported
        ollama_tools = []
        if tools and self.supports_feature("tools"):
            for tool in tools:
                # Ollama expects a specific format for tools
                if "function" in tool:
                    fn = tool["function"]
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": fn.get("name"),
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters", {})
                        }
                    })
                else:
                    # Pass through other tool formats
                    ollama_tools.append(tool)
        elif tools:
            log.warning(f"Tools provided but {self.model} doesn't support tools according to configuration")
        
        # Build Ollama options from kwargs
        ollama_options = self._build_ollama_options(kwargs)
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
        }
        
        # Add tools if provided and supported
        if ollama_tools:
            request_params["tools"] = ollama_tools
        
        # Add options if provided
        if ollama_options:
            request_params["options"] = ollama_options
        
        # Make the non-streaming sync call
        response = self.sync_client.chat(**request_params)
        
        # Process response
        return self._parse_response(response)
    
    def _build_ollama_options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Ollama options dict from OpenAI-style parameters.
        
        Ollama parameters go in an 'options' dict, not directly in chat().
        """
        ollama_options = {}
        
        # Map OpenAI-style parameters to Ollama options
        parameter_mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "num_predict",  # Ollama uses num_predict instead of max_tokens
            "stop": "stop",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "top_k": "top_k",
            "seed": "seed",
        }
        
        for openai_param, ollama_param in parameter_mapping.items():
            if openai_param in kwargs:
                value = kwargs[openai_param]
                ollama_options[ollama_param] = value
                log.debug(f"Mapped {openai_param}={value} to Ollama option {ollama_param}")
        
        # Handle any Ollama-specific options passed directly
        if "options" in kwargs and isinstance(kwargs["options"], dict):
            ollama_options.update(kwargs["options"])
        
        return ollama_options

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse Ollama response to standardized format with configuration awareness."""
        main_text = ""
        tool_calls = []
        
        # Get message from response
        message = getattr(response, "message", None)
        if message:
            # Get content
            main_text = getattr(message, "content", "No response")
            
            # Process tool calls if any and if tools are supported
            raw_tool_calls = getattr(message, "tool_calls", None)
            if raw_tool_calls and self.supports_feature("tools"):
                for tc in raw_tool_calls:
                    tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                    
                    fn_name = getattr(tc.function, "name", "")
                    fn_args = getattr(tc.function, "arguments", {})
                    
                    # Ensure arguments are in string format
                    if isinstance(fn_args, dict):
                        fn_args_str = json.dumps(fn_args)
                    elif isinstance(fn_args, str):
                        fn_args_str = fn_args
                    else:
                        fn_args_str = str(fn_args)
                    
                    tool_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": fn_args_str
                        }
                    })
            elif raw_tool_calls:
                log.warning(f"Received tool calls but {self.model} doesn't support tools according to configuration")
        
        # If we have tool calls and no content, return null content
        if tool_calls and not main_text:
            return {"response": None, "tool_calls": tool_calls}
        
        return {"response": main_text, "tool_calls": tool_calls}
    
    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Configuration-aware completion generation with real streaming support.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tools
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to the underlying API
            
        Returns:
            When stream=True: AsyncIterator that yields chunks in real-time
            When stream=False: Awaitable that resolves to completion dict
        """
        # Validate request against configuration
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        if validated_stream:
            # Return async generator directly for real streaming
            return self._stream_completion_async(validated_messages, validated_tools, **validated_kwargs)
        else:
            # Return awaitable for non-streaming
            return self._regular_completion(validated_messages, validated_tools, **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real streaming using Ollama's AsyncClient with configuration awareness.
        This provides true real-time streaming from Ollama's API.
        """
        try:
            log.debug(f"Starting Ollama streaming for model: {self.model}")
            
            # Prepare messages for Ollama with configuration-aware processing
            ollama_messages = self._prepare_ollama_messages(messages)
            
            # Convert tools to Ollama format if supported
            ollama_tools = []
            if tools and self.supports_feature("tools"):
                for tool in tools:
                    if "function" in tool:
                        fn = tool["function"]
                        ollama_tools.append({
                            "type": "function",
                            "function": {
                                "name": fn.get("name"),
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {})
                            }
                        })
                    else:
                        ollama_tools.append(tool)
            elif tools:
                log.warning(f"Tools provided but {self.model} doesn't support tools according to configuration")
            
            # Build Ollama options from kwargs
            ollama_options = self._build_ollama_options(kwargs)
            
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,
            }
            
            # Add tools if provided and supported
            if ollama_tools:
                request_params["tools"] = ollama_tools
            
            # Add options if provided
            if ollama_options:
                request_params["options"] = ollama_options
            
            # Use async client for real streaming
            stream = await self.async_client.chat(**request_params)
            
            chunk_count = 0
            aggregated_tool_calls = []
            
            # Process each chunk in the stream immediately
            async for chunk in stream:
                chunk_count += 1
                
                # Get content from chunk
                content = ""
                if hasattr(chunk, 'message') and chunk.message:
                    content = getattr(chunk.message, "content", "")
                
                # Check for tool calls (only if tools are supported)
                new_tool_calls = []
                if (hasattr(chunk, 'message') and chunk.message and 
                    self.supports_feature("tools")):
                    chunk_tool_calls = getattr(chunk.message, "tool_calls", None)
                    if chunk_tool_calls:
                        for tc in chunk_tool_calls:
                            tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                            
                            fn_name = getattr(tc.function, "name", "")
                            fn_args = getattr(tc.function, "arguments", {})
                            
                            # Process arguments
                            if isinstance(fn_args, dict):
                                fn_args_str = json.dumps(fn_args)
                            elif isinstance(fn_args, str):
                                fn_args_str = fn_args
                            else:
                                fn_args_str = str(fn_args)
                            
                            tool_call = {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": fn_name,
                                    "arguments": fn_args_str
                                }
                            }
                            new_tool_calls.append(tool_call)
                            aggregated_tool_calls.append(tool_call)
                
                # Yield chunk immediately if we have content or tool calls
                if content or new_tool_calls:
                    yield {
                        "response": content,
                        "tool_calls": new_tool_calls
                    }
                
                # Allow other async tasks to run periodically
                if chunk_count % 5 == 0:
                    await asyncio.sleep(0)
            
            log.debug(f"Ollama streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            log.error(f"Error in Ollama streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _regular_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Non-streaming completion using async execution with configuration awareness."""
        try:
            log.debug(f"Starting Ollama completion for model: {self.model}")
            
            result = await asyncio.to_thread(self._create_sync, messages, tools, **kwargs)
            
            log.debug(f"Ollama completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}")
            
            return result
        except Exception as e:
            log.error(f"Error in Ollama completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }