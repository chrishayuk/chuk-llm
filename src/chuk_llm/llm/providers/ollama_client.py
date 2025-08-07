# chuk_llm/llm/providers/ollama_client.py
"""
Ollama chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration-driven capabilities with local model support.
ENHANCED with GPT-OSS and reasoning model support.
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
    
    ENHANCED: Now supports reasoning models like GPT-OSS with thinking streams.
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
        ENHANCED: Now includes reasoning model detection.
        """
        # Get base info from configuration
        info = super().get_model_info()
        
        # Add Ollama-specific metadata only if no error occurred
        if not info.get("error"):
            is_reasoning = self._is_reasoning_model()
            
            info.update({
                "ollama_specific": {
                    "host": self.api_base,
                    "local_deployment": True,
                    "model_family": self._detect_model_family(),
                    "supports_custom_models": True,
                    "no_api_key_required": True,
                    "is_reasoning_model": is_reasoning,
                    "supports_thinking_stream": is_reasoning,
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
        elif "gpt-oss" in model_lower:
            return "gpt-oss"
        elif "codellama" in model_lower or "code" in model_lower:
            return "code"
        else:
            return "unknown"

    def _is_reasoning_model(self) -> bool:
        """
        ENHANCED: Check if the current model is a reasoning model that uses thinking.
        
        Reasoning models output their thought process in a 'thinking' field
        and may have empty 'content' during thinking phases.
        """
        reasoning_patterns = [
            "gpt-oss", "qwq", "marco-o1", "deepseek-r1", 
            "reasoning", "think", "r1", "o1"
        ]
        model_lower = self.model.lower()
        is_reasoning = any(pattern in model_lower for pattern in reasoning_patterns)
        
        if is_reasoning:
            log.debug(f"Detected reasoning model: {self.model}")
        
        return is_reasoning

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
        """
        ENHANCED: Parse Ollama response with support for reasoning models.
        
        Reasoning models like GPT-OSS use 'thinking' field for their reasoning process
        and may have empty 'content'. This method handles both cases properly.
        """
        main_text = ""
        tool_calls = []
        thinking_text = ""
        
        # Get message from response
        message = getattr(response, "message", None)
        if message:
            # Get content and thinking
            main_text = getattr(message, "content", "")
            thinking_text = getattr(message, "thinking", "")
            
            # For reasoning models, if content is empty but thinking exists, use thinking
            if not main_text and thinking_text and self._is_reasoning_model():
                main_text = thinking_text
                log.debug(f"Using thinking content as main response for reasoning model: '{thinking_text[:100]}...'")
            
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
        
        result = {
            "response": main_text if main_text else None,
            "tool_calls": tool_calls
        }
        
        # Add reasoning metadata for reasoning models
        if self._is_reasoning_model():
            result["reasoning"] = {
                "thinking": thinking_text,
                "content": getattr(message, "content", "") if message else "",
                "model_type": "reasoning"
            }
        
        return result
    
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
        ENHANCED: Now supports reasoning model streaming with thinking process.
        
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
        ENHANCED: Real streaming using Ollama's AsyncClient with support for reasoning models.
        
        This now properly handles reasoning models like GPT-OSS that stream their thinking
        process in the 'thinking' field rather than 'content'.
        
        Key improvements:
        - Detects reasoning models automatically
        - Streams thinking content in real-time 
        - Maintains tool call detection
        - Preserves backward compatibility with regular models
        """
        try:
            is_reasoning_model = self._is_reasoning_model()
            log.debug(f"Starting Ollama streaming for {'reasoning' if is_reasoning_model else 'regular'} model: {self.model}")
            
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
            total_thinking_chars = 0
            total_content_chars = 0
            aggregated_tool_calls = []
            
            # Process each chunk in the stream immediately
            async for chunk in stream:
                chunk_count += 1
                
                # ENHANCED: Extract both content and thinking
                content = ""
                thinking = ""
                
                if hasattr(chunk, 'message') and chunk.message:
                    content = getattr(chunk.message, "content", "")
                    thinking = getattr(chunk.message, "thinking", "")  # NEW: Support thinking field
                
                # Track statistics
                if content:
                    total_content_chars += len(content)
                if thinking:
                    total_thinking_chars += len(thinking)
                
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
                
                # ENHANCED: Determine what to stream based on model type
                stream_content = ""
                
                if is_reasoning_model and thinking:
                    # For reasoning models like GPT-OSS, stream the thinking process
                    stream_content = thinking
                    if chunk_count <= 5:  # Log first few chunks for debugging
                        log.debug(f"Streaming thinking chunk {chunk_count}: '{thinking[:30]}...'")
                elif content:
                    # For regular models, stream the content
                    stream_content = content
                    if chunk_count <= 5:  # Log first few chunks for debugging
                        log.debug(f"Streaming content chunk {chunk_count}: '{content[:30]}...'")
                
                # ENHANCED: Yield chunk if we have content, thinking, or tool calls
                if stream_content or new_tool_calls:
                    chunk_data = {
                        "response": stream_content,
                        "tool_calls": new_tool_calls
                    }
                    
                    # Add reasoning metadata if applicable
                    if is_reasoning_model:
                        chunk_data["reasoning"] = {
                            "is_thinking": bool(thinking and not content),
                            "thinking_content": thinking if thinking else None,
                            "regular_content": content if content else None,
                            "chunk_type": "thinking" if thinking else "content"
                        }
                    
                    yield chunk_data
                elif chunk_count <= 3:
                    # Log empty chunks only for first few chunks to debug issues
                    log.debug(f"Empty chunk {chunk_count}: content='{content}', thinking='{thinking}', tools={len(new_tool_calls)}")
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            # Final statistics
            log.debug(f"Ollama streaming completed: {chunk_count} chunks, "
                     f"thinking={total_thinking_chars} chars, content={total_content_chars} chars, "
                     f"tools={len(aggregated_tool_calls)} for {'reasoning' if is_reasoning_model else 'regular'} model")
        
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
        """
        ENHANCED: Non-streaming completion using async execution with reasoning model support.
        """
        try:
            is_reasoning_model = self._is_reasoning_model()
            log.debug(f"Starting Ollama completion for {'reasoning' if is_reasoning_model else 'regular'} model: {self.model}")
            
            result = await asyncio.to_thread(self._create_sync, messages, tools, **kwargs)
            
            log.debug(f"Ollama completion result: "
                     f"response={len(str(result.get('response', ''))) if result.get('response') else 0} chars, "
                     f"tool_calls={len(result.get('tool_calls', []))}, "
                     f"reasoning={'yes' if result.get('reasoning') else 'no'}")
            
            return result
        except Exception as e:
            log.error(f"Error in Ollama completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }