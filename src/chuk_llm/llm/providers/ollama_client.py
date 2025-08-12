# chuk_llm/llm/providers/ollama_client.py
"""
Ollama chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration-driven capabilities with local model support.
ENHANCED with GPT-OSS and reasoning model support - FIXED DUPLICATE TOOL CALLS.
FIXED: Proper context memory preservation for multi-turn conversations.
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
    FIXED: Deduplicates tool calls that appear in multiple locations in chunks.
    FIXED: Proper context memory preservation for multi-turn conversations.
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

        # Look up model capabilities reported by Ollama
        self._model_capabilities = []
        try:
            model_info = self.sync_client.show(model=self.model)
            self._model_capabilities = model_info.capabilities
        except ollama.ResponseError:
            log.debug("Unable to get model capabilities from Ollama")

    def supports_feature(self, feature_name: str) -> bool:
        if feature_name in self._model_capabilities:
            return True
        return super().supports_feature(feature_name)

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
        
        First checks model capabilities reported by Ollama, then falls back to name patterns.
        """
        # Check if model has "thinking" capability reported by Ollama
        if "thinking" in self._model_capabilities:
            log.debug(f"Detected reasoning model from capabilities: {self.model}")
            return True
        
        # Fall back to pattern matching in model name
        reasoning_patterns = [
            "gpt-oss", "qwq", "marco-o1", "deepseek-r1", 
            "reasoning", "think", "r1", "o1"
        ]
        model_lower = self.model.lower()
        is_reasoning = any(pattern in model_lower for pattern in reasoning_patterns)
        
        if is_reasoning:
            log.debug(f"Detected reasoning model from name pattern: {self.model}")
        
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
        FIXED: Prepare messages for Ollama with proper context preservation.
        
        This enhanced version ensures that:
        1. Full conversation history is maintained
        2. Tool calls in assistant messages are properly formatted
        3. Tool responses are correctly included in the context
        """
        ollama_messages = []
        
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            
            # Handle different message roles
            if role == "system":
                # Check if system messages are supported
                if self.supports_feature("system_messages"):
                    message = {"role": "system", "content": content}
                else:
                    # Convert to user message as fallback
                    message = {"role": "user", "content": f"System: {content}"}
                    log.debug(f"Converting system message to user message - {self.model} doesn't support system messages")
                
            elif role == "assistant":
                # CRITICAL FIX: Properly handle assistant messages with tool calls
                message = {"role": "assistant", "content": content if content else ""}
                
                # Handle tool calls in assistant messages (for context)
                if tool_calls := m.get("tool_calls"):
                    formatted_tool_calls = []
                    for tool_call in tool_calls:
                        # Create a properly formatted tool call for Ollama
                        tc_formatted = {
                            "function": {
                                "name": tool_call.get("function", {}).get("name", ""),
                                "arguments": {}
                            }
                        }
                        
                        # Parse arguments if they're a string
                        args = tool_call.get("function", {}).get("arguments", {})
                        if isinstance(args, str):
                            try:
                                tc_formatted["function"]["arguments"] = json.loads(args)
                            except json.JSONDecodeError:
                                tc_formatted["function"]["arguments"] = {"raw": args}
                        elif isinstance(args, dict):
                            tc_formatted["function"]["arguments"] = args
                        else:
                            tc_formatted["function"]["arguments"] = {}
                        
                        formatted_tool_calls.append(tc_formatted)
                    
                    message["tool_calls"] = formatted_tool_calls
                    
                    # Some Ollama models expect tool calls to be mentioned in content too
                    if not content and formatted_tool_calls:
                        # Add a description of tool calls to content for models that need it
                        tool_names = [tc["function"]["name"] for tc in formatted_tool_calls]
                        message["content"] = f"[Called tools: {', '.join(tool_names)}]"
                        
            elif role == "tool":
                # CRITICAL FIX: Handle tool response messages properly
                # Ollama expects tool responses as user messages with special formatting
                tool_name = m.get("name", "unknown_tool")
                tool_content = m.get("content", "")
                
                # Format tool response for Ollama
                message = {
                    "role": "user",  # Ollama treats tool responses as user messages
                    "content": f"Tool Response from {tool_name}: {tool_content}"
                }
                
                # Let Ollama handle metadata if it supports it
                # The API will ignore metadata if not supported
                message["metadata"] = {
                    "type": "tool_response",
                    "tool_name": tool_name
                }
                    
            elif role == "user":
                message = {"role": "user", "content": content}
                
                # Handle images if present in the message content
                if isinstance(content, list):
                    has_images = any(item.get("type") in ["image", "image_url"] for item in content)
                    
                    if has_images and not self.supports_feature("vision"):
                        # Extract only text content
                        text_content = " ".join([
                            item.get("text", "") for item in content 
                            if item.get("type") == "text"
                        ])
                        message["content"] = text_content or "[Image content removed - not supported by model]"
                        log.warning(f"Removed vision content - {self.model} doesn't support vision")
                    else:
                        # Process images for Ollama format
                        text_parts = []
                        images = []
                        
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") in ["image", "image_url"]:
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
                                    # Extract base64 data
                                    import base64
                                    _, encoded = image_url.split(",", 1)
                                    images.append(base64.b64decode(encoded))
                                else:
                                    images.append(image_url)
                        
                        message["content"] = " ".join(text_parts)
                        if images:
                            message["images"] = images
            else:
                # Handle any other role types
                message = {"role": role, "content": content}
            
            ollama_messages.append(message)
        
        # CRITICAL: Log the conversation context for debugging
        if len(ollama_messages) > 1:
            log.debug(f"Prepared {len(ollama_messages)} messages for Ollama with full context:")
            for i, msg in enumerate(ollama_messages[-3:]):  # Log last 3 messages
                role = msg.get("role", "unknown")
                content_preview = str(msg.get("content", ""))[:100]
                has_tools = "tool_calls" in msg
                log.debug(f"  Message {i}: role={role}, content='{content_preview}...', has_tools={has_tools}")
        
        return ollama_messages

    def _validate_conversation_context(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Validate that the conversation context is properly structured.
        
        Returns True if context is valid, False otherwise.
        """
        if not messages:
            return True
        
        # Check for proper role alternation (with tool messages allowed)
        last_role = None
        for msg in messages:
            role = msg.get("role")
            
            # Tool messages can appear anywhere
            if role == "tool":
                continue
                
            # Check for duplicate consecutive roles (except system at start)
            if last_role == role and role != "system":
                log.warning(f"Duplicate consecutive {role} messages detected - may cause context issues")
                
            last_role = role
        
        # Check that tool calls have corresponding tool responses
        pending_tool_calls = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", tc.get("function", {}).get("name", "unknown"))
                    pending_tool_calls[tc_id] = True
                    
            elif msg.get("role") == "tool":
                tool_id = msg.get("tool_call_id", msg.get("name", "unknown"))
                if tool_id in pending_tool_calls:
                    del pending_tool_calls[tool_id]
        
        if pending_tool_calls:
            log.warning(f"Found {len(pending_tool_calls)} tool calls without responses - context may be incomplete")
        
        return True

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
        
        # Handle think parameter for sync client
        if 'think' in kwargs:
            think_value = kwargs.get('think')
            if isinstance(think_value, bool) and think_value:
                think_value = 'medium'
            
            if think_value in ['low', 'medium', 'high']:
                try:
                    # Check if sync client supports think parameter
                    import inspect
                    chat_signature = inspect.signature(self.sync_client.chat)
                    if 'think' in chat_signature.parameters:
                        request_params["think"] = think_value
                        log.debug(f"Enabled thinking mode ({think_value}) for sync call")
                    else:
                        log.debug(f"Think parameter not supported by sync client")
                except Exception as e:
                    log.debug(f"Could not check think parameter support: {e}")
        
        # Make the non-streaming sync call
        response = self.sync_client.chat(**request_params)
        
        # Process response
        return self._parse_response(response)
    
    def _build_ollama_options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Ollama options dict from OpenAI-style parameters.
        
        Ollama parameters go in an 'options' dict, not directly in chat().
        Special parameters like 'think' are handled separately at the request level.
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
        
        # List of special parameters that are handled at request level, not in options
        special_params = ['think', 'stream', 'tools', 'messages', 'model']
        for param in special_params:
            if param in kwargs:
                log.debug(f"Parameter '{param}' will be handled at request level, not in options")
        
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
        ENHANCED: Configuration-aware completion with proper context preservation.
        """
        # CRITICAL: Validate conversation context before processing
        if not self._validate_conversation_context(messages):
            log.warning("Conversation context validation failed - responses may lack context")
        
        # Log conversation length for debugging
        log.debug(f"Creating completion with {len(messages)} messages in context")
        if len(messages) > 1:
            # Log the conversation flow
            roles = [msg.get("role", "unknown") for msg in messages[-5:]]  # Last 5 messages
            log.debug(f"Recent conversation flow: {' -> '.join(roles)}")
        
        # Continue with existing validation and processing...
        validated_messages, validated_tools, validated_stream, validated_kwargs = self._validate_request_with_config(
            messages, tools, stream, **kwargs
        )
        
        if validated_stream:
            return self._stream_completion_async(validated_messages, validated_tools, **validated_kwargs)
        else:
            return self._regular_completion(validated_messages, validated_tools, **validated_kwargs)

    async def _stream_completion_async(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        ENHANCED: Real streaming using Ollama's AsyncClient with support for reasoning models.
        FIXED: Deduplicates tool calls that appear in multiple locations in the same chunk.
        
        This now properly handles:
        - Granite models that send tool calls in early chunks with no content
        - Regular models that send content and tool calls together
        - Reasoning models that stream thinking content
        - Duplicate tool calls in the same chunk (GPT-OSS issue)
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
            
            # Add think parameter if specified (supported in latest Ollama)
            if 'think' in kwargs:
                think_value = kwargs.get('think')
                if isinstance(think_value, bool) and think_value:
                    think_value = 'medium'
                
                if think_value in ['low', 'medium', 'high']:
                    request_params["think"] = think_value
                    log.debug(f"Enabled thinking mode ({think_value}) for model: {self.model}")
                elif think_value:
                    # Pass through any other think value
                    request_params["think"] = think_value
                    log.debug(f"Using think parameter: {think_value}")
            
            # Use async client for real streaming
            stream = await self.async_client.chat(**request_params)
            
            chunk_count = 0
            total_thinking_chars = 0
            total_content_chars = 0
            aggregated_tool_calls = []
            empty_chunk_count = 0
            
            # Track if we've seen tool calls to avoid duplicates
            seen_tool_call_keys = set()
            
            # CRITICAL FIX: Track if we've already yielded tool calls
            tool_calls_yielded = False
            
            # Helper function to create unique key for tool call
            def make_tool_call_key(tc_dict):
                """Create a unique key for a tool call based on its content"""
                return f"{tc_dict['function']['name']}:{tc_dict['function']['arguments']}"
            
            # Process each chunk in the stream immediately
            async for chunk in stream:
                chunk_count += 1
                
                # ENHANCED: Extract both content and thinking
                content = ""
                thinking = ""
                chunk_tool_calls = []
                
                # Helper function to extract tool calls from various formats
                def extract_tool_calls(tc_list):
                    extracted = []
                    if tc_list:
                        for tc in tc_list:
                            # Generate or get ID
                            tc_id = None
                            if hasattr(tc, 'id'):
                                tc_id = tc.id
                            elif hasattr(tc, 'Id'):
                                tc_id = tc.Id
                            
                            if not tc_id:
                                tc_id = f"call_{uuid.uuid4().hex[:8]}"
                            
                            # Try multiple ways to get the function
                            func = None
                            fn_name = ""
                            fn_args = {}
                            
                            # Try different attribute names
                            for func_attr in ['function', 'Function', 'func', 'Func']:
                                if hasattr(tc, func_attr):
                                    func = getattr(tc, func_attr)
                                    break
                            
                            if func:
                                # Try different ways to get name
                                for name_attr in ['name', 'Name', 'function_name', 'FunctionName']:
                                    if hasattr(func, name_attr):
                                        fn_name = getattr(func, name_attr, "")
                                        break
                                
                                # Try different ways to get arguments
                                for args_attr in ['arguments', 'Arguments', 'args', 'Args', 'parameters', 'Parameters']:
                                    if hasattr(func, args_attr):
                                        fn_args = getattr(func, args_attr, {})
                                        break
                            
                            # Also check if tc itself has name/arguments directly
                            if not fn_name:
                                for name_attr in ['name', 'Name']:
                                    if hasattr(tc, name_attr):
                                        fn_name = getattr(tc, name_attr, "")
                                        break
                            
                            if not fn_args:
                                for args_attr in ['arguments', 'Arguments', 'args', 'Args']:
                                    if hasattr(tc, args_attr):
                                        fn_args = getattr(tc, args_attr, {})
                                        break
                            
                            if fn_name:  # Only add if we found a function name
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
                                
                                # Check for duplicates based on content
                                tc_key = make_tool_call_key(tool_call)
                                if tc_key not in seen_tool_call_keys:
                                    seen_tool_call_keys.add(tc_key)
                                    extracted.append(tool_call)
                    return extracted
                
                # Track what tool calls we get from message to avoid duplicates
                message_tool_calls = None
                
                # Extract content and thinking from message
                if hasattr(chunk, 'message') and chunk.message:
                    content = getattr(chunk.message, "content", "")
                    thinking = getattr(chunk.message, "thinking", "")
                    
                    # Check for tool calls in message
                    if self.supports_feature("tools"):
                        message_tool_calls = getattr(chunk.message, "tool_calls", None)
                        if message_tool_calls:
                            extracted = extract_tool_calls(message_tool_calls)
                            chunk_tool_calls.extend(extracted)
                            aggregated_tool_calls.extend(extracted)
                
                # Also check for tool calls at the chunk level (some models put them here)
                # BUT skip if they're the same object as message.tool_calls
                if hasattr(chunk, 'tool_calls') and self.supports_feature("tools"):
                    chunk_level_tool_calls = chunk.tool_calls
                    
                    # Only process if this is a different object from message.tool_calls
                    if chunk_level_tool_calls is not message_tool_calls:
                        extracted = extract_tool_calls(chunk_level_tool_calls)
                        chunk_tool_calls.extend(extracted)
                        aggregated_tool_calls.extend(extracted)
                
                # Check if this is a final chunk with additional data
                if hasattr(chunk, 'done') and chunk.done:
                    # Some models send tool calls only in the final chunk
                    # Check multiple possible locations
                    final_tool_calls = None
                    
                    # Try message.tool_calls
                    if hasattr(chunk, 'message') and chunk.message:
                        final_tool_calls = getattr(chunk.message, "tool_calls", None)
                    
                    # Try chunk.tool_calls if not found in message and not same object
                    if not final_tool_calls and hasattr(chunk, 'tool_calls'):
                        potential_final = chunk.tool_calls
                        if potential_final is not message_tool_calls:
                            final_tool_calls = potential_final
                    
                    # Process any final tool calls found
                    if final_tool_calls:
                        extracted = extract_tool_calls(final_tool_calls)
                        # Only add if we haven't already processed them
                        for tc in extracted:
                            if tc not in chunk_tool_calls:
                                chunk_tool_calls.append(tc)
                                aggregated_tool_calls.append(tc)
                
                # Track statistics
                if content:
                    total_content_chars += len(content)
                if thinking:
                    total_thinking_chars += len(thinking)
                
                # FIXED: Determine what to stream based on what we have
                stream_content = ""
                
                if is_reasoning_model and thinking:
                    # For reasoning models like GPT-OSS, stream the thinking process
                    stream_content = thinking
                    if chunk_count <= 5:  # Log first few chunks for debugging
                        log.debug(f"Streaming thinking chunk {chunk_count}: '{thinking[:50]}...'")
                elif content:
                    # For regular models or when content is available, stream the content
                    stream_content = content
                    if chunk_count <= 5:  # Log first few chunks for debugging
                        log.debug(f"Streaming content chunk {chunk_count}: '{content[:50]}...'")
                
                # CRITICAL FIX: Check if chunk has meaningful data
                has_content = bool(stream_content)
                has_tools = bool(chunk_tool_calls)
                
                # CRITICAL FIX FOR GRANITE:
                # Granite sends tool calls in first chunk with empty content,
                # then sends content in subsequent chunks.
                # We need to yield the tool calls immediately when we see them,
                # even if there's no content yet.
                
                if has_tools and not tool_calls_yielded:
                    # Yield tool calls immediately when we first see them
                    chunk_data = {
                        "response": stream_content if stream_content else "",  # Empty string instead of None
                        "tool_calls": chunk_tool_calls
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
                    tool_calls_yielded = True
                    
                    # Log tool calls
                    log.debug(f"Model yielded {len(chunk_tool_calls)} tool calls in chunk {chunk_count}")
                    
                elif has_content:
                    # Yield content chunks (without repeating tool calls)
                    chunk_data = {
                        "response": stream_content,
                        "tool_calls": []  # Don't repeat tool calls in content chunks
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
                else:
                    empty_chunk_count += 1
                    # Accurate debug logging
                    if empty_chunk_count <= 3:
                        log.debug(f"Empty chunk {empty_chunk_count}: content='{content}', thinking='{thinking}', tools={len(chunk_tool_calls)}")
                
                # Allow other async tasks to run periodically
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            # FIXED: Final statistics with accurate counts
            log.debug(f"Ollama streaming completed: {chunk_count} total chunks, "
                    f"{empty_chunk_count} empty chunks, "
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