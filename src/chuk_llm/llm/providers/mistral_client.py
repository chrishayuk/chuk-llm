# src/chuk_llm/llm/providers/mistral_client.py

"""
Mistral Le Plateforme chat-completion adapter

Features
--------
* Full support for Mistral's API including vision, function calling, and streaming
* Dynamic model capability detection using patterns
* Real async streaming without buffering
* Vision capabilities for supported models
* Function calling support for compatible models
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# Import Mistral SDK
try:
    from mistralai import Mistral
except ImportError:
    raise ImportError(
        "mistralai package is required for Mistral provider. "
        "Install with: pip install mistralai"
    )

# Base imports
from chuk_llm.llm.core.base import BaseLLMClient

log = logging.getLogger(__name__)

class MistralLLMClient(BaseLLMClient):
    """
    Adapter for Mistral Le Plateforme API with full feature support.
    
    Uses dynamic capability detection based on model patterns rather than
    hardcoded model lists for better scalability.
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        self.provider_name = "mistral"
        
        # Initialize Mistral client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["server_url"] = api_base
            
        self.client = Mistral(**client_kwargs)
        
        log.info(f"MistralLLMClient initialized with model: {model}")

    def _supports_function_calling(self) -> bool:
        """Check if current model supports function calling based on patterns"""
        function_calling_patterns = [
            "mistral-large", "mistral-medium", "mistral-small",
            "codestral", "devstral", "ministral", "pixtral", "nemo"
        ]
        return any(pattern in self.model.lower() for pattern in function_calling_patterns)

    def _supports_vision(self) -> bool:
        """Check if current model supports vision based on patterns"""
        vision_patterns = ["pixtral", "mistral-small", "mistral-medium"]
        return any(pattern in self.model.lower() for pattern in vision_patterns)

    def _convert_messages_to_mistral_format(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert ChatML messages to Mistral format"""
        mistral_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # Handle different message types
            if role == "system":
                # Mistral supports system messages directly
                mistral_messages.append({
                    "role": "system",
                    "content": content
                })
            
            elif role == "user":
                if isinstance(content, str):
                    # Simple text message
                    mistral_messages.append({
                        "role": "user", 
                        "content": content
                    })
                elif isinstance(content, list):
                    # Multimodal message (text + images)
                    mistral_content = []
                    for item in content:
                        if item.get("type") == "text":
                            mistral_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") == "image_url":
                            # Handle both URL and base64 formats
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = str(image_url)
                            
                            mistral_content.append({
                                "type": "image_url",
                                "image_url": url
                            })
                    
                    mistral_messages.append({
                        "role": "user",
                        "content": mistral_content
                    })
            
            elif role == "assistant":
                # Handle assistant messages with potential tool calls
                if msg.get("tool_calls"):
                    # Convert tool calls to Mistral format
                    tool_calls = []
                    for tc in msg["tool_calls"]:
                        tool_calls.append({
                            "id": tc.get("id"),
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        })
                    
                    mistral_messages.append({
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": tool_calls
                    })
                else:
                    mistral_messages.append({
                        "role": "assistant",
                        "content": content or ""
                    })
            
            elif role == "tool":
                # Tool response messages
                mistral_messages.append({
                    "role": "tool",
                    "name": msg.get("name", ""),
                    "content": content or "",
                    "tool_call_id": msg.get("tool_call_id", "")
                })
        
        return mistral_messages

    def _normalize_mistral_response(self, response: Any) -> Dict[str, Any]:
        """Convert Mistral response to standard format"""
        # Handle both response types
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message
            
            content = getattr(message, 'content', '') or ''
            tool_calls = []
            
            # Extract tool calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            # Return standard format
            if tool_calls:
                return {"response": None, "tool_calls": tool_calls}
            else:
                return {"response": content, "tool_calls": []}
        
        # Fallback for unexpected response format
        return {"response": str(response), "tool_calls": []}

    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[AsyncIterator[Dict[str, Any]], Any]:
        """
        Create completion with Mistral API.
        
        Args:
            messages: ChatML-style messages
            tools: OpenAI-style tool definitions
            stream: Whether to stream response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            AsyncIterator for streaming, awaitable for non-streaming
        """
        # Convert messages to Mistral format
        mistral_messages = self._convert_messages_to_mistral_format(messages)
        
        # Validate tool usage
        if tools and not self._supports_function_calling():
            log.warning(f"Model {self.model} does not support function calling")
            tools = None
        
        # Check for vision content
        has_vision = any(
            isinstance(msg.get("content"), list) 
            for msg in messages
        )
        if has_vision and not self._supports_vision():
            log.warning(f"Model {self.model} does not support vision")
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": mistral_messages,
            **kwargs
        }
        
        # Add tools if provided and supported
        if tools:
            request_params["tools"] = tools
            # Set tool_choice to "auto" by default if not specified
            if "tool_choice" not in kwargs:
                request_params["tool_choice"] = "auto"
        
        if stream:
            return self._stream_completion_async(request_params)
        else:
            return self._regular_completion(request_params)

    async def _stream_completion_async(
        self, 
        request_params: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Real streaming using Mistral's async streaming API.
        """
        try:
            log.debug("Starting Mistral streaming...")
            
            # Use Mistral's streaming endpoint
            stream = self.client.chat.stream(
                **request_params
            )
            
            chunk_count = 0
            
            # Process streaming response
            for chunk in stream:
                chunk_count += 1
                
                if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                    choices = chunk.data.choices
                    if choices:
                        choice = choices[0]
                        
                        # Extract content from delta
                        content = ""
                        tool_calls = []
                        
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                            
                            # Get content
                            if hasattr(delta, 'content') and delta.content:
                                content = delta.content
                            
                            # Get tool calls
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                for tc in delta.tool_calls:
                                    tool_calls.append({
                                        "id": getattr(tc, 'id', f"call_{uuid.uuid4().hex[:8]}"),
                                        "type": getattr(tc, 'type', 'function'),
                                        "function": {
                                            "name": getattr(tc.function, 'name', ''),
                                            "arguments": getattr(tc.function, 'arguments', '')
                                        }
                                    })
                        
                        # Yield chunk if it has content
                        if content or tool_calls:
                            yield {
                                "response": content,
                                "tool_calls": tool_calls
                            }
                
                # Allow other async tasks to run
                if chunk_count % 10 == 0:
                    await asyncio.sleep(0)
            
            log.debug(f"Mistral streaming completed with {chunk_count} chunks")
        
        except Exception as e:
            log.error(f"Error in Mistral streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    async def _regular_completion(
        self, 
        request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming completion using async execution."""
        try:
            def _sync_completion():
                return self.client.chat.complete(**request_params)
            
            # Run sync call in thread to avoid blocking
            response = await asyncio.to_thread(_sync_completion)
            
            # Normalize response
            return self._normalize_mistral_response(response)
            
        except Exception as e:
            log.error(f"Error in Mistral completion: {e}")
            return {
                "response": f"Error: {str(e)}",
                "tool_calls": [],
                "error": True
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "mistral",
            "model": self.model,
            "supports_function_calling": self._supports_function_calling(),
            "supports_vision": self._supports_vision(),
            "supports_streaming": True,
            "supports_system_messages": True
        }

    async def close(self):
        """Cleanup resources"""
        # Mistral client doesn't require explicit cleanup
        pass