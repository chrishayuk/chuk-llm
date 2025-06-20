# chuk_llm/api/core.py
"""
Core ask/stream functions with unified configuration and automatic session tracking
==================================================================================

Main API functions using the unified configuration system with integrated session management.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
import logging
import os

from chuk_llm.configuration import get_config, ConfigValidator, Feature
from chuk_llm.api.config import get_current_config
from chuk_llm.llm.client import get_client

# Try to import session manager
try:
    from chuk_ai_session_manager import SessionManager
    _SESSION_AVAILABLE = True
except ImportError:
    _SESSION_AVAILABLE = False
    SessionManager = None

logger = logging.getLogger(__name__)

# Global session manager instance
_global_session_manager = None

# Check if sessions should be disabled via environment variable
_SESSIONS_ENABLED = _SESSION_AVAILABLE and os.getenv("CHUK_LLM_DISABLE_SESSIONS", "").lower() not in ("true", "1", "yes")


def _get_session_manager() -> Optional['SessionManager']:
    """Get or create the global session manager."""
    global _global_session_manager
    
    if not _SESSIONS_ENABLED:
        return None
    
    if _global_session_manager is None:
        try:
            # Get system prompt from config if available
            config = get_current_config()
            system_prompt = config.get("system_prompt")
            
            _global_session_manager = SessionManager(
                system_prompt=system_prompt,
                infinite_context=True,  # Enable by default
                token_threshold=4000    # Reasonable default
            )
        except Exception as e:
            logger.debug(f"Could not create session manager: {e}")
            return None
    
    return _global_session_manager


async def ask(
    prompt: str,
    *,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    tools: List[Dict[str, Any]] = None,
    json_mode: bool = False,
    context: str = None,
    previous_messages: List[Dict[str, str]] = None,
    **kwargs
) -> str:
    """
    Ask a question and get a response with unified configuration and automatic session tracking.
    
    Args:
        prompt: The question/prompt to send
        provider: LLM provider (uses config default if not specified)
        model: Model name (uses provider default if not specified)
        system_prompt: System prompt override
        temperature: Temperature override
        max_tokens: Max tokens override
        tools: Function tools for the LLM
        json_mode: Enable JSON mode response
        context: Additional context for the question (stateless)
        previous_messages: Previous messages for context (stateless)
        **kwargs: Additional arguments
        
    Returns:
        The LLM's response as a string
    """
    # Get session manager
    session_manager = _get_session_manager()
    
    # Track user message if session is available
    if session_manager:
        try:
            await session_manager.user_says(prompt)
        except Exception as e:
            logger.debug(f"Session tracking error (user): {e}")
    
    # Get base configuration
    config = get_current_config()
    
    # Determine effective provider and model
    effective_provider = provider or config["provider"]
    effective_model = model or config["model"]
    
    # Resolve provider-specific settings when provider is overridden
    config_manager = get_config()
    
    if provider is not None:
        # Provider override - resolve all provider-specific settings
        try:
            provider_config = config_manager.get_provider(provider)
            effective_api_key = config_manager.get_api_key(provider)
            effective_api_base = provider_config.api_base
            
            # Resolve model if needed
            if model is None:
                effective_model = provider_config.default_model
                
        except Exception as e:
            logger.warning(f"Could not resolve provider '{provider}': {e}")
            # Fallback to cached config
            effective_api_key = config["api_key"] 
            effective_api_base = config["api_base"]
    else:
        # No provider override - use cached config
        effective_api_key = config["api_key"]
        effective_api_base = config["api_base"]
        
        # Still resolve model if needed
        if not effective_model:
            try:
                provider_config = config_manager.get_provider(effective_provider)
                effective_model = provider_config.default_model
            except Exception:
                pass
    
    # Update session system prompt if provided
    if system_prompt and session_manager:
        try:
            await session_manager.update_system_prompt(system_prompt)
        except Exception as e:
            logger.debug(f"Could not update session system prompt: {e}")
    
    # Build effective configuration
    effective_config = {
        "provider": effective_provider,
        "model": effective_model,
        "api_key": effective_api_key,
        "api_base": effective_api_base,
        "system_prompt": system_prompt or config.get("system_prompt"),
        "temperature": temperature if temperature is not None else config.get("temperature"),
        "max_tokens": max_tokens if max_tokens is not None else config.get("max_tokens"),
    }
    
    # Validate request compatibility
    is_valid, issues = ConfigValidator.validate_request_compatibility(
        provider_name=effective_provider,
        model=effective_model,
        tools=tools,
        stream=False,
        **{"response_format": "json" if json_mode else None}
    )
    
    if not is_valid:
        # Log warnings but don't fail - allow fallbacks
        for issue in issues:
            logger.warning(f"Request compatibility issue: {issue}")
    
    # Get client with correct parameters
    client = get_client(
        provider=effective_config["provider"],
        model=effective_config["model"],
        api_key=effective_config["api_key"],
        api_base=effective_config["api_base"]
    )
    
    # Build messages with intelligent system prompt handling
    messages = _build_messages(
        prompt=prompt,
        system_prompt=effective_config.get("system_prompt"),
        tools=tools,
        provider=effective_provider,
        model=effective_model,
        context=context,
        previous_messages=previous_messages
    )
    
    # Remove context and previous_messages from kwargs if present
    completion_kwargs = kwargs.copy()
    completion_kwargs.pop('context', None)
    completion_kwargs.pop('previous_messages', None)
    
    # Prepare completion arguments
    completion_args = {"messages": messages}
    
    # Add tools if supported
    if tools:
        try:
            if config_manager.supports_feature(effective_provider, Feature.TOOLS, effective_model):
                completion_args["tools"] = tools
            else:
                logger.warning(f"{effective_provider}/{effective_model} doesn't support tools, ignoring")
        except Exception:
            # Unknown provider, try anyway
            completion_args["tools"] = tools
    
    # Add JSON mode if requested and supported
    if json_mode:
        try:
            if config_manager.supports_feature(effective_provider, Feature.JSON_MODE, effective_model):
                if effective_provider == "openai":
                    completion_args["response_format"] = {"type": "json_object"}
                elif effective_provider == "gemini":
                    completion_args.setdefault("generation_config", {})["response_mime_type"] = "application/json"
                # For other providers, we'll add instruction to system message
            else:
                logger.warning(f"{effective_provider}/{effective_model} doesn't support JSON mode")
                # Add JSON instruction to system message
                _add_json_instruction_to_messages(messages)
        except Exception:
            # Unknown provider, try anyway
            if effective_provider == "openai":
                completion_args["response_format"] = {"type": "json_object"}
    
    # Add temperature and max_tokens
    if effective_config.get("temperature") is not None:
        completion_args["temperature"] = effective_config["temperature"]
    if effective_config.get("max_tokens") is not None:
        completion_args["max_tokens"] = effective_config["max_tokens"]
    
    # Add any additional kwargs
    completion_args.update(completion_kwargs)
    
    # Make the request
    try:
        response = await client.create_completion(**completion_args)
        
        # Extract response
        if isinstance(response, dict):
            if response.get("error"):
                raise Exception(f"LLM Error: {response.get('error_message', 'Unknown error')}")
            response_text = response.get("response", "")
        else:
            response_text = str(response)
        
        # Track AI response if session is available
        if session_manager:
            try:
                await session_manager.ai_responds(
                    response_text,
                    model=effective_model,
                    provider=effective_provider
                )
                
                # Track tool usage if tools were provided
                if tools and isinstance(response, dict) and "tool_calls" in response:
                    for tool_call in response["tool_calls"]:
                        await session_manager.tool_used(
                            tool_name=tool_call.get("name", "unknown"),
                            arguments=tool_call.get("arguments", {}),
                            result=tool_call.get("result", {})
                        )
            except Exception as e:
                logger.debug(f"Session tracking error (response): {e}")
        
        return response_text
        
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise


async def stream(prompt: str, context: str = None, previous_messages: List[Dict[str, str]] = None, **kwargs) -> AsyncIterator[str]:
    """
    Stream a response token by token with unified configuration and automatic session tracking.
    
    Args:
        prompt: The question/prompt to send
        context: Additional context for the question (stateless)
        previous_messages: Previous messages for context (stateless)
        **kwargs: Same arguments as ask() plus streaming-specific options
        
    Yields:
        str: Individual tokens/chunks from the LLM response
    """
    # Get session manager
    session_manager = _get_session_manager()
    
    # Track user message if session is available
    if session_manager:
        try:
            await session_manager.user_says(prompt)
        except Exception as e:
            logger.debug(f"Session tracking error (user): {e}")
    
    # Collect full response for session tracking
    full_response = ""
    
    # Get base configuration
    config = get_current_config()
    
    # Extract parameters
    provider = kwargs.get('provider')
    model = kwargs.get('model')
    
    # Determine effective provider and settings (same logic as ask())
    effective_provider = provider or config["provider"]
    effective_model = model or config["model"]
    
    # Resolve provider-specific settings
    config_manager = get_config()
    
    if provider is not None:
        try:
            provider_config = config_manager.get_provider(provider)
            effective_api_key = config_manager.get_api_key(provider)
            effective_api_base = provider_config.api_base
            
            if model is None:
                effective_model = provider_config.default_model
                
        except Exception as e:
            logger.warning(f"Could not resolve provider '{provider}': {e}")
            effective_api_key = config["api_key"]
            effective_api_base = config["api_base"]
    else:
        effective_api_key = config["api_key"]
        effective_api_base = config["api_base"]
        
        if not effective_model:
            try:
                provider_config = config_manager.get_provider(effective_provider)
                effective_model = provider_config.default_model
            except Exception:
                pass
    
    # Update session system prompt if provided
    system_prompt = kwargs.get("system_prompt")
    if system_prompt and session_manager:
        try:
            await session_manager.update_system_prompt(system_prompt)
        except Exception as e:
            logger.debug(f"Could not update session system prompt: {e}")
    
    # Build effective configuration
    effective_config = {
        "provider": effective_provider,
        "model": effective_model,
        "api_key": effective_api_key,
        "api_base": effective_api_base,
        "system_prompt": system_prompt or config.get("system_prompt"),
        "temperature": kwargs.get("temperature") if "temperature" in kwargs else config.get("temperature"),
        "max_tokens": kwargs.get("max_tokens") if "max_tokens" in kwargs else config.get("max_tokens"),
    }
    
    # Validate streaming support
    try:
        if not config_manager.supports_feature(effective_provider, Feature.STREAMING, effective_model):
            logger.warning(f"{effective_provider}/{effective_model} doesn't support streaming")
            # Fall back to non-streaming
            response = await ask(prompt, **kwargs)
            
            # Track the response if session is available
            if session_manager:
                try:
                    await session_manager.ai_responds(
                        response,
                        model=effective_model,
                        provider=effective_provider
                    )
                except Exception as e:
                    logger.debug(f"Session tracking error (response): {e}")
            
            yield response
            return
    except Exception:
        pass  # Unknown provider, proceed anyway
    
    # Get client
    client = get_client(
        provider=effective_config["provider"],
        model=effective_config["model"],
        api_key=effective_config["api_key"],
        api_base=effective_config["api_base"]
    )
    
    # Build messages
    messages = _build_messages(
        prompt=prompt,
        system_prompt=effective_config.get("system_prompt"),
        tools=kwargs.get("tools"),
        provider=effective_provider,
        model=effective_model,
        context=context,
        previous_messages=previous_messages
    )
    
    # Remove context and previous_messages from kwargs if present
    completion_kwargs = kwargs.copy()
    completion_kwargs.pop('context', None)
    completion_kwargs.pop('previous_messages', None)
    
    # Prepare streaming arguments
    completion_args = {
        "messages": messages,
        "stream": True,
    }
    
    # Add tools if supported
    tools = kwargs.get("tools")
    if tools:
        try:
            if config_manager.supports_feature(effective_provider, Feature.TOOLS, effective_model):
                completion_args["tools"] = tools
        except Exception:
            completion_args["tools"] = tools
    
    # Add non-config kwargs
    non_config_kwargs = {k: v for k, v in completion_kwargs.items() 
                        if k not in ['provider', 'model', 'system_prompt', 'temperature', 'max_tokens']}
    completion_args.update(non_config_kwargs)
    
    # Add config parameters
    if effective_config.get("temperature") is not None:
        completion_args["temperature"] = effective_config["temperature"]
    if effective_config.get("max_tokens") is not None:
        completion_args["max_tokens"] = effective_config["max_tokens"]
    
    # Stream the response
    try:
        # When streaming, create_completion returns an async generator directly
        response_stream = client.create_completion(**completion_args)
        
        # Check if it's already an async generator (no await needed)
        if hasattr(response_stream, '__aiter__'):
            async for chunk in response_stream:
                if isinstance(chunk, dict):
                    if chunk.get("error"):
                        error_msg = f"[Error: {chunk.get('error_message', 'Unknown error')}]"
                        full_response += error_msg
                        yield error_msg
                        return
                    content = chunk.get("response", "")
                    if content:
                        full_response += content
                        yield content
                else:
                    chunk_str = str(chunk)
                    full_response += chunk_str
                    yield chunk_str
        else:
            # It might be a coroutine that needs awaiting
            result = await response_stream
            
            if hasattr(result, '__aiter__'):
                # Result is an async generator
                async for chunk in result:
                    if isinstance(chunk, dict):
                        if chunk.get("error"):
                            error_msg = f"[Error: {chunk.get('error_message', 'Unknown error')}]"
                            full_response += error_msg
                            yield error_msg
                            return
                        content = chunk.get("response", "")
                        if content:
                            full_response += content
                            yield content
                    else:
                        chunk_str = str(chunk)
                        full_response += chunk_str
                        yield chunk_str
            else:
                # Handle non-streaming response
                if isinstance(result, dict):
                    if result.get("error"):
                        error_msg = f"[Error: {result.get('error_message', 'Unknown error')}]"
                        full_response = error_msg
                        yield error_msg
                    else:
                        content = result.get("response", "")
                        full_response = content
                        yield content
                else:
                    result_str = str(result)
                    full_response = result_str
                    yield result_str
        
        # Track complete response if session is available
        if session_manager and full_response:
            try:
                await session_manager.ai_responds(
                    full_response,
                    model=effective_model,
                    provider=effective_provider
                )
            except Exception as e:
                logger.debug(f"Session tracking error (response): {e}")
                
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield f"[Streaming Error: {str(e)}]"


def _build_messages(
    prompt: str, 
    system_prompt: Optional[str], 
    tools: Optional[List[Dict[str, Any]]], 
    provider: str,
    model: Optional[str],
    context: Optional[str] = None,
    previous_messages: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    """Build messages array with intelligent system prompt handling"""
    messages = []
    
    # Determine system prompt
    if system_prompt:
        system_content = system_prompt
    elif tools:
        # Generate system prompt for tools
        try:
            from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
            generator = SystemPromptGenerator()
            system_content = generator.generate_prompt(tools)
        except ImportError:
            system_content = "You are a helpful AI assistant with access to function calling tools."
    else:
        # Default system prompt
        system_content = "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
    
    # Add context to system prompt if provided
    if context:
        system_content += f"\n\nContext: {context}"
    
    # Add system message if provider supports it
    try:
        config_manager = get_config()
        if config_manager.supports_feature(provider, Feature.SYSTEM_MESSAGES, model):
            messages.append({"role": "system", "content": system_content})
        else:
            # Prepend system content to user message for providers that don't support system messages
            prompt = f"System: {system_content}\n\nUser: {prompt}"
    except Exception:
        # Unknown provider, assume it supports system messages
        messages.append({"role": "system", "content": system_content})
    
    # Add previous messages if provided (for stateless context)
    if previous_messages:
        messages.extend(previous_messages)
    
    messages.append({"role": "user", "content": prompt})
    return messages


def _add_json_instruction_to_messages(messages: List[Dict[str, Any]]):
    """Add JSON mode instruction to system message for providers without native support"""
    json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. Do not include any text outside the JSON structure."
    
    # Find system message and add instruction
    for message in messages:
        if message.get("role") == "system":
            message["content"] += json_instruction
            return
    
    # No system message found, add one
    messages.insert(0, {
        "role": "system",
        "content": f"You are a helpful AI assistant.{json_instruction}"
    })


# Session management functions
async def get_session_stats(include_all_segments: bool = False) -> Dict[str, Any]:
    """Get current session statistics."""
    session_manager = _get_session_manager()
    if session_manager:
        try:
            return await session_manager.get_stats(include_all_segments=include_all_segments)
        except Exception as e:
            logger.debug(f"Could not get session stats: {e}")
    
    return {
        "sessions_enabled": _SESSIONS_ENABLED,
        "session_available": False,
        "message": "No active session"
    }


async def get_session_history(include_all_segments: bool = False) -> List[Dict[str, Any]]:
    """Get current session conversation history."""
    session_manager = _get_session_manager()
    if session_manager:
        try:
            return await session_manager.get_conversation(include_all_segments=include_all_segments)
        except Exception as e:
            logger.debug(f"Could not get session history: {e}")
    
    return []


def get_current_session_id() -> Optional[str]:
    """Get the current session ID if available."""
    session_manager = _get_session_manager()
    return session_manager.session_id if session_manager else None


def reset_session():
    """Reset the current session (start a new one)."""
    global _global_session_manager
    _global_session_manager = None
    logger.info("Session reset - new session will be created on next call")


def disable_sessions():
    """Disable session tracking for the current process."""
    global _SESSIONS_ENABLED, _global_session_manager
    _SESSIONS_ENABLED = False
    _global_session_manager = None
    logger.info("Session tracking disabled")


def enable_sessions():
    """Re-enable session tracking if available."""
    global _SESSIONS_ENABLED
    if _SESSION_AVAILABLE:
        _SESSIONS_ENABLED = True
        logger.info("Session tracking enabled")
    else:
        logger.warning("Cannot enable sessions - chuk-ai-session-manager not installed")


# Enhanced convenience functions
async def ask_with_tools(
    prompt: str,
    tools: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Ask with function calling tools and return structured response"""
    response = await ask(prompt, tools=tools, **kwargs)
    
    return {
        "response": response,
        "tools_used": tools,
        "provider": kwargs.get("provider") or get_current_config()["provider"],
        "model": kwargs.get("model") or get_current_config()["model"],
        "session_id": get_current_session_id()
    }


async def ask_json(prompt: str, **kwargs) -> str:
    """Ask for a JSON response"""
    return await ask(prompt, json_mode=True, **kwargs)


async def quick_ask(prompt: str, provider: str = None) -> str:
    """Quick ask with optional provider override"""
    return await ask(prompt, provider=provider)


async def multi_provider_ask(prompt: str, providers: List[str]) -> Dict[str, str]:
    """Ask the same question to multiple providers"""
    import asyncio
    
    async def ask_provider(provider: str) -> tuple[str, str]:
        try:
            response = await ask(prompt, provider=provider)
            return provider, response
        except Exception as e:
            return provider, f"Error: {e}"
    
    tasks = [ask_provider(provider) for provider in providers]
    results = await asyncio.gather(*tasks)
    
    return dict(results)


# Validation helpers
def validate_request(
    prompt: str,
    provider: str = None,
    model: str = None,
    tools: List[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validate a request before sending"""
    config = get_current_config()
    effective_provider = provider or config["provider"]
    effective_model = model or config["model"]
    
    # Build fake messages to check for vision content
    messages = [{"role": "user", "content": prompt}]
    
    is_valid, issues = ConfigValidator.validate_request_compatibility(
        provider_name=effective_provider,
        model=effective_model,
        messages=messages,
        tools=tools,
        stream=kwargs.get("stream", False),
        **kwargs
    )
    
    return {
        "valid": is_valid,
        "issues": issues,
        "provider": effective_provider,
        "model": effective_model
    }