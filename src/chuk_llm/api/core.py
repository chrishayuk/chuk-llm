# src/chuk_llm/api/core.py
"""Core ask/stream functions for the simple API."""

from typing import List, Dict, Any, Optional, AsyncIterator
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
from .config import get_current_config, get_client_for_config
from .provider_utils import get_provider_default_model

async def ask(
    prompt: str,
    *,
    provider: str = None,
    model: str = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    tools: List[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Ask a question and get a response.
    
    Args:
        prompt: The question/prompt to send
        provider: LLM provider (defaults from global config)
        model: Model name (defaults from provider config)
        system_prompt: Override default system prompt
        temperature: Override default temperature
        max_tokens: Override default max tokens
        tools: Function tools for the LLM to use
        **kwargs: Additional arguments
        
    Returns:
        The LLM's response as a string
    """
    # Get the current global config
    final_config = get_current_config().copy()
    
    # Apply parameter overrides
    if provider is not None:
        final_config["provider"] = provider
    if model is not None:
        final_config["model"] = model
    if system_prompt is not None:
        final_config["system_prompt"] = system_prompt
    if temperature is not None:
        final_config["temperature"] = temperature
    if max_tokens is not None:
        final_config["max_tokens"] = max_tokens
    
    # If model is still None or we changed provider, get the default for the provider
    if final_config.get("model") is None or (provider is not None and model is None):
        provider_default = get_provider_default_model(final_config["provider"])
        if provider_default:
            final_config["model"] = provider_default
        else:
            # Fallback to trying the existing provider config system
            try:
                from chuk_llm.llm.configuration.provider_config import ProviderConfig
                config_mgr = ProviderConfig()
                provider_config = config_mgr.get_provider_config(final_config["provider"])
                fallback_model = provider_config.get("default_model")
                if fallback_model:
                    final_config["model"] = fallback_model
            except Exception:
                # Keep the existing model from global config as last resort
                pass
    
    # Get the client
    client = get_client_for_config(final_config)
    
    # Build messages
    messages = []
    
    # Add system prompt
    if final_config.get("system_prompt"):
        messages.append({"role": "system", "content": final_config["system_prompt"]})
    elif tools:
        # Use your existing system prompt generator only when tools are provided
        system_generator = SystemPromptGenerator()
        system_content = system_generator.generate_prompt(tools)
        messages.append({"role": "system", "content": system_content})
    else:
        # Simple system prompt for basic conversations without tools
        simple_prompt = "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
        messages.append({"role": "system", "content": simple_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Prepare completion arguments
    completion_args = {"messages": messages}
    if tools:
        completion_args["tools"] = tools
    if final_config.get("temperature") is not None:
        completion_args["temperature"] = final_config["temperature"]
    if final_config.get("max_tokens") is not None:
        completion_args["max_tokens"] = final_config["max_tokens"]
    
    # Add any additional kwargs
    completion_args.update(kwargs)
    
    # Make the request using your client
    response = await client.create_completion(**completion_args)
    
    # Extract response text
    if isinstance(response, dict):
        if response.get("error"):
            raise Exception(f"LLM Error: {response.get('error_message', 'Unknown error')}")
        return response.get("response", "")
    
    return str(response)

async def stream(prompt: str, **kwargs) -> AsyncIterator[str]:
    """Stream a response token by token.
    
    Args:
        prompt: The question/prompt to send
        **kwargs: Same arguments as ask(), plus streaming options
        
    Yields:
        str: Individual tokens/chunks from the LLM response
    """
    # Get config and apply overrides
    final_config = get_current_config().copy()
    
    # Extract config-relevant kwargs
    provider = kwargs.get('provider')
    model = kwargs.get('model')
    
    if provider is not None:
        final_config["provider"] = provider
    if model is not None:
        final_config["model"] = model
        
    # Apply other config overrides
    config_keys = ['system_prompt', 'temperature', 'max_tokens']
    for key in config_keys:
        if key in kwargs:
            final_config[key] = kwargs[key]
    
    # If model is still None or we changed provider, get the default for the provider
    if final_config.get("model") is None or (provider is not None and model is None):
        provider_default = get_provider_default_model(final_config["provider"])
        if provider_default:
            final_config["model"] = provider_default
    
    # Get client
    client = get_client_for_config(final_config)
    
    # Build messages
    messages = []
    if final_config.get("system_prompt"):
        messages.append({"role": "system", "content": final_config["system_prompt"]})
    elif kwargs.get("tools"):
        system_generator = SystemPromptGenerator()
        system_content = system_generator.generate_prompt(kwargs.get("tools"))
        messages.append({"role": "system", "content": system_content})
    else:
        # Simple system prompt for basic conversations
        simple_prompt = "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
        messages.append({"role": "system", "content": simple_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Prepare streaming arguments
    completion_args = {
        "messages": messages,
        "stream": True,
    }
    
    # Add non-config kwargs
    non_config_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['provider', 'model', 'system_prompt', 'temperature', 'max_tokens']}
    completion_args.update(non_config_kwargs)
    
    # Add config parameters
    if final_config.get("temperature") is not None:
        completion_args["temperature"] = final_config["temperature"]
    if final_config.get("max_tokens") is not None:
        completion_args["max_tokens"] = final_config["max_tokens"]
    
    # Stream using client - FIXED VERSION
    try:
        # DON'T await - the client.create_completion returns an async generator directly when stream=True
        response_stream = client.create_completion(**completion_args)
        
        # Check if it's an async generator/iterator
        if hasattr(response_stream, '__aiter__'):
            # It's an async iterator - iterate directly
            async for chunk in response_stream:
                if isinstance(chunk, dict):
                    if chunk.get("error"):
                        yield f"[Error: {chunk.get('error_message', 'Unknown error')}]"
                        return  # Stop streaming on error
                    content = chunk.get("response", "")
                    if content:
                        yield content
                else:
                    # Handle direct string chunks
                    yield str(chunk)
        else:
            # If it's not an async iterator, we need to await it
            awaited_response = await response_stream
            if isinstance(awaited_response, dict):
                if awaited_response.get("error"):
                    yield f"[Error: {awaited_response.get('error_message', 'Unknown error')}]"
                else:
                    yield awaited_response.get("response", "")
            else:
                yield str(awaited_response)
            
    except Exception as e:
        yield f"[Streaming Error: {str(e)}]"