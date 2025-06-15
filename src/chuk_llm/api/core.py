# chuk_llm/api/core.py
"""
Core ask/stream functions - clean and simple
============================================

Main API functions using dynamic configuration.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from chuk_llm.configuration.config import get_config
from chuk_llm.api.config import get_current_config
from chuk_llm.llm.client import get_client


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
    """
    Ask a question and get a response.
    
    Args:
        prompt: The question/prompt to send
        provider: LLM provider (uses config default if not specified)
        model: Model name (uses provider default if not specified)
        system_prompt: System prompt override
        temperature: Temperature override
        max_tokens: Max tokens override
        tools: Function tools for the LLM
        **kwargs: Additional arguments
        
    Returns:
        The LLM's response as a string
    """
    # Get base configuration
    config = get_current_config()
    
    # Apply parameter overrides
    if provider is not None:
        config["provider"] = provider
    if model is not None:
        config["model"] = model
    if system_prompt is not None:
        config["system_prompt"] = system_prompt
    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    
    # Resolve model if needed
    if not config.get("model") and provider:
        config_manager = get_config()
        try:
            provider_config = config_manager.get_provider(provider)
            config["model"] = provider_config.default_model
        except ValueError:
            pass
    
    # Validate features if needed
    if tools:
        config_manager = get_config()
        try:
            if not config_manager.supports_feature(config["provider"], "tools"):
                raise ValueError(f"Provider {config['provider']} doesn't support function calling")
        except ValueError:
            pass  # Unknown provider, proceed anyway
    
    # Get client
    from chuk_llm.llm.client import get_client
    client = get_client(
        provider=config["provider"],
        model=config["model"],
        api_key=config["api_key"],
        api_base=config["api_base"]
    )
    
    # Build messages
    messages = []
    
    # Add system prompt
    if config.get("system_prompt"):
        messages.append({"role": "system", "content": config["system_prompt"]})
    elif tools:
        # Generate system prompt for tools
        from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
        generator = SystemPromptGenerator()
        system_content = generator.generate_prompt(tools)
        messages.append({"role": "system", "content": system_content})
    else:
        # Default system prompt
        messages.append({
            "role": "system", 
            "content": "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
        })
    
    messages.append({"role": "user", "content": prompt})
    
    # Prepare completion arguments
    completion_args = {"messages": messages}
    
    if tools:
        completion_args["tools"] = tools
    if config.get("temperature") is not None:
        completion_args["temperature"] = config["temperature"]
    if config.get("max_tokens") is not None:
        completion_args["max_tokens"] = config["max_tokens"]
    
    completion_args.update(kwargs)
    
    # Make the request
    response = await client.create_completion(**completion_args)
    
    # Extract response
    if isinstance(response, dict):
        if response.get("error"):
            raise Exception(f"LLM Error: {response.get('error_message', 'Unknown error')}")
        return response.get("response", "")
    
    return str(response)


async def stream(prompt: str, **kwargs) -> AsyncIterator[str]:
    """
    Stream a response token by token.
    
    Args:
        prompt: The question/prompt to send
        **kwargs: Same arguments as ask()
        
    Yields:
        str: Individual tokens/chunks from the LLM response
    """
    # Get base configuration
    config = get_current_config()
    
    # Apply parameter overrides
    provider = kwargs.get('provider')
    model = kwargs.get('model')
    
    if provider is not None:
        config["provider"] = provider
    if model is not None:
        config["model"] = model
    
    # Apply other config overrides
    for key in ['system_prompt', 'temperature', 'max_tokens']:
        if key in kwargs:
            config[key] = kwargs[key]
    
    # Resolve model if needed
    if not config.get("model") and provider:
        config_manager = get_config()
        try:
            provider_config = config_manager.get_provider(provider)
            config["model"] = provider_config.default_model
        except ValueError:
            pass
    
    # Validate streaming support
    config_manager = get_config()
    try:
        if not config_manager.supports_feature(config["provider"], "streaming"):
            raise ValueError(f"Provider {config['provider']} doesn't support streaming")
    except ValueError:
        pass  # Unknown provider, proceed anyway
    
    # Get client
    from chuk_llm.llm.client import get_client
    client = get_client(
        provider=config["provider"],
        model=config["model"],
        api_key=config["api_key"],
        api_base=config["api_base"]
    )
    
    # Build messages (same logic as ask())
    messages = []
    
    if config.get("system_prompt"):
        messages.append({"role": "system", "content": config["system_prompt"]})
    elif kwargs.get("tools"):
        from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator
        generator = SystemPromptGenerator()
        system_content = generator.generate_prompt(kwargs.get("tools"))
        messages.append({"role": "system", "content": system_content})
    else:
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
        })
    
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
    if config.get("temperature") is not None:
        completion_args["temperature"] = config["temperature"]
    if config.get("max_tokens") is not None:
        completion_args["max_tokens"] = config["max_tokens"]
    
    # Stream the response
    try:
        response_stream = client.create_completion(**completion_args)
        
        if hasattr(response_stream, '__aiter__'):
            async for chunk in response_stream:
                if isinstance(chunk, dict):
                    if chunk.get("error"):
                        yield f"[Error: {chunk.get('error_message', 'Unknown error')}]"
                        return
                    content = chunk.get("response", "")
                    if content:
                        yield content
                else:
                    yield str(chunk)
        else:
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