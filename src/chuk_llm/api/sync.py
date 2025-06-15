# chuk_llm/api/sync.py
"""
Clean synchronous wrappers
==========================

Simple sync wrappers for the async API.
"""

import asyncio
import threading
from typing import List, Dict, Any

from .core import ask, stream

# Thread-local storage for event loops
_thread_local = threading.local()


def _get_or_create_loop():
    """Get or create an event loop for the current thread"""
    try:
        loop = asyncio.get_running_loop()
        # We're already in an async context
        raise RuntimeError(
            "Cannot call sync functions from async context. "
            "Use the async version instead."
        )
    except RuntimeError as e:
        if "Cannot call sync functions" in str(e):
            raise e
        # No running loop, we can create one
        pass
    
    # Check if we have a loop for this thread
    if not hasattr(_thread_local, 'loop') or _thread_local.loop.is_closed():
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)
    
    return _thread_local.loop


def _run_async(coro):
    """Run async coroutine safely"""
    try:
        # Check if we're already in an async context
        asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot call sync functions from async context. "
            "Use the async version instead."
        )
    except RuntimeError as e:
        if "Cannot call sync functions" in str(e):
            raise e
    
    # We're not in an async context, safe to use asyncio.run
    return asyncio.run(coro)


def ask_sync(prompt: str, **kwargs) -> str:
    """
    Synchronous version of ask().
    
    Args:
        prompt: The message to send
        **kwargs: All the same arguments as ask()
        
    Returns:
        The LLM's response as a string
    """
    return _run_async(ask(prompt, **kwargs))


def stream_sync(prompt: str, **kwargs) -> List[str]:
    """
    Synchronous version of stream() - returns list of chunks.
    
    Args:
        prompt: The message to send
        **kwargs: All the same arguments as stream()
        
    Returns:
        List of response chunks
    """
    async def collect_chunks():
        chunks = []
        async for chunk in stream(prompt, **kwargs):
            chunks.append(chunk)
        return chunks
    
    return _run_async(collect_chunks())


def compare_providers(question: str, providers: List[str] = None) -> Dict[str, str]:
    """
    Ask the same question to multiple providers.
    
    Args:
        question: The question to ask
        providers: List of provider names (uses available providers if not specified)
        
    Returns:
        Dictionary mapping provider names to responses
    """
    if providers is None:
        from chuk_llm.configuration.config import get_config
        config = get_config()
        all_providers = config.get_all_providers()
        providers = all_providers[:3] if len(all_providers) >= 3 else all_providers
    
    results = {}
    for provider in providers:
        try:
            results[provider] = ask_sync(question, provider=provider)
        except Exception as e:
            results[provider] = f"Error: {str(e)}"
    
    return results


def quick_question(question: str, provider: str = None) -> str:
    """
    Quick one-off question using default or specified provider.
    
    Args:
        question: The question to ask
        provider: Provider to use (uses global default if not specified)
        
    Returns:
        The response
    """
    if not provider:
        from chuk_llm.configuration.config import get_config
        config = get_config()
        settings = config.get_global_settings()
        provider = settings.get("active_provider", "openai")
    
    return ask_sync(question, provider=provider)