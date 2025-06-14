"""Synchronous wrappers for simple scripts."""

import asyncio
import warnings
from typing import List, Dict, Any
from .core import ask, stream

def _run_async_safe(coro):
    """Helper to run async functions, handling existing event loop safely."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in an event loop, we can't use asyncio.run()
        raise RuntimeError(
            "Cannot call sync functions from within an async context. "
            "Use the async version instead (e.g., ask() instead of ask_sync())"
        )
    except RuntimeError as e:
        if "cannot call sync functions" in str(e).lower():
            raise e
    
    # Import the persistent loop runner from providers module
    from .providers import _run_async_on_persistent_loop
    return _run_async_on_persistent_loop(coro)

def ask_sync(prompt: str, **kwargs) -> str:
    """Synchronous version of ask().
    
    Args:
        prompt: The message to send
        **kwargs: All the same arguments as ask() (provider, model, temperature, etc.)
        
    Returns:
        The LLM's response as a string
        
    Examples:
        response = ask_sync("Hello!")
        response = ask_sync("Hello!", provider="anthropic")
        response = ask_sync("Hello!", provider="openai", model="gpt-4o")
    """
    return _run_async_safe(ask(prompt, **kwargs))

def stream_sync(prompt: str, **kwargs) -> List[str]:
    """Synchronous version of stream() - returns list of chunks.
    
    Args:
        prompt: The message to send
        **kwargs: All the same arguments as stream()
        
    Returns:
        List of response chunks
        
    Examples:
        chunks = stream_sync("Write a story")
        full_response = "".join(chunks)
    """
    async def collect_stream():
        chunks = []
        async for chunk in stream(prompt, **kwargs):
            chunks.append(chunk)
        return chunks
    
    return _run_async_safe(collect_stream())

def compare_providers(question: str, providers: List[str] = None) -> Dict[str, str]:
    """Ask the same question to multiple providers synchronously.
    
    Args:
        question: The question to ask all providers
        providers: List of provider names (defaults to ["openai", "anthropic"])
        
    Returns:
        Dictionary mapping provider names to responses
        
    Examples:
        results = compare_providers("What is AI?")
        results = compare_providers("Hello!", ["openai", "anthropic", "groq"])
    """
    if providers is None:
        providers = ["openai", "anthropic"]
    
    results = {}
    for provider in providers:
        try:
            results[provider] = ask_sync(question, provider=provider)
        except Exception as e:
            results[provider] = f"Error: {str(e)}"
    
    return results