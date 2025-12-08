"""
Client Registry - Automatic Client Reuse
=========================================

Provides automatic client caching and reuse to eliminate the overhead
of creating duplicate clients (saves ~12ms per duplicate creation).

Features:
- Automatic client caching by (provider, model, config)
- Thread-safe for concurrent access
- Automatic cleanup on process exit
- Manual cleanup when needed
- Stats and diagnostics

Usage:
    # Automatic reuse - creates once, caches forever
    from chuk_llm.client_registry import get_cached_client

    client1 = get_cached_client("openai", model="gpt-4o", api_key=key)
    client2 = get_cached_client("openai", model="gpt-4o", api_key=key)
    # client2 is the SAME instance as client1 (no 12ms overhead!)

    # Manual cleanup when needed
    from chuk_llm.client_registry import cleanup_registry
    await cleanup_registry()
"""

import asyncio
import atexit
import hashlib
import logging
from threading import Lock

from chuk_llm.llm.core.base import BaseLLMClient

log = logging.getLogger(__name__)

# Global registry - maps cache key to client instance
_client_cache: dict[str, BaseLLMClient] = {}

# Lock for thread-safe access
_cache_lock = Lock()

# Stats
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "total_clients": 0,
}


def _make_cache_key(provider: str, model: str, **kwargs) -> str:
    """
    Create a cache key from provider, model, and config parameters.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4o")
        **kwargs: Additional config (api_key, base_url, etc.)

    Returns:
        Unique cache key string
    """
    # Sort kwargs for consistent key generation
    sorted_kwargs = sorted(kwargs.items())

    # Create key from provider, model, and kwargs (convert ALL to strings)
    key_parts = [str(provider), str(model)] + [
        f"{k}={str(v)}" for k, v in sorted_kwargs
    ]
    key_string = ":".join(key_parts)

    # Hash for privacy (don't expose API keys in logs)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]

    # Readable prefix + hash
    return f"{provider}:{model}:{key_hash}"


def get_cached_client(
    provider: str,
    model: str | None = None,
    **kwargs,
) -> BaseLLMClient:
    """
    Get or create a client with automatic caching.

    This function automatically caches clients by their configuration.
    Identical configurations return the same client instance, avoiding
    the ~12ms overhead of creating duplicate clients.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (optional, uses provider default if None)
        **kwargs: Additional config (api_key, api_base, etc.)

    Returns:
        Cached or newly created client

    Examples:
        >>> # First call creates client (~12ms)
        >>> client1 = get_cached_client("openai", model="gpt-4o", api_key=key)

        >>> # Second call returns cached client (~0.001ms)
        >>> client2 = get_cached_client("openai", model="gpt-4o", api_key=key)
        >>> assert client1 is client2  # Same instance!

        >>> # Different config creates new client
        >>> client3 = get_cached_client("openai", model="gpt-4o-mini", api_key=key)
        >>> assert client3 is not client1  # Different model
    """
    # Import internal function to avoid circular dependency
    from chuk_llm.llm.client import _create_client_internal

    # Create cache key
    cache_key = _make_cache_key(provider, model or "", **kwargs)

    # Thread-safe cache access
    with _cache_lock:
        # Check if client already exists
        if cache_key in _client_cache:
            _cache_stats["hits"] += 1
            log.debug(f"Cache HIT: {cache_key} (total hits: {_cache_stats['hits']})")
            return _client_cache[cache_key]

        # Cache miss - create new client
        _cache_stats["misses"] += 1
        _cache_stats["total_clients"] += 1

        log.debug(
            f"Cache MISS: {cache_key} "
            f"(creating client #{_cache_stats['total_clients']})"
        )

        # Create client using internal function (no caching recursion)
        client = _create_client_internal(provider, model=model, **kwargs)

        # Cache for future use
        _client_cache[cache_key] = client

        return client


def invalidate_client(client: BaseLLMClient) -> bool:
    """
    Remove a client from the cache.

    This should be called when a client is closed to prevent returning
    closed clients from the cache.

    Args:
        client: The client instance to remove

    Returns:
        True if client was found and removed, False otherwise
    """
    with _cache_lock:
        # Find the cache key for this client instance
        for cache_key, cached_client in list(_client_cache.items()):
            if cached_client is client:
                del _client_cache[cache_key]
                log.debug(f"Invalidated cached client: {cache_key}")
                return True
        return False


def is_client_cached(provider: str, model: str | None = None, **kwargs) -> bool:
    """
    Check if a client with this config is already cached.

    Args:
        provider: Provider name
        model: Model name
        **kwargs: Additional config

    Returns:
        True if client is cached
    """
    cache_key = _make_cache_key(provider, model or "", **kwargs)
    with _cache_lock:
        return cache_key in _client_cache


def get_cached_client_count() -> int:
    """
    Get the number of cached clients.

    Returns:
        Number of clients in cache
    """
    with _cache_lock:
        return len(_client_cache)


def get_cache_stats() -> dict[str, int]:
    """
    Get cache statistics.

    Returns:
        Dict with cache hits, misses, and total clients

    Example:
        >>> stats = get_cache_stats()
        >>> print(f"Cache hit rate: {stats['hits'] / (stats['hits'] + stats['misses']):.1%}")
    """
    with _cache_lock:
        return dict(_cache_stats)


async def cleanup_registry() -> None:
    """
    Clean up all cached clients.

    Closes all client connections and clears the cache.
    Should be called when shutting down the application.

    Example:
        >>> # At application shutdown
        >>> await cleanup_registry()
    """
    with _cache_lock:
        clients = list(_client_cache.values())
        _client_cache.clear()
        _cache_stats["total_clients"] = 0

    # Close all clients (outside lock to avoid blocking)
    log.info(f"Cleaning up {len(clients)} cached clients...")

    for client in clients:
        try:
            if hasattr(client, "close"):
                await client.close()
        except Exception as e:
            # Ignore "Event loop is closed" and "different loop" errors during cleanup
            error_str = str(e)
            if (
                "Event loop is closed" not in error_str
                and "different loop" not in error_str
            ):
                log.debug(f"Error closing client: {e}")

    log.info("Client registry cleanup complete")


def cleanup_registry_sync() -> None:
    """
    Synchronous version of cleanup_registry().

    For use in synchronous contexts (e.g., atexit handlers).
    """
    try:
        asyncio.run(cleanup_registry())
    except RuntimeError:
        # Already in event loop - try to schedule cleanup
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(cleanup_registry())
        else:
            asyncio.run(cleanup_registry())


def clear_cache(reset_stats: bool = True) -> int:
    """
    Clear the cache without closing clients.

    Use this when you want to force recreation of clients
    without properly closing existing ones.

    Args:
        reset_stats: Whether to reset cache statistics (default: True)

    Returns:
        Number of clients removed from cache

    Warning:
        This does NOT close existing clients. Use cleanup_registry()
        for proper cleanup with connection closing.
    """
    with _cache_lock:
        count = len(_client_cache)
        _client_cache.clear()
        if reset_stats:
            _cache_stats["hits"] = 0
            _cache_stats["misses"] = 0
            _cache_stats["total_clients"] = 0
        log.info(f"Cleared {count} clients from cache")
        return count


def print_registry_stats() -> None:
    """
    Print human-readable cache statistics.

    Example output:
        Client Registry Statistics
        ==========================
        Cached clients:     5
        Cache hits:        42
        Cache misses:       5
        Hit rate:        89.4%
        Time saved:     ~504ms (42 hits × 12ms)
    """
    stats = get_cache_stats()
    total_requests = stats["hits"] + stats["misses"]
    hit_rate = stats["hits"] / total_requests * 100 if total_requests > 0 else 0
    time_saved_ms = stats["hits"] * 12  # ~12ms per saved creation

    print("\nClient Registry Statistics")
    print("=" * 70)
    print(f"Cached clients:     {stats['total_clients']}")
    print(f"Cache hits:        {stats['hits']}")
    print(f"Cache misses:       {stats['misses']}")
    print(f"Hit rate:        {hit_rate:.1f}%")
    print(f"Time saved:     ~{time_saved_ms}ms ({stats['hits']} hits × 12ms)")
    print("=" * 70)


# Register cleanup on exit
atexit.register(cleanup_registry_sync)
