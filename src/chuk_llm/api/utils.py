# chuk_llm/api/utils.py
"""Utility functions for metrics, health checks, and diagnostics."""

from typing import Dict, Any, Optional, List
import asyncio
from .config import _cached_client, get_current_config

def get_metrics() -> Dict[str, Any]:
    """Get metrics from the current client if metrics are enabled.
    
    Returns:
        Dictionary with metrics data, or empty dict if no metrics available
        
    Examples:
        metrics = get_metrics()
        print(f"Total requests: {metrics.get('total_requests', 0)}")
        print(f"Average duration: {metrics.get('average_duration', 0):.2f}s")
    """
    if _cached_client and hasattr(_cached_client, 'middleware_stack'):
        for middleware in _cached_client.middleware_stack.middlewares:
            if hasattr(middleware, 'get_metrics'):
                return middleware.get_metrics()
    return {}

async def health_check() -> Dict[str, Any]:
    """Get health status using your resource manager.
    
    Returns:
        Dictionary with health status information
        
    Examples:
        health = await health_check()
        print(f"Status: {health.get('status', 'unknown')}")
        print(f"Active clients: {health.get('total_clients', 0)}")
    """
    try:
        from chuk_llm.llm.connection_pool import get_llm_health_status
        return await get_llm_health_status()
    except ImportError:
        return {
            "status": "unknown",
            "error": "Health check not available - connection pool not found"
        }

def health_check_sync() -> Dict[str, Any]:
    """Synchronous version of health_check()."""
    return asyncio.run(health_check())

def get_current_client_info() -> Dict[str, Any]:
    """Get information about the currently cached client.
    
    Returns:
        Dictionary with client information
    """
    if not _cached_client:
        return {"status": "no_client", "message": "No client currently cached"}
    
    config = get_current_config()
    
    info = {
        "status": "active",
        "provider": config.get("provider", "unknown"),
        "model": config.get("model", "unknown"),
        "client_type": type(_cached_client).__name__,
        "has_middleware": hasattr(_cached_client, 'middleware_stack'),
    }
    
    # Add middleware info if available
    if hasattr(_cached_client, 'middleware_stack'):
        middleware_names = [type(m).__name__ for m in _cached_client.middleware_stack.middlewares]
        info["middleware"] = middleware_names
    
    return info

async def test_connection(
    provider: str = None,
    model: str = None,
    test_prompt: str = "Hello, this is a connection test."
) -> Dict[str, Any]:
    """Test connection to a specific provider/model.
    
    Args:
        provider: Provider to test (uses current config if None)
        model: Model to test (uses current config if None)
        test_prompt: Simple prompt to send for testing
        
    Returns:
        Dictionary with test results
        
    Examples:
        # Test current configuration
        result = await test_connection()
        
        # Test specific provider
        result = await test_connection(provider="anthropic", model="claude-3-sonnet")
        print(f"Success: {result['success']}")
    """
    from .core import ask
    
    config = get_current_config()
    test_provider = provider or config.get("provider")
    test_model = model or config.get("model")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        response = await ask(
            test_prompt,
            provider=test_provider,
            model=test_model,
            max_tokens=50  # Keep it short for testing
        )
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        return {
            "success": True,
            "provider": test_provider,
            "model": test_model,
            "duration": duration,
            "response_length": len(response),
            "response_preview": response[:100] + "..." if len(response) > 100 else response
        }
        
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        return {
            "success": False,
            "provider": test_provider,
            "model": test_model,
            "duration": duration,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_connection_sync(
    provider: str = None,
    model: str = None,
    test_prompt: str = "Hello, this is a connection test."
) -> Dict[str, Any]:
    """Synchronous version of test_connection()."""
    return asyncio.run(test_connection(provider, model, test_prompt))

async def test_all_providers(
    providers: List[str] = None,
    test_prompt: str = "Hello, this is a connection test."
) -> Dict[str, Dict[str, Any]]:
    """Test connections to multiple providers.
    
    Args:
        providers: List of providers to test (defaults to common ones)
        test_prompt: Simple prompt to send for testing
        
    Returns:
        Dictionary mapping provider names to test results
        
    Examples:
        results = await test_all_providers()
        for provider, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {provider}: {result.get('duration', 0):.2f}s")
    """
    if providers is None:
        providers = ["openai", "anthropic", "google"]
    
    results = {}
    
    # Test providers concurrently
    tasks = []
    for provider in providers:
        task = test_connection(provider=provider, test_prompt=test_prompt)
        tasks.append((provider, task))
    
    responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    for (provider, _), response in zip(tasks, responses):
        if isinstance(response, Exception):
            results[provider] = {
                "success": False,
                "provider": provider,
                "error": str(response),
                "error_type": type(response).__name__
            }
        else:
            results[provider] = response
    
    return results

def test_all_providers_sync(
    providers: List[str] = None,
    test_prompt: str = "Hello, this is a connection test."
) -> Dict[str, Dict[str, Any]]:
    """Synchronous version of test_all_providers()."""
    return asyncio.run(test_all_providers(providers, test_prompt))

def print_diagnostics():
    """Print diagnostic information about the current setup.
    
    This is a convenience function for debugging and troubleshooting.
    """
    print("🔍 ChukLLM Diagnostics")
    print("=" * 50)
    
    # Current config
    config = get_current_config()
    print("\n📋 Current Configuration:")
    for key, value in config.items():
        if key == "api_key" and value:
            value = f"{value[:8]}..." if len(value) > 8 else "***"
        print(f"  {key}: {value}")
    
    # Client info
    print("\n🔧 Client Information:")
    client_info = get_current_client_info()
    for key, value in client_info.items():
        print(f"  {key}: {value}")
    
    # Metrics
    print("\n📊 Metrics:")
    metrics = get_metrics()
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("  No metrics available (enable_metrics=False)")
    
    # Health check
    print("\n🏥 Health Check:")
    try:
        health = health_check_sync()
        for key, value in health.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")

async def cleanup():
    """Cleanup all LLM resources.
    
    This function cleans up connection pools, cached clients, and other resources.
    Useful for proper shutdown in applications.
    """
    global _cached_client
    
    try:
        from chuk_llm.llm.connection_pool import cleanup_llm_resources
        await cleanup_llm_resources()
    except ImportError:
        pass
    
    # Clear cached client
    if _cached_client and hasattr(_cached_client, 'close'):
        await _cached_client.close()
    
    _cached_client = None

def cleanup_sync():
    """Synchronous version of cleanup()."""
    asyncio.run(cleanup())