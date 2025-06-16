# chuk_llm/api/utils.py
"""Utility functions for metrics, health checks, and diagnostics."""

from typing import Dict, Any, Optional, List
import asyncio
from chuk_llm.configuration.unified_config import get_config
from chuk_llm.api.config import get_current_config

# Global cached client for compatibility with tests
_cached_client = None


def get_metrics() -> Dict[str, Any]:
    """Get metrics from cached client middleware."""
    global _cached_client
    
    if _cached_client is None:
        return {}
    
    # Check if client has middleware stack
    if not hasattr(_cached_client, 'middleware_stack'):
        return {}
    
    # Look for middleware with get_metrics method
    try:
        for middleware in _cached_client.middleware_stack.middlewares:
            if hasattr(middleware, 'get_metrics'):
                return middleware.get_metrics()
    except (AttributeError, TypeError):
        pass
    
    return {}


async def health_check() -> Dict[str, Any]:
    """Get health status using clean configuration system."""
    try:
        # Try to import connection pool module
        try:
            from chuk_llm.llm.connection_pool import get_llm_health_status
            return await get_llm_health_status()
        except ImportError:
            return {
                "status": "unknown",
                "error": "Health check not available - connection pool not found"
            }
        
    except Exception as e:
        # Re-raise other exceptions
        raise e


def health_check_sync() -> Dict[str, Any]:
    """Synchronous version of health_check()."""
    return asyncio.run(health_check())


def get_current_client_info() -> Dict[str, Any]:
    """Get information about the current cached client."""
    global _cached_client
    
    if _cached_client is None:
        return {
            "status": "no_client",
            "message": "No client currently cached"
        }
    
    try:
        current_config = get_current_config()
        
        info = {
            "status": "active",
            "provider": current_config.get("provider", "unknown"),
            "model": current_config.get("model", "unknown"),
            "client_type": _cached_client.__class__.__name__,
            "has_middleware": hasattr(_cached_client, 'middleware_stack'),
        }
        
        # Add middleware info if available
        if hasattr(_cached_client, 'middleware_stack'):
            try:
                middleware_names = [
                    middleware.__class__.__name__ 
                    for middleware in _cached_client.middleware_stack.middlewares
                ]
                info["middleware"] = middleware_names
            except (AttributeError, TypeError):
                pass
        
        return info
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


async def test_connection(
    provider: str = None,
    model: str = None,
    test_prompt: str = "Hello, this is a connection test."
) -> Dict[str, Any]:
    """Test connection to a specific provider/model."""
    try:
        from chuk_llm.api.core import ask
        
        # Use current config if not specified
        if not provider:
            current_config = get_current_config()
            provider = current_config.get("provider", "openai")
        
        if not model:
            # Get the model from current config or fall back to provider default
            current_config = get_current_config()
            if current_config.get("provider") == provider:
                model = current_config.get("model")
            
            # If still no model, get provider default
            if not model:
                config_manager = get_config()
                try:
                    provider_config = config_manager.get_provider(provider)
                    model = provider_config.default_model
                except ValueError:
                    model = "gpt-4o-mini"  # Fallback
        
        # Use asyncio event loop time for consistent timing with tests
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        
        response = await ask(
            test_prompt,
            provider=provider,
            model=model,
            max_tokens=50  # Keep it short for testing
        )
        
        end_time = loop.time()
        duration = end_time - start_time
        
        # Handle long responses (truncate preview)
        response_preview = response
        if len(response) > 100:
            response_preview = response[:100] + "..."
        
        return {
            "success": True,
            "provider": provider,
            "model": model,
            "duration": duration,
            "response_length": len(response),
            "response_preview": response_preview
        }
        
    except Exception as e:
        loop = asyncio.get_event_loop()
        end_time = loop.time()
        # Use a reasonable default duration if timing fails
        duration = getattr(test_connection, '_start_time', 1.0)
        
        return {
            "success": False,
            "provider": provider or "unknown",
            "model": model or "unknown", 
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
    """Test connections to multiple providers."""
    if providers is None:
        # Default providers for testing
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
    """Print diagnostic information about the current setup."""
    print("🔧 ChukLLM Diagnostics")
    print("=" * 50)
    
    try:
        # Current config
        current_config = get_current_config()
        print("\n📋 Current Configuration:")
        for key, value in current_config.items():
            if key == "api_key" and value:
                # Mask API key
                if len(value) > 8:
                    value = f"{value[:8]}..."
                else:
                    value = "***"
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
            
    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")


async def cleanup():
    """Cleanup resources."""
    global _cached_client
    
    try:
        # Try to cleanup connection pool
        try:
            from chuk_llm.llm.connection_pool import cleanup_llm_resources
            await cleanup_llm_resources()
        except ImportError:
            # Connection pool not available
            pass
        
        # Close cached client if it exists
        if _cached_client is not None:
            if hasattr(_cached_client, 'close'):
                if asyncio.iscoroutinefunction(_cached_client.close):
                    await _cached_client.close()
                else:
                    _cached_client.close()
        
    finally:
        # Always clear the cached client
        _cached_client = None


def cleanup_sync():
    """Synchronous version of cleanup()."""
    asyncio.run(cleanup())