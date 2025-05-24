# chuk_llm/llm/connection_pool.py
import asyncio
import weakref
from typing import Dict, Any, Optional
import httpx
from contextlib import asynccontextmanager

class ConnectionPool:
    """Shared connection pool for all providers"""
    _instance = None
    _pools: Dict[str, httpx.AsyncClient] = {}
    _locks: Dict[str, asyncio.Lock] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._pools = {}
            self._locks = {}
    
    async def get_client(
        self, 
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_connections: int = 100,
        **kwargs
    ) -> httpx.AsyncClient:
        """Get or create HTTP client for a base URL"""
        key = base_url or "default"
        
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            if key not in self._pools:
                self._pools[key] = httpx.AsyncClient(
                    base_url=base_url,
                    timeout=httpx.Timeout(timeout),
                    limits=httpx.Limits(max_connections=max_connections),
                    **kwargs
                )
        
        return self._pools[key]
    
    async def close_all(self):
        """Close all connection pools"""
        for client in self._pools.values():
            await client.aclose()
        self._pools.clear()
        self._locks.clear()
    
    async def close_pool(self, base_url: Optional[str] = None):
        """Close specific connection pool"""
        key = base_url or "default"
        if key in self._pools:
            await self._pools[key].aclose()
            del self._pools[key]
            if key in self._locks:
                del self._locks[key]

# Context manager for automatic cleanup
@asynccontextmanager
async def managed_connection_pool():
    """Context manager for connection pool lifecycle"""
    pool = ConnectionPool()
    try:
        yield pool
    finally:
        await pool.close_all()

# Resource manager for LLM clients
class LLMResourceManager:
    """Manages lifecycle of LLM clients and connections"""
    
    def __init__(self):
        self._clients: weakref.WeakSet = weakref.WeakSet()
        self._connection_pool = ConnectionPool()
    
    def register_client(self, client):
        """Register a client for lifecycle management"""
        self._clients.add(client)
    
    async def cleanup_all(self):
        """Cleanup all registered clients"""
        # Close all clients
        for client in list(self._clients):
            if hasattr(client, 'close'):
                await client.close()
        
        # Close connection pool
        await self._connection_pool.close_all()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all registered clients"""
        health_status = {
            "total_clients": len(self._clients),
            "connection_pools": len(self._connection_pool._pools),
            "clients": []
        }
        
        for client in self._clients:
            try:
                # Basic health check - could be expanded
                client_status = {
                    "type": type(client).__name__,
                    "provider": getattr(client, 'provider_name', 'unknown'),
                    "model": getattr(client, 'model', 'unknown'),
                    "status": "healthy"
                }
            except Exception as e:
                client_status = {
                    "type": type(client).__name__,
                    "status": "error",
                    "error": str(e)
                }
            
            health_status["clients"].append(client_status)
        
        return health_status

# Global resource manager instance
_resource_manager = LLMResourceManager()

async def cleanup_llm_resources():
    """Global cleanup function"""
    await _resource_manager.cleanup_all()

async def get_llm_health_status():
    """Get health status of all LLM resources"""
    return await _resource_manager.health_check()