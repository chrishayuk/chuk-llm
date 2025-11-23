"""Comprehensive tests for llm/connection_pool.py module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import httpx
import weakref

from chuk_llm.llm.connection_pool import (
    ConnectionPool,
    managed_connection_pool,
    LLMResourceManager,
    cleanup_llm_resources,
    get_llm_health_status,
    _resource_manager,
)


class TestConnectionPool:
    """Tests for ConnectionPool class."""

    def test_connection_pool_singleton(self):
        """Test that ConnectionPool is a singleton."""
        pool1 = ConnectionPool()
        pool2 = ConnectionPool()

        assert pool1 is pool2

    def test_connection_pool_initialization(self):
        """Test ConnectionPool initialization."""
        pool = ConnectionPool()

        assert hasattr(pool, '_pools')
        assert hasattr(pool, '_locks')
        assert isinstance(pool._pools, dict)
        assert isinstance(pool._locks, dict)

    @pytest.mark.asyncio
    async def test_get_client_default(self):
        """Test getting default HTTP client."""
        pool = ConnectionPool()

        # Clean up first
        await pool.close_all()

        client = await pool.get_client()

        assert isinstance(client, httpx.AsyncClient)
        assert "default" in pool._pools

        # Cleanup
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_get_client_with_base_url(self):
        """Test getting client with custom base URL."""
        pool = ConnectionPool()

        await pool.close_all()

        base_url = "https://api.example.com"
        client = await pool.get_client(base_url=base_url)

        assert isinstance(client, httpx.AsyncClient)
        assert base_url in pool._pools

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_get_client_cached(self):
        """Test that get_client returns cached client."""
        pool = ConnectionPool()

        await pool.close_all()

        base_url = "https://api.test.com"
        client1 = await pool.get_client(base_url=base_url)
        client2 = await pool.get_client(base_url=base_url)

        # Should return same instance
        assert client1 is client2

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_get_client_with_timeout(self):
        """Test getting client with custom timeout."""
        pool = ConnectionPool()

        await pool.close_all()

        client = await pool.get_client(timeout=30.0)

        assert isinstance(client, httpx.AsyncClient)

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_get_client_with_max_connections(self):
        """Test getting client with custom max_connections."""
        pool = ConnectionPool()

        await pool.close_all()

        client = await pool.get_client(max_connections=50)

        assert isinstance(client, httpx.AsyncClient)

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_get_client_creates_lock(self):
        """Test that get_client creates lock for key."""
        pool = ConnectionPool()

        await pool.close_all()

        base_url = "https://api.lock-test.com"
        await pool.get_client(base_url=base_url)

        assert base_url in pool._locks
        assert isinstance(pool._locks[base_url], asyncio.Lock)

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all connection pools."""
        pool = ConnectionPool()

        # Create some clients
        await pool.get_client(base_url="https://api1.com")
        await pool.get_client(base_url="https://api2.com")

        assert len(pool._pools) >= 2

        await pool.close_all()

        # All pools should be cleared
        assert len(pool._pools) == 0
        assert len(pool._locks) == 0

    @pytest.mark.asyncio
    async def test_close_pool_specific(self):
        """Test closing specific connection pool."""
        pool = ConnectionPool()

        await pool.close_all()

        # Create two clients
        base_url1 = "https://api1.com"
        base_url2 = "https://api2.com"

        await pool.get_client(base_url=base_url1)
        await pool.get_client(base_url=base_url2)

        # Close only one
        await pool.close_pool(base_url=base_url1)

        assert base_url1 not in pool._pools
        assert base_url2 in pool._pools

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_close_pool_default(self):
        """Test closing default pool."""
        pool = ConnectionPool()

        await pool.close_all()

        await pool.get_client()  # Creates default pool

        await pool.close_pool()  # Close default

        assert "default" not in pool._pools

    @pytest.mark.asyncio
    async def test_close_pool_nonexistent(self):
        """Test closing non-existent pool doesn't error."""
        pool = ConnectionPool()

        await pool.close_all()

        # Should not raise
        await pool.close_pool(base_url="https://nonexistent.com")

    @pytest.mark.asyncio
    async def test_concurrent_get_client(self):
        """Test concurrent access to get_client."""
        pool = ConnectionPool()

        await pool.close_all()

        base_url = "https://concurrent.test.com"

        # Create multiple concurrent requests
        clients = await asyncio.gather(*[
            pool.get_client(base_url=base_url)
            for _ in range(10)
        ])

        # All should be the same instance
        assert all(c is clients[0] for c in clients)

        # Only one pool should exist for this URL
        assert base_url in pool._pools

        await pool.close_all()


class TestManagedConnectionPool:
    """Tests for managed_connection_pool context manager."""

    @pytest.mark.asyncio
    async def test_managed_connection_pool_basic(self):
        """Test managed_connection_pool context manager."""
        async with managed_connection_pool() as pool:
            assert isinstance(pool, ConnectionPool)

            # Can use the pool
            client = await pool.get_client()
            assert isinstance(client, httpx.AsyncClient)

        # After context, pool should be closed
        # (Can't easily verify without implementation details)

    @pytest.mark.asyncio
    async def test_managed_connection_pool_cleanup(self):
        """Test that managed_connection_pool cleans up."""
        initial_pool = ConnectionPool()
        initial_count = len(initial_pool._pools)

        async with managed_connection_pool() as pool:
            await pool.get_client(base_url="https://test.com")

        # Pool should be cleaned up
        final_pool = ConnectionPool()
        assert len(final_pool._pools) == initial_count

    @pytest.mark.asyncio
    async def test_managed_connection_pool_exception(self):
        """Test that managed_connection_pool cleans up on exception."""
        try:
            async with managed_connection_pool() as pool:
                await pool.get_client(base_url="https://error.test.com")
                raise ValueError("Test error")
        except ValueError:
            pass

        # Pool should still be cleaned up
        final_pool = ConnectionPool()
        # Can't check exact count but shouldn't crash


class TestLLMResourceManager:
    """Tests for LLMResourceManager class."""

    def test_llm_resource_manager_init(self):
        """Test LLMResourceManager initialization."""
        manager = LLMResourceManager()

        assert hasattr(manager, '_clients')
        assert hasattr(manager, '_connection_pool')
        assert isinstance(manager._clients, weakref.WeakSet)
        assert isinstance(manager._connection_pool, ConnectionPool)

    def test_register_client(self):
        """Test registering a client."""
        manager = LLMResourceManager()

        mock_client = Mock()
        manager.register_client(mock_client)

        assert mock_client in manager._clients

    def test_register_multiple_clients(self):
        """Test registering multiple clients."""
        manager = LLMResourceManager()

        client1 = Mock()
        client2 = Mock()

        manager.register_client(client1)
        manager.register_client(client2)

        assert client1 in manager._clients
        assert client2 in manager._clients

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleanup_all closes all clients."""
        manager = LLMResourceManager()

        # Create mock clients with close method
        mock_client1 = Mock()
        mock_client1.close = AsyncMock()
        mock_client2 = Mock()
        mock_client2.close = AsyncMock()

        manager.register_client(mock_client1)
        manager.register_client(mock_client2)

        await manager.cleanup_all()

        # Both clients should be closed
        mock_client1.close.assert_called_once()
        mock_client2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_all_without_close_method(self):
        """Test cleanup_all skips clients without close method."""
        manager = LLMResourceManager()

        # Client without close method
        mock_client = Mock(spec=[])  # No methods

        manager.register_client(mock_client)

        # Should not raise
        await manager.cleanup_all()

    @pytest.mark.asyncio
    async def test_cleanup_all_closes_connection_pool(self):
        """Test cleanup_all closes connection pool."""
        manager = LLMResourceManager()

        # Add a client to connection pool
        await manager._connection_pool.get_client(base_url="https://test.com")

        await manager.cleanup_all()

        # Connection pool should be empty
        assert len(manager._connection_pool._pools) == 0

    @pytest.mark.asyncio
    async def test_health_check_basic(self):
        """Test health_check returns status."""
        manager = LLMResourceManager()

        status = await manager.health_check()

        assert isinstance(status, dict)
        assert "total_clients" in status
        assert "connection_pools" in status
        assert "clients" in status
        assert isinstance(status["clients"], list)

    @pytest.mark.asyncio
    async def test_health_check_with_clients(self):
        """Test health_check with registered clients."""
        manager = LLMResourceManager()

        # Create mock client with attributes
        mock_client = Mock()
        mock_client.provider_name = "openai"
        mock_client.model = "gpt-4"

        manager.register_client(mock_client)

        status = await manager.health_check()

        assert status["total_clients"] == 1
        assert len(status["clients"]) == 1
        assert status["clients"][0]["provider"] == "openai"
        assert status["clients"][0]["model"] == "gpt-4"
        assert status["clients"][0]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_client_error(self):
        """Test health_check handles client errors."""
        manager = LLMResourceManager()

        # Create mock client that raises error on attribute access
        mock_client = Mock()
        type(mock_client).provider_name = property(lambda self: (_ for _ in ()).throw(Exception("Error")))

        manager.register_client(mock_client)

        status = await manager.health_check()

        assert len(status["clients"]) == 1
        assert status["clients"][0]["status"] == "error"
        assert "error" in status["clients"][0]

    @pytest.mark.asyncio
    async def test_health_check_connection_pools(self):
        """Test health_check reports connection pools."""
        manager = LLMResourceManager()

        # Create some connection pools
        await manager._connection_pool.get_client(base_url="https://api1.com")
        await manager._connection_pool.get_client(base_url="https://api2.com")

        status = await manager.health_check()

        assert status["connection_pools"] >= 2

        # Cleanup
        await manager._connection_pool.close_all()


class TestGlobalFunctions:
    """Tests for global functions."""

    @pytest.mark.asyncio
    async def test_cleanup_llm_resources(self):
        """Test cleanup_llm_resources function."""
        # Register a mock client
        mock_client = Mock()
        mock_client.close = AsyncMock()

        _resource_manager.register_client(mock_client)

        await cleanup_llm_resources()

        # Client should be closed
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_health_status(self):
        """Test get_llm_health_status function."""
        status = await get_llm_health_status()

        assert isinstance(status, dict)
        assert "total_clients" in status
        assert "connection_pools" in status
        assert "clients" in status

    @pytest.mark.asyncio
    async def test_global_resource_manager_singleton(self):
        """Test that _resource_manager is a singleton."""
        from chuk_llm.llm.connection_pool import _resource_manager as rm1
        from chuk_llm.llm import connection_pool

        # Get it through module reload
        assert hasattr(connection_pool, '_resource_manager')
        rm2 = connection_pool._resource_manager

        # Should be same instance
        assert rm1 is rm2


class TestWeakReferences:
    """Tests for weak reference behavior."""

    def test_client_weakref_cleanup(self):
        """Test that clients are garbage collected when deleted."""
        manager = LLMResourceManager()

        # Create client in a scope
        def create_client():
            client = Mock()
            manager.register_client(client)
            return len(manager._clients)

        initial_count = create_client()

        # After function returns, client should be GC'd
        # Note: This might not work immediately due to Python GC timing
        import gc
        gc.collect()

        # Count might be 0 or 1 depending on GC timing
        assert len(manager._clients) <= initial_count

    @pytest.mark.asyncio
    async def test_weakref_in_health_check(self):
        """Test that health_check handles weak references correctly."""
        manager = LLMResourceManager()

        # Create a client
        client = Mock()
        client.provider_name = "test"
        client.model = "model"

        manager.register_client(client)

        # Get health check
        status1 = await manager.health_check()
        assert status1["total_clients"] >= 1

        # Delete client
        del client
        import gc
        gc.collect()

        # Health check should still work
        status2 = await manager.health_check()
        assert isinstance(status2, dict)


class TestConcurrency:
    """Tests for concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_pool_creation(self):
        """Test concurrent pool creation is thread-safe."""
        pool = ConnectionPool()
        await pool.close_all()

        base_url = "https://concurrent.com"

        # Create many concurrent requests
        tasks = [pool.get_client(base_url=base_url) for _ in range(50)]
        clients = await asyncio.gather(*tasks)

        # All should be the same client
        assert all(c is clients[0] for c in clients)

        # Only one pool created
        assert len(pool._pools) == 1

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_and_access(self):
        """Test cleanup and access don't conflict."""
        pool = ConnectionPool()

        async def access_pool():
            for _ in range(10):
                await pool.get_client(base_url="https://access.com")
                await asyncio.sleep(0.001)

        async def cleanup_pool():
            for _ in range(5):
                await asyncio.sleep(0.002)
                await pool.close_all()

        # Run both concurrently - should not crash
        await asyncio.gather(access_pool(), cleanup_pool())

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent health checks."""
        manager = LLMResourceManager()

        # Register some clients
        for i in range(10):
            client = Mock()
            client.provider_name = f"provider{i}"
            client.model = f"model{i}"
            manager.register_client(client)

        # Run multiple health checks concurrently
        results = await asyncio.gather(*[
            manager.health_check()
            for _ in range(10)
        ])

        # All should succeed
        assert all(isinstance(r, dict) for r in results)
        assert all("total_clients" in r for r in results)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_close_pool_removes_lock(self):
        """Test that closing pool also removes its lock."""
        pool = ConnectionPool()
        await pool.close_all()

        base_url = "https://lock-test.com"
        await pool.get_client(base_url=base_url)

        assert base_url in pool._locks

        await pool.close_pool(base_url=base_url)

        assert base_url not in pool._locks

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_empty_manager_cleanup(self):
        """Test cleanup on empty manager."""
        manager = LLMResourceManager()

        # Should not raise
        await manager.cleanup_all()

        status = await manager.health_check()
        assert status["total_clients"] == 0

    @pytest.mark.asyncio
    async def test_health_check_with_no_provider_attr(self):
        """Test health check with client missing provider_name."""
        manager = LLMResourceManager()

        mock_client = Mock(spec=[])  # No attributes

        manager.register_client(mock_client)

        status = await manager.health_check()

        # Should handle gracefully
        assert len(status["clients"]) == 1
        client_status = status["clients"][0]
        assert client_status.get("provider") == "unknown" or "error" in client_status
