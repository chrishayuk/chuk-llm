"""Comprehensive tests for client_registry module"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from chuk_llm.client_registry import (
    get_cached_client,
    is_client_cached,
    get_cached_client_count,
    get_cache_stats,
    cleanup_registry,
    cleanup_registry_sync,
    clear_cache,
    print_registry_stats,
    _make_cache_key,
    _client_cache,
    _cache_stats,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before and after each test"""
    # Clear cache before test
    clear_cache(reset_stats=True)
    yield
    # Clear cache after test
    clear_cache(reset_stats=True)


class TestCacheKeyGeneration:
    """Test cache key generation"""

    def test_make_cache_key_basic(self):
        """Test basic cache key generation"""
        key = _make_cache_key("openai", "gpt-4o")
        assert key.startswith("openai:gpt-4o:")
        assert len(key.split(":")) == 3

    def test_make_cache_key_with_kwargs(self):
        """Test cache key with additional kwargs"""
        key1 = _make_cache_key("openai", "gpt-4o", api_key="sk-123", base_url="https://api.openai.com")
        key2 = _make_cache_key("openai", "gpt-4o", api_key="sk-123", base_url="https://api.openai.com")
        assert key1 == key2

    def test_make_cache_key_different_kwargs(self):
        """Test that different kwargs produce different keys"""
        key1 = _make_cache_key("openai", "gpt-4o", api_key="sk-123")
        key2 = _make_cache_key("openai", "gpt-4o", api_key="sk-456")
        assert key1 != key2

    def test_make_cache_key_kwargs_order_independent(self):
        """Test that kwargs order doesn't matter"""
        key1 = _make_cache_key("openai", "gpt-4o", api_key="sk-123", base_url="https://api.openai.com")
        key2 = _make_cache_key("openai", "gpt-4o", base_url="https://api.openai.com", api_key="sk-123")
        assert key1 == key2

    def test_make_cache_key_privacy(self):
        """Test that API keys are hashed for privacy"""
        key = _make_cache_key("openai", "gpt-4o", api_key="sk-secret-key-12345")
        # API key should not appear in the key
        assert "sk-secret-key-12345" not in key

    def test_make_cache_key_with_none_model(self):
        """Test cache key generation with None model"""
        key = _make_cache_key("openai", None)
        assert key.startswith("openai:None:")


class TestGetCachedClient:
    """Test get_cached_client function"""

    def test_get_cached_client_creates_new(self):
        """Test that first call creates a new client"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            client = get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            assert client == mock_client
            mock_create.assert_called_once_with("openai", model="gpt-4o", api_key="sk-123")

    def test_get_cached_client_returns_cached(self):
        """Test that second call returns cached client"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            # First call
            client1 = get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            # Second call with same params
            client2 = get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            assert client1 is client2
            # Should only create once
            assert mock_create.call_count == 1

    def test_get_cached_client_different_params(self):
        """Test that different params create different clients"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client1 = Mock()
            mock_client2 = Mock()
            mock_create.side_effect = [mock_client1, mock_client2]

            # Different models
            client1 = get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            client2 = get_cached_client("openai", model="gpt-4o-mini", api_key="sk-123")

            assert client1 is not client2
            assert mock_create.call_count == 2

    def test_get_cached_client_without_model(self):
        """Test getting client without specifying model"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            client = get_cached_client("openai", api_key="sk-123")

            assert client == mock_client
            mock_create.assert_called_once_with("openai", model=None, api_key="sk-123")

    def test_get_cached_client_updates_stats(self):
        """Test that cache stats are updated"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            # First call - miss
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            stats = get_cache_stats()
            assert stats["misses"] == 1
            assert stats["hits"] == 0
            assert stats["total_clients"] == 1

            # Second call - hit
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            stats = get_cache_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["total_clients"] == 1


class TestIsClientCached:
    """Test is_client_cached function"""

    def test_is_client_cached_false_initially(self):
        """Test that client is not cached initially"""
        assert not is_client_cached("openai", model="gpt-4o", api_key="sk-123")

    def test_is_client_cached_true_after_creation(self):
        """Test that client is cached after creation"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            assert is_client_cached("openai", model="gpt-4o", api_key="sk-123")

    def test_is_client_cached_different_params(self):
        """Test that different params are not cached"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            assert not is_client_cached("openai", model="gpt-4o-mini", api_key="sk-123")


class TestGetCachedClientCount:
    """Test get_cached_client_count function"""

    def test_get_cached_client_count_zero(self):
        """Test count is zero initially"""
        assert get_cached_client_count() == 0

    def test_get_cached_client_count_increments(self):
        """Test count increments as clients are created"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            assert get_cached_client_count() == 1

            get_cached_client("anthropic", model="claude-3", api_key="sk-456")
            assert get_cached_client_count() == 2

            # Same params - should not increment
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            assert get_cached_client_count() == 2


class TestGetCacheStats:
    """Test get_cache_stats function"""

    def test_get_cache_stats_initial(self):
        """Test initial stats"""
        stats = get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_clients"] == 0

    def test_get_cache_stats_after_operations(self):
        """Test stats after cache operations"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            # First call - miss
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            # Second call - hit
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            # Third call - hit
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            stats = get_cache_stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["total_clients"] == 1

    def test_get_cache_stats_returns_copy(self):
        """Test that stats returns a copy, not the original dict"""
        stats1 = get_cache_stats()
        stats2 = get_cache_stats()
        assert stats1 is not stats2
        assert stats1 == stats2


class TestCleanupRegistry:
    """Test cleanup_registry function"""

    @pytest.mark.asyncio
    async def test_cleanup_registry_clears_cache(self):
        """Test that cleanup clears the cache"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()

            async def mock_close():
                pass

            mock_client.close = mock_close
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            assert get_cached_client_count() == 1

            await cleanup_registry()

            assert get_cached_client_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_registry_closes_clients(self):
        """Test that cleanup closes clients"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            close_called = False

            async def mock_close():
                nonlocal close_called
                close_called = True

            mock_client.close = mock_close
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            await cleanup_registry()

            # close should have been called
            assert close_called

    @pytest.mark.asyncio
    async def test_cleanup_registry_handles_close_errors(self):
        """Test that cleanup handles client close errors gracefully"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            # Simulate close error
            async def raise_error():
                raise Exception("Close error")
            mock_client.close = raise_error
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            # Should not raise
            await cleanup_registry()

            assert get_cached_client_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_registry_handles_event_loop_closed(self):
        """Test that cleanup handles 'Event loop is closed' errors"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            # Simulate event loop closed error
            async def raise_loop_error():
                raise RuntimeError("Event loop is closed")
            mock_client.close = raise_loop_error
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            # Should not raise or log warning
            await cleanup_registry()

    @pytest.mark.asyncio
    async def test_cleanup_registry_without_close_method(self):
        """Test cleanup with clients that don't have close method"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock(spec=[])  # No close method
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            # Should not raise
            await cleanup_registry()

    @pytest.mark.asyncio
    async def test_cleanup_registry_resets_total_clients(self):
        """Test that cleanup resets total_clients stat"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()

            async def mock_close():
                pass

            mock_client.close = mock_close
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            stats = get_cache_stats()
            assert stats["total_clients"] == 1

            await cleanup_registry()

            stats = get_cache_stats()
            assert stats["total_clients"] == 0


class TestCleanupRegistrySync:
    """Test cleanup_registry_sync function"""

    def test_cleanup_registry_sync_basic(self):
        """Test synchronous cleanup"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()

            async def mock_close():
                pass

            mock_client.close = mock_close
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            assert get_cached_client_count() == 1

            cleanup_registry_sync()

            assert get_cached_client_count() == 0

    def test_cleanup_registry_sync_with_running_loop(self):
        """Test sync cleanup when event loop is running"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            # Mock asyncio.run to raise RuntimeError (loop already running)
            with patch('asyncio.run', side_effect=RuntimeError("Event loop already running")):
                # Mock get_event_loop and create_task
                mock_loop = Mock()
                mock_loop.is_running.return_value = True
                with patch('asyncio.get_event_loop', return_value=mock_loop):
                    cleanup_registry_sync()

                    # Should have tried to create a task
                    mock_loop.create_task.assert_called_once()


class TestClearCache:
    """Test clear_cache function"""

    def test_clear_cache_basic(self):
        """Test basic cache clearing"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("anthropic", model="claude-3", api_key="sk-456")

            count = clear_cache()

            assert count == 2
            assert get_cached_client_count() == 0

    def test_clear_cache_resets_stats(self):
        """Test that clear_cache resets stats by default"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")  # Hit

            clear_cache(reset_stats=True)

            stats = get_cache_stats()
            assert stats["hits"] == 0
            assert stats["misses"] == 0
            assert stats["total_clients"] == 0

    def test_clear_cache_preserve_stats(self):
        """Test that clear_cache can preserve stats"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")  # Hit

            clear_cache(reset_stats=False)

            stats = get_cache_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            # Note: total_clients is NOT reset when reset_stats=False
            # It represents the cumulative total created
            assert stats["total_clients"] == 1

    def test_clear_cache_returns_count(self):
        """Test that clear_cache returns the number of clients removed"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("anthropic", model="claude-3", api_key="sk-456")
            get_cached_client("groq", model="mixtral", api_key="sk-789")

            count = clear_cache()

            assert count == 3


class TestPrintRegistryStats:
    """Test print_registry_stats function"""

    def test_print_registry_stats_basic(self, capsys):
        """Test printing basic stats"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")  # Hit

            print_registry_stats()

            captured = capsys.readouterr()
            assert "Client Registry Statistics" in captured.out
            assert "Cached clients:" in captured.out
            assert "Cache hits:" in captured.out
            assert "Cache misses:" in captured.out
            assert "Hit rate:" in captured.out
            assert "Time saved:" in captured.out

    def test_print_registry_stats_with_zero_requests(self, capsys):
        """Test printing stats when no requests made"""
        print_registry_stats()

        captured = capsys.readouterr()
        assert "0.0%" in captured.out  # Hit rate should be 0%
        assert "0ms" in captured.out  # Time saved should be 0

    def test_print_registry_stats_calculates_hit_rate(self, capsys):
        """Test that hit rate is calculated correctly"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            # 1 miss, 3 hits = 75% hit rate
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            print_registry_stats()

            captured = capsys.readouterr()
            assert "75.0%" in captured.out


class TestThreadSafety:
    """Test thread-safety of the registry"""

    def test_concurrent_access(self):
        """Test that concurrent access is thread-safe"""
        import threading

        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            def get_client():
                for _ in range(10):
                    get_cached_client("openai", model="gpt-4o", api_key="sk-123")

            threads = [threading.Thread(target=get_client) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should only create one client despite 50 concurrent calls
            assert mock_create.call_count == 1
            stats = get_cache_stats()
            assert stats["total_clients"] == 1
            assert stats["misses"] == 1
            assert stats["hits"] == 49


class TestIntegration:
    """Integration tests for the registry"""

    def test_full_lifecycle(self):
        """Test full lifecycle of registry usage"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_client1 = Mock()
            mock_client2 = Mock()
            mock_create.side_effect = [mock_client1, mock_client2]

            # Create multiple clients
            client1 = get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            client2 = get_cached_client("openai", model="gpt-4o", api_key="sk-123")
            client3 = get_cached_client("anthropic", model="claude-3", api_key="sk-456")

            # Verify caching
            assert client1 is client2
            assert client1 is not client3

            # Check stats
            stats = get_cache_stats()
            assert stats["total_clients"] == 2
            assert stats["hits"] == 1
            assert stats["misses"] == 2

            # Verify cached
            assert is_client_cached("openai", model="gpt-4o", api_key="sk-123")
            assert is_client_cached("anthropic", model="claude-3", api_key="sk-456")

            # Clear cache
            count = clear_cache()
            assert count == 2
            assert get_cached_client_count() == 0

    def test_multiple_providers_and_models(self):
        """Test registry with multiple providers and models"""
        with patch('chuk_llm.llm.client._create_client_internal') as mock_create:
            mock_create.return_value = Mock()

            # Different providers
            get_cached_client("openai", model="gpt-4o")
            get_cached_client("anthropic", model="claude-3")
            get_cached_client("groq", model="mixtral")

            # Same provider, different models
            get_cached_client("openai", model="gpt-4o-mini")
            get_cached_client("openai", model="gpt-3.5-turbo")

            # Same provider and model (should hit cache)
            get_cached_client("openai", model="gpt-4o")

            assert get_cached_client_count() == 5
            stats = get_cache_stats()
            assert stats["total_clients"] == 5
            assert stats["misses"] == 5
            assert stats["hits"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
