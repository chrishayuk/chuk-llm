"""
Comprehensive tests for the registry cache system.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from chuk_llm.core.enums import Provider
from chuk_llm.registry.cache import RegistryCache
from chuk_llm.registry.models import ModelCapabilities, ModelSpec, QualityTier


class TestRegistryCache:
    """Test the registry cache system."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a RegistryCache instance with temporary directory."""
        return RegistryCache(cache_dir=temp_cache_dir, ttl_hours=24)

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization creates directory and sets up cache."""
        cache = RegistryCache(cache_dir=temp_cache_dir)

        # Directory should be created
        assert cache.cache_dir.exists()
        # Cache file path should be set
        assert cache.cache_file == temp_cache_dir / "registry_cache.json"

    def test_set_and_get_capabilities(self, cache):
        """Test setting and getting capabilities."""
        spec = ModelSpec(
            provider=Provider.OPENAI.value,
            name="gpt-4o",
            family="gpt-4o",
        )

        caps = ModelCapabilities(
            max_context=128_000,
            supports_tools=True,
            supports_vision=True,
            quality_tier=QualityTier.BEST,
        )

        # Set capabilities
        cache.set_capabilities(spec, caps)

        # Get capabilities
        retrieved = cache.get_model(spec.provider, spec.name)

        assert retrieved is not None
        assert retrieved.spec.name == spec.name
        assert retrieved.capabilities.max_context == 128_000
        assert retrieved.capabilities.supports_tools is True

    def test_get_nonexistent_model(self, cache):
        """Test getting a model that doesn't exist."""
        result = cache.get_model(Provider.OPENAI.value, "nonexistent-model")

        assert result is None

    def test_cache_expiration(self, temp_cache_dir):
        """Test that expired cache entries are not returned."""
        # Create cache with 0 hour TTL (immediate expiration)
        cache = RegistryCache(cache_dir=temp_cache_dir, ttl_hours=0)

        spec = ModelSpec(
            provider=Provider.OPENAI.value,
            name="gpt-4o",
        )

        caps = ModelCapabilities(max_context=128_000)

        # Set capabilities
        cache.set_capabilities(spec, caps)

        # Wait a moment to ensure expiration
        time.sleep(0.1)

        # Try to get - should be expired
        result = cache.get_model(spec.provider, spec.name)

        assert result is None

    def test_cache_persistence(self, temp_cache_dir):
        """Test that cache persists across instances."""
        spec = ModelSpec(
            provider=Provider.OPENAI.value,
            name="gpt-4o",
        )

        caps = ModelCapabilities(
            max_context=128_000,
            supports_tools=True,
        )

        # Create first cache instance and set data
        cache1 = RegistryCache(cache_dir=temp_cache_dir)
        cache1.set_capabilities(spec, caps)

        # Create second cache instance and retrieve data
        cache2 = RegistryCache(cache_dir=temp_cache_dir)
        retrieved = cache2.get_model(spec.provider, spec.name)

        assert retrieved is not None
        assert retrieved.capabilities.max_context == 128_000

    def test_clear_cache(self, cache):
        """Test clearing the cache."""
        spec = ModelSpec(
            provider=Provider.OPENAI.value,
            name="gpt-4o",
        )

        caps = ModelCapabilities(max_context=128_000)

        # Add some data
        cache.set_capabilities(spec, caps)

        # Clear cache
        cache.clear()

        # Verify data is gone
        result = cache.get_model(spec.provider, spec.name)
        assert result is None

    def test_get_stats(self, cache):
        """Test cache statistics."""
        # Add some models
        for i in range(3):
            spec = ModelSpec(
                provider=Provider.OPENAI.value,
                name=f"model-{i}",
            )
            caps = ModelCapabilities(max_context=128_000)
            cache.set_capabilities(spec, caps)

        stats = cache.get_stats()

        assert stats["total_entries"] == 3
        assert stats["valid_entries"] == 3
        assert stats["cache_file"] == str(cache.cache_file)
        assert "cache_size_bytes" in stats

    def test_expired_entries_in_stats(self, temp_cache_dir):
        """Test that stats correctly counts expired entries."""
        cache = RegistryCache(cache_dir=temp_cache_dir, ttl_hours=24)

        # Add a valid entry
        spec1 = ModelSpec(provider=Provider.OPENAI.value, name="model-1")
        caps1 = ModelCapabilities(max_context=128_000)
        cache.set_capabilities(spec1, caps1)

        # Manually add an expired entry to the cache
        expired_key = cache._cache_key(Provider.OPENAI.value, "expired-model")
        cache._cache[expired_key] = {
            "provider": Provider.OPENAI.value,
            "model": "expired-model",
            "family": None,
            "capabilities": {"max_context": 100000},
            "cached_at": (datetime.now() - timedelta(hours=48)).isoformat(),
        }
        cache._save_cache()

        stats = cache.get_stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 1  # Only one is valid

    def test_multiple_providers(self, cache):
        """Test cache with multiple providers."""
        providers = [
            Provider.OPENAI.value,
            Provider.ANTHROPIC.value,
            Provider.GEMINI.value,
        ]

        for provider in providers:
            spec = ModelSpec(provider=provider, name=f"{provider}-model")
            caps = ModelCapabilities(max_context=128_000)
            cache.set_capabilities(spec, caps)

        # Verify all providers are cached
        for provider in providers:
            result = cache.get_model(provider, f"{provider}-model")
            assert result is not None

    def test_update_existing_entry(self, cache):
        """Test updating an existing cache entry."""
        spec = ModelSpec(
            provider=Provider.OPENAI.value,
            name="gpt-4o",
        )

        # Set initial capabilities
        caps1 = ModelCapabilities(max_context=100_000)
        cache.set_capabilities(spec, caps1)

        # Update capabilities
        caps2 = ModelCapabilities(
            max_context=128_000,
            supports_tools=True,
        )
        cache.set_capabilities(spec, caps2)

        # Verify updated values
        retrieved = cache.get_model(spec.provider, spec.name)
        assert retrieved.capabilities.max_context == 128_000
        assert retrieved.capabilities.supports_tools is True

    def test_cache_file_corruption_recovery(self, temp_cache_dir):
        """Test that cache recovers from corrupted file."""
        # Create cache
        cache = RegistryCache(cache_dir=temp_cache_dir)

        # Corrupt the file
        with open(cache.cache_file, "w") as f:
            f.write("INVALID JSON{{{")

        # Create new cache instance - should recover
        cache2 = RegistryCache(cache_dir=temp_cache_dir)

        # Should work normally
        spec = ModelSpec(provider=Provider.OPENAI.value, name="test")
        caps = ModelCapabilities(max_context=100_000)
        cache2.set_capabilities(spec, caps)

        result = cache2.get_model(spec.provider, spec.name)
        assert result is not None

    def test_concurrent_access(self, temp_cache_dir):
        """Test concurrent cache access."""
        spec1 = ModelSpec(provider=Provider.OPENAI.value, name="model-1")
        spec2 = ModelSpec(provider=Provider.ANTHROPIC.value, name="model-2")

        caps1 = ModelCapabilities(max_context=100_000)
        caps2 = ModelCapabilities(max_context=200_000)

        # Create first instance and write
        cache1 = RegistryCache(cache_dir=temp_cache_dir)
        cache1.set_capabilities(spec1, caps1)

        # Create second instance and write
        cache2 = RegistryCache(cache_dir=temp_cache_dir)
        cache2.set_capabilities(spec2, caps2)

        # Create third instance to verify both entries exist
        cache3 = RegistryCache(cache_dir=temp_cache_dir)
        result1 = cache3.get_model(spec1.provider, spec1.name)
        result2 = cache3.get_model(spec2.provider, spec2.name)

        assert result1 is not None
        assert result2 is not None
        assert result1.capabilities.max_context == 100_000
        assert result2.capabilities.max_context == 200_000

    def test_cache_size_bytes(self, cache):
        """Test cache size calculation."""
        # Add some data
        for i in range(5):
            spec = ModelSpec(provider=Provider.OPENAI.value, name=f"model-{i}")
            caps = ModelCapabilities(max_context=128_000)
            cache.set_capabilities(spec, caps)

        stats = cache.get_stats()

        # Cache should have non-zero size
        assert stats["cache_size_bytes"] > 0

    def test_empty_cache_stats(self, cache):
        """Test statistics for empty cache."""
        stats = cache.get_stats()

        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0
        assert "cache_file" in stats

    def test_cache_known_params_set(self, cache):
        """Test caching of known_params as a set."""
        spec = ModelSpec(provider=Provider.OPENAI.value, name="gpt-4o")

        caps = ModelCapabilities(
            max_context=128_000,
            known_params={"temperature", "max_tokens", "top_p"},
        )

        cache.set_capabilities(spec, caps)

        retrieved = cache.get_model(spec.provider, spec.name)

        assert isinstance(retrieved.capabilities.known_params, set)
        assert "temperature" in retrieved.capabilities.known_params
        assert len(retrieved.capabilities.known_params) == 3

    def test_get_capabilities_method(self, cache):
        """Test get_capabilities method separately from get_model."""
        spec = ModelSpec(provider=Provider.OPENAI.value, name="gpt-4o")
        caps = ModelCapabilities(max_context=128_000, supports_tools=True)

        cache.set_capabilities(spec, caps)
        retrieved_caps = cache.get_capabilities(spec)

        assert retrieved_caps is not None
        assert retrieved_caps.max_context == 128_000
        assert retrieved_caps.supports_tools is True

    def test_get_capabilities_nonexistent(self, cache):
        """Test get_capabilities for nonexistent model."""
        spec = ModelSpec(provider=Provider.OPENAI.value, name="nonexistent")
        caps = cache.get_capabilities(spec)
        assert caps is None

    def test_get_capabilities_expired(self, temp_cache_dir):
        """Test get_capabilities for expired entry."""
        cache = RegistryCache(cache_dir=temp_cache_dir, ttl_hours=0)
        spec = ModelSpec(provider=Provider.OPENAI.value, name="gpt-4o")
        caps = ModelCapabilities(max_context=128_000)

        cache.set_capabilities(spec, caps)
        time.sleep(0.1)

        retrieved = cache.get_capabilities(spec)
        assert retrieved is None

    def test_clear_provider(self, cache):
        """Test clearing cache for specific provider."""
        spec1 = ModelSpec(provider=Provider.OPENAI.value, name="model-1")
        spec2 = ModelSpec(provider=Provider.ANTHROPIC.value, name="model-2")
        caps = ModelCapabilities(max_context=128_000)

        cache.set_capabilities(spec1, caps)
        cache.set_capabilities(spec2, caps)

        cache.clear_provider(Provider.OPENAI.value)

        assert cache.get_model(Provider.OPENAI.value, "model-1") is None
        assert cache.get_model(Provider.ANTHROPIC.value, "model-2") is not None

    def test_set_capabilities_with_none_values(self, cache):
        """Test caching capabilities with None values."""
        spec = ModelSpec(provider=Provider.OPENAI.value, name="test-model")
        caps = ModelCapabilities(max_context=None, supports_tools=None)

        cache.set_capabilities(spec, caps)
        retrieved = cache.get_model(spec.provider, spec.name)

        assert retrieved is not None
        assert retrieved.capabilities.max_context is None
