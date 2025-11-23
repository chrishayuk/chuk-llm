"""
Registry caching system.

Caches discovered models and resolved capabilities to avoid repeated API calls.
Cache is stored in ~/.cache/chuk-llm/ for fast lookups.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chuk_llm.registry.models import ModelCapabilities, ModelSpec, ModelWithCapabilities


class RegistryCache:
    """
    Persistent cache for model registry.

    Stores discovered models and their capabilities in a JSON file.
    Automatically expires stale entries.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_hours: int = 24,
    ):
        """
        Initialize registry cache.

        Args:
            cache_dir: Cache directory (defaults to ~/.cache/chuk-llm/)
            ttl_hours: Time-to-live for cache entries in hours
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "chuk-llm"

        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "registry_cache.json"
        self.ttl = timedelta(hours=ttl_hours)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        self._cache: dict[str, Any] = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # Corrupt cache - start fresh
            return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError:
            # Failed to save - not critical
            pass

    def _is_expired(self, timestamp_str: str) -> bool:
        """Check if cache entry is expired."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return datetime.now() - timestamp > self.ttl
        except (ValueError, TypeError):
            # Invalid timestamp - consider expired
            return True

    def _cache_key(self, provider: str, model_name: str) -> str:
        """Generate cache key for a model."""
        return f"{provider}:{model_name}"

    def get_capabilities(self, spec: ModelSpec) -> ModelCapabilities | None:
        """
        Get cached capabilities for a model.

        Args:
            spec: Model specification

        Returns:
            Cached capabilities or None if not found/expired
        """
        key = self._cache_key(spec.provider, spec.name)
        entry = self._cache.get(key)

        if not entry:
            return None

        # Check if expired
        if self._is_expired(entry.get("cached_at", "")):
            # Remove expired entry
            del self._cache[key]
            self._save_cache()
            return None

        # Parse capabilities
        try:
            cap_data = entry.get("capabilities", {})
            return ModelCapabilities(**cap_data)
        except (TypeError, ValueError):
            # Invalid cache entry
            return None

    def set_capabilities(
        self, spec: ModelSpec, capabilities: ModelCapabilities
    ) -> None:
        """
        Cache capabilities for a model.

        Args:
            spec: Model specification
            capabilities: Model capabilities to cache
        """
        key = self._cache_key(spec.provider, spec.name)

        # Store entry (convert sets to lists for JSON serialization)
        cap_dict = capabilities.model_dump(exclude_none=True)
        # Convert known_params set to list
        if "known_params" in cap_dict and isinstance(cap_dict["known_params"], set):
            cap_dict["known_params"] = list(cap_dict["known_params"])

        self._cache[key] = {
            "provider": spec.provider,
            "model": spec.name,
            "family": spec.family,
            "capabilities": cap_dict,
            "cached_at": datetime.now().isoformat(),
        }

        self._save_cache()

    def get_model(self, provider: str, model_name: str) -> ModelWithCapabilities | None:
        """
        Get cached model with capabilities.

        Args:
            provider: Provider name
            model_name: Model name

        Returns:
            Cached model or None if not found/expired
        """
        key = self._cache_key(provider, model_name)
        entry = self._cache.get(key)

        if not entry:
            return None

        # Check if expired
        if self._is_expired(entry.get("cached_at", "")):
            del self._cache[key]
            self._save_cache()
            return None

        # Parse model
        try:
            spec = ModelSpec(
                provider=entry["provider"],
                name=entry["model"],
                family=entry.get("family"),
            )
            cap_data = entry.get("capabilities", {})
            capabilities = ModelCapabilities(**cap_data)

            return ModelWithCapabilities(
                spec=spec,
                capabilities=capabilities,
            )
        except (TypeError, ValueError, KeyError):
            return None

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()

    def clear_provider(self, provider: str) -> None:
        """
        Clear cache entries for a specific provider.

        Args:
            provider: Provider name to clear
        """
        keys_to_remove = [key for key in self._cache if key.startswith(f"{provider}:")]

        for key in keys_to_remove:
            del self._cache[key]

        self._save_cache()

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self._cache)
        expired_entries = sum(
            1
            for entry in self._cache.values()
            if self._is_expired(entry.get("cached_at", ""))
        )
        providers = list({entry["provider"] for entry in self._cache.values()})

        return {
            "total_entries": total_entries,
            "valid_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "providers": providers,
            "cache_file": str(self.cache_file),
            "cache_size_bytes": (
                self.cache_file.stat().st_size if self.cache_file.exists() else 0
            ),
        }
