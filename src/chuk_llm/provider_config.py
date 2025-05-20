# chuk_llm/openai_style_mixin.py
"""
Provider configuration management.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# global defaults
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "__global__": {
        "active_provider": "openai",
        "active_model": "gpt-4o-mini",
    },
    "openai": {
        "client": "chuk_llm.providers.openai_client:OpenAILLMClient",
        "api_key_env": "OPENAI_API_KEY",
        "api_base": None,
        "api_key": None,
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "client": "chuk_llm.providers.groq_client:GroqAILLMClient",
        "api_key_env": "GROQ_API_KEY",
        "api_key": None,
        "api_base": None,
        "default_model": "llama-3.3-70b-versatile",
    },
    "ollama": {
        "client": "chuk_llm.providers.ollama_client:OllamaLLMClient",
        "api_key_env": None,
        "api_base": "http://localhost:11434",
        "api_key": None,
        "default_model": "qwen3",
    },
    "gemini": {
        "client": "chuk_llm.providers.gemini_client:GeminiLLMClient",
        "api_key_env": None,
        "api_key": None,
        "default_model": "gemini-2.0-flash",
    },
    "anthropic": {
        "client": "chuk_llm.providers.anthropic_client:AnthropicLLMClient",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_base": None,
        "api_key": None,
        "default_model": "claude-3-7-sonnet",
    },
}


class ProviderConfig:
    """Provider configuration with sensible defaults."""

    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Initialize provider configuration.
        
        Args:
            config: Optional dictionary of provider configurations to overlay on defaults.
                   If None, only the built-in defaults will be used.
        """
        # Deep copy defaults then overlay user config
        self.providers = json.loads(json.dumps(DEFAULTS))
        
        if config:
            for provider, provider_config in config.items():
                if provider not in self.providers:
                    self.providers[provider] = provider_config
                else:
                    self.providers[provider].update(provider_config)

    def _ensure_section(self, name: str) -> None:
        """Ensure a provider section exists."""
        if name not in self.providers:
            self.providers[name] = {}

    def _merge_env_key(self, cfg: Dict[str, Any]) -> None:
        """Add API key from environment if configured and not explicitly provided."""
        if not cfg.get("api_key") and (env := cfg.get("api_key_env")):
            cfg["api_key"] = os.getenv(env)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a provider, with environment variables applied."""
        self._ensure_section(provider)
        # Start with defaults for this provider, then apply user config
        cfg = {**DEFAULTS.get(provider, {}), **self.providers[provider]}
        self._merge_env_key(cfg)
        return cfg

    def update_provider_config(self, provider: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a provider."""
        self._ensure_section(provider)
        self.providers[provider].update(updates)

    # ── active provider / model ──────────────────────────────────────
    @property
    def _glob(self) -> Dict[str, Any]:
        """Get the global configuration section."""
        self._ensure_section("__global__")
        return self.providers["__global__"]

    def get_active_provider(self) -> str:
        """Get the active provider name."""
        return self._glob.get("active_provider", DEFAULTS["__global__"]["active_provider"])

    def set_active_provider(self, provider: str) -> None:
        """Set the active provider name."""
        self._glob["active_provider"] = provider

    def get_active_model(self) -> str:
        """Get the active model name."""
        return self._glob.get("active_model", DEFAULTS["__global__"]["active_model"])

    def set_active_model(self, model: str) -> None:
        """Set the active model name."""
        self._glob["active_model"] = model

    # ── convenience getters ─────────────────────────────────────────
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get the API key for a provider."""
        return self.get_provider_config(provider).get("api_key")

    def get_api_base(self, provider: str) -> Optional[str]:
        """Get the API base URL for a provider."""
        return self.get_provider_config(provider).get("api_base")

    def get_default_model(self, provider: str) -> str:
        """Get the default model for a provider."""
        return self.get_provider_config(provider).get("default_model", "")