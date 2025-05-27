# chuk_llm/llm/config/provider_config.py
"""
Provider configuration management
=================================

* Built-in **DEFAULTS** are defined in this file.
* A user-supplied *providers.yaml* (location rules below) can **extend /
  override** any entry – including deep keys – and may use
  `inherits: other_provider` for syntactic sugar.
* In-memory *overlays* passed to ``ProviderConfig({...})`` are applied **last**
  (highest priority) and use the same merge rules.

YAML search order
-----------------
1.  Environment variable ``CHUK_LLM_PROVIDERS_YAML`` (absolute or relative).
2.  *providers.yaml* in the current working directory.
3.  *providers.yaml* in the package root (``src/chuk_llm``).

Environment variables referenced via ``*_env`` keys are resolved **lazily** on
*every* call to :py:meth:`get_provider_config`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# 1.  Built-in defaults                                                       #
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "__global__": {"active_provider": "openai", "active_model": "gpt-4o-mini"},
    # (only a subset of keys is shown – add/change freely)
    "openai": {
        "client": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
        "api_key_env": "OPENAI_API_KEY",
        "api_base": None,
        "api_key": None,
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "client": "chuk_llm.llm.providers.anthropic_client:AnthropicLLMClient",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_base": None,
        "api_key": None,
        "default_model": "claude-3-7-sonnet-20250219",
    },
    "groq": {
        "client": "chuk_llm.llm.providers.groq_client:GroqAILLMClient",
        "api_key_env": "GROQ_API_KEY",
        "api_base": "https://api.groq.com",
        "api_key": None,
        # **no vision feature – matches tests**
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "client": "chuk_llm.llm.providers.gemini_client:GeminiLLMClient",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key": None,
        "default_model": "gemini-2.0-flash",
    },
    "ollama": {
        "client": "chuk_llm.llm.providers.ollama_client:OllamaLLMClient",
        "api_key_env": None,
        "api_base": "http://localhost:11434",
        "api_key": None,
        "default_model": "qwen3",
    },
    "mistral": {
        "client": "chuk_llm.llm.providers.mistral_client:MistralLLMClient",
        "api_key_env": "MISTRAL_API_KEY",
        "api_base": None,
        "api_key": None,
        "default_model": "mistral-large-latest",
    },
    "watsonx": {
        "client": "chuk_llm.llm.providers.watsonx_client:WatsonXLLMClient",
        "api_key_env": "WATSONX_API_KEY",
        "api_key_fallback_env": "IBM_CLOUD_API_KEY",
        "project_id_env": "WATSONX_PROJECT_ID",
        "watsonx_ai_url_env": "WATSONX_AI_URL",
        "space_id_env": "WATSONX_SPACE_ID",
        "api_base": None,
        "api_key": None,
        "default_model": "ibm/granite-3-8b-instruct",
    },
    "deepseek": {
        "client": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_base": "https://api.deepseek.com",
        "api_key": None,
        "default_model": "deepseek-chat",
    },
}

# --------------------------------------------------------------------------- #
# 2.  YAML overlay helper                                                     #
# --------------------------------------------------------------------------- #
_YAML_ENV = "CHUK_LLM_PROVIDERS_YAML"
_YAML_NAME = "providers.yaml"


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict update (similar to ``dict.update`` but deep)."""
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml_if_any() -> Dict[str, Dict[str, Any]]:
    path_candidates = []
    if os.getenv(_YAML_ENV):
        path_candidates.append(Path(os.getenv(_YAML_ENV)))
    path_candidates.append(Path.cwd() / _YAML_NAME)
    path_candidates.append(Path(__file__).resolve().parent.parent.parent / _YAML_NAME)

    for p in path_candidates:
        if p.is_file():
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "pyyaml is required to read provider YAML files; "
                    f"install it or remove {p}"
                ) from exc
            return yaml.safe_load(p.read_text()) or {}
    return {}


def _apply_overlay(
    base: Dict[str, Dict[str, Any]],
    overlay: Dict[str, Dict[str, Any]],
) -> None:
    """Merge *overlay* into *base* (handles ``inherits``)."""
    for name, block in overlay.items():
        if name == "__global__":
            _deep_update(base["__global__"], block)
            continue

        parent_name = block.pop("inherits", None)
        parent_cfg = json.loads(json.dumps(base.get(parent_name, {}))) if parent_name else {}

        merged = _deep_update(parent_cfg, block)
        base[name] = _deep_update(base.get(name, {}), merged)


# --------------------------------------------------------------------------- #
# 3.  ProviderConfig class                                                    #
# --------------------------------------------------------------------------- #
class ProviderConfig:
    """Unified view over default, YAML and runtime overlays."""

    def __init__(self, runtime_overlay: Dict[str, Dict[str, Any]] | None = None) -> None:
        # deep-copy defaults
        self.providers: Dict[str, Dict[str, Any]] = json.loads(json.dumps(DEFAULTS))

        # YAML overlay (medium priority)
        _apply_overlay(self.providers, _load_yaml_if_any())

        # constructor overlay (highest priority)
        if runtime_overlay:
            _apply_overlay(self.providers, runtime_overlay)

    # ── helpers ────────────────────────────────────────────────────────────
    def _ensure_section(self, name: str) -> None:
        if name not in self.providers:
            self.providers[name] = {}

    @staticmethod
    def _merge_env_key(cfg: Dict[str, Any]) -> None:
        """Populate ``api_key`` from env if necessary."""
        if cfg.get("api_key"):
            return
        env = cfg.get("api_key_env") or cfg.get("api_key_fallback_env")
        if env:
            cfg["api_key"] = os.getenv(env)

    # ── public API ─────────────────────────────────────────────────────────
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        self._ensure_section(provider)
        cfg = json.loads(json.dumps(self.providers[provider]))  # copy
        self._merge_env_key(cfg)
        return cfg

    def update_provider_config(self, provider: str, updates: Dict[str, Any]) -> None:
        self._ensure_section(provider)
        _deep_update(self.providers[provider], updates)

    # global getters / setters --------------------------------------------
    @property
    def _glob(self) -> Dict[str, Any]:
        self._ensure_section("__global__")
        return self.providers["__global__"]

    def get_active_provider(self) -> str:
        return self._glob.get("active_provider", DEFAULTS["__global__"]["active_provider"])

    def set_active_provider(self, provider: str) -> None:
        self._glob["active_provider"] = provider

    def get_active_model(self) -> str:
        return self._glob.get("active_model", DEFAULTS["__global__"]["active_model"])

    def set_active_model(self, model: str) -> None:
        self._glob["active_model"] = model

    # convenience ---------------------------------------------------------
    def get_api_key(self, provider: str) -> Optional[str]:
        return self.get_provider_config(provider).get("api_key")

    def get_api_base(self, provider: str) -> Optional[str]:
        return self.get_provider_config(provider).get("api_base")

    def get_default_model(self, provider: str) -> str:
        return self.get_provider_config(provider).get("default_model", "")

