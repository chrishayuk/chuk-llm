# chuk_llm/llm/configuration/capabilities.py
"""
Model-/Provider-capability registry with YAML overlay
=====================================================

* Built-in defaults stay in code (minimal stub – OpenAI only).
* A `llm_capabilities.yaml` can extend/override everything.
  Search order (first match wins):

  1.  $CHUK_LLM_CAPABILITIES_FILE  env-var
  2.  llm_capabilities.yaml  in the CWD
  3.  llm_capabilities.yml   in the CWD
  4.  llm_capabilities.{yaml,yml} **in the package root**
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – graceful degradation
    yaml = None  # noqa: N816

logger = logging.getLogger(__name__)

# ──────────────────────────────── feature flags ───────────────────────────────
class Feature(str, Enum):
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    PARALLEL_CALLS = "parallel_calls"
    SYSTEM_MESSAGES = "system_messages"
    MULTIMODAL = "multimodal"

    @classmethod
    def from_raw(cls, raw: str) -> "Feature":
        try:
            return cls(raw)
        except ValueError as exc:  # pragma: no cover – dev error
            raise ValueError(f"Unknown feature string: {raw}") from exc


# ─────────────────────────── data-classes (unchanged) ─────────────────────────
@dataclass
class ModelCapabilities:
    pattern: str
    features: Set[Feature]
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None

    def matches(self, model_name: str) -> bool:  # noqa: D401
        return bool(re.match(self.pattern, model_name, flags=re.I))


@dataclass
class ProviderCapabilities:
    name: str
    features: Set[Feature]
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    rate_limits: Optional[Dict[str, int]] = None
    model_capabilities: Optional[List[ModelCapabilities]] = None

    # helpers
    def supports(self, feat: Feature) -> bool:
        return feat in self.features

    def get_rate_limit(self, tier: str = "default") -> Optional[int]:
        return (self.rate_limits or {}).get(tier)

    def get_model_capabilities(self, model: str | None) -> ModelCapabilities:
        if model and self.model_capabilities:
            for mc in self.model_capabilities:
                if mc.matches(model):
                    return mc
        # fallback
        return ModelCapabilities(
            pattern=".*",
            features=self.features,
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens,
        )


# ──────────────────────────── built-in minimal stub ───────────────────────────
_DEFAULT_REGISTRY: Dict[str, ProviderCapabilities] = {
    "openai": ProviderCapabilities(
        name="OpenAI",
        features={
            Feature.STREAMING,
            Feature.TOOLS,
            Feature.VISION,
            Feature.JSON_MODE,
            Feature.PARALLEL_CALLS,
            Feature.SYSTEM_MESSAGES,
        },
        max_context_length=128_000,
        max_output_tokens=4_096,
        rate_limits={"default": 3_500},
    ),
}

# ─────────────────────────── YAML overlay helper ──────────────────────────────
# Added ➊ & ➋  ➜ package-root look-ups
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
_YAML_CANDIDATES = (
    os.getenv("CHUK_LLM_CAPABILITIES_FILE"),          # 0
    "llm_capabilities.yaml",                          # 1
    "llm_capabilities.yml",                           # 2
    _PKG_ROOT / "llm_capabilities.yaml",              # ➊
    _PKG_ROOT / "llm_capabilities.yml",               # ➋
)


def _find_yaml() -> Optional[Path]:
    for candidate in _YAML_CANDIDATES:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.is_file():
            return p
    return None


def _parse_feature_set(raw: List[str | Feature]) -> Set[Feature]:
    return {f if isinstance(f, Feature) else Feature.from_raw(str(f)) for f in raw}


def _overlay_from_yaml(reg: Dict[str, ProviderCapabilities]) -> None:
    cfg_path = _find_yaml()
    if not cfg_path:
        logger.debug("capabilities: no YAML file found – using built-ins only")
        return
    if yaml is None:
        logger.warning("pyyaml missing – cannot read %s; using built-ins", cfg_path)
        return

    logger.info("capabilities: loading overrides from %s", cfg_path)
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            doc: Dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover
        logger.error("capabilities: failed to parse %s: %s", cfg_path, exc)
        return

    for pname, pdata in doc.items():
        base = reg.get(pname) or ProviderCapabilities(pname, set())
        reg[pname] = base

        # scalar fields
        for key in ("max_context_length", "max_output_tokens", "rate_limits"):
            if key in pdata:
                setattr(base, key, pdata[key])

        if "features" in pdata:
            base.features = _parse_feature_set(pdata["features"])

        # model-specific entries
        models: list[ModelCapabilities] = []
        for m in pdata.get("models", []):
            models.append(
                ModelCapabilities(
                    pattern=m.get("pattern", ".*"),
                    features=_parse_feature_set(m.get("features", list(base.features))),
                    max_context_length=m.get("max_context_length", base.max_context_length),
                    max_output_tokens=m.get("max_output_tokens", base.max_output_tokens),
                )
            )
        if models:
            base.model_capabilities = models


# build public registry
PROVIDER_CAPABILITIES: Dict[str, ProviderCapabilities] = {
    k: v for k, v in _DEFAULT_REGISTRY.items()
}
_overlay_from_yaml(PROVIDER_CAPABILITIES)

# ────────────────────────────── public helper API ─────────────────────────────
class CapabilityChecker:
    """Query helpers for *PROVIDER_CAPABILITIES*."""

    @staticmethod
    def can_handle_request(
        provider: str,
        model: str | None = None,
        *,
        has_tools: bool = False,
        has_vision: bool = False,
        needs_streaming: bool = False,
        needs_json: bool = False,
    ) -> tuple[bool, List[str]]:
        if provider not in PROVIDER_CAPABILITIES:
            return False, [f"Unknown provider: {provider}"]

        caps = PROVIDER_CAPABILITIES[provider].get_model_capabilities(model)
        feats = caps.features

        problems: list[str] = []
        if has_tools and Feature.TOOLS not in feats:
            problems.append("tools not supported")
        if has_vision and Feature.VISION not in feats:
            problems.append("vision not supported")
        if needs_streaming and Feature.STREAMING not in feats:
            problems.append("streaming not supported")
        if needs_json and Feature.JSON_MODE not in feats:
            problems.append("JSON mode not supported")
        return (not problems), problems

    @staticmethod
    def get_best_provider(
        requirements: Set[Feature],
        exclude: Optional[Set[str]] = None,
    ) -> Optional[str]:
        exclude = exclude or set()
        choices: list[tuple[str, int]] = []
        for name, caps in PROVIDER_CAPABILITIES.items():
            if name in exclude:
                continue
            if requirements.issubset(caps.features):
                choices.append((name, caps.get_rate_limit() or 0))
        return max(choices, key=lambda t: t[1])[0] if choices else None

    @staticmethod
    def get_model_info(provider: str, model: str) -> Dict[str, Any]:
        if provider not in PROVIDER_CAPABILITIES:
            return {"error": f"Unknown provider: {provider}"}
        caps = PROVIDER_CAPABILITIES[provider].get_model_capabilities(model)
        return {
            "provider": provider,
            "model": model,
            "features": [f.value for f in caps.features],
            "max_context_length": caps.max_context_length,
            "max_output_tokens": caps.max_output_tokens,
            "supports_streaming": Feature.STREAMING in caps.features,
            "supports_tools": Feature.TOOLS in caps.features,
            "supports_vision": Feature.VISION in caps.features,
            "supports_json_mode": Feature.JSON_MODE in caps.features,
        }
