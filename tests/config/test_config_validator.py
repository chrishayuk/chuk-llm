# tests/test_config_validator.py
from __future__ import annotations

import os
from typing import Any, Dict, Set
from unittest.mock import patch

import pytest

from chuk_llm.llm.configuration.config_validator import (
    ConfigValidator,
    ValidatedProviderConfig,
)
from chuk_llm.llm.configuration.capabilities import (
    Feature,
    CapabilityChecker,
    PROVIDER_CAPABILITIES,
)
from chuk_llm.llm.configuration.provider_config import ProviderConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def _with_env(**vars_: str):
    """Temporarily patch `os.environ` for the lifetime of a single test."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            with patch.dict(os.environ, vars_, clear=False):
                return fn(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Provider-level configuration validation
# ──────────────────────────────────────────────────────────────────────────────
class TestValidateProviderConfig:
    @_with_env(OPENAI_API_KEY="sk-test")
    def test_openai_minimal_passes(self):
        cfg = {
            "client": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
            "default_model": "gpt-4o-mini",
        }
        ok, issues = ConfigValidator.validate_provider_config("openai", cfg)
        assert ok, issues

    @_with_env(ANTHROPIC_API_KEY="sk-ant-test")
    def test_anthropic_passes(self):
        cfg = {
            "client": "chuk_llm.llm.providers.anthropic_client:AnthropicLLMClient",
            "default_model": "claude-3-7-sonnet-20250219",
        }
        ok, issues = ConfigValidator.validate_provider_config("anthropic", cfg)
        assert ok, issues

    def test_unknown_provider_fails(self):
        ok, issues = ConfigValidator.validate_provider_config("nope", {})
        assert not ok and "Unsupported provider" in issues[0]


# ──────────────────────────────────────────────────────────────────────────────
# ValidatedProviderConfig integration
# ──────────────────────────────────────────────────────────────────────────────
class TestValidatedProviderConfig:
    @_with_env(OPENAI_API_KEY="sk-x", DEEPSEEK_API_KEY="sk-deep")
    def test_yaml_like_overlay(self):
        overlay = {
            "openai": {"api_base": "https://alt.example.com"},
            "deepseek": {
                "inherits": "openai",
                "api_key_env": "DEEPSEEK_API_KEY",
                "api_base": "https://api.deepseek.com",
                "default_model": "deepseek-chat",
            },
        }

        # non-strict: we may override only a subset of keys
        cfg = ValidatedProviderConfig(overlay, strict=False)

        openai_cfg = cfg.get_provider_config("openai")
        deepseek_cfg = cfg.get_provider_config("deepseek")

        # overlay took effect
        assert openai_cfg.get("api_base", "").startswith("https://alt.")
        assert deepseek_cfg.get("api_base", "").endswith("deepseek.com")

    def test_provider_config_facade_methods(self):
        pc = ProviderConfig()  # defaults + YAML merge
        assert pc.get_active_provider()
        assert pc.get_active_model()

        pc.set_active_provider("groq")
        pc.set_active_model("llama-3.3-70b-versatile")

        assert pc.get_active_provider() == "groq"
        assert pc.get_active_model().startswith("llama-3")


# ──────────────────────────────────────────────────────────────────────────────
# CapabilityChecker sanity
# ──────────────────────────────────────────────────────────────────────────────
class TestCapabilityChecker:
    @pytest.mark.parametrize(
        "provider,model,feature",
        [
            ("openai", "gpt-4o-mini", Feature.VISION),
            ("mistral", "pixtral-12b-latest", Feature.VISION),
            ("gemini", "gemini-2.0-flash", Feature.JSON_MODE),
            ("groq", "llama-3.3-70b-versatile", Feature.STREAMING),
        ],
    )
    def test_models_report_expected_feature(
        self,
        provider: str,
        model: str,
        feature: Feature,
    ):
        info = CapabilityChecker.get_model_info(provider, model)
        assert feature.value in info["features"], info

    def test_best_provider_chooser(self):
        wants: Set[Feature] = {Feature.STREAMING, Feature.TOOLS}
        best = CapabilityChecker.get_best_provider(wants)
        assert best in PROVIDER_CAPABILITIES
        assert wants.issubset(PROVIDER_CAPABILITIES[best].features)


# ──────────────────────────────────────────────────────────────────────────────
# Request-compatibility checks
# ──────────────────────────────────────────────────────────────────────────────
class TestRequestCompatibility:
    @_with_env(OPENAI_API_KEY="sk-foo")
    def test_streaming_and_tools_ok(self):
        msgs = [{"role": "user", "content": "hello"}]
        tools = [
            {"type": "function", "function": {"name": "dummy", "parameters": {}}},
        ]
        ok, issues = ConfigValidator.validate_request_compatibility(
            "openai",
            msgs,
            stream=True,
            tools=tools,
        )
        assert ok, issues

    def test_vision_rejected_for_nonvision_provider(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                ],
            }
        ]
        ok, issues = ConfigValidator.validate_request_compatibility("groq", msgs)

        # Groq has no VISION support, so request must be refused
        assert not ok
        assert issues  # at least one explanatory issue is present


# ──────────────────────────────────────────────────────────────────────────────
# Performance sanity
# ──────────────────────────────────────────────────────────────────────────────
def test_context_estimator_fast():
    huge = [{"role": "user", "content": "A" * 500_000}]
    tokens = ConfigValidator._estimate_token_count(huge)
    assert tokens == 125_000  # 4-char ≈ 1 token heuristic
