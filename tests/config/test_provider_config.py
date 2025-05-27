# tests/test_provider_config.py
import os
from unittest.mock import patch

from chuk_llm.llm.configuration.provider_config import ProviderConfig, DEFAULTS


class TestProviderConfig:
    """Tests for the ProviderConfig class."""

    # ── Construction ─────────────────────────────────────────────────────────
    def test_init_with_defaults(self):
        pc = ProviderConfig()
        assert "openai" in pc.providers
        assert pc.providers["openai"]["default_model"] == DEFAULTS["openai"]["default_model"]
        assert "__global__" in pc.providers

    def test_custom_overlay(self):
        overlay = {
            "openai": {"default_model": "custom-model"},
            "demo": {"client": "demo:Client"},
        }
        pc = ProviderConfig(overlay)
        assert pc.get_provider_config("openai")["default_model"] == "custom-model"
        assert pc.get_provider_config("demo")["client"] == "demo:Client"

    # ── Env merge helpers ────────────────────────────────────────────────────
    def test_env_merge_for_api_key(self):
        with patch.dict(os.environ, {"FOO_KEY": "from-env"}):
            pc = ProviderConfig({"foo": {"api_key_env": "FOO_KEY"}})
            assert pc.get_provider_config("foo").get("api_key") == "from-env"

    # ── Simple getters / setters ────────────────────────────────────────────
    def test_active_provider_and_model(self):
        pc = ProviderConfig()
        pc.set_active_provider("groq")
        pc.set_active_model("llama-3.3-70b-versatile")
        assert pc.get_active_provider() == "groq"
        assert pc.get_active_model().startswith("llama-3")

    def test_active_provider_and_model_setters(self):
        pc = ProviderConfig()
        pc.set_active_provider("groq")
        pc.set_active_model("llama-x")
        assert pc.get_active_provider() == "groq"
        assert pc.get_active_model()     == "llama-x"
