# tests/test_config_validator.py
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from chuk_llm.configuration.config_validator import ConfigValidator


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
            "client_class": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
            "api_key_env": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini",
        }
        ok, issues = ConfigValidator.validate_provider_config("openai", cfg)
        assert ok, issues

    @_with_env(ANTHROPIC_API_KEY="sk-ant-test")
    def test_anthropic_passes(self):
        cfg = {
            "client_class": "chuk_llm.llm.providers.anthropic_client:AnthropicLLMClient",
            "api_key_env": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-7-sonnet-20250219",
        }
        ok, issues = ConfigValidator.validate_provider_config("anthropic", cfg)
        assert ok, issues

    def test_missing_client_class_fails(self):
        cfg = {
            "default_model": "gpt-4o-mini",
        }
        ok, issues = ConfigValidator.validate_provider_config("openai", cfg)
        assert not ok
        assert "Missing 'client_class'" in issues[0]

    def test_missing_default_model_fails(self):
        cfg = {
            "client_class": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
        }
        ok, issues = ConfigValidator.validate_provider_config("openai", cfg)
        assert not ok
        assert "Missing 'default_model'" in issues[0]

    def test_missing_api_key_fails(self):
        cfg = {
            "client_class": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
            "api_key_env": "MISSING_API_KEY",
            "default_model": "gpt-4o-mini",
        }
        ok, issues = ConfigValidator.validate_provider_config("openai", cfg)
        assert not ok
        assert "Missing API key" in issues[0]

    def test_invalid_api_base_fails(self):
        cfg = {
            "client_class": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
            "api_base": "not-a-url",
            "default_model": "gpt-4o-mini",
        }
        ok, issues = ConfigValidator.validate_provider_config("openai", cfg)
        assert not ok
        assert "Invalid API base URL" in issues[0]

    def test_valid_api_base_passes(self):
        cfg = {
            "client_class": "chuk_llm.llm.providers.openai_client:OpenAILLMClient",
            "api_base": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
        }
        ok, issues = ConfigValidator.validate_provider_config("openai", cfg)
        assert ok, issues

    def test_ollama_no_api_key_required(self):
        cfg = {
            "client_class": "chuk_llm.llm.providers.ollama_client:OllamaLLMClient",
            "default_model": "llama3",
        }
        ok, issues = ConfigValidator.validate_provider_config("ollama", cfg)
        assert ok, issues


# ──────────────────────────────────────────────────────────────────────────────
# Request-compatibility checks
# ──────────────────────────────────────────────────────────────────────────────
class TestRequestCompatibility:
    def test_basic_request_passes(self):
        msgs = [{"role": "user", "content": "hello"}]
        ok, issues = ConfigValidator.validate_request_compatibility("openai", msgs)
        assert ok, issues

    def test_vision_content_detection(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                ],
            }
        ]
        
        # Test that vision content is detected
        has_vision = ConfigValidator._has_vision_content(msgs)
        assert has_vision

    def test_no_vision_content_detection(self):
        msgs = [{"role": "user", "content": "hello"}]
        has_vision = ConfigValidator._has_vision_content(msgs)
        assert not has_vision

    def test_empty_messages_no_vision(self):
        has_vision = ConfigValidator._has_vision_content([])
        assert not has_vision

    def test_none_messages_no_vision(self):
        has_vision = ConfigValidator._has_vision_content(None)
        assert not has_vision


# ──────────────────────────────────────────────────────────────────────────────
# URL validation tests
# ──────────────────────────────────────────────────────────────────────────────
class TestUrlValidation:
    def test_valid_https_url(self):
        assert ConfigValidator._is_valid_url("https://api.openai.com/v1")

    def test_valid_http_url(self):
        assert ConfigValidator._is_valid_url("http://localhost:8080")

    def test_valid_localhost_url(self):
        assert ConfigValidator._is_valid_url("http://localhost:11434")

    def test_valid_ip_url(self):
        assert ConfigValidator._is_valid_url("http://192.168.1.1:8080")

    def test_invalid_no_protocol(self):
        assert not ConfigValidator._is_valid_url("api.openai.com")

    def test_invalid_empty_url(self):
        assert not ConfigValidator._is_valid_url("")

    def test_invalid_none_url(self):
        assert not ConfigValidator._is_valid_url(None)

    def test_invalid_malformed_url(self):
        assert not ConfigValidator._is_valid_url("not-a-url")


# ──────────────────────────────────────────────────────────────────────────────
# Token estimation tests
# ──────────────────────────────────────────────────────────────────────────────
class TestTokenEstimation:
    def test_simple_message_token_count(self):
        msgs = [{"role": "user", "content": "hello world"}]
        tokens = ConfigValidator._estimate_token_count(msgs)
        # "hello world" = 11 chars, so ~2-3 tokens
        assert 2 <= tokens <= 3

    def test_large_message_token_count(self):
        huge_content = "A" * 500_000
        msgs = [{"role": "user", "content": huge_content}]
        tokens = ConfigValidator._estimate_token_count(msgs)
        # 500,000 chars / 4 = 125,000 tokens
        assert tokens == 125_000

    def test_multimodal_message_token_count(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}
                ]
            }
        ]
        tokens = ConfigValidator._estimate_token_count(msgs)
        # Only text content is counted: "hello" = 5 chars, so ~1 token
        assert tokens == 1

    def test_empty_messages_zero_tokens(self):
        tokens = ConfigValidator._estimate_token_count([])
        assert tokens == 0

    def test_none_messages_zero_tokens(self):
        tokens = ConfigValidator._estimate_token_count(None)
        assert tokens == 0

    def test_multiple_messages_token_count(self):
        msgs = [
            {"role": "user", "content": "hello"},      # 5 chars
            {"role": "assistant", "content": "hi"},    # 2 chars  
            {"role": "user", "content": "goodbye"}     # 7 chars
        ]
        tokens = ConfigValidator._estimate_token_count(msgs)
        # Total: 14 chars / 4 = 3.5, rounded down = 3 tokens
        assert tokens == 3