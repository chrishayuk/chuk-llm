# tests/llm/providers/test_groq_client_comprehensive.py
"""
Comprehensive test suite for GroqAILLMClient

Tests all Groq-specific functionality including:
- Smart model detection and feature support
- Known production and preview models
- Groq-specific error handling and retries
- Tool call normalization (always list, never None)
- Model family detection
- Optimization profiles
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_llm.core.enums import MessageRole
from chuk_llm.llm.providers.groq_client import GroqAILLMClient


class TestGroqClientInit:
    """Test suite for Groq client initialization"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        client = GroqAILLMClient()
        assert client.model == "llama-3.3-70b-versatile"
        assert client.detected_provider == "groq"
        assert client.provider_name == "groq"
        assert "groq" in client.api_base

    def test_init_with_custom_model(self):
        """Test initialization with custom model"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        assert client.model == "llama-3.1-8b-instant"
        assert client.detected_provider == "groq"

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base"""
        custom_base = "https://custom.groq.com/v1"
        client = GroqAILLMClient(api_base=custom_base)
        assert client.api_base == custom_base


class TestGroqKnownModels:
    """Test suite for known Groq models"""

    @pytest.mark.parametrize("model,expected_context", [
        ("llama-3.1-8b-instant", 131072),
        ("llama-3.3-70b-versatile", 131072),
        ("meta-llama/llama-guard-4-12b", 131072),
    ])
    def test_production_model_context(self, model, expected_context):
        """Test that production models have correct context lengths"""
        client = GroqAILLMClient(model=model)
        info = client._get_known_model_info(model)
        assert info is not None
        assert info["context"] == expected_context

    @pytest.mark.parametrize("model,expected_family", [
        ("llama-3.1-8b-instant", "llama"),
        ("llama-3.3-70b-versatile", "llama"),
        ("whisper-large-v3", "whisper"),
        ("deepseek-r1-distill-llama-70b", "deepseek"),
        ("openai/gpt-oss-120b", "gpt-oss"),
    ])
    def test_known_model_families(self, model, expected_family):
        """Test that known models return correct families"""
        client = GroqAILLMClient(model=model)
        info = client._get_known_model_info(model)
        assert info is not None
        assert info["family"] == expected_family

    def test_preview_model_detection(self):
        """Test that preview models are detected correctly"""
        client = GroqAILLMClient(model="deepseek-r1-distill-llama-70b")
        info = client._get_known_model_info("deepseek-r1-distill-llama-70b")
        assert info is not None
        assert "reasoning" in info["features"]

    def test_unknown_model_returns_none(self):
        """Test that unknown models return None"""
        client = GroqAILLMClient(model="unknown-model")
        info = client._get_known_model_info("unknown-model")
        assert info is None


class TestGroqSmartDefaults:
    """Test suite for Groq smart defaults"""

    @pytest.mark.parametrize("model,expected_features", [
        ("llama-3.1-8b-instant", {"text", "streaming", "tools", "system_messages", "json_mode"}),
        ("whisper-large-v3", {"audio", "transcription"}),
        ("playai-tts", {"text_to_speech", "audio"}),
        ("meta-llama/llama-guard-4-12b", {"text", "streaming"}),
        ("openai/gpt-oss-120b", {"text", "streaming", "tools", "reasoning", "system_messages", "json_mode"}),
        ("deepseek-r1-distill-llama-70b", {"text", "streaming", "tools", "reasoning", "system_messages", "json_mode"}),
    ])
    def test_smart_default_features(self, model, expected_features):
        """Test that smart defaults return correct features"""
        features = GroqAILLMClient._get_smart_default_features(model)
        assert expected_features.issubset(features)

    @pytest.mark.parametrize("model", [
        "llama-3.1-70b-versatile",
        "llama-4-maverick-17b",
        "qwen/qwen3-32b",
        "unknown-new-model",
    ])
    def test_smart_defaults_for_text_models(self, model):
        """Test that text models get reasonable defaults"""
        features = GroqAILLMClient._get_smart_default_features(model)
        # All text models should have at least these
        assert "text" in features
        assert "streaming" in features
        assert "system_messages" in features

    def test_smart_default_parameters_llama(self):
        """Test smart default parameters for Llama models"""
        params = GroqAILLMClient._get_smart_default_parameters("llama-3.1-70b-versatile")
        assert params["max_context_length"] == 131072
        assert params["max_output_tokens"] == 32768
        assert params["supports_tools"] is True
        assert params["ultra_fast_inference"] is True

    def test_smart_default_parameters_gpt_oss(self):
        """Test smart default parameters for GPT-OSS models"""
        params = GroqAILLMClient._get_smart_default_parameters("openai/gpt-oss-120b")
        assert params["max_context_length"] == 131072
        assert params["supports_reasoning"] is True
        assert params["ultra_fast_inference"] is True
        assert params["open_source"] is True

    def test_smart_default_parameters_unknown_model(self):
        """Test smart defaults for unknown models are reasonable"""
        params = GroqAILLMClient._get_smart_default_parameters("unknown-future-model")
        assert params["max_context_length"] == 131072  # Groq standard
        assert params["supports_tools"] is True  # Be optimistic
        assert params["ultra_fast_inference"] is True


class TestGroqFeatureSupport:
    """Test suite for Groq feature support"""

    def test_supports_feature_with_known_model(self):
        """Test feature support for known models"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")

        # Mock the parent's supports_feature to return None (no config)
        with patch.object(GroqAILLMClient.__bases__[0], 'supports_feature', return_value=None):
            assert client.supports_feature("tools") is True
            assert client.supports_feature("streaming") is True
            assert client.supports_feature("json_mode") is True

    def test_supports_feature_audio_model(self):
        """Test feature support for audio models"""
        client = GroqAILLMClient(model="whisper-large-v3")

        with patch.object(GroqAILLMClient.__bases__[0], 'supports_feature', return_value=None):
            assert client.supports_feature("audio") is True
            assert client.supports_feature("transcription") is True
            assert client.supports_feature("tools") is False

    def test_supports_feature_fallback_on_error(self):
        """Test that feature support fails gracefully"""
        client = GroqAILLMClient(model="test-model")

        # Mock parent to raise exception
        with patch.object(GroqAILLMClient.__bases__[0], 'supports_feature', side_effect=Exception("Test error")):
            # Should fall back to optimistic defaults
            assert client.supports_feature("text") is True
            assert client.supports_feature("streaming") is True
            assert client.supports_feature("tools") is True


class TestGroqModelInfo:
    """Test suite for Groq model info"""

    def test_get_model_info_production_model(self):
        """Test model info for production models"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")

        # Mock parent's get_model_info
        with patch.object(GroqAILLMClient.__bases__[0], 'get_model_info', return_value={}):
            info = client.get_model_info()

            assert info["provider"] == "groq"
            assert info["detected_provider"] == "groq"
            assert info["model_status"] == "production"
            assert "groq_specific" in info
            assert info["groq_specific"]["ultra_fast_inference"] is True
            assert info["using_known_model"] is True

    def test_get_model_info_preview_model(self):
        """Test model info for preview models"""
        client = GroqAILLMClient(model="deepseek-r1-distill-llama-70b")

        with patch.object(GroqAILLMClient.__bases__[0], 'get_model_info', return_value={}):
            info = client.get_model_info()

            assert info["model_status"] == "preview"
            assert info["using_known_model"] is True

    def test_get_model_info_unknown_model(self):
        """Test model info for unknown models uses smart defaults"""
        client = GroqAILLMClient(model="future-model-v2")

        with patch.object(GroqAILLMClient.__bases__[0], 'get_model_info', return_value={}):
            with patch.object(client, '_has_explicit_model_config', return_value=False):
                info = client.get_model_info()

                assert info["model_status"] == "unknown"
                assert "using_smart_defaults" in info
                assert "smart_default_features" in info

    def test_unsupported_parameters(self):
        """Test that Groq-specific unsupported parameters are listed"""
        client = GroqAILLMClient()

        with patch.object(GroqAILLMClient.__bases__[0], 'get_model_info', return_value={}):
            info = client.get_model_info()

            unsupported = info["unsupported_parameters"]
            assert "frequency_penalty" in unsupported
            assert "presence_penalty" in unsupported
            assert "logit_bias" in unsupported


class TestGroqModelFamily:
    """Test suite for Groq model family detection"""

    @pytest.mark.parametrize("model,expected_family", [
        ("llama-3.1-8b-instant", "llama"),
        ("llama-4-maverick-17b", "llama4"),
        ("whisper-large-v3", "whisper"),
        ("playai-tts-arabic", "tts"),
        ("meta-llama/llama-guard-4-12b", "llama-guard"),
        ("openai/gpt-oss-120b", "gpt-oss"),
        ("deepseek-r1-distill-llama-70b", "deepseek"),
        ("qwen/qwen3-32b", "qwen"),
        ("moonshotai/kimi-k2-instruct", "kimi"),
        ("compound-beta-mini", "compound"),
        ("unknown-model", "unknown"),
    ])
    def test_detect_groq_model_family(self, model, expected_family):
        """Test model family detection for various models"""
        client = GroqAILLMClient(model=model)
        family = client._detect_groq_model_family()
        assert family == expected_family


class TestGroqOptimizationProfile:
    """Test suite for Groq optimization profiles"""

    @pytest.mark.parametrize("model,expected_profile", [
        ("whisper-large-v3", "audio_processing"),
        ("playai-tts", "audio_processing"),
        ("meta-llama/llama-guard-4-12b", "safety_filtering"),
        ("llama-3.3-70b-versatile", "high_throughput"),
        ("llama-3.1-8b-instant", "low_latency"),
        ("deepseek-r1-distill-llama-70b", "deep_reasoning"),
        ("compound-beta-mini", "multi_model_system"),
        ("llama-3.1-70b-versatile", "high_throughput"),
    ])
    def test_get_optimization_profile(self, model, expected_profile):
        """Test optimization profile detection"""
        client = GroqAILLMClient(model=model)
        profile = client._get_optimization_profile()
        assert profile == expected_profile


class TestGroqNormalization:
    """Test suite for Groq-specific normalization"""

    def test_normalize_message_ensures_tool_calls_list(self):
        """Test that tool_calls is always a list, never None"""
        client = GroqAILLMClient()

        # Create a mock message with tool_calls=None
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Test response"
        mock_message.tool_calls = None

        # Mock the parent's _normalise_message to return tool_calls=None
        with patch.object(GroqAILLMClient.__bases__[0], '_normalise_message', return_value={
            "role": "assistant",
            "content": "Test response",
            "tool_calls": None
        }):
            result = client._normalise_message(mock_message)

            # CRITICAL: tool_calls must be a list, not None
            assert result["tool_calls"] == []
            assert isinstance(result["tool_calls"], list)
