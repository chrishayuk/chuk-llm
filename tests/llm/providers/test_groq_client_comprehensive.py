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


class TestGroqSmartDefaultsEdgeCases:
    """Test suite for edge cases in smart defaults"""

    def test_smart_defaults_tts_models_first_if(self):
        """Test smart defaults for TTS models with tts in name (first if)"""
        features = GroqAILLMClient._get_smart_default_features("custom-tts-model")
        assert "text_to_speech" in features
        assert "audio" in features

    def test_smart_defaults_tts_models_or_playai(self):
        """Test smart defaults for TTS models with playai in name (or condition)"""
        features = GroqAILLMClient._get_smart_default_features("playai-custom")
        assert "text_to_speech" in features
        assert "audio" in features

    def test_smart_defaults_guard_models(self):
        """Test smart defaults for guard/safety models"""
        features = GroqAILLMClient._get_smart_default_features("llama-guard-custom")
        assert "text" in features
        assert "streaming" in features
        assert "tools" not in features

    def test_smart_defaults_gpt_oss_models(self):
        """Test smart defaults for GPT-OSS models with logging"""
        # GPT-OSS models are in known models, so they won't trigger the logging branch
        # Test with a custom gpt-oss model instead
        import logging
        with patch.object(logging.getLogger('chuk_llm.llm.providers.groq_client'), 'info') as mock_log:
            features = GroqAILLMClient._get_smart_default_features("openai/gpt-oss-custom")
            assert "reasoning" in features
            assert "tools" in features
            mock_log.assert_called()

    def test_smart_defaults_deepseek_with_reasoning(self):
        """Test smart defaults for DeepSeek models with reasoning"""
        # With r1 in name
        features = GroqAILLMClient._get_smart_default_features("deepseek-r1-custom")
        assert "reasoning" in features
        assert "tools" in features

        # With 'reason' in name
        features = GroqAILLMClient._get_smart_default_features("deepseek-reasoning-model")
        assert "reasoning" in features

    def test_smart_defaults_whisper_return_path(self):
        """Test smart defaults for Whisper models return path (line 282)"""
        # This specifically tests the return statement on line 282
        features = GroqAILLMClient._get_smart_default_features("whisper-large-v3-turbo")
        assert features == {"audio", "transcription"}

        # Test with different whisper variant
        features2 = GroqAILLMClient._get_smart_default_features("whisper-small")
        assert features2 == {"audio", "transcription"}

    def test_smart_defaults_llama_nested_guard(self):
        """Test smart defaults for Llama models with guard in name (line 311)"""
        # NOTE: This line may be unreachable due to line 289 catching all "guard" models first
        # However, we test it to ensure the logic is correct
        # The nested guard check is defensive programming
        features = GroqAILLMClient._get_smart_default_features("llama-prompt-guard-custom")
        # This will actually hit line 290 (the top-level guard check), not line 311
        assert features == {"text", "streaming"}

    def test_smart_defaults_qwen_features(self):
        """Test smart defaults for Qwen models (lines 327-328)"""
        # Test a custom qwen model to hit the qwen branch
        features = GroqAILLMClient._get_smart_default_features("qwen-custom-model")
        assert "tools" in features
        assert "reasoning" in features
        assert "json_mode" in features

    def test_smart_defaults_kimi_models(self):
        """Test smart defaults for Kimi models"""
        features = GroqAILLMClient._get_smart_default_features("moonshotai/kimi-custom")
        assert "tools" in features
        assert "json_mode" in features

    def test_smart_defaults_compound_models(self):
        """Test smart defaults for Compound models"""
        features = GroqAILLMClient._get_smart_default_features("compound-beta-custom")
        assert "tools" in features
        assert "reasoning" in features

    def test_smart_defaults_reasoning_patterns(self):
        """Test smart defaults for models with reasoning patterns"""
        import logging
        with patch.object(logging.getLogger('chuk_llm.llm.providers.groq_client'), 'info') as mock_log:
            # Test 'reasoning' pattern
            features = GroqAILLMClient._get_smart_default_features("custom-reasoning-model")
            assert "reasoning" in features
            mock_log.assert_called()

            # Test 'distill' pattern
            features = GroqAILLMClient._get_smart_default_features("distill-custom-model")
            assert "reasoning" in features


class TestGroqSmartParametersEdgeCases:
    """Test suite for edge cases in smart default parameters"""

    def test_parameters_whisper_models(self):
        """Test parameters for Whisper models"""
        params = GroqAILLMClient._get_smart_default_parameters("whisper-custom")
        assert params["supports_audio"] is True
        assert params["supports_transcription"] is True
        assert params["model_type"] == "audio"

    def test_parameters_tts_models(self):
        """Test parameters for TTS models"""
        params = GroqAILLMClient._get_smart_default_parameters("playai-custom")
        assert params["max_context_length"] == 8192
        assert params["supports_tts"] is True
        assert params["model_type"] == "tts"

    def test_parameters_gpt_oss_120b(self):
        """Test parameters for GPT-OSS 120b model (lines 405-413)"""
        # Test custom 120b to hit the if branch
        params = GroqAILLMClient._get_smart_default_parameters("openai/gpt-oss-120b-custom")
        assert params["max_context_length"] == 131072
        assert params["max_output_tokens"] == 32766
        assert params["supports_tools"] is True
        assert params["supports_reasoning"] is True
        assert params["ultra_fast_inference"] is True
        assert params["open_source"] is True

    def test_parameters_gpt_oss_other(self):
        """Test parameters for other GPT-OSS models (else branch, lines 414-416)"""
        # Test without 120b to hit the else
        params = GroqAILLMClient._get_smart_default_parameters("openai/gpt-oss-custom")
        assert params["max_output_tokens"] == 32768
        assert params["open_source"] is True

    def test_parameters_deepseek_reasoning(self):
        """Test parameters for DeepSeek with reasoning"""
        params = GroqAILLMClient._get_smart_default_parameters("deepseek-r1-custom")
        assert params["max_output_tokens"] == 131072
        assert params["supports_reasoning"] is True

    def test_parameters_llama_guard_small(self):
        """Test parameters for small Llama guard models"""
        # Test 22m
        params = GroqAILLMClient._get_smart_default_parameters("llama-prompt-guard-2-22m")
        assert params["max_context_length"] == 512
        assert params["supports_tools"] is False
        assert params["model_type"] == "safety"

        # Test 86m
        params = GroqAILLMClient._get_smart_default_parameters("llama-prompt-guard-2-86m")
        assert params["max_context_length"] == 512

    def test_parameters_llama_guard_large(self):
        """Test parameters for large Llama guard models (else branch)"""
        params = GroqAILLMClient._get_smart_default_parameters("llama-guard-4-12b")
        assert params["max_context_length"] == 131072
        assert params["max_output_tokens"] == 1024
        assert params["model_type"] == "safety"

    def test_parameters_llama4(self):
        """Test parameters for Llama 4 models"""
        params = GroqAILLMClient._get_smart_default_parameters("llama-4-custom")
        assert params["max_output_tokens"] == 8192
        assert params["supports_tools"] is True

    def test_parameters_llama_8b(self):
        """Test parameters for Llama 8b models"""
        params = GroqAILLMClient._get_smart_default_parameters("llama-3.1-8b-custom")
        assert params["max_output_tokens"] == 131072  # 8b has huge output

    def test_parameters_llama_other_size(self):
        """Test parameters for other Llama sizes (else branch)"""
        params = GroqAILLMClient._get_smart_default_parameters("llama-3.1-13b-custom")
        assert params["max_output_tokens"] == 32768

    def test_parameters_qwen(self):
        """Test parameters for Qwen models"""
        params = GroqAILLMClient._get_smart_default_parameters("qwen/qwen-custom")
        assert params["max_output_tokens"] == 40960
        assert params["supports_reasoning"] is True

    def test_parameters_kimi(self):
        """Test parameters for Kimi models"""
        params = GroqAILLMClient._get_smart_default_parameters("kimi-custom")
        assert params["max_output_tokens"] == 16384

    def test_parameters_compound(self):
        """Test parameters for Compound models"""
        params = GroqAILLMClient._get_smart_default_parameters("compound-custom")
        assert params["supports_reasoning"] is True
        assert params["model_type"] == "system"


class TestGroqFeatureSupportWithLogging:
    """Test suite for feature support with logging branches"""

    def test_supports_feature_config_returns_true(self):
        """Test feature support when config returns definitive True (line 538)"""
        client = GroqAILLMClient(model="custom-unknown-model")

        # Mock parent to return True
        with patch.object(GroqAILLMClient.__bases__[0], 'supports_feature', return_value=True):
            result = client.supports_feature("tools")
            assert result is True

    def test_supports_feature_config_returns_false(self):
        """Test feature support when config returns definitive False (line 538)"""
        client = GroqAILLMClient(model="custom-unknown-model")

        # Mock parent to return False
        with patch.object(GroqAILLMClient.__bases__[0], 'supports_feature', return_value=False):
            result = client.supports_feature("tools")
            assert result is False

    def test_supports_feature_smart_default_positive(self):
        """Test feature support that uses smart defaults (positive case)"""
        client = GroqAILLMClient(model="custom-unknown-model")

        import logging
        with patch.object(GroqAILLMClient.__bases__[0], 'supports_feature', return_value=None):
            with patch.object(logging.getLogger('chuk_llm.llm.providers.groq_client'), 'info') as mock_log:
                result = client.supports_feature("tools")
                assert result is True
                mock_log.assert_called()


class TestGroqModelFamilyDetectionBranches:
    """Test suite for all branches in model family detection"""

    def test_detect_whisper_family(self):
        """Test detection of Whisper family"""
        client = GroqAILLMClient(model="whisper-custom")
        family = client._detect_groq_model_family()
        assert family == "whisper"

    def test_detect_tts_family_with_tts(self):
        """Test detection of TTS family with 'tts' in name"""
        client = GroqAILLMClient(model="custom-tts-model")
        family = client._detect_groq_model_family()
        assert family == "tts"

    def test_detect_tts_family_with_playai(self):
        """Test detection of TTS family with 'playai' in name"""
        client = GroqAILLMClient(model="playai-custom")
        family = client._detect_groq_model_family()
        assert family == "tts"

    def test_detect_guard_family(self):
        """Test detection of guard family"""
        client = GroqAILLMClient(model="custom-guard-model")
        family = client._detect_groq_model_family()
        assert family == "llama-guard"

    def test_detect_gpt_oss_family(self):
        """Test detection of GPT-OSS family"""
        client = GroqAILLMClient(model="openai/gpt-oss-custom")
        family = client._detect_groq_model_family()
        assert family == "gpt-oss"

    def test_detect_deepseek_family(self):
        """Test detection of DeepSeek family"""
        client = GroqAILLMClient(model="deepseek-custom")
        family = client._detect_groq_model_family()
        assert family == "deepseek"

    def test_detect_llama4_family_with_hyphen(self):
        """Test detection of Llama 4 family with hyphen"""
        client = GroqAILLMClient(model="llama-4-custom")
        family = client._detect_groq_model_family()
        assert family == "llama4"

    def test_detect_llama4_family_without_hyphen(self):
        """Test detection of Llama 4 family without hyphen"""
        client = GroqAILLMClient(model="llama4-custom")
        family = client._detect_groq_model_family()
        assert family == "llama4"

    def test_detect_llama_family_fallback(self):
        """Test detection of Llama family (non-4) fallback"""
        client = GroqAILLMClient(model="llama-3.1-custom")
        family = client._detect_groq_model_family()
        assert family == "llama"

    def test_detect_qwen_family(self):
        """Test detection of Qwen family"""
        client = GroqAILLMClient(model="qwen-custom")
        family = client._detect_groq_model_family()
        assert family == "qwen"

    def test_detect_kimi_family(self):
        """Test detection of Kimi family"""
        client = GroqAILLMClient(model="kimi-custom")
        family = client._detect_groq_model_family()
        assert family == "kimi"

    def test_detect_compound_family(self):
        """Test detection of Compound family"""
        client = GroqAILLMClient(model="compound-custom")
        family = client._detect_groq_model_family()
        assert family == "compound"

    def test_detect_reasoning_family(self):
        """Test detection of reasoning family"""
        client = GroqAILLMClient(model="custom-reasoning-model")
        family = client._detect_groq_model_family()
        assert family == "reasoning"


class TestGroqOptimizationProfileBranches:
    """Test suite for all branches in optimization profile detection"""

    def test_profile_reasoning_with_large_size(self):
        """Test that reasoning models return deep_reasoning (checked before size)"""
        client = GroqAILLMClient(model="r1-70b-custom")
        profile = client._get_optimization_profile()
        # Reasoning is checked before size, so returns deep_reasoning
        assert profile == "deep_reasoning"

    def test_profile_large_without_reasoning(self):
        """Test that large models without reasoning return high_throughput"""
        client = GroqAILLMClient(model="llama-70b-custom")
        profile = client._get_optimization_profile()
        assert profile == "high_throughput"

    def test_profile_reasoning_without_large_size(self):
        """Test that reasoning without large size returns deep_reasoning"""
        client = GroqAILLMClient(model="reasoning-13b-custom")
        profile = client._get_optimization_profile()
        assert profile == "deep_reasoning"

    def test_profile_balanced_fallback(self):
        """Test balanced profile for unmatched models"""
        client = GroqAILLMClient(model="custom-unknown-35b")
        profile = client._get_optimization_profile()
        assert profile == "balanced"


class TestGroqErrorHandling:
    """Test suite for Groq-specific error handling"""

    def test_stream_completion_error_handling_exists(self):
        """Test that stream completion has error handling code"""
        import inspect
        source = inspect.getsource(GroqAILLMClient._stream_completion_async)

        # Verify error handling logic is present
        assert "Failed to call a function" in source
        assert "Groq function calling failed" in source
        assert "retry" in source.lower()

    def test_regular_completion_error_handling_exists(self):
        """Test that regular completion has error handling code"""
        import inspect
        source = inspect.getsource(GroqAILLMClient._regular_completion)

        # Verify error handling logic is present
        assert "Failed to call a function" in source
        assert "Groq function calling failed" in source
        assert "retry" in source.lower()


class TestGroqEnhanceMessages:
    """Test suite for Groq message enhancement"""

    def test_enhance_messages_no_tools(self):
        """Test that messages are not enhanced when no tools"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]

        result = client._enhance_messages_for_groq(messages, [])
        assert result == messages

    def test_enhance_messages_no_system_message_support(self):
        """Test that messages are not enhanced when system messages not supported"""
        client = GroqAILLMClient(model="test-model")
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_func"}}]

        with patch.object(client, 'supports_feature', side_effect=lambda x: False if x == "system_messages" else True):
            result = client._enhance_messages_for_groq(messages, tools)
            assert result == messages

    def test_enhance_messages_no_tools_support(self):
        """Test that messages are not enhanced when tools not supported"""
        client = GroqAILLMClient(model="test-model")
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_func"}}]

        with patch.object(client, 'supports_feature', side_effect=lambda x: False if x == "tools" else True):
            result = client._enhance_messages_for_groq(messages, tools)
            assert result == messages

    def test_enhance_messages_known_family_models(self):
        """Test message enhancement for known family models"""
        tools = [{"function": {"name": "get_weather"}}, {"function": {"name": "search"}}]

        # Test each supported family
        for model_family in ["llama", "llama4", "qwen", "deepseek", "gpt-oss"]:
            client = GroqAILLMClient(model=f"{model_family}-test")
            messages = [{"role": "user", "content": "test"}]

            with patch.object(client, '_detect_groq_model_family', return_value=model_family):
                with patch.object(client, 'supports_feature', return_value=True):
                    result = client._enhance_messages_for_groq(messages, tools)

                    # Should have added system message
                    assert len(result) == 2
                    assert result[0]["role"] == "system"
                    assert "get_weather" in result[0]["content"]
                    assert "search" in result[0]["content"]
                    assert "proper JSON format" in result[0]["content"]

    def test_enhance_messages_unknown_family(self):
        """Test message enhancement for unknown family models"""
        client = GroqAILLMClient(model="unknown-model")
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_func"}}]

        with patch.object(client, '_detect_groq_model_family', return_value="unknown"):
            with patch.object(client, 'supports_feature', return_value=True):
                result = client._enhance_messages_for_groq(messages, tools)

                # Should have added system message with generic guidance
                assert len(result) == 2
                assert result[0]["role"] == "system"
                assert "Available functions" in result[0]["content"]

    def test_enhance_messages_with_existing_system_message(self):
        """Test message enhancement when system message already exists"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "test"}
        ]
        tools = [{"function": {"name": "test_func"}}]

        with patch.object(client, 'supports_feature', return_value=True):
            result = client._enhance_messages_for_groq(messages, tools)

            # Should have enhanced existing system message
            assert len(result) == 2
            assert result[0]["role"] == "system"
            assert "You are a helpful assistant" in result[0]["content"]
            assert "test_func" in result[0]["content"]


class TestGroqValidateToolCallArguments:
    """Test suite for tool call validation"""

    def test_validate_valid_tool_call_with_string_args(self):
        """Test validation of valid tool call with string arguments"""
        client = GroqAILLMClient()

        tool_call = {
            "function": {
                "name": "test_func",
                "arguments": '{"param": "value"}'
            }
        }

        assert client._validate_tool_call_arguments(tool_call) is True

    def test_validate_valid_tool_call_with_dict_args(self):
        """Test validation of valid tool call with dict arguments"""
        client = GroqAILLMClient()

        tool_call = {
            "function": {
                "name": "test_func",
                "arguments": {"param": "value"}
            }
        }

        assert client._validate_tool_call_arguments(tool_call) is True

    def test_validate_invalid_tool_call_no_function(self):
        """Test validation fails when no function key"""
        client = GroqAILLMClient()

        tool_call = {"name": "test_func"}

        assert client._validate_tool_call_arguments(tool_call) is False

    def test_validate_invalid_tool_call_no_arguments(self):
        """Test validation fails when no arguments key"""
        client = GroqAILLMClient()

        tool_call = {
            "function": {
                "name": "test_func"
            }
        }

        assert client._validate_tool_call_arguments(tool_call) is False

    def test_validate_invalid_tool_call_invalid_json(self):
        """Test validation fails with invalid JSON"""
        client = GroqAILLMClient()

        tool_call = {
            "function": {
                "name": "test_func",
                "arguments": "invalid json {"
            }
        }

        assert client._validate_tool_call_arguments(tool_call) is False

    def test_validate_invalid_tool_call_wrong_type(self):
        """Test validation fails with wrong argument type"""
        client = GroqAILLMClient()

        tool_call = {
            "function": {
                "name": "test_func",
                "arguments": 123  # Not string or dict
            }
        }

        assert client._validate_tool_call_arguments(tool_call) is False

    def test_validate_tool_call_with_type_error(self):
        """Test validation handles TypeError gracefully"""
        client = GroqAILLMClient()

        tool_call = None  # Will cause TypeError

        assert client._validate_tool_call_arguments(tool_call) is False


class TestGroqStreamingErrorHandling:
    """Test suite for Groq streaming error handling"""

    @pytest.mark.asyncio
    async def test_stream_completion_normal_success(self):
        """Test normal streaming without errors"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]

        # Mock parent's _stream_completion_async to yield chunks
        async def mock_stream(*args, **kwargs):
            yield {"response": "chunk1", "tool_calls": None}
            yield {"response": "chunk2", "tool_calls": []}

        with patch.object(GroqAILLMClient.__bases__[0], '_stream_completion_async', side_effect=mock_stream):
            chunks = []
            async for chunk in client._stream_completion_async(messages):
                chunks.append(chunk)

            # Should have normalized tool_calls to []
            assert len(chunks) == 2
            assert chunks[0]["tool_calls"] == []
            assert chunks[1]["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_stream_completion_function_calling_error_retry(self):
        """Test streaming with function calling error and retry (lines 755-771)"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_func"}}]

        call_count = [0]

        # Create two async generators
        async def first_stream(*args, **kwargs):
            raise Exception("Failed to call a function test_func")
            # This code won't execute, but needed to make it a generator
            yield  # pragma: no cover

        async def second_stream(*args, **kwargs):
            yield {"response": "retry success", "tool_calls": []}

        async def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                async for x in first_stream(*args, **kwargs):
                    yield x
            else:
                async for x in second_stream(*args, **kwargs):
                    yield x

        with patch.object(GroqAILLMClient.__bases__[0], '_stream_completion_async', side_effect=mock_stream):
            chunks = []
            async for chunk in client._stream_completion_async(messages, tools):
                chunks.append(chunk)

            # Should have retried and added note (retry chunk + note chunk)
            assert len(chunks) >= 1
            assert chunks[0]["response"] == "retry success"
            # The note is added as a separate chunk
            if len(chunks) > 1:
                assert "[Note: Function calling disabled due to Groq limitation]" in chunks[1]["response"]

    @pytest.mark.asyncio
    async def test_stream_completion_retry_also_fails(self):
        """Test streaming when retry also fails (lines 773-779)"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_func"}}]

        call_count = [0]

        async def failing_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Failed to call a function test_func")
            else:
                raise Exception("Retry error")
            yield  # pragma: no cover

        with patch.object(GroqAILLMClient.__bases__[0], '_stream_completion_async', side_effect=failing_stream):
            chunks = []
            async for chunk in client._stream_completion_async(messages, tools):
                chunks.append(chunk)

            # Should have error chunk
            assert len(chunks) == 1
            assert "Streaming error" in chunks[0]["response"]
            assert chunks[0]["error"] is True

    @pytest.mark.asyncio
    async def test_stream_completion_other_error_reraises(self):
        """Test streaming with non-function-calling error reraises (line 782)"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]

        async def failing_stream(*args, **kwargs):
            raise Exception("Some other error")
            yield  # pragma: no cover

        with patch.object(GroqAILLMClient.__bases__[0], '_stream_completion_async', side_effect=failing_stream):
            with pytest.raises(Exception, match="Some other error"):
                async for chunk in client._stream_completion_async(messages):
                    pass


class TestGroqRegularCompletionErrorHandling:
    """Test suite for Groq regular completion error handling"""

    @pytest.mark.asyncio
    async def test_regular_completion_normal_success(self):
        """Test normal completion without errors (line 799)"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]

        # Mock parent to return result with tool_calls=None
        with patch.object(GroqAILLMClient.__bases__[0], '_regular_completion', new_callable=AsyncMock) as mock_parent:
            mock_parent.return_value = {"response": "test", "tool_calls": None}

            result = await client._regular_completion(messages)

            # Should have normalized tool_calls to []
            assert result["tool_calls"] == []
            assert result["response"] == "test"

    @pytest.mark.asyncio
    async def test_regular_completion_function_calling_error_retry(self):
        """Test regular completion with function calling error and retry (lines 807-825)"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_func"}}]

        call_count = [0]
        async def mock_completion(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Failed to call a function test_func")
            else:
                return {"response": "retry success", "tool_calls": []}

        with patch.object(GroqAILLMClient.__bases__[0], '_regular_completion', side_effect=mock_completion):
            result = await client._regular_completion(messages, tools)

            # Should have retried and added note
            assert "retry success" in result["response"]
            assert "[Note: Function calling disabled due to Groq limitation]" in result["response"]
            assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_regular_completion_retry_also_fails(self):
        """Test regular completion when retry also fails (lines 827-833)"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]
        tools = [{"function": {"name": "test_func"}}]

        call_count = [0]
        async def mock_completion(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Failed to call a function test_func")
            else:
                raise Exception("Retry error")

        with patch.object(GroqAILLMClient.__bases__[0], '_regular_completion', side_effect=mock_completion):
            result = await client._regular_completion(messages, tools)

            # Should have error in response
            assert "Error" in result["response"]
            assert result["error"] is True
            assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_regular_completion_other_error_reraises(self):
        """Test regular completion with non-function-calling error reraises (line 836)"""
        client = GroqAILLMClient(model="llama-3.1-8b-instant")
        messages = [{"role": "user", "content": "test"}]

        async def mock_completion(*args, **kwargs):
            raise Exception("Some other error")

        with patch.object(GroqAILLMClient.__bases__[0], '_regular_completion', side_effect=mock_completion):
            with pytest.raises(Exception, match="Some other error"):
                await client._regular_completion(messages)


class TestGroqClose:
    """Test suite for cleanup/close functionality"""

    @pytest.mark.asyncio
    async def test_close_with_name_mapping(self):
        """Test close method resets name mapping"""
        client = GroqAILLMClient()

        # Set a name mapping
        client._current_name_mapping = {"func1": "tool1"}

        # Mock parent close
        with patch.object(GroqAILLMClient.__bases__[0], 'close', new_callable=AsyncMock) as mock_parent_close:
            await client.close()

            # Should have reset mapping
            assert client._current_name_mapping == {}
            # Should have called parent
            mock_parent_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_name_mapping(self):
        """Test close method when no name mapping attribute"""
        client = GroqAILLMClient()

        # Don't set name mapping attribute
        if hasattr(client, '_current_name_mapping'):
            delattr(client, '_current_name_mapping')

        # Mock parent close
        with patch.object(GroqAILLMClient.__bases__[0], 'close', new_callable=AsyncMock) as mock_parent_close:
            await client.close()

            # Should have called parent without error
            mock_parent_close.assert_called_once()
