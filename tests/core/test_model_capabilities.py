"""
Tests for chuk_llm.core.model_capabilities
==========================================

Test model capability registry and parameter support checks.
"""

import pytest

# Import the module to ensure coverage
import chuk_llm.core.model_capabilities as model_capabilities
from chuk_llm.core.enums import Provider
from chuk_llm.core.protocol import ModelInfo


# =============================================================================
# MODEL_CAPABILITIES Registry Tests
# =============================================================================

def test_model_capabilities_registry_exists():
    """Test that MODEL_CAPABILITIES registry is defined"""
    assert hasattr(model_capabilities, "MODEL_CAPABILITIES")
    assert isinstance(model_capabilities.MODEL_CAPABILITIES, dict)
    assert len(model_capabilities.MODEL_CAPABILITIES) > 0


def test_model_capabilities_have_required_fields():
    """Test that all models in registry have required fields"""
    for model_name, info in model_capabilities.MODEL_CAPABILITIES.items():
        assert isinstance(info, ModelInfo)
        assert hasattr(info, "provider")
        assert hasattr(info, "model")
        assert hasattr(info, "is_reasoning")
        assert hasattr(info, "supports_tools")
        assert hasattr(info, "supports_streaming")


def test_reasoning_models_registered():
    """Test that reasoning models are in the registry"""
    reasoning_models = ["gpt-5", "gpt-5-mini", "o1", "o1-preview", "o1-mini", "o3-mini"]

    for model in reasoning_models:
        assert model in model_capabilities.MODEL_CAPABILITIES
        info = model_capabilities.MODEL_CAPABILITIES[model]
        assert info.is_reasoning is True


def test_standard_models_registered():
    """Test that standard GPT-4 models are in the registry"""
    standard_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

    for model in standard_models:
        assert model in model_capabilities.MODEL_CAPABILITIES
        info = model_capabilities.MODEL_CAPABILITIES[model]
        assert info.is_reasoning is False


# =============================================================================
# get_model_capabilities() Tests
# =============================================================================

def test_get_model_capabilities_exact_match():
    """Test getting capabilities with exact model name"""
    info = model_capabilities.get_model_capabilities("gpt-5")
    assert info is not None
    assert isinstance(info, ModelInfo)
    assert info.model == "gpt-5"
    assert info.provider == Provider.OPENAI.value
    assert info.is_reasoning is True


def test_get_model_capabilities_gpt4o():
    """Test getting capabilities for GPT-4o"""
    info = model_capabilities.get_model_capabilities("gpt-4o")
    assert info is not None
    assert info.model == "gpt-4o"
    assert info.is_reasoning is False
    assert info.supports_tools is True
    assert info.supports_streaming is True
    assert info.supports_vision is True
    assert info.supports_temperature is True
    assert info.supports_top_p is True


def test_get_model_capabilities_o1():
    """Test getting capabilities for O1 models"""
    info = model_capabilities.get_model_capabilities("o1")
    assert info is not None
    assert info.model == "o1"
    assert info.is_reasoning is True
    assert info.supports_temperature is False
    assert info.supports_top_p is False


def test_get_model_capabilities_versioned_model():
    """Test getting capabilities for versioned model name"""
    # Should match "gpt-4o" base model
    info = model_capabilities.get_model_capabilities("gpt-4o-2024-05-13")
    assert info is not None
    assert info.model == "gpt-4o"


def test_get_model_capabilities_another_versioned():
    """Test another versioned model"""
    # Should match "gpt-5" base model
    info = model_capabilities.get_model_capabilities("gpt-5-preview-2024")
    assert info is not None
    assert info.model == "gpt-5"


def test_get_model_capabilities_unknown_model():
    """Test getting capabilities for unknown model"""
    info = model_capabilities.get_model_capabilities("unknown-model-xyz")
    assert info is None


def test_get_model_capabilities_empty_string():
    """Test getting capabilities with empty string"""
    info = model_capabilities.get_model_capabilities("")
    assert info is None


# =============================================================================
# model_supports_parameter() Tests
# =============================================================================

def test_model_supports_parameter_reasoning_model():
    """Test parameter support for reasoning models"""
    # GPT-5 does not support temperature
    assert model_capabilities.model_supports_parameter("gpt-5", "temperature") is False
    assert model_capabilities.model_supports_parameter("gpt-5", "top_p") is False
    assert model_capabilities.model_supports_parameter("gpt-5", "frequency_penalty") is False
    assert model_capabilities.model_supports_parameter("gpt-5", "presence_penalty") is False

    # But supports tools and streaming
    assert model_capabilities.model_supports_parameter("gpt-5", "tools") is True
    assert model_capabilities.model_supports_parameter("gpt-5", "stream") is True


def test_model_supports_parameter_standard_model():
    """Test parameter support for standard models"""
    # GPT-4o supports all standard parameters
    assert model_capabilities.model_supports_parameter("gpt-4o", "temperature") is True
    assert model_capabilities.model_supports_parameter("gpt-4o", "top_p") is True
    assert model_capabilities.model_supports_parameter("gpt-4o", "max_tokens") is True
    assert model_capabilities.model_supports_parameter("gpt-4o", "frequency_penalty") is True
    assert model_capabilities.model_supports_parameter("gpt-4o", "presence_penalty") is True
    assert model_capabilities.model_supports_parameter("gpt-4o", "logit_bias") is True
    assert model_capabilities.model_supports_parameter("gpt-4o", "logprobs") is True


def test_model_supports_parameter_o1_models():
    """Test parameter support for O1 series"""
    for model in ["o1", "o1-preview", "o1-mini", "o3-mini"]:
        assert model_capabilities.model_supports_parameter(model, "temperature") is False
        assert model_capabilities.model_supports_parameter(model, "top_p") is False
        assert model_capabilities.model_supports_parameter(model, "frequency_penalty") is False


def test_model_supports_parameter_max_completion_tokens():
    """Test max_completion_tokens parameter mapping"""
    # Should map to supports_max_tokens
    assert model_capabilities.model_supports_parameter("gpt-4o", "max_completion_tokens") is True


def test_model_supports_parameter_unknown_model():
    """Test parameter support for unknown model (should default to True)"""
    # Unknown models are assumed to support all parameters (safe default)
    assert model_capabilities.model_supports_parameter("unknown-model", "temperature") is True
    assert model_capabilities.model_supports_parameter("unknown-model", "top_p") is True
    assert model_capabilities.model_supports_parameter("unknown-model", "any_param") is True


def test_model_supports_parameter_unknown_parameter():
    """Test checking unknown parameter (should default to True)"""
    # Unknown parameters are assumed to be supported
    assert model_capabilities.model_supports_parameter("gpt-4o", "unknown_param") is True
    assert model_capabilities.model_supports_parameter("gpt-5", "unknown_param") is True


def test_model_supports_parameter_empty_values():
    """Test edge cases with empty values"""
    assert model_capabilities.model_supports_parameter("", "temperature") is True
    assert model_capabilities.model_supports_parameter("gpt-4o", "") is True


# =============================================================================
# Integration Tests
# =============================================================================

def test_all_reasoning_models_have_restricted_params():
    """Test that all reasoning models have parameter restrictions"""
    reasoning_models = [
        name for name, info in model_capabilities.MODEL_CAPABILITIES.items()
        if info.is_reasoning
    ]

    for model in reasoning_models:
        # All reasoning models should NOT support these parameters
        assert model_capabilities.model_supports_parameter(model, "temperature") is False
        assert model_capabilities.model_supports_parameter(model, "top_p") is False


def test_all_standard_models_have_full_support():
    """Test that standard models have full parameter support"""
    standard_models = [
        name for name, info in model_capabilities.MODEL_CAPABILITIES.items()
        if not info.is_reasoning
    ]

    for model in standard_models:
        # All standard models should support these parameters
        assert model_capabilities.model_supports_parameter(model, "temperature") is True
        assert model_capabilities.model_supports_parameter(model, "top_p") is True
        assert model_capabilities.model_supports_parameter(model, "max_tokens") is True


def test_all_models_have_openai_provider():
    """Test that all registered models are OpenAI models"""
    for model_name, info in model_capabilities.MODEL_CAPABILITIES.items():
        assert info.provider == Provider.OPENAI.value


# =============================================================================
# Specific Model Configuration Tests
# =============================================================================

def test_gpt5_configuration():
    """Test GPT-5 specific configuration"""
    info = model_capabilities.get_model_capabilities("gpt-5")
    assert info is not None
    assert info.is_reasoning is True
    assert info.supports_tools is True
    assert info.supports_streaming is True
    assert info.supports_vision is True
    assert info.supports_temperature is False
    assert info.supports_top_p is False
    assert info.supports_frequency_penalty is False
    assert info.supports_presence_penalty is False
    assert info.supports_logit_bias is False
    assert info.supports_logprobs is False


def test_gpt5_mini_configuration():
    """Test GPT-5-mini specific configuration"""
    info = model_capabilities.get_model_capabilities("gpt-5-mini")
    assert info is not None
    assert info.is_reasoning is True
    assert info.supports_tools is True
    assert info.supports_vision is True
    assert info.supports_temperature is False


def test_o1_configuration():
    """Test O1 specific configuration"""
    info = model_capabilities.get_model_capabilities("o1")
    assert info is not None
    assert info.is_reasoning is True
    assert info.supports_tools is False  # O1 doesn't support tools
    assert info.supports_vision is False  # O1 doesn't support vision
    assert info.supports_temperature is False


def test_gpt4o_mini_configuration():
    """Test GPT-4o-mini specific configuration"""
    info = model_capabilities.get_model_capabilities("gpt-4o-mini")
    assert info is not None
    assert info.is_reasoning is False
    assert info.supports_tools is True
    assert info.supports_streaming is True
    assert info.supports_vision is True
    assert info.supports_temperature is True
    assert info.supports_top_p is True
    assert info.supports_frequency_penalty is True
    assert info.supports_presence_penalty is True
