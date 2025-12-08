"""
Model Capabilities Registry
============================

Defines parameter support for different models.
Reasoning models (GPT-5, O-series) have restricted parameters.
"""

from .enums import Provider
from .protocol import ModelInfo

# Model capability definitions
MODEL_CAPABILITIES: dict[str, ModelInfo] = {
    # GPT-5 Series - Reasoning models with restricted parameters
    "gpt-5": ModelInfo(
        provider=Provider.OPENAI.value,
        model="gpt-5",
        is_reasoning=True,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        supports_system_messages=True,  # GPT-5 supports system messages (unlike o1)
        # GPT-5 does not support these parameters
        supports_temperature=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        supports_logit_bias=False,
        supports_logprobs=False,
    ),
    "gpt-5-mini": ModelInfo(
        provider=Provider.OPENAI.value,
        model="gpt-5-mini",
        is_reasoning=True,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        supports_system_messages=True,  # GPT-5 supports system messages (unlike o1)
        # GPT-5-mini does not support these parameters
        supports_temperature=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        supports_logit_bias=False,
        supports_logprobs=False,
    ),
    # O-series - Reasoning models
    "o1": ModelInfo(
        provider=Provider.OPENAI.value,
        model="o1",
        is_reasoning=True,
        supports_tools=False,
        supports_streaming=True,
        supports_vision=False,
        supports_system_messages=False,  # o1 does NOT support system messages
        supports_temperature=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        supports_logit_bias=False,
        supports_logprobs=False,
    ),
    "o1-preview": ModelInfo(
        provider=Provider.OPENAI.value,
        model="o1-preview",
        is_reasoning=True,
        supports_tools=False,
        supports_streaming=True,
        supports_vision=False,
        supports_system_messages=False,  # o1 does NOT support system messages
        supports_temperature=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        supports_logit_bias=False,
        supports_logprobs=False,
    ),
    "o1-mini": ModelInfo(
        provider=Provider.OPENAI.value,
        model="o1-mini",
        is_reasoning=True,
        supports_tools=False,
        supports_streaming=True,
        supports_vision=False,
        supports_system_messages=False,  # o1 does NOT support system messages
        supports_temperature=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        supports_logit_bias=False,
        supports_logprobs=False,
    ),
    "o3-mini": ModelInfo(
        provider=Provider.OPENAI.value,
        model="o3-mini",
        is_reasoning=True,
        supports_tools=False,
        supports_streaming=True,
        supports_vision=False,
        supports_system_messages=True,  # o3+ supports system messages (unlike o1)
        supports_temperature=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        supports_logit_bias=False,
        supports_logprobs=False,
    ),
    # GPT-4 Series - Full support
    "gpt-4o": ModelInfo(
        provider=Provider.OPENAI.value,
        model="gpt-4o",
        is_reasoning=False,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_tokens=True,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
        supports_logit_bias=True,
        supports_logprobs=True,
    ),
    "gpt-4o-mini": ModelInfo(
        provider=Provider.OPENAI.value,
        model="gpt-4o-mini",
        is_reasoning=False,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_tokens=True,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
        supports_logit_bias=True,
        supports_logprobs=True,
    ),
    "gpt-4-turbo": ModelInfo(
        provider=Provider.OPENAI.value,
        model="gpt-4-turbo",
        is_reasoning=False,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_tokens=True,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
        supports_logit_bias=True,
        supports_logprobs=True,
    ),
}


def get_model_capabilities(model: str) -> ModelInfo | None:
    """
    Get model capabilities by model name.

    Args:
        model: Model name (e.g., "gpt-5", "gpt-4o-mini")

    Returns:
        ModelInfo with capabilities, or None if model not found
    """
    # Try exact match first
    if model in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model]

    # Try pattern matching for versioned models
    # e.g., "gpt-4o-2024-05-13" -> "gpt-4o"
    base_model = model.split("-")[0:2]  # "gpt-4o"
    if len(base_model) >= 2:
        base_key = "-".join(base_model)
        if base_key in MODEL_CAPABILITIES:
            return MODEL_CAPABILITIES[base_key]

    # Unknown model - return default (full support)
    return None


def model_supports_parameter(model: str, parameter: str) -> bool:
    """
    Check if a model supports a specific parameter.

    Args:
        model: Model name
        parameter: Parameter name (e.g., "temperature", "top_p")

    Returns:
        True if supported or unknown, False if known to be unsupported
    """
    capabilities = get_model_capabilities(model)
    if capabilities is None:
        # Unknown model - assume it supports everything (safe default)
        return True

    # Map parameter names to capability flags
    param_map = {
        "temperature": "supports_temperature",
        "top_p": "supports_top_p",
        "max_tokens": "supports_max_tokens",
        "max_completion_tokens": "supports_max_tokens",
        "frequency_penalty": "supports_frequency_penalty",
        "presence_penalty": "supports_presence_penalty",
        "logit_bias": "supports_logit_bias",
        "logprobs": "supports_logprobs",
    }

    capability_name = param_map.get(parameter)
    if capability_name is None:
        # Unknown parameter - assume supported
        return True

    return getattr(capabilities, capability_name, True)
