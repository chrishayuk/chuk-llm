# chuk_llm/llm/configuration/capabilities.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum
import re

class Feature(Enum):
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    PARALLEL_CALLS = "parallel_calls"
    SYSTEM_MESSAGES = "system_messages"
    MULTIMODAL = "multimodal"

@dataclass 
class ModelCapabilities:
    """Capabilities for a specific model or model pattern"""
    pattern: str  # Regex pattern to match model names
    features: Set[Feature]
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    
    def matches(self, model_name: str) -> bool:
        return bool(re.match(self.pattern, model_name, re.IGNORECASE))

@dataclass
class ProviderCapabilities:
    name: str
    features: Set[Feature]  # Default features for all models
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    rate_limits: Optional[Dict[str, int]] = None
    model_capabilities: Optional[List[ModelCapabilities]] = None  # Model-specific overrides
    
    def supports(self, feature: Feature) -> bool:
        return feature in self.features
    
    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific model"""
        if self.model_capabilities:
            for model_cap in self.model_capabilities:
                if model_cap.matches(model_name):
                    return model_cap
        
        # Return default capabilities
        return ModelCapabilities(
            pattern=".*",
            features=self.features,
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens
        )
    
    def get_rate_limit(self, tier: str = "default") -> Optional[int]:
        if self.rate_limits:
            return self.rate_limits.get(tier)
        return None

# Pattern-based model capabilities for Mistral
MISTRAL_MODEL_CAPABILITIES = [
    # Codestral models - high context, coding focus
    ModelCapabilities(
        pattern=r".*codestral.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=256000,
        max_output_tokens=8192
    ),
    
    # Pixtral models - vision capabilities
    ModelCapabilities(
        pattern=r".*pixtral.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.SYSTEM_MESSAGES, Feature.MULTIMODAL
        },
        max_context_length=128000,
        max_output_tokens=8192
    ),
    
    # Ministral models - edge/efficient models
    ModelCapabilities(
        pattern=r".*ministral.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=128000,
        max_output_tokens=4096
    ),
    
    # Mistral Small with vision
    ModelCapabilities(
        pattern=r"mistral-small.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION,
            Feature.SYSTEM_MESSAGES, Feature.MULTIMODAL
        },
        max_context_length=128000,
        max_output_tokens=8192
    ),
    
    # Mistral Medium with vision
    ModelCapabilities(
        pattern=r"mistral-medium.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION,
            Feature.SYSTEM_MESSAGES, Feature.MULTIMODAL
        },
        max_context_length=128000,
        max_output_tokens=8192
    ),
    
    # Saba models - multilingual
    ModelCapabilities(
        pattern=r".*saba.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=32000,
        max_output_tokens=4096
    ),
    
    # OCR models - document processing
    ModelCapabilities(
        pattern=r".*ocr.*",
        features={
            Feature.STREAMING, Feature.VISION, Feature.SYSTEM_MESSAGES
        },
        max_context_length=32000,
        max_output_tokens=8192
    ),
    
    # Moderation models
    ModelCapabilities(
        pattern=r".*moderation.*",
        features={
            Feature.STREAMING, Feature.SYSTEM_MESSAGES
        },
        max_context_length=8000,
        max_output_tokens=1000
    ),
    
    # Embed models
    ModelCapabilities(
        pattern=r".*embed.*",
        features=set(),  # Embedding models don't use chat features
        max_context_length=8000,
        max_output_tokens=None
    ),
    
    # Devstral models - development focused
    ModelCapabilities(
        pattern=r".*devstral.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=128000,
        max_output_tokens=8192
    ),
    
    # Nemo models - multilingual
    ModelCapabilities(
        pattern=r".*nemo.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=128000,
        max_output_tokens=8192
    ),
    
    # Mamba models - architecture variant
    ModelCapabilities(
        pattern=r".*mamba.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=256000,
        max_output_tokens=8192
    ),
    
    # Math models - specialized
    ModelCapabilities(
        pattern=r".*math.*",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=32000,
        max_output_tokens=4096
    )
]

# Registry of provider capabilities - now with pattern-based model support
PROVIDER_CAPABILITIES = {
    "openai": ProviderCapabilities(
        name="OpenAI",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.JSON_MODE, Feature.PARALLEL_CALLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=128000,
        max_output_tokens=4096,
        rate_limits={"default": 3500, "tier_1": 500}
    ),
    "anthropic": ProviderCapabilities(
        name="Anthropic",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.PARALLEL_CALLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=200000,
        max_output_tokens=4096,
        rate_limits={"default": 4000}
    ),
    "groq": ProviderCapabilities(
        name="Groq",
        features={Feature.STREAMING, Feature.TOOLS, Feature.PARALLEL_CALLS},
        max_context_length=32768,
        max_output_tokens=8192,
        rate_limits={"default": 30}
    ),
    "gemini": ProviderCapabilities(
        name="Google Gemini",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION,
            Feature.JSON_MODE, Feature.SYSTEM_MESSAGES
        },
        max_context_length=1000000,
        max_output_tokens=8192,
        rate_limits={"default": 1500}
    ),
    "ollama": ProviderCapabilities(
        name="Ollama",
        features={Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES},
        max_context_length=None,
        max_output_tokens=None,
        rate_limits=None
    ),
    # NEW: Mistral Le Plateforme capabilities - pattern-based and scalable
    "mistral": ProviderCapabilities(
        name="Mistral Le Plateforme",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION,
            Feature.SYSTEM_MESSAGES, Feature.PARALLEL_CALLS
        },
        max_context_length=128000,  # Default for most models
        max_output_tokens=8192,
        rate_limits={"default": 1000, "premium": 5000},
        model_capabilities=MISTRAL_MODEL_CAPABILITIES  # Use pattern-based matching
    )
}

class CapabilityChecker:
    """Utility for checking provider capabilities"""
    
    @staticmethod
    def can_handle_request(
        provider: str, 
        model: str = None,
        has_tools: bool = False,
        has_vision: bool = False,
        needs_streaming: bool = False,
        needs_json: bool = False
    ) -> tuple[bool, List[str]]:
        """Check if provider can handle the request"""
        if provider not in PROVIDER_CAPABILITIES:
            return False, [f"Unknown provider: {provider}"]
        
        caps = PROVIDER_CAPABILITIES[provider]
        
        # Get model-specific capabilities if model is provided
        if model:
            model_caps = caps.get_model_capabilities(model)
            features = model_caps.features
        else:
            features = caps.features
        
        issues = []
        
        if has_tools and Feature.TOOLS not in features:
            issues.append(f"{provider} model {model or 'default'} doesn't support tools")
        
        if has_vision and Feature.VISION not in features:
            issues.append(f"{provider} model {model or 'default'} doesn't support vision")
        
        if needs_streaming and Feature.STREAMING not in features:
            issues.append(f"{provider} model {model or 'default'} doesn't support streaming")
        
        if needs_json and Feature.JSON_MODE not in features:
            issues.append(f"{provider} model {model or 'default'} doesn't support JSON mode")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def get_best_provider(
        requirements: Set[Feature],
        exclude: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Find the best provider for given requirements"""
        exclude = exclude or set()
        
        candidates = []
        for provider, caps in PROVIDER_CAPABILITIES.items():
            if provider in exclude:
                continue
            
            if requirements.issubset(caps.features):
                # Score based on rate limits (higher is better)
                rate_limit = caps.get_rate_limit() or 0
                candidates.append((provider, rate_limit))
        
        if candidates:
            # Return provider with highest rate limit
            return max(candidates, key=lambda x: x[1])[0]
        
        return None
    
    @staticmethod
    def get_model_info(provider: str, model: str) -> Dict[str, any]:
        """Get detailed information about a specific model"""
        if provider not in PROVIDER_CAPABILITIES:
            return {"error": f"Unknown provider: {provider}"}
        
        caps = PROVIDER_CAPABILITIES[provider]
        model_caps = caps.get_model_capabilities(model)
        
        return {
            "provider": provider,
            "model": model,
            "features": [f.value for f in model_caps.features],
            "max_context_length": model_caps.max_context_length,
            "max_output_tokens": model_caps.max_output_tokens,
            "supports_streaming": Feature.STREAMING in model_caps.features,
            "supports_tools": Feature.TOOLS in model_caps.features,
            "supports_vision": Feature.VISION in model_caps.features,
            "supports_json_mode": Feature.JSON_MODE in model_caps.features,
        }