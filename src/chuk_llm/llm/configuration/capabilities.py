# chuk_llm/llm/configuration/capabilities.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum

class Feature(Enum):
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    PARALLEL_CALLS = "parallel_calls"
    SYSTEM_MESSAGES = "system_messages"
    MULTIMODAL = "multimodal"

@dataclass
class ProviderCapabilities:
    name: str
    features: Set[Feature]
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    rate_limits: Optional[Dict[str, int]] = None  # requests per minute
    supported_models: Optional[List[str]] = None
    
    def supports(self, feature: Feature) -> bool:
        return feature in self.features
    
    def get_rate_limit(self, tier: str = "default") -> Optional[int]:
        if self.rate_limits:
            return self.rate_limits.get(tier)
        return None

# Registry of provider capabilities
PROVIDER_CAPABILITIES = {
    "openai": ProviderCapabilities(
        name="OpenAI",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.JSON_MODE, Feature.PARALLEL_CALLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=128000,
        max_output_tokens=4096,
        rate_limits={"default": 3500, "tier_1": 500},
        supported_models=[
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
        ]
    ),
    "anthropic": ProviderCapabilities(
        name="Anthropic",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.PARALLEL_CALLS, Feature.SYSTEM_MESSAGES
        },
        max_context_length=200000,
        max_output_tokens=4096,
        rate_limits={"default": 4000},
        supported_models=[
            "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"
        ]
    ),
    "groq": ProviderCapabilities(
        name="Groq",
        features={Feature.STREAMING, Feature.TOOLS, Feature.PARALLEL_CALLS},
        max_context_length=32768,
        max_output_tokens=8192,
        rate_limits={"default": 30},  # Very limited
        supported_models=[
            "llama-3.3-70b-versatile", "mixtral-8x7b-32768"
        ]
    ),
    "gemini": ProviderCapabilities(
        name="Google Gemini",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION,
            Feature.JSON_MODE, Feature.SYSTEM_MESSAGES
        },
        max_context_length=1000000,
        max_output_tokens=8192,
        rate_limits={"default": 1500},
        supported_models=["gemini-2.0-flash", "gemini-1.5-pro"]
    ),
    "ollama": ProviderCapabilities(
        name="Ollama",
        features={Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES},
        max_context_length=None,  # Depends on model
        max_output_tokens=None,   # Depends on model
        rate_limits=None,         # Local, no limits
        supported_models=None     # Dynamic based on installation
    )
}

class CapabilityChecker:
    """Utility for checking provider capabilities"""
    
    @staticmethod
    def can_handle_request(
        provider: str, 
        has_tools: bool = False,
        has_vision: bool = False,
        needs_streaming: bool = False,
        needs_json: bool = False
    ) -> tuple[bool, List[str]]:
        """Check if provider can handle the request"""
        if provider not in PROVIDER_CAPABILITIES:
            return False, [f"Unknown provider: {provider}"]
        
        caps = PROVIDER_CAPABILITIES[provider]
        issues = []
        
        if has_tools and not caps.supports(Feature.TOOLS):
            issues.append(f"{provider} doesn't support tools")
        
        if has_vision and not caps.supports(Feature.VISION):
            issues.append(f"{provider} doesn't support vision")
        
        if needs_streaming and not caps.supports(Feature.STREAMING):
            issues.append(f"{provider} doesn't support streaming")
        
        if needs_json and not caps.supports(Feature.JSON_MODE):
            issues.append(f"{provider} doesn't support JSON mode")
        
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