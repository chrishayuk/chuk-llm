# tests/core/test_capabilities.py
"""
Unit tests for provider capabilities registry and checking
"""
import pytest
from typing import Set, Optional, List
from dataclasses import dataclass
from enum import Enum

# Note: These imports will need to be adjusted based on your actual structure
# For now, creating mock implementations since the capability classes don't exist yet

class Feature(Enum):
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    JSON_MODE = "json_mode"
    PARALLEL_CALLS = "parallel_calls"
    SYSTEM_MESSAGES = "system_messages"
    MULTIMODAL = "multimodal"
    STRUCTURED_OUTPUT = "structured_output"
    FUNCTION_CALLING = "function_calling"


@dataclass
class ProviderCapabilities:
    name: str
    features: Set[Feature]
    max_context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    rate_limits: Optional[dict] = None  # requests per minute
    supported_models: Optional[List[str]] = None
    
    def supports(self, feature: Feature) -> bool:
        return feature in self.features
    
    def get_rate_limit(self, tier: str = "default") -> Optional[int]:
        if self.rate_limits:
            return self.rate_limits.get(tier)
        return None


# Mock registry of provider capabilities
PROVIDER_CAPABILITIES = {
    "openai": ProviderCapabilities(
        name="OpenAI",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.JSON_MODE, Feature.PARALLEL_CALLS, Feature.SYSTEM_MESSAGES,
            Feature.FUNCTION_CALLING, Feature.STRUCTURED_OUTPUT
        },
        max_context_length=128000,
        max_output_tokens=4096,
        rate_limits={"default": 3500, "tier_1": 500, "tier_2": 10000},
        supported_models=[
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
        ]
    ),
    "anthropic": ProviderCapabilities(
        name="Anthropic",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.PARALLEL_CALLS, Feature.SYSTEM_MESSAGES,
            Feature.FUNCTION_CALLING
        },
        max_context_length=200000,
        max_output_tokens=4096,
        rate_limits={"default": 4000, "enterprise": 20000},
        supported_models=[
            "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        ]
    ),
    "groq": ProviderCapabilities(
        name="Groq",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.PARALLEL_CALLS,
            Feature.FUNCTION_CALLING, Feature.JSON_MODE
        },
        max_context_length=32768,
        max_output_tokens=8192,
        rate_limits={"default": 30, "pro": 6000},  # Very limited for free tier
        supported_models=[
            "llama-3.3-70b-versatile", "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
    ),
    "gemini": ProviderCapabilities(
        name="Google Gemini",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION,
            Feature.JSON_MODE, Feature.SYSTEM_MESSAGES, Feature.MULTIMODAL,
            Feature.FUNCTION_CALLING
        },
        max_context_length=1000000,
        max_output_tokens=8192,
        rate_limits={"default": 1500, "paid": 10000},
        supported_models=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
    ),
    "ollama": ProviderCapabilities(
        name="Ollama",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.SYSTEM_MESSAGES,
            Feature.FUNCTION_CALLING
        },
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
        needs_json: bool = False,
        needs_system_messages: bool = False,
        estimated_tokens: Optional[int] = None
    ) -> tuple[bool, List[str]]:
        """Check if provider can handle the request"""
        if provider not in PROVIDER_CAPABILITIES:
            return False, [f"Unknown provider: {provider}"]
        
        caps = PROVIDER_CAPABILITIES[provider]
        issues = []
        
        if has_tools and not caps.supports(Feature.TOOLS):
            issues.append(f"{provider} doesn't support tools/function calling")
        
        if has_vision and not caps.supports(Feature.VISION):
            issues.append(f"{provider} doesn't support vision/image inputs")
        
        if needs_streaming and not caps.supports(Feature.STREAMING):
            issues.append(f"{provider} doesn't support streaming")
        
        if needs_json and not caps.supports(Feature.JSON_MODE):
            issues.append(f"{provider} doesn't support JSON mode")
        
        if needs_system_messages and not caps.supports(Feature.SYSTEM_MESSAGES):
            issues.append(f"{provider} doesn't support system messages")
        
        # Check context length limits
        if estimated_tokens and caps.max_context_length:
            if estimated_tokens > caps.max_context_length:
                issues.append(
                    f"Estimated tokens ({estimated_tokens}) exceeds {provider} "
                    f"context limit ({caps.max_context_length})"
                )
        
        return len(issues) == 0, issues
    
    @staticmethod
    def get_best_provider(
        requirements: Set[Feature],
        exclude: Optional[Set[str]] = None,
        prefer_higher_limits: bool = True
    ) -> Optional[str]:
        """Find the best provider for given requirements"""
        exclude = exclude or set()
        
        candidates = []
        for provider, caps in PROVIDER_CAPABILITIES.items():
            if provider in exclude:
                continue
            
            if requirements.issubset(caps.features):
                # Score based on various factors
                score = 0
                
                # Higher rate limits are better
                rate_limit = caps.get_rate_limit() or 0
                score += rate_limit / 1000  # Normalize
                
                # Higher context length is better
                if caps.max_context_length:
                    score += caps.max_context_length / 100000  # Normalize
                
                # More features are better
                score += len(caps.features) * 0.1
                
                candidates.append((provider, score))
        
        if candidates:
            # Return provider with highest score
            return max(candidates, key=lambda x: x[1])[0]
        
        return None
    
    @staticmethod
    def get_provider_info(provider: str) -> Optional[ProviderCapabilities]:
        """Get detailed information about a provider"""
        return PROVIDER_CAPABILITIES.get(provider)
    
    @staticmethod
    def list_providers_with_feature(feature: Feature) -> List[str]:
        """List all providers that support a specific feature"""
        return [
            provider for provider, caps in PROVIDER_CAPABILITIES.items()
            if caps.supports(feature)
        ]
    
    @staticmethod
    def compare_providers(providers: List[str]) -> dict:
        """Compare capabilities of multiple providers"""
        comparison = {}
        
        for provider in providers:
            caps = PROVIDER_CAPABILITIES.get(provider)
            if caps:
                comparison[provider] = {
                    "features": [f.value for f in caps.features],
                    "max_context_length": caps.max_context_length,
                    "max_output_tokens": caps.max_output_tokens,
                    "rate_limits": caps.rate_limits,
                    "supported_models": caps.supported_models
                }
            else:
                comparison[provider] = {"error": "Provider not found"}
        
        return comparison


class TestProviderCapabilities:
    """Test suite for ProviderCapabilities dataclass"""
    
    def test_provider_capabilities_creation(self):
        """Test creating ProviderCapabilities instance"""
        caps = ProviderCapabilities(
            name="Test Provider",
            features={Feature.STREAMING, Feature.TOOLS},
            max_context_length=50000,
            rate_limits={"default": 1000}
        )
        
        assert caps.name == "Test Provider"
        assert Feature.STREAMING in caps.features
        assert Feature.TOOLS in caps.features
        assert caps.max_context_length == 50000
        assert caps.get_rate_limit() == 1000
    
    def test_supports_feature(self):
        """Test supports() method"""
        caps = ProviderCapabilities(
            name="Test",
            features={Feature.STREAMING, Feature.VISION}
        )
        
        assert caps.supports(Feature.STREAMING) is True
        assert caps.supports(Feature.VISION) is True
        assert caps.supports(Feature.TOOLS) is False
        assert caps.supports(Feature.JSON_MODE) is False
    
    def test_get_rate_limit_with_tiers(self):
        """Test rate limit retrieval with different tiers"""
        caps = ProviderCapabilities(
            name="Test",
            features={Feature.STREAMING},
            rate_limits={"default": 100, "premium": 1000, "enterprise": 10000}
        )
        
        assert caps.get_rate_limit() == 100  # Default
        assert caps.get_rate_limit("default") == 100
        assert caps.get_rate_limit("premium") == 1000
        assert caps.get_rate_limit("enterprise") == 10000
        assert caps.get_rate_limit("nonexistent") is None
    
    def test_get_rate_limit_no_limits(self):
        """Test rate limit when no limits are set"""
        caps = ProviderCapabilities(
            name="Test",
            features={Feature.STREAMING},
            rate_limits=None
        )
        
        assert caps.get_rate_limit() is None
        assert caps.get_rate_limit("any_tier") is None
    
    def test_provider_capabilities_defaults(self):
        """Test default values for optional fields"""
        caps = ProviderCapabilities(
            name="Minimal",
            features={Feature.STREAMING}
        )
        
        assert caps.max_context_length is None
        assert caps.max_output_tokens is None
        assert caps.rate_limits is None
        assert caps.supported_models is None


class TestProviderCapabilitiesRegistry:
    """Test suite for the provider capabilities registry"""
    
    def test_all_providers_exist(self):
        """Test that all expected providers are in registry"""
        expected_providers = ["openai", "anthropic", "groq", "gemini", "ollama"]
        
        for provider in expected_providers:
            assert provider in PROVIDER_CAPABILITIES, f"Provider {provider} not found"
    
    def test_openai_capabilities(self):
        """Test OpenAI provider capabilities"""
        caps = PROVIDER_CAPABILITIES["openai"]
        
        assert caps.name == "OpenAI"
        assert caps.supports(Feature.STREAMING)
        assert caps.supports(Feature.TOOLS)
        assert caps.supports(Feature.VISION)
        assert caps.supports(Feature.JSON_MODE)
        assert caps.max_context_length == 128000
        assert caps.max_output_tokens == 4096
        assert "gpt-4o" in caps.supported_models
    
    def test_anthropic_capabilities(self):
        """Test Anthropic provider capabilities"""
        caps = PROVIDER_CAPABILITIES["anthropic"]
        
        assert caps.name == "Anthropic"
        assert caps.supports(Feature.STREAMING)
        assert caps.supports(Feature.TOOLS)
        assert caps.supports(Feature.VISION)
        assert not caps.supports(Feature.JSON_MODE)  # Anthropic doesn't have native JSON mode
        assert caps.max_context_length == 200000
        assert "claude-3-5-sonnet-20241022" in caps.supported_models
    
    def test_groq_capabilities(self):
        """Test Groq provider capabilities"""
        caps = PROVIDER_CAPABILITIES["groq"]
        
        assert caps.name == "Groq"
        assert caps.supports(Feature.STREAMING)
        assert caps.supports(Feature.TOOLS)
        assert not caps.supports(Feature.VISION)  # Groq doesn't support vision
        assert caps.max_context_length == 32768
        assert caps.get_rate_limit() == 30  # Very limited free tier
        assert caps.get_rate_limit("pro") == 6000
    
    def test_gemini_capabilities(self):
        """Test Gemini provider capabilities"""
        caps = PROVIDER_CAPABILITIES["gemini"]
        
        assert caps.name == "Google Gemini"
        assert caps.supports(Feature.STREAMING)
        assert caps.supports(Feature.VISION)
        assert caps.supports(Feature.MULTIMODAL)
        assert caps.max_context_length == 1000000  # Very large context
        assert "gemini-2.0-flash" in caps.supported_models
    
    def test_ollama_capabilities(self):
        """Test Ollama provider capabilities"""
        caps = PROVIDER_CAPABILITIES["ollama"]
        
        assert caps.name == "Ollama"
        assert caps.supports(Feature.STREAMING)
        assert caps.supports(Feature.TOOLS)
        assert not caps.supports(Feature.VISION)  # Most Ollama models don't support vision
        assert caps.max_context_length is None  # Depends on model
        assert caps.rate_limits is None  # No limits for local
        assert caps.supported_models is None  # Dynamic


class TestCapabilityChecker:
    """Test suite for CapabilityChecker utility class"""
    
    def test_can_handle_simple_request(self):
        """Test capability checking for simple text request"""
        can_handle, issues = CapabilityChecker.can_handle_request("openai")
        
        assert can_handle is True
        assert len(issues) == 0
    
    def test_can_handle_request_with_tools(self):
        """Test capability checking for request with tools"""
        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai", 
            has_tools=True
        )
        
        assert can_handle is True
        assert len(issues) == 0
        
        # Test provider without tools
        can_handle, issues = CapabilityChecker.can_handle_request(
            "ollama",  # Assuming ollama supports tools in our mock
            has_tools=True
        )
        
        assert can_handle is True  # Ollama supports tools in our mock
    
    def test_can_handle_request_with_vision(self):
        """Test capability checking for vision requests"""
        # Provider with vision support
        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai",
            has_vision=True
        )
        
        assert can_handle is True
        assert len(issues) == 0
        
        # Provider without vision support
        can_handle, issues = CapabilityChecker.can_handle_request(
            "groq",
            has_vision=True
        )
        
        assert can_handle is False
        assert len(issues) == 1
        assert "doesn't support vision" in issues[0]
    
    def test_can_handle_request_with_streaming(self):
        """Test capability checking for streaming requests"""
        can_handle, issues = CapabilityChecker.can_handle_request(
            "anthropic",
            needs_streaming=True
        )
        
        assert can_handle is True
        assert len(issues) == 0
    
    def test_can_handle_request_with_json_mode(self):
        """Test capability checking for JSON mode requests"""
        # Provider with JSON mode
        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai",
            needs_json=True
        )
        
        assert can_handle is True
        assert len(issues) == 0
        
        # Provider without JSON mode
        can_handle, issues = CapabilityChecker.can_handle_request(
            "anthropic",
            needs_json=True
        )
        
        assert can_handle is False
        assert "doesn't support JSON mode" in issues[0]
    
    def test_can_handle_request_context_length_check(self):
        """Test context length validation"""
        # Within limits
        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai",
            estimated_tokens=50000  # Well within 128k limit
        )
        
        assert can_handle is True
        assert len(issues) == 0
        
        # Exceeding limits
        can_handle, issues = CapabilityChecker.can_handle_request(
            "groq",
            estimated_tokens=50000  # Exceeds 32k limit
        )
        
        assert can_handle is False
        assert "exceeds" in issues[0]
        assert "context limit" in issues[0]
    
    def test_can_handle_request_multiple_requirements(self):
        """Test capability checking with multiple requirements"""
        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai",
            has_tools=True,
            has_vision=True,
            needs_streaming=True,
            needs_json=True,
            estimated_tokens=10000
        )
        
        assert can_handle is True
        assert len(issues) == 0
        
        # Test provider that can't handle all requirements
        can_handle, issues = CapabilityChecker.can_handle_request(
            "groq",
            has_tools=True,
            has_vision=True,  # Groq doesn't support vision
            needs_streaming=True,
            needs_json=True
        )
        
        assert can_handle is False
        assert len(issues) == 1  # Should have vision issue
        assert "vision" in issues[0]
    
    def test_can_handle_unknown_provider(self):
        """Test handling of unknown provider"""
        can_handle, issues = CapabilityChecker.can_handle_request("unknown_provider")
        
        assert can_handle is False
        assert len(issues) == 1
        assert "Unknown provider" in issues[0]
    
    def test_get_best_provider_simple(self):
        """Test finding best provider for simple requirements"""
        best = CapabilityChecker.get_best_provider({Feature.STREAMING})
        
        assert best is not None
        assert best in PROVIDER_CAPABILITIES
        assert PROVIDER_CAPABILITIES[best].supports(Feature.STREAMING)
    
    def test_get_best_provider_complex_requirements(self):
        """Test finding best provider for complex requirements"""
        requirements = {Feature.STREAMING, Feature.TOOLS, Feature.VISION, Feature.JSON_MODE}
        best = CapabilityChecker.get_best_provider(requirements)
        
        assert best is not None
        caps = PROVIDER_CAPABILITIES[best]
        
        # Should support all requirements
        for feature in requirements:
            assert caps.supports(feature)
    
    def test_get_best_provider_with_exclusions(self):
        """Test finding best provider with exclusions"""
        requirements = {Feature.STREAMING}
        excluded = {"openai", "anthropic"}
        
        best = CapabilityChecker.get_best_provider(requirements, exclude=excluded)
        
        assert best is not None
        assert best not in excluded
        assert PROVIDER_CAPABILITIES[best].supports(Feature.STREAMING)
    
    def test_get_best_provider_impossible_requirements(self):
        """Test when no provider can meet requirements"""
        # Create impossible requirements
        impossible_requirements = {
            Feature.STREAMING, Feature.TOOLS, Feature.VISION, 
            Feature.JSON_MODE, Feature.MULTIMODAL, Feature.STRUCTURED_OUTPUT
        }
        
        best = CapabilityChecker.get_best_provider(impossible_requirements)
        
        # Might return None if no provider supports all features
        # Or might return the best available provider
        if best is not None:
            caps = PROVIDER_CAPABILITIES[best]
            # Should support as many as possible
            supported_count = sum(1 for req in impossible_requirements if caps.supports(req))
            assert supported_count > 0
    
    def test_get_provider_info(self):
        """Test getting provider information"""
        info = CapabilityChecker.get_provider_info("openai")
        
        assert info is not None
        assert info.name == "OpenAI"
        assert info.supports(Feature.STREAMING)
        
        # Test unknown provider
        info = CapabilityChecker.get_provider_info("unknown")
        assert info is None
    
    def test_list_providers_with_feature(self):
        """Test listing providers with specific feature"""
        streaming_providers = CapabilityChecker.list_providers_with_feature(Feature.STREAMING)
        
        assert len(streaming_providers) > 0
        assert "openai" in streaming_providers
        assert "anthropic" in streaming_providers
        
        # All returned providers should support streaming
        for provider in streaming_providers:
            caps = PROVIDER_CAPABILITIES[provider]
            assert caps.supports(Feature.STREAMING)
        
        # Test less common feature
        vision_providers = CapabilityChecker.list_providers_with_feature(Feature.VISION)
        
        assert "openai" in vision_providers
        assert "anthropic" in vision_providers
        assert "gemini" in vision_providers
        assert "groq" not in vision_providers  # Groq doesn't support vision
    
    def test_compare_providers(self):
        """Test provider comparison functionality"""
        comparison = CapabilityChecker.compare_providers(["openai", "anthropic", "groq"])
        
        assert "openai" in comparison
        assert "anthropic" in comparison
        assert "groq" in comparison
        
        # Check structure of comparison
        openai_info = comparison["openai"]
        assert "features" in openai_info
        assert "max_context_length" in openai_info
        assert "rate_limits" in openai_info
        
        # Check that features are properly listed
        assert "streaming" in openai_info["features"]
        assert "tools" in openai_info["features"]
        
        # Test with unknown provider
        comparison = CapabilityChecker.compare_providers(["openai", "unknown"])
        
        assert "openai" in comparison
        assert "unknown" in comparison
        assert "error" in comparison["unknown"]


class TestFeatureEnum:
    """Test suite for Feature enum"""
    
    def test_feature_enum_values(self):
        """Test that all expected features are defined"""
        expected_features = [
            "streaming", "tools", "vision", "json_mode", 
            "parallel_calls", "system_messages", "multimodal",
            "structured_output", "function_calling"
        ]
        
        feature_values = [f.value for f in Feature]
        
        for expected in expected_features:
            assert expected in feature_values
    
    def test_feature_enum_uniqueness(self):
        """Test that feature values are unique"""
        feature_values = [f.value for f in Feature]
        assert len(feature_values) == len(set(feature_values))


class TestCapabilityIntegration:
    """Integration tests for capability system"""
    
    def test_real_world_scenario_basic_chat(self):
        """Test capability checking for basic chat scenario"""
        # Simple chat request
        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai",
            needs_streaming=True,
            estimated_tokens=1000
        )
        
        assert can_handle is True
        assert len(issues) == 0
    
    def test_real_world_scenario_function_calling(self):
        """Test capability checking for function calling scenario"""
        # Function calling with streaming
        can_handle, issues = CapabilityChecker.can_handle_request(
            "anthropic",
            has_tools=True,
            needs_streaming=True,
            estimated_tokens=5000
        )
        
        assert can_handle is True
        assert len(issues) == 0
    
    def test_real_world_scenario_vision_analysis(self):
        """Test capability checking for vision analysis scenario"""
        # Vision analysis with large context
        can_handle, issues = CapabilityChecker.can_handle_request(
            "gemini",
            has_vision=True,
            needs_streaming=True,
            estimated_tokens=50000
        )
        
        assert can_handle is True
        assert len(issues) == 0
    
    def test_real_world_scenario_json_generation(self):
        """Test capability checking for JSON generation scenario"""
        # JSON mode with function calling
        can_handle, issues = CapabilityChecker.can_handle_request(
            "openai",
            has_tools=True,
            needs_json=True,
            estimated_tokens=2000
        )
        
        assert can_handle is True
        assert len(issues) == 0
    
    def test_provider_selection_workflow(self):
        """Test complete provider selection workflow"""
        # User wants vision + tools + streaming
        requirements = {Feature.VISION, Feature.TOOLS, Feature.STREAMING}
        
        # Get best provider
        best_provider = CapabilityChecker.get_best_provider(requirements)
        assert best_provider is not None
        
        # Verify it can handle the request
        can_handle, issues = CapabilityChecker.can_handle_request(
            best_provider,
            has_vision=True,
            has_tools=True,
            needs_streaming=True
        )
        
        assert can_handle is True
        assert len(issues) == 0
        
        # Get provider info for display
        info = CapabilityChecker.get_provider_info(best_provider)
        assert info is not None
        assert info.supports(Feature.VISION)
        assert info.supports(Feature.TOOLS)
        assert info.supports(Feature.STREAMING)


# Fixtures for common test data
@pytest.fixture
def basic_capabilities():
    """Basic capabilities for testing"""
    return ProviderCapabilities(
        name="Test Provider",
        features={Feature.STREAMING, Feature.TOOLS},
        max_context_length=10000,
        rate_limits={"default": 100}
    )


@pytest.fixture
def advanced_capabilities():
    """Advanced capabilities for testing"""
    return ProviderCapabilities(
        name="Advanced Provider",
        features={
            Feature.STREAMING, Feature.TOOLS, Feature.VISION,
            Feature.JSON_MODE, Feature.MULTIMODAL
        },
        max_context_length=100000,
        max_output_tokens=8192,
        rate_limits={"default": 1000, "premium": 10000},
        supported_models=["model-v1", "model-v2"]
    )


# Parametrized tests
@pytest.mark.parametrize("provider,expected_features", [
    ("openai", {Feature.STREAMING, Feature.TOOLS, Feature.VISION, Feature.JSON_MODE}),
    ("anthropic", {Feature.STREAMING, Feature.TOOLS, Feature.VISION}),
    ("groq", {Feature.STREAMING, Feature.TOOLS}),
    ("gemini", {Feature.STREAMING, Feature.TOOLS, Feature.VISION, Feature.MULTIMODAL}),
    ("ollama", {Feature.STREAMING, Feature.TOOLS}),
])
def test_provider_feature_sets(provider, expected_features):
    """Test that providers have expected core features"""
    caps = PROVIDER_CAPABILITIES[provider]
    
    for feature in expected_features:
        assert caps.supports(feature), f"{provider} should support {feature.value}"


@pytest.mark.parametrize("feature,min_provider_count", [
    (Feature.STREAMING, 4),  # Most providers support streaming
    (Feature.TOOLS, 4),      # Most providers support tools  
    (Feature.VISION, 2),     # At least 2 providers support vision
    (Feature.JSON_MODE, 2),  # At least 2 providers support JSON mode
])
def test_feature_availability(feature, min_provider_count):
    """Test that features are available across multiple providers"""
    providers_with_feature = CapabilityChecker.list_providers_with_feature(feature)
    
    assert len(providers_with_feature) >= min_provider_count, \
        f"Expected at least {min_provider_count} providers to support {feature.value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])