# diagnostics/capabilities/utils/result_models.py
"""
Data models for diagnostic results.
Updated to be compatible with the new test runners and configuration system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

@dataclass
class ProviderResult:
    """Results for a single provider's diagnostic tests"""
    provider: str
    models: Dict[str, str]       # capability → model

    # Test results
    text_completion: Optional[bool] = None
    streaming_text: Optional[bool] = None
    function_call: Optional[bool] = None
    streaming_function_call: Optional[bool] = None
    vision: Optional[bool] = None

    # Error tracking and timing
    errors: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    def record(self, capability: str, result: Optional[bool]) -> None:
        """Record a test result using capability names"""
        capability_map = {
            "text_completion": "text_completion",
            "streaming_text": "streaming_text", 
            "function_call": "function_call",
            "streaming_function_call": "streaming_function_call",
            "vision": "vision"
        }
        
        attr_name = capability_map.get(capability, capability)
        if hasattr(self, attr_name):
            setattr(self, attr_name, result)

    @property
    def feature_set(self) -> Set[str]:
        """Get set of supported features based on successful tests"""
        features: Set[str] = set()
        if self.text_completion:
            features.add("text")
        if self.streaming_text:
            features.add("streaming")
        if self.function_call:
            features.add("tools")
        if self.streaming_function_call:
            features.add("streaming_tools")
        if self.vision:
            features.add("vision")
        return features

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate (excluding None/skipped tests)"""
        tests = [
            self.text_completion,
            self.streaming_text,
            self.function_call,
            self.streaming_function_call,
            self.vision
        ]
        # Only count tests that were actually attempted (not None)
        attempted_tests = [test for test in tests if test is not None]
        successful_tests = [test for test in attempted_tests if test is True]
        
        return len(successful_tests) / len(attempted_tests) if attempted_tests else 0.0

    @property
    def total_capabilities(self) -> int:
        """Get total number of capabilities actually tested (not skipped)"""
        tests = [
            self.text_completion,
            self.streaming_text,
            self.function_call,
            self.streaming_function_call,
            self.vision
        ]
        return len([test for test in tests if test is not None])

    @property
    def successful_capabilities(self) -> int:
        """Get number of successful capabilities"""
        tests = [
            self.text_completion,
            self.streaming_text,
            self.function_call,
            self.streaming_function_call,
            self.vision
        ]
        return sum(1 for test in tests if test is True)

    def get_capability_status(self, capability: str) -> Optional[bool]:
        """Get status of a specific capability"""
        capability_map = {
            "text": self.text_completion,
            "streaming": self.streaming_text,
            "tools": self.function_call,
            "streaming_tools": self.streaming_function_call,
            "vision": self.vision
        }
        return capability_map.get(capability)

    def has_errors(self) -> bool:
        """Check if there are any errors recorded"""
        return len(self.errors) > 0

    def get_fastest_capability(self) -> Optional[str]:
        """Get the capability with the fastest response time"""
        if not self.timings:
            return None
        fastest_item = min(self.timings.items(), key=lambda x: x[1])
        return f"{fastest_item[0]} ({fastest_item[1]:.2f}s)"

    def get_slowest_capability(self) -> Optional[str]:
        """Get the capability with the slowest response time"""
        if not self.timings:
            return None
        slowest_item = max(self.timings.items(), key=lambda x: x[1])
        return f"{slowest_item[0]} ({slowest_item[1]:.2f}s)"

    def get_error_summary(self) -> str:
        """Get a summary of errors for this provider"""
        if not self.errors:
            return "No errors"
        
        error_types = []
        for stage, error in self.errors.items():
            error_lower = error.lower()
            if "api key" in error_lower or "authentication" in error_lower:
                error_types.append("auth")
            elif "not support" in error_lower or "capability" in error_lower:
                error_types.append("unsupported")
            elif "rate limit" in error_lower:
                error_types.append("rate_limit")
            else:
                error_types.append("other")
        
        return f"{len(self.errors)} errors: {', '.join(set(error_types))}"

    def is_fully_functional(self) -> bool:
        """Check if provider supports all major capabilities"""
        return len(self.feature_set) >= 4  # text, streaming, tools, vision

    def get_model_summary(self, max_length: int = 50) -> str:
        """Get a truncated summary of models used"""
        if not self.models:
            return "No models"
        
        # Show unique models used
        unique_models = set(self.models.values())
        if len(unique_models) == 1:
            model = list(unique_models)[0]
            return model[:max_length] + "..." if len(model) > max_length else model
        else:
            summary = f"{len(unique_models)} models"
            return summary

    def __str__(self) -> str:
        """String representation for debugging"""
        return (f"ProviderResult({self.provider}, "
                f"success_rate={self.success_rate:.1%}, "
                f"features={len(self.feature_set)}, "
                f"errors={len(self.errors)})")


@dataclass
class DiagnosticSummary:
    """Summary of all diagnostic results"""
    results: List[ProviderResult] = field(default_factory=list)
    total_time: float = 0.0
    
    @property
    def total_providers(self) -> int:
        return len(self.results)
    
    @property
    def successful_providers(self) -> List[str]:
        """Get providers with no errors and good success rate"""
        return [r.provider for r in self.results if not r.has_errors() and r.success_rate >= 0.75]
    
    @property
    def problematic_providers(self) -> List[str]:
        """Get providers with significant errors"""
        return [r.provider for r in self.results if r.has_errors() or r.success_rate < 0.5]
    
    @property
    def full_featured_providers(self) -> List[str]:
        """Get providers supporting all major capabilities"""
        return [r.provider for r in self.results if r.is_fully_functional()]
    
    @property
    def overall_success_rate(self) -> float:
        """Calculate overall success rate across all providers"""
        if not self.results:
            return 0.0
        return sum(r.success_rate for r in self.results) / len(self.results)
    
    def get_fastest_providers(self, n: int = 3) -> List[tuple[str, float]]:
        """Get the N fastest providers by average response time"""
        provider_speeds = []
        for result in self.results:
            if result.timings:
                avg_time = sum(result.timings.values()) / len(result.timings)
                provider_speeds.append((result.provider, avg_time))
        
        provider_speeds.sort(key=lambda x: x[1])
        return provider_speeds[:n]
    
    def get_capability_matrix(self) -> Dict[str, Dict[str, Optional[bool]]]:
        """Get a matrix of capabilities vs providers"""
        capabilities = ["text_completion", "streaming_text", "function_call", 
                       "streaming_function_call", "vision"]
        matrix = {}
        
        for capability in capabilities:
            matrix[capability] = {}
            for result in self.results:
                matrix[capability][result.provider] = getattr(result, capability)
        
        return matrix