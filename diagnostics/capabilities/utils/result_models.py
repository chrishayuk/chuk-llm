# diagnostics/capabilities/result_models.py
"""
Data models for diagnostic results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

@dataclass
class ProviderResult:
    """Results for a single provider's diagnostic tests"""
    provider: str
    models: Dict[str, str]       # capability → model

    text_completion: Optional[bool] = None
    streaming_text: Optional[bool] = None
    function_call: Optional[bool] = None
    streaming_function_call: Optional[bool] = None
    vision: Optional[bool] = None

    errors: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    def record(self, attr: str, value: Optional[bool]) -> None:
        """Record a test result"""
        setattr(self, attr, value)

    @property
    def feature_set(self) -> Set[str]:
        """Get set of supported features"""
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
        """Calculate overall success rate"""
        tests = [
            self.text_completion,
            self.streaming_text,
            self.function_call,
            self.streaming_function_call,
            self.vision
        ]
        successful = sum(1 for test in tests if test is True)
        total = len([test for test in tests if test is not None])
        return successful / total if total > 0 else 0.0

    @property
    def total_capabilities(self) -> int:
        """Get total number of capabilities tested"""
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
        """Check if there are any errors"""
        return len(self.errors) > 0

    def get_fastest_capability(self) -> Optional[str]:
        """Get the capability with the fastest response time"""
        if not self.timings:
            return None
        return min(self.timings.items(), key=lambda x: x[1])[0]

    def get_slowest_capability(self) -> Optional[str]:
        """Get the capability with the slowest response time"""
        if not self.timings:
            return None
        return max(self.timings.items(), key=lambda x: x[1])[0]