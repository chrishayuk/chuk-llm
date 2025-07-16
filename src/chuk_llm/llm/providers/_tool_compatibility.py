# chuk_llm/llm/providers/_tool_compatibility.py
"""
Universal Tool Name Compatibility System
========================================

Standardized tool name sanitization and restoration for seamless integration
across all LLM providers. Handles naming restrictions and ensures consistent
behavior regardless of provider requirements.

Features:
- Universal sanitization algorithm
- Bidirectional name mapping with restoration
- Provider-specific optimization
- Comprehensive test framework
- Configuration-driven approach

Works with any tool naming convention including:
- MCP tools (stdio.read_query, filesystem.read_file)
- Dot notation (web.api.search, database.query.execute)
- Colon notation (service:method, namespace:function)
- Mixed conventions (tool.name:method, api.v1:get_data)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from dataclasses import dataclass

log = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Tool name compatibility levels"""
    NATIVE = "native"                    # Provider accepts complex names as-is
    SANITIZED = "sanitized"             # Provider needs sanitization
    AGGRESSIVE = "aggressive"           # Provider needs aggressive sanitization
    ENTERPRISE = "enterprise"           # Enterprise-grade restrictions


@dataclass
class ProviderToolRequirements:
    """Tool naming requirements for a provider"""
    pattern: Optional[str] = None       # Regex pattern for valid names
    max_length: int = 64               # Maximum name length
    compatibility_level: CompatibilityLevel = CompatibilityLevel.SANITIZED
    forbidden_chars: Set[str] = None   # Explicitly forbidden characters
    required_prefix: Optional[str] = None  # Required name prefix
    case_sensitive: bool = True        # Whether names are case sensitive
    
    def __post_init__(self):
        if self.forbidden_chars is None:
            self.forbidden_chars = {'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}


# Provider-specific requirements
PROVIDER_REQUIREMENTS = {
    "mistral": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        max_length=64,
        compatibility_level=CompatibilityLevel.SANITIZED,
        forbidden_chars={'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
    
    "anthropic": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_-]{1,128}$",
        max_length=128,
        compatibility_level=CompatibilityLevel.SANITIZED,
        forbidden_chars={'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
    
    # OpenAI actually requires sanitization - dots and colons are forbidden
    "openai": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",  # Same as Mistral - no dots or colons
        max_length=64,
        compatibility_level=CompatibilityLevel.SANITIZED,  # CHANGED from NATIVE to SANITIZED
        forbidden_chars={'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
    
    # Azure OpenAI should match OpenAI requirements
    "azure_openai": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",  # Same as OpenAI
        max_length=64,
        compatibility_level=CompatibilityLevel.SANITIZED,  # CHANGED from NATIVE to SANITIZED
        forbidden_chars={'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
    
    "gemini": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        max_length=64,
        compatibility_level=CompatibilityLevel.SANITIZED,
        forbidden_chars={'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
    
    "groq": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        max_length=64,
        compatibility_level=CompatibilityLevel.SANITIZED,
        forbidden_chars={'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
    
    "watsonx": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        max_length=64,
        compatibility_level=CompatibilityLevel.ENTERPRISE,
        forbidden_chars={'.', ':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
    
    "ollama": ProviderToolRequirements(
        pattern=r"^[a-zA-Z0-9_.-]{1,64}$",
        max_length=64,
        compatibility_level=CompatibilityLevel.NATIVE,  # Local deployment, usually flexible
        forbidden_chars={':', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '[', ']', '{', '}', '|', '\\', '/', '?', '<', '>', ',', ';', '"', "'", '`', '~'}
    ),
}


class ToolNameSanitizer:
    """Universal tool name sanitizer for all providers"""
    
    @staticmethod
    def sanitize_universal(name: str, max_length: int = 64) -> str:
        """
        Universal sanitization algorithm that works across all providers.
        
        This is the most aggressive approach that ensures compatibility
        with even the strictest providers.
        
        Examples:
            stdio.read_query -> stdio_read_query
            web.api:search -> web_api_search
            namespace:function -> namespace_function
            complex.tool:method.v1 -> complex_tool_method_v1
        """
        if not name or not isinstance(name, str):
            return "unnamed_function"
        
        # 1. Replace all forbidden characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        
        # 2. Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # 3. Ensure starts with letter or underscore (not number or dash)
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'tool_' + sanitized
        
        # 4. Truncate to max length and clean trailing underscores/dashes
        sanitized = sanitized[:max_length].rstrip('_-')
        
        # 5. Ensure we have a valid result
        if not sanitized or len(sanitized) == 0:
            sanitized = "unnamed_function"
        
        return sanitized
    
    @staticmethod
    def sanitize_for_provider(name: str, provider: str) -> str:
        """
        Provider-specific sanitization that optimizes for each provider's requirements.
        """
        if not name or not isinstance(name, str):
            return "unnamed_function"
        
        requirements = PROVIDER_REQUIREMENTS.get(provider, PROVIDER_REQUIREMENTS["mistral"])
        
        # For native providers, do minimal sanitization
        if requirements.compatibility_level == CompatibilityLevel.NATIVE:
            return ToolNameSanitizer._sanitize_minimal(name, requirements)
        
        # For other providers, use appropriate level of sanitization
        elif requirements.compatibility_level == CompatibilityLevel.SANITIZED:
            return ToolNameSanitizer._sanitize_standard(name, requirements)
        
        elif requirements.compatibility_level == CompatibilityLevel.AGGRESSIVE:
            return ToolNameSanitizer._sanitize_aggressive(name, requirements)
        
        elif requirements.compatibility_level == CompatibilityLevel.ENTERPRISE:
            return ToolNameSanitizer._sanitize_enterprise(name, requirements)
        
        else:
            return ToolNameSanitizer.sanitize_universal(name, requirements.max_length)
    
    @staticmethod
    def _sanitize_minimal(name: str, requirements: ProviderToolRequirements) -> str:
        """Minimal sanitization for flexible providers"""
        # Only replace characters that are explicitly forbidden
        sanitized = name
        for char in requirements.forbidden_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Clean up multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Truncate if needed
        if len(sanitized) > requirements.max_length:
            sanitized = sanitized[:requirements.max_length].rstrip('_-')
        
        return sanitized or "unnamed_function"
    
    @staticmethod
    def _sanitize_standard(name: str, requirements: ProviderToolRequirements) -> str:
        """Standard sanitization for most providers"""
        # Replace forbidden characters with underscores
        sanitized = name
        for char in requirements.forbidden_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Ensure valid start
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'tool_' + sanitized
        
        # Truncate and clean
        sanitized = sanitized[:requirements.max_length].rstrip('_-')
        
        return sanitized or "unnamed_function"
    
    @staticmethod
    def _sanitize_aggressive(name: str, requirements: ProviderToolRequirements) -> str:
        """Aggressive sanitization for strict providers"""
        # Only allow alphanumeric, underscore, and dash
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Ensure starts with letter or underscore
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'func_' + sanitized
        
        # Truncate and clean
        sanitized = sanitized[:requirements.max_length].rstrip('_-')
        
        return sanitized or "unnamed_function"
    
    @staticmethod
    def _sanitize_enterprise(name: str, requirements: ProviderToolRequirements) -> str:
        """Enterprise-grade sanitization with maximum compatibility"""
        # Very conservative approach
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # No dashes for maximum compatibility
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Ensure starts with letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'enterprise_tool_' + sanitized
        
        # Truncate and clean
        sanitized = sanitized[:requirements.max_length].rstrip('_')
        
        return sanitized or "enterprise_function"
    
    @staticmethod
    def validate_name(name: str, provider: str) -> bool:
        """Validate if a name meets provider requirements"""
        if not name:
            return False
        
        requirements = PROVIDER_REQUIREMENTS.get(provider)
        if not requirements:
            return True  # Unknown provider, assume valid
        
        # Check length
        if len(name) > requirements.max_length:
            return False
        
        # Check pattern if specified
        if requirements.pattern:
            if not re.match(requirements.pattern, name):
                return False
        
        # Check forbidden characters
        if requirements.forbidden_chars:
            if any(char in name for char in requirements.forbidden_chars):
                return False
        
        return True


class ToolCompatibilityMixin:
    """
    Mixin class that provides standardized tool compatibility
    for all LLM provider clients.
    
    Handles any tool naming convention:
    - stdio.read_query -> stdio_read_query
    - web.api:search -> web_api_search  
    - database.sql.execute -> database_sql_execute
    - service:method -> service_method
    """
    
    def __init__(self, provider_name: str = "unknown"):
        """Initialize with provider name for proper sanitization"""
        self.provider_name = provider_name
        self._current_name_mapping: Dict[str, str] = {}
        self._sanitizer = ToolNameSanitizer()
    
    def get_tool_requirements(self) -> ProviderToolRequirements:
        """Get tool naming requirements for this provider"""
        return PROVIDER_REQUIREMENTS.get(self.provider_name, PROVIDER_REQUIREMENTS["mistral"])
    
    def get_tool_compatibility_info(self) -> Dict[str, Any]:
        """Get comprehensive tool compatibility information"""
        requirements = self.get_tool_requirements()
        
        return {
            "provider": self.provider_name,
            "tool_name_requirements": requirements.pattern or "flexible",
            "tool_compatibility": self._get_compatibility_status(),
            "max_tool_name_length": requirements.max_length,
            "requires_sanitization": requirements.compatibility_level != CompatibilityLevel.NATIVE,
            "compatibility_level": requirements.compatibility_level.value,
            "forbidden_characters": list(requirements.forbidden_chars) if requirements.forbidden_chars else [],
            "sample_transformations": self._get_sample_transformations(),
            "case_sensitive": requirements.case_sensitive,
        }
    
    def _get_compatibility_status(self) -> str:
        """Get tool compatibility status description"""
        requirements = self.get_tool_requirements()
        
        if requirements.compatibility_level == CompatibilityLevel.NATIVE:
            return "native_support"
        elif requirements.compatibility_level == CompatibilityLevel.SANITIZED:
            return "requires_sanitization_with_restoration"
        elif requirements.compatibility_level == CompatibilityLevel.AGGRESSIVE:
            return "requires_aggressive_sanitization"
        elif requirements.compatibility_level == CompatibilityLevel.ENTERPRISE:
            return "enterprise_grade_sanitization"
        else:
            return "unknown"
    
    def _get_sample_transformations(self) -> Dict[str, str]:
        """Get sample tool name transformations for this provider"""
        samples = [
            "stdio.read_query",
            "filesystem.read_file",
            "web.api:search",
            "database.sql.execute",
            "service:method",
            "namespace:function",
            "complex.tool:method.v1"
        ]
        
        return {
            original: self._sanitizer.sanitize_for_provider(original, self.provider_name)
            for original in samples
        }
    
    def _sanitize_tool_names(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """
        Standardized tool name sanitization with bidirectional mapping.
        
        This method should be called by all provider clients before sending
        tools to the API. It ensures compatibility while preserving original
        names for restoration.
        
        Works with any tool naming convention:
        - MCP style: stdio.read_query
        - API style: web.api:search  
        - Service style: database.sql.execute
        - Namespace style: service:method
        """
        if not tools:
            return tools
        
        # Reset mapping for new request
        self._current_name_mapping = {}
        
        sanitized_tools, self._current_name_mapping = self._sanitize_tools_with_mapping(tools)
        
        # Log sanitization activity
        if self._current_name_mapping:
            log.debug(f"[{self.provider_name}] Sanitized {len(self._current_name_mapping)} tool names for compatibility")
            for sanitized, original in self._current_name_mapping.items():
                if sanitized != original:
                    log.debug(f"[{self.provider_name}] {original} -> {sanitized}")
        
        return sanitized_tools
    
    def _sanitize_tools_with_mapping(self, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Sanitize tools and create bidirectional mapping for restoration.
        
        Returns:
            Tuple of (sanitized_tools, name_mapping)
            name_mapping: Dict[sanitized_name, original_name]
        """
        sanitized_tools = []
        name_mapping = {}
        
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
                original_name = tool["function"].get("name", "")
                
                if original_name:
                    # Sanitize the name
                    sanitized_name = self._sanitizer.sanitize_for_provider(original_name, self.provider_name)
                    
                    # Store mapping
                    name_mapping[sanitized_name] = original_name
                    
                    # Create sanitized tool
                    sanitized_tool = tool.copy()
                    sanitized_tool["function"] = tool["function"].copy()
                    sanitized_tool["function"]["name"] = sanitized_name
                    
                    sanitized_tools.append(sanitized_tool)
                else:
                    # Tool without name, add default
                    sanitized_tool = tool.copy()
                    sanitized_tool["function"] = tool["function"].copy()
                    sanitized_tool["function"]["name"] = "unnamed_function"
                    sanitized_tools.append(sanitized_tool)
            else:
                # Non-function tools pass through unchanged
                sanitized_tools.append(tool)
        
        return sanitized_tools, name_mapping
    
    def _restore_tool_names_in_response(self, response: Dict[str, Any], name_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Restore original tool names in response using bidirectional mapping.
        
        This makes sanitization completely transparent to users - they send
        any tool naming convention and receive the same names back.
        
        Examples:
        - User sends: stdio.read_query
        - API receives: stdio_read_query  
        - User gets back: stdio.read_query
        """
        if name_mapping is None:
            name_mapping = self._current_name_mapping
        
        if not name_mapping or not response.get("tool_calls"):
            return response
        
        # Create copy to avoid modifying original
        restored_response = response.copy()
        restored_tool_calls = []
        
        for tool_call in response["tool_calls"]:
            if isinstance(tool_call, dict) and "function" in tool_call and "name" in tool_call["function"]:
                sanitized_name = tool_call["function"]["name"]
                original_name = name_mapping.get(sanitized_name, sanitized_name)
                
                # Restore original name
                restored_tool_call = tool_call.copy()
                restored_tool_call["function"] = tool_call["function"].copy()
                restored_tool_call["function"]["name"] = original_name
                
                restored_tool_calls.append(restored_tool_call)
                
                # Log restoration if name changed
                if original_name != sanitized_name:
                    log.debug(f"[{self.provider_name}] Restored tool name: {sanitized_name} -> {original_name}")
            else:
                restored_tool_calls.append(tool_call)
        
        restored_response["tool_calls"] = restored_tool_calls
        return restored_response
    
    def validate_tool_names(self, tools: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate tool names against provider requirements.
        
        Returns:
            Tuple of (all_valid, list_of_issues)
        """
        issues = []
        all_valid = True
        
        for i, tool in enumerate(tools):
            if tool.get("type") == "function" and "function" in tool:
                name = tool["function"].get("name", "")
                
                if not name:
                    issues.append(f"Tool {i}: Missing name")
                    all_valid = False
                    continue
                
                if not self._sanitizer.validate_name(name, self.provider_name):
                    issues.append(f"Tool {i} '{name}': Does not meet {self.provider_name} requirements")
                    all_valid = False
                    
                    # Suggest sanitized version
                    sanitized = self._sanitizer.sanitize_for_provider(name, self.provider_name)
                    issues.append(f"  Suggested: '{sanitized}'")
        
        return all_valid, issues


# Comprehensive test framework
class ToolCompatibilityTester:
    """Comprehensive testing framework for tool compatibility across providers"""
    
    def __init__(self):
        self.test_cases = [
            # Common tool naming patterns
            "stdio.read_query",
            "filesystem.read_file", 
            "web.api:search",
            "database.sql.execute",
            "service:method",
            "namespace:function",
            "complex.tool:method.v1",
            
            # MCP-style tools
            "mcp.server:get_data",
            "stdio.list_tables",
            "stdio.describe_table",
            
            # API-style tools
            "api.v1:get_users",
            "rest.api:post_data",
            "graphql:query",
            
            # Standard patterns
            "already_valid_name",
            "tool-with-dashes",
            "tool_with_underscores",
            "UPPERCASE_TOOL",
            "mixedCaseToolName",
            
            # Problematic cases
            "123invalid_start",
            "tool@with#special!chars",
            "tool with spaces",
            "tool/with\\slashes", 
            "very.long.tool.name.that.exceeds.normal.limits.and.should.be.truncated.properly",
            "",  # Empty name
            "a",  # Single character
            "a" * 100,  # Very long
        ]
    
    def test_provider_compatibility(self, provider: str) -> Dict[str, Any]:
        """Test compatibility for a specific provider"""
        sanitizer = ToolNameSanitizer()
        results = {}
        
        for test_case in self.test_cases:
            try:
                sanitized = sanitizer.sanitize_for_provider(test_case, provider)
                is_valid = sanitizer.validate_name(sanitized, provider)
                
                results[test_case] = {
                    "original": test_case,
                    "sanitized": sanitized,
                    "changed": test_case != sanitized,
                    "valid": is_valid,
                    "length": len(sanitized),
                    "status": "✅ PASS" if is_valid else "❌ FAIL"
                }
            except Exception as e:
                results[test_case] = {
                    "original": test_case,
                    "error": str(e),
                    "status": "❌ ERROR"
                }
        
        return results
    
    def test_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """Test compatibility across all providers"""
        all_results = {}
        
        for provider in PROVIDER_REQUIREMENTS.keys():
            all_results[provider] = self.test_provider_compatibility(provider)
        
        return all_results
    
    def generate_compatibility_report(self) -> str:
        """Generate a comprehensive compatibility report"""
        results = self.test_all_providers()
        
        report = ["# Universal Tool Name Compatibility Report", ""]
        
        for provider, provider_results in results.items():
            report.append(f"## {provider.upper()}")
            report.append("")
            
            requirements = PROVIDER_REQUIREMENTS[provider]
            report.append(f"- **Compatibility Level**: {requirements.compatibility_level.value}")
            report.append(f"- **Max Length**: {requirements.max_length}")
            report.append(f"- **Pattern**: `{requirements.pattern or 'flexible'}`")
            report.append("")
            
            # Success rate
            total = len(provider_results)
            passed = sum(1 for r in provider_results.values() if r.get("valid", False))
            success_rate = (passed / total * 100) if total > 0 else 0
            
            report.append(f"**Success Rate**: {passed}/{total} ({success_rate:.1f}%)")
            report.append("")
            
            # Sample transformations
            report.append("**Sample Transformations**:")
            report.append("```")
            for test_case, result in list(provider_results.items())[:10]:  # First 10
                if "sanitized" in result:
                    status = "✅" if result.get("valid", False) else "❌"
                    report.append(f"{test_case:<30} -> {result['sanitized']:<30} {status}")
            report.append("```")
            report.append("")
        
        return "\n".join(report)


# Export public interface
__all__ = [
    'ToolCompatibilityMixin',
    'ToolNameSanitizer', 
    'ProviderToolRequirements',
    'CompatibilityLevel',
    'PROVIDER_REQUIREMENTS',
    'ToolCompatibilityTester'
]