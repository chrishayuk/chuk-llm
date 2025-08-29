#!/usr/bin/env python3
"""
Base URL Environment Variable Diagnostic
========================================

This diagnostic tests the base URL environment variable functionality,
ensuring providers can correctly resolve base URLs from various sources.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_llm.configuration.unified_config import get_config
from chuk_llm.api.dynamic_providers import (
    register_openai_compatible,
    unregister_provider
)

class BaseURLDiagnostic:
    """Diagnostic for base URL environment variable functionality."""
    
    def __init__(self):
        self.config = get_config()
        self.test_results: List[Tuple[str, bool, str]] = []
        self.original_env: Dict[str, Optional[str]] = {}
    
    def save_env(self, *env_vars):
        """Save original environment variable values."""
        for var in env_vars:
            self.original_env[var] = os.getenv(var)
    
    def restore_env(self):
        """Restore original environment variable values."""
        for var, value in self.original_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value
        self.original_env.clear()
    
    def add_result(self, test_name: str, passed: bool, details: str):
        """Add a test result."""
        self.test_results.append((test_name, passed, details))
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if details:
            print(f"         {details}")
    
    def test_standard_patterns(self):
        """Test standard environment variable patterns."""
        print("\n1. Testing Standard Environment Patterns")
        print("-" * 40)
        
        test_provider = "openai"
        patterns = [
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "OPENAI_API_URL",
            "OPENAI_ENDPOINT"
        ]
        
        self.save_env(*patterns)
        
        for pattern in patterns:
            # Set the environment variable
            test_url = f"https://test.{pattern.lower()}.com/v1"
            os.environ[pattern] = test_url
            
            # Get the resolved base URL
            try:
                resolved = self.config.get_api_base(test_provider)
                
                # Check if it matches
                if resolved == test_url:
                    self.add_result(
                        f"Pattern {pattern}",
                        True,
                        f"Resolved to {test_url}"
                    )
                else:
                    self.add_result(
                        f"Pattern {pattern}",
                        False,
                        f"Expected {test_url}, got {resolved}"
                    )
            except Exception as e:
                self.add_result(
                    f"Pattern {pattern}",
                    False,
                    f"Error: {e}"
                )
            
            # Clear for next test
            del os.environ[pattern]
        
        self.restore_env()
    
    def test_custom_env_variable(self):
        """Test custom environment variable names."""
        print("\n2. Testing Custom Environment Variables")
        print("-" * 40)
        
        custom_env = "MY_CUSTOM_LLM_ENDPOINT"
        custom_url = "https://custom.endpoint.com/v1"
        
        self.save_env(custom_env)
        os.environ[custom_env] = custom_url
        
        try:
            # Register provider with custom env var
            provider = register_openai_compatible(
                name="custom_test",
                api_base_env=custom_env,
                models=["test-model"],
                default_model="test-model"
            )
            
            # Get resolved URL
            resolved = self.config.get_api_base("custom_test")
            
            if resolved == custom_url:
                self.add_result(
                    "Custom env variable",
                    True,
                    f"Custom var {custom_env} resolved correctly"
                )
            else:
                self.add_result(
                    "Custom env variable",
                    False,
                    f"Expected {custom_url}, got {resolved}"
                )
            
            # Clean up
            unregister_provider("custom_test")
            
        except Exception as e:
            self.add_result(
                "Custom env variable",
                False,
                f"Error: {e}"
            )
        
        self.restore_env()
    
    def test_priority_order(self):
        """Test priority order of base URL resolution."""
        print("\n3. Testing Priority Order")
        print("-" * 40)
        
        configured_url = "https://configured.com/v1"
        standard_url = "https://standard.com/v1"
        custom_url = "https://custom.com/v1"
        
        self.save_env("TEST_PRIO_CUSTOM", "TEST_PRIO_API_BASE")
        
        try:
            # Register with configured URL and custom env
            provider = register_openai_compatible(
                name="test_prio",
                api_base=configured_url,
                api_base_env="TEST_PRIO_CUSTOM",
                models=["test-model"],
                default_model="test-model"
            )
            
            # Test 1: Only configured URL
            resolved = self.config.get_api_base("test_prio")
            if resolved == configured_url:
                self.add_result(
                    "Configured URL only",
                    True,
                    "Uses configured URL when no env vars"
                )
            else:
                self.add_result(
                    "Configured URL only",
                    False,
                    f"Expected {configured_url}, got {resolved}"
                )
            
            # Test 2: Standard pattern overrides configured
            os.environ["TEST_PRIO_API_BASE"] = standard_url
            resolved = self.config.get_api_base("test_prio")
            if resolved == standard_url:
                self.add_result(
                    "Standard pattern override",
                    True,
                    "Standard pattern overrides configured URL"
                )
            else:
                self.add_result(
                    "Standard pattern override",
                    False,
                    f"Expected {standard_url}, got {resolved}"
                )
            
            # Test 3: Custom env overrides standard
            os.environ["TEST_PRIO_CUSTOM"] = custom_url
            resolved = self.config.get_api_base("test_prio")
            if resolved == custom_url:
                self.add_result(
                    "Custom env override",
                    True,
                    "Custom env overrides standard pattern"
                )
            else:
                self.add_result(
                    "Custom env override",
                    False,
                    f"Expected {custom_url}, got {resolved}"
                )
            
            # Clean up
            unregister_provider("test_prio")
            
        except Exception as e:
            self.add_result(
                "Priority order",
                False,
                f"Error: {e}"
            )
        
        self.restore_env()
    
    def test_update_with_env(self):
        """Test updating provider with api_base_env."""
        print("\n4. Testing Update with Environment Variable")
        print("-" * 40)
        
        update_env = "UPDATED_BASE_URL"
        update_url = "https://updated.com/v1"
        
        self.save_env(update_env)
        
        try:
            # Register initial provider
            provider = register_openai_compatible(
                name="update_test",
                api_base="https://initial.com/v1",
                models=["test-model"],
                default_model="test-model"
            )
            
            # Update to use environment variable
            from chuk_llm.api.dynamic_providers import update_provider
            updated = update_provider(
                "update_test",
                api_base_env=update_env
            )
            
            # Set environment variable
            os.environ[update_env] = update_url
            
            # Check resolution
            resolved = self.config.get_api_base("update_test")
            
            if resolved == update_url:
                self.add_result(
                    "Update with api_base_env",
                    True,
                    "Provider updated to use environment variable"
                )
            else:
                self.add_result(
                    "Update with api_base_env",
                    False,
                    f"Expected {update_url}, got {resolved}"
                )
            
            # Clean up
            unregister_provider("update_test")
            
        except Exception as e:
            self.add_result(
                "Update with api_base_env",
                False,
                f"Error: {e}"
            )
        
        self.restore_env()
    
    def test_existing_providers(self):
        """Test environment variables with existing providers."""
        print("\n5. Testing Existing Providers")
        print("-" * 40)
        
        # Test OpenAI provider
        test_url = "https://proxy.test.com/v1"
        self.save_env("OPENAI_API_BASE")
        
        try:
            os.environ["OPENAI_API_BASE"] = test_url
            resolved = self.config.get_api_base("openai")
            
            if resolved == test_url:
                self.add_result(
                    "OpenAI provider env override",
                    True,
                    "Existing provider uses environment variable"
                )
            else:
                self.add_result(
                    "OpenAI provider env override",
                    False,
                    f"Expected {test_url}, got {resolved}"
                )
            
        except Exception as e:
            self.add_result(
                "OpenAI provider env override",
                False,
                f"Error: {e}"
            )
        
        self.restore_env()
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("\n" + "=" * 60)
        print("BASE URL ENVIRONMENT VARIABLE DIAGNOSTIC")
        print("=" * 60)
        
        self.test_standard_patterns()
        self.test_custom_env_variable()
        self.test_priority_order()
        self.test_update_with_env()
        self.test_existing_providers()
        
        # Summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, p, _ in self.test_results if p)
        total = len(self.test_results)
        
        print(f"\nTests Passed: {passed}/{total}")
        
        if passed == total:
            print("\n✅ ALL TESTS PASSED!")
            print("Base URL environment variable support is working correctly.")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("\nFailed tests:")
            for name, passed, details in self.test_results:
                if not passed:
                    print(f"  - {name}: {details}")
        
        print("\n" + "=" * 60)
        print("CONFIGURATION GUIDE")
        print("=" * 60)
        print("""
To use base URL environment variables:

1. Standard Patterns (automatic):
   export OPENAI_API_BASE=https://proxy.com/v1
   export ANTHROPIC_BASE_URL=https://custom.anthropic.com

2. Custom Environment Variables:
   register_openai_compatible(
       name="custom_provider",
       api_base_env="MY_CUSTOM_ENDPOINT",
       ...
   )

3. Priority Order:
   - Custom env var (api_base_env) - highest
   - Standard patterns ({PROVIDER}_API_BASE, etc.)
   - Configured api_base - lowest

4. Common Use Cases:
   - Corporate proxies
   - OpenAI-compatible services
   - Environment-specific endpoints
   - Local development servers
""")

def main():
    """Run the diagnostic."""
    diagnostic = BaseURLDiagnostic()
    diagnostic.run_all_tests()

if __name__ == "__main__":
    main()