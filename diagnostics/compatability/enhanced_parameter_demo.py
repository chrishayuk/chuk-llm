#!/usr/bin/env python3
# diagnostics/compatibility/enhanced_parameter_demo.py
"""
Enhanced LLM Provider Parameter Compatibility Demo
Shows live parameter compatibility testing with improved error handling and insights
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')


class EnhancedParameterCompatibilityDemo:
    """Enhanced parameter compatibility demonstration with comprehensive testing"""
    
    def __init__(self):
        self.providers = self._detect_available_providers()
        
        # Enhanced parameter definitions with universal config mapping
        self.universal_parameters = {
            "temperature": {
                "description": "Controls randomness (0.0 = deterministic, 2.0 = very random)",
                "type": "float",
                "universal_range": (0.0, 2.0),
                "provider_ranges": {
                    "openai": (0.0, 2.0),
                    "anthropic": (0.0, 1.0),
                    "gemini": (0.0, 2.0),
                    "groq": (0.0, 2.0),
                    "ollama": (0.0, 2.0),
                    "mistral": (0.0, 1.0),
                    "deepseek": (0.0, 2.0),
                    "perplexity": (0.0, 2.0)
                },
                "test_values": [0.0, 0.1, 0.7, 1.0, 1.5, 2.0],
                "edge_cases": [-0.1, 3.0]
            },
            "max_tokens": {
                "description": "Maximum tokens in response",
                "type": "int",
                "universal_range": (1, 4000),
                "provider_mappings": {
                    "gemini": "max_output_tokens",
                    "watsonx": "max_new_tokens"
                },
                "provider_ranges": {
                    "openai": (1, 4000),
                    "anthropic": (1, 4000),
                    "gemini": (1, 2048),
                    "groq": (1, 2000),
                    "ollama": (1, 2000),
                    "mistral": (1, 2000)
                },
                "test_values": [1, 10, 50, 100, 500, 1000],
                "edge_cases": [0, -1]
            },
            "top_p": {
                "description": "Nucleus sampling (0.1 = focused, 1.0 = diverse)",
                "type": "float",
                "universal_range": (0.0, 1.0),
                "test_values": [0.0, 0.1, 0.5, 0.9, 1.0],
                "edge_cases": [-0.1, 1.5]
            },
            "top_k": {
                "description": "Top-k sampling (1 = deterministic, 100 = diverse)",
                "type": "int",
                "universal_range": (1, 100),
                "supported_providers": ["anthropic", "gemini", "ollama", "watsonx"],
                "test_values": [1, 5, 20, 50, 100],
                "edge_cases": [0, 101]
            },
            "stop": {
                "description": "Stop sequences to end generation",
                "type": "list|str",
                "supported_providers": ["openai", "groq", "ollama", "mistral", "deepseek", "perplexity"],
                "provider_mappings": {
                    "anthropic": "stop_sequences",
                    "gemini": "stop_sequences"
                },
                "test_values": [None, ".", [".", "!"], ["\n", "END"]],
                "edge_cases": [123, {"invalid": "dict"}]
            },
            "frequency_penalty": {
                "description": "Penalize frequent tokens (-2.0 to 2.0)",
                "type": "float",
                "universal_range": (-2.0, 2.0),
                "supported_providers": ["openai", "gemini", "groq", "ollama", "mistral"],
                "test_values": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "edge_cases": [-3.0, 3.0]
            },
            "presence_penalty": {
                "description": "Penalize new topics (-2.0 to 2.0)",
                "type": "float", 
                "universal_range": (-2.0, 2.0),
                "supported_providers": ["openai", "gemini", "groq", "ollama", "mistral"],
                "test_values": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "edge_cases": [-3.0, 3.0]
            },
            "system": {
                "description": "System message for conversation context",
                "type": "str",
                "supported_providers": ["openai", "anthropic", "gemini", "groq", "mistral", "deepseek", "perplexity"],
                "test_values": ["You are a helpful assistant.", ""],
                "edge_cases": [None, 123]
            }
        }
        
        # Test scenarios for parameter validation
        self.test_scenarios = [
            {
                "name": "Conservative (Low Temperature)",
                "params": {"temperature": 0.1, "max_tokens": 30},
                "message": "What is the capital of France?",
                "expected": "deterministic, factual response"
            },
            {
                "name": "Balanced (Medium Temperature)",
                "params": {"temperature": 0.7, "max_tokens": 50},
                "message": "Write a creative greeting",
                "expected": "creative but coherent response"
            },
            {
                "name": "Creative (High Temperature)",
                "params": {"temperature": 1.2, "max_tokens": 40},
                "message": "Invent a funny word",
                "expected": "highly creative, diverse responses"
            },
            {
                "name": "Focused Generation",
                "params": {"temperature": 0.3, "top_p": 0.8, "max_tokens": 60},
                "message": "Explain quantum computing in simple terms",
                "expected": "focused, coherent explanation"
            }
        ]
    
    def _detect_available_providers(self) -> List[str]:
        """Detect which providers have API keys configured"""
        provider_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "groq": "GROQ_API_KEY",
            "watsonx": ["WATSONX_API_KEY", "IBM_CLOUD_API_KEY"],
            "ollama": None  # Usually doesn't need API key
        }
        
        available = []
        for provider, env_vars in provider_env_vars.items():
            if env_vars is None:
                available.append(provider)  # Ollama
            elif isinstance(env_vars, list):
                if any(os.getenv(var) for var in env_vars):
                    available.append(provider)
            elif os.getenv(env_vars):
                available.append(provider)
        
        return available
    
    def _map_parameter_for_provider(self, provider: str, param_name: str, param_value: Any) -> tuple[str, Any]:
        """Map parameter to provider-specific name and validate value"""
        param_config = self.universal_parameters.get(param_name, {})
        
        # Check if parameter is supported by provider
        supported_providers = param_config.get("supported_providers")
        if supported_providers and provider not in supported_providers:
            return None, None
        
        # Map parameter name if needed
        provider_mappings = param_config.get("provider_mappings", {})
        actual_param_name = provider_mappings.get(provider, param_name)
        
        # Validate and clamp value to provider's range
        provider_ranges = param_config.get("provider_ranges", {})
        if provider in provider_ranges and isinstance(param_value, (int, float)):
            min_val, max_val = provider_ranges[provider]
            param_value = max(min_val, min(max_val, param_value))
        
        return actual_param_name, param_value
    
    def _prepare_params_for_provider(self, provider: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for a specific provider"""
        provider_params = {}
        
        for param_name, param_value in params.items():
            if param_value is None:
                continue
                
            mapped_name, mapped_value = self._map_parameter_for_provider(provider, param_name, param_value)
            if mapped_name and mapped_value is not None:
                provider_params[mapped_name] = mapped_value
        
        # Add required parameters for specific providers
        if provider == "anthropic" and "max_tokens" not in provider_params:
            provider_params["max_tokens"] = 100
        
        return provider_params
    
    async def run_enhanced_demo(self):
        """Run enhanced parameter compatibility demonstration"""
        print("\n" + "="*100)
        print("🧪 ENHANCED LLM PROVIDER PARAMETER COMPATIBILITY DEMO".center(100))
        print("="*100)
        
        print(f"📋 Available Providers: {', '.join(self.providers)}")
        print(f"🧪 Universal Parameters: {len(self.universal_parameters)}")
        print(f"🎯 Test Scenarios: {len(self.test_scenarios)}")
        
        if not self.providers:
            print("❌ No providers available. Please configure API keys.")
            return
        
        # Run comprehensive parameter testing
        await self._test_parameter_support_matrix()
        await self._test_parameter_ranges()
        await self._test_parameter_scenarios()
        await self._show_parameter_best_practices()
        await self._generate_compatibility_summary()
    
    async def _test_parameter_support_matrix(self):
        """Test parameter support across all providers"""
        print(f"\n📊 Universal Parameter Support Matrix")
        print("-" * 60)
        
        # Test each parameter for basic support
        results = {}
        for param_name, param_config in self.universal_parameters.items():
            print(f"   Testing {param_name}...")
            results[param_name] = {}
            
            for provider in self.providers:
                try:
                    # Use a safe test value
                    test_values = param_config.get("test_values", [])
                    if not test_values:
                        continue
                    
                    safe_value = test_values[1] if len(test_values) > 1 else test_values[0]
                    test_params = {param_name: safe_value}
                    
                    # Add required params
                    if param_name != "max_tokens":
                        test_params["max_tokens"] = 50
                    if param_name != "temperature":
                        test_params["temperature"] = 0.7
                    
                    provider_params = self._prepare_params_for_provider(provider, test_params)
                    
                    if provider_params:
                        client = get_client(provider)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            response = await client.create_completion(
                                [{"role": "user", "content": "Say hello"}],
                                **provider_params
                            )
                            
                            if response and response.get("response"):
                                results[param_name][provider] = "✅"
                            else:
                                results[param_name][provider] = "⚠️"
                    else:
                        results[param_name][provider] = "❌"
                        
                except Exception as e:
                    error_str = str(e).lower()
                    if "not supported" in error_str or "invalid" in error_str:
                        results[param_name][provider] = "❌"
                    elif any(x in error_str for x in ["401", "auth", "key"]):
                        results[param_name][provider] = "🔑"
                    else:
                        results[param_name][provider] = "🔧"
        
        # Display matrix
        self._print_support_matrix(results)
    
    def _print_support_matrix(self, results: Dict[str, Dict[str, str]]):
        """Print parameter support matrix table"""
        print(f"\n{'Parameter':<20} | " + " | ".join(f"{p:<10}" for p in self.providers))
        print("-" * (22 + 13 * len(self.providers)))
        
        for param, provider_results in results.items():
            row = f"{param:<20} | "
            for provider in self.providers:
                status = provider_results.get(provider, "❓")
                row += f"{status:<10} | "
            print(row)
        
        print("\nLegend:")
        print("  ✅ Full Support    ⚠️ Partial Support    ❌ Not Supported")
        print("  🔧 Needs Mapping  🔑 Auth Issue         ❓ Not Tested")
    
    async def _test_parameter_ranges(self):
        """Test parameter ranges and edge cases"""
        print(f"\n🔧 Parameter Range Validation Testing")
        print("-" * 60)
        
        range_results = {}
        
        for param_name, param_config in self.universal_parameters.items():
            if param_config.get("type") not in ["float", "int"]:
                continue
                
            print(f"\n🎯 Testing {param_name.upper()} Ranges")
            print(f"   {param_config['description']}")
            
            range_results[param_name] = {}
            
            for provider in self.providers:
                print(f"     Testing {provider}...")
                
                # Test edge cases
                edge_cases = param_config.get("edge_cases", [])
                edge_results = []
                
                for edge_value in edge_cases:
                    try:
                        test_params = {param_name: edge_value, "max_tokens": 20, "temperature": 0.5}
                        provider_params = self._prepare_params_for_provider(provider, test_params)
                        
                        if provider_params:
                            client = get_client(provider)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                
                                response = await client.create_completion(
                                    [{"role": "user", "content": "Hello"}],
                                    **provider_params
                                )
                                edge_results.append("✅")  # Accepted edge case
                        else:
                            edge_results.append("🚫")  # Parameter not supported
                            
                    except Exception:
                        edge_results.append("❌")  # Correctly rejected edge case
                
                # Determine provider range support
                provider_ranges = param_config.get("provider_ranges", {})
                if provider in provider_ranges:
                    min_val, max_val = provider_ranges[provider]
                    range_status = f"✅ {min_val} to {max_val}"
                    edge_status = "✅ Strict" if all(r == "❌" for r in edge_results) else "⚠️ Lenient"
                else:
                    range_status = "🔧 Universal"
                    edge_status = "❓ Unknown"
                
                range_results[param_name][provider] = {
                    "range": range_status,
                    "edge_handling": edge_status,
                    "overall": "✅" if "✅" in range_status else "🔧"
                }
        
        # Display range results
        self._print_range_results(range_results)
    
    def _print_range_results(self, results: Dict[str, Dict[str, Dict[str, str]]]):
        """Print parameter range testing results"""
        for param_name, provider_results in results.items():
            print(f"\n    📊 {param_name.upper()} Range Summary:")
            print(f"    {'Provider':<12} | {'Range':<15} | {'Edge Cases':<12} | {'Status':<8}")
            print("    " + "-" * 55)
            
            for provider in self.providers:
                result = provider_results.get(provider, {})
                range_info = result.get("range", "❓ Unknown")[:15]
                edge_info = result.get("edge_handling", "❓ Unknown")
                status = result.get("overall", "❓")
                
                print(f"    {provider:<12} | {range_info:<15} | {edge_info:<12} | {status:<8}")
    
    async def _test_parameter_scenarios(self):
        """Test different parameter scenarios across providers"""
        print(f"\n🎭 Parameter Scenario Testing")
        print("-" * 60)
        
        scenario_results = {}
        
        for scenario in self.test_scenarios:
            print(f"\n🧪 Scenario: {scenario['name']}")
            print(f"   Parameters: {scenario['params']}")
            print(f"   Message: \"{scenario['message']}\"")
            print(f"   Expected: {scenario['expected']}")
            
            scenario_results[scenario['name']] = {}
            
            for provider in self.providers:
                try:
                    provider_params = self._prepare_params_for_provider(provider, scenario['params'])
                    
                    if not provider_params:
                        scenario_results[scenario['name']][provider] = {
                            "status": "❌ No supported params",
                            "response": "",
                            "time": 0
                        }
                        continue
                    
                    client = get_client(provider)
                    
                    start_time = time.time()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        response = await client.create_completion(
                            [{"role": "user", "content": scenario['message']}],
                            **provider_params
                        )
                    
                    duration = time.time() - start_time
                    
                    if response and response.get("response"):
                        response_text = response["response"].strip()
                        scenario_results[scenario['name']][provider] = {
                            "status": "✅ Success",
                            "response": response_text[:100] + ("..." if len(response_text) > 100 else ""),
                            "time": duration
                        }
                    else:
                        scenario_results[scenario['name']][provider] = {
                            "status": "⚠️ No response",
                            "response": "",
                            "time": duration
                        }
                        
                except Exception as e:
                    error_str = str(e).lower()
                    if any(x in error_str for x in ["401", "auth", "key"]):
                        status = "🔑 Auth Error"
                    elif "503" in error_str or "unavailable" in error_str:
                        status = "⏳ Service Unavailable"
                    else:
                        status = "❌ Error"
                    
                    scenario_results[scenario['name']][provider] = {
                        "status": status,
                        "response": str(e)[:50] + "..." if len(str(e)) > 50 else str(e),
                        "time": 0
                    }
        
        # Display scenario results
        self._print_scenario_results(scenario_results)
    
    def _print_scenario_results(self, results: Dict[str, Dict[str, Dict[str, Any]]]):
        """Print scenario testing results in tables"""
        for scenario_name, provider_results in results.items():
            print(f"\n    📊 Results for: {scenario_name}")
            print(f"    {'Provider':<12} | {'Response':<50} | {'Status':<15} | {'Time':<8}")
            print("    " + "-" * 90)
            
            for provider in self.providers:
                result = provider_results.get(provider, {})
                response = result.get("response", "No data")[:48]
                status = result.get("status", "❓ Unknown")
                time_str = f"{result.get('time', 0):.2f}s"
                
                print(f"    {provider:<12} | {response:<50} | {status:<15} | {time_str:<8}")
    
    async def _show_parameter_best_practices(self):
        """Show parameter best practices and recommendations"""
        print(f"\n💡 Parameter Best Practices & Universal Config Insights")
        print("-" * 70)
        
        print("\n🎯 Universal Parameter Guidelines:")
        universal_recommendations = [
            ("temperature", "Use 0.7 as universal default, clamp to provider ranges"),
            ("max_tokens", "Always specify to prevent runaway generation"),
            ("top_p", "0.9 works well across most providers"),
            ("system", "Use universal system parameter, fallback to messages array"),
        ]
        
        for param, recommendation in universal_recommendations:
            print(f"   • {param}: {recommendation}")
        
        print("\n🔧 Provider-Specific Handling:")
        provider_specifics = [
            ("anthropic", "Requires max_tokens, doesn't support stop sequences"),
            ("gemini", "Auto-maps max_tokens→max_output_tokens, uses generation_config"),
            ("groq", "Full OpenAI compatibility, fast inference"),
            ("ollama", "Local deployment, supports most parameters"),
        ]
        
        for provider, specific in provider_specifics:
            if provider in self.providers:
                print(f"   • {provider}: {specific}")
        
        print("\n⚡ Performance Optimizations:")
        optimizations = [
            "Cache parameter validation results per provider",
            "Pre-filter unsupported parameters before API calls",
            "Use provider-specific parameter profiles",
            "Implement parameter auto-adjustment based on provider capabilities",
            "Monitor parameter success rates and adjust defaults"
        ]
        
        for opt in optimizations:
            print(f"   • {opt}")
    
    async def _generate_compatibility_summary(self):
        """Generate final compatibility summary and recommendations"""
        print(f"\n📊 Universal Config Parameter Compatibility Summary")
        print("=" * 70)
        
        # Calculate overall statistics
        total_providers = len(self.providers)
        total_parameters = len(self.universal_parameters)
        
        # Test a few key parameters quickly for summary
        compatibility_scores = {}
        key_params = ["temperature", "max_tokens", "top_p", "system"]
        
        for provider in self.providers:
            working_params = 0
            for param in key_params:
                try:
                    test_params = {param: 0.5 if param == "temperature" else 50 if param == "max_tokens" else 0.8 if param == "top_p" else "Test"}
                    if param != "max_tokens":
                        test_params["max_tokens"] = 30
                    
                    provider_params = self._prepare_params_for_provider(provider, test_params)
                    if provider_params and param in provider_params:
                        working_params += 1
                except:
                    pass
            
            compatibility_scores[provider] = (working_params / len(key_params)) * 100
        
        # Display summary
        print(f"\n🏆 Provider Parameter Compatibility Scores:")
        sorted_providers = sorted(compatibility_scores.items(), key=lambda x: x[1], reverse=True)
        
        for provider, score in sorted_providers:
            status = "🎉 Excellent" if score >= 90 else "✅ Good" if score >= 75 else "⚠️ Limited" if score >= 50 else "❌ Poor"
            print(f"   {provider:<12}: {score:5.1f}% - {status}")
        
        overall_score = sum(compatibility_scores.values()) / len(compatibility_scores) if compatibility_scores else 0
        
        print(f"\n📊 Overall Universal Config Compatibility: {overall_score:.1f}%")
        
        if overall_score >= 90:
            print("🎉 EXCELLENT! Universal parameter handling is working beautifully!")
        elif overall_score >= 75:
            print("✅ GOOD! Strong universal parameter support with minor provider-specific handling needed")
        elif overall_score >= 60:
            print("⚠️ MODERATE: Universal config working but needs provider-specific optimizations")
        else:
            print("🔧 NEEDS WORK: Universal parameter handling requires significant improvement")
        
        print(f"\n🎯 Key Recommendations:")
        recommendations = [
            f"✅ {len([s for s in compatibility_scores.values() if s >= 75])} providers have good parameter support",
            f"🔧 Implement parameter mapping for provider-specific needs",
            f"💡 Use parameter validation pipeline before API calls",
            f"⚡ Consider provider-specific parameter profiles for optimization"
        ]
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print("\n" + "="*70)


async def main():
    """Main demo function"""
    demo = EnhancedParameterCompatibilityDemo()
    await demo.run_enhanced_demo()


if __name__ == "__main__":
    asyncio.run(main())