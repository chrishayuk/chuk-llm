# diagnostics/compatibility/provider_aware_parameter_test.py
"""
Provider-aware parameter compatibility testing that accounts for provider differences
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress warnings
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client


class ProviderAwareParameterTester:
    """Test parameter compatibility with provider-specific parameter mapping"""
    
    def __init__(self):
        self.test_providers = ["openai", "anthropic", "gemini", "groq", "ollama", "mistral"]
        
        # Provider-specific parameter mappings and constraints
        self.provider_constraints = {
            "openai": {
                "temperature": {"min": 0.0, "max": 2.0},
                "max_tokens": {"min": 1, "max": 4000},
                "top_p": {"min": 0.0, "max": 1.0},
                "stop": {"supported": True},
                "frequency_penalty": {"min": -2.0, "max": 2.0},
                "presence_penalty": {"min": -2.0, "max": 2.0}
            },
            "anthropic": {
                "temperature": {"min": 0.0, "max": 1.0},  # Anthropic: 0-1 range
                "max_tokens": {"min": 1, "max": 4000, "required": True},
                "top_p": {"min": 0.0, "max": 1.0},
                "stop": {"supported": False},  # Anthropic doesn't support stop
                "top_k": {"min": 1, "max": 100}
            },
            "gemini": {
                "temperature": {"min": 0.0, "max": 2.0, "param_name": "temperature"},
                "max_tokens": {"supported": False},  # Gemini uses different parameter
                "top_p": {"min": 0.0, "max": 1.0, "param_name": "top_p"},
                "top_k": {"min": 1, "max": 100, "param_name": "top_k"},
                "stop": {"supported": False}  # Gemini handles differently
            },
            "groq": {
                "temperature": {"min": 0.0, "max": 2.0},
                "max_tokens": {"min": 1, "max": 2000},
                "top_p": {"min": 0.0, "max": 1.0},
                "stop": {"supported": True},
                "frequency_penalty": {"min": -2.0, "max": 2.0},
                "presence_penalty": {"min": -2.0, "max": 2.0}
            },
            "ollama": {
                "temperature": {"min": 0.0, "max": 2.0},
                "max_tokens": {"min": 1, "max": 2000},
                "top_p": {"min": 0.0, "max": 1.0},
                "top_k": {"min": 1, "max": 100},
                "stop": {"supported": True}
            },
            "mistral": {
                "temperature": {"min": 0.0, "max": 1.0},
                "max_tokens": {"min": 1, "max": 2000},
                "top_p": {"min": 0.0, "max": 1.0},
                "stop": {"supported": True}
            }
        }
        
        # Test messages
        self.test_messages = [
            {"role": "user", "content": "Say 'Hello' and nothing else"}
        ]
    
    def _get_valid_parameter_value(self, provider: str, param_name: str, test_value: Any) -> tuple[Any, bool]:
        """Get a valid parameter value for the provider, return (value, is_supported)"""
        
        if provider not in self.provider_constraints:
            return test_value, True
        
        constraints = self.provider_constraints[provider].get(param_name, {})
        
        # Check if parameter is supported
        if constraints.get("supported") is False:
            return None, False
        
        # Apply constraints
        if isinstance(test_value, (int, float)):
            if "min" in constraints and test_value < constraints["min"]:
                return constraints["min"], True
            if "max" in constraints and test_value > constraints["max"]:
                return constraints["max"], True
        
        return test_value, True
    
    def _build_provider_kwargs(self, provider: str, **params) -> Dict[str, Any]:
        """Build kwargs appropriate for the specific provider"""
        
        kwargs = {}
        constraints = self.provider_constraints.get(provider, {})
        
        for param_name, param_value in params.items():
            if param_value is None:
                continue
            
            param_constraints = constraints.get(param_name, {})
            
            # Check if parameter is supported
            if param_constraints.get("supported") is False:
                continue
            
            # Use provider-specific parameter name if different
            actual_param_name = param_constraints.get("param_name", param_name)
            
            # Add the parameter
            kwargs[actual_param_name] = param_value
        
        # Add required parameters
        for param_name, param_constraints in constraints.items():
            if param_constraints.get("required") and param_name not in kwargs:
                if param_name == "max_tokens":
                    kwargs["max_tokens"] = 100  # Default value
        
        return kwargs
    
    async def run_provider_aware_test(self, providers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run provider-aware parameter compatibility testing"""
        
        test_providers = providers or self.test_providers
        available_providers = [p for p in test_providers if p in self.test_providers]
        
        print("🧪 Running provider-aware parameter compatibility testing...")
        print(f"📋 Testing providers: {', '.join(available_providers)}")
        
        results = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "providers_tested": available_providers,
                "test_version": "3.0"
            },
            "provider_results": {},
            "parameter_matrix": {},
            "compatibility_summary": {}
        }
        
        # Test each provider
        for provider in available_providers:
            print(f"\n🔧 Testing {provider}...")
            results["provider_results"][provider] = await self._test_provider_parameters(provider)
        
        # Generate parameter support matrix
        results["parameter_matrix"] = self._generate_parameter_matrix(results["provider_results"])
        
        # Generate compatibility summary
        results["compatibility_summary"] = self._generate_compatibility_summary(results)
        
        return results
    
    async def _test_provider_parameters(self, provider: str) -> Dict[str, Any]:
        """Test parameters for a specific provider"""
        
        results = {
            "temperature_tests": {},
            "max_tokens_tests": {},
            "top_p_tests": {},
            "stop_tests": {},
            "kwargs_tests": {},
            "edge_case_tests": {},
            "overall_score": 0.0
        }
        
        try:
            client = get_client(provider)
            total_tests = 0
            passed_tests = 0
            
            # Test temperature with provider-appropriate values
            print(f"    📊 Testing temperature...")
            temp_values = [0.0, 0.5, 1.0]
            if provider not in ["anthropic", "mistral"]:  # These have max temp of 1.0
                temp_values.append(1.5)
            
            for temp in temp_values:
                total_tests += 1
                valid_temp, is_supported = self._get_valid_parameter_value(provider, "temperature", temp)
                
                if not is_supported:
                    results["temperature_tests"][str(temp)] = {"status": "🚫 Not supported"}
                    continue
                
                try:
                    kwargs = self._build_provider_kwargs(provider, temperature=valid_temp)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        response = await client.create_completion(
                            self.test_messages,
                            **kwargs
                        )
                        
                        if response and response.get("response"):
                            results["temperature_tests"][str(temp)] = {
                                "status": "✅ Pass",
                                "actual_value": valid_temp,
                                "response_length": len(response["response"])
                            }
                            passed_tests += 1
                        else:
                            results["temperature_tests"][str(temp)] = {
                                "status": "⚠️ No response",
                                "actual_value": valid_temp
                            }
                            
                except Exception as e:
                    results["temperature_tests"][str(temp)] = {
                        "status": "❌ Error",
                        "actual_value": valid_temp,
                        "error": str(e)[:100]
                    }
            
            # Test max_tokens (if supported)
            print(f"    📏 Testing max_tokens...")
            if self.provider_constraints.get(provider, {}).get("max_tokens", {}).get("supported", True):
                for max_tok in [10, 50, 100]:
                    total_tests += 1
                    try:
                        kwargs = self._build_provider_kwargs(provider, max_tokens=max_tok, temperature=0.5)
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            response = await client.create_completion(
                                [{"role": "user", "content": "Write a short paragraph about AI"}],
                                **kwargs
                            )
                            
                            if response and response.get("response"):
                                response_text = response["response"]
                                estimated_tokens = len(response_text) // 4
                                
                                results["max_tokens_tests"][str(max_tok)] = {
                                    "status": "✅ Pass",
                                    "response_length": len(response_text),
                                    "estimated_tokens": estimated_tokens,
                                    "respected_limit": estimated_tokens <= max_tok * 1.5
                                }
                                passed_tests += 1
                            else:
                                results["max_tokens_tests"][str(max_tok)] = {"status": "⚠️ No response"}
                                
                    except Exception as e:
                        results["max_tokens_tests"][str(max_tok)] = {
                            "status": "❌ Error",
                            "error": str(e)[:100]
                        }
            else:
                results["max_tokens_tests"]["info"] = "🚫 Parameter not supported by provider"
            
            # Test top_p
            print(f"    🎯 Testing top_p...")
            for top_p in [0.1, 0.9, 1.0]:
                total_tests += 1
                try:
                    kwargs = self._build_provider_kwargs(provider, top_p=top_p, temperature=0.5)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        response = await client.create_completion(
                            self.test_messages,
                            **kwargs
                        )
                        
                        if response and response.get("response"):
                            results["top_p_tests"][str(top_p)] = {"status": "✅ Pass"}
                            passed_tests += 1
                        else:
                            results["top_p_tests"][str(top_p)] = {"status": "⚠️ No response"}
                            
                except Exception as e:
                    results["top_p_tests"][str(top_p)] = {
                        "status": "❌ Error",
                        "error": str(e)[:100]
                    }
            
            # Test stop parameter (if supported)
            print(f"    🛑 Testing stop...")
            if self.provider_constraints.get(provider, {}).get("stop", {}).get("supported", True):
                stop_cases = [None, ".", [".", "!"]]
                for i, stop in enumerate(stop_cases):
                    total_tests += 1
                    try:
                        kwargs = self._build_provider_kwargs(provider, stop=stop, temperature=0.5)
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            response = await client.create_completion(
                                [{"role": "user", "content": "Count: 1, 2, 3, 4, 5."}],
                                **kwargs
                            )
                            
                            if response and response.get("response"):
                                results["stop_tests"][f"case_{i}"] = {
                                    "status": "✅ Pass",
                                    "stop_value": stop,
                                    "response": response["response"][:50] + "..." if len(response["response"]) > 50 else response["response"]
                                }
                                passed_tests += 1
                            else:
                                results["stop_tests"][f"case_{i}"] = {
                                    "status": "⚠️ No response",
                                    "stop_value": stop
                                }
                                
                    except Exception as e:
                        results["stop_tests"][f"case_{i}"] = {
                            "status": "❌ Error",
                            "stop_value": stop,
                            "error": str(e)[:100]
                        }
            else:
                results["stop_tests"]["info"] = "🚫 Parameter not supported by provider"
            
            # Test unknown kwargs handling
            print(f"    📦 Testing unknown kwargs...")
            total_tests += 1
            try:
                base_kwargs = self._build_provider_kwargs(provider, temperature=0.5)
                unknown_kwargs = {
                    "unknown_param": "test",
                    "made_up_setting": 42
                }
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    response = await client.create_completion(
                        self.test_messages,
                        **base_kwargs,
                        **unknown_kwargs
                    )
                    
                    if response and response.get("response"):
                        results["kwargs_tests"]["unknown_params"] = {
                            "status": "✅ Handled gracefully",
                            "message": "Unknown parameters ignored"
                        }
                        passed_tests += 1
                    else:
                        results["kwargs_tests"]["unknown_params"] = {
                            "status": "⚠️ No response"
                        }
                        
            except Exception as e:
                error_str = str(e).lower()
                if any(param in error_str for param in ["unknown_param", "made_up_setting"]):
                    results["kwargs_tests"]["unknown_params"] = {
                        "status": "⚠️ Rejects unknown params",
                        "error": str(e)[:100]
                    }
                else:
                    results["kwargs_tests"]["unknown_params"] = {
                        "status": "❌ Unexpected error",
                        "error": str(e)[:100]
                    }
            
            # Test edge cases with provider-appropriate values
            print(f"    ⚠️  Testing edge cases...")
            edge_cases = []
            
            # Temperature edge cases
            temp_constraints = self.provider_constraints.get(provider, {}).get("temperature", {})
            if temp_constraints:
                min_temp = temp_constraints.get("min", 0.0)
                max_temp = temp_constraints.get("max", 2.0)
                
                # Test below minimum
                edge_cases.append(("temperature", min_temp - 0.1, f"below min ({min_temp})"))
                # Test above maximum  
                edge_cases.append(("temperature", max_temp + 0.1, f"above max ({max_temp})"))
            
            for param, value, description in edge_cases:
                total_tests += 1
                try:
                    kwargs = self._build_provider_kwargs(provider, **{param: value})
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        response = await client.create_completion(
                            self.test_messages,
                            **kwargs
                        )
                        
                        # If this succeeds, provider is lenient
                        results["edge_case_tests"][description] = {
                            "status": "⚠️ Accepted edge value",
                            "value": value
                        }
                        
                except Exception as e:
                    # Expected - provider should reject edge values
                    results["edge_case_tests"][description] = {
                        "status": "✅ Correctly rejected",
                        "value": value,
                        "error": str(e)[:100]
                    }
                    passed_tests += 1
            
            # Calculate overall score
            results["overall_score"] = passed_tests / total_tests if total_tests > 0 else 0.0
            results["tests_passed"] = passed_tests
            results["total_tests"] = total_tests
            
        except Exception as e:
            results["error"] = str(e)
            results["overall_score"] = 0.0
        
        return results
    
    def _generate_parameter_matrix(self, provider_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a parameter support matrix across providers"""
        
        parameters = ["temperature", "max_tokens", "top_p", "stop", "kwargs"]
        matrix = {}
        
        for param in parameters:
            matrix[param] = {}
            
            for provider, results in provider_results.items():
                param_test_key = f"{param}_tests"
                if param_test_key in results:
                    param_results = results[param_test_key]
                    
                    if isinstance(param_results, dict):
                        if "info" in param_results and "not supported" in param_results["info"]:
                            matrix[param][provider] = "🚫 Not supported"
                        else:
                            # Count successful tests
                            total_tests = len([k for k in param_results.keys() if k != "info"])
                            passed_tests = len([k for k, v in param_results.items() 
                                              if isinstance(v, dict) and v.get("status", "").startswith("✅")])
                            
                            if total_tests == 0:
                                matrix[param][provider] = "❓ Not tested"
                            elif passed_tests == total_tests:
                                matrix[param][provider] = "✅ Full support"
                            elif passed_tests > 0:
                                matrix[param][provider] = f"⚠️ Partial ({passed_tests}/{total_tests})"
                            else:
                                matrix[param][provider] = "❌ Not working"
                    else:
                        matrix[param][provider] = "❓ Unknown"
                else:
                    matrix[param][provider] = "❓ Not tested"
        
        return matrix
    
    def _generate_compatibility_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compatibility summary"""
        
        provider_results = results["provider_results"]
        
        summary = {
            "overall_parameter_compatibility": 0.0,
            "provider_scores": {},
            "best_parameter_support": [],
            "parameter_issues": [],
            "recommendations": []
        }
        
        # Calculate provider scores
        for provider, results_data in provider_results.items():
            score = results_data.get("overall_score", 0.0)
            summary["provider_scores"][provider] = score
        
        # Overall compatibility
        if summary["provider_scores"]:
            summary["overall_parameter_compatibility"] = sum(summary["provider_scores"].values()) / len(summary["provider_scores"])
        
        # Best performing providers
        sorted_providers = sorted(summary["provider_scores"].items(), key=lambda x: x[1], reverse=True)
        summary["best_parameter_support"] = [p[0] for p in sorted_providers if p[1] >= 0.8]
        
        # Identify issues
        for provider, score in summary["provider_scores"].items():
            if score < 0.7:
                summary["parameter_issues"].append(f"{provider}: {score:.1%} compatibility")
        
        # Generate recommendations
        overall_score = summary["overall_parameter_compatibility"]
        if overall_score >= 0.85:
            summary["recommendations"].append("✅ Excellent parameter compatibility across providers")
        elif overall_score >= 0.7:
            summary["recommendations"].append("👍 Good parameter compatibility with provider-specific handling")
        else:
            summary["recommendations"].append("⚠️ Parameter compatibility needs improvement")
        
        if summary["best_parameter_support"]:
            summary["recommendations"].append(f"🏆 Best parameter support: {', '.join(summary['best_parameter_support'])}")
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print parameter compatibility results"""
        
        print("\n" + "="*80)
        print("🧪 PROVIDER-AWARE PARAMETER COMPATIBILITY REPORT")
        print("="*80)
        
        summary = results["compatibility_summary"]
        overall_score = summary["overall_parameter_compatibility"]
        
        print(f"📊 Overall Parameter Compatibility: {overall_score:.1%}")
        
        if overall_score >= 0.85:
            print("🎉 EXCELLENT: Parameters work well across providers!")
        elif overall_score >= 0.7:
            print("👍 GOOD: Most parameters work with provider-specific handling")
        else:
            print("⚠️  NEEDS ATTENTION: Significant parameter compatibility issues")
        
        # Provider scores
        print("\n📋 Provider Parameter Scores:")
        for provider, score in summary["provider_scores"].items():
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            print(f"   {status} {provider}: {score:.1%}")
        
        # Parameter support matrix
        print("\n📊 Parameter Support Matrix:")
        matrix = results["parameter_matrix"]
        
        print("   Parameter      | " + " | ".join(f"{p:10s}" for p in summary["provider_scores"].keys()))
        print("   " + "-" * 15 + "|" + "|".join("-" * 12 for _ in summary["provider_scores"]))
        
        for param, provider_support in matrix.items():
            row = f"   {param:14s} | "
            for provider in summary["provider_scores"].keys():
                support = provider_support.get(provider, "❓ Unknown")
                # Truncate status for display
                display_support = support.split()[0] if support else "❓"
                row += f"{display_support:10s} | "
            print(row)
        
        # Recommendations
        print("\n🎯 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   {rec}")
        
        print("="*80)
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"provider_aware_parameter_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Provider-aware parameter report saved to: {filename}")
        return filename


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Provider-aware parameter compatibility testing")
    parser.add_argument("--providers", nargs="*", help="Providers to test")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    providers = args.providers or ["openai", "anthropic", "gemini", "groq", "ollama", "mistral"]
    
    tester = ProviderAwareParameterTester()
    results = await tester.run_provider_aware_test(providers)
    
    # Print results
    tester.print_results(results)
    
    # Save detailed report
    filename = tester.save_results(results, args.output)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())