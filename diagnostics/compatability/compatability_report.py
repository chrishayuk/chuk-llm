# diagnostics/compatibility/compatibility_report.py
"""
Generate comprehensive compatibility reports - Enhanced with warning suppression
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Apply comprehensive warning suppression at the start
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*non-text parts.*')
warnings.filterwarnings('ignore', message='.*function_call.*')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client


class CompatibilityReportGenerator:
    """Generate comprehensive compatibility reports with enhanced error handling"""
    
    def __init__(self):
        self.test_providers = ["openai", "anthropic", "gemini", "groq", "ollama", "mistral"]
        self.required_methods = ["create_completion"]
        self.standard_parameters = {
            "messages": list,
            "tools": (list, type(None)),
            "stream": bool,
            "max_tokens": (int, type(None)),
            "temperature": (float, type(None))
        }
        
        # Simple weather tool for testing
        self.test_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }
    
    async def generate_full_compatibility_report(self, providers: List[str]) -> Dict[str, Any]:
        """Generate complete compatibility report with enhanced error handling"""
        print("🔍 Running comprehensive compatibility analysis...")
        
        # Filter to available providers
        available_providers = [p for p in providers if p in self.test_providers]
        if not available_providers:
            print(f"❌ No valid providers specified. Available: {', '.join(self.test_providers)}")
            return {}
        
        # Run all compatibility tests with warning suppression
        print("  📋 Testing interface consistency...")
        interface_results = await self._test_interface_consistency(available_providers)
        
        print("  📄 Testing response format consistency...")
        format_results = await self._test_response_format_consistency(available_providers)
        
        print("  🔧 Testing tool compatibility...")
        tool_results = await self._test_tool_compatibility(available_providers)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary({
            "interface": interface_results,
            "format": format_results,
            "tools": tool_results
        })
        
        # Create final report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "providers_tested": available_providers,
                "test_version": "1.1"
            },
            "executive_summary": executive_summary,
            "detailed_results": {
                "interface_consistency": interface_results,
                "response_format_consistency": format_results,
                "tool_calling_compatibility": tool_results
            },
            "recommendations": self._generate_recommendations(executive_summary)
        }
        
        return report
    
    async def _test_interface_consistency(self, providers: List[str]) -> Dict[str, Any]:
        """Test interface consistency across providers"""
        results = {}
        
        for provider in providers:
            try:
                client = get_client(provider)
                
                # Test method availability
                method_compat = {}
                for method_name in self.required_methods:
                    has_method = hasattr(client, method_name)
                    is_callable = callable(getattr(client, method_name, None))
                    method_compat[method_name] = has_method and is_callable
                
                # Test parameter compatibility
                param_compat = {}
                method = getattr(client, "create_completion", None)
                if method:
                    sig = inspect.signature(method)
                    for param_name, expected_type in self.standard_parameters.items():
                        param = sig.parameters.get(param_name)
                        if param is None:
                            # Check if it can be passed as **kwargs
                            has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                            param_compat[param_name] = has_kwargs
                        else:
                            param_compat[param_name] = True
                
                results[provider] = {
                    "method_compatibility": method_compat,
                    "parameter_compatibility": param_compat,
                    "overall_score": self._calculate_interface_score(method_compat, param_compat)
                }
                
            except Exception as e:
                results[provider] = {
                    "error": str(e),
                    "overall_score": 0.0
                }
        
        # Calculate overall consistency
        scores = [r.get("overall_score", 0.0) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "provider_results": results,
            "compatibility_analysis": {
                "overall_consistency": avg_score >= 0.9,
                "compatibility_score": avg_score,
                "inconsistencies": [
                    f"{provider}: {result.get('error', 'Interface issues')}"
                    for provider, result in results.items()
                    if result.get("overall_score", 0.0) < 0.9
                ]
            }
        }
    
    def _calculate_interface_score(self, method_compat: Dict, param_compat: Dict) -> float:
        """Calculate interface compatibility score"""
        all_checks = list(method_compat.values()) + list(param_compat.values())
        if not all_checks:
            return 0.0
        return sum(all_checks) / len(all_checks)
    
    async def _test_response_format_consistency(self, providers: List[str]) -> Dict[str, Any]:
        """Test response format consistency with enhanced error handling"""
        results = {}
        
        test_message = [{"role": "user", "content": "Say hello in one word"}]
        
        for provider in providers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    client = get_client(provider)
                    
                    # Test non-streaming response
                    response = await client.create_completion(test_message, stream=False)
                    non_streaming_valid = self._validate_response_format(response)
                    
                    # Test streaming response
                    streaming_valid = await self._test_streaming_format(client, test_message)
                    
                    results[provider] = {
                        "non_streaming": {"valid": non_streaming_valid, "response": response},
                        "streaming": {"valid": streaming_valid},
                        "overall_score": (non_streaming_valid + streaming_valid) / 2
                    }
                
            except Exception as e:
                results[provider] = {
                    "error": str(e),
                    "overall_score": 0.0
                }
        
        # Calculate consistency
        scores = [r.get("overall_score", 0.0) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "provider_results": results,
            "compatibility_analysis": {
                "overall_consistency": avg_score >= 0.9,
                "compatibility_score": avg_score
            }
        }
    
    def _validate_response_format(self, response: Any) -> bool:
        """Validate response format"""
        if not isinstance(response, dict):
            return False
        
        required_fields = ["response", "tool_calls"]
        for field in required_fields:
            if field not in response:
                return False
        
        if not isinstance(response.get("tool_calls"), list):
            return False
        
        return True
    
    async def _test_streaming_format(self, client, messages: List[Dict]) -> bool:
        """Test streaming response format with enhanced error handling"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                stream = client.create_completion(messages, stream=True)
                
                if not hasattr(stream, "__aiter__"):
                    return False
                
                chunk_count = 0
                async for chunk in stream:
                    chunk_count += 1
                    
                    if not isinstance(chunk, dict):
                        return False
                    
                    if "response" not in chunk or "tool_calls" not in chunk:
                        return False
                    
                    if chunk_count >= 3:  # Test a few chunks
                        break
                
                return chunk_count > 0
            
        except Exception:
            return False
    
    async def _test_tool_compatibility(self, providers: List[str]) -> Dict[str, Any]:
        """Test tool calling compatibility with enhanced error handling and bug fixes"""
        results = {}
        
        test_message = [{"role": "user", "content": "What's the weather in London? Use the weather function."}]
        
        for provider in providers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    client = get_client(provider)
                    
                    # Test tool calling with enhanced error handling
                    try:
                        response = await client.create_completion(
                            test_message, 
                            tools=[self.test_tool],
                            stream=False
                        )
                        
                        # Safe tool calls extraction with None handling
                        tool_calls = response.get("tool_calls") if response else None
                        tool_calls = tool_calls if tool_calls is not None else []
                        
                        has_tool_calls = len(tool_calls) > 0
                        
                        # Check if response indicates tools were disabled (Groq fallback)
                        response_text = response.get("response", "") if response else ""
                        groq_fallback = ("Function calling disabled due to provider limitation" in response_text 
                                       if response_text else False)
                        
                        # If Groq fallback occurred, still count as partial success
                        if provider == "groq" and groq_fallback:
                            has_tool_calls = 0.5  # Partial score for graceful degradation
                        
                        # Validate tool call structure if present
                        tool_structure_valid = True
                        if tool_calls and len(tool_calls) > 0:
                            for tool_call in tool_calls:
                                if not self._validate_tool_call_structure(tool_call):
                                    tool_structure_valid = False
                                    break
                        else:
                            # If no tool calls but graceful degradation, partial score
                            tool_structure_valid = 0.5 if groq_fallback else False
                        
                        # Test streaming with tools
                        streaming_tools_valid = await self._test_streaming_tools(client, test_message)
                        
                        results[provider] = {
                            "tool_calling": {
                                "success": has_tool_calls, 
                                "structure_valid": tool_structure_valid,
                                "graceful_degradation": groq_fallback
                            },
                            "streaming_tools": {"success": streaming_tools_valid},
                            "overall_score": (
                                (has_tool_calls if isinstance(has_tool_calls, (int, float)) else int(has_tool_calls)) + 
                                (tool_structure_valid if isinstance(tool_structure_valid, (int, float)) else int(tool_structure_valid)) + 
                                int(streaming_tools_valid)
                            ) / 3
                        }
                        
                    except Exception as tool_error:
                        error_str = str(tool_error)
                        
                        # Handle specific error types
                        if "argument of type 'NoneType' is not iterable" in error_str:
                            # This is likely a bug in our validation code
                            results[provider] = {
                                "tool_calling": {"success": False, "reason": "validation_bug"},
                                "streaming_tools": {"success": False, "reason": "validation_bug"},
                                "overall_score": 0.0,
                                "error": "Tool validation error (internal bug)"
                            }
                        elif provider == "groq" and "Failed to call a function" in error_str:
                            results[provider] = {
                                "tool_calling": {
                                    "success": False, 
                                    "reason": "groq_function_error",
                                    "needs_enhancement": True
                                },
                                "streaming_tools": {"success": False, "reason": "groq_function_error"},
                                "overall_score": 0.3  # Some credit for attempting
                            }
                        else:
                            # Other tool errors
                            results[provider] = {
                                "tool_calling": {"success": False, "reason": "tool_error"},
                                "streaming_tools": {"success": False, "reason": "tool_error"},
                                "overall_score": 0.0,
                                "error": error_str
                            }
                
            except Exception as e:
                # General provider errors
                error_str = str(e)
                results[provider] = {
                    "error": error_str,
                    "overall_score": 0.0
                }
        
        # Calculate compatibility
        scores = [r.get("overall_score", 0.0) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "provider_results": results,
            "compatibility_analysis": {
                "compatibility_score": avg_score,
                "providers_supporting_tools": [
                    provider for provider, result in results.items()
                    if result.get("tool_calling", {}).get("success", False)
                ],
                "providers_needing_enhancement": [
                    provider for provider, result in results.items()
                    if result.get("tool_calling", {}).get("needs_enhancement", False)
                ]
            }
        }
    
    def _validate_tool_call_structure(self, tool_call: Any) -> bool:
        """Validate tool call structure"""
        if not isinstance(tool_call, dict):
            return False
        
        required_fields = ["id", "type", "function"]
        for field in required_fields:
            if field not in tool_call:
                return False
        
        func = tool_call.get("function", {})
        if not isinstance(func, dict):
            return False
        
        if "name" not in func or "arguments" not in func:
            return False
        
        # Validate arguments is JSON
        try:
            json.loads(func["arguments"])
        except (json.JSONDecodeError, TypeError):
            return False
        
        return True
    
    async def _test_streaming_tools(self, client, messages: List[Dict]) -> bool:
        """Test streaming with tools"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                stream = client.create_completion(
                    messages,
                    tools=[self.test_tool],
                    stream=True
                )
                
                chunk_count = 0
                found_tool_calls = False
                
                async for chunk in stream:
                    chunk_count += 1
                    if chunk.get("tool_calls"):
                        found_tool_calls = True
                        break
                    
                    if chunk_count > 20:  # Limit chunks
                        break
                
                return found_tool_calls
                
        except Exception:
            return False
    
    def _generate_executive_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary with enhanced analysis"""
        summary = {
            "overall_compatibility_score": 0.0,
            "fully_compatible_providers": [],
            "providers_with_issues": [],
            "providers_needing_enhancement": [],
            "test_results_summary": {}
        }
        
        # Calculate area scores
        area_scores = {}
        for area, results in all_results.items():
            if isinstance(results, dict) and "compatibility_analysis" in results:
                analysis = results["compatibility_analysis"]
                score = analysis.get("compatibility_score", 0.0)
                area_scores[area] = score
                summary["test_results_summary"][area] = {
                    "score": score,
                    "status": "✅ Pass" if score >= 0.8 else "⚠️ Issues" if score >= 0.5 else "❌ Fail"
                }
        
        # Overall score
        if area_scores:
            summary["overall_compatibility_score"] = sum(area_scores.values()) / len(area_scores)
        
        # Identify provider issues and enhancements needed
        all_providers = set()
        for results in all_results.values():
            if isinstance(results, dict) and "provider_results" in results:
                all_providers.update(results["provider_results"].keys())
        
        for provider in all_providers:
            scores = []
            issues = []
            needs_enhancement = False
            
            for area, results in all_results.items():
                if isinstance(results, dict) and "provider_results" in results:
                    provider_result = results["provider_results"].get(provider, {})
                    score = provider_result.get("overall_score", 0.0)
                    scores.append(score)
                    
                    if "error" in provider_result:
                        issues.append(f"{area}: {provider_result['error']}")
                    elif score < 0.8:
                        issues.append(f"{area}: Low compatibility ({score:.1%})")
                    
                    # Check for enhancement needs (e.g., Groq function calling)
                    if (area == "tools" and 
                        provider_result.get("tool_calling", {}).get("needs_enhancement", False)):
                        needs_enhancement = True
            
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= 0.9:
                    summary["fully_compatible_providers"].append(provider)
                elif avg_score < 0.7 or needs_enhancement:
                    provider_info = {
                        "provider": provider,
                        "score": avg_score,
                        "issues": issues
                    }
                    if needs_enhancement:
                        provider_info["enhancement_needed"] = True
                        summary["providers_needing_enhancement"].append(provider)
                    summary["providers_with_issues"].append(provider_info)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations with enhanced guidance"""
        recommendations = []
        
        overall_score = summary.get("overall_compatibility_score", 0.0)
        
        if overall_score >= 0.95:
            recommendations.append("✅ Excellent compatibility! All providers work consistently.")
        elif overall_score >= 0.8:
            recommendations.append("👍 Good compatibility with minor issues to address.")
        else:
            recommendations.append("⚠️ Significant compatibility issues need attention.")
        
        fully_compatible = summary.get("fully_compatible_providers", [])
        if fully_compatible:
            recommendations.append(f"🏆 Fully compatible providers: {', '.join(fully_compatible)}")
        
        providers_with_issues = summary.get("providers_with_issues", [])
        if providers_with_issues:
            issue_providers = [p["provider"] if isinstance(p, dict) else p for p in providers_with_issues]
            recommendations.append(f"🔧 Focus on fixing: {', '.join(issue_providers)}")
        
        providers_needing_enhancement = summary.get("providers_needing_enhancement", [])
        if providers_needing_enhancement:
            recommendations.append(f"🚀 Consider enhancements for: {', '.join(providers_needing_enhancement)}")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compatibility_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Compatibility report saved to: {filename}")
        return filename
    
    def print_summary(self, report: Dict[str, Any]):
        """Print report summary with enhanced formatting"""
        summary = report["executive_summary"]
        
        print("\n" + "="*80)
        print("🔍 PROVIDER COMPATIBILITY REPORT")
        print("="*80)
        
        score = summary["overall_compatibility_score"]
        print(f"📊 Overall Compatibility Score: {score:.1%}")
        
        if score >= 0.95:
            print("🎉 EXCELLENT: All providers are highly compatible!")
        elif score >= 0.8:
            print("👍 GOOD: Minor compatibility issues detected")
        else:
            print("⚠️  NEEDS ATTENTION: Significant compatibility issues")
        
        # Test results summary
        print("\n📋 Test Results:")
        test_summary = summary.get("test_results_summary", {})
        for area, result in test_summary.items():
            print(f"   {area}: {result['status']} ({result['score']:.1%})")
        
        # Provider status
        fully_compatible = summary.get("fully_compatible_providers", [])
        if fully_compatible:
            print(f"\n✅ Fully Compatible: {', '.join(fully_compatible)}")
        
        providers_with_issues = summary.get("providers_with_issues", [])
        if providers_with_issues:
            print("\n❌ Providers with Issues:")
            for provider_info in providers_with_issues:
                if isinstance(provider_info, dict):
                    provider = provider_info["provider"]
                    score = provider_info["score"]
                    print(f"   • {provider}: {score:.1%} compatibility")
                    if provider_info.get("enhancement_needed"):
                        print(f"     ⚡ Enhancement available")
                    for issue in provider_info["issues"][:2]:  # Show first 2 issues
                        print(f"     - {issue}")
                else:
                    print(f"   • {provider_info}")
        
        providers_needing_enhancement = summary.get("providers_needing_enhancement", [])
        if providers_needing_enhancement:
            print(f"\n⚡ Enhancement Opportunities: {', '.join(providers_needing_enhancement)}")
        
        print("\n🎯 Recommendations:")
        for recommendation in report["recommendations"]:
            print(f"   {recommendation}")
        
        print("="*80)


async def main():
    """CLI entry point with enhanced error handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test provider compatibility")
    parser.add_argument("--providers", nargs="*", help="Providers to test")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    providers = args.providers or ["openai", "anthropic", "gemini", "groq", "mistral", "ollama"]
    
    generator = CompatibilityReportGenerator()
    report = await generator.generate_full_compatibility_report(providers)
    
    if report:  # Only proceed if we got a valid report
        # Print summary
        generator.print_summary(report)
        
        # Save detailed report
        filename = generator.save_report(report, args.output)
    
    return report


if __name__ == "__main__":
    asyncio.run(main())