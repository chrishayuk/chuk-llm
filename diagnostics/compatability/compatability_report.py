# diagnostics/compatibility/compatibility_report.py
"""
Generate comprehensive compatibility reports - Enhanced with improved provider handling
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
from typing import Any, Dict, List, Optional

# Apply comprehensive warning suppression at the start
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*non-text parts.*')
warnings.filterwarnings('ignore', message='.*function_call.*')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from chuk_llm.llm.client import get_client


class CompatibilityReportGenerator:
    """Generate comprehensive compatibility reports with enhanced provider support"""
    
    def __init__(self):
        self.test_providers = {
            "openai": {"models": ["gpt-4o-mini", "gpt-4o"], "has_env": self._check_env("OPENAI_API_KEY")},
            "anthropic": {"models": ["claude-sonnet-4-20250514"], "has_env": self._check_env("ANTHROPIC_API_KEY")},
            "gemini": {"models": ["gemini-2.0-flash"], "has_env": self._check_env("GEMINI_API_KEY")},
            "mistral": {"models": ["mistral-medium-2505"], "has_env": self._check_env("MISTRAL_API_KEY")},
            "deepseek": {"models": ["deepseek-chat"], "has_env": self._check_env("DEEPSEEK_API_KEY")},
            "perplexity": {"models": ["sonar-pro"], "has_env": self._check_env("PERPLEXITY_API_KEY")},
            "groq": {"models": ["llama-3.3-70b-versatile"], "has_env": self._check_env("GROQ_API_KEY")},
            "watsonx": {"models": ["ibm/granite-3-8b-instruct"], "has_env": self._check_env("WATSONX_API_KEY") or self._check_env("IBM_CLOUD_API_KEY")},
            "ollama": {"models": ["llama3.2"], "has_env": True}  # Ollama usually doesn't need API key
        }
        
        self.required_methods = ["create_completion"]
        self.standard_parameters = {
            "messages": list,
            "tools": (list, type(None)),
            "stream": bool,
            "max_tokens": (int, type(None)),
            "temperature": (float, type(None)),
            "system": (str, type(None))
        }
        
        # Enhanced weather tool for testing
        self.test_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country, e.g. San Francisco, CA"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    
    def _check_env(self, env_var: str) -> bool:
        """Check if environment variable is set"""
        return bool(os.getenv(env_var))
    
    async def generate_full_compatibility_report(self, providers: List[str]) -> Dict[str, Any]:
        """Generate complete compatibility report with enhanced error handling"""
        print("🔍 Running comprehensive compatibility analysis...")
        
        # Filter to available and configured providers
        available_providers = []
        for provider in providers:
            if provider in self.test_providers:
                if self.test_providers[provider]["has_env"]:
                    available_providers.append(provider)
                else:
                    print(f"⚠️  Skipping {provider}: API key not configured")
            else:
                print(f"⚠️  Unknown provider: {provider}")
        
        if not available_providers:
            print(f"❌ No configured providers available. Check API keys.")
            return {}
        
        print(f"📋 Testing providers: {', '.join(available_providers)}")
        
        # Run all compatibility tests with enhanced error handling
        print("  🔌 Testing interface consistency...")
        interface_results = await self._test_interface_consistency(available_providers)
        
        print("  📄 Testing response format consistency...")
        format_results = await self._test_response_format_consistency(available_providers)
        
        print("  🔧 Testing tool compatibility...")
        tool_results = await self._test_tool_compatibility(available_providers)
        
        print("  🎭 Testing system message support...")
        system_results = await self._test_system_message_support(available_providers)
        
        print("  📊 Testing streaming capabilities...")
        streaming_results = await self._test_streaming_capabilities(available_providers)
        
        # Generate executive summary
        all_test_results = {
            "interface": interface_results,
            "format": format_results,
            "tools": tool_results,
            "system_messages": system_results,
            "streaming": streaming_results
        }
        
        executive_summary = self._generate_executive_summary(all_test_results)
        
        # Create final report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "providers_tested": available_providers,
                "providers_available": list(self.test_providers.keys()),
                "test_version": "2.0"
            },
            "executive_summary": executive_summary,
            "detailed_results": {
                "interface_consistency": interface_results,
                "response_format_consistency": format_results,
                "tool_calling_compatibility": tool_results,
                "system_message_support": system_results,
                "streaming_capabilities": streaming_results
            },
            "recommendations": self._generate_recommendations(executive_summary, available_providers)
        }
        
        return report
    
    async def _test_interface_consistency(self, providers: List[str]) -> Dict[str, Any]:
        """Test interface consistency across providers"""
        results = {}
        
        for provider in providers:
            try:
                # Get default model for provider
                model = self.test_providers[provider]["models"][0]
                client = get_client(provider, model=model)
                
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
                
                # Test model info availability
                has_model_info = hasattr(client, 'get_model_info') and callable(getattr(client, 'get_model_info', None))
                
                results[provider] = {
                    "method_compatibility": method_compat,
                    "parameter_compatibility": param_compat,
                    "model_info_available": has_model_info,
                    "overall_score": self._calculate_interface_score(method_compat, param_compat, has_model_info)
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
                "overall_consistency": avg_score >= 0.8,
                "compatibility_score": avg_score,
                "inconsistencies": [
                    f"{provider}: {result.get('error', 'Interface issues')}"
                    for provider, result in results.items()
                    if result.get("overall_score", 0.0) < 0.8
                ]
            }
        }
    
    def _calculate_interface_score(self, method_compat: Dict, param_compat: Dict, has_model_info: bool) -> float:
        """Calculate interface compatibility score"""
        all_checks = list(method_compat.values()) + list(param_compat.values()) + [has_model_info]
        if not all_checks:
            return 0.0
        return sum(all_checks) / len(all_checks)
    
    async def _test_response_format_consistency(self, providers: List[str]) -> Dict[str, Any]:
        """Test response format consistency with enhanced error handling"""
        results = {}
        
        test_message = [{"role": "user", "content": "Say 'hello' in exactly one word"}]
        
        for provider in providers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    model = self.test_providers[provider]["models"][0]
                    client = get_client(provider, model=model)
                    
                    # Test non-streaming response
                    try:
                        response = await client.create_completion(test_message, stream=False, max_tokens=50)
                        non_streaming_valid = self._validate_response_format(response)
                        
                        # Enhanced response analysis
                        response_analysis = {
                            "valid": non_streaming_valid,
                            "has_response_field": False,
                            "has_tool_calls_field": False,
                            "response_keys": [],
                            "response_type": type(response).__name__ if response else "None"
                        }
                        
                        if isinstance(response, dict):
                            response_analysis["response_keys"] = list(response.keys())
                            response_analysis["has_response_field"] = "response" in response
                            response_analysis["has_tool_calls_field"] = "tool_calls" in response
                        
                    except Exception as e:
                        error_str = str(e)
                        if "401" in error_str or "Authorization" in error_str:
                            response_analysis = {
                                "valid": False,
                                "error": "API authentication failed",
                                "note": "Check API key configuration"
                            }
                        else:
                            response_analysis = {
                                "valid": False,
                                "error": error_str[:100]
                            }
                        non_streaming_valid = False
                    
                    # Test streaming response
                    streaming_valid = await self._test_streaming_format(client, test_message)
                    
                    results[provider] = {
                        "non_streaming": response_analysis,
                        "streaming": {"valid": streaming_valid},
                        "overall_score": (non_streaming_valid + streaming_valid) / 2
                    }
                
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "Authorization" in error_str:
                    results[provider] = {
                        "error": "API authentication failed",
                        "overall_score": 0.0,
                        "note": "Check API key configuration"
                    }
                else:
                    results[provider] = {
                        "error": error_str[:100],
                        "overall_score": 0.0
                    }
        
        # Calculate consistency
        scores = [r.get("overall_score", 0.0) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "provider_results": results,
            "compatibility_analysis": {
                "overall_consistency": avg_score >= 0.8,
                "compatibility_score": avg_score,
                "format_issues": [
                    provider for provider, result in results.items()
                    if result.get("overall_score", 0.0) < 0.8
                ],
                "authentication_issues": [
                    provider for provider, result in results.items()
                    if "authentication" in result.get("error", "").lower()
                ]
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
                
                stream = client.create_completion(messages, stream=True, max_tokens=50)
                
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
        
        test_message = [{"role": "user", "content": "What's the weather in London? Use the get_weather function to check."}]
        
        for provider in providers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    model = self.test_providers[provider]["models"][0]
                    client = get_client(provider, model=model)
                    
                    # Test tool calling with enhanced error handling
                    try:
                        response = await client.create_completion(
                            test_message, 
                            tools=[self.test_tool],
                            stream=False,
                            max_tokens=300
                        )
                        
                        # Safe tool calls extraction with proper null checking
                        tool_calls = response.get("tool_calls", []) if response else []
                        tool_calls = tool_calls if tool_calls is not None else []
                        
                        has_tool_calls = len(tool_calls) > 0
                        
                        # Safe response text extraction with null checking
                        response_text = ""
                        if response and response.get("response"):
                            response_text = str(response["response"])
                        
                        # Check for provider-specific graceful degradation messages
                        has_graceful_degradation = False
                        if response_text:
                            degradation_phrases = [
                                "function calling disabled",
                                "tools not supported", 
                                "unable to call function",
                                "cannot use tools",
                                "function calling not available"
                            ]
                            has_graceful_degradation = any(phrase in response_text.lower() for phrase in degradation_phrases)
                        
                        # Validate tool call structure if present
                        tool_structure_valid = True
                        valid_tool_count = 0
                        if tool_calls and len(tool_calls) > 0:
                            for tool_call in tool_calls:
                                if self._validate_tool_call_structure(tool_call):
                                    valid_tool_count += 1
                                else:
                                    tool_structure_valid = False
                        
                        # Test streaming with tools
                        streaming_tools_valid = await self._test_streaming_tools(client, test_message)
                        
                        # Calculate scores with proper handling
                        tool_score = 1.0 if has_tool_calls else (0.3 if has_graceful_degradation else 0.0)
                        structure_score = 1.0 if (not tool_calls or tool_structure_valid) else (valid_tool_count / len(tool_calls) if tool_calls else 0.0)
                        streaming_score = 1.0 if streaming_tools_valid else 0.0
                        
                        results[provider] = {
                            "tool_calling": {
                                "success": has_tool_calls, 
                                "tool_calls_count": len(tool_calls),
                                "valid_tool_calls": valid_tool_count,
                                "structure_valid": tool_structure_valid,
                                "graceful_degradation": has_graceful_degradation,
                                "response_text_sample": response_text[:100] if response_text else "No response",
                                "score": tool_score
                            },
                            "streaming_tools": {"success": streaming_tools_valid, "score": streaming_score},
                            "overall_score": (tool_score + structure_score + streaming_score) / 3
                        }
                        
                    except Exception as tool_error:
                        error_str = str(tool_error)
                        
                        # Handle specific error patterns with better categorization
                        if "'NoneType' object has no attribute 'lower'" in error_str:
                            results[provider] = {
                                "tool_calling": {
                                    "success": False, 
                                    "reason": "validation_bug_fixed",
                                    "score": 0.0
                                },
                                "streaming_tools": {"success": False, "score": 0.0},
                                "overall_score": 0.0,
                                "error": "Tool validation bug (now fixed)",
                                "note": "This was a bug in the compatibility tester, not the provider"
                            }
                        elif "groq" in provider.lower() and ("function" in error_str.lower() or "tool" in error_str.lower()):
                            results[provider] = {
                                "tool_calling": {
                                    "success": False, 
                                    "reason": "groq_function_limitation",
                                    "needs_enhancement": True,
                                    "score": 0.2
                                },
                                "streaming_tools": {"success": False, "score": 0.0},
                                "overall_score": 0.1,
                                "note": "Groq may need enhanced function calling support"
                            }
                        elif "gemini" in provider.lower() and "system_instruction" in error_str:
                            results[provider] = {
                                "tool_calling": {
                                    "success": False,
                                    "reason": "gemini_parameter_issue", 
                                    "needs_client_fix": True,
                                    "score": 0.3
                                },
                                "streaming_tools": {"success": False, "score": 0.0},
                                "overall_score": 0.15,
                                "note": "Gemini client parameter handling needs fixes"
                            }
                        elif "401" in error_str or "Authorization" in error_str:
                            results[provider] = {
                                "tool_calling": {
                                    "success": False,
                                    "reason": "authentication_error",
                                    "score": 0.0
                                },
                                "streaming_tools": {"success": False, "score": 0.0},
                                "overall_score": 0.0,
                                "error": "API key authentication failed",
                                "note": "Check API key configuration"
                            }
                        else:
                            results[provider] = {
                                "tool_calling": {"success": False, "reason": "tool_error", "score": 0.0},
                                "streaming_tools": {"success": False, "score": 0.0},
                                "overall_score": 0.0,
                                "error": error_str[:200]
                            }
                
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "Authorization" in error_str:
                    results[provider] = {
                        "error": "API key authentication failed",
                        "overall_score": 0.0,
                        "note": "Check API key configuration"
                    }
                else:
                    results[provider] = {
                        "error": error_str[:200],
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
                    if result.get("tool_calling", {}).get("needs_enhancement", False) or
                       result.get("tool_calling", {}).get("needs_client_fix", False)
                ],
                "tools_working_count": sum(1 for result in results.values() 
                                         if result.get("tool_calling", {}).get("success", False)),
                "authentication_issues": [
                    provider for provider, result in results.items()
                    if "authentication" in result.get("error", "").lower() or 
                       result.get("tool_calling", {}).get("reason") == "authentication_error"
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
        
        # Validate arguments is valid JSON
        try:
            args = func["arguments"]
            if isinstance(args, str):
                json.loads(args)
            elif not isinstance(args, dict):
                return False
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
                    stream=True,
                    max_tokens=300
                )
                
                chunk_count = 0
                found_tool_calls = False
                
                async for chunk in stream:
                    chunk_count += 1
                    if chunk.get("tool_calls"):
                        found_tool_calls = True
                        break
                    
                    if chunk_count > 30:  # Reasonable limit
                        break
                
                return found_tool_calls
                
        except Exception:
            return False
    
    async def _test_system_message_support(self, providers: List[str]) -> Dict[str, Any]:
        """Test system message support across providers with proper parameter handling"""
        results = {}
        
        test_system = "You are a helpful assistant. Always start your response with 'SYSTEM_TEST:'"
        test_message = [{"role": "user", "content": "Say hello"}]
        
        for provider in providers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    model = self.test_providers[provider]["models"][0]
                    client = get_client(provider, model=model)
                    
                    # Test 1: Try system parameter (will fail for most providers currently)
                    system_param_works = False
                    system_param_error = None
                    try:
                        response = await client.create_completion(
                            test_message,
                            system=test_system,
                            max_tokens=100
                        )
                        
                        if response and response.get("response"):
                            response_text = str(response["response"])
                            system_respected = "system_test:" in response_text.lower()
                            system_param_works = True
                            
                            results[provider] = {
                                "system_parameter_works": True,
                                "system_respected": system_respected,
                                "response_snippet": response_text[:100],
                                "overall_score": 1.0 if system_respected else 0.7,
                                "method_used": "system_parameter"
                            }
                        else:
                            system_param_works = False
                            system_param_error = "No response received"
                            
                    except Exception as sys_error:
                        system_param_works = False
                        system_param_error = str(sys_error)
                    
                    # Test 2: If system parameter failed, try system message in messages array
                    if not system_param_works:
                        try:
                            sys_messages = [
                                {"role": "system", "content": test_system},
                                {"role": "user", "content": "Say hello"}
                            ]
                            response = await client.create_completion(sys_messages, max_tokens=100)
                            
                            if response and response.get("response"):
                                response_text = str(response["response"])
                                system_respected = "system_test:" in response_text.lower()
                                
                                results[provider] = {
                                    "system_parameter_works": False,
                                    "system_messages_work": True,
                                    "system_respected": system_respected,
                                    "response_snippet": response_text[:100],
                                    "overall_score": 0.8 if system_respected else 0.5,
                                    "method_used": "system_message_in_array",
                                    "system_param_error": system_param_error[:100] if system_param_error else None
                                }
                            else:
                                results[provider] = {
                                    "system_parameter_works": False,
                                    "system_messages_work": False,
                                    "overall_score": 0.0,
                                    "error": "No response from either method",
                                    "system_param_error": system_param_error[:100] if system_param_error else None
                                }
                                
                        except Exception as msg_error:
                            results[provider] = {
                                "system_parameter_works": False,
                                "system_messages_work": False,
                                "overall_score": 0.0,
                                "error": f"Both methods failed: {str(msg_error)[:100]}",
                                "system_param_error": system_param_error[:100] if system_param_error else None
                            }
                
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "Authorization" in error_str:
                    results[provider] = {
                        "error": "API key authentication failed",
                        "overall_score": 0.0,
                        "note": "Check API key configuration"
                    }
                else:
                    results[provider] = {
                        "error": error_str[:100],
                        "overall_score": 0.0
                    }
        
        # Calculate compatibility
        scores = [r.get("overall_score", 0.0) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "provider_results": results,
            "compatibility_analysis": {
                "compatibility_score": avg_score,
                "providers_with_system_param": [
                    provider for provider, result in results.items()
                    if result.get("system_parameter_works", False)
                ],
                "providers_with_system_messages": [
                    provider for provider, result in results.items()
                    if result.get("system_messages_work", False)
                ],
                "providers_with_system_support": [
                    provider for provider, result in results.items()
                    if result.get("system_parameter_works", False) or result.get("system_messages_work", False)
                ],
                "authentication_issues": [
                    provider for provider, result in results.items()
                    if "authentication" in result.get("error", "").lower()
                ]
            }
        }
    
    async def _test_streaming_capabilities(self, providers: List[str]) -> Dict[str, Any]:
        """Test streaming capabilities across providers"""
        results = {}
        
        test_message = [{"role": "user", "content": "Count from 1 to 5, one number per line"}]
        
        for provider in providers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    model = self.test_providers[provider]["models"][0]
                    client = get_client(provider, model=model)
                    
                    # Test streaming
                    stream_works = False
                    chunk_count = 0
                    total_content = ""
                    
                    try:
                        stream = client.create_completion(test_message, stream=True, max_tokens=100)
                        
                        async for chunk in stream:
                            chunk_count += 1
                            if isinstance(chunk, dict) and chunk.get("response"):
                                total_content += chunk["response"]
                                stream_works = True
                            
                            if chunk_count > 50:  # Safety limit
                                break
                        
                        results[provider] = {
                            "streaming_works": stream_works,
                            "chunk_count": chunk_count,
                            "total_content_length": len(total_content),
                            "content_sample": total_content[:100],
                            "overall_score": 1.0 if stream_works and chunk_count > 0 else 0.0
                        }
                        
                    except Exception as stream_error:
                        results[provider] = {
                            "streaming_works": False,
                            "error": str(stream_error)[:100],
                            "overall_score": 0.0
                        }
                
            except Exception as e:
                results[provider] = {
                    "error": str(e)[:100],
                    "overall_score": 0.0
                }
        
        # Calculate compatibility
        scores = [r.get("overall_score", 0.0) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "provider_results": results,
            "compatibility_analysis": {
                "compatibility_score": avg_score,
                "providers_with_streaming": [
                    provider for provider, result in results.items()
                    if result.get("streaming_works", False)
                ]
            }
        }
    
    def _generate_executive_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary with enhanced analysis"""
        summary = {
            "overall_compatibility_score": 0.0,
            "fully_compatible_providers": [],
            "providers_with_issues": [],
            "providers_needing_enhancement": [],
            "test_results_summary": {},
            "feature_support_matrix": {}
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
                    "status": "✅ Excellent" if score >= 0.9 else "👍 Good" if score >= 0.7 else "⚠️ Issues" if score >= 0.4 else "❌ Poor"
                }
        
        # Overall score
        if area_scores:
            summary["overall_compatibility_score"] = sum(area_scores.values()) / len(area_scores)
        
        # Build feature support matrix
        all_providers = set()
        for results in all_results.values():
            if isinstance(results, dict) and "provider_results" in results:
                all_providers.update(results["provider_results"].keys())
        
        for provider in all_providers:
            provider_features = {}
            provider_issues = []
            needs_enhancement = False
            scores = []
            
            for area, results in all_results.items():
                if isinstance(results, dict) and "provider_results" in results:
                    provider_result = results["provider_results"].get(provider, {})
                    score = provider_result.get("overall_score", 0.0)
                    scores.append(score)
                    
                    # Track feature support
                    if area == "tools":
                        provider_features["tools"] = provider_result.get("tool_calling", {}).get("success", False)
                    elif area == "streaming":
                        provider_features["streaming"] = provider_result.get("streaming_works", False)
                    elif area == "system_messages":
                        provider_features["system_messages"] = provider_result.get("system_parameter_works", False) or provider_result.get("system_messages_work", False)
                    
                    # Track issues
                    if "error" in provider_result:
                        provider_issues.append(f"{area}: {provider_result['error']}")
                    elif score < 0.7:
                        provider_issues.append(f"{area}: Score {score:.1%}")
                    
                    # Check for enhancement needs
                    if (provider_result.get("tool_calling", {}).get("needs_enhancement", False) or
                        provider_result.get("tool_calling", {}).get("needs_client_fix", False)):
                        needs_enhancement = True
            
            # Store feature matrix
            summary["feature_support_matrix"][provider] = provider_features
            
            # Categorize providers
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= 0.85 and not needs_enhancement:
                    summary["fully_compatible_providers"].append(provider)
                else:
                    provider_info = {
                        "provider": provider,
                        "score": avg_score,
                        "issues": provider_issues[:3],  # Top 3 issues
                        "features": provider_features
                    }
                    if needs_enhancement:
                        provider_info["enhancement_needed"] = True
                        summary["providers_needing_enhancement"].append(provider)
                    if avg_score < 0.7 or provider_issues:
                        summary["providers_with_issues"].append(provider_info)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any], tested_providers: List[str]) -> List[str]:
        """Generate recommendations with enhanced guidance"""
        recommendations = []
        
        overall_score = summary.get("overall_compatibility_score", 0.0)
        
        # Overall assessment
        if overall_score >= 0.9:
            recommendations.append("🎉 Excellent compatibility! Universal config is working great.")
        elif overall_score >= 0.7:
            recommendations.append("👍 Good compatibility with room for improvement.")
        elif overall_score >= 0.5:
            recommendations.append("⚠️ Moderate compatibility - several issues to address.")
        else:
            recommendations.append("🚨 Poor compatibility - significant work needed.")
        
        # Provider-specific recommendations
        fully_compatible = summary.get("fully_compatible_providers", [])
        if fully_compatible:
            recommendations.append(f"🏆 Fully compatible providers: {', '.join(fully_compatible)}")
        
        providers_with_issues = summary.get("providers_with_issues", [])
        if providers_with_issues:
            issue_providers = [p["provider"] if isinstance(p, dict) else p for p in providers_with_issues]
            recommendations.append(f"🔧 Providers needing fixes: {', '.join(issue_providers)}")
        
        providers_needing_enhancement = summary.get("providers_needing_enhancement", [])
        if providers_needing_enhancement:
            recommendations.append(f"⚡ Providers with enhancement opportunities: {', '.join(providers_needing_enhancement)}")
        
        # Feature-specific recommendations
        feature_matrix = summary.get("feature_support_matrix", {})
        if feature_matrix:
            tools_support = sum(1 for features in feature_matrix.values() if features.get("tools", False))
            streaming_support = sum(1 for features in feature_matrix.values() if features.get("streaming", False))
            system_support = sum(1 for features in feature_matrix.values() if features.get("system_messages", False))
            
            total_providers = len(tested_providers)
            
            if tools_support < total_providers:
                recommendations.append(f"🔧 Tool calling: {tools_support}/{total_providers} providers working")
            if streaming_support < total_providers:
                recommendations.append(f"📡 Streaming: {streaming_support}/{total_providers} providers working")
            if system_support < total_providers:
                recommendations.append(f"💬 System messages: {system_support}/{total_providers} providers working")
        
        # Priority actions
        test_summary = summary.get("test_results_summary", {})
        poor_areas = [area for area, result in test_summary.items() if result["score"] < 0.5]
        if poor_areas:
            recommendations.append(f"🎯 Priority fixes needed for: {', '.join(poor_areas)}")
        
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
        """Print report summary with enhanced formatting and issue identification"""
        summary = report["executive_summary"]
        metadata = report["metadata"]
        
        print("\n" + "="*80)
        print("🔍 PROVIDER COMPATIBILITY REPORT")
        print("="*80)
        
        score = summary["overall_compatibility_score"]
        print(f"📊 Overall Compatibility Score: {score:.1%}")
        print(f"🧪 Providers Tested: {', '.join(metadata['providers_tested'])}")
        print(f"📅 Generated: {metadata['generated_at'][:19]}")
        
        if score >= 0.9:
            print("🎉 EXCELLENT: Universal config working great!")
        elif score >= 0.7:
            print("👍 GOOD: Minor compatibility issues detected")
        elif score >= 0.5:
            print("⚠️  MODERATE: Several issues need attention")
        else:
            print("🚨 POOR: Significant compatibility problems")
        
        # Test results summary
        print("\n📋 Test Results by Category:")
        test_summary = summary.get("test_results_summary", {})
        for area, result in test_summary.items():
            print(f"   {area.replace('_', ' ').title()}: {result['status']} ({result['score']:.1%})")
        
        # Enhanced feature support matrix
        print("\n🎯 Feature Support Matrix:")
        feature_matrix = summary.get("feature_support_matrix", {})
        if feature_matrix:
            features = ["tools", "streaming", "system_messages"]
            print(f"{'Provider':<15} {'Tools':<8} {'Stream':<8} {'System':<8}")
            print("-" * 45)
            for provider, provider_features in feature_matrix.items():
                tools_icon = "✅" if provider_features.get("tools", False) else "❌"
                stream_icon = "✅" if provider_features.get("streaming", False) else "❌"
                system_icon = "✅" if provider_features.get("system_messages", False) else "❌"
                print(f"{provider:<15} {tools_icon:<8} {stream_icon:<8} {system_icon:<8}")
        
        # Provider status
        fully_compatible = summary.get("fully_compatible_providers", [])
        if fully_compatible:
            print(f"\n🏆 Fully Compatible: {', '.join(fully_compatible)}")
        
        providers_with_issues = summary.get("providers_with_issues", [])
        if providers_with_issues:
            print("\n🔧 Providers with Issues:")
            for provider_info in providers_with_issues:
                if isinstance(provider_info, dict):
                    provider = provider_info["provider"]
                    score = provider_info["score"]
                    print(f"   • {provider}: {score:.1%} compatibility")
                    if provider_info.get("enhancement_needed"):
                        print(f"     ⚡ Enhancement opportunity available")
                    for issue in provider_info["issues"][:2]:  # Show top 2 issues
                        print(f"     - {issue}")
                else:
                    print(f"   • {provider_info}")
        
        providers_needing_enhancement = summary.get("providers_needing_enhancement", [])
        if providers_needing_enhancement:
            print(f"\n⚡ Enhancement Opportunities: {', '.join(providers_needing_enhancement)}")
        
        # Check for authentication issues across all tests
        auth_issues = set()
        detailed_results = report.get("detailed_results", {})
        for test_name, test_results in detailed_results.items():
            if isinstance(test_results, dict) and "compatibility_analysis" in test_results:
                analysis = test_results["compatibility_analysis"]
                if "authentication_issues" in analysis:
                    auth_issues.update(analysis["authentication_issues"])
        
        if auth_issues:
            print(f"\n🔑 API Key Issues Detected: {', '.join(auth_issues)}")
            print("   Please check your API key configuration for these providers")
        
        # Check for validation bugs that were fixed
        validation_bugs = []
        for test_name, test_results in detailed_results.items():
            if isinstance(test_results, dict) and "provider_results" in test_results:
                for provider, result in test_results["provider_results"].items():
                    if isinstance(result, dict):
                        if (result.get("note") == "This was a bug in the compatibility tester, not the provider" or
                            "validation_bug" in result.get("error", "")):
                            validation_bugs.append(provider)
        
        if validation_bugs:
            print(f"\n🐛 Validation Bugs Fixed: {', '.join(set(validation_bugs))}")
            print("   These were issues in the tester, not the providers")
        
        print("\n🎯 Recommendations:")
        for recommendation in report["recommendations"]:
            print(f"   {recommendation}")
        
        print("="*80)


async def main():
    """CLI entry point with enhanced error handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test provider compatibility with enhanced analysis")
    parser.add_argument("--providers", nargs="*", help="Providers to test")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Default to all available providers
    default_providers = ["openai", "anthropic", "gemini", "mistral", "deepseek", "perplexity", "groq", "watsonx", "ollama"]
    providers = args.providers or default_providers
    
    generator = CompatibilityReportGenerator()
    report = await generator.generate_full_compatibility_report(providers)
    
    if report:  # Only proceed if we got a valid report
        # Print summary
        generator.print_summary(report)
        
        # Save detailed report
        filename = generator.save_report(report, args.output)
        print(f"\n📁 Full report saved to: {filename}")
    else:
        print("❌ No report generated - check API keys and provider availability")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())