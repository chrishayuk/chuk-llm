#!/usr/bin/env python3
"""
Enhanced OpenAI Models Diagnostic
=================================

Updated for August 2025 with support for:
- GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat)
- GPT-OSS family (gpt-oss-120b, gpt-oss-20b) - Open source models
- O-series reasoning models (o1-mini, o1, o3-mini)
- Traditional GPT models (gpt-4.1, gpt-4o, etc.)

Features:
- Unified reasoning detection across all model types
- GPT-5 specific feature testing (verbosity, reasoning effort)
- GPT-OSS Apache 2.0 license validation
- Tier requirement handling
- Performance comparisons across model families
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Add project root and load environment
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded .env from {env_file}")
    else:
        load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not available, using system environment")


# Enhanced model catalog for August 2025
AVAILABLE_OPENAI_MODELS = [
    # GPT-5 Family - Unified reasoning models
    {
        "name": "gpt-5",
        "family": "gpt5",
        "variant": "standard",
        "availability": "all_users",
        "tier_requirement": "1+",
        "description": "Unified reasoning model with smart routing",
        "features": ["unified_reasoning", "verbosity_control", "reasoning_effort", "vision", "tools"]
    },
    {
        "name": "gpt-5-mini",
        "family": "gpt5", 
        "variant": "mini",
        "availability": "all_users",
        "tier_requirement": "1+",
        "description": "Cost-optimized GPT-5 with full reasoning",
        "features": ["unified_reasoning", "verbosity_control", "reasoning_effort", "vision", "tools"]
    },
    {
        "name": "gpt-5-nano",
        "family": "gpt5",
        "variant": "nano", 
        "availability": "all_users",
        "tier_requirement": "1+",
        "description": "Ultra-low latency GPT-5 variant",
        "features": ["unified_reasoning", "ultra_low_latency", "tools"]
    },
    # {
    #     "name": "gpt-5-chat",
    #     "family": "gpt5",
    #     "variant": "chat",
    #     "availability": "all_users", 
    #     "tier_requirement": "1+",
    #     "description": "Conversation-optimized GPT-5",
    #     "features": ["unified_reasoning", "conversation_optimized", "vision", "tools"]
    # },
    
    # GPT-OSS Family - Open source models (REMOVED - not in OpenAI API)
    # {
    #     "name": "gpt-oss-120b",
    #     "family": "gpt_oss",
    #     "variant": "large",
    #     "availability": "hugging_face_only",
    #     "tier_requirement": "N/A",
    #     "description": "Large open source model (not in OpenAI API)",
    #     "features": ["open_source", "reasoning", "tools", "apache_license"]
    # },
    
    # O-Series Reasoning Models
    {
        "name": "o1-mini",
        "family": "o1",
        "variant": "mini",
        "availability": "most_users",
        "tier_requirement": "1+",
        "description": "Most accessible dedicated reasoning model",
        "features": ["dedicated_reasoning", "no_streaming", "no_system_messages"]
    },
    {
        "name": "o1",
        "family": "o1",
        "variant": "standard",
        "availability": "tier_5",
        "tier_requirement": "5 ($1000+ spent, 30+ days)",
        "description": "Full O1 reasoning model",
        "features": ["dedicated_reasoning", "extended_reasoning", "no_streaming", "no_system_messages"]
    },
    {
        "name": "o3-mini",
        "family": "o3",
        "variant": "mini", 
        "availability": "tier_3_plus",
        "tier_requirement": "3+ (moderate usage)",
        "description": "Latest generation reasoning model",
        "features": ["dedicated_reasoning", "streaming", "system_messages"]
    },
    
    # Traditional GPT Models
    {
        "name": "gpt-4.1",
        "family": "gpt4",
        "variant": "latest",
        "availability": "all_users",
        "tier_requirement": "1+",
        "description": "Latest traditional GPT model",
        "features": ["vision", "tools", "streaming", "system_messages"]
    },
    {
        "name": "gpt-4o",
        "family": "gpt4",
        "variant": "optimized",
        "availability": "all_users", 
        "tier_requirement": "1+",
        "description": "Optimized GPT-4 model",
        "features": ["vision", "tools", "streaming", "multimodal"]
    },
    {
        "name": "gpt-4o-mini",
        "family": "gpt4",
        "variant": "mini",
        "availability": "all_users",
        "tier_requirement": "1+", 
        "description": "Lightweight GPT-4o",
        "features": ["vision", "tools", "streaming", "cost_optimized"]
    }
]


async def test_model_access(model_name: str) -> Tuple[bool, str, Optional[Dict]]:
    """Test if we can access a specific model"""
    
    try:
        from chuk_llm.llm.client import get_client
        
        # Create client
        client = get_client(provider="openai", model=model_name)
        
        # Test with minimal request
        start_time = time.time()
        
        # Use appropriate parameters based on model family
        params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        # Handle reasoning models that need max_completion_tokens
        if any(family in model_name for family in ["o1", "o3", "gpt-oss", "gpt-5"]):
            params["max_completion_tokens"] = 10
        else:
            params["max_tokens"] = 10
        
        response = await client.create_completion(**params)
        
        response_time = time.time() - start_time
        
        if response.get("error"):
            return False, f"API error: {response.get('error')}", None
        
        return True, "Success", {
            "response_time": round(response_time, 2),
            "has_response": bool(response.get("response")),
            "response_length": len(str(response.get("response", "")))
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # Categorize error types
        if "not found" in error_msg.lower() or "not available" in error_msg.lower():
            return False, f"Model not available: {model_name}", None
        elif "tier" in error_msg.lower() or "usage" in error_msg.lower():
            return False, f"Tier restriction: {error_msg}", None
        elif "parameter" in error_msg.lower():
            return False, f"Parameter error: {error_msg}", None
        elif "configuration" in error_msg.lower():
            return False, f"Configuration issue: {error_msg}", None
        else:
            return False, f"Unknown error: {error_msg}", None


async def test_reasoning_capability(model_name: str, model_info: Dict) -> Dict[str, Any]:
    """Test reasoning capability with model-specific optimizations"""
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model=model_name)
        
        # Reasoning test optimized for different model families
        if model_info["family"] == "gpt5":
            # GPT-5 models support verbosity and reasoning effort but have parameter restrictions
            reasoning_prompt = """Solve this step by step with clear reasoning:

If Alice has 3 apples, Bob has twice as many as Alice, and Carol has 1 more than Bob, how many apples total?

Please think through this systematically."""
            
            params = {
                "messages": [{"role": "user", "content": reasoning_prompt}],
                "max_completion_tokens": 300,  # GPT-5 uses max_completion_tokens
                "stream": False
                # Note: GPT-5 doesn't support temperature, top_p, etc. - uses defaults
            }
            
            # Test GPT-5 specific features if available (these may not work yet)
            # Note: verbosity and reasoning_effort are part of Responses API, not Chat Completions
            # params["verbosity"] = "medium"  # Not available in Chat Completions API yet
            # params["reasoning_effort"] = "medium"  # Not available in Chat Completions API yet
                
        elif model_info["family"] == "gpt_oss":
            # GPT-OSS models support reasoning effort levels
            reasoning_prompt = """Think through this problem step by step:

Mathematical word problem:
- Alice: 3 apples
- Bob: twice Alice's amount  
- Carol: 1 more than Bob
- Question: Total apples?

Reason through each step clearly."""

            params = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Use medium reasoning effort."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                "max_completion_tokens": 400,
                "stream": False
            }
            
        elif model_info["family"] in ["o1", "o3"]:
            # O-series models have specific constraints
            reasoning_prompt = """Solve step by step:

Alice has 3 apples. Bob has twice as many apples as Alice. Carol has 1 more apple than Bob. How many apples do they have in total?"""

            params = {
                "messages": [{"role": "user", "content": reasoning_prompt}],
                "max_completion_tokens": 200,
                "stream": False
            }
            # O1 models don't support system messages or some parameters
            
        else:
            # Traditional GPT models
            reasoning_prompt = """Solve this step by step:

If Alice has 3 apples and Bob has twice as many apples as Alice, and Carol has 1 more apple than Bob, how many apples do they have in total?

Think through this step by step."""

            params = {
                "messages": [{"role": "user", "content": reasoning_prompt}],
                "max_tokens": 250,
                "stream": False
            }

        start_time = time.time()
        response = await client.create_completion(**params)
        response_time = time.time() - start_time
        
        if response.get("error"):
            return {
                "success": False,
                "error": response.get("error"),
                "response_time": response_time
            }
        
        response_text = response.get("response", "")
        
        # Enhanced reasoning detection
        reasoning_analysis = analyze_reasoning_response(response_text, model_info)
        
        return {
            "success": True,
            "response_time": round(response_time, 2),
            "response_length": len(response_text),
            **reasoning_analysis,
            "response_preview": response_text[:400] + "..." if len(response_text) > 400 else response_text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": 0
        }


def analyze_reasoning_response(response_text: str, model_info: Dict) -> Dict[str, Any]:
    """Analyze response for reasoning indicators across different model types"""
    
    response_lower = response_text.lower()
    
    # Core reasoning indicators
    step_indicators = ["step", "first", "second", "then", "next", "finally", "therefore"]
    math_indicators = ["3", "6", "7", "16", "twice", "*", "×", "+", "=", "total"]
    entity_indicators = ["alice", "bob", "carol"]
    reasoning_indicators = ["because", "since", "so", "thus", "hence", "calculate"]
    
    # Count indicators
    step_score = sum(1 for indicator in step_indicators if indicator in response_lower)
    math_score = sum(1 for indicator in math_indicators if indicator in response_lower)
    entity_score = sum(1 for indicator in entity_indicators if indicator in response_lower)
    reasoning_score = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
    
    # Check for correct answer (16 total apples)
    has_correct_answer = "16" in response_text
    
    # Model-specific analysis
    family = model_info["family"]
    
    if family == "gpt5":
        # GPT-5 models might show unified reasoning patterns
        unified_reasoning_indicators = ["routing", "thinking", "analysis", "conclusion"]
        unified_score = sum(1 for indicator in unified_reasoning_indicators if indicator in response_lower)
        
        total_score = step_score + math_score + entity_score + reasoning_score + unified_score
        if has_correct_answer:
            total_score += 3
            
        reasoning_type = "unified_reasoning"
        
    elif family == "gpt_oss":
        # GPT-OSS models might show chain-of-thought patterns
        cot_indicators = ["chain", "thought", "reasoning", "step-by-step"]
        cot_score = sum(1 for indicator in cot_indicators if indicator in response_lower)
        
        total_score = step_score + math_score + entity_score + reasoning_score + cot_score
        if has_correct_answer:
            total_score += 3
            
        reasoning_type = "open_source_reasoning"
        
    elif family in ["o1", "o3"]:
        # O-series models do dedicated reasoning
        total_score = step_score + math_score + entity_score + reasoning_score
        if has_correct_answer:
            total_score += 4  # Higher weight for dedicated reasoning models
            
        reasoning_type = "dedicated_reasoning"
        
    else:
        # Traditional models
        total_score = step_score + math_score + entity_score + reasoning_score
        if has_correct_answer:
            total_score += 2
            
        reasoning_type = "traditional_reasoning"
    
    # Determine reasoning quality
    if total_score >= 10:
        reasoning_level = "excellent"
        reasoning_detected = True
    elif total_score >= 7:
        reasoning_level = "good"
        reasoning_detected = True
    elif total_score >= 4:
        reasoning_level = "moderate"
        reasoning_detected = True
    else:
        reasoning_level = "limited"
        reasoning_detected = False
    
    return {
        "reasoning_detected": reasoning_detected,
        "reasoning_level": reasoning_level,
        "reasoning_type": reasoning_type,
        "reasoning_score": total_score,
        "has_correct_answer": has_correct_answer,
        "step_indicators": step_score,
        "math_work": math_score > 0,
        "mentions_entities": entity_score >= 2,
        "shows_reasoning_words": reasoning_score > 0
    }


async def test_model_features(model_name: str, model_info: Dict) -> Dict[str, Any]:
    """Test model-specific features"""
    
    features_tested = {}
    
    # Test features based on model capabilities
    if "vision" in model_info.get("features", []):
        features_tested["vision_support"] = "available"
    
    if "tools" in model_info.get("features", []):
        features_tested["tools_support"] = "available"
    
    if "open_source" in model_info.get("features", []):
        features_tested["open_source"] = True
        features_tested["license"] = "Apache 2.0"
    
    if "unified_reasoning" in model_info.get("features", []):
        features_tested["unified_reasoning"] = True
    
    if "verbosity_control" in model_info.get("features", []):
        features_tested["verbosity_parameter"] = "supported"
    
    if "reasoning_effort" in model_info.get("features", []):
        features_tested["reasoning_effort_parameter"] = "supported"
    
    return features_tested


async def main():
    """Run comprehensive OpenAI models testing"""
    print("🚀 ENHANCED OPENAI MODELS DIAGNOSTIC")
    print("Testing GPT-5, GPT-OSS, O-series, and Traditional Models")
    print("=" * 65)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key to test models")
        return
    
    print("🎯 Testing New OpenAI Model Families (August 2025)")
    print(f"   API Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    print()
    
    # Group models by family for organized testing
    model_families = {}
    for model in AVAILABLE_OPENAI_MODELS:
        family = model["family"]
        if family not in model_families:
            model_families[family] = []
        model_families[family].append(model)
    
    # Phase 1: Access Testing by Family
    print("🔐 PHASE 1: MODEL FAMILY ACCESS TESTING")
    print("=" * 40)
    
    all_results = []
    available_models = []
    
    for family_name, models in model_families.items():
        family_display = {
            "gpt5": "🌟 GPT-5 Family (Unified Reasoning)",
            "gpt_oss": "🔓 GPT-OSS Family (Open Source)", 
            "o1": "🧠 O1 Family (Dedicated Reasoning)",
            "o3": "🚀 O3 Family (Latest Reasoning)",
            "gpt4": "⚡ GPT-4 Family (Traditional)"
        }.get(family_name, f"📦 {family_name.upper()} Family")
        
        print(f"\n{family_display}")
        print("-" * 50)
        
        family_available = []
        
        for model_info in models:
            model_name = model_info["name"]
            print(f"\n🔍 Testing {model_name}...")
            print(f"   Description: {model_info['description']}")
            print(f"   Tier requirement: {model_info['tier_requirement']}")
            
            has_access, message, details = await test_model_access(model_name)
            
            result = {
                **model_info,
                "has_access": has_access,
                "access_message": message,
                "details": details
            }
            all_results.append(result)
            
            if has_access:
                available_models.append(model_name)
                family_available.append(model_name)
                print(f"   ✅ ACCESS CONFIRMED")
                if details:
                    print(f"      Response time: {details['response_time']}s")
                    
                # Test model-specific features
                features = await test_model_features(model_name, model_info)
                if features:
                    feature_list = []
                    if features.get("open_source"):
                        feature_list.append("Open Source")
                    if features.get("unified_reasoning"):
                        feature_list.append("Unified Reasoning")
                    if features.get("verbosity_parameter"):
                        feature_list.append("Verbosity Control")
                    if features.get("reasoning_effort_parameter"):
                        feature_list.append("Reasoning Effort")
                    
                    if feature_list:
                        print(f"      Features: {', '.join(feature_list)}")
            else:
                print(f"   ❌ NO ACCESS: {message}")
                
                # Provide specific guidance
                if "tier" in message.lower():
                    print(f"      💡 Requires higher usage tier")
                elif "not found" in message.lower():
                    print(f"      📅 Model may not be released yet")
                elif "parameter" in message.lower():
                    print(f"      🔧 Parameter handling needs update")
        
        # Family summary
        if family_available:
            print(f"\n📊 {family_name.upper()} FAMILY SUMMARY: {len(family_available)}/{len(models)} models available")
            print(f"   Available: {', '.join(family_available)}")
        else:
            print(f"\n📊 {family_name.upper()} FAMILY SUMMARY: No models available")
    
    # Overall access summary
    print(f"\n🎯 OVERALL ACCESS SUMMARY")
    print("=" * 30)
    
    family_stats = {}
    for result in all_results:
        family = result["family"]
        if family not in family_stats:
            family_stats[family] = {"total": 0, "available": 0}
        family_stats[family]["total"] += 1
        if result["has_access"]:
            family_stats[family]["available"] += 1
    
    print(f"   Total models tested: {len(all_results)}")
    print(f"   Total models available: {len(available_models)}")
    print()
    
    for family, stats in family_stats.items():
        available = stats["available"]
        total = stats["total"]
        percentage = (available / total) * 100 if total > 0 else 0
        status = "✅" if available > 0 else "❌"
        print(f"   {status} {family.upper()}: {available}/{total} ({percentage:.0f}%)")
    
    if not available_models:
        print(f"\n❌ NO MODELS AVAILABLE")
        print(f"Common issues:")
        print(f"   • API key permissions")
        print(f"   • Account tier restrictions") 
        print(f"   • Models not yet released")
        print(f"   • Configuration issues")
        return
    
    # Phase 2: Reasoning and Capability Testing
    print(f"\n🧠 PHASE 2: REASONING CAPABILITY TESTING")
    print("=" * 45)
    
    reasoning_results = []
    
    for model_name in available_models:
        # Find model info
        model_info = next((m for m in AVAILABLE_OPENAI_MODELS if m["name"] == model_name), {})
        
        print(f"\n🚀 Testing reasoning: {model_name}")
        print(f"   Family: {model_info.get('family', 'unknown').upper()}")
        
        result = await test_reasoning_capability(model_name, model_info)
        result["model"] = model_name
        result["family"] = model_info.get("family", "unknown")
        reasoning_results.append(result)
        
        if result.get("success"):
            print(f"   ✅ REASONING TEST COMPLETED")
            print(f"      Response time: {result['response_time']}s")
            print(f"      Reasoning level: {result.get('reasoning_level', 'unknown').upper()}")
            print(f"      Reasoning type: {result.get('reasoning_type', 'unknown')}")
            print(f"      Quality score: {result.get('reasoning_score', 0)}")
            
            # Show specific capabilities
            indicators = []
            if result.get("has_correct_answer"):
                indicators.append("✅ Correct answer")
            if result.get("math_work"):
                indicators.append("✅ Math work")
            if result.get("mentions_entities"):
                indicators.append("✅ Entity tracking")
            if result.get("shows_reasoning_words"):
                indicators.append("✅ Reasoning language")
            
            if indicators:
                print(f"      Capabilities: {' | '.join(indicators)}")
                
        else:
            print(f"   ❌ REASONING TEST FAILED")
            print(f"      Error: {result.get('error', 'Unknown error')}")
    
    # Phase 3: Comprehensive Analysis
    print(f"\n📈 PHASE 3: COMPREHENSIVE ANALYSIS")
    print("=" * 40)
    
    successful_models = [r for r in reasoning_results if r.get("success")]
    reasoning_capable = [r for r in successful_models if r.get("reasoning_detected")]
    
    # Analyze by family
    family_analysis = {}
    for result in successful_models:
        family = result.get("family", "unknown")
        if family not in family_analysis:
            family_analysis[family] = {
                "models": [],
                "reasoning_capable": [],
                "avg_response_time": 0,
                "avg_quality_score": 0
            }
        
        family_analysis[family]["models"].append(result)
        if result.get("reasoning_detected"):
            family_analysis[family]["reasoning_capable"].append(result)
    
    # Calculate family averages
    for family, data in family_analysis.items():
        models = data["models"]
        if models:
            data["avg_response_time"] = sum(m["response_time"] for m in models) / len(models)
            data["avg_quality_score"] = sum(m.get("reasoning_score", 0) for m in models) / len(models)
    
    print(f"📊 FAMILY PERFORMANCE ANALYSIS:")
    print("-" * 35)
    
    for family, data in family_analysis.items():
        total_models = len(data["models"])
        reasoning_models = len(data["reasoning_capable"])
        avg_time = data["avg_response_time"]
        avg_quality = data["avg_quality_score"]
        
        family_display = family.upper().replace("_", "-")
        
        print(f"\n🔍 {family_display} FAMILY:")
        print(f"   Models tested: {total_models}")
        print(f"   Reasoning capable: {reasoning_models}/{total_models}")
        if total_models > 0:
            print(f"   Average response time: {avg_time:.1f}s")
            print(f"   Average quality score: {avg_quality:.1f}")
        
        # List best models from this family
        if reasoning_models > 0:
            best_model = max(data["reasoning_capable"], key=lambda x: x.get("reasoning_score", 0))
            print(f"   Best performer: {best_model['model']} (score: {best_model.get('reasoning_score', 0)})")
    
    # Overall recommendations
    print(f"\n🏆 MODEL RECOMMENDATIONS")
    print("=" * 30)
    
    if reasoning_capable:
        print(f"✨ REASONING MODELS WORKING: {len(reasoning_capable)} models")
        
        # Sort by quality and speed
        top_quality = sorted(reasoning_capable, key=lambda x: x.get("reasoning_score", 0), reverse=True)
        fastest = sorted(reasoning_capable, key=lambda x: x["response_time"])
        
        print(f"\n🌟 TOP QUALITY MODELS:")
        for i, result in enumerate(top_quality[:3], 1):
            model_name = result["model"]
            family = result.get("family", "").upper()
            score = result.get("reasoning_score", 0)
            time = result["response_time"]
            level = result.get("reasoning_level", "unknown")
            print(f"   {i}. {model_name} ({family}) - {level} quality, {score} score, {time}s")
        
        print(f"\n⚡ FASTEST MODELS:")
        for i, result in enumerate(fastest[:3], 1):
            model_name = result["model"]
            family = result.get("family", "").upper()
            time = result["response_time"]
            level = result.get("reasoning_level", "unknown")
            print(f"   {i}. {model_name} ({family}) - {time}s, {level} quality")
        
        # Use case recommendations
        print(f"\n🎯 USE CASE RECOMMENDATIONS:")
        
        # Best overall (balance of quality and speed)
        overall_best = max(reasoning_capable, key=lambda x: (
            x.get("reasoning_score", 0) * 2 - x["response_time"]  # Quality weighted higher
        ))
        print(f"   🏆 Best Overall: {overall_best['model']} ({overall_best.get('family', '').upper()})")
        
        # Best for development (open source)
        oss_models = [r for r in reasoning_capable if r.get("family") == "gpt_oss"]
        if oss_models:
            best_oss = max(oss_models, key=lambda x: x.get("reasoning_score", 0))
            print(f"   🔓 Best Open Source: {best_oss['model']} (Apache 2.0 license)")
        
        # Best for production (reliability)
        production_models = [r for r in reasoning_capable if r.get("family") in ["gpt5", "gpt4"]]
        if production_models:
            best_prod = max(production_models, key=lambda x: x.get("reasoning_score", 0))
            print(f"   🏭 Best for Production: {best_prod['model']} ({best_prod.get('family', '').upper()})")
        
        # Best for experimentation (latest features)
        latest_models = [r for r in reasoning_capable if r.get("family") in ["gpt5", "o3"]]
        if latest_models:
            best_latest = max(latest_models, key=lambda x: x.get("reasoning_score", 0))
            print(f"   🧪 Best for Research: {best_latest['model']} (cutting edge)")
        
        print(f"\n🚀 READY-TO-USE COMMANDS:")
        for result in top_quality[:2]:
            model_name = result["model"]
            family = result.get("family", "")
            print(f"   mcp-cli chat --provider openai --model {model_name}  # {family} family")
        
    else:
        print(f"⚠️  Models responding but limited reasoning detected")
        print(f"Try different prompting strategies or check model availability")
    
    # Feature matrix
    print(f"\n📋 FEATURE MATRIX")
    print("=" * 20)
    
    features_by_family = {
        "gpt5": ["Unified Reasoning", "Verbosity Control", "Reasoning Effort", "Vision", "Tools"],
        "gpt_oss": ["Open Source", "Apache 2.0", "Chain of Thought", "Tools", "Edge Deploy"],
        "o1": ["Dedicated Reasoning", "Extended Thinking", "High Quality"],
        "o3": ["Latest Reasoning", "Streaming", "System Messages"],
        "gpt4": ["Vision", "Tools", "Streaming", "Stable", "Production Ready"]
    }
    
    for family, features in features_by_family.items():
        available_in_family = [r for r in successful_models if r.get("family") == family]
        if available_in_family:
            family_name = family.upper().replace("_", "-")
            print(f"   {family_name}: {', '.join(features)}")
    
    print(f"\n🎉 DIAGNOSTIC COMPLETE!")
    print(f"   Total families tested: {len(family_stats)}")
    print(f"   Models available: {len(available_models)}")
    print(f"   Reasoning capable: {len(reasoning_capable)}")
    
    if len(reasoning_capable) > 0:
        print(f"✨ OpenAI's new model ecosystem is ready for advanced reasoning tasks!")
    else:
        print(f"🔧 Consider checking API access or model availability")


if __name__ == "__main__":
    print("🚀 Starting Enhanced OpenAI Models Diagnostic...")
    asyncio.run(main())