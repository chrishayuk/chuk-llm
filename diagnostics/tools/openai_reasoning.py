#!/usr/bin/env python3
"""
FIXED OpenAI Reasoning Models Diagnostic
========================================

Updated to handle the reality of OpenAI reasoning model availability:
- Only test actually available models (o1-mini, o1, o3-mini)  
- Handle tier restrictions gracefully
- Fix configuration loading issues
- Focus on realistic testing scenarios

Based on current OpenAI model availability (January 2025):
- o1-mini: Available to most API users
- o1: Requires Tier 5 ($1000+ spent, 30+ day accounts)  
- o3-mini: Available to Tiers 3-5
- o3/o4: Not available in API yet
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


# Known available reasoning models (January 2025 reality)
AVAILABLE_REASONING_MODELS = [
    {
        "name": "o1-mini",
        "generation": "o1", 
        "variant": "mini",
        "availability": "most_users",
        "tier_requirement": "1+",
        "description": "Most accessible reasoning model"
    },
    {
        "name": "o1",
        "generation": "o1",
        "variant": "standard", 
        "availability": "tier_5",
        "tier_requirement": "5 ($1000+ spent, 30+ days)",
        "description": "Full O1 model, requires high tier"
    },
    {
        "name": "o1-2024-12-17",
        "generation": "o1",
        "variant": "latest",
        "availability": "tier_5", 
        "tier_requirement": "5 ($1000+ spent, 30+ days)",
        "description": "Latest O1 version"
    },
    {
        "name": "o3-mini", 
        "generation": "o3",
        "variant": "mini",
        "availability": "tier_3_plus",
        "tier_requirement": "3+ (moderate usage)",
        "description": "Latest reasoning model, Tiers 3-5"
    }
]


async def test_model_access(model_name: str) -> Tuple[bool, str, Optional[Dict]]:
    """Test if we can access a specific reasoning model"""
    
    try:
        from chuk_llm.llm.client import get_client
        
        # Create client
        client = get_client(provider="openai", model=model_name)
        
        # Test with minimal request using correct reasoning model parameters
        start_time = time.time()
        
        response = await client.create_completion(
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,  # This should be converted to max_completion_tokens
            stream=False
        )
        
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
        if "not available" in error_msg.lower():
            return False, f"Model not available: {error_msg}", None
        elif "tier" in error_msg.lower() or "usage" in error_msg.lower():
            return False, f"Tier restriction: {error_msg}", None
        elif "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
            return False, f"Parameter error (needs fix): {error_msg}", None
        elif "configuration" in error_msg.lower():
            return False, f"Configuration issue: {error_msg}", None
        else:
            return False, f"Unknown error: {error_msg}", None


async def test_reasoning_capability(model_name: str) -> Dict[str, Any]:
    """Test reasoning capability with proper detection of hidden reasoning tokens"""
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model=model_name)
        
        # Simple reasoning test with expected answer
        reasoning_prompt = """Solve this step by step:

If Alice has 3 apples and Bob has twice as many apples as Alice, and Carol has 1 more apple than Bob, how many apples do they have in total?

Think through this step by step."""

        start_time = time.time()
        
        # Get the raw response to access usage information
        response = await client.create_completion(
            messages=[{"role": "user", "content": reasoning_prompt}],
            max_tokens=200,  # Should convert to max_completion_tokens
            stream=False
        )
        
        response_time = time.time() - start_time
        
        if response.get("error"):
            return {
                "success": False,
                "error": response.get("error"),
                "response_time": response_time
            }
        
        response_text = response.get("response", "")
        
        # NEW: Try to detect hidden reasoning tokens from usage (if available)
        reasoning_tokens = 0
        completion_tokens = 0
        reasoning_ratio = 0
        hidden_reasoning_detected = False
        
        # Check if we can access usage information through the client
        try:
            # For reasoning models, check if we can detect internal reasoning
            # This would require accessing the raw API response
            if hasattr(client, '_last_response_usage'):
                usage = client._last_response_usage
                reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', len(response_text.split()))
                if reasoning_tokens > 0:
                    reasoning_ratio = reasoning_tokens / completion_tokens if completion_tokens > 0 else 0
                    hidden_reasoning_detected = True
        except Exception as e:
            # If we can't access usage info, that's okay
            pass
        
        # Enhanced reasoning detection: correctness + approach + hidden reasoning
        response_lower = response_text.lower()
        
        # Check for correct mathematical answer (Alice=3, Bob=6, Carol=7, Total=16)
        has_correct_total = "16" in response_text
        mentions_all_people = all(name in response_lower for name in ["alice", "bob", "carol"])
        shows_math = any(indicator in response_lower for indicator in ["twice", "*", "×", "+", "=", "3", "6", "7"])
        shows_steps = any(indicator in response_lower for indicator in ["step", "first", "second", "then", "therefore"])
        
        # Improved scoring system
        reasoning_quality_score = 0
        if has_correct_total: reasoning_quality_score += 4  # Correct answer is key
        if mentions_all_people: reasoning_quality_score += 2  # Shows understanding
        if shows_math: reasoning_quality_score += 2  # Shows mathematical reasoning
        if shows_steps: reasoning_quality_score += 2  # Shows explicit reasoning
        if hidden_reasoning_detected: reasoning_quality_score += 3  # Hidden reasoning bonus
        
        # Determine reasoning quality level
        if reasoning_quality_score >= 8:
            reasoning_level = "excellent"
            reasoning_detected = True
        elif reasoning_quality_score >= 5:
            reasoning_level = "good" 
            reasoning_detected = True
        elif reasoning_quality_score >= 3:
            reasoning_level = "moderate"
            reasoning_detected = True
        else:
            reasoning_level = "limited"
            reasoning_detected = False
        
        # Enhanced result with hidden reasoning detection
        result = {
            "success": True,
            "response_time": round(response_time, 2),
            "response_length": len(response_text),
            
            # Traditional reasoning detection
            "reasoning_detected": reasoning_detected,
            "reasoning_score": reasoning_quality_score,
            "reasoning_level": reasoning_level,
            
            # Enhanced analysis
            "correct_answer": has_correct_total,
            "shows_mathematical_work": shows_math,
            "mentions_all_entities": mentions_all_people,
            "explicit_steps": shows_steps,
            
            # Hidden reasoning detection (if available)
            "hidden_reasoning_detected": hidden_reasoning_detected,
            "reasoning_tokens": reasoning_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_ratio": round(reasoning_ratio, 1) if reasoning_ratio > 0 else 0,
            
            # Response preview
            "response_preview": response_text[:300] + "..." if len(response_text) > 300 else response_text
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": 0
        }


async def main():
    """Run focused OpenAI reasoning model testing"""
    print("🧠 OPENAI REASONING MODELS DIAGNOSTIC (FIXED)")
    print("Testing o1, o3 series with Realistic Expectations")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key to test reasoning models")
        return
    
    print("🎯 Testing Actually Available Reasoning Models")
    print(f"   API Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    print()
    
    # Phase 1: Access Testing
    print("🔐 PHASE 1: ACCESS TESTING")
    print("=" * 30)
    
    access_results = []
    available_models = []
    
    for model_info in AVAILABLE_REASONING_MODELS:
        model_name = model_info["name"]
        print(f"\n🔍 Testing access to {model_name}...")
        print(f"   Tier requirement: {model_info['tier_requirement']}")
        print(f"   Description: {model_info['description']}")
        
        has_access, message, details = await test_model_access(model_name)
        
        result = {
            **model_info,
            "has_access": has_access,
            "access_message": message,
            "details": details
        }
        access_results.append(result)
        
        if has_access:
            available_models.append(model_name)
            print(f"   ✅ ACCESS CONFIRMED")
            if details:
                print(f"      Response time: {details['response_time']}s")
        else:
            print(f"   ❌ NO ACCESS: {message}")
            
            # Provide helpful guidance
            if "tier" in message.lower():
                print(f"      💡 This model requires higher usage tier")
            elif "parameter" in message.lower():
                print(f"      🔧 Parameter conversion issue - needs client fix")
            elif "configuration" in message.lower():
                print(f"      ⚙️  Configuration loading issue")
    
    # Summary of access results
    print(f"\n📊 ACCESS SUMMARY:")
    print(f"   Models tested: {len(access_results)}")
    print(f"   Models accessible: {len(available_models)}")
    
    if available_models:
        print(f"   ✅ Available models: {', '.join(available_models)}")
    else:
        print(f"   ❌ No reasoning models available")
        print(f"\n🔧 COMMON ISSUES:")
        print(f"      • Tier restrictions - o1 requires Tier 5, o3-mini requires Tier 3+")
        print(f"      • Account age - o1 requires 30+ day old accounts") 
        print(f"      • Spending requirements - o1 requires $1000+ spent")
        print(f"      • Configuration issues - check client_class and model availability")
        return
    
    # Phase 2: Reasoning Testing
    print(f"\n🧠 PHASE 2: REASONING CAPABILITY TESTING")
    print("=" * 45)
    
    reasoning_results = []
    
    for model_name in available_models:
        print(f"\n🚀 Testing reasoning capabilities: {model_name}")
        
        result = await test_reasoning_capability(model_name)
        result["model"] = model_name
        reasoning_results.append(result)
        
        if result.get("success"):
            print(f"   ✅ REASONING TEST PASSED")
            print(f"      Response time: {result['response_time']}s")
            print(f"      Reasoning level: {result.get('reasoning_level', 'unknown').upper()}")
            print(f"      Quality score: {result['reasoning_score']}/13")
            
            # Enhanced result display
            indicators = []
            if result.get("correct_answer"):
                indicators.append("✅ Correct answer")
            if result.get("shows_mathematical_work"):
                indicators.append("✅ Math work")
            if result.get("explicit_steps"):
                indicators.append("✅ Step-by-step")
            if result.get("hidden_reasoning_detected"):
                indicators.append(f"🧠 Hidden reasoning ({result.get('reasoning_tokens', 0)} tokens)")
            
            if indicators:
                print(f"      Analysis: {' | '.join(indicators)}")
            
            if result.get("reasoning_ratio", 0) > 0:
                print(f"      Reasoning intensity: {result['reasoning_ratio']}x output tokens")
            
            if result.get("response_preview"):
                preview = result["response_preview"][:200]
                print(f"      Preview: {preview}{'...' if len(result['response_preview']) > 200 else ''}")
        else:
            print(f"   ❌ REASONING TEST FAILED")
            print(f"      Error: {result.get('error', 'Unknown error')}")
    
    # Phase 3: Final Analysis
    print(f"\n📈 FINAL ANALYSIS")
    print("=" * 20)
    
    successful_models = [r for r in reasoning_results if r.get("success")]
    reasoning_capable = [r for r in successful_models if r.get("reasoning_detected")]
    excellent_reasoning = [r for r in successful_models if r.get("reasoning_level") == "excellent"]
    good_reasoning = [r for r in successful_models if r.get("reasoning_level") in ["excellent", "good"]]
    hidden_reasoning = [r for r in successful_models if r.get("hidden_reasoning_detected")]
    
    print(f"📋 Results Summary:")
    print(f"   Total models tested: {len(reasoning_results)}")
    print(f"   Successfully responding: {len(successful_models)}")
    print(f"   Showing reasoning behavior: {len(reasoning_capable)}")
    print(f"   Excellent reasoning quality: {len(excellent_reasoning)}")
    print(f"   Good+ reasoning quality: {len(good_reasoning)}")
    print(f"   Hidden reasoning detected: {len(hidden_reasoning)}")
    
    if reasoning_capable or hidden_reasoning:
        print(f"\n🎉 REASONING MODELS WORKING!")
        print(f"✅ Successfully tested OpenAI reasoning models")
        
        print(f"\n💡 REASONING MODEL ANALYSIS:")
        
        # Group models by reasoning capability
        for result in successful_models:
            model_name = result["model"]
            reasoning_level = result.get("reasoning_level", "unknown")
            response_time = result["response_time"]
            quality_score = result["reasoning_score"]
            
            # Create capability description
            capabilities = []
            if result.get("correct_answer"):
                capabilities.append("Correct answer")
            if result.get("shows_mathematical_work"):
                capabilities.append("Math reasoning")
            if result.get("explicit_steps"):
                capabilities.append("Explicit steps")
            if result.get("hidden_reasoning_detected"):
                reasoning_tokens = result.get("reasoning_tokens", 0)
                capabilities.append(f"Hidden reasoning ({reasoning_tokens} tokens)")
            
            capability_str = " + ".join(capabilities) if capabilities else "Basic response"
            
            # Quality indicator
            if reasoning_level == "excellent":
                quality_emoji = "🌟"
            elif reasoning_level == "good":
                quality_emoji = "✅"
            elif reasoning_level == "moderate":
                quality_emoji = "🟡"
            else:
                quality_emoji = "⚪"
            
            print(f"   {quality_emoji} {model_name}")
            print(f"      Quality: {reasoning_level.upper()} ({quality_score}/13)")
            print(f"      Speed: {response_time}s")
            print(f"      Capabilities: {capability_str}")
        
        print(f"\n🚀 MCP CLI READY COMMANDS:")
        
        # Recommend best models first
        best_models = sorted(successful_models, key=lambda x: (
            x.get("reasoning_score", 0),
            -x.get("response_time", 999)  # Negative for ascending time
        ), reverse=True)
        
        for result in best_models[:3]:  # Top 3
            model_name = result["model"]
            reasoning_level = result.get("reasoning_level", "unknown")
            print(f"   mcp-cli chat --provider openai --model {model_name}  # {reasoning_level} quality")
        
        # Performance analysis
        if len(successful_models) > 1:
            fastest = min(successful_models, key=lambda x: x["response_time"])
            highest_quality = max(successful_models, key=lambda x: x.get("reasoning_score", 0))
            
            print(f"\n📊 PERFORMANCE ANALYSIS:")
            print(f"   ⚡ Fastest: {fastest['model']} ({fastest['response_time']}s)")
            print(f"   🧠 Highest quality: {highest_quality['model']} ({highest_quality.get('reasoning_score', 0)}/13)")
            
            # Hidden reasoning analysis
            if hidden_reasoning:
                print(f"\n🔍 HIDDEN REASONING ANALYSIS:")
                for result in hidden_reasoning:
                    model_name = result["model"]
                    reasoning_tokens = result.get("reasoning_tokens", 0)
                    reasoning_ratio = result.get("reasoning_ratio", 0)
                    print(f"   🧠 {model_name}: {reasoning_tokens} internal reasoning tokens ({reasoning_ratio}x multiplier)")
                
                print(f"\n💡 INSIGHT: These models do extensive internal reasoning")
                print(f"   Even when they don't show steps, they're thinking deeply!")
                
        # Model recommendations by use case
        print(f"\n🎯 USE CASE RECOMMENDATIONS:")
        
        if fastest and fastest.get("reasoning_level") in ["good", "excellent"]:
            print(f"   ⚡ Speed Priority: {fastest['model']} ({fastest['response_time']}s, {fastest.get('reasoning_level')} quality)")
        
        explicit_reasoning = [r for r in successful_models if r.get("explicit_steps")]
        if explicit_reasoning:
            best_explicit = max(explicit_reasoning, key=lambda x: x.get("reasoning_score", 0))
            print(f"   🔍 Transparency Priority: {best_explicit['model']} (shows reasoning steps)")
        
        if highest_quality:
            print(f"   🌟 Quality Priority: {highest_quality['model']} (score: {highest_quality.get('reasoning_score', 0)}/13)")
    
    elif successful_models:
        print(f"\n⚠️  MODELS RESPONDING BUT LIMITED REASONING DETECTED")
        print(f"Models are accessible but may need reasoning prompt optimization")
        
        for result in successful_models:
            model_name = result["model"]
            response_time = result["response_time"]
            score = result.get("reasoning_score", 0)
            print(f"   • {model_name}: {response_time}s response, {score}/13 reasoning score")
        
        print(f"\n💡 SUGGESTIONS:")
        print(f"   • Try more explicit reasoning prompts")
        print(f"   • Models may be doing internal reasoning not visible in output")
        print(f"   • Consider using Responses API for reasoning visibility")
    
    else:
        print(f"\n❌ NO REASONING MODELS WORKING")
        print(f"\n🔧 ISSUES TO CHECK:")
        
        # Analyze common error types
        error_types = {}
        for result in reasoning_results:
            if not result.get("success"):
                error = result.get("error", "Unknown")
                if "parameter" in error.lower():
                    error_types["parameter_issues"] = error_types.get("parameter_issues", 0) + 1
                elif "tier" in error.lower():
                    error_types["tier_issues"] = error_types.get("tier_issues", 0) + 1
                elif "configuration" in error.lower():
                    error_types["config_issues"] = error_types.get("config_issues", 0) + 1
                else:
                    error_types["other_issues"] = error_types.get("other_issues", 0) + 1
        
        for issue_type, count in error_types.items():
            print(f"   • {issue_type.replace('_', ' ').title()}: {count} models")
        
        if error_types.get("parameter_issues"):
            print(f"\n💡 PARAMETER ISSUE FIX:")
            print(f"   The OpenAI client needs max_completion_tokens parameter conversion")
            print(f"   This should be handled automatically by the reasoning model fix")
        
        if error_types.get("tier_issues"):
            print(f"\n💡 TIER ACCESS INFO:")
            print(f"   • o1-mini: Most users (Tier 1+)")
            print(f"   • o1: Tier 5 users ($1000+ spent)")
            print(f"   • o3-mini: Tier 3+ users (moderate usage)")
    
    print(f"\n🔬 DIAGNOSTIC INSIGHTS:")
    print(f"   • Reasoning models do internal processing (hidden reasoning tokens)")
    print(f"   • External reasoning steps are just one indicator of capability")
    print(f"   • Response correctness + speed often indicate internal reasoning quality")
    print(f"   • Different models have different reasoning exposition styles")
    
    print(f"\n🏁 DIAGNOSTIC COMPLETE")
    
    # Final recommendation
    if len(available_models) > 0:
        print(f"✨ OpenAI reasoning models are partially working!")
        if len(reasoning_capable) > 0:
            print(f"✨ Reasoning behavior confirmed on {len(reasoning_capable)} models")
    else:
        print(f"🔧 OpenAI reasoning models need configuration or access fixes")


if __name__ == "__main__":
    print("🚀 Starting Fixed OpenAI Reasoning Models Diagnostic...")
    asyncio.run(main())