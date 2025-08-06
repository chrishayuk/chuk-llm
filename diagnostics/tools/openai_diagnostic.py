#!/usr/bin/env python3
"""
OpenAI Reasoning Models Diagnostic
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
    """Test reasoning capability with a simple problem"""
    
    try:
        from chuk_llm.llm.client import get_client
        
        client = get_client(provider="openai", model=model_name)
        
        # Simple reasoning test that should work quickly
        reasoning_prompt = """Solve this step by step:

If Alice has 3 apples and Bob has twice as many apples as Alice, and Carol has 1 more apple than Bob, how many apples do they have in total?

Think through this step by step."""

        start_time = time.time()
        
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
        
        # Check for reasoning indicators
        response_text = response.get("response", "").lower()
        reasoning_indicators = [
            "step", "first", "second", "then", "therefore", 
            "alice", "bob", "carol", "total", "3", "6", "7", "16"
        ]
        
        reasoning_score = sum(1 for indicator in reasoning_indicators if indicator in response_text)
        has_reasoning = reasoning_score >= 5
        
        return {
            "success": True,
            "response_time": round(response_time, 2),
            "response_length": len(response.get("response", "")),
            "reasoning_detected": has_reasoning,
            "reasoning_score": reasoning_score,
            "response_preview": response.get("response", "")[:200] + "..." if response.get("response") else ""
        }
        
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
            print(f"      Reasoning detected: {'Yes' if result['reasoning_detected'] else 'No'}")
            print(f"      Reasoning score: {result['reasoning_score']}/10")
            if result.get("response_preview"):
                print(f"      Preview: {result['response_preview']}")
        else:
            print(f"   ❌ REASONING TEST FAILED")
            print(f"      Error: {result.get('error', 'Unknown error')}")
    
    # Phase 3: Final Analysis
    print(f"\n📈 FINAL ANALYSIS")
    print("=" * 20)
    
    successful_models = [r for r in reasoning_results if r.get("success")]
    reasoning_capable = [r for r in successful_models if r.get("reasoning_detected")]
    
    print(f"📋 Results Summary:")
    print(f"   Total models tested: {len(reasoning_results)}")
    print(f"   Successfully responding: {len(successful_models)}")
    print(f"   Showing reasoning behavior: {len(reasoning_capable)}")
    
    if reasoning_capable:
        print(f"\n🎉 REASONING MODELS WORKING!")
        print(f"✅ Successfully tested OpenAI reasoning models")
        
        print(f"\n💡 WORKING MODELS:")
        for result in reasoning_capable:
            model_name = result["model"]
            response_time = result["response_time"]
            print(f"   • {model_name} (avg: {response_time}s)")
        
        print(f"\n🚀 MCP CLI READY COMMANDS:")
        for result in reasoning_capable[:2]:  # Top 2
            model_name = result["model"]
            print(f"   mcp-cli chat --provider openai --model {model_name}")
        
        # Performance comparison
        if len(successful_models) > 1:
            fastest = min(successful_models, key=lambda x: x["response_time"])
            print(f"\n⚡ Fastest model: {fastest['model']} ({fastest['response_time']}s)")
            
            best_reasoning = max(reasoning_capable, key=lambda x: x["reasoning_score"])
            print(f"🧠 Best reasoning: {best_reasoning['model']} (score: {best_reasoning['reasoning_score']}/10)")
    
    elif successful_models:
        print(f"\n⚠️  MODELS RESPONDING BUT NO REASONING DETECTED")
        print(f"Models are accessible but may need reasoning prompt tuning")
        
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