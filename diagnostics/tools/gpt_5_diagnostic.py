#!/usr/bin/env python3
"""
Manual GPT-5 Parameter Test
===========================
Test GPT-5 models with correct parameters to verify they work
when the parameter conversion is done manually.
"""

import asyncio
import os
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

async def test_gpt5_manually():
    """Test GPT-5 models with manual parameter conversion"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Direct OpenAI API test
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        
        print("🧪 Manual GPT-5 Testing (Bypassing chuk_llm)")
        print("=" * 50)
        
        models_to_test = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        
        for model_name in models_to_test:
            print(f"\n🔍 Testing {model_name} manually...")
            
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": "Hello! Can you solve 2+2?"}
                    ],
                    max_completion_tokens=50,  # Correct parameter for GPT-5
                    # temperature=1.0  # GPT-5 only supports default temperature (1.0)
                )
                
                message = response.choices[0].message.content
                print(f"   ✅ SUCCESS: {model_name}")
                print(f"      Response: {message[:100]}...")
                print(f"      Usage: {response.usage}")
                
                # Test reasoning capability
                print(f"\n🧠 Testing reasoning capability...")
                reasoning_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": "If Alice has 3 apples and Bob has twice as many, how many do they have total?"}
                    ],
                    max_completion_tokens=200
                    # No temperature parameter - GPT-5 uses default
                )
                
                reasoning_text = reasoning_response.choices[0].message.content
                print(f"      Reasoning response: {reasoning_text[:150]}...")
                
                # Check for correct answer (9 total)
                has_correct_answer = "9" in reasoning_text
                shows_work = any(word in reasoning_text.lower() for word in ["twice", "6", "3"])
                
                print(f"      ✅ Correct answer: {has_correct_answer}")
                print(f"      ✅ Shows work: {shows_work}")
                
            except Exception as e:
                error_msg = str(e)
                if "does not exist" in error_msg:
                    print(f"   ❌ Model not available: {model_name}")
                elif "max_tokens" in error_msg:
                    print(f"   🔧 Parameter error (should be fixed): {error_msg}")
                else:
                    print(f"   ❌ Error: {error_msg}")
        
        print(f"\n📊 MANUAL TEST SUMMARY:")
        print(f"   This test bypasses chuk_llm and uses OpenAI API directly")
        print(f"   If models work here but not in chuk_llm, the issue is in the client")
        
    except ImportError:
        print("❌ openai package not available for direct testing")
        print("   Install with: pip install openai")
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_gpt5_manually())