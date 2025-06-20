#!/usr/bin/env python3
# examples/test_session_integration.py
"""
Minimal test for CHUK LLM session integration
"""

import asyncio
import sys
import os

# Add src to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()


async def test_basic_session():
    """Test basic session functionality"""
    print("Testing CHUK LLM Session Integration\n")
    
    try:
        # Import the core functions
        from chuk_llm import ask, get_session_stats, get_current_session_id
        
        print("1. Making first request...")
        response = await ask("Hello! What's 2+2?")
        print(f"   Response: {response}\n")
        
        print("2. Checking session...")
        session_id = get_current_session_id()
        if session_id:
            print(f"   ✅ Session ID: {session_id[:8]}...")
        else:
            print(f"   ⚠️  No session ID (sessions might be disabled)")
        
        print("\n3. Making follow-up request...")
        response = await ask("What did I just ask you?")
        print(f"   Response: {response}\n")
        
        print("4. Getting session stats...")
        stats = await get_session_stats()
        if stats.get('sessions_enabled', True):
            print(f"   ✅ Sessions enabled")
            print(f"   Total messages: {stats.get('total_messages', 0)}")
            print(f"   Total tokens: {stats.get('total_tokens', 0)}")
            print(f"   Estimated cost: ${stats.get('estimated_cost', 0):.6f}")
        else:
            print(f"   ⚠️  Sessions not available")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure chuk-ai-session-manager is installed:")
        print("   uv add chuk-ai-session-manager")
        print("2. Check your Python path")
        print("3. Make sure you're running from the project root")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have API keys in .env file")


if __name__ == "__main__":
    asyncio.run(test_basic_session())