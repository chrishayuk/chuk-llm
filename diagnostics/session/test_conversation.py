#!/usr/bin/env python3
# test_conversation_context.py
"""
Test conversation context with session tracking
"""

import asyncio
import sys
import os

# Add src to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from chuk_llm import conversation, ask, get_session_stats, get_session_history


async def test_conversation_context():
    """Test that conversation context is properly maintained"""
    print("🧪 Testing Conversation Context with Sessions\n")
    
    # Test 1: Using conversation API (should maintain context)
    print("1. Testing conversation API:")
    async with conversation(provider="openai") as conv:
        print("   User: My name is Alice")
        response1 = await conv.say("My name is Alice")
        print(f"   AI: {response1}\n")
        
        print("   User: What's my name?")
        response2 = await conv.say("What's my name?")
        print(f"   AI: {response2}\n")
        
        # Check if session is tracking
        if conv.has_session:
            stats = await conv.get_session_stats()
            print(f"   ✅ Session tracking: {conv.session_id[:8]}...")
            print(f"   📊 Messages: {stats['total_messages']}, Tokens: {stats['total_tokens']}\n")
    
    # Test 2: Using direct ask() calls (session tracking but no context)
    print("2. Testing direct ask() calls:")
    response3 = await ask("My name is Bob")
    print(f"   First ask: {response3}\n")
    
    response4 = await ask("What's my name?")
    print(f"   Second ask: {response4}\n")
    
    # Check session history
    history = await get_session_history()
    print("3. Session History:")
    for i, msg in enumerate(history[-4:], 1):  # Last 4 messages
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:80] + '...' if len(msg.get('content', '')) > 80 else msg.get('content', '')
        print(f"   {i}. {role}: {content}")
    
    # Final stats
    stats = await get_session_stats()
    print(f"\n📊 Final Session Stats:")
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Estimated cost: ${stats['estimated_cost']:.6f}")
    
    print("\n💡 Key Insights:")
    print("   - conversation() maintains context between messages")
    print("   - ask() tracks sessions but doesn't maintain context")
    print("   - Both approaches contribute to the same session")


if __name__ == "__main__":
    asyncio.run(test_conversation_context())