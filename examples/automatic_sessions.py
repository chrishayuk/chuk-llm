#!/usr/bin/env python3
# examples/automatic_sessions.py
"""
CHUK LLM - Automatic Session Tracking Example
============================================

Sessions are tracked automatically under the hood!

Setup:
    pip install chuk-llm chuk-ai-session-manager python-dotenv
    
    Create .env file:
    OPENAI_API_KEY=your-key-here
    ANTHROPIC_API_KEY=your-key-here
"""

import asyncio
import sys
import os

# Add src to path if running from examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Regular CHUK LLM imports - no special session imports needed!
from chuk_llm import (
    ask,
    stream,
    conversation,
    get_session_stats,
    get_session_history,
    get_current_session_id,
    reset_session,
)


async def demo_automatic_sessions():
    """Sessions are tracked automatically - just use the API normally!"""
    print("=== Automatic Session Tracking ===\n")
    
    # Just ask normally - sessions happen automatically
    response = await ask("What are the benefits of Python for data science?")
    print(f"Response: {response}\n")
    
    # Continue the conversation - context is maintained automatically
    response = await ask("Can you give me a code example?")
    print(f"Follow-up: {response}\n")
    
    # Check session stats whenever you want
    stats = await get_session_stats()
    print("📊 Session Stats (automatic!):")
    print(f"  Session ID: {get_current_session_id()}")
    print(f"  Total tokens: {stats.get('total_tokens', 0)}")
    print(f"  Estimated cost: ${stats.get('estimated_cost', 0):.6f}")
    print(f"  Messages: {stats.get('total_messages', 0)}")


async def demo_conversation_with_sessions():
    """Conversations automatically track sessions too!"""
    print("\n=== Conversation with Automatic Sessions ===\n")
    
    # Just use conversation normally - sessions are automatic
    async with conversation(
        provider="openai",
        system_prompt="You are a helpful Python tutor. Keep responses concise."
    ) as conv:
        # First question
        print("User: How do I read a CSV file in Python?")
        response = await conv.say("How do I read a CSV file in Python?")
        print(f"AI: {response}\n")
        
        # Follow-up with context
        print("User: What if the file has a different delimiter?")
        response = await conv.say("What if the file has a different delimiter?")
        print(f"AI: {response}\n")
        
        # Sessions are tracked automatically!
        if conv.has_session:
            print(f"✅ Session automatically created: {conv.session_id}")
            stats = await conv.get_session_stats()
            print(f"📊 Tokens used: {stats.get('total_tokens', 0)}")
            print(f"💰 Cost: ${stats.get('estimated_cost', 0):.6f}")


async def demo_streaming_with_automatic_sessions():
    """Streaming also gets automatic session tracking!"""
    print("\n=== Streaming with Automatic Sessions ===\n")
    
    print("Streaming response: ", end="", flush=True)
    
    # Stream normally - sessions happen automatically
    async for chunk in stream("Write a haiku about programming"):
        print(chunk, end="", flush=True)
    
    print("\n")
    
    # Check what got tracked
    stats = await get_session_stats()
    print(f"📊 Automatically tracked - Tokens: {stats.get('total_tokens', 0)}")


async def demo_session_history():
    """Access conversation history that's tracked automatically."""
    print("\n=== Automatic History Tracking ===\n")
    
    # Have a normal conversation
    await ask("My name is Alice and I'm learning Python.")
    await ask("What's a good first project for beginners?")
    await ask("Can you remind me what my name is?")
    
    # Get the automatically tracked history
    history = await get_session_history()
    
    print("📜 Automatically Tracked History:")
    for i, turn in enumerate(history):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if len(content) > 100:
            content = content[:100] + "..."
        print(f"  {i+1}. {role}: {content}")
    
    # Session stats
    stats = await get_session_stats()
    print(f"\n📊 Session Summary:")
    print(f"  Session ID: {get_current_session_id()}")
    print(f"  Total messages: {stats.get('total_messages', 0)}")
    print(f"  Total tokens: {stats.get('total_tokens', 0)}")
    print(f"  Estimated cost: ${stats.get('estimated_cost', 0):.6f}")


async def demo_multiple_sessions():
    """Show how to reset sessions when needed."""
    print("\n=== Multiple Sessions (Manual Reset) ===\n")
    
    # First session
    await ask("Tell me about machine learning")
    session1 = get_current_session_id()
    print(f"Session 1: {session1[:8]}...")
    
    # Reset to start a new session
    reset_session()
    print("🔄 Session reset")
    
    # Second session
    await ask("Tell me about web development")
    session2 = get_current_session_id()
    print(f"Session 2: {session2[:8]}...")
    
    # Sessions are different
    print(f"✅ Sessions are different: {session1[:8]} != {session2[:8]}")


async def demo_provider_specific_sessions():
    """Sessions work across different providers."""
    print("\n=== Cross-Provider Sessions ===\n")
    
    # Use different providers in the same session
    try:
        response1 = await ask("What's 2+2?", provider="openai")
        print(f"OpenAI: {response1}")
        
        response2 = await ask("What did I just ask you?", provider="anthropic")
        print(f"Anthropic: {response2}")
        
        # Check unified session stats
        stats = await get_session_stats()
        print(f"\n📊 Unified Session Stats:")
        print(f"  Messages across providers: {stats.get('total_messages', 0)}")
        print(f"  Total cost: ${stats.get('estimated_cost', 0):.6f}")
    except Exception as e:
        print(f"Note: Cross-provider demo requires multiple API keys configured")


async def main():
    """Run all demos."""
    print("🚀 CHUK LLM - Automatic Session Tracking")
    print("=" * 50)
    print("Sessions are tracked automatically - no extra code needed!\n")
    
    try:
        # Check if sessions are available
        initial_stats = await get_session_stats()
        if initial_stats.get("sessions_enabled", True):
            print("✅ Session tracking is enabled and automatic!\n")
        else:
            print("⚠️ Session tracking is disabled (CHUK_LLM_DISABLE_SESSIONS=true)\n")
        
        # Run demos
        await demo_automatic_sessions()
        await demo_conversation_with_sessions()
        await demo_streaming_with_automatic_sessions()
        await demo_session_history()
        await demo_multiple_sessions()
        await demo_provider_specific_sessions()
        
        print("\n✨ All demos completed!")
        print("\n💡 Key Points:")
        print("  • Sessions are tracked automatically - no special imports needed")
        print("  • Every ask(), stream(), and conversation is tracked")
        print("  • Access stats and history anytime with get_session_stats()")
        print("  • Reset sessions manually with reset_session() when needed")
        print("  • Disable with environment variable: CHUK_LLM_DISABLE_SESSIONS=true")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have API keys configured in .env")


if __name__ == "__main__":
    asyncio.run(main())